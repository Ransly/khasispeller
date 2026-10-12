"""
limits.py — what stops one visitor, or one oversized request, from holding
the HTTP service.

Four independent guards, each off when its setting is 0:

  rate limit        per visitor address: text checks per minute, word
                    look-ups per minute and per hour. Answered with 429 and
                    a Retry-After header.
  request size      a body larger than the cap is refused with 413 before it
                    is parsed.
  busy gate         at most N text checks run at once; a few more wait a
                    short while; the rest are told to retry (503). Checking
                    is CPU-bound, so more threads only make every check
                    slower.
  text length       the largest text /check, /correct and /realword accept
                    (enforced by the request schemas in api.py).

The rate limit is per address, and a classroom or an office usually shares
one. The defaults are sized for that: 30 people pressing "Check" in the same
minute stay inside them. The busy gate does not depend on addresses at all,
so it holds even when the address a client reports cannot be trusted.

Settings (environment variables):
  KHASI_SPELL_MAX_TEXT_CHARS          20000
  KHASI_SPELL_MAX_BODY_BYTES          524288
  KHASI_SPELL_RATE_CHECKS_PER_MIN     90
  KHASI_SPELL_RATE_LOOKUPS_PER_MIN    180
  KHASI_SPELL_RATE_LOOKUPS_PER_HOUR   2000
  KHASI_SPELL_MAX_CONCURRENT_CHECKS   2
  KHASI_SPELL_MAX_WAITING_CHECKS      8
  KHASI_SPELL_BUSY_WAIT_SECONDS       15
"""
from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Optional

# Routes that check running text, and routes that look one word up. The
# second group returns the lexicon's glosses, which are licensed: the hourly
# limit on it is what slows copying them in bulk.
CHECK_PATHS = frozenset({"/check", "/correct", "/realword", "/batch", "/v1/spellcheck"})
LOOKUP_PATHS = frozenset({"/word", "/analyse"})


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        print(f"[khasi-spell] {name}={raw!r} is not a whole number; using {default}")
        return default


@dataclass(frozen=True)
class Settings:
    max_text_chars: int = 20_000
    max_body_bytes: int = 512 * 1024
    checks_per_min: int = 90
    lookups_per_min: int = 180
    lookups_per_hour: int = 2_000
    max_concurrent_checks: int = 2
    max_waiting_checks: int = 8
    busy_wait_seconds: int = 15

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        return cls(
            # The text limit cannot be switched off: an unbounded text is
            # the case every other guard exists to survive.
            max_text_chars=_env_int("KHASI_SPELL_MAX_TEXT_CHARS", d.max_text_chars) or d.max_text_chars,
            max_body_bytes=_env_int("KHASI_SPELL_MAX_BODY_BYTES", d.max_body_bytes),
            checks_per_min=_env_int("KHASI_SPELL_RATE_CHECKS_PER_MIN", d.checks_per_min),
            lookups_per_min=_env_int("KHASI_SPELL_RATE_LOOKUPS_PER_MIN", d.lookups_per_min),
            lookups_per_hour=_env_int("KHASI_SPELL_RATE_LOOKUPS_PER_HOUR", d.lookups_per_hour),
            max_concurrent_checks=_env_int("KHASI_SPELL_MAX_CONCURRENT_CHECKS", d.max_concurrent_checks),
            max_waiting_checks=_env_int("KHASI_SPELL_MAX_WAITING_CHECKS", d.max_waiting_checks),
            busy_wait_seconds=_env_int("KHASI_SPELL_BUSY_WAIT_SECONDS", d.busy_wait_seconds),
        )

    def rules(self) -> dict:
        """{category: [(limit, window seconds), ...]}, limits of 0 left out."""
        rules = {
            "checks": [(self.checks_per_min, 60)],
            "lookups": [(self.lookups_per_min, 60), (self.lookups_per_hour, 3600)],
        }
        return {k: [(n, w) for n, w in v if n > 0] for k, v in rules.items()}

    def public(self) -> dict:
        """What a caller may be told, for /health and /api."""
        return {
            "max_text_chars": self.max_text_chars,
            "max_request_bytes": self.max_body_bytes or None,
            "text_checks_per_minute": self.checks_per_min or None,
            "word_lookups_per_minute": self.lookups_per_min or None,
            "word_lookups_per_hour": self.lookups_per_hour or None,
        }


class RateLimiter:
    """Counts requests per (visitor, category) in fixed windows.

    A window opens at a visitor's first request and closes `window` seconds
    later; within it at most `limit` requests pass. Fixed windows let a
    burst straddle two of them, which is acceptable here: the limit exists
    to stop sustained hammering, and a counter per window costs two numbers
    per visitor where a sliding log costs one timestamp per request.
    """

    def __init__(self, rules: dict, clock: Callable[[], float] = time.monotonic,
                 max_entries: int = 50_000):
        self._rules = {k: list(v) for k, v in rules.items()}
        self._clock = clock
        self._max_entries = max_entries
        self._state: dict = {}            # (key, category, window) -> [opened, count]
        self._lock = threading.Lock()

    def hit(self, key: str, category: str) -> Optional[int]:
        """Record one request. None when it may pass, else seconds to wait."""
        rules = self._rules.get(category)
        if not rules:
            return None
        now = self._clock()
        with self._lock:
            wait = 0.0
            slots = []
            for limit, window in rules:
                slot = self._state.get((key, category, window))
                if slot is None or now - slot[0] >= window:
                    slot = [now, 0]
                    self._state[(key, category, window)] = slot
                slots.append(slot)
                if slot[1] >= limit:
                    wait = max(wait, slot[0] + window - now)
            if wait > 0:
                return max(1, int(wait + 0.999))
            # Counted only when every rule passes: a request refused by the
            # hourly rule must not also use up the minute's allowance.
            for slot in slots:
                slot[1] += 1
            if len(self._state) > self._max_entries:
                self._purge(now)
            return None

    def _purge(self, now: float) -> None:
        dead = [k for k, (opened, _n) in self._state.items() if now - opened >= k[2]]
        for k in dead:
            del self._state[k]
        if len(self._state) > self._max_entries:      # still full: all live
            self._state.clear()


class Busy(Exception):
    """No slot for a text check became free in time."""


class HeavyGate:
    """Lets `slots` text checks run at once and `max_waiting` more queue.

    A request beyond that, or one that has queued for `wait_seconds`, gets
    Busy. Without the queue cap every waiting request would hold one of the
    server's worker threads, and enough of them leave none for /health.
    """

    def __init__(self, slots: int, max_waiting: int, wait_seconds: float):
        self._enabled = slots > 0
        self._sem = threading.Semaphore(slots) if self._enabled else None
        self._max_waiting = max(0, max_waiting)
        self._wait_seconds = max(0.0, float(wait_seconds))
        self._waiting = 0
        self._lock = threading.Lock()

    @contextmanager
    def slot(self):
        if not self._enabled:
            yield
            return
        if not self._sem.acquire(blocking=False):
            with self._lock:
                if self._waiting >= self._max_waiting:
                    raise Busy()
                self._waiting += 1
            try:
                got = self._sem.acquire(timeout=self._wait_seconds)
            finally:
                with self._lock:
                    self._waiting -= 1
            if not got:
                raise Busy()
        try:
            yield
        finally:
            self._sem.release()


def client_address(scope) -> tuple:
    """(address, where it was read from) for an ASGI http scope.

    Behind a proxy the socket's peer is the proxy, the same for every
    visitor; keyed on it, one busy visitor would use up everybody's
    allowance. The service runs behind Cloudflare, which writes the
    visitor's address into CF-Connecting-IP itself (a value the client sent
    is replaced). X-Forwarded-For is the fallback; its first entry can be
    set by the client, so it identifies honest visitors only. Neither is
    proof of identity: this decides whose allowance a request is counted
    against, nothing more, and the busy gate does not rely on it.
    """
    headers = {}
    for name, value in scope.get("headers") or ():
        headers.setdefault(name, value)
    cf = headers.get(b"cf-connecting-ip", b"").decode("latin-1").strip()
    if cf:
        return cf[:64], "cf-connecting-ip"
    xff = headers.get(b"x-forwarded-for", b"").decode("latin-1").split(",")[0].strip()
    if xff:
        return xff[:64], "x-forwarded-for"
    client = scope.get("client")
    return (client[0] if client else "unknown"), "connection"


async def _send_json(send, status: int, body: dict, headers: Optional[dict] = None) -> None:
    payload = json.dumps(body).encode("utf-8")
    raw = [(b"content-type", b"application/json"),
           (b"content-length", str(len(payload)).encode("ascii"))]
    for k, v in (headers or {}).items():
        raw.append((k.lower().encode("ascii"), str(v).encode("ascii")))
    await send({"type": "http.response.start", "status": status, "headers": raw})
    await send({"type": "http.response.body", "body": payload})


class ServiceLimits:
    """ASGI middleware: the rate limit and the request-size cap.

    Only POST requests are touched; the page, /health and the favicon are
    GETs. Replies use FastAPI's own error shape, {"detail": "..."}, which is
    what the page displays.
    """

    def __init__(self, app, settings: Settings, limiter: RateLimiter):
        self.app = app
        self.settings = settings
        self.limiter = limiter
        self._logged: dict = {}           # (address, category) -> time of last log line

    def _log_refusal(self, address: str, category: str, wait: int) -> None:
        """One log line per visitor and category a minute, not one per request:
        a flood of refused requests must not become a flood of log lines."""
        now = time.monotonic()
        if now - self._logged.get((address, category), -1e9) < 60:
            return
        if len(self._logged) > 5_000:
            self._logged.clear()
        self._logged[(address, category)] = now
        print(f"[khasi-spell] rate limit: {category} from {address}, retry in {wait}s")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return

        path = (scope.get("path") or "").rstrip("/") or "/"
        category = ("checks" if path in CHECK_PATHS else
                    "lookups" if path in LOOKUP_PATHS else None)
        if category:
            address, _source = client_address(scope)
            wait = self.limiter.hit(address, category)
            if wait is not None:
                self._log_refusal(address, category, wait)
                await _send_json(
                    send, 429,
                    {"detail": "Too many requests from your network. Please wait "
                               f"{wait} second{'s' if wait != 1 else ''} and try again."},
                    {"Retry-After": wait})
                return

        cap = self.settings.max_body_bytes
        if not cap:
            await self.app(scope, receive, send)
            return

        too_large = {"detail": "This request is too large. Please send the text in smaller parts."}
        declared = None
        for name, value in scope.get("headers") or ():
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = None
                break
        if declared is not None:
            # The server holds the body to its declared length, so checking
            # the declaration is enough.
            if declared > cap:
                await _send_json(send, 413, too_large)
                return
            await self.app(scope, receive, send)
            return

        # No declared length (a chunked body): read it here, at most `cap`
        # bytes of it, and hand it on only if it fits. Raising from inside
        # the app's own read does not work: the framework reports any error
        # while reading a body as a plain 400.
        chunks, seen = [], 0
        while True:
            message = await receive()
            if message["type"] != "http.request":       # the client went away
                return
            body = message.get("body") or b""
            seen += len(body)
            if seen > cap:
                await _send_json(send, 413, too_large)
                return
            chunks.append(body)
            if not message.get("more_body"):
                break
        payload, replayed = b"".join(chunks), False

        async def replay():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": payload, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)
