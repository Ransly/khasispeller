"""The service's own guards (khasi_spell/limits.py): the per-visitor rate
limit, the request-size cap, the busy gate and how a visitor is told apart.

Before these (audit of 2026-10-04, item 8) nothing stopped a script from
sending the heaviest legal request in a loop: 50,000 characters of unknown
words held a worker for half a minute, and the lexicon's glosses could be
copied one /word call at a time.

None of this needs the lexicon: the middleware is exercised on a three-line
app. The routes of the real service are covered in test_audit_regressions.
"""
import threading
import time

import pytest

from khasi_spell import limits


# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------

def test_defaults_leave_room_for_a_classroom_on_one_address():
    s = limits.Settings()
    # One press of "Check" is two requests (/check, then /realword).
    assert s.checks_per_min >= 30 * 2
    assert s.max_text_chars == 20_000


def test_settings_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("KHASI_SPELL_MAX_TEXT_CHARS", "5000")
    monkeypatch.setenv("KHASI_SPELL_RATE_CHECKS_PER_MIN", "0")
    monkeypatch.setenv("KHASI_SPELL_MAX_BODY_BYTES", "not a number")
    s = limits.Settings.from_env()
    assert s.max_text_chars == 5000
    assert "checks" in s.rules() and s.rules()["checks"] == []     # switched off
    assert s.max_body_bytes == limits.Settings().max_body_bytes     # bad value ignored


def test_the_text_limit_cannot_be_switched_off(monkeypatch):
    monkeypatch.setenv("KHASI_SPELL_MAX_TEXT_CHARS", "0")
    assert limits.Settings.from_env().max_text_chars == limits.Settings().max_text_chars


# ----------------------------------------------------------------------
# Rate limiter
# ----------------------------------------------------------------------

class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_requests_pass_up_to_the_limit_then_wait_is_reported():
    clock = Clock()
    rl = limits.RateLimiter({"checks": [(3, 60)]}, clock=clock)
    assert [rl.hit("a", "checks") for _ in range(3)] == [None, None, None]
    assert rl.hit("a", "checks") == 60
    clock.now += 45
    assert rl.hit("a", "checks") == 15           # what is left of the window
    clock.now += 15
    assert rl.hit("a", "checks") is None         # a new window


def test_each_visitor_and_each_category_has_its_own_allowance():
    rl = limits.RateLimiter({"checks": [(1, 60)], "lookups": [(1, 60)]}, clock=Clock())
    assert rl.hit("a", "checks") is None
    assert rl.hit("b", "checks") is None         # another visitor
    assert rl.hit("a", "lookups") is None        # another category
    assert rl.hit("a", "checks") is not None


def test_the_hourly_rule_holds_after_the_minute_resets():
    clock = Clock()
    rl = limits.RateLimiter({"lookups": [(2, 60), (3, 3600)]}, clock=clock)
    assert rl.hit("a", "lookups") is None
    assert rl.hit("a", "lookups") is None
    assert rl.hit("a", "lookups") == 60          # the minute is used up
    clock.now += 60
    assert rl.hit("a", "lookups") is None        # third of the hour
    wait = rl.hit("a", "lookups")                # minute has room, the hour has not
    assert wait is not None and wait > 3000


def test_a_refused_request_does_not_use_up_the_other_rule():
    clock = Clock()
    rl = limits.RateLimiter({"lookups": [(5, 60), (1, 3600)]}, clock=clock)
    assert rl.hit("a", "lookups") is None
    for _ in range(10):                          # all refused by the hourly rule
        assert rl.hit("a", "lookups") is not None
    assert rl._state[("a", "lookups", 60)][1] == 1


def test_an_unlisted_category_is_not_limited():
    rl = limits.RateLimiter({"checks": []}, clock=Clock())
    assert all(rl.hit("a", c) is None for c in ("checks", "other") for _ in range(50))


def test_old_visitors_are_forgotten():
    clock = Clock()
    rl = limits.RateLimiter({"checks": [(5, 60)]}, clock=clock, max_entries=100)
    for i in range(100):
        rl.hit(f"v{i}", "checks")
    clock.now += 61
    rl.hit("new", "checks")                      # over max_entries: purge runs
    assert len(rl._state) == 1


# ----------------------------------------------------------------------
# Busy gate
# ----------------------------------------------------------------------

def test_gate_refuses_when_full_and_nobody_may_wait():
    gate = limits.HeavyGate(slots=1, max_waiting=0, wait_seconds=0)
    with gate.slot():
        with pytest.raises(limits.Busy):
            with gate.slot():
                pass
    with gate.slot():                            # released: free again
        pass


def test_gate_gives_up_after_the_wait():
    gate = limits.HeavyGate(slots=1, max_waiting=1, wait_seconds=0.05)
    with gate.slot():
        with pytest.raises(limits.Busy):
            with gate.slot():
                pass
    assert gate._waiting == 0


def test_a_waiting_check_runs_when_a_slot_frees():
    gate = limits.HeavyGate(slots=1, max_waiting=1, wait_seconds=5)
    ran = []

    def waiter():
        with gate.slot():
            ran.append(1)

    with gate.slot():
        t = threading.Thread(target=waiter)
        t.start()
        deadline = time.monotonic() + 2
        while gate._waiting == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert gate._waiting == 1 and ran == []      # queued, not running
    t.join(timeout=5)
    assert ran == [1]


def test_slot_is_released_when_the_check_fails():
    gate = limits.HeavyGate(slots=1, max_waiting=0, wait_seconds=0)
    with pytest.raises(ValueError):
        with gate.slot():
            raise ValueError("the check itself failed")
    with gate.slot():
        pass


def test_gate_with_no_slots_is_off():
    gate = limits.HeavyGate(slots=0, max_waiting=0, wait_seconds=0)
    with gate.slot(), gate.slot(), gate.slot():
        pass


# ----------------------------------------------------------------------
# Who the visitor is
# ----------------------------------------------------------------------

def _scope(headers=(), client=("10.0.0.7", 5555)):
    return {"type": "http", "headers": [(k.lower().encode(), v.encode()) for k, v in headers],
            "client": client}


def test_cloudflare_header_wins_over_forwarded_for():
    scope = _scope([("X-Forwarded-For", "6.6.6.6, 198.51.100.4"), ("CF-Connecting-IP", "203.0.113.9")])
    assert limits.client_address(scope) == ("203.0.113.9", "cf-connecting-ip")


def test_forwarded_for_uses_its_first_entry():
    scope = _scope([("X-Forwarded-For", "198.51.100.4, 10.1.2.3")])
    assert limits.client_address(scope) == ("198.51.100.4", "x-forwarded-for")


def test_no_proxy_headers_means_the_connection():
    assert limits.client_address(_scope()) == ("10.0.0.7", "connection")
    assert limits.client_address(_scope(client=None)) == ("unknown", "connection")


# ----------------------------------------------------------------------
# Middleware, on a small app
# ----------------------------------------------------------------------

@pytest.fixture()
def tiny():
    """A three-route app behind ServiceLimits with small limits."""
    fastapi = pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    settings = limits.Settings(max_body_bytes=200, checks_per_min=2,
                               lookups_per_min=3, lookups_per_hour=0)
    app = fastapi.FastAPI()

    @app.post("/check")
    def check(body: dict):
        return {"n": len(body.get("text", ""))}

    @app.post("/word")
    def word(body: dict):
        return {"ok": True}

    @app.post("/other")
    def other(body: dict):
        return {"ok": True}

    @app.get("/health")
    def health():
        return {"ok": True}

    app.add_middleware(limits.ServiceLimits, settings=settings,
                       limiter=limits.RateLimiter(settings.rules()))
    with TestClient(app) as client:
        yield client


def test_too_many_checks_get_429_with_retry_after(tiny):
    assert tiny.post("/check", json={"text": "ka"}).status_code == 200
    assert tiny.post("/check", json={"text": "ka"}).status_code == 200
    r = tiny.post("/check", json={"text": "ka"})
    assert r.status_code == 429
    assert 1 <= int(r.headers["retry-after"]) <= 60
    assert "Too many requests" in r.json()["detail"]


def test_checks_and_lookups_are_counted_apart(tiny):
    for _ in range(2):
        tiny.post("/check", json={"text": "ka"})
    assert tiny.post("/check", json={"text": "ka"}).status_code == 429
    assert tiny.post("/word", json={"word": "ka"}).status_code == 200


def test_visitors_are_told_apart_by_the_proxy_header(tiny):
    a = {"CF-Connecting-IP": "203.0.113.1"}
    b = {"CF-Connecting-IP": "203.0.113.2"}
    for _ in range(2):
        assert tiny.post("/check", json={"text": "ka"}, headers=a).status_code == 200
    assert tiny.post("/check", json={"text": "ka"}, headers=a).status_code == 429
    assert tiny.post("/check", json={"text": "ka"}, headers=b).status_code == 200


def test_get_requests_and_other_routes_are_never_limited(tiny):
    for _ in range(20):
        assert tiny.get("/health").status_code == 200
        assert tiny.post("/other", json={}).status_code == 200


def test_declared_oversized_body_is_refused_before_parsing(tiny):
    r = tiny.post("/other", json={"text": "x" * 500})
    assert r.status_code == 413
    assert "too large" in r.json()["detail"]


def test_chunked_oversized_body_is_refused_too(tiny):
    def chunks():
        yield b'{"text": "'
        for _ in range(50):
            yield b"x" * 20
        yield b'"}'
    r = tiny.post("/other", content=chunks(), headers={"Content-Type": "application/json"})
    assert r.status_code == 413


def test_a_body_at_the_cap_passes(tiny):
    body = b'{"text": "' + b"x" * 150 + b'"}'
    assert len(body) <= 200
    r = tiny.post("/other", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 200
