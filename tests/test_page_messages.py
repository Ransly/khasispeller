"""What the page tells the user when the server refuses a request
(apiErrorMessage and countNote in khasi_spell/static/index.html).

A text over the length limit comes back as a list of validation problems,
and the page printed that list as "[object Object]". The service's own
refusals (too many requests, busy, too large) are plain text and must be
shown as they are. These run the page's own functions under Node.js.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parents[1] / "khasi_spell" / "static" / "index.html"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js not installed")


def _function(name: str) -> str:
    src = PAGE.read_text(encoding="utf-8")
    m = re.search(r"^function %s\(.*?^\}" % re.escape(name), src, re.S | re.M)
    assert m, f"{name} not found in index.html"
    return m.group(0)


def _call(name: str, *args):
    script = (_function(name)
              + "\nconst args = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
              + f"\nprocess.stdout.write(JSON.stringify({name}(...args)));")
    out = subprocess.run([NODE, "-e", script], input=json.dumps(list(args)),
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_too_long_text_is_explained_with_the_limit():
    data = {"detail": [{"type": "string_too_long", "loc": ["body", "text"],
                        "msg": "String should have at most 20000 characters",
                        "ctx": {"max_length": 20000}}]}
    msg = _call("apiErrorMessage", 422, "Unprocessable Entity", data)
    assert msg == ("The text is too long: the limit is 20,000 characters. "
                   "Please check it in parts.")


@pytest.mark.parametrize("detail", [
    "Too many requests from your network. Please wait 12 seconds and try again.",
    "The checker is busy with other requests. Please try again in a few seconds.",
    "This request is too large. Please send the text in smaller parts.",
])
def test_the_services_own_messages_are_shown_as_written(detail):
    assert _call("apiErrorMessage", 429, "Too Many Requests", {"detail": detail}) == detail


def test_other_validation_problems_show_their_message_not_object_object():
    data = {"detail": [{"type": "missing", "msg": "Field required"}]}
    assert _call("apiErrorMessage", 422, "Unprocessable Entity", data) == "Field required"


@pytest.mark.parametrize("data", [{}, None, {"detail": ""}, {"detail": []}])
def test_an_empty_reply_falls_back_to_the_status(data):
    assert _call("apiErrorMessage", 502, "Bad Gateway", data) == "502 Bad Gateway"


def test_the_counter_says_when_the_text_is_over_the_limit():
    assert _call("countNote", 1234, 200, 20000) == "1,234 characters · 200 words"
    assert _call("countNote", 20000, 3000, 20000) == "20,000 characters · 3000 words"
    assert _call("countNote", 20001, 3000, 20000) == (
        "20,001 characters · 3000 words · over the limit of 20,000, check it in parts")
