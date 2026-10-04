"""The page's offset conversion (toPageOffsets in khasi_spell/static/index.html).

The server reports offsets in code points; the page slices UTF-16 strings. An
emoji before a flagged word shifted its highlight, and "Fix" replaced the wrong
letters: "😀😀 ka shnongg bha" came back "😀😀 kshnonggg bha" (2026-10-04).
These run the page's own function under Node.js.
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


def _function_source() -> str:
    src = PAGE.read_text(encoding="utf-8")
    m = re.search(r"^function toPageOffsets\(res\)\{.*?^\}", src, re.S | re.M)
    assert m, "toPageOffsets not found in index.html"
    return m.group(0)


def _slices(res: dict) -> dict:
    """Convert *res* with the page's function; return the text each span covers."""
    script = (_function_source()
              + "\nconst res = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
              + "\ntoPageOffsets(res);"
              + "\nconst cut = l => (l || []).map(x => typeof x.start === 'number'"
              + " ? res.text.slice(x.start, x.end) : null);"
              + "\nprocess.stdout.write(JSON.stringify({c: cut(res.corrections),"
              + " v: cut(res.variants), s: cut(res.skipped), r: cut(res.realword)}));")
    out = subprocess.run([NODE, "-e", script], input=json.dumps(res),
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def _span(text: str, word: str) -> dict:
    i = text.index(word)
    return {"start": i, "end": i + len(word)}       # code points, as the server counts


@pytest.mark.parametrize("text", [
    "ka shnongg bha",                 # nothing to convert
    "😀😀 ka shnongg bha",             # emoji before the word
    "ka shnongg😀 bha",                # emoji touching it
])
def test_a_flag_marks_its_own_word(text):
    assert _slices({"text": text, "corrections": [_span(text, "shnongg")]})["c"] == ["shnongg"]


def test_every_list_is_converted():
    text = "ka 👍 shnongg 👍 iathuh 😀 Shillong u riew"
    got = _slices({
        "text": text,
        "corrections": [_span(text, "shnongg")],
        "variants": [_span(text, "iathuh")],
        "skipped": [_span(text, "Shillong"), {"word": "x"}],    # a skip may carry no span
        "realword": [_span(text, "riew")],
    })
    assert got == {"c": ["shnongg"], "v": ["iathuh"], "s": ["Shillong", None], "r": ["riew"]}
