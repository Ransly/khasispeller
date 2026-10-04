"""The page's word pattern (WORD_SRC in khasi_spell/static/index.html) must cut
text exactly as the server does (khasi_engine.tokens), or a click on a word
opens a different word. Runs the page's own definitions under Node.js."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from khasi_engine import tokens as T

PAGE = Path(__file__).resolve().parents[1] / "khasi_spell" / "static" / "index.html"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js not installed")

SAMPLES = [
    "ka sngewrém ka úd, u nga’m u, ka i^p-shnong",
    "'nong- ka jiap-jiap",
    "ka shn\u00adong bha, ka shn\u200bongg",
    "ka \uff53\uff48\uff4e\uff4f\uff4e\uff47 bha",
    "ka shn\u043eng bha \u0441\u043eн \u03bf\u03b9",
]


def test_the_page_cuts_words_as_the_server_does():
    src = PAGE.read_text(encoding="utf-8")
    m = re.search(r"^const W_LETTERS = .*?^const wordRe = .*?;$", src, re.S | re.M)
    assert m, "word pattern not found in index.html"
    script = (m.group(0)
              + "\nconst xs = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
              + "\nprocess.stdout.write(JSON.stringify(xs.map(s => s.match(wordRe()) || [])));")
    out = subprocess.run([NODE, "-e", script], input=json.dumps(SAMPLES),
                         capture_output=True, text=True, check=True)
    page = json.loads(out.stdout)
    server = [[t.group(0) for t in T.tokens(s)] for s in SAMPLES]
    assert page == server
