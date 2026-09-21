"""
Golden-corpus regression test.

Every maintainer linguistic ruling lives in tests/golden_corpus.json as a
word → expected-behaviour pair. This test runs the full analyser over the
corpus and fails loudly when any ruling regresses — e.g. a lexicon
cleanup pass that re-introduces a fossilised-prefix decomposition, or an
engine change that un-gates a stripping path.

Add a new entry to the JSON whenever the maintainer makes a ruling.
Runs in CI via .github/workflows/smoke.yml (unittest discovery).
"""
import json
import unittest
from pathlib import Path

from khasi_engine.analyser import KhasiAnalyser

CORPUS_PATH = Path(__file__).parent / "golden_corpus.json"


def _load_corpus() -> dict:
    data = json.loads(CORPUS_PATH.read_text())
    merged = {}
    for section in ("valid_roots", "derived_forms", "rejected_fakes"):
        merged.update(data.get(section, {}))
    return merged


class TestGoldenCorpus(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        cls.analyser = KhasiAnalyser()
        cls.corpus = _load_corpus()

    def test_corpus_not_empty(self):
        self.assertGreater(len(self.corpus), 30)

    def test_all_rulings_hold(self):
        failures = []
        for word, expected in self.corpus.items():
            result = self.analyser.analyse(word)
            verdict = result.get("verdict")
            root = (result.get("phase2") or {}).get("root")

            if verdict != expected["verdict"]:
                failures.append(
                    f"  {word!r}: verdict {verdict!r} != expected "
                    f"{expected['verdict']!r}  ({expected.get('ruling', '')})"
                )
                continue
            if "root" in expected and root != expected["root"]:
                failures.append(
                    f"  {word!r}: root {root!r} != expected "
                    f"{expected['root']!r}  ({expected.get('ruling', '')})"
                )

        if failures:
            self.fail(
                f"{len(failures)} golden-corpus ruling(s) regressed:\n"
                + "\n".join(failures)
            )


if __name__ == "__main__":
    unittest.main()
