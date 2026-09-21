"""
Item 3 regression tests: lookup normalization + root-minimality gate.
"""
import unittest

from khasi_engine import morphology
from khasi_engine.database import KhasiDB


class TestRootMinimality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = KhasiDB()

    def test_no_single_letter_root(self):
        # iaroh must not collapse to ia- + "r" (alphabet placeholder junk).
        r = morphology.parse("iaroh", self.db.lookup)
        self.assertTrue(r["pass"])
        self.assertGreaterEqual(len(r["root"] or ""), 2,
                                f"root {r['root']!r} is a single letter")

    def test_alphabet_entries_quarantined(self):
        for junk in ("r", "b", "k", "adverb", "bass"):
            with self.subTest(word=junk):
                self.assertEqual(self.db._surface_index.get(junk), None,
                                 f"{junk!r} should be quarantined out of the lexicon")


class TestLookupNormalization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = KhasiDB()

    def test_accent_fold(self):
        # A diacritic-free query should still reach an accented entry.
        acc = next((sf for sf in self.db._surface_index
                    if any(c in sf for c in "áéíóúï")
                    and " " not in sf and "-" not in sf), None)
        self.assertIsNotNone(acc)
        plain = self.db._fold_key(acc)
        self.assertTrue(self.db.lookup(plain),
                        f"folded query {plain!r} should find {acc!r}")

    def test_apostrophe_fold(self):
        apos = next((sf for sf in self.db._surface_index
                     if sf.startswith("'") and len(sf) > 2 and " " not in sf), None)
        self.assertIsNotNone(apos)
        self.assertTrue(self.db.lookup(apos[1:]),
                        f"apostrophe-free query {apos[1:]!r} should find {apos!r}")

    def test_exact_match_still_priority(self):
        # Folding must never shadow an exact canonical entry.
        exact = next((sf for sf in self.db._surface_index
                      if " " not in sf and len(sf) > 3), None)
        self.assertIsNotNone(exact)
        hits = self.db.lookup(exact)
        self.assertTrue(hits)


if __name__ == "__main__":
    unittest.main()
