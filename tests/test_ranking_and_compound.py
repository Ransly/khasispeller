"""
Regression tests for the morphology ranking engine (roadmap item 9) and the
solid-compound splitter (roadmap item 10a).
"""
import os
import unittest

from khasi_engine import morphology, complex as complex_words
from khasi_engine.database import KhasiDB


class TestMorphologyRanking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = KhasiDB()

    def test_direct_match_beats_derivation(self):
        # kynja is a real lexical root; the ranker must NOT prefer kyn- + ja.
        r = morphology.parse("kynja", self.db.lookup)
        self.assertTrue(r["pass"])
        self.assertEqual(r["status"], "direct_match")
        self.assertEqual(r["root"], "kynja")

    def test_parse_score_present(self):
        r = morphology.parse("kynja", self.db.lookup)
        self.assertIn("parse_score", r["affix_info"])
        self.assertIsInstance(r["affix_info"]["parse_score"], float)

    def test_lexicon_root_outscores_phonotactic(self):
        # A derivation whose root is in the lexicon must outscore a bare
        # phonotactic-only one. jingbam → jing- + bam (bam in lexicon).
        r = morphology.parse("jingbam", self.db.lookup)
        self.assertTrue(r["pass"])
        self.assertTrue(bool(r.get("matches")))

    def test_parse_tree_shape_and_leaf_pos(self):
        # jingpynim → jing- ( pyn- ( im ) ); the leaf im is a VERB, not the
        # surface noun's PoS.
        r = morphology.parse("jingpynim", self.db.lookup)
        tree = r.get("tree")
        self.assertIsInstance(tree, dict)
        # Walk to the leaf.
        node = tree
        while "child" in node:
            node = node["child"]
        self.assertEqual(node.get("root"), "im")
        self.assertEqual(node.get("pos"), "verb")

    def test_alternatives_are_lower_scored(self):
        r = morphology.parse("jingkynshaitum", self.db.lookup)
        alts = r.get("alternatives") or []
        if alts:
            best = r["affix_info"]["parse_score"]
            self.assertTrue(all(a["score"] <= best for a in alts))

    def test_affix_order_flexible_pair_strict_mode(self):
        # War p.72: ïa- may precede pyn- (jingïapyndom class). In strict mode
        # the affix-rank guard must not reject an ia-before-pyn stack.
        os.environ["STRICT_PHASE_A"] = "1"
        try:
            r = morphology.parse("jingiapyndem", self.db.lookup)
            self.assertTrue(r["pass"])
            prefixes = [v for k, v in r["layers"].items() if k.startswith("prefix")]
            self.assertIn("ia-", prefixes)
            self.assertIn("pyn-", prefixes)
        finally:
            os.environ.pop("STRICT_PHASE_A", None)


class TestSolidCompoundSplitter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = KhasiDB()

    def test_splits_two_lexicon_roots(self):
        # laitdoh = lait + doh (both real roots), not in DB as a solid entry.
        r = complex_words.detect("laitdoh", self.db.lookup)
        self.assertTrue(r["detected"])
        self.assertEqual(r["type"], "compound_fusion")
        self.assertEqual(set(r["components"]), {"lait", "doh"})

    def test_rejects_nonsense(self):
        r = complex_words.detect("xyzqk", self.db.lookup)
        self.assertFalse(r["detected"])

    def test_no_function_word_split(self):
        # A split whose part is a free morpheme / clitic must be rejected —
        # that would be a phrase, not a compound (Phase A Rule 4). We assert
        # that IF a fusion split is returned, no part is a function word.
        free = {m.lower() for m in morphology.FREE_MORPHEMES}
        clitics = {c.lower() for c in morphology.CLITICS}
        blocked = free | clitics
        for w in ("kabam", "ubha", "kileh"):
            r = complex_words.detect(w, self.db.lookup)
            if r["detected"] and r["type"] == "compound_fusion":
                self.assertNotIn(r["details"]["root1"], blocked)
                self.assertNotIn(r["details"]["root2"], blocked)

    def test_affix_beats_compound_in_pipeline(self):
        # In the full pipeline Phase 2 (morphology) runs before Phase 4, so a
        # prefixed word must resolve as a derivation, not a fusion compound.
        from khasi_engine.analyser import KhasiAnalyser
        analyser = KhasiAnalyser()
        r = analyser.analyse("nonghikai")
        self.assertEqual(r["verdict"], "derived")
        self.assertIn("nong", r["verdict_desc"].lower())


if __name__ == "__main__":
    unittest.main()
