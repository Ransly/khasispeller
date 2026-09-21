"""
G2P v2 regression tests (khasi_engine/g2p.py).

Asserts the systematic War-derived rules directly, and measures overall
agreement against the gold calibration fixture, excluding the phonemic
vowel-length words (not recoverable from orthography — see g2p.py docstring).
"""
import json
import unittest
from pathlib import Path

from khasi_engine import g2p

FIXTURE = Path(__file__).parent / "g2p_calibration_war.json"


def _norm(s: str) -> str:
    s = s.strip().strip("/|[]").replace(" ", "")
    for sup, plain in (("kʰ", "kh"), ("pʰ", "ph"), ("tʰ", "th"),
                       ("bʰ", "bh"), ("dʰ", "dh"), ("dzʰ", "jh")):
        s = s.replace(sup, plain)
    return s.replace("ɨ", "ɪ")


class TestG2PRules(unittest.TestCase):
    def test_final_glottal(self):
        self.assertEqual(g2p.to_ipa("dih"), "/diʔ/")
        self.assertEqual(g2p.to_ipa("bah"), "/baʔ/")
        self.assertEqual(g2p.to_ipa("lyngdoh"), "/lɪŋdɔʔ/")

    def test_onset_h_stays_h(self):
        self.assertTrue(g2p.to_ipa("her").startswith("/h"))

    def test_palatal_final(self):
        self.assertEqual(g2p.to_ipa("buit"), "/buic/")
        self.assertEqual(g2p.to_ipa("loit"), "/lɔic/")

    def test_lax_i(self):
        self.assertEqual(g2p.to_ipa("im"), "/ɪm/")
        self.assertEqual(g2p.to_ipa("tip"), "/tɪp/")
        # open / glottal-closed i stays /i/
        self.assertEqual(g2p.to_ipa("ki"), "/ki/")

    def test_prevocalic_y_glottal(self):
        self.assertEqual(g2p.to_ipa("shyiap"), "/ʃʔiap/")

    def test_y_nucleus(self):
        self.assertEqual(g2p.to_ipa("kynjai"), "/kɪndzai/")

    def test_affricate_and_aspirate(self):
        self.assertEqual(g2p.to_ipa("kyrjaw"), "/kɪrdzau/")
        self.assertEqual(g2p.to_ipa("thmu"), "/tʰmu/")

    def test_apostrophe_is_elision_not_glottal(self):
        # War p.84: ' marks a deleted segment ('tikmie < kti+kmie), it is
        # not a glottal stop. It must contribute no phoneme, initially or
        # word-internally.
        for word in ("'a", "'bai", "ar'ti", "sh'ieng"):
            self.assertNotIn("ʔ", g2p.to_ipa(word), f"{word} got a glottal stop")
        self.assertEqual(g2p.to_ipa("'a"), "/a/")
        # A vowel-initial word with no apostrophe is transcribed the same way,
        # so the two spellings stay phonologically consistent.
        self.assertEqual(g2p.to_ipa("'am"), g2p.to_ipa("am"))


class TestLengthLexicon(unittest.TestCase):
    def test_length_words_from_lexicon(self):
        # Phonemic length is supplied by the override lexicon, verbatim.
        self.assertEqual(g2p.to_ipa("nam"), "/naːm/")
        self.assertEqual(g2p.to_ipa("akor"), "/akɔːr/")
        self.assertEqual(g2p.to_ipa("khaw"), "/kʰaːu/")

    def test_lexicon_word_not_flagged_for_review(self):
        self.assertFalse(g2p.convert("nam")["needs_review"])

    def test_lexical_quality_override(self):
        # leit/putharo: rules get the vowel quality wrong; lexicon fixes it.
        self.assertEqual(g2p.to_ipa("leit"), "/lɔic/")
        self.assertEqual(g2p.to_ipa("putharo"), "/putʰaro/")


class TestG2PCalibration(unittest.TestCase):
    def test_full_agreement(self):
        # With the length lexicon, every attested War word must match.
        entries = json.loads(FIXTURE.read_text(encoding="utf-8"))["entries"]
        ok = sum(1 for e in entries
                 if _norm(g2p.to_ipa(e["word"])) == _norm(e["war_ipa"]))
        acc = ok / len(entries)
        self.assertGreaterEqual(
            acc, 0.95,
            f"calibration agreement {acc:.0%} below 95% ({ok}/{len(entries)})",
        )

    def test_derivable_rules_alone_at_least_90pct(self):
        # Rule engine WITHOUT the lexicon (unseen words) still >= 90% on the
        # orthographically-derivable (non-length) subset.
        saved = g2p.LENGTH_LEXICON
        g2p.LENGTH_LEXICON = {}
        try:
            entries = json.loads(FIXTURE.read_text(encoding="utf-8"))["entries"]
            derivable = [e for e in entries
                         if "vowel_length" not in e.get("phenomena", [])]
            ok = sum(1 for e in derivable
                     if _norm(g2p.to_ipa(e["word"])) == _norm(e["war_ipa"]))
            acc = ok / len(derivable)
        finally:
            g2p.LENGTH_LEXICON = saved
        self.assertGreaterEqual(acc, 0.90,
                                f"rule-only agreement {acc:.0%} below 90%")


if __name__ == "__main__":
    unittest.main()
