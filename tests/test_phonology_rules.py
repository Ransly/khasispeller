"""
Phonology-rule regression tests (roadmap items 1, 2, 6).

Asserts phonology.validate() behaviour directly (not through the analyser),
grounded in War (2001) Lynnong III-VI:
  • orthographic y is a vowel nucleus  (lyngdoh, nyngkong, jngai are valid)
  • final aspirates are forbidden       (bakh, bath are invalid)
  • lʰ / rʰ onsets are legal             (lhuh, rhah are valid)
  • loan-diagnostic finals -l/-s/-j warn but do not fail (bol, bus)
"""
import unittest

from khasi_engine import phonology


class TestYNucleus(unittest.TestCase):
    def test_y_words_valid(self):
        for w in ["lyngdoh", "lyngkha", "lyngngoh", "nyngkong", "kyrmen",
                  "kynthei", "pyrthei", "symboh", "tyngkai"]:
            with self.subTest(word=w):
                self.assertTrue(phonology.validate(w)["pass"],
                                f"{w} should be phonotactically valid (y = vowel)")

    def test_jngai_cluster_valid(self):
        # jng- is an attested Khasi onset (jngai 'far'); must not be rejected.
        self.assertTrue(phonology.validate("jngai")["pass"])

    def test_syllabic_y_words_no_exception_list_needed(self):
        # These used to require the phonotactic_exceptions crutch.
        for w in ["ym", "yn", "pyn", "bym", "byr", "myr"]:
            with self.subTest(word=w):
                self.assertTrue(phonology.validate(w)["pass"])


class TestFinalAspirate(unittest.TestCase):
    def test_final_aspirate_invalid(self):
        # War p.45: aspirates never occur word-finally (loans deaspirate).
        for w in ["bakh", "bath", "raph", "sokh"]:
            with self.subTest(word=w):
                self.assertFalse(phonology.validate(w)["pass"],
                                 f"{w} ends in an aspirate — should be invalid")

    def test_plain_final_h_valid(self):
        # Final orthographic -h (= glottal /ʔ/) is fine: soh, duh, leh.
        for w in ["soh", "duh", "leh", "bah"]:
            with self.subTest(word=w):
                self.assertTrue(phonology.validate(w)["pass"])


class TestAspiratedSonorants(unittest.TestCase):
    def test_lh_rh_onsets_valid(self):
        # War p.55: lʰ (lhuh) and rʰ (rhah) are phonemes; onsets are legal.
        for w in ["lhuh", "rhah"]:
            with self.subTest(word=w):
                self.assertTrue(phonology.validate(w)["pass"])


class TestLoanFinals(unittest.TestCase):
    def test_loan_finals_warn_not_fail(self):
        # Native Khasi has no final -l/-s/-j; loans do (bol, bus). These
        # should PASS (still valid words) but carry a loan-signal warning.
        for w in ["bol", "bus"]:
            with self.subTest(word=w):
                res = phonology.validate(w)
                self.assertTrue(res["pass"], f"{w} should still be valid")
                self.assertTrue(
                    any("loan" in warn.lower() for warn in res["warnings"]),
                    f"{w} should carry a loan-signal warning",
                )


if __name__ == "__main__":
    unittest.main()


# ----------------------------------------------------------------------
# Word-final coda validation  (added 2026-08-26)
# ----------------------------------------------------------------------
#
# ALLOWED_FINAL_CONSONANTS, _DIGRAPHS and _SPECIAL were loaded from the
# data and never consulted, so no coda check ran and 'bamsh' validated.

class TestFinalCoda(unittest.TestCase):

    def test_illegal_coda_is_rejected(self):
        result = phonology.validate("bamsh")
        self.assertFalse(result["pass"])
        self.assertTrue(any("coda" in e for e in result["errors"]),
                        f"expected a coda error, got {result['errors']}")

    def test_permitted_codas_pass(self):
        # p b d t k ' m n r, plus ng, ñ, a vowel, -h and -w
        for word in ("shnong", "briew", "ksew", "soh", "lyngdoh",
                     "bam", "kynthup", "ïathuh"):
            with self.subTest(word=word):
                self.assertTrue(phonology.validate(word)["pass"],
                                f"{word} wrongly rejected")

    def test_final_h_is_the_glottal_stop_not_an_aspirate(self):
        """
        forbidden_final used to list 'h', contradicting 838 entries. A bare
        final -h is /ʔ/; only the aspirate digraphs are barred.
        """
        self.assertTrue(phonology.validate("soh")["pass"])
        self.assertFalse(phonology.validate("bakh")["pass"])

    def test_final_w_offglide_allowed(self):
        """378 entries end in -w; it is a diphthong offglide, not a coda."""
        self.assertTrue(phonology.validate("ksew")["pass"])

    def test_loans_warn_rather_than_fail(self):
        """
        148 entries end in -l/-s/-j and every one inspected is a borrowing,
        so these are marked, not rejected.
        """
        for word in ("angel", "baptis", "awaj"):
            with self.subTest(word=word):
                r = phonology.validate(word)
                self.assertTrue(r["pass"], f"{word} should pass with a warning")

    def test_stale_constants_are_not_applied(self):
        """
        VALID_INITIALS excludes 'ï' and would reject every ïa- word; it must
        stay unconsumed. This guards against someone wiring it up.
        """
        self.assertNotIn("ï", phonology.VALID_INITIALS)
        self.assertTrue(phonology.validate("ïabeit")["pass"])
