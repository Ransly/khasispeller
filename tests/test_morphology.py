import unittest

from khasi_engine import morphology
from khasi_engine.database import KhasiDB


class TestMorphologySuffixParsing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = KhasiDB()

    def test_jingkynshaitum_stack_suffix(self):
        # 2026-06-10: kyn- is fossilised, so kynshait is an atomic root.
        # 2026-07-06 (item 5): -m is the negation clitic ym contracted onto a
        # CLOSED host set (u, i, ka, …) — NOT a productive stem suffix. So the
        # engine must NOT invent a -um suffix on the open-class stem kynshait.
        # jingkynshaitum is not a real word; the honest analysis is the
        # productive prefix jing- over an OOV remainder (root not in lexicon).
        result = morphology.parse('jingkynshaitum', self.db.lookup)

        self.assertTrue(result['pass'])
        self.assertEqual(result['layers'].get('prefix'), 'jing-')
        # No spurious -m/-um suffix layer.
        self.assertIsNone(result['layers'].get('suffix'))

    def test_shaitum_no_spurious_negation_suffix(self):
        # War pp.74-75: -m only contracts onto a clitic host. 'shait' is a
        # verb, not a host, so 'shaitum' must NOT be decomposed as shait+(-m).
        result = morphology.parse('shaitum', self.db.lookup)
        if result['pass']:
            self.assertNotEqual(result['layers'].get('suffix'), '-um')
            self.assertNotEqual(result['layers'].get('suffix'), '-m')

    def test_clitic_negation_forms_valid(self):
        # The valid contractions are whole lexicon entries and resolve by
        # direct match: um (u+m 'he not'), im (i+m 'it not').
        for w in ('um', 'im'):
            with self.subTest(word=w):
                r = morphology.parse(w, self.db.lookup)
                self.assertTrue(r['pass'], f"{w} should resolve")


if __name__ == '__main__':
    unittest.main()
