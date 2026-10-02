"""
Tokenisation, normalisation and case — khasi_engine.tokens.

No lexicon needed: these are pure functions every component relies on, and
the ways they used to disagree corrupted correct text.
"""
from khasi_engine import tokens as T


def _words(text):
    return [m.group(0) for m in T.tokens(text)]


def test_accented_vowels_are_letters():
    # `sngewrém` is a lexicon word; it used to split into `sngewr` + `m`.
    assert _words("ka sngewrém ka úd") == ["ka", "sngewrém", "ka", "úd"]


def test_typographic_apostrophe_joins_a_word():
    # Phones and word processors turn ' into ’; `nga’m` used to split.
    assert _words("u nga’m u") == ["u", "nga’m", "u"]
    assert _words("u ngaʼm u") == ["u", "ngaʼm", "u"]


def test_scan_debris_joins_a_word():
    assert _words("ka i^p-shnong ka") == ["ka", "i^p-shnong", "ka"]


def test_joiners_at_the_edge_are_not_part_of_a_word():
    assert _words("'nong- ka") == ["nong", "ka"]


def test_canonical_keeps_offsets():
    text = "nga’m"
    assert T.canonical(text) == "nga'm"
    assert len(T.canonical(text)) == len(text)


def test_match_case():
    assert T.match_case("Shnng", "shnong") == "Shnong"
    assert T.match_case("SHNNG", "shnong") == "SHNONG"
    assert T.match_case("shnng", "shnong") == "shnong"
    assert T.match_case("Iaid", "ïaid") == "Ïaid"
    assert T.match_case("Jongki", "jong ki") == "Jong ki"


def test_sentence_starts():
    text = "Ka shnong ka bha\nShnng ka bha"
    assert T.starts_sentence(text, text.index("Shnng"))          # a new line
    t2 = "Kane ka: Shnng ka bha"
    assert T.starts_sentence(t2, t2.index("Shnng"))               # after a colon
    t3 = "- Shnng ka bha"
    assert T.starts_sentence(t3, t3.index("Shnng"))               # a list item
    t4 = 'U la ong. "Shnng ka bha'
    assert T.starts_sentence(t4, t4.index("Shnng"))               # opening quote
    t5 = "ka la iaid sha Ardent"
    assert not T.starts_sentence(t5, t5.index("Ardent"))          # mid-sentence


def test_sentence_index_follows_the_ngram_builder():
    text = "Ka la wan. U la leit\nKi la ong"
    spans = [(m.start(), m.end()) for m in T.tokens(text)]
    assert T.sentence_index(text, spans) == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert T.sentence_index("3.5 ka", [(4, 6)]) == [0]
