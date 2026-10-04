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


def test_invisible_characters_inside_a_word_join_it():
    # Soft hyphens and zero-width spaces come in with text copied from PDFs
    # and web pages; `shn<soft hyphen>ong` used to split into `shn` + `ong`.
    for ch in "\u00ad\u200b\u200c\u200d\u2060\ufeff":
        assert _words(f"ka shn{ch}ong bha") == ["ka", f"shn{ch}ong", "bha"]
        assert T.canonical(f"shn{ch}ong") == "shnong"


def test_full_width_letters_are_letters():
    w = "\uff53\uff48\uff4e\uff4f\uff4e\uff47"          # \uff53\uff48\uff4e\uff4f\uff4e\uff47
    assert _words(f"ka {w} bha") == ["ka", w, "bha"]
    assert T.canonical(w) == "shnong"


def test_a_look_alike_letter_stays_in_its_word_and_is_not_hidden():
    w = "shn\u043eng"                                   # Cyrillic \u043e
    assert _words(f"ka {w} bha") == ["ka", w, "bha"]
    assert T.canonical(w) == w                          # flagged, not silently fixed
    assert T.latinise(w) == "shnong"


def test_cyrillic_and_greek_text_stays_outside_the_checker():
    assert _words("ka \u0441\u043eн bha \u03bf\u03b9 ka") == ["ka", "bha", "ka"]


def test_a_long_run_of_look_alikes_is_scanned_quickly():
    import time
    t0 = time.time()
    assert _words("\u043e" * 50000) == []
    assert time.time() - t0 < 1


def test_addresses_and_letters_on_numbers_are_not_checked():
    text = "ka 12th, 1st, 79.2mm, 5:30pm, ML04A, covid19, www.mpsc.nic.in, a.b@gmail.com"
    skip, digit_words = T.unchecked_spans(text)
    for piece in ["th", "st", "mm", "pm", "ML04A", "covid19", "www.mpsc.nic.in", "a.b@gmail.com"]:
        i = text.index(piece)
        assert any(a <= i < b for a, b in skip), piece
    assert digit_words == []


def test_a_digit_inside_a_word_is_found():
    assert T.unchecked_spans("ka shn0ng bha")[1] == [(3, 9, "shn0ng")]
