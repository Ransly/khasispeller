"""
khasi_engine — Khasi spellchecking engine.

Extracted from the MorpSpeller / Khasi NLP Explorer platform as a
standalone spellchecker, and extended here (see the README for what differs
from the parent). The modules:

    tokens         word tokenisation, normalisation and case, shared by all
    paths          where the data files live (KHASI_DATA_DIR)
    phonology      Phase 1 — phonotactic validation (gates 0 and 1)
    morphology     Phase 2 — affix cascade, the validity oracle
    assimilation   Phase 3 — surface/underlying reconstruction
    complex        Phase 4 — reduplication, compounds, loans
    database       lexicon store and word-list source
    generate       on-demand morphological candidate generation
    spell_checker  the morphology-gated checker itself
    analyser       orchestrator; injects the morphology gate
    g2p            rule-based pronunciation, used by the /analyse panel
    w2v_client     optional FastText re-ranking, local model or remote Space

Named-entity recognition, relation extraction, governance, exports, TTS and
the parent's REST surface are deliberately absent.

Most callers want `khasi_spell.KhasiSpeller`, not this package directly.

The three public classes are imported lazily. Importing any submodule —
`khasi_engine.tokens`, say — used to import the analyser and with it the
phonology and morphology modules, which read the lexicon at import time.
That happened before the HTTP service could load its `.env`, so a
DATABASE_URL kept there arrived too late and the 64 MB JSON was parsed
anyway.
"""

__all__ = ["KhasiAnalyser", "KhasiDB", "KhasiSpellChecker"]
__version__ = "1.0.0"


def __getattr__(name):
    if name == "KhasiAnalyser":
        from khasi_engine.analyser import KhasiAnalyser
        return KhasiAnalyser
    if name == "KhasiDB":
        from khasi_engine.database import KhasiDB
        return KhasiDB
    if name == "KhasiSpellChecker":
        from khasi_engine.spell_checker import KhasiSpellChecker
        return KhasiSpellChecker
    raise AttributeError(f"module 'khasi_engine' has no attribute {name!r}")
