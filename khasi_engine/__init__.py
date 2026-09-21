"""
khasi_engine — Khasi spellchecking engine.

Extracted from the MorpSpeller / Khasi NLP Explorer platform as a
standalone spellchecker. This package contains only the modules the
spelling path actually depends on:

    phonology      Phase 1 — phonotactic validation (gates 0 and 1)
    morphology     Phase 2 — affix cascade, the validity oracle
    assimilation   Phase 3 — surface/underlying reconstruction
    complex        Phase 4 — reduplication, compounds, loans
    database       lexicon store and word-list source
    spell_checker  the morphology-gated checker itself
    analyser       orchestrator; injects the morphology gate
    w2v_client     optional FastText re-ranking over HTTP

Everything else from the parent platform (NER, relation extraction,
governance, exports, G2P, TTS, the REST surface) is deliberately absent.

Most callers want `khasi_spell.KhasiSpeller`, not this package directly.
"""

from khasi_engine.analyser import KhasiAnalyser
from khasi_engine.database import KhasiDB
from khasi_engine.spell_checker import KhasiSpellChecker

__all__ = ["KhasiAnalyser", "KhasiDB", "KhasiSpellChecker"]
__version__ = "1.0.0"
