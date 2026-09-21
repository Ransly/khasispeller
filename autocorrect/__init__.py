# Python 3 Spelling Corrector — Khasi-enhanced edition
#
# Copyright 2014 Jonas McCallum.
# Updated for Python 3, based on Peter Norvig's
# 2007 version: http://norvig.com/spell-correct.html
#
# Khasi morphological improvements:
#   [1] Dictionary expansion   — prefix+root combinations added at load time
#       so productively-derived words (jingbha, nonghikai…) are always known.
#   [2] Prefix stripping       — autocorrect_word strips a known prefix,
#       corrects the bare root, then reattaches for better recall.
#   [3] Morpheme score bonus   — candidates that are prefix+known-root get
#       a score boost so they rank above edit-distance noise.
#   [4] Infix handling         — done in typos.py (_tokenize_khasi pass 2).
#   [5] Pronoun protection     — inflected pronoun forms (un, um, kan, kam…)
#       are recognised as grammatically valid and never over-corrected.

import json
import re
import tarfile
import textwrap
import warnings
from contextlib import closing
from functools import lru_cache
from itertools import chain
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

from autocorrect.constants import (
    word_regexes,
    backup_urls,
    ipfs_gateways,
    ipfs_paths,
    KHASI_PREFIXES,
    KHASI_SUFFIXES,
    KHASI_PRONOUN_ROOTS,
)
from autocorrect.typos import Word

__all__ = ["Speller", "spell"]

PATH = Path(__file__).parent

# Prefixes sorted longest-first so "nong" is tried before "ng", etc.
_PREFIXES_SORTED: list[str] = sorted(KHASI_PREFIXES, key=len, reverse=True)


# ---------------------------------------------------------------------------
# Download progress helper
# ---------------------------------------------------------------------------

class ProgressBar:
    """Simple ASCII progress bar for dictionary downloads."""

    def __init__(self):
        self.old_percent = 0
        print("_" * 50)

    def download_progress_hook(self, count: int, blockSize: int, totalSize: int) -> None:
        if totalSize <= 0:
            return
        percent = int(count * blockSize * 100 / totalSize)
        if percent >= self.old_percent + 2:
            self.old_percent = percent
            print(">", end="", flush=True)
        if percent >= 100:
            print("\ndone!")


# ---------------------------------------------------------------------------
# Dictionary loading (cached at module level)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _load_cached(lang: str, file_name: str = "word_count.json") -> dict:
    """
    Load and cache the word-frequency dictionary for *lang*.

    Resolution order:
      1. <data_dir>/<lang>.json    (plain JSON — useful for development / offline)
      2. <data_dir>/<lang>.tar.gz  (compressed archive)
      3. Download from IPFS / backup URLs, then cache as .tar.gz

    All keys are lowercased on load so that lookups are always
    case-insensitive without needing per-call .lower() on the dict itself.
    """
    if lang not in word_regexes:
        supported = ", ".join(sorted(word_regexes.keys()))
        raise NotImplementedError(
            textwrap.dedent(f"""\
                Language '{lang}' is not supported.
                Supported languages: {supported}
                To add Khasi support follow the instructions at
                https://github.com/fsondej/autocorrect/tree/master#adding-new-languages
            """)
        )

    data_dir = PATH / "data"
    data_dir.mkdir(exist_ok=True)

    json_path = data_dir / f"{lang}.json"
    archive_path = data_dir / f"{lang}.tar.gz"

    # Fast path: plain JSON (development / offline use)
    if json_path.is_file():
        with open(json_path, encoding="utf-8") as f:
            raw = json.load(f)
        return {k.lower(): v for k, v in raw.items()}

    # Compressed archive
    if archive_path.is_file():
        return _read_tar(archive_path, file_name)

    # Download from IPFS gateways then backup URLs
    all_urls = [
        gateway + path
        for gateway in ipfs_gateways
        for path in ipfs_paths.get(lang, [])
    ] + backup_urls.get(lang, [])

    if not all_urls:
        raise FileNotFoundError(
            f"No download URLs configured for language '{lang}'.\n"
            f"Place a '{lang}.json' or '{lang}.tar.gz' file in {data_dir}."
        )

    last_error: Optional[Exception] = None
    for url in all_urls:
        print(f"Downloading dictionary from {url} …")
        progress = ProgressBar()
        try:
            urlretrieve(url, archive_path, progress.download_progress_hook)
            last_error = None
            break
        except Exception as exc:
            print(f"  failed ({exc}), trying next URL …")
            last_error = exc
            if archive_path.is_file():
                archive_path.unlink()

    if last_error is not None:
        raise ConnectionError(
            f"{last_error}\n"
            "Fix your network connection, or manually place the dictionary file\n"
            f"  {archive_path}\nor\n  {json_path}"
        )

    return _read_tar(archive_path, file_name)


def _read_tar(archive_path: Path, file_name: str) -> dict:
    """Extract and parse the JSON word-count file from a .tar.gz archive."""
    with closing(tarfile.open(archive_path, "r:gz")) as tarf:
        with closing(tarf.extractfile(file_name)) as f:
            raw = json.load(f)
    return {k.lower(): v for k, v in raw.items()}


def load_from_tar(lang: str, file_name: str = "word_count.json") -> dict:
    """Public alias kept for backward compatibility."""
    return _load_cached(lang, file_name)


# ---------------------------------------------------------------------------
# Morphology helpers (module-level, language-agnostic interface)
# ---------------------------------------------------------------------------

def _expand_with_morphology(nlp_data: dict) -> None:
    """
    [Improvement 1] In-place expansion of *nlp_data* with prefix+root forms.

    Iterates every known root and every Khasi prefix.  If the derived form
    (prefix + root) is not already in the dictionary, it is added with a
    frequency derived from the root's frequency divided by the prefix's
    divisor.  This ensures that productively-formed words like
    'jingbha', 'nonghikai', 'pynstad' are always recognised even if they
    were absent from the raw corpus.

    Only called for lang='kh'.
    """
    roots = list(nlp_data.keys())           # snapshot — don't iterate while mutating
    for prefix, divisor in KHASI_PREFIXES.items():
        for root in roots:
            derived = prefix + root
            if derived not in nlp_data:
                nlp_data[derived] = max(1, nlp_data[root] // divisor)


def _strip_prefix(word: str) -> tuple[str, str]:
    """
    [Improvement 2 — helper] Return (prefix, root) if *word* starts with a
    known Khasi prefix and the remaining root is at least 2 characters long.
    Returns ('', word) when no prefix matches.

    Prefixes are tried longest-first so 'nong' is matched before 'no', etc.
    """
    for prefix in _PREFIXES_SORTED:
        if word.startswith(prefix) and len(word) > len(prefix) + 1:
            return prefix, word[len(prefix):]
    return "", word


def _morpheme_bonus(candidate: str, nlp_data: dict) -> int:
    """
    [Improvement 3] Return a score bonus for candidates that are a valid
    prefix + known root combination.

    This lifts morphologically transparent words above edit-distance noise
    that happens to share the same edit distance.  A candidate like
    'jingbha' (jing + bha, both known) scores +10; a noise form like
    'jingbhx' scores 0.
    """
    for prefix in _PREFIXES_SORTED:
        if candidate.startswith(prefix):
            root = candidate[len(prefix):]
            if root in nlp_data:
                return 10
    return 0


def _is_valid_pronoun_form(word: str) -> bool:
    """
    [Improvement 5] Return True if *word* is a known pronoun root with a
    valid tense or negation suffix attached.

    Recognised forms:
      u+n=un, u+m=um, ka+n=kan, ka+m=kam,
      i+n=in, i+m=im, phi+n=phin, phi+m=phim,
      ha+n=han, ha+m=ham, ki+n=kin, ki+m=kim

    These are grammatically well-formed and must not be corrected away.
    """
    for root in KHASI_PRONOUN_ROOTS:
        if word.startswith(root):
            suffix = word[len(root):]
            if suffix in KHASI_SUFFIXES:
                # Confirm this root is in the allowed set for that suffix
                if root in KHASI_SUFFIXES[suffix]:
                    return True
    return False


# ---------------------------------------------------------------------------
# Speller
# ---------------------------------------------------------------------------

class Speller:
    """
    Autocorrecting spell checker with full Khasi morphological support.

    Parameters
    ----------
    lang : str
        Language code.  Use 'kh' for Khasi.
    threshold : int
        Ignore words with frequency < threshold (reduces memory, may hurt recall).
    nlp_data : dict | None
        Provide your own word-frequency dict instead of loading from disk.
    fast : bool
        If True, only explore 1-edit candidates (faster but less accurate).
    only_replacements : bool
        If True, only generate replacement edits (no inserts/deletes/transposes).
    custom_words : dict | None
        Extra {word: frequency} pairs merged into nlp_data at init time.
    expand_morphology : bool
        If True (default for lang='kh'), automatically add prefix+root derived
        forms to the dictionary at initialisation time.  Set to False to skip
        this step (e.g. when memory is very constrained).
    """

    def __init__(
        self,
        lang: str = "kh",
        threshold: int = 0,
        nlp_data: Optional[dict] = None,
        fast: bool = False,
        only_replacements: bool = False,
        custom_words: Optional[dict] = None,
        expand_morphology: bool = True,
    ):
        self.lang = lang
        self.threshold = threshold
        self.fast = fast
        self.only_replacements = only_replacements

        # Load base dictionary (keys are already lowercased by _load_cached)
        self.nlp_data: dict = (
            dict(load_from_tar(lang)) if nlp_data is None else
            {k.lower(): v for k, v in nlp_data.items()}
        )

        # Apply frequency threshold
        if threshold > 0:
            self.nlp_data = {
                k: v for k, v in self.nlp_data.items() if v >= threshold
            }

        # [Improvement 1] Expand dictionary with morphologically derived forms.
        if lang == "kh" and expand_morphology:
            _expand_with_morphology(self.nlp_data)

        # Merge custom words (custom entries always win on frequency conflicts)
        if custom_words:
            for word, freq in custom_words.items():
                key = word.lower()
                self.nlp_data[key] = max(self.nlp_data.get(key, 0), freq)

    # ------------------------------------------------------------------
    # Dictionary management
    # ------------------------------------------------------------------

    def add_word(self, word: str, frequency: int = 1) -> None:
        """
        Add a single word to the in-memory dictionary.

        >>> speller.add_word("Sohra")
        """
        key = word.lower()
        self.nlp_data[key] = max(self.nlp_data.get(key, 0), frequency)

    def add_words(self, words) -> None:
        """
        Add multiple words at once.

        Accepts any iterable of strings, or a dict {word: frequency}.

        >>> speller.add_words(["Sohra", "Nongkrem", "Weiking"])
        >>> speller.add_words({"Sohra": 100, "Nongkrem": 50})
        """
        if isinstance(words, dict):
            for word, freq in words.items():
                self.add_word(word, freq)
        else:
            for word in words:
                self.add_word(word)

    def save_custom_words(self, path: str) -> None:
        """Persist the current in-memory dictionary to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.nlp_data, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def existing(self, words) -> set:
        """
        Filter *words* to those present in the dictionary.

        Each candidate is lowercased before the lookup so that generated
        strings always match the normalised (all-lowercase) dict keys.
        Returns lowercased strings so nlp_data[c] is always safe.
        """
        return {word.lower() for word in words if word.lower() in self.nlp_data}

    def _scored_candidates(
        self,
        exact: set[str],
        one_edit: set[str],
        two_edit: set[str],
    ) -> list[tuple[int, str]]:
        """
        Build a scored result list from the three candidate tiers.

        Scoring:
          exact match  → frequency × 3  + morpheme bonus
          1-edit away  → frequency × 2  + morpheme bonus
          2-edit away  → frequency × 1  + morpheme bonus

        [Improvement 3] The morpheme bonus (+10) is added whenever a
        candidate is a recognised prefix + known root, lifting valid
        derived forms above spurious edit-distance neighbours.
        """
        results: list[tuple[int, str]] = []
        for c in exact:
            bonus = _morpheme_bonus(c, self.nlp_data) if self.lang == "kh" else 0
            results.append((self.nlp_data[c] * 3 + bonus, c))
        for c in one_edit - exact:
            bonus = _morpheme_bonus(c, self.nlp_data) if self.lang == "kh" else 0
            results.append((self.nlp_data[c] * 2 + bonus, c))
        for c in two_edit - one_edit - exact:
            bonus = _morpheme_bonus(c, self.nlp_data) if self.lang == "kh" else 0
            results.append((self.nlp_data[c] + bonus, c))
        return results

    def get_candidates(self, word: str) -> list[tuple[int, str]]:
        """
        Return a list of (score, candidate) tuples for *word*.

        All edit operations run on the lowercased input so generated
        candidates match the normalised dictionary keys.

        Tiers:
          1. Exact match     — score = frequency × 3  + morpheme bonus
          2. One edit away   — score = frequency × 2  + morpheme bonus
          3. Two edits away  — score = frequency × 1  + morpheme bonus
             (only computed when tiers 1 and 2 are both empty)

        Never empty — falls back to [(0, word)] when nothing is found.
        """
        lower_word = word.lower()
        w = Word(lower_word, self.lang, self.only_replacements)

        exact: set[str] = {lower_word} if lower_word in self.nlp_data else set()

        one_edit_strings: list[str] = list(w.typos())
        one_edit: set[str] = self.existing(one_edit_strings)

        if self.fast or exact or one_edit:
            two_edit: set[str] = set()
        else:
            two_edit = self.existing(
                chain.from_iterable(
                    Word(e, self.lang, self.only_replacements).typos()
                    for e in one_edit_strings
                )
            )

        results = self._scored_candidates(exact, one_edit, two_edit)
        return results or [(0, word)]

    def get_suggestions(self, word: str, n: int = 5) -> list[str]:
        """
        Return up to *n* ranked spelling suggestions for *word*.

        Unlike get_candidates(), this method always computes all three tiers
        so the caller receives as many high-quality suggestions as possible.
        The original word is excluded unless it is a valid dictionary entry.

        Parameters
        ----------
        word : str
            The potentially misspelled input word.
        n : int
            Number of suggestions to return (default 5).

        Returns
        -------
        list[str]
            Up to *n* suggestions, best-first, with original casing restored.
        """
        if not word:
            return []

        lower_word = word.lower()
        w = Word(lower_word, self.lang, self.only_replacements)

        exact: set[str] = {lower_word} if lower_word in self.nlp_data else set()
        one_edit_strings: list[str] = list(w.typos())
        one_edit: set[str] = self.existing(one_edit_strings)

        # Always compute tier 3 for suggestions so we can fill n slots.
        two_edit: set[str] = self.existing(
            chain.from_iterable(
                Word(e, self.lang, self.only_replacements).typos()
                for e in one_edit_strings
            )
        )

        scored = self._scored_candidates(exact, one_edit, two_edit)
        scored.sort(reverse=True)

        suggestions = []
        for _, candidate in scored[:n]:
            suggestions.append(self._restore_case(word, candidate))
        return suggestions

    # ------------------------------------------------------------------
    # Word correction
    # ------------------------------------------------------------------

    @staticmethod
    def _restore_case(original: str, corrected: str) -> str:
        """Restore the capitalisation pattern of *original* onto *corrected*."""
        if original.isupper():
            return corrected.upper()
        if original[0].isupper():
            return corrected[0].upper() + corrected[1:]
        return corrected

    def autocorrect_word(self, word: str) -> str:
        """
        Return the most likely correction for *word* (up to 2 edits away).

        For Khasi, three additional strategies are applied before falling
        back to plain edit-distance:

        [Improvement 5] Pronoun protection — inflected pronoun forms such as
        'un' (he will), 'um' (he not), 'kan' (she will) are grammatically
        valid and returned unchanged immediately.

        [Improvement 2] Prefix stripping — if the word starts with a known
        Khasi prefix, the bare root is corrected and the prefix reattached.
        The prefixed form is then added to the candidate pool so it can win
        over a spurious whole-word edit-distance candidate.

        All candidates are scored with the [Improvement 3] morpheme bonus
        so that valid prefix+root forms rank above edit-distance noise.
        """
        if not word:
            return word

        lower = word.lower()

        # [Improvement 5] Do not correct grammatically valid pronoun forms.
        if self.lang == "kh" and _is_valid_pronoun_form(lower):
            return word

        # Standard edit-distance candidates
        candidates = self.get_candidates(word)

        # [Improvement 2] Prefix-strip → correct root → reattach
        if self.lang == "kh":
            prefix, root = _strip_prefix(lower)
            if prefix and root:
                root_candidates = self.get_candidates(root)
                if root_candidates:
                    best_root = max(root_candidates)[1]
                    prefixed = prefix + best_root
                    # Score the reattached form: use its dict frequency if known,
                    # else inherit the root's score as a proxy.
                    if prefixed in self.nlp_data:
                        freq = self.nlp_data[prefixed]
                        bonus = _morpheme_bonus(prefixed, self.nlp_data)
                        candidates.append((freq * 2 + bonus, prefixed))
                    else:
                        root_freq = self.nlp_data.get(best_root, 1)
                        candidates.append((root_freq + 5, prefixed))

        best_word = max(candidates)[1]
        return self._restore_case(word, best_word)

    def _correct_token(self, token: str) -> str:
        """
        Correct a single token that may be hyphenated or contain an apostrophe.

        Each sub-part is corrected independently and rejoined so that
        compound Khasi words like 'jong-u' or 'da'n' are handled gracefully.
        """
        sep_pattern = r"([-'])"
        parts = re.split(sep_pattern, token)
        corrected = [
            part if re.match(sep_pattern, part)
            else (self.autocorrect_word(part) if part else part)
            for part in parts
        ]
        return "".join(corrected)

    # ------------------------------------------------------------------
    # Sentence / text correction
    # ------------------------------------------------------------------

    def autocorrect_sentence(self, sentence: str) -> str:
        """
        Correct all words in *sentence*, preserving surrounding punctuation
        and whitespace.
        """
        return re.sub(
            word_regexes[self.lang],
            lambda m: self._correct_token(m.group(0)),
            sentence,
        )

    def check_sentence(self, sentence: str) -> list[dict]:
        """
        Return correction suggestions for *sentence* without modifying it.

        Each entry in the returned list is::

            {
                "original":   "teh",
                "suggestion": "the",
                "start":      4,
                "end":        7,
            }

        Words that are already correct are omitted.
        """
        suggestions = []
        for match in re.finditer(word_regexes[self.lang], sentence):
            original = match.group(0)
            suggestion = self._correct_token(original)
            if suggestion != original:
                suggestions.append({
                    "original":   original,
                    "suggestion": suggestion,
                    "start":      match.start(),
                    "end":        match.end(),
                })
        return suggestions

    def is_known(self, word: str) -> bool:
        """Return True if *word* (case-insensitive) is in the dictionary."""
        return word.lower() in self.nlp_data

    # Make the speller callable — backward compatible with autocorrect_sentence
    __call__ = autocorrect_sentence


# ---------------------------------------------------------------------------
# Backward compatibility shim
# ---------------------------------------------------------------------------

class LazySpeller:
    """Deprecated — use Speller() directly."""

    def __init__(self):
        self._speller: Optional[Speller] = None

    def __call__(self, sentence: str) -> str:
        warnings.warn(
            "autocorrect.spell is deprecated, use autocorrect.Speller instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._speller is None:
            self._speller = Speller()
        return self._speller(sentence)


spell = LazySpeller()
