# constants.py
# Khasi language support for autocorrect spell checker
#
# Khasi (Sohra dialect / standard romanized orthography) uses the Latin script
# with a defined set of letters and special characters.
#
# Special characters confirmed in Khasi orthography:
#   ï / Ï  — diaeresis-i, represents a distinct back unrounded vowel
#   ñ / Ñ  — tilde-n, represents the palatal nasal sound
#
# Digraphs (multi-character sequences that represent a SINGLE phoneme):
#   ng, sh, kh, ph, th, bh, dh, jh, ie
# These are treated as atomic units during all edit-distance operations so
# that the spell checker never generates phonemically invalid fragments.

# ---------------------------------------------------------------------------
# Word-boundary regex patterns
# ---------------------------------------------------------------------------
# Khasi words may contain:
#   - Basic Latin letters (case-insensitive)
#   - ï / Ï  (diaeresis i)
#   - ñ / Ñ  (tilde n — palatal nasal)
#   - Hyphens within compound words  (e.g. "jong-u")
#   - Apostrophes for elision        (e.g. "da'n")
# The pattern avoids matching lone hyphens or apostrophes.
word_regexes = {
    "en": r"[A-Za-z]+(?:['\-][A-Za-z]+)*",
    "kh": r"[A-Za-zÏïÑñ]+(?:['\-][A-Za-zÏïÑñ]+)*",
}

# ---------------------------------------------------------------------------
# Single-character alphabets used for single-char edit operations
# ---------------------------------------------------------------------------
# Core Khasi letters (lowercase listed first; uppercase added for casing support):
#   Vowels     : a  e  i  ï  o  u
#   Nasals     : m  n  ñ  (ng is a digraph, handled separately)
#   Stops      : b  d  g  k  p  t
#   Aspirated  : bh dh jh kh ph th  (all digraphs, handled separately)
#   Fricatives : s  h
#   Affricates : j
#   Liquids    : l  r
#   Glides     : w  y
# Loanword letters (c, f, q, v, x, z) included for robustness.
alphabets = {
    "en": "abcdefghijklmnopqrstuvwxyz",
    "kh": (
        "abdeghi\u00efjklmn\u00f1oprstuwycfqvxz"
        "ABDEGHI\u00cfJKLMN\u00d1OPRSTUWYCFQVXZ"
    ),
}

# ---------------------------------------------------------------------------
# Khasi digraphs — treated as atomic units during edit-distance operations
# ---------------------------------------------------------------------------
# Sorted longest-first so trigraph candidates are matched before digraphs
# during tokenisation (e.g. "ngi" before "ng").
#
# Full confirmed list:
#   ng  — velar nasal (as in "sing")
#   sh  — voiceless postalveolar fricative
#   kh  — aspirated velar stop
#   ph  — aspirated bilabial stop
#   th  — aspirated alveolar stop
#   bh  — aspirated/breathy bilabial stop
#   dh  — aspirated/breathy alveolar stop
#   jh  — aspirated palatal affricate
#   ie  — falling diphthong (front vowel glide)
khasi_digraphs: list[str] = [
    # trigraphs first (longest match wins)
    "ngi", "nge", "ngm",
    # confirmed digraphs
    "ng", "sh", "kh", "ph", "th", "bh", "dh", "jh", "ie",
]

# Pre-sorted longest-first for use in tokenisation (avoids re-sorting on every call).
khasi_digraphs_sorted: list[str] = sorted(khasi_digraphs, key=len, reverse=True)

# ---------------------------------------------------------------------------
# Khasi morphology
# ---------------------------------------------------------------------------
# Prefixes are highly productive in Khasi and are used to change a word's
# part of speech or add nuances like causation, agency, or continuity.
#
# Each entry maps prefix → frequency_divisor.  The divisor controls how
# conservatively a derived word is scored relative to its root:
#   - Low divisor  (e.g. 2) → derived form scores close to the root
#   - High divisor (e.g. 8) → derived form is scored very conservatively
#
# Prefixes are sorted longest-first so that "nong" is tried before "no",
# and "jai" before "ja", preventing partial matches.
KHASI_PREFIXES: dict[str, int] = {
    # Nominalizer: turns verbs/adjectives into abstract nouns
    # e.g. jingbha "goodness" ← bha "good"
    "jing": 2,

    # Attributive / stative: forms adjectives showing a state
    # e.g. bastad "wise" ← stad "wisdom"
    "ba":   4,

    # Causative: makes a verb causative
    # e.g. pynbha "to make good", pynhiar "to cause to go down"
    "pyn":  2,

    # Agent noun: the person who does the action
    # e.g. nonghikai "teacher" ← hikai "to teach"
    "nong": 2,

    # Continuative: expresses ongoing or continued action
    # e.g. iaileit "to continue going" ← leit "to go"
    "iai":  3,

    # Progressive: expresses an action currently in progress
    # e.g. nangthoh "is writing" ← thoh "to write"
    "nang": 3,

    # Future marker: prefixed to indicate future time reference
    # e.g. lashai "tomorrow"
    "la":   6,

    # Plural / reciprocal: denotes plurality or mutual action
    # e.g. ki ia kren "they are talking to each other"
    "ia":   8,
}

# Suffixes are primarily inflectional (tense, negation).
#
# Each entry maps suffix → set of pronoun roots it validly attaches to.
# An empty set means the suffix is unrestricted (applies to any root).
#
#   -n : future tense on pronouns  — u + n = un "he will"
#   -m : negation on pronouns      — u + m = um "he not / he does not"
KHASI_SUFFIXES: dict[str, set[str]] = {
    "n": {"u", "ka", "i", "phi", "ha", "ki"},
    "m": {"u", "ka", "i", "phi", "ha", "ki"},
}

# Pronoun roots — used to identify and protect inflected pronoun forms
# (e.g. "um", "un", "kam", "kan") from being over-corrected.
KHASI_PRONOUN_ROOTS: set[str] = {"u", "ka", "i", "phi", "ha", "ki"}

# Infix "-yn-": inserted after the first consonant to change a noun to a verb.
# e.g. kjat "leg" → kynhjat "to kick"
#      kshaid "waterfall" → kynshaid "to splash"
# Stored as (infix_string, expected_first_char) pairs — the infix is only
# valid when the word begins with 'k' (the most productive pattern).
KHASI_INFIX_YN: tuple[str, str] = ("yn", "k")

# ---------------------------------------------------------------------------
# IPFS gateway mirrors (fastest first)
# ---------------------------------------------------------------------------
ipfs_gateways: list[str] = [
    "https://cf-ipfs.com/ipfs/",
    "http://ipfs.io/ipfs/",
    "https://gateway.pinata.cloud/ipfs/",
]

# ---------------------------------------------------------------------------
# IPFS content paths per language
# ---------------------------------------------------------------------------
ipfs_paths: dict[str, list[str]] = {
    "en": ["QmbRSZvfJV6zN12zzWhecphcvE9ZBeQdAJGQ9c9ttJXzcg/en.tar.gz"],
    "pl": ["QmbRSZvfJV6zN12zzWhecphcvE9ZBeQdAJGQ9c9ttJXzcg/pl.tar.gz"],
    "ru": ["QmbRSZvfJV6zN12zzWhecphcvE9ZBeQdAJGQ9c9ttJXzcg/ru.tar.gz"],
    "uk": ["QmbRSZvfJV6zN12zzWhecphcvE9ZBeQdAJGQ9c9ttJXzcg/uk.tar.gz"],
    "tr": ["QmbRSZvfJV6zN12zzWhecphcvE9ZBeQdAJGQ9c9ttJXzcg/tr.tar.gz"],
    "es": ["QmbRSZvfJV6zN12zzWhecphcvE9ZBeQdAJGQ9c9ttJXzcg/es.tar.gz"],
    "cs": ["QmbRSZvfJV6zN12zzWhecphcvE9ZBeQdAJGQ9c9ttJXzcg/cs.tar.gz"],
    "pt": ["QmbRSZvfJV6zN12zzWhecphcvE9ZBeQdAJGQ9c9ttJXzcg/pt.tar.gz"],
    "el": ["QmbRSZvfJV6zN12zzWhecphcvE9ZBeQdAJGQ9c9ttJXzcg/el.tar.gz"],
    "it": ["QmbRSZvfJV6zN12zzWhecphcvE9ZBeQdAJGQ9c9ttJXzcg/it.tar.gz"],
    "fr": ["QmPRNDmUDTXikq8gWnGcw3ZGmnoBfvekmAyeyX8y6onf23/fr.tar.gz"],
    "vi": ["QmRRJj5i7nkpzTRSKhFe23XMjLRw7f2zD6FLKDrRfzco7f/vi.tar.gz"],
    # Khasi — place your kh.tar.gz IPFS CID here once published
    "kh": [],
}

# ---------------------------------------------------------------------------
# HTTP backup download URLs per language
# ---------------------------------------------------------------------------
backup_urls: dict[str, list[str]] = {
    "en": ["https://dl.dropboxusercontent.com/s/grxjmtw4db814g1/en.tar.gz?dl=0"],
    "pl": ["https://dl.dropboxusercontent.com/s/40orabi1l3dfqpp/pl.tar.gz?dl=0"],
    "ru": [
        "https://dl.dropboxusercontent.com/s/mpas7xqn8yl3wej/ru.tar.gz?dl=0",
        "https://dl.dropboxusercontent.com/s/6tzfxy34xx34mm7/ru.tar.gz?dl=0",
    ],
    "uk": [
        "https://dl.dropboxusercontent.com/s/s64ot0l4lj3a0ec/uk.tar.gz?dl=0",
        "https://dl.dropboxusercontent.com/s/b76p4sc1lld96lw/uk.tar.gz?dl=0",
    ],
    "tr": [
        "https://dl.dropboxusercontent.com/s/mj2d3t158ucwhwx/tr.tar.gz?dl=0",
        "https://dl.dropboxusercontent.com/s/1wy01nq5fpq8iay/tr.tar.gz?dl=0",
    ],
    "es": [
        "https://dl.dropboxusercontent.com/s/jh0212sou1qbs7t/es.tar.gz?dl=0",
        "https://dl.dropboxusercontent.com/s/k6g5vj3x0rx7mjz/es.tar.gz?dl=0",
    ],
    "cs": [
        "https://dl.dropboxusercontent.com/s/8ptuuh8kcr3kufy/cs.tar.gz?dl=0",
        "https://dl.dropboxusercontent.com/s/369wplqb0w2ax21/cs.tar.gz?dl=0",
    ],
    "pt": ["https://dl.dropboxusercontent.com/s/6xnko882tsjgeaw/pt.tar.gz?dl=0"],
    "el": ["https://dl.dropboxusercontent.com/s/2zdewe1p1od9vu0/el.tar.gz?dl=0"],
    "it": ["https://dl.dropboxusercontent.com/s/6xci1wfb387zk23/it.tar.gz?dl=0"],
    "fr": ["https://mega.nz/file/kQByQJAb#rMbmF0HG09MLQQ-FDafHrPAgXigJIpmC1zhtxRMp2dQ"],
    # Add your Khasi dictionary backup URL here
    "kh": [],
}
