"""
preprocessing.py
Text cleaning pipeline for the phishing email detector.

Pipeline (per spec):
  1. Strip HTML
  2. Strip URLs
  3. Strip email addresses
  4. Strip digits / special characters
  5. Lowercase
  6. Tokenize
  7. Remove stop-words
  8. Lemmatize
"""

import re

# --------------------------------------------------------------------------
# NLTK is used when its data is available locally. If it isn't (e.g. no
# internet access at setup time), we fall back to a small built-in
# stop-word list and a lightweight suffix-stripping "lemmatizer" so the
# pipeline still runs end-to-end without crashing.
# --------------------------------------------------------------------------
_NLTK_READY = False
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.data.find(
                f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}"
            )
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass

    STOP_WORDS = set(stopwords.words("english"))
    _lemmatizer = WordNetLemmatizer()
    _NLTK_READY = True
except Exception:
    _NLTK_READY = False

# Fallback stop-word list (used only if NLTK data could not be loaded)
_FALLBACK_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "to", "of",
    "in", "on", "at", "for", "with", "as", "by", "is", "are", "was", "were",
    "be", "been", "being", "this", "that", "these", "those", "it", "its",
    "i", "you", "he", "she", "we", "they", "your", "my", "our", "their",
    "from", "have", "has", "had", "not", "no", "do", "does", "did", "will",
    "would", "can", "could", "should", "just", "than", "too", "very",
    "please", "up", "out", "about", "into", "over", "again",
}

_HTML_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")


def _simple_stem(word: str) -> str:
    """Crude suffix stripper used only when NLTK/WordNet is unavailable."""
    for suffix in ("ing", "edly", "ed", "ly", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def clean_text(text: str) -> str:
    """Run the full cleaning + normalization pipeline and return a single
    space-joined string of processed tokens, ready for the TF-IDF
    vectorizer."""
    if not text:
        return ""

    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = text.lower()
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()

    if _NLTK_READY:
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
        tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    else:
        tokens = text.split()
        tokens = [t for t in tokens if t not in _FALLBACK_STOP_WORDS and len(t) > 1]
        tokens = [_simple_stem(t) for t in tokens]

    return " ".join(tokens)


# A small set of words that, if present in the *raw* email, are surfaced to
# the user as "notable signals" in the API response. This is independent of
# the model itself (which uses TF-IDF over the full cleaned text) and is
# only for the human-readable explanation shown in the UI.
_SIGNAL_WORDS = [
    "urgent", "verify", "suspended", "click", "confirm", "password",
    "account", "bank", "security", "update", "immediately", "winner",
    "prize", "free", "limited", "act now", "expire", "login", "ssn",
    "social security", "wire transfer", "gift card", "invoice", "refund",
]


def extract_signals(raw_text: str, top_n: int = 5) -> list:
    """Return up to top_n notable phishing-style keywords found in the raw
    (uncleaned) email text, preserving a sensible display order."""
    if not raw_text:
        return []
    lowered = raw_text.lower()
    found = [w for w in _SIGNAL_WORDS if w in lowered]
    return found[:top_n]
