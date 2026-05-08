# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

"""Text preparation pipeline for Qwen3-TTS.

Three concerns, all language-aware, all opt-out:

1. Normalization (``normalize_text``) — strip the parts a TTS model can't
   pronounce sensibly (emojis, markdown formatting markers, URLs, raw
   currency symbols) and expand the parts it would mispronounce
   character-by-character (numbers via num2words, common abbreviations
   like "Sr." → "señor"). Defaults are tuned for chat / LLM-output content
   — the typical input shape for a Qwen3-TTS deployment.

2. Sentence splitting (``split_sentences``) — chunk a long input into
   TTS-friendly segments. ICL-style autoregressive TTS models (Qwen3-TTS
   included) start drifting in timbre, accent, and pronunciation past
   ~25-30 seconds of synthesized audio per call; chunking + per-chunk
   regeneration is the difference between "demoable" and "deployable".

3. Failure heuristics (``estimate_duration``, ``detect_failure``) and
   stitching (``crossfade_concat``) — the building blocks the chunked
   generation path uses to decide whether a chunk is acceptable, and to
   join multiple accepted chunks without seams.

The 10 supported languages match Qwen3-TTS's ``codec_language_id`` map:
chinese, english, french, german, italian, japanese, korean, portuguese,
russian, spanish.

Per-language coverage is best-effort and grows over time. Spanish and
English are well-tuned because we run them in production; others are
correct in shape but the abbreviation dictionaries are minimal. PRs to
expand them are welcome.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from num2words import num2words

    _HAS_NUM2WORDS = True
except ImportError:  # pragma: no cover - graceful degradation
    _HAS_NUM2WORDS = False


# ---------------------------------------------------------------------------
# Language tables
# ---------------------------------------------------------------------------

# Map Qwen3-TTS language names to num2words ISO codes. num2words supports
# 25+ languages; this is the subset Qwen3-TTS itself supports. Chinese is
# omitted because num2words renders Chinese numbers in pinyin words rather
# than the natural form a Chinese speaker would say (e.g. "三千" vs "san qian
# ling"). For Chinese the digit-by-digit fallback is closer to natural.
_NUM2WORDS_LANG: Dict[str, str] = {
    "spanish": "es",
    "english": "en",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "japanese": "ja",
    "korean": "ko",
    "russian": "ru",
}

# Currency symbol → spoken word, per language. Symbols are matched both as
# prefix ("$10") and suffix ("10€") because conventions vary.
_CURRENCY: Dict[str, Dict[str, str]] = {
    "spanish":    {"€": "euros", "$": "dólares", "£": "libras",   "¥": "yenes"},
    "english":    {"€": "euros", "$": "dollars", "£": "pounds",   "¥": "yen"},
    "french":     {"€": "euros", "$": "dollars", "£": "livres",   "¥": "yens"},
    "german":     {"€": "Euro",  "$": "Dollar",  "£": "Pfund",    "¥": "Yen"},
    "italian":    {"€": "euro",  "$": "dollari", "£": "sterline", "¥": "yen"},
    "portuguese": {"€": "euros", "$": "dólares", "£": "libras",   "¥": "ienes"},
    "japanese":   {"€": "ユーロ", "$": "ドル",    "£": "ポンド",    "¥": "円"},
    "korean":     {"€": "유로",   "$": "달러",    "£": "파운드",    "¥": "엔"},
    "chinese":    {"€": "欧元",   "$": "美元",    "£": "英镑",      "¥": "元"},
    "russian":    {"€": "евро",  "$": "долларов","£": "фунтов",    "¥": "иен"},
}

# Per-language abbreviation expansion. Keys are word-boundary regex patterns,
# values are spoken-form replacements. Conservative dictionaries — only the
# expansions whose context is unambiguous regardless of surrounding text.
# Long lists hurt more than help (false-positive expansions are jarring).
_ABBREVIATIONS: Dict[str, Dict[str, str]] = {
    "spanish": {
        r"\bSr\.": "señor",
        r"\bSra\.": "señora",
        r"\bSrta\.": "señorita",
        r"\bDr\.": "doctor",
        r"\bDra\.": "doctora",
        r"\bD\.": "don",
        r"\bDña\.": "doña",
        r"\bUd\.": "usted",
        r"\bUds\.": "ustedes",
        r"\betc\.": "etcétera",
        r"\bp\.\s?ej\.": "por ejemplo",
        r"\bEE\.\s?UU\.": "Estados Unidos",
        r"\bS\.\s?A\.": "sociedad anónima",
        r"\bS\.\s?L\.": "sociedad limitada",
        r"\bvs\.": "contra",
    },
    "english": {
        r"\bMr\.": "Mister",
        r"\bMrs\.": "Misses",
        r"\bMs\.": "Miss",
        r"\bDr\.": "Doctor",
        r"\bSt\.": "Saint",
        r"\bJr\.": "Junior",
        r"\bSr\.": "Senior",
        r"\betc\.": "et cetera",
        r"\be\.g\.": "for example",
        r"\bi\.e\.": "that is",
        r"\bvs\.": "versus",
        r"\bU\.S\.A\.": "United States",
        r"\bU\.K\.": "United Kingdom",
        r"\bU\.S\.": "United States",
    },
    "french": {
        r"\bM\.": "monsieur",
        r"\bMme\.": "madame",
        r"\bMlle\.": "mademoiselle",
        r"\bDr\.": "docteur",
        r"\betc\.": "et cetera",
        r"\bp\.\s?ex\.": "par exemple",
    },
    "german": {
        r"\bHr\.": "Herr",
        r"\bFr\.": "Frau",
        r"\bDr\.": "Doktor",
        r"\bz\.\s?B\.": "zum Beispiel",
        r"\bd\.\s?h\.": "das heißt",
        r"\busw\.": "und so weiter",
    },
    "italian": {
        r"\bSig\.": "signore",
        r"\bSig\.ra": "signora",
        r"\bDr\.": "dottore",
        r"\becc\.": "eccetera",
        r"\bes\.": "esempio",
    },
    "portuguese": {
        r"\bSr\.": "senhor",
        r"\bSra\.": "senhora",
        r"\bDr\.": "doutor",
        r"\bDra\.": "doutora",
        r"\betc\.": "et cetera",
    },
}

# Heuristic synthesis rate per language: characters of input text per second
# of generated audio at speed=1.0. Used for duration sanity checks (regen
# decisions in the chunked generation pipeline). Numbers come from measuring
# Qwen3-TTS Base output on Genesis 1:1-5 across the 6 languages we have
# samples for; entries for the rest are interpolated. They're rough — the
# detect_failure thresholds are 0.4x/2.5x to give wide tolerance.
_CHARS_PER_SEC: Dict[str, float] = {
    "spanish":    14.0,
    "english":    14.5,
    "french":     14.0,
    "german":     13.0,
    "italian":    14.0,
    "portuguese": 14.0,
    "russian":    13.0,
    "japanese":   7.0,
    "korean":     8.0,
    "chinese":    5.0,
}


# ---------------------------------------------------------------------------
# Compiled patterns (module-level so we don't recompile per call)
# ---------------------------------------------------------------------------

_URL_PATTERN = re.compile(
    r"https?://\S+|www\.[A-Za-z0-9.-]+\.[A-Za-z]{2,}\S*",
    re.IGNORECASE,
)
_EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
)

# Markdown: handle code blocks first (they shadow inline patterns).
_MD_CODE_BLOCK = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_MD_INLINE_CODE = re.compile(r"`+([^`]+)`+")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
# Bold/italic/strike: keep inner text. Order matters: longest delim first.
_MD_BOLD_ITALIC = re.compile(r"(\*\*\*|\*\*|\*|_{1,3}|~~)(.+?)\1")
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_MD_BLOCKQUOTE = re.compile(r"^\s{0,3}>\s+", re.MULTILINE)
_MD_HR = re.compile(r"^\s*[-*_]{3,}\s*$", re.MULTILINE)
_MD_LIST_BULLET = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_MD_LIST_NUM = re.compile(r"^\s*\d+\.\s+", re.MULTILINE)

_WHITESPACE = re.compile(r"\s+")

# Number matcher. Matches digit runs with optional thousands separators and
# decimal mark. The decimal/thousands convention differs by language and is
# resolved per call in _expand_numbers.
_NUMBER = re.compile(r"\d[\d.,]*\d|\d")


# ---------------------------------------------------------------------------
# Cleaners
# ---------------------------------------------------------------------------


def _strip_emojis(text: str) -> str:
    """Remove emoji, pictograph, dingbat, and zero-width formatting codepoints.

    Uses the Unicode general-category database (``unicodedata``) so we don't
    need an external emoji table. ``So`` covers most emoji; ``Sk`` catches
    skin-tone modifiers; we also drop variation selectors (U+FE00..FE0F),
    zero-width joiner, and similar invisible glue codepoints that emoji
    sequences are built from.
    """
    out: List[str] = []
    for c in text:
        cat = unicodedata.category(c)
        if cat in ("So", "Sk"):
            continue
        cp = ord(c)
        if 0xFE00 <= cp <= 0xFE0F:  # variation selectors
            continue
        if cp in (0x200D, 0x2060, 0xFEFF):  # ZWJ, WJ, BOM
            continue
        out.append(c)
    return "".join(out)


def _strip_urls(text: str) -> str:
    """Remove URLs (http/https/www) and email addresses entirely."""
    text = _URL_PATTERN.sub(" ", text)
    text = _EMAIL_PATTERN.sub(" ", text)
    return text


def _strip_markdown(text: str) -> str:
    """Strip Markdown formatting markers, keeping the underlying text.

    Code blocks are removed entirely (they're rarely speech-friendly).
    Inline code keeps its contents. Links/images keep their alt text. Bold,
    italic, and strikethrough markers are stripped while preserving content.
    Heading/blockquote/list markers are removed; horizontal rules are dropped.
    """
    text = _MD_CODE_BLOCK.sub(" ", text)
    text = _MD_IMAGE.sub(r"\1", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_INLINE_CODE.sub(r"\1", text)
    text = _MD_BOLD_ITALIC.sub(r"\2", text)
    text = _MD_HEADING.sub("", text)
    text = _MD_BLOCKQUOTE.sub("", text)
    text = _MD_HR.sub("", text)
    text = _MD_LIST_BULLET.sub("", text)
    text = _MD_LIST_NUM.sub("", text)
    return text


# ---------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------


def _expand_currency(text: str, lang: str) -> str:
    """Replace currency symbols with their spoken word.

    Matches the symbol whether it appears before ("$10") or after ("10€")
    a number, and inserts a space so the number-expansion pass can pick up
    the digits cleanly.
    """
    cur = _CURRENCY.get(lang) or _CURRENCY["english"]
    for sym, word in cur.items():
        esc = re.escape(sym)
        text = re.sub(rf"(\d+(?:[.,]\d+)*)\s*{esc}", rf"\1 {word}", text)
        text = re.sub(rf"{esc}\s*(\d+(?:[.,]\d+)*)", rf"\1 {word}", text)
    return text


def _expand_abbreviations(text: str, lang: str) -> str:
    """Apply per-language abbreviation expansion (case-sensitive regex)."""
    table = _ABBREVIATIONS.get(lang)
    if not table:
        return text
    for pattern, replacement in table.items():
        text = re.sub(pattern, replacement, text)
    return text


def _expand_numbers(text: str, lang: str) -> str:
    """Expand numeric tokens to spoken words via num2words.

    Resolves locale-specific separators: Spanish/German/French/Italian/
    Portuguese use period for thousands and comma for decimal; English uses
    the opposite. Falls back to leaving the original token in place if
    parsing fails or num2words doesn't support the language.
    """
    if not _HAS_NUM2WORDS:
        return text
    iso = _NUM2WORDS_LANG.get(lang)
    if iso is None:
        return text

    use_comma_decimal = lang in (
        "spanish", "german", "french", "italian", "portuguese", "russian"
    )

    def _replace(m: "re.Match[str]") -> str:
        tok = m.group(0)
        # Single bare digit shortcut — keep cheap path
        if len(tok) == 1:
            try:
                return num2words(int(tok), lang=iso)
            except (NotImplementedError, ValueError):
                return tok

        if use_comma_decimal:
            # period = thousands, comma = decimal
            cleaned = tok.replace(".", "").replace(",", ".")
        else:
            # comma = thousands, period = decimal
            cleaned = tok.replace(",", "")

        # Reject ambiguous tokens (multiple decimal points after cleaning)
        if cleaned.count(".") > 1:
            return tok

        try:
            num = float(cleaned)
        except ValueError:
            return tok

        if num.is_integer():
            num = int(num)
        try:
            return num2words(num, lang=iso)
        except (NotImplementedError, ValueError):
            return tok

    return _NUMBER.sub(_replace, text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_text(
    text: str,
    lang: str = "english",
    *,
    strip_emojis: bool = True,
    strip_urls: bool = True,
    strip_markdown: bool = True,
    expand_abbreviations: bool = True,
    expand_numbers: bool = True,
    extra_abbreviations: Optional[Dict[str, str]] = None,
) -> str:
    """Normalize text for TTS synthesis.

    Designed for chat / LLM-output content: strips formatting, emojis, and
    URLs; expands numbers and common abbreviations to their spoken form.
    Order is significant — strips run before expansions so that, e.g., a
    URL containing digits doesn't get half-expanded by num2words.

    Args:
        text: Input text.
        lang: Qwen3-TTS language name (``spanish``, ``english``, ...).
        strip_emojis: Drop emoji and pictograph codepoints.
        strip_urls: Drop URLs and email addresses.
        strip_markdown: Strip Markdown formatting markers (keeps inner text).
        expand_abbreviations: Apply the per-language abbreviation dictionary.
        expand_numbers: Expand digits and currency to spoken words.
        extra_abbreviations: Additional regex→replacement mappings applied
            after the built-in dictionary (typically domain glossaries from
            the consumer side).

    Returns:
        Normalized text, ready to feed to the TTS model.
    """
    if strip_markdown:
        text = _strip_markdown(text)
    if strip_urls:
        text = _strip_urls(text)
    if strip_emojis:
        text = _strip_emojis(text)
    if expand_abbreviations:
        text = _expand_abbreviations(text, lang)
        if extra_abbreviations:
            for pattern, replacement in extra_abbreviations.items():
                text = re.sub(pattern, replacement, text)
    if expand_numbers:
        text = _expand_currency(text, lang)
        text = _expand_numbers(text, lang)

    text = _WHITESPACE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

# Latin-script terminators. We split on whitespace AFTER terminator+optional
# closing punctuation. The lookbehind keeps the punctuation attached to the
# preceding sentence (where TTS prosody expects it).
_SENT_LATIN = re.compile(r"(?<=[.!?])[\"')\]]?\s+")
# Spanish opens questions/exclamations with ¿¡; we still split on closing
# punctuation, but tolerate the openers as start-of-sentence markers.
_SENT_LATIN_ES = _SENT_LATIN
# CJK terminators have no following whitespace requirement.
_SENT_CJK = re.compile(r"(?<=[。！？])\s*")


def split_sentences(text: str, lang: str = "english") -> List[str]:
    """Split text into sentence-sized chunks suitable for chunked TTS.

    Per-language regex; conservative on edge cases (does not parse
    abbreviations, decimal numbers, etc. — those are expected to have been
    normalized first via :func:`normalize_text`).

    The output is intended to be fed one segment at a time into the model
    so each call stays under the ~25-30s decay window of the autoregressive
    decoder. Empty / whitespace-only segments are dropped.
    """
    if not text.strip():
        return []
    if lang in ("chinese", "japanese"):
        sents = _SENT_CJK.split(text)
    elif lang == "spanish":
        sents = _SENT_LATIN_ES.split(text)
    else:
        sents = _SENT_LATIN.split(text)
    return [s.strip() for s in sents if s.strip()]


# ---------------------------------------------------------------------------
# Failure heuristics & stitching
# ---------------------------------------------------------------------------


def estimate_duration(text: str, lang: str = "english", speed: float = 1.0) -> float:
    """Estimate expected speech duration in seconds for ``text``.

    Uses a per-language character-rate constant (see ``_CHARS_PER_SEC``).
    Rough; consumed only by :func:`detect_failure` which has wide tolerance.
    """
    rate = _CHARS_PER_SEC.get(lang, 14.0)
    if rate <= 0 or not text:
        return 0.0
    return len(text) / rate / max(speed, 0.1)


def detect_failure(
    audio_duration: float,
    expected_duration: float,
    *,
    hit_token_cap: bool,
    short_ratio: float = 0.4,
    long_ratio: float = 2.5,
) -> Tuple[bool, str]:
    """Heuristic check: did the generation likely fail?

    Three signals:
    - ``hit_token_cap``: the decoder ran into max_tokens before EOS. Almost
      always indicates a loop or infinite hallucination.
    - duration too short: the audio is much shorter than the text would
      take to read aloud (cut-off / EOS too early).
    - duration too long: the audio is much longer than expected (loop or
      hallucinated tail).

    Defaults are wide on purpose: we want this to fire only on clear breakage,
    not on natural prosodic variation. Consumers can tighten the ratios for
    stricter QA.
    """
    if hit_token_cap:
        return True, "hit max_tokens (likely loop)"
    if expected_duration > 0:
        ratio = audio_duration / expected_duration
        if ratio < short_ratio:
            return (
                True,
                f"audio too short ({audio_duration:.1f}s vs ~{expected_duration:.1f}s expected, "
                f"ratio {ratio:.2f} < {short_ratio:.2f})",
            )
        if ratio > long_ratio:
            return (
                True,
                f"audio too long ({audio_duration:.1f}s vs ~{expected_duration:.1f}s expected, "
                f"ratio {ratio:.2f} > {long_ratio:.2f})",
            )
    return False, ""


def crossfade_concat(
    chunks: List[np.ndarray],
    sample_rate: int,
    fade_ms: int = 50,
) -> np.ndarray:
    """Concatenate audio chunks with a linear crossfade between adjacent pairs.

    Plain concatenation produces audible clicks because chunk boundaries don't
    end / start on zero-crossings. A short linear crossfade (default 50ms)
    eliminates this without losing perceptible content. Set ``fade_ms=0`` to
    disable the crossfade and concatenate directly.
    """
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0].astype(np.float32, copy=False)

    fade_samples = int(sample_rate * fade_ms / 1000) if fade_ms > 0 else 0
    out = chunks[0].astype(np.float32, copy=True)

    for chunk in chunks[1:]:
        chunk = chunk.astype(np.float32, copy=False)
        if fade_samples > 0 and len(out) >= fade_samples and len(chunk) >= fade_samples:
            fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
            mixed = out[-fade_samples:] * fade_out + chunk[:fade_samples] * fade_in
            out = np.concatenate([out[:-fade_samples], mixed, chunk[fade_samples:]])
        else:
            out = np.concatenate([out, chunk])

    return out
