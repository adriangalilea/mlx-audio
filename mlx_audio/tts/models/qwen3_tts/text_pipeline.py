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

# Decimal separator word per language. num2words' built-in float handling
# defaults to "punto" / "point" / digit-by-digit, which is not how natives
# read decimals — Spanish uses "coma", French uses "virgule", etc. We split
# the integer and fractional parts ourselves and join with this word.
_DECIMAL_WORD: Dict[str, str] = {
    "spanish":    "coma",
    "english":    "point",
    "french":     "virgule",
    "german":     "Komma",
    "italian":    "virgola",
    "portuguese": "vírgula",
    "russian":    "запятая",
    "japanese":   "テン",
    "korean":     "쩜",
    "chinese":    "点",
}

# "X percent" expansion per language. Applied before number expansion so the
# bare digits get spelled out by num2words afterwards.
_PERCENT_WORD: Dict[str, str] = {
    "spanish":    "por ciento",
    "english":    "percent",
    "french":     "pour cent",
    "german":     "Prozent",
    "italian":    "per cento",
    "portuguese": "por cento",
    "russian":    "процентов",
    "japanese":   "パーセント",
    "korean":     "퍼센트",
    "chinese":    "百分之",
}

# Month names for date expansion (numeric → spoken). Indexed 1..12.
# Sentinel at index 0 keeps the access ergonomic.
_MONTH_NAMES: Dict[str, List[str]] = {
    "spanish": ["", "enero", "febrero", "marzo", "abril", "mayo", "junio",
                "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"],
    "english": ["", "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"],
    "french":  ["", "janvier", "février", "mars", "avril", "mai", "juin",
                "juillet", "août", "septembre", "octobre", "novembre", "décembre"],
    "german":  ["", "Januar", "Februar", "März", "April", "Mai", "Juni",
                "Juli", "August", "September", "Oktober", "November", "Dezember"],
    "italian": ["", "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
                "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"],
    "portuguese": ["", "janeiro", "fevereiro", "março", "abril", "maio", "junho",
                   "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"],
}

# Date phrasing per language: how to glue "day {} month {} year" together.
# Tuple: (between_day_and_month, between_month_and_year).
_DATE_GLUE: Dict[str, Tuple[str, str]] = {
    "spanish":    (" de ", " de "),
    "english":    (" ", ", "),
    "french":     (" ", " "),
    "german":     (". ", " "),
    "italian":    (" ", " "),
    "portuguese": (" de ", " de "),
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

# Compound patterns that must run BEFORE the bare number expander, because
# they wrap multiple digit groups into a single linguistic unit (a date is
# not three separate numbers, a time is not two separate numbers).
_DATE_DMY = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
_DATE_ISO = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_TIME_HM = re.compile(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b")
_PERCENT = re.compile(r"(\d+(?:[.,]\d+)*)\s*%")

# Number matcher. Matches digit runs with optional thousands separators and
# decimal mark. The decimal/thousands convention differs by language and is
# resolved per call in _expand_numbers.
_NUMBER = re.compile(r"\d[\d.,]*\d|\d")

# End-of-sentence punctuation we restore when abbreviation expansion eats it.
_SENT_TERMINAL = ".!?"


# ---------------------------------------------------------------------------
# Cleaners
# ---------------------------------------------------------------------------


def _strip_emojis(text: str) -> str:
    """Remove emoji, pictograph, dingbat, and zero-width formatting codepoints.

    Uses the Unicode general-category database (``unicodedata``) so we don't
    need an external emoji table. ``So`` covers most emoji; ``Sk`` catches
    skin-tone modifiers; visible emoji codepoints are replaced with a space
    so that "wow🔥cool" becomes "wow cool" rather than "wowcool" (the
    surrounding whitespace cleanup pass will collapse runs back to one).
    Variation selectors and zero-width glue codepoints are dropped silently —
    they're invisible by definition and don't need a space replacement.
    """
    out: List[str] = []
    for c in text:
        cat = unicodedata.category(c)
        if cat in ("So", "Sk"):
            out.append(" ")
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


def _expand_percent(text: str, lang: str) -> str:
    """Replace ``N%`` with ``N <percent_word>`` so the bare digits get
    spelled out by the number-expansion pass that follows.

    Falls back to English ("percent") when the language has no entry.
    """
    word = _PERCENT_WORD.get(lang, _PERCENT_WORD["english"])
    return _PERCENT.sub(rf"\1 {word}", text)


def _expand_abbreviations(text: str, lang: str) -> str:
    """Apply per-language abbreviation expansion, preserving final punctuation.

    The abbreviation regex consumes its trailing period (``\\.`` is part of the
    match), which would silently swallow end-of-sentence terminators —
    "Vivo en EE.UU." → "Vivo en Estados Unidos" loses the ``.`` even though the
    original ended a sentence. We capture the final terminator before the pass
    and restore it afterward if the expansion ate it.
    """
    table = _ABBREVIATIONS.get(lang)
    if not table:
        return text

    stripped = text.rstrip()
    final = stripped[-1] if stripped else ""
    had_terminator = final in _SENT_TERMINAL
    trailing_ws = text[len(stripped):] if had_terminator else ""

    for pattern, replacement in table.items():
        text = re.sub(pattern, replacement, text)

    if had_terminator and not text.rstrip().endswith(final):
        text = text.rstrip() + final + trailing_ws

    return text


def _expand_dates(text: str, lang: str) -> str:
    """Expand DD/MM/YYYY and YYYY-MM-DD date patterns to spoken form.

    Only covers languages with a populated month name table; for the rest
    the date passes through and gets read digit-group-by-digit-group.

    DD/MM/YYYY is the European convention; the US would write MM/DD/YYYY.
    Since we only run this normalizer when ``lang`` is set, and the languages
    in our table all use DD/MM/YYYY, we don't try to disambiguate.
    """
    months = _MONTH_NAMES.get(lang)
    if not months:
        return text
    sep_dm, sep_my = _DATE_GLUE.get(lang, (" ", " "))

    def _expand_dmy(m: "re.Match[str]") -> str:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not (1 <= d <= 31 and 1 <= mo <= 12):
            return m.group(0)
        return f"{d}{sep_dm}{months[mo]}{sep_my}{y}"

    def _expand_iso(m: "re.Match[str]") -> str:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not (1 <= d <= 31 and 1 <= mo <= 12):
            return m.group(0)
        return f"{d}{sep_dm}{months[mo]}{sep_my}{y}"

    text = _DATE_DMY.sub(_expand_dmy, text)
    text = _DATE_ISO.sub(_expand_iso, text)
    return text


def _expand_times(text: str, lang: str) -> str:
    """Expand ``HH:MM`` (and ``HH:MM:SS``) to natural spoken form.

    Two passes:
      1. Common fractional minutes (``:15``, ``:30``) get the natural phrase
         a native speaker would use ("y media", "y cuarto" / "et quart" /
         "e mezza" / "e meia"). Only applies when there are no seconds
         (HH:MM:SS) — for full timestamps the natural readings break down.
      2. Everything else falls back to ``HH<conjunction>MM``. When MM == 00
         and there's no SS, only ``HH`` is emitted, since the surrounding
         text usually has its own qualifier ("en punto", "o'clock").

    German is intentionally NOT special-cased — German "halb elf" anchors
    to the next hour (10:30 → "halb elf"), and getting that wrong is more
    jarring than reading "zehn dreißig" verbatim.

    The minute group must be exactly 2 digits to avoid clashing with
    non-time constructs (``2:5`` is not a time; ``15:7`` is unusual).
    Hours are 0-29 to allow 24h schedules; out-of-range matches pass
    through unchanged.
    """
    # Per-language quarter/half phrases. The phrase already contains the
    # connective ("y", "et", "e") so it's appended with a single space rather
    # than via the join token below. Empty dict means "no special handling".
    fractional = {
        "spanish":    {15: "y cuarto", 30: "y media"},
        "french":     {15: "et quart", 30: "et demie"},
        "italian":    {15: "e un quarto", 30: "e mezza"},
        "portuguese": {15: "e quinze",   30: "e meia"},
    }.get(lang, {})

    # Per-language conjunction for the fallback HH<join>MM form.
    join = {
        "spanish":    " y ",
        "english":    " ",
        "french":     " ",
        "german":     " ",
        "italian":    " e ",
        "portuguese": " e ",
    }.get(lang, " ")

    def _expand(m: "re.Match[str]") -> str:
        h, mn = int(m.group(1)), int(m.group(2))
        s = m.group(3)
        if not (0 <= h <= 29 and 0 <= mn <= 59):
            return m.group(0)
        if mn == 0 and s is None:
            return str(h)
        if mn in fractional and s is None:
            return f"{h} {fractional[mn]}"
        out = f"{h}{join}{mn:02d}"
        if s is not None:
            ss = int(s)
            if 0 <= ss <= 59:
                out += f"{join}{ss:02d}"
        return out

    return _TIME_HM.sub(_expand, text)


def _expand_numbers(text: str, lang: str) -> str:
    """Expand numeric tokens to spoken words via num2words.

    Resolves locale-specific separators: Spanish/German/French/Italian/
    Portuguese/Russian use period for thousands and comma for decimal;
    English uses the opposite. Decimals are NOT delegated to num2words'
    built-in float handling — that path tends to produce digit-by-digit
    fractional readings ("cero punto siete cinco") instead of natural
    cardinal readings ("cero coma setenta y cinco"). We split integer and
    fractional parts ourselves and join with the language-specific
    decimal word (see ``_DECIMAL_WORD``).

    Fractional parts up to 3 digits get the cardinal reading
    ("setenta y cinco"); longer ones default to digit-by-digit
    ("uno cuatro uno cinco nueve") which is how mathematicians read them.

    Tokens that fail to parse pass through unchanged.
    """
    if not _HAS_NUM2WORDS:
        return text
    iso = _NUM2WORDS_LANG.get(lang)
    if iso is None:
        return text

    use_comma_decimal = lang in (
        "spanish", "german", "french", "italian", "portuguese", "russian"
    )
    decimal_word = _DECIMAL_WORD.get(lang, "point")

    def _spell_int(n: int) -> str:
        try:
            return num2words(n, lang=iso)
        except (NotImplementedError, ValueError):
            return str(n)

    def _replace(m: "re.Match[str]") -> str:
        tok = m.group(0)
        # Single bare digit shortcut — keep cheap path
        if len(tok) == 1:
            return _spell_int(int(tok))

        if use_comma_decimal:
            # period = thousands, comma = decimal
            normalized = tok.replace(".", "").replace(",", ".")
        else:
            # comma = thousands, period = decimal
            normalized = tok.replace(",", "")

        # Reject ambiguous tokens (multiple decimal points after cleaning)
        if normalized.count(".") > 1:
            return tok

        # Split integer/fractional parts manually — bypass num2words' float
        # path so we control how the decimal joiner reads.
        if "." in normalized:
            int_part_str, frac_part_str = normalized.split(".", 1)
            if not int_part_str:
                int_part_str = "0"
            try:
                int_part = int(int_part_str)
            except ValueError:
                return tok

            # Strip trailing zeros so "3,50" reads "tres coma cinco" not
            # "tres coma cincuenta" (which would suggest 3,5 == 3,50).
            # Note: this changes semantics. Disable if mathematical fidelity
            # matters more than naturalness.
            frac_trimmed = frac_part_str.rstrip("0") or "0"

            int_words = _spell_int(int_part)
            if len(frac_trimmed) <= 3:
                # Cardinal reading for short fractions
                try:
                    frac_words = _spell_int(int(frac_trimmed))
                except ValueError:
                    return tok
            else:
                # Digit-by-digit for long fractions (mathematical reading)
                frac_words = " ".join(_spell_int(int(c)) for c in frac_trimmed)

            return f"{int_words} {decimal_word} {frac_words}"

        # Pure integer
        try:
            return _spell_int(int(normalized))
        except ValueError:
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
        # Order matters: dates/times/percent must consume their compound
        # patterns *before* the bare number expander pulls digits apart.
        # E.g. "10:30" must be matched as a single time, not as "10" and
        # "30" separately joined by a literal colon.
        text = _expand_dates(text, lang)
        text = _expand_times(text, lang)
        text = _expand_currency(text, lang)
        text = _expand_percent(text, lang)
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
