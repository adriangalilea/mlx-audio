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
from urllib.parse import urlparse

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

# Per-language word for the dot in vocalized hostnames ("example punto com")
# and for the @ in vocalized email addresses ("foo arroba bar"). Distinct
# from _DECIMAL_WORD which is what gets read for numeric decimals — Spanish
# uses "punto" for URL dots but "coma" for decimal numbers. Native readers
# really do say different words for the same character in different
# contexts.
_URL_DOT_WORD: Dict[str, str] = {
    "spanish":    "punto",
    "english":    "dot",
    "french":     "point",
    "german":     "Punkt",
    "italian":    "punto",
    "portuguese": "ponto",
    "japanese":   "ドット",
    "korean":     "점",
    "chinese":    "点",
    "russian":    "точка",
}
# Letter names per language for letter-by-letter spelling (used inside
# vocalized hostnames for "www" / "ftp" / etc.). When a language has no
# entry, the letter is emitted bare and the TTS pronounces it however its
# training picked up — fine for English ("dub-ya dub-ya dub-ya"), but
# Spanish reads bare "w" as "uu" or worse, so we explicitly spell it as
# "uve doble" (RAE official). Only entries that meaningfully differ from
# the bare letter are populated; the rest fall through to bare emission.
_LETTER_NAMES: Dict[str, Dict[str, str]] = {
    "spanish": {
        "a": "a",        "b": "be",       "c": "ce",       "d": "de",
        "e": "e",        "f": "efe",      "g": "ge",       "h": "hache",
        "i": "i",        "j": "jota",     "k": "ka",       "l": "ele",
        "m": "eme",      "n": "ene",      "ñ": "eñe",      "o": "o",
        "p": "pe",       "q": "cu",       "r": "erre",     "s": "ese",
        "t": "te",       "u": "u",        "v": "uve",      "w": "uve doble",
        "x": "equis",    "y": "ye",       "z": "zeta",
    },
}

_AT_SIGN_WORD: Dict[str, str] = {
    "spanish":    "arroba",
    "english":    "at",
    "french":     "arobase",
    "german":     "at",
    "italian":    "chiocciola",
    "portuguese": "arroba",
    "japanese":   "アット",
    "korean":     "골뱅이",
    "chinese":    "at",
    "russian":    "собака",
}

# URL / email placeholder per language. Used for *complex* URLs only —
# anything with a path, query, or fragment beyond the bare host. Stripping
# the URL to empty would leave dangling sentences ("Más info en " with
# nothing after); a placeholder keeps the grammar intact. Pass an empty
# string to normalize_text(url_placeholder="") to strip instead, in which
# case the empty-bracket cleanup pass also runs.
#
# Simple URLs (just a hostname, optional protocol, optional trailing slash)
# bypass this and get vocalized directly: "www.example.com" reads
# "w w w punto example punto com" so a listener can transcribe it back.
_URL_PLACEHOLDER: Dict[str, str] = {
    "spanish":    "enlace",
    "english":    "link",
    "french":     "lien",
    "german":     "Link",
    "italian":    "collegamento",
    "portuguese": "ligação",
    "japanese":   "リンク",
    "korean":     "링크",
    "chinese":    "链接",
    "russian":    "ссылка",
}
_EMAIL_PLACEHOLDER: Dict[str, str] = {
    "spanish":    "correo",
    "english":    "email",
    "french":     "courriel",
    "german":     "E-Mail",
    "italian":    "email",
    "portuguese": "email",
    "japanese":   "メール",
    "korean":     "이메일",
    "chinese":    "电子邮件",
    "russian":    "адрес",
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

# Unit acronyms expanded to spoken words per language. Tricky boundary
# semantics: "\b" alone matches between digit and letter, so "\bGB\b"
# would FAIL to match "512GB" (the boundary is suppressed because both
# '2' and 'G' are \w characters). We need letter-only lookbehind/
# lookahead so the unit matches whether glued to digits ("512GB") or
# standalone ("the GB drive"). Each entry below uses
# (?<![a-zA-Z])UNIT(?![a-zA-Z]); replacements are prefixed with a space
# so glued forms get a clean separator (the whitespace collapse pass
# fixes any double-spaces in the standalone case).
#
# Order in the dict matters: longer/more-specific patterns must come
# before shorter ones so "Mbps" doesn't get partially eaten by "MB".
# Python 3.7+ preserves insertion order — this is intentional.
#
# Coverage is selective on purpose. Standalone single-letter units like
# "g" (gramos) or "m" (metros) are too risky — they'd match inside or
# adjacent to ordinary words. Only acronyms unambiguous in normal text
# get expanded.
_UNITS: Dict[str, Dict[str, str]] = {
    "spanish": {
        r"(?<![a-zA-Z])Mbps(?![a-zA-Z])": " megabits por segundo",
        r"(?<![a-zA-Z])Gbps(?![a-zA-Z])": " gigabits por segundo",
        r"(?<![a-zA-Z])Kbps(?![a-zA-Z])": " kilobits por segundo",
        r"(?<![a-zA-Z])kWh(?![a-zA-Z])":  " kilovatios hora",
        r"(?<![a-zA-Z])GHz(?![a-zA-Z])":  " gigahercios",
        r"(?<![a-zA-Z])MHz(?![a-zA-Z])":  " megahercios",
        r"(?<![a-zA-Z])kHz(?![a-zA-Z])":  " kilohercios",
        r"(?<![a-zA-Z])RPM(?![a-zA-Z])":  " revoluciones por minuto",
        r"(?<![a-zA-Z])FPS(?![a-zA-Z])":  " fotogramas por segundo",
        r"(?<![a-zA-Z])DPI(?![a-zA-Z])":  " puntos por pulgada",
        r"(?<![a-zA-Z])GB(?![a-zA-Z])":   " gigabytes",
        r"(?<![a-zA-Z])MB(?![a-zA-Z])":   " megabytes",
        r"(?<![a-zA-Z])KB(?![a-zA-Z])":   " kilobytes",
        r"(?<![a-zA-Z])TB(?![a-zA-Z])":   " terabytes",
        r"(?<![a-zA-Z])PB(?![a-zA-Z])":   " petabytes",
        r"(?<![a-zA-Z])MP(?![a-zA-Z])":   " megapíxeles",
        r"(?<![a-zA-Z])kW(?![a-zA-Z])":   " kilovatios",
        r"(?<![a-zA-Z])MW(?![a-zA-Z])":   " megavatios",
        r"(?<![a-zA-Z])GW(?![a-zA-Z])":   " gigavatios",
        r"(?<![a-zA-Z])Hz(?![a-zA-Z])":   " hercios",
        r"(?<![a-zA-Z])kg(?![a-zA-Z])":   " kilogramos",
        r"(?<![a-zA-Z])mg(?![a-zA-Z])":   " miligramos",
        r"(?<![a-zA-Z])ml(?![a-zA-Z])":   " mililitros",
        r"(?<![a-zA-Z])km(?![a-zA-Z])":   " kilómetros",
        r"(?<![a-zA-Z])cm(?![a-zA-Z])":   " centímetros",
        r"(?<![a-zA-Z])mm(?![a-zA-Z])":   " milímetros",
        r"(?<![a-zA-Z])ms(?![a-zA-Z])":   " milisegundos",
    },
    "english": {
        r"(?<![a-zA-Z])Mbps(?![a-zA-Z])": " megabits per second",
        r"(?<![a-zA-Z])Gbps(?![a-zA-Z])": " gigabits per second",
        r"(?<![a-zA-Z])Kbps(?![a-zA-Z])": " kilobits per second",
        r"(?<![a-zA-Z])kWh(?![a-zA-Z])":  " kilowatt-hours",
        r"(?<![a-zA-Z])GHz(?![a-zA-Z])":  " gigahertz",
        r"(?<![a-zA-Z])MHz(?![a-zA-Z])":  " megahertz",
        r"(?<![a-zA-Z])kHz(?![a-zA-Z])":  " kilohertz",
        r"(?<![a-zA-Z])RPM(?![a-zA-Z])":  " revolutions per minute",
        r"(?<![a-zA-Z])FPS(?![a-zA-Z])":  " frames per second",
        r"(?<![a-zA-Z])DPI(?![a-zA-Z])":  " dots per inch",
        r"(?<![a-zA-Z])GB(?![a-zA-Z])":   " gigabytes",
        r"(?<![a-zA-Z])MB(?![a-zA-Z])":   " megabytes",
        r"(?<![a-zA-Z])KB(?![a-zA-Z])":   " kilobytes",
        r"(?<![a-zA-Z])TB(?![a-zA-Z])":   " terabytes",
        r"(?<![a-zA-Z])PB(?![a-zA-Z])":   " petabytes",
        r"(?<![a-zA-Z])MP(?![a-zA-Z])":   " megapixels",
        r"(?<![a-zA-Z])kW(?![a-zA-Z])":   " kilowatts",
        r"(?<![a-zA-Z])MW(?![a-zA-Z])":   " megawatts",
        r"(?<![a-zA-Z])GW(?![a-zA-Z])":   " gigawatts",
        r"(?<![a-zA-Z])Hz(?![a-zA-Z])":   " hertz",
        r"(?<![a-zA-Z])kg(?![a-zA-Z])":   " kilograms",
        r"(?<![a-zA-Z])mg(?![a-zA-Z])":   " milligrams",
        r"(?<![a-zA-Z])ml(?![a-zA-Z])":   " milliliters",
        r"(?<![a-zA-Z])km(?![a-zA-Z])":   " kilometers",
        r"(?<![a-zA-Z])cm(?![a-zA-Z])":   " centimeters",
        r"(?<![a-zA-Z])mm(?![a-zA-Z])":   " millimeters",
        r"(?<![a-zA-Z])ms(?![a-zA-Z])":   " milliseconds",
    },
}

# Masculine-form words that have feminine variants when they precede a
# feminine noun. Used by _gender_concordance for Spanish — the most common
# case is hundreds ("doscientos personas" → "doscientas personas") plus
# the unit "uno" / "veintiuno" / "alguno" / "ninguno". Only Spanish has
# this in our language set so it's hardcoded here rather than a per-lang
# table.
_ES_MASC_TO_FEM: Dict[str, str] = {
    "doscientos":   "doscientas",
    "trescientos":  "trescientas",
    "cuatrocientos":"cuatrocientas",
    "quinientos":   "quinientas",
    "seiscientos":  "seiscientas",
    "setecientos":  "setecientas",
    "ochocientos":  "ochocientas",
    "novecientos":  "novecientas",
    "uno":          "una",
    "veintiún":     "veintiuna",
    "veintiuno":    "veintiuna",
    "alguno":       "alguna",
    "ninguno":      "ninguna",
}

# Apocope: words that drop their final -o (with optional accent shift) when
# they precede a masculine singular noun. "tengo uno gigabit" → "tengo un
# gigabit". This applies AFTER the gender-concordance check has decided
# the next word is masculine — feminine cases use _ES_MASC_TO_FEM instead.
# Only Spanish.
_ES_APOCOPE: Dict[str, str] = {
    "uno":       "un",
    "veintiuno": "veintiún",
    "alguno":    "algún",
    "ninguno":   "ningún",
}

# Words that should NOT trigger apocope when they follow "uno" — typical
# function words/prepositions/conjunctions where "uno" is being used as a
# pronoun rather than a quantifier ("uno por uno", "uno de cada", "uno y
# otro"). The list is conservative; false-negatives just leave "uno"
# untouched, which is grammatically valid even if not always natural.
_ES_APOCOPE_IDIOM_GUARDS: set = {
    # conjunctions
    "y", "o", "e", "u", "ni", "pero", "sino",
    # prepositions
    "a", "ante", "bajo", "con", "contra", "de", "desde", "durante", "en",
    "entre", "hacia", "hasta", "mediante", "para", "por", "según", "sin",
    "sobre", "tras",
    # relative / interrogative
    "que", "cual", "cuales",
    # comparison / coordination
    "como", "menos", "más",
    # cardinal pronouns repeating
    "uno", "una", "otro", "otra", "otros", "otras",
}

# Feminine nouns whose plural ends in "-es" rather than "-as", so the
# suffix-based heuristic in _gender_concordance can't infer gender from
# spelling alone (compare "mujeres" fem vs "hombres" masc, identical
# suffix). When the next word after a masculine number is in this set we
# apply concordance even though the suffix doesn't match.
#
# Bounded list — Spanish has a finite number of common feminine -es
# nouns. Grows with audit findings; not exhaustive.
_ES_FEM_ES_NOUNS: set = {
    # singular (-e ending) — for "una/veintiuna" before fem singular
    "mujer", "flor", "red", "ley", "luz", "voz", "raíz", "sal", "fe",
    "miel", "piel", "edad", "ciudad", "verdad", "libertad", "vez",
    "paz", "cruz", "nariz", "perdiz",
    # plural (-es ending) — for hundreds before fem plural
    "mujeres", "flores", "redes", "leyes", "luces", "voces", "raíces",
    "sales", "veces", "paces", "cruces", "narices",
    "edades", "ciudades", "verdades", "libertades",
    "noches", "tardes", "clases", "bases", "claves",
    "partes", "fuentes", "frases", "llaves", "naves",
    "madres", "carnes", "muertes", "mentes", "fuentes",
    "leches", "miel", "pieles", "fieles",
    "aves",  # technically uses "el" in singular, fem otherwise
}

# Words that LOOK feminine plural (end in "-as") but aren't — masculine
# nouns whose lemma happens to end in -a. The gender-concordance heuristic
# would mis-fire on these ("trescientos días" is correct masc, not
# "trescientas días"). The list is intentionally short: only nouns common
# enough to come up in real text. Add to it as false positives surface.
_ES_FALSE_AS_FEM: set = {
    "día", "días",
    "mapa", "mapas",
    "problema", "problemas",
    "tema", "temas",
    "programa", "programas",
    "clima", "climas",
    "sistema", "sistemas",
    "idioma", "idiomas",
    "fantasma", "fantasmas",
    "drama", "dramas",
    "diagrama", "diagramas",
    "planeta", "planetas",
    "esquema", "esquemas",
    "poeta", "poetas",
    "cometa", "cometas",  # the celestial body — masc
    "panorama", "panoramas",
    "trauma", "traumas",
    "diploma", "diplomas",
    "axioma", "axiomas",
    "carisma", "carismas",
    "lema", "lemas",
    "enigma", "enigmas",
    "dilema", "dilemas",
    "teorema", "teoremas",
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
        # Note: S.A. / S.L. intentionally NOT expanded — native speakers more
        # often read them as letter sequences ("ese a", "ese ele") than as
        # the full "sociedad anónima" / "sociedad limitada", and the
        # expansion sounds overly formal for chat content. If a future use
        # case needs them, pass extra_abbreviations={r"\bS\.\s?A\.": ...}.
        r"\bvs\.": "contra",
        # ISO-4217 currency codes that LLM output frequently produces in
        # parens or alongside numbers ("$7,000 (USD)"). Expand to the
        # spoken word; the bare-number expander handles the digits.
        r"\bUSD\b": "dólares",
        r"\bEUR\b": "euros",
        r"\bGBP\b": "libras",
        r"\bJPY\b": "yenes",
        r"\bCNY\b": "yuanes",
        r"\bCHF\b": "francos suizos",
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
        r"\bUSD\b": "dollars",
        r"\bEUR\b": "euros",
        r"\bGBP\b": "pounds",
        r"\bJPY\b": "yen",
        r"\bCNY\b": "yuan",
        r"\bCHF\b": "Swiss francs",
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

# "X or more" suffix per language for "1500+" → "1500 o más".
_PLUS_SUFFIX_WORD: Dict[str, str] = {
    "spanish":    "o más",
    "english":    "or more",
    "french":     "ou plus",
    "german":     "oder mehr",
    "italian":    "o più",
    "portuguese": "ou mais",
    "russian":    "или больше",
    "japanese":   "以上",
    "korean":     "이상",
    "chinese":    "或更多",
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
_MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
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
# "<digits>+" or "<digits>%+" written immediately after a number/percent
# meaning "or more" / "and up" in user-facing copy ("$10,000+", "100%+").
# The optional %? lets the plus-suffix run BEFORE the percent expander
# without losing the % character; the percent expander runs afterward
# on the captured group as usual. Lookahead excludes "++" and word
# boundaries so "C++" / "Java++" stay intact.
_PLUS_SUFFIX = re.compile(r"(\d+(?:[.,]\d+)*\s*%?)\+(?![\w+])")
# Pure thousand-separator patterns. ASCII-only digit groups separated by
# either ',' or '.' with EXACTLY 3 digits per group. Used in
# _expand_numbers to disambiguate "7,000" vs "7,5" before applying the
# locale-specific decimal rule.
_THOUSANDS_COMMA = re.compile(r"^\d{1,3}(,\d{3})+$")
_THOUSANDS_DOT   = re.compile(r"^\d{1,3}(\.\d{3})+$")

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


_EMPTY_BRACKETS = re.compile(r"\(\s*\)|\[\s*\]|\{\s*\}|<\s*>")
_TRAILING_PUNCT = ".,;:!?)]}"


def _vocalize_host(host: str, dot_word: str, lang: str) -> str:
    """Convert a hostname into a TTS-friendly spoken sequence.

    Splits on dots and joins with the language's dot word
    ("punto" / "dot" / "Punkt"). Segments matching well-known unpronounceable
    prefixes ("www", "ftp", "smtp") are spelled out letter-by-letter; the
    per-language ``_LETTER_NAMES`` table is consulted so Spanish "www"
    renders as "uve doble uve doble uve doble" instead of three bare 'w'
    characters that the TTS reads as "wuwuwu" or worse.
    """
    spell_letterwise = {"www", "ftp", "smtp", "imap", "pop", "ssh", "http", "https"}
    letter_names = _LETTER_NAMES.get(lang)
    parts = host.split(".")
    rendered = []
    for p in parts:
        if p.lower() in spell_letterwise:
            if letter_names:
                rendered.append(" ".join(letter_names.get(c.lower(), c) for c in p))
            else:
                rendered.append(" ".join(p.lower()))
        else:
            rendered.append(p)
    return f" {dot_word} ".join(rendered)


def _classify_url(matched: str) -> Tuple[bool, str]:
    """Decide if a URL is "simple" (host only) and return its host.

    Simple = no path beyond ``/``, no query, no fragment, no auth, no port.
    Anything past the bare host belongs to the placeholder branch — listening
    to "barra docs barra api versión uno" is not useful.
    """
    candidate = matched if "://" in matched else f"//{matched}"
    try:
        parsed = urlparse(candidate, scheme="http")
    except ValueError:
        return False, ""
    netloc = parsed.netloc
    # Strip port + auth from netloc for vocalization (also automatic disqualifier)
    if "@" in netloc or ":" in netloc:
        return False, ""
    if not netloc:
        return False, ""
    is_simple = parsed.path in ("", "/") and not parsed.query and not parsed.fragment
    return is_simple, netloc


def _replace_urls(
    text: str,
    lang: str,
    url_placeholder: Optional[str],
    email_placeholder: Optional[str],
) -> str:
    """Replace URLs and emails with TTS-friendly forms.

    Two paths:
      - Simple URL (just a host, optionally with protocol or trailing /)
        → drop the protocol and vocalize the host, joining segments with
        the language's dot word and spelling out known acronym prefixes
        ("www" → "w w w"). The listener can transcribe the URL back from
        the audio.
      - Complex URL (any path/query/fragment) → use a placeholder word.
        Reading "barra docs interrogación q igual..." aloud is useless.

    Emails are always vocalized: "foo arroba bar punto com". The email
    placeholder is only used as a fallback if the @ split fails.

    Trailing terminal punctuation captured by the greedy regex
    ("https://example.com." matches the trailing dot too) is trimmed
    before vocalization so it survives as part of the surrounding
    sentence.

    Pass an empty placeholder to bypass vocalization for that path
    (force-strip mode); in that case empty bracket-pairs left behind
    are cleaned up.
    """
    url_word = url_placeholder
    email_word = email_placeholder
    if url_word is None:
        url_word = _URL_PLACEHOLDER.get(lang, _URL_PLACEHOLDER["english"])
    if email_word is None:
        email_word = _EMAIL_PLACEHOLDER.get(lang, _EMAIL_PLACEHOLDER["english"])

    dot_word = _URL_DOT_WORD.get(lang, _URL_DOT_WORD["english"])
    at_word = _AT_SIGN_WORD.get(lang, _AT_SIGN_WORD["english"])

    def _split_trailing(matched: str) -> Tuple[str, str]:
        trailing = ""
        while matched and matched[-1] in _TRAILING_PUNCT:
            trailing = matched[-1] + trailing
            matched = matched[:-1]
        return matched, trailing

    def _replace_url(m: "re.Match[str]") -> str:
        matched, trailing = _split_trailing(m.group(0))
        # Force-strip mode (caller asked for empty placeholder)
        if not url_word.strip():
            return url_word + trailing
        is_simple, host = _classify_url(matched)
        if is_simple and host:
            return _vocalize_host(host, dot_word, lang) + trailing
        return url_word + trailing

    def _replace_email(m: "re.Match[str]") -> str:
        matched, trailing = _split_trailing(m.group(0))
        if not email_word.strip():
            return email_word + trailing
        local, sep, host = matched.partition("@")
        if sep and host:
            return f"{local} {at_word} {_vocalize_host(host, dot_word, lang)}" + trailing
        return email_word + trailing

    text = _URL_PATTERN.sub(_replace_url, text)
    text = _EMAIL_PATTERN.sub(_replace_email, text)
    if not url_word.strip() or not email_word.strip():
        text = _EMPTY_BRACKETS.sub("", text)
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
    # Preserve both anchor text AND URL: "Visita [mi web](https://example.com)"
    # used to drop the URL entirely. Now we keep "mi web https://example.com"
    # and let the URL-replacement pass downstream vocalize the URL ("mi web
    # example punto com"). Listeners can hear the link target — the same
    # information a sighted reader gets from the rendered link.
    text = _MD_LINK.sub(r"\1 \2", text)
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

    Spanish has an idiomatic exception: "100%" reads as "cien por cien",
    not "cien por ciento" — both are technically valid but native speakers
    consistently use the doubled form for exactly 100. Other percentages
    (87%, 12,5%) take the regular "<n> por ciento" form.

    Falls back to English ("percent") when the language has no entry.
    """
    word = _PERCENT_WORD.get(lang, _PERCENT_WORD["english"])
    if lang == "spanish":
        # Match "100%" (whole-number 100, no decimals) before the generic
        # rule so it gets the idiomatic reading.
        text = re.sub(r"\b100\s*%(?!\d)", "cien por cien", text)
    return _PERCENT.sub(rf"\1 {word}", text)


def _expand_plus_suffix(text: str, lang: str) -> str:
    """Replace ``<digits>+`` with ``<digits> <or-more-word>``.

    Common in copy LLMs produce — "$10,000+", "1500+ users", "10+
    minutes". Without this the model reads the trailing "+" as a literal
    "más" / "plus" or just stops short, both of which sound wrong.

    Falls back to English when the language has no entry. The bare
    number expander downstream handles the digit group.
    """
    word = _PLUS_SUFFIX_WORD.get(lang, _PLUS_SUFFIX_WORD["english"])
    return _PLUS_SUFFIX.sub(rf"\1 {word}", text)


def _expand_units(text: str, lang: str) -> str:
    """Expand unit acronyms (GB, kg, MHz, …) to their spoken word.

    Runs *before* the bare-number expander so a glued form like "512GB"
    becomes "512 gigabytes" first; the number expander then handles "512"
    in isolation and produces "quinientos doce gigabytes" with the
    expected space between number and unit.

    Languages with no table fall through unchanged.
    """
    table = _UNITS.get(lang)
    if not table:
        return text
    for pattern, replacement in table.items():
        text = re.sub(pattern, replacement, text)
    return text


def _gender_concordance(text: str, lang: str) -> str:
    """Apply masculine→feminine conversion before feminine-plural nouns (es).

    Spanish numbers in 200..900 (and the standalone "uno") have a feminine
    variant when they precede a feminine noun: ``trescientas mujeres``
    rather than ``trescientos mujeres``. num2words doesn't know about the
    surrounding noun, so we post-process the expanded text: when one of
    these masculine words is immediately followed by what looks like a
    feminine plural ("-as" ending), swap to the feminine form.

    The heuristic mis-fires on masculine nouns that happen to end in
    "-as" (días, problemas, sistemas, …). _ES_FALSE_AS_FEM lists the
    common ones; words in that set are skipped.

    Only Spanish is covered. Italian/French/Portuguese have similar but
    much narrower concordance issues; add them per-language if the audit
    surfaces a real failure.
    """
    if lang != "spanish":
        return text

    # The pattern matches a masculine word from our table followed by
    # whitespace and another word. The "another word" decision is made in
    # the callback so we can consult the false-positive set.
    masc_pattern = r"\b(" + "|".join(re.escape(k) for k in _ES_MASC_TO_FEM) + r")\b(\s+)(\w+)"

    def _swap(m: "re.Match[str]") -> str:
        masc = m.group(1)
        gap = m.group(2)
        next_word = m.group(3)
        nw_lower = next_word.lower()
        is_apocope_word = masc in _ES_APOCOPE

        # Idiom guard for apocope words: "uno por uno", "uno de cada", etc.
        # When the following word is a function word, "uno" stays as a
        # pronoun and doesn't apocopate. Only applies to apocope words —
        # the hundreds (doscientos, etc.) don't have this issue because
        # they're never used as standalone pronouns.
        if is_apocope_word and nw_lower in _ES_APOCOPE_IDIOM_GUARDS:
            return m.group(0)

        # Common masc nouns ending in -as: keep masculine for hundreds
        # ("trescientos días"), but still apocopate uno-forms ("uno día"
        # → "un día").
        if nw_lower in _ES_FALSE_AS_FEM:
            if is_apocope_word:
                return f"{_ES_APOCOPE[masc]}{gap}{next_word}"
            return m.group(0)

        # Explicit feminine nouns (typically -es endings where the suffix
        # is ambiguous: "mujeres" fem vs "hombres" masc): apply concordance.
        if nw_lower in _ES_FEM_ES_NOUNS:
            return f"{_ES_MASC_TO_FEM[masc]}{gap}{next_word}"

        # Feminine plural marker (-as): apply concordance.
        if nw_lower.endswith("as") and len(nw_lower) > 2:
            return f"{_ES_MASC_TO_FEM[masc]}{gap}{next_word}"

        # Feminine singular: only meaningful for apocope words ("una hija"
        # rather than "uno hija"). Hundreds always stay masculine before
        # a singular -a noun (those are exceedingly rare anyway: "200 agua"
        # is not idiomatic Spanish).
        if is_apocope_word and nw_lower.endswith("a") and nw_lower not in _ES_FALSE_AS_FEM:
            return f"{_ES_MASC_TO_FEM[masc]}{gap}{next_word}"

        # Default: assume next word is masculine. For apocope words,
        # apply apocope (uno→un, veintiuno→veintiún, etc.). Hundreds stay
        # as-is since there's nothing to apocopate on them.
        if is_apocope_word:
            return f"{_ES_APOCOPE[masc]}{gap}{next_word}"

        return m.group(0)

    return re.sub(masc_pattern, _swap, text)


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
        # Glue protection: when a digit run is glued to a trailing letter
        # (e.g. "512GB", "100kg", "60Hz"), the bare expansion would produce
        # "quinientos doceGB" with no audible space between the number and
        # the unit. Detect this from the surrounding context and append a
        # space to the words so the unit stays a separate spoken token.
        end = m.end()
        full = m.string
        needs_trailing_space = end < len(full) and full[end].isalpha()

        # Single bare digit shortcut — keep cheap path
        if len(tok) == 1:
            words = _spell_int(int(tok))
            return words + " " if needs_trailing_space else words

        # Thousand-only patterns first, regardless of language. "7,000"
        # in Spanish text is almost certainly en-US thousands convention
        # (LLMs produce currency this way: "$7,000"). Native Spanish for
        # the decimal 7.5 is "7,5" — the 3-digit-after-comma pattern is
        # essentially never a decimal in any locale. Same for periods
        # ("1.234.567" in Spanish, "1.234.567" copied from Spanish into
        # English). Detecting up-front avoids the float() conversion
        # eating trailing zeros (7.000 → 7.0 → "siete").
        if _THOUSANDS_COMMA.fullmatch(tok):
            normalized = tok.replace(",", "")
        elif _THOUSANDS_DOT.fullmatch(tok):
            normalized = tok.replace(".", "")
        elif use_comma_decimal:
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

            words = f"{int_words} {decimal_word} {frac_words}"
            return words + " " if needs_trailing_space else words

        # Pure integer
        try:
            words = _spell_int(int(normalized))
        except ValueError:
            return tok
        return words + " " if needs_trailing_space else words

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
    url_placeholder: Optional[str] = None,
    email_placeholder: Optional[str] = None,
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
        strip_urls: Replace URLs and email addresses (see ``url_placeholder``
            and ``email_placeholder`` for how).
        strip_markdown: Strip Markdown formatting markers (keeps inner text).
        expand_abbreviations: Apply the per-language abbreviation dictionary.
        expand_numbers: Expand digits and currency to spoken words.
        url_placeholder: Word to replace URLs with. ``None`` (default) means
            "use the per-language placeholder from ``_URL_PLACEHOLDER``"
            ("enlace" / "link" / etc.). Pass an empty string to strip URLs;
            in that case any empty bracket-pairs left behind are also removed.
        email_placeholder: Same idea for email addresses, with its own
            per-language defaults ("correo" / "email" / etc.).
        extra_abbreviations: Additional regex→replacement mappings applied
            after the built-in dictionary (typically domain glossaries from
            the consumer side).

    Returns:
        Normalized text, ready to feed to the TTS model.
    """
    if strip_markdown:
        text = _strip_markdown(text)
    if strip_urls:
        text = _replace_urls(text, lang, url_placeholder, email_placeholder)
    if strip_emojis:
        text = _strip_emojis(text)
    if expand_abbreviations:
        text = _expand_abbreviations(text, lang)
        if extra_abbreviations:
            for pattern, replacement in extra_abbreviations.items():
                text = re.sub(pattern, replacement, text)
    if expand_numbers:
        # Order matters: compound expanders that consume "<digits><suffix>"
        # patterns must run before the bare number expander pulls digits
        # apart. Plus-suffix runs BEFORE currency/percent because the "+"
        # is glued to the digits/% — once currency or percent consumes its
        # symbol the "+" gets orphaned and the plus-suffix regex can't
        # match it back to the number ("$10,000+" → "10,000 dólares+"
        # if currency runs first). After plus-suffix and the symbol
        # expanders, the bare number expander handles remaining digits.
        # The gender-concordance pass at the end cleans up masc→fem
        # before feminine-plural nouns ("quinientos personas" →
        # "quinientas personas").
        text = _expand_dates(text, lang)
        text = _expand_times(text, lang)
        text = _expand_plus_suffix(text, lang)
        text = _expand_currency(text, lang)
        text = _expand_percent(text, lang)
        text = _expand_units(text, lang)
        text = _expand_numbers(text, lang)
        text = _gender_concordance(text, lang)

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
