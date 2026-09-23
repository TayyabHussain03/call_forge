"""Small deterministic script and lexical language classifier."""

from __future__ import annotations

import re

from app.knowledge.processing.contracts import DocumentLanguage

_LATIN_WORD = re.compile(r"[A-Za-z]+")
_URDU = re.compile(r"[\u0600-\u06ff]")
_HINDI = re.compile(r"[\u0900-\u097f]")
_ROMAN_URDU = frozenset(
    {
        "aap",
        "acha",
        "apka",
        "hai",
        "hain",
        "hum",
        "kaise",
        "karna",
        "kya",
        "mein",
        "nahi",
        "theek",
        "yeh",
    }
)
_ENGLISH = frozenset(
    {"and", "are", "business", "for", "how", "is", "service", "the", "this", "with"}
)


def detect_language(text: str) -> DocumentLanguage:
    """Classify supported languages from visible scripts and bounded word sets."""
    urdu = len(_URDU.findall(text))
    hindi = len(_HINDI.findall(text))
    words = [word.casefold() for word in _LATIN_WORD.findall(text)]
    roman = sum(word in _ROMAN_URDU for word in words)
    english = sum(word in _ENGLISH for word in words)
    present = sum((urdu > 0, hindi > 0, roman >= 2, english >= 2))
    if present > 1:
        return DocumentLanguage.MIXED
    if urdu:
        return DocumentLanguage.URDU
    if hindi:
        return DocumentLanguage.HINDI
    if roman >= 2:
        return DocumentLanguage.ROMAN_URDU
    if words:
        return DocumentLanguage.ENGLISH
    return DocumentLanguage.UNKNOWN
