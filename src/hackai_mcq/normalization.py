from __future__ import annotations

import re
import unicodedata
from functools import lru_cache


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text or "")
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("\x00", " ")).strip()


@lru_cache(maxsize=8192)
def canonical(text: str) -> str:
    return normalize_space(strip_accents(text).lower())


def contains_any(text: str, terms: tuple[str, ...] | list[str] | set[str]) -> bool:
    c = canonical(text)
    return any(canonical(t) in c for t in terms)


def truncate(text: str, max_chars: int) -> str:
    text = normalize_space(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    # avoid cutting in the middle of a word when possible
    pos = max(cut.rfind(" "), cut.rfind("\n"))
    return (cut[:pos] if pos > max_chars * 0.7 else cut).rstrip() + "…"
