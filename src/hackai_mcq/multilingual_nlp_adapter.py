from __future__ import annotations

"""Lightweight multilingual signals for MCQ routing.

This module is deliberately dependency-light. The contest model is fixed by the
rules, so the competitive edge comes from better routing/prompting around that
model. We detect scripts, negation/exception wording, broad domains, and token
pressure across Vietnamese, English, CJK, Korean, Arabic, Cyrillic, Thai and
Latin-script European languages.
"""

import re
import unicodedata
from dataclasses import dataclass, field

from .normalization import canonical, normalize_space, strip_accents

_SCRIPT_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "latin": ((0x0041, 0x024F), (0x1E00, 0x1EFF)),
    "cjk": ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF)),
    "hiragana": ((0x3040, 0x309F),),
    "katakana": ((0x30A0, 0x30FF),),
    "hangul": ((0xAC00, 0xD7AF), (0x1100, 0x11FF)),
    "arabic": ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF)),
    "cyrillic": ((0x0400, 0x04FF),),
    "thai": ((0x0E00, 0x0E7F),),
    "devanagari": ((0x0900, 0x097F),),
}

NEGATION_TERMS_BY_LANG: dict[str, tuple[str, ...]] = {
    "vi": (
        "không", "khong", "sai", "ngoại trừ", "ngoai tru", "không đúng", "khong dung",
        "không phải", "khong phai", "chưa chính xác", "chua chinh xac", "trừ",
    ),
    "en": (
        "not", "no", "never", "none", "cannot", "can't", "incorrect", "false", "least",
        "except", "excluding", "wrong", "isn't", "aren't", "doesn't", "do not", "which is not",
    ),
    "fr": ("ne pas", "n'est pas", "pas", "sauf", "excepté", "incorrect", "faux", "moins"),
    "es": ("no", "nunca", "excepto", "salvo", "incorrecto", "falso", "menos"),
    "de": ("nicht", "kein", "außer", "falsch", "inkorrekt", "wenigsten"),
    "zh": ("不", "不是", "没有", "沒", "错误", "錯誤", "不正确", "不正確", "除了", "除外", "非"),
    "ja": ("ない", "ではない", "じゃない", "誤り", "間違", "除く", "以外", "不正確"),
    "ko": ("아니다", "아닌", "않", "못", "제외", "틀린", "잘못", "부정확"),
    "ru": ("не", "нет", "кроме", "исключ", "невер", "ошиб", "лож"),
    "ar": ("لا", "ليس", "ليست", "غير", "باستثناء", "خطأ", "خاطئ"),
    "th": ("ไม่", "มิ", "ยกเว้น", "ผิด", "ไม่ถูก"),
}

DOMAIN_TERMS_MULTI: dict[str, tuple[str, ...]] = {
    "law": (
        "law", "legal", "contract", "liability", "regulation", "pháp luật", "hop dong", "hợp đồng",
        "luật", "商业法", "法律", "契約", "법", "закон", "قانون",
    ),
    "finance": (
        "finance", "bank", "banking", "interest", "revenue", "cost", "gdp", "inflation", "exchange rate",
        "ngân hàng", "ngan hang", "tài chính", "tai chinh", "金融", "銀行", "금융", "банк", "اقتصاد",
    ),
    "health": (
        "health", "medical", "patient", "disease", "hospital", "clinical", "y tế", "sức khỏe", "benh nhan",
        "医疗", "健康", "病", "医療", "환자", "здоров", "مرض",
    ),
    "education": (
        "education", "student", "teacher", "school", "learning", "exam", "học", "sinh viên", "giáo viên",
        "教育", "学生", "学校", "教育", "학생", "учен", "تعليم",
    ),
    "science": (
        "physics", "chemistry", "biology", "math", "calculate", "equation", "toán", "vật lý", "hóa học", "sinh học",
        "数学", "物理", "化学", "生物", "수학", "физ", "хим", "رياض",
    ),
    "technology": (
        "ai", "machine learning", "algorithm", "computer", "data", "network", "công nghệ", "thuật toán",
        "人工智能", "算法", "컴퓨터", "алгоритм", "ذكاء اصطناعي",
    ),
}

HARD_MARKERS_BY_LANG: dict[str, tuple[str, ...]] = {
    "vi": ("đúng nhất", "phù hợp nhất", "suy luận", "hệ quả", "nguyên nhân"),
    "en": ("best", "most likely", "infer", "inference", "consequence", "cause", "primarily", "mainly"),
    "fr": ("le plus", "probable", "déduire", "cause", "conséquence"),
    "es": ("más probable", "mejor", "inferir", "causa", "consecuencia"),
    "de": ("am besten", "wahrscheinlich", "schlussfolger", "ursache", "folge"),
    "zh": ("最", "推断", "推論", "原因", "结果", "結果"),
    "ja": ("最も", "推論", "原因", "結果"),
    "ko": ("가장", "추론", "원인", "결과"),
    "ru": ("наиболее", "вероят", "вывод", "причин", "следств"),
    "ar": ("الأكثر", "استنتج", "سبب", "نتيجة"),
    "th": ("มากที่สุด", "น่าจะ", "สรุป", "สาเหตุ", "ผล"),
}

@dataclass(slots=True)
class MultilingualSignals:
    normalized: str
    language: str
    scripts: dict[str, int]
    primary_script: str
    has_negation: bool
    domains: list[str] = field(default_factory=list)
    hard_markers: list[str] = field(default_factory=list)
    token_count: int = 0
    has_number: bool = False
    is_mixed_language: bool = False
    cjk_char_count: int = 0


def _char_script(ch: str) -> str | None:
    cp = ord(ch)
    for script, ranges in _SCRIPT_RANGES.items():
        for lo, hi in ranges:
            if lo <= cp <= hi:
                return script
    return None


def script_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ch in text:
        if ch.isspace() or ch.isdigit() or unicodedata.category(ch).startswith("P"):
            continue
        s = _char_script(ch)
        if s:
            counts[s] = counts.get(s, 0) + 1
    return counts


def detect_language(text: str) -> tuple[str, str, dict[str, int]]:
    norm = normalize_space(text)
    counts = script_counts(norm)
    if not counts:
        return "unknown", "unknown", counts
    primary = max(counts, key=counts.get)
    total = sum(counts.values())
    latin_ratio = counts.get("latin", 0) / max(1, total)
    # Script-first languages.
    if primary == "cjk":
        # Japanese normally mixes CJK with kana.
        if counts.get("hiragana", 0) + counts.get("katakana", 0) > max(2, counts.get("cjk", 0) * 0.08):
            return "ja", primary, counts
        return "zh", primary, counts
    if primary in {"hiragana", "katakana"}:
        return "ja", primary, counts
    if primary == "hangul":
        return "ko", primary, counts
    if primary == "arabic":
        return "ar", primary, counts
    if primary == "cyrillic":
        return "ru", primary, counts
    if primary == "thai":
        return "th", primary, counts
    if primary == "devanagari":
        return "hi", primary, counts
    # Latin-script heuristic: distinguish Vietnamese by diacritics/common words.
    c = canonical(norm)
    raw_lower = norm.lower()
    vi_hits = sum(1 for t in ("câu", "hỏi", "đáp", "đúng", "không", "nào", "là", "trong", "của", "và") if t in raw_lower)
    vi_hits += sum(1 for t in ("cau", "hoi", "dap", "dung", "khong", "nao", "trong", "cua") if t in c)
    if vi_hits >= 2 or re.search(r"[ăâđêôơưĂÂĐÊÔƠƯ]", norm):
        return "vi", "latin", counts
    if latin_ratio > 0.35:
        return "en", "latin", counts
    return "mixed", primary, counts


def multilingual_tokenize(text: str) -> list[str]:
    """Tokenize Latin-space languages and CJK/Thai script enough for routing.

    This is not intended to replace a real tokenizer; it prevents CJK/Thai rows
    from being seen as zero-token/low-difficulty rows by the router.
    """
    norm = normalize_space(text)
    tokens = re.findall(r"[A-Za-zÀ-ỹ0-9]+", norm)
    # Treat CJK/Kana/Hangul/Thai/Arabic/Cyrillic chars as signal tokens when no spaces.
    script_chars: list[str] = []
    for ch in norm:
        s = _char_script(ch)
        if s in {"cjk", "hiragana", "katakana", "hangul", "thai", "arabic", "cyrillic", "devanagari"}:
            script_chars.append(ch)
    # Use single chars for CJK/Kana/Hangul and rough 2-char chunks for Thai-like no-space scripts.
    if script_chars:
        tokens.extend(script_chars)
    return [t for t in tokens if t.strip()]


def _contains_any_multilingual(text: str, terms: tuple[str, ...]) -> bool:
    raw = (text or "").lower()
    c = canonical(text)
    accentless = strip_accents(raw).lower()
    for term in terms:
        t = term.lower()
        if t in raw or canonical(t) in c or strip_accents(t).lower() in accentless:
            return True
    return False


def analyze_multilingual(text: str) -> MultilingualSignals:
    norm = normalize_space(unicodedata.normalize("NFC", text or ""))
    lang, primary_script, counts = detect_language(norm)
    scripts_present = [s for s, n in counts.items() if n >= 2]
    is_mixed = len(scripts_present) > 1 and not (set(scripts_present) <= {"latin"})

    # Check primary language, English and Vietnamese by default because mixed tests often use English metadata.
    neg_langs = [lang]
    for common in ("en", "vi"):
        if common not in neg_langs:
            neg_langs.append(common)
    if lang == "mixed":
        neg_langs.extend(k for k in NEGATION_TERMS_BY_LANG if k not in neg_langs)
    has_neg = any(_contains_any_multilingual(norm, NEGATION_TERMS_BY_LANG.get(l, ())) for l in neg_langs)

    domains = [name for name, terms in DOMAIN_TERMS_MULTI.items() if _contains_any_multilingual(norm, terms)]
    hard_langs = [lang]
    for common in ("en", "vi"):
        if common not in hard_langs:
            hard_langs.append(common)
    hard = [l for l in hard_langs if _contains_any_multilingual(norm, HARD_MARKERS_BY_LANG.get(l, ())) ]
    toks = multilingual_tokenize(norm)
    return MultilingualSignals(
        normalized=norm,
        language=lang,
        scripts=counts,
        primary_script=primary_script,
        has_negation=has_neg,
        domains=domains,
        hard_markers=hard,
        token_count=len(toks),
        has_number=bool(re.search(r"\d", norm)),
        is_mixed_language=is_mixed,
        cjk_char_count=counts.get("cjk", 0) + counts.get("hiragana", 0) + counts.get("katakana", 0),
    )
