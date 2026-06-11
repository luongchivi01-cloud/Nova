from __future__ import annotations

from .multilingual_nlp_adapter import analyze_multilingual
from .normalization import truncate
from .schema import MCQItem

SYSTEM_RULE = """You are a multilingual multiple-choice solver. Detect the question language and solve it faithfully in that language. The only final output allowed is one answer letter: A, B, C, or D. Do not explain."""

LANGUAGE_HINTS = {
    "vi": "Vietnamese detected: chú ý phủ định như không/sai/ngoại trừ.",
    "en": "English/Latin detected: watch for not/except/incorrect/least/most likely.",
    "zh": "Chinese detected: watch for 不/不是/错误/除了/最 and preserve the original meaning.",
    "ja": "Japanese detected: watch for ない/ではない/誤り/除く/最も.",
    "ko": "Korean detected: watch for 아니다/않/제외/틀린/가장.",
    "ru": "Cyrillic/Russian detected: watch for не/кроме/неверно/наиболее.",
    "ar": "Arabic detected: watch for لا/ليس/غير/باستثناء/الأكثر.",
    "th": "Thai detected: watch for ไม่/ยกเว้น/ผิด/มากที่สุด.",
    "mixed": "Mixed-language detected: preserve entities and do not assume Vietnamese-only wording.",
    "unknown": "Unknown language: rely on the options and output only A/B/C/D.",
}


def _base(item: MCQItem, context: str = "") -> str:
    sig = analyze_multilingual(item.text_for_retrieval())
    lang_hint = LANGUAGE_HINTS.get(sig.language, LANGUAGE_HINTS.get("unknown", ""))
    signal_line = (
        f"Detected language={sig.language}; primary_script={sig.primary_script}; "
        f"negation={sig.has_negation}; domains={','.join(sig.domains) or 'none'}; "
        f"mixed={sig.is_mixed_language}. {lang_hint}"
    )
    ctx = f"\nOffline reference context, if relevant:\n{truncate(context.strip(), 1800)}\n" if context and context.strip() else ""
    return f"""{SYSTEM_RULE}
{signal_line}
{ctx}
Question:
{truncate(item.question.strip(), 2600)}

Choices:
A. {truncate(item.options.get('A','').strip(), 900)}
B. {truncate(item.options.get('B','').strip(), 900)}
C. {truncate(item.options.get('C','').strip(), 900)}
D. {truncate(item.options.get('D','').strip(), 900)}
"""


def direct_prompt(item: MCQItem, context: str = "") -> str:
    return _base(item, context) + """
Choose the single best answer. If the question asks for NOT/EXCEPT/incorrect/least/false in any language, invert the target correctly.
Output exactly one letter: A, B, C, or D.
Answer:"""


def elimination_prompt(item: MCQItem, context: str = "") -> str:
    return _base(item, context) + """
Eliminate wrong or less relevant choices first. Be careful with negation/exception words in Vietnamese, English, Chinese, Japanese, Korean, Arabic, Cyrillic, Thai or mixed language.
Output only one letter A/B/C/D.
Answer:"""


def scoring_prompt(item: MCQItem, context: str = "") -> str:
    return _base(item, context) + """
Score A, B, C and D by how well each choice answers the question in its original language. Pick the highest-scoring choice.
Output only short JSON: {"answer":"A"} or {"answer":"B"} or {"answer":"C"} or {"answer":"D"}.
JSON:"""


def constrained_choice_prompt(item: MCQItem, context: str = "") -> str:
    return _base(item, context) + """
Return exactly one token from the set {A, B, C, D}. No other text.
"""

def cpu_choice_prompt(item: MCQItem, context: str = "") -> str:
    """Compact prompt for CPU inference where prefill latency dominates."""
    ctx = truncate(context.strip(), 900)
    context_line = f"\nReference:\n{ctx}\n" if ctx else ""
    return (
        "Choose the best answer. Handle NOT/EXCEPT/incorrect carefully. "
        "Reply with exactly A, B, C, or D.\n"
        f"{context_line}Question: {truncate(item.question.strip(), 1800)}\n"
        f"A. {truncate(item.options.get('A', '').strip(), 600)}\n"
        f"B. {truncate(item.options.get('B', '').strip(), 600)}\n"
        f"C. {truncate(item.options.get('C', '').strip(), 600)}\n"
        f"D. {truncate(item.options.get('D', '').strip(), 600)}\nAnswer:"
    )


def cpu_verifier_prompt(item: MCQItem, candidate: str, context: str = "") -> str:
    return cpu_choice_prompt(item, context) + f"\nVerify proposed answer {candidate}. Reply only A, B, C, or D:"


def negation_guard_prompt(item: MCQItem, context: str = "") -> str:
    return _base(item, context) + """
This row may contain negation or exception wording in any language. Determine whether it asks for the correct choice or the false/not/except choice, then answer accordingly.
Output exactly one letter A/B/C/D.
Answer:"""


def translation_sanity_prompt(item: MCQItem, context: str = "") -> str:
    return _base(item, context) + """
For non-Vietnamese or mixed-language rows, internally normalize the meaning into a simple question, preserve named entities/numbers, then choose the best option.
Do not print the normalization. Output exactly one letter A/B/C/D.
Answer:"""


def verifier_prompt(item: MCQItem, candidate: str, context: str = "") -> str:
    return _base(item, context) + f"""
A solver proposed answer {candidate}. Independently verify it in the question's original language. If {candidate} is best, output {candidate}; otherwise output the better answer.
Output exactly one letter A/B/C/D.
Verified answer:"""


def pairwise_prompt(item: MCQItem, left: str, right: str, context: str = "") -> str:
    left = left.upper()[:1]
    right = right.upper()[:1]
    return _base(item, context) + f"""
Compare only choices {left} and {right}. Which one answers the question better in its original language?
Output only {left} or {right}. No other text.
Answer:"""


def judge_prompt(item: MCQItem, votes: dict[str, str], context: str = "") -> str:
    vote_text = ", ".join(f"{k}={v}" for k, v in sorted(votes.items())) or "none"
    return _base(item, context) + f"""
Sub-solvers proposed: {vote_text}.
Re-check carefully, especially negation/exception wording, numbers, and near-duplicate choices across any language.
Output exactly one letter A/B/C/D.
Final answer:"""
