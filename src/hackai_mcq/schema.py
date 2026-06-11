from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

QID_COLUMNS = ("qid", "id", "question_id", "ma_cau_hoi", "mã_câu_hỏi", "index")
QUESTION_COLUMNS = ("question", "cau_hoi", "câu_hỏi", "prompt", "content", "noi_dung", "nội_dung")

OPTION_COLUMNS: dict[str, tuple[str, ...]] = {
    "A": ("A", "a", "option_a", "answer_a", "ans_a", "choice_a", "dap_an_a", "đáp_án_a"),
    "B": ("B", "b", "option_b", "answer_b", "ans_b", "choice_b", "dap_an_b", "đáp_án_b"),
    "C": ("C", "c", "option_c", "answer_c", "ans_c", "choice_c", "dap_an_c", "đáp_án_c"),
    "D": ("D", "d", "option_d", "answer_d", "ans_d", "choice_d", "dap_an_d", "đáp_án_d"),
}

VALID_ANSWERS = {"A", "B", "C", "D"}

@dataclass(slots=True)
class MCQItem:
    qid: str
    question: str
    options: dict[str, str]
    raw: dict[str, Any] = field(default_factory=dict)

    def option_block(self) -> str:
        return "\n".join(f"{k}. {self.options.get(k, '').strip()}" for k in "ABCD")

    def text_for_retrieval(self) -> str:
        return self.question + "\n" + self.option_block()

@dataclass(slots=True)
class SolverResult:
    qid: str
    answer: str
    confidence: float = 0.0
    strategy: str = "unknown"
    votes: dict[str, str] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    notes: str = ""

@dataclass(slots=True)
class RunStats:
    rows: int = 0
    seconds: float = 0.0
    backend: str = ""
    mode: str = ""
    fallback_count: int = 0
    vote_count: int = 0
    judge_count: int = 0
    direct_count: int = 0
    errors: int = 0
