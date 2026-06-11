from __future__ import annotations

"""Same-model answer repair utilities.

This is not a heuristic fallback.  It uses the official mounted backend again with
stricter prompts when a generation was unparsable, preserving strict no-fallback
semantics while improving row-level reliability.
"""

from typing import Protocol

from .answer_parser import parse_answer
from .prompts import constrained_choice_prompt, judge_prompt
from .schema import MCQItem, VALID_ANSWERS


class RepairBackend(Protocol):
    def generate(self, prompt: str, item: MCQItem) -> str: ...


def repair_with_same_backend(backend: RepairBackend, item: MCQItem, context: str = "", votes: dict[str, str] | None = None) -> str | None:
    prompts = [
        constrained_choice_prompt(item, context),
        judge_prompt(item, votes or {}, context),
        constrained_choice_prompt(item, ""),
    ]
    for prompt in prompts:
        try:
            raw = backend.generate(prompt, item)
            ans = parse_answer(raw)
            if ans in VALID_ANSWERS:
                return ans
        except Exception:
            continue
    return None
