from __future__ import annotations

"""Option-order stability checks.

Small LLMs can over-pick A/B/C by position. A cheap way to detect this is to
permute choices, ask one direct constrained question, and map the answer back to
original labels. If the answer remains stable, confidence increases; if not, the
solver escalates.
"""

from dataclasses import dataclass
from typing import Callable

from .schema import MCQItem, VALID_ANSWERS


PERMUTATIONS: tuple[tuple[str, str, str, str], ...] = (
    ("C", "A", "D", "B"),
    ("B", "D", "A", "C"),
)


@dataclass(slots=True)
class PermutationResult:
    votes: dict[str, str]
    consistent: bool | None
    agreement: float
    notes: str


def permute_item(item: MCQItem, order: tuple[str, str, str, str]) -> tuple[MCQItem, dict[str, str]]:
    """Return a new item and map new_label -> original_label."""
    new_labels = "ABCD"
    new_opts: dict[str, str] = {}
    inverse: dict[str, str] = {}
    for new_label, old_label in zip(new_labels, order):
        new_opts[new_label] = item.options.get(old_label, "")
        inverse[new_label] = old_label
    return MCQItem(qid=f"{item.qid}__perm", question=item.question, options=new_opts, raw=dict(item.raw)), inverse


def run_permutation_check(
    item: MCQItem,
    ask_fn: Callable[[MCQItem], str | None],
    *,
    current_answer: str | None = None,
    max_checks: int = 2,
) -> PermutationResult:
    votes: dict[str, str] = {}
    for i, order in enumerate(PERMUTATIONS[: max(0, max_checks)], start=1):
        pitem, inverse = permute_item(item, order)
        ans = ask_fn(pitem)
        if ans and ans in inverse:
            mapped = inverse[ans]
            if mapped in VALID_ANSWERS:
                votes[f"perm_{i}"] = mapped
    if not votes:
        return PermutationResult(votes, None, 0.0, "no permutation votes")
    target = current_answer if current_answer in VALID_ANSWERS else max(set(votes.values()), key=list(votes.values()).count)
    agree = sum(1 for v in votes.values() if v == target) / max(1, len(votes))
    consistent = agree >= 0.75
    return PermutationResult(votes, consistent, agree, f"target={target};agree={agree:.2f}")
