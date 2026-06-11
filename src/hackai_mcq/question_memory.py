from __future__ import annotations

"""In-run question memory for private-test stability.

The official task can include repeated questions, paraphrases with the same
option texts, or duplicated rows where only option order changes. This module is
strictly in-process and deterministic: it never calls the network and never
stores private test content outside the current run unless the caller explicitly
writes traces.
"""

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Iterable

from .normalization import canonical
from .schema import MCQItem, VALID_ANSWERS


def _norm_text(text: str) -> str:
    return " ".join(canonical(text).split())


def _option_signature(item: MCQItem) -> tuple[str, ...]:
    # Option text multiset signature; robust to A/B/C/D permutation.
    return tuple(sorted(_norm_text(item.options.get(k, "")) for k in "ABCD"))


def _question_signature(item: MCQItem) -> str:
    return _norm_text(item.question)


@dataclass(slots=True)
class MemoryHit:
    answer: str
    confidence: float
    source_qid: str
    reason: str


@dataclass
class QuestionMemory:
    exact: dict[str, MemoryHit] = field(default_factory=dict)
    option_set: dict[tuple[str, tuple[str, ...]], MemoryHit] = field(default_factory=dict)
    question_bank: list[tuple[str, tuple[str, ...], MemoryHit]] = field(default_factory=list)
    fuzzy_threshold: float = 0.94
    enabled: bool = True

    def _exact_key(self, item: MCQItem) -> str:
        opts = "||".join(f"{k}:{_norm_text(item.options.get(k, ''))}" for k in "ABCD")
        return _question_signature(item) + "||" + opts

    def _option_key(self, item: MCQItem) -> tuple[str, tuple[str, ...]]:
        return (_question_signature(item), _option_signature(item))

    def lookup(self, item: MCQItem) -> MemoryHit | None:
        if not self.enabled:
            return None
        exact = self.exact.get(self._exact_key(item))
        if exact:
            return MemoryHit(exact.answer, min(0.99, exact.confidence + 0.03), exact.source_qid, "exact_duplicate")
        opt_key = self._option_key(item)
        by_opts = self.option_set.get(opt_key)
        if by_opts:
            # If the exact same option texts are present but order changed, map
            # the remembered answer text to the current letter.
            prev_answer = by_opts.answer
            prev_text = None
            # Stored reason can encode answer text after the final separator.
            if "answer_text=" in by_opts.reason:
                prev_text = by_opts.reason.split("answer_text=", 1)[-1]
            if prev_text:
                target = _norm_text(prev_text)
                for k in "ABCD":
                    if _norm_text(item.options.get(k, "")) == target:
                        return MemoryHit(k, min(0.98, by_opts.confidence), by_opts.source_qid, "option_permutation_duplicate")
            if prev_answer in VALID_ANSWERS:
                return MemoryHit(prev_answer, min(0.95, by_opts.confidence), by_opts.source_qid, "same_question_option_set")
        q = _question_signature(item)
        opts = _option_signature(item)
        best: tuple[float, MemoryHit] | None = None
        for old_q, old_opts, hit in self.question_bank[-512:]:
            if old_opts != opts:
                continue
            sim = SequenceMatcher(None, q, old_q).ratio()
            if sim >= self.fuzzy_threshold and (best is None or sim > best[0]):
                best = (sim, hit)
        if best:
            sim, hit = best
            return MemoryHit(hit.answer, min(hit.confidence, 0.90) * sim, hit.source_qid, f"fuzzy_duplicate:{sim:.3f}")
        return None

    def remember(self, item: MCQItem, answer: str, confidence: float = 0.0) -> None:
        if not self.enabled or answer not in VALID_ANSWERS:
            return
        answer_text = item.options.get(answer, "")
        hit = MemoryHit(answer, max(0.0, min(1.0, confidence)), item.qid, "answer_text=" + answer_text)
        self.exact[self._exact_key(item)] = hit
        self.option_set[self._option_key(item)] = hit
        self.question_bank.append((_question_signature(item), _option_signature(item), hit))
        if len(self.question_bank) > 4096:
            self.question_bank = self.question_bank[-2048:]
