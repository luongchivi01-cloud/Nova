from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping

from .schema import VALID_ANSWERS


def softmax(scores: Mapping[str, float], temperature: float = 1.0) -> dict[str, float]:
    if not scores:
        return {}
    temperature = max(1e-6, temperature)
    vals = {k: float(v) / temperature for k, v in scores.items() if k in VALID_ANSWERS}
    if not vals:
        return {}
    m = max(vals.values())
    exps = {k: math.exp(max(-50.0, min(50.0, v - m))) for k, v in vals.items()}
    z = sum(exps.values()) or 1.0
    return {k: v / z for k, v in exps.items()}


def top_margin(prob: Mapping[str, float]) -> tuple[str | None, float]:
    items = sorted(((k, float(v)) for k, v in prob.items() if k in VALID_ANSWERS), key=lambda kv: kv[1], reverse=True)
    if not items:
        return None, 0.0
    if len(items) == 1:
        return items[0][0], items[0][1]
    return items[0][0], items[0][1] - items[1][1]


@dataclass(slots=True)
class VoteState:
    votes: dict[str, str]
    scores: dict[str, float]
    weighted: dict[str, float]
    answer: str | None
    confidence: float
    margin: float


def weighted_vote(votes: Mapping[str, str | None], scores: Mapping[str, float] | None = None, weights: Mapping[str, float] | None = None) -> VoteState:
    weights = dict(weights or {})
    acc: dict[str, float] = defaultdict(float)
    valid_votes: dict[str, str] = {}

    # scores are already normalized probabilities in most call sites; use them as a soft prior.
    if scores:
        total_score = sum(max(0.0, float(v)) for v in scores.values())
        if total_score > 0:
            for k, v in scores.items():
                if k in VALID_ANSWERS:
                    acc[k] += 0.85 * (max(0.0, float(v)) / total_score)

    for name, ans in votes.items():
        if not ans:
            continue
        a = str(ans).strip().upper()[:1]
        if a not in VALID_ANSWERS:
            continue
        valid_votes[str(name)] = a
        # scoring/token outputs are a little stronger than plain generation; judge is strongest when called.
        w = weights.get(name)
        if w is None:
            if "token" in name or "score" in name:
                w = 1.15
            elif "judge" in name or "pair" in name:
                w = 1.25
            elif "multilingual" in name or "translation" in name:
                w = 1.08
            elif "direct" in name:
                w = 0.95
            else:
                w = 1.0
        acc[a] += float(w)

    if not acc:
        return VoteState(valid_votes, dict(scores or {}), {}, None, 0.0, 0.0)
    total = sum(acc.values()) or 1.0
    prob = {k: acc.get(k, 0.0) / total for k in "ABCD"}
    ans, margin = top_margin(prob)
    # confidence conservative: consensus + margin, capped to avoid overclaim.
    confidence = min(0.96, 0.35 + 0.55 * prob.get(ans or "A", 0.0) + 0.10 * margin)
    return VoteState(valid_votes, dict(scores or {}), prob, ans, confidence, margin)
