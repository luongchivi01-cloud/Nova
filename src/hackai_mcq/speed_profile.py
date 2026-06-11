from __future__ import annotations

"""Speed profile planner for the official MCQ runtime.

The goal is not to make the system weaker.  It spends expensive calls only on
rows where they are likely to change the answer, while keeping strict output
contracts and model-only official decisions.
"""

from dataclasses import dataclass, field
from typing import Iterable

from .schema import MCQItem


@dataclass(slots=True)
class SpeedPlan:
    profile: str
    effective_mode: str
    rag_k: int
    max_context_chars: int
    allow_fast_token_exit: bool
    token_exit_margin: float
    skip_rag: bool
    skip_option_matrix: bool
    skip_permutation: bool
    skip_verifier: bool
    skip_pairwise: bool
    skip_multilingual_sanity: bool
    max_pairwise_calls: int
    notes: str = ""

    @property
    def is_turbo(self) -> bool:
        return self.profile in {"fast", "turbo"}


def normalize_speed_profile(raw: str | None) -> str:
    p = (raw or "balanced").strip().lower()
    aliases = {
        "safe": "balanced",
        "normal": "balanced",
        "speed": "fast",
        "quick": "fast",
        "max": "accuracy",
        "max_accuracy": "accuracy",
    }
    p = aliases.get(p, p)
    if p not in {"turbo", "fast", "balanced", "accuracy"}:
        return "balanced"
    return p


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def build_speed_plan(
    *,
    profile: str,
    configured_mode: str,
    base_mode: str,
    difficulty: float,
    risk_score: float,
    time_pressure: bool,
    token_margin: float | None = None,
    rag_k: int = 3,
    rag_max_chars: int = 1800,
    max_pairwise_calls: int = 3,
) -> SpeedPlan:
    """Create a deterministic row-level speed plan.

    Profiles:
    - turbo: fastest safe route. Uses token scoring/direct for low-risk rows and
      deepens only for high-risk rows.
    - fast: similar, but keeps a bit more retrieval and vote logic.
    - balanced: default competition profile.
    - accuracy: spends more compute; still skips wasteful work under time pressure.
    """
    profile = normalize_speed_profile(profile)
    configured_mode = (configured_mode or "adaptive").lower()
    mode = base_mode
    notes: list[str] = [f"speed={profile}"]

    # Dynamic retrieval budget.  Keep evidence short for speed; expand only for
    # genuinely risky rows or explicit accuracy mode.
    dynamic_k = max(0, int(rag_k))
    dynamic_chars = max(256, int(rag_max_chars))
    skip_rag = False
    if profile == "turbo":
        dynamic_k = 0 if risk_score < 0.45 and difficulty < 0.55 else min(dynamic_k, 1)
        dynamic_chars = min(dynamic_chars, 700)
        skip_rag = dynamic_k == 0
        if risk_score < 0.62 and difficulty < 0.78:
            mode = "direct"
    elif profile == "fast":
        dynamic_k = min(dynamic_k, 1 if risk_score < 0.55 else 2)
        dynamic_chars = min(dynamic_chars, 950 if risk_score < 0.70 else 1250)
        if risk_score < 0.50 and difficulty < 0.68:
            mode = "direct"
    elif profile == "balanced":
        dynamic_k = min(dynamic_k, 2 if risk_score < 0.70 else max(2, dynamic_k))
        dynamic_chars = min(dynamic_chars, 1400 if risk_score < 0.70 else dynamic_chars)
    else:  # accuracy
        dynamic_k = max(1, dynamic_k)
        dynamic_chars = max(dynamic_chars, 1600)
        if configured_mode == "max_accuracy" and not time_pressure:
            mode = "max_accuracy"

    if time_pressure:
        notes.append("time_pressure")
        dynamic_k = min(dynamic_k, 1 if risk_score < 0.80 else 2)
        dynamic_chars = min(dynamic_chars, 900 if risk_score < 0.80 else 1200)
        if mode == "max_accuracy":
            mode = "judge" if risk_score >= 0.75 else "vote"
        elif mode == "judge" and risk_score < 0.65:
            mode = "vote"
        elif mode == "vote" and risk_score < 0.40:
            mode = "direct"

    # Cheap early exit only when the row is clearly safe.  It still uses the
    # legal model's A/B/C/D score, not a heuristic fallback.
    base_exit = {"turbo": 0.18, "fast": 0.22, "balanced": 0.30, "accuracy": 0.42}[profile]
    risk_penalty = 0.16 * risk_score + 0.08 * difficulty
    exit_margin = _clamp(base_exit + risk_penalty, 0.18, 0.62)
    allow_fast_exit = profile in {"turbo", "fast", "balanced"} and risk_score < 0.58 and difficulty < 0.75
    if token_margin is not None and token_margin >= exit_margin and allow_fast_exit:
        notes.append(f"token_exit_ready:{token_margin:.3f}>={exit_margin:.3f}")

    skip_option_matrix = profile in {"turbo", "fast"} and risk_score < (0.70 if profile == "turbo" else 0.52)
    skip_permutation = profile in {"turbo", "fast"} and risk_score < (0.82 if profile == "turbo" else 0.62)
    skip_verifier = profile in {"turbo", "fast"} and risk_score < (0.84 if profile == "turbo" else 0.68)
    skip_pairwise = profile in {"turbo", "fast"} and risk_score < (0.90 if profile == "turbo" else 0.72)
    skip_multilingual_sanity = profile in {"turbo", "fast"} and risk_score < 0.62

    pairwise_calls = max(0, int(max_pairwise_calls))
    if skip_pairwise:
        pairwise_calls = 0
    elif profile == "fast":
        pairwise_calls = min(pairwise_calls, 1)
    elif profile == "balanced" or time_pressure:
        pairwise_calls = min(pairwise_calls, 2)

    return SpeedPlan(
        profile=profile,
        effective_mode=mode,
        rag_k=dynamic_k,
        max_context_chars=dynamic_chars,
        allow_fast_token_exit=allow_fast_exit,
        token_exit_margin=exit_margin,
        skip_rag=skip_rag,
        skip_option_matrix=skip_option_matrix,
        skip_permutation=skip_permutation,
        skip_verifier=skip_verifier,
        skip_pairwise=skip_pairwise,
        skip_multilingual_sanity=skip_multilingual_sanity,
        max_pairwise_calls=pairwise_calls,
        notes=";".join(notes),
    )


@dataclass(slots=True)
class SpeedTelemetry:
    rows: int = 0
    fast_token_exits: int = 0
    direct_rows: int = 0
    deep_rows: int = 0
    rag_skips: int = 0
    option_matrix_skips: int = 0
    generation_calls_avoided: int = 0
    notes: list[str] = field(default_factory=list)

    def record(self, result_strategy: str, plan: SpeedPlan, fast_exit: bool = False) -> None:
        self.rows += 1
        if fast_exit:
            self.fast_token_exits += 1
            self.generation_calls_avoided += 1
        if result_strategy in {"direct", "token_fast_exit", "memory_reuse"}:
            self.direct_rows += 1
        else:
            self.deep_rows += 1
        if plan.skip_rag:
            self.rag_skips += 1
        if plan.skip_option_matrix:
            self.option_matrix_skips += 1

    def summary(self) -> dict[str, int | float | str]:
        return {
            "rows": self.rows,
            "fast_token_exits": self.fast_token_exits,
            "direct_rows": self.direct_rows,
            "deep_rows": self.deep_rows,
            "rag_skips": self.rag_skips,
            "option_matrix_skips": self.option_matrix_skips,
            "generation_calls_avoided_estimate": self.generation_calls_avoided,
            "fast_exit_rate": round(self.fast_token_exits / self.rows, 4) if self.rows else 0.0,
        }
