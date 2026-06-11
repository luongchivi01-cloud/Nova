"""
V8 batch-aware official solve loop.

This module is wired into cli.py automatically. It accelerates the normal
/data -> /output/pred.csv path without asking the judge to run extra commands.

Design:
- Phase 1: batch token scoring for all rows when backend supports it.
- Phase 2: batch direct generation for remaining uncertain rows.
- Phase 3: strict deep solver for genuinely hard rows.
No heuristic fallback is used in strict official mode.
"""
from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .solver import AdaptiveSolver
    from .schema import MCQItem, SolverResult


def batch_backend_available(backend: Any) -> bool:
    return callable(getattr(backend, "generate_batch", None)) or callable(getattr(backend, "batch_score_choices", None))


def compute_optimal_batch_size(vram_gb: float | None, model_size_b: float = 7.0, n_ctx: int = 1536) -> int:
    if vram_gb is None:
        return 4
    try:
        vram = float(vram_gb)
    except Exception:
        return 4
    model_mem_gb = max(1.0, float(model_size_b) * 0.55)
    available_for_kv = max(0.5, vram - model_mem_gb)
    kv_per_item_gb = 0.030 * (max(256, int(n_ctx)) / 1024)
    safe_batch = int(available_for_kv / kv_per_item_gb)
    return max(1, min(32, safe_batch))


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _profile_direct_limit(profile: str) -> float:
    profile = (profile or "balanced").strip().lower()
    if profile in {"turbo", "fast"}:
        return 0.42
    if profile in {"accuracy", "max_accuracy"}:
        return 0.24
    return 0.32


def run_batch_solve_loop(
    solver: "AdaptiveSolver",
    items: list["MCQItem"],
    batch_size: int = 8,
    log_every: int = 50,
    strict: bool = True,
) -> list["SolverResult"]:
    from .answer_parser import parse_answer
    from .accuracy_stability_guard import evaluate_early_exit
    from .features import estimate_difficulty
    from .prompts import constrained_choice_prompt, direct_prompt
    from .risk_gate import assess_risk
    from .schema import SolverResult

    backend = solver.backend
    results: list[SolverResult | None] = [None] * len(items)
    uncertain_indices: list[int] = list(range(len(items)))

    # Phase 1: batched token-choice scores.
    if callable(getattr(backend, "batch_score_choices", None)) and getattr(solver.config, "use_token_scoring", True):
        t0 = time.time()
        prompts: list[str] = []
        contexts: list[str] = []
        for item in items:
            diff = estimate_difficulty(item)
            ctx = solver._context(item, diff)
            contexts.append(ctx)
            prompts.append(constrained_choice_prompt(item, ctx))
        score_maps = backend.batch_score_choices(prompts, items)  # type: ignore[attr-defined]
        threshold = float(os.getenv("TOKEN_FAST_EXIT_MARGIN", str(getattr(solver.config, "token_fast_exit_margin", 0.35))))
        next_uncertain: list[int] = []
        fast = 0
        for i, (item, scores) in enumerate(zip(items, score_maps)):
            if not scores:
                next_uncertain.append(i)
                continue
            ordered = sorted(((float(v), k) for k, v in scores.items() if k in "ABCD"), reverse=True)
            if not ordered:
                next_uncertain.append(i)
                continue
            best_score, best = ordered[0]
            second = ordered[1][0] if len(ordered) > 1 else best_score
            margin = best_score - second
            difficulty = estimate_difficulty(item)
            risk = assess_risk(item) if getattr(solver.config, "use_risk_gate", True) else None
            risk_score = float(getattr(risk, "score", 0.0) or 0.0)
            # Accuracy guard: token logits are excellent for easy rows, but high-risk
            # negation/calculation/domain rows should not skip the verifier path on
            # a merely OK margin.  Raise the fast-exit threshold instead of turning
            # batching off completely.
            guarded_threshold = threshold
            if difficulty >= 0.64 or risk_score >= 0.62 or bool(getattr(risk, "should_deepen", False)):
                guarded_threshold += _env_float("TOKEN_FAST_EXIT_HARD_EXTRA", 0.20)
            elif difficulty >= 0.45 or risk_score >= 0.45:
                guarded_threshold += _env_float("TOKEN_FAST_EXIT_MEDIUM_EXTRA", 0.10)
            scores_clean = {k: float(v) for k, v in scores.items() if k in "ABCD"}
            confidence = min(0.96, 0.58 + max(0.0, margin) * 0.75)
            guard = evaluate_early_exit(
                item,
                answer=best,
                strategy="batch_token_fast_exit",
                difficulty=difficulty,
                risk_score=risk_score,
                token_margin=margin,
                vote_margin=margin,
                confidence=confidence,
                votes={"batch_token": best},
                score_map=scores_clean,
                mode=getattr(solver.config, "mode", "adaptive"),
                time_pressure=False,
                profile=getattr(solver.config, "speed_profile", "balanced"),
            )
            if getattr(solver.config, "token_fast_exit", True) and margin >= guarded_threshold and guard.allow:
                results[i] = solver._finish(item, SolverResult(
                    qid=item.qid,
                    answer=best,
                    confidence=confidence,
                    strategy="batch_token_fast_exit",
                    votes={"batch_token": best},
                    scores=scores_clean,
                    notes=f"v9_1_batch_phase1;margin={margin:.4f};threshold={guarded_threshold:.4f};difficulty={difficulty:.3f};risk={risk_score:.3f};{guard.note}",
                ))
                fast += 1
            else:
                next_uncertain.append(i)
        uncertain_indices = next_uncertain
        print(f"[hackai-v8] batch phase1 token_score fast={fast} uncertain={len(uncertain_indices)} elapsed={time.time()-t0:.1f}s", flush=True)

    # Phase 2: batched direct generation for uncertain rows.
    if uncertain_indices and callable(getattr(backend, "generate_batch", None)):
        t0 = time.time()
        batch_items = [items[i] for i in uncertain_indices]
        prompts = []
        for item in batch_items:
            diff = estimate_difficulty(item)
            prompts.append(direct_prompt(item, solver._context(item, diff)))
        raw_outputs = backend.generate_batch(prompts, batch_items)  # type: ignore[attr-defined]
        next_uncertain: list[int] = []
        accepted = 0
        direct_accept_all = _env_bool("BATCH_DIRECT_ACCEPT_ALL", False)
        direct_max_diff = _env_float("BATCH_DIRECT_MAX_DIFFICULTY", _profile_direct_limit(getattr(solver.config, "speed_profile", "balanced")))
        direct_max_risk = _env_float("BATCH_DIRECT_MAX_RISK", 0.42)
        for orig_i, item, raw in zip(uncertain_indices, batch_items, raw_outputs):
            ans = parse_answer(raw)
            difficulty = estimate_difficulty(item)
            risk = assess_risk(item) if getattr(solver.config, "use_risk_gate", True) else None
            risk_score = float(getattr(risk, "score", 0.0) or 0.0)
            should_deepen = bool(getattr(risk, "should_deepen", False))
            # V8.1 accuracy fix: do not let a parsable direct answer bypass
            # verifier/vote/judge for difficult private-test rows.  Batch direct is
            # only a safe fast exit for clearly easy, low-risk rows, or when the
            # operator deliberately enables BATCH_DIRECT_ACCEPT_ALL / direct mode.
            guard = evaluate_early_exit(
                item,
                answer=ans,
                strategy="batch_direct",
                difficulty=difficulty,
                risk_score=risk_score,
                token_margin=0.0,
                vote_margin=0.0,
                confidence=0.66,
                votes={"batch_direct": ans} if ans else {},
                score_map={ans: 0.66} if ans else {},
                mode=getattr(solver.config, "mode", "adaptive"),
                time_pressure=False,
                profile=getattr(solver.config, "speed_profile", "balanced"),
            )
            can_fast_accept = bool(ans) and (
                direct_accept_all
                or (getattr(solver.config, "mode", "adaptive") == "direct" and guard.allow)
                or (difficulty <= direct_max_diff and risk_score <= direct_max_risk and not should_deepen and guard.allow)
            )
            if can_fast_accept:
                results[orig_i] = solver._finish(item, SolverResult(
                    qid=item.qid,
                    answer=ans,
                    confidence=0.66,
                    strategy="batch_direct",
                    votes={"batch_direct": ans},
                    scores={ans: 0.66},
                    notes=f"v9_1_batch_phase2;difficulty={difficulty:.3f};risk={risk_score:.3f};{guard.note}",
                ))
                accepted += 1
            else:
                next_uncertain.append(orig_i)
        uncertain_indices = next_uncertain
        print(f"[hackai-v8] batch phase2 direct accepted={accepted} deep={len(uncertain_indices)} elapsed={time.time()-t0:.1f}s", flush=True)

    # Phase 3: full adaptive solver only for rows that need it.
    if uncertain_indices:
        print(f"[hackai-v8] batch phase3 deep_rows={len(uncertain_indices)}", flush=True)
    for count, i in enumerate(uncertain_indices, start=1):
        item = items[i]
        try:
            results[i] = solver.solve(item, index=i + 1)
        except Exception:
            if strict:
                raise
            from .answer_parser import deterministic_fallback
            ans = deterministic_fallback(item.qid, item.question)
            results[i] = SolverResult(item.qid, ans, 0.0, "batch_error_fallback", {}, {ans: 0.0}, "non_strict_batch_error")
        if log_every and count % max(1, log_every) == 0:
            print(f"[hackai-v8] deep solved={count}/{len(uncertain_indices)}", flush=True)

    final: list[SolverResult] = []
    for i, maybe in enumerate(results):
        if maybe is None:
            final.append(solver.solve(items[i], index=i + 1))
        else:
            final.append(maybe)
    return final
