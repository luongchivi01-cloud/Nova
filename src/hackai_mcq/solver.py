from __future__ import annotations

import time
from dataclasses import dataclass

from .answer_parser import deterministic_fallback, parse_answer
from .answer_repair import repair_with_same_backend
from .answer_quality_gate import assess_result_quality, normalize_result
from .calibration import AnswerPriorCalibrator
from .cache import PromptCache
from .config import RuntimeConfig
from .ensembling import weighted_vote
from .features import choose_solver_mode, estimate_difficulty, has_negation
from .confidence_calibrator import calibrate_confidence
from .decision_arbitrator import arbitrate_decision
from .option_permutation import run_permutation_check
from .time_budget_controller import TimeBudgetController
from .speed_profile import build_speed_plan
from .accuracy_stability_guard import evaluate_early_exit, guard_note
from .prompt_compactor import compact_context, trim_prompt
from .token_choice_scorer import score_with_backend
from .knowledge_option_scorer import score_options_from_evidence
from .evidence_consensus import evaluate_evidence_consensus, consensus_score_map
from .multilingual_nlp_adapter import analyze_multilingual
from .question_memory import QuestionMemory
from .risk_gate import assess_risk
from .option_evidence_matrix import build_option_evidence_matrix
from .model_backends import Backend, ChoiceScoringBackend
from .prompts import (
    constrained_choice_prompt,
    cpu_choice_prompt,
    cpu_verifier_prompt,
    direct_prompt,
    elimination_prompt,
    judge_prompt,
    negation_guard_prompt,
    pairwise_prompt,
    translation_sanity_prompt,
    scoring_prompt,
    verifier_prompt,
)
from .rag import LexicalRAG
from .schema import MCQItem, SolverResult


@dataclass
class AdaptiveSolver:
    backend: Backend
    config: RuntimeConfig
    rag: LexicalRAG | None = None
    calibrator: AnswerPriorCalibrator | None = None
    cache: PromptCache | None = None
    question_memory: QuestionMemory | None = None
    start_time: float = 0.0
    total_rows: int = 0


    def _finish(self, item: MCQItem, result: SolverResult) -> SolverResult:
        result = normalize_result(item, result)
        quality = assess_result_quality(item, result, confidence_threshold=self.config.confidence_threshold)
        if not quality.ok and self.config.strict_no_fallback:
            raise RuntimeError(f"quality gate rejected qid={item.qid}: {quality.reason}")
        if quality.needs_rescue:
            result = SolverResult(
                result.qid,
                result.answer,
                result.confidence,
                result.strategy,
                result.votes,
                result.weighted,
                (result.notes or "") + f";quality_gate={quality.reason}",
            )
        if self.question_memory and result.answer:
            self.question_memory.remember(item, result.answer, result.confidence)
        return result

    def _evidence(self, item: MCQItem, difficulty: float, k: int | None = None, skip: bool = False):
        if skip or not self.rag or not self.config.enable_rag:
            return []
        effective_k = self.config.rag_k if k is None else max(0, int(k))
        if effective_k <= 0:
            return []
        # In the self-loading knowledge build, use evidence more often: it is local, deterministic,
        # and helps weak models. Time pressure still keeps k small through config.
        if self.config.mode not in {"max_accuracy", "judge"} and difficulty < 0.32:
            return []
        try:
            return self.rag.search(item, k=effective_k)
        except AttributeError:
            try:
                return self.rag.search(item.text_for_retrieval(), k=effective_k)
            except Exception:
                if self.config.strict_no_fallback:
                    raise
                return []

    def _context_from_evidence(self, item: MCQItem, evidence, max_chars: int | None = None) -> str:
        if not evidence:
            return ""
        try:
            from .evidence_compressor import compress_evidence
            raw = compress_evidence(item.text_for_retrieval(), evidence, max_chars=max_chars or self.config.rag_max_chars)
            return compact_context(raw, max_chars=max_chars or self.config.rag_max_chars)
        except Exception:
            raw = "\n".join(getattr(e, "text", str(e)) for e in list(evidence)[: self.config.rag_k])
            return compact_context(raw, max_chars=max_chars or self.config.rag_max_chars)

    def _context(self, item: MCQItem, difficulty: float) -> str:
        return self._context_from_evidence(item, self._evidence(item, difficulty))

    def _ask(self, prompt: str, item: MCQItem, allowed: set[str] | None = None) -> str | None:
        prompt = trim_prompt(prompt, getattr(self.config, "prompt_hard_limit_chars", 4500))
        last: str | None = None
        last_error: Exception | None = None
        for _ in range(max(1, self.config.max_retries + 1)):
            try:
                cached = self.cache.get(prompt, item.qid) if self.cache else None
                if cached is not None:
                    raw = cached
                else:
                    raw = self.backend.generate(prompt, item)
                    if self.cache:
                        self.cache.set(prompt, raw, item.qid)
                last = parse_answer(raw)
                if last and (allowed is None or last in allowed):
                    return last
            except Exception as e:
                last_error = e
                last = None
        if last_error is not None and self.config.strict_no_fallback:
            raise RuntimeError(f"backend generation failed for qid={item.qid}") from last_error
        if last and allowed is not None and last not in allowed:
            return None
        return last

    def _score_choice(self, item: MCQItem, context: str) -> tuple[str | None, float, dict[str, float]]:
        if not self.config.use_token_scoring or not isinstance(self.backend, ChoiceScoringBackend):
            return None, 0.0, {}
        try:
            res = score_with_backend(self.backend, constrained_choice_prompt(item, context), item, temperature=1.0)
            probs = res.probabilities
            if self.calibrator and probs:
                probs = self.calibrator.apply(probs)
                # Re-normalize through weighted_vote path by keeping probabilities.
                top = max(probs, key=probs.get) if probs else res.answer
                sorted_vals = sorted(probs.values(), reverse=True)
                margin = sorted_vals[0] - sorted_vals[1] if len(sorted_vals) > 1 else sorted_vals[0] if sorted_vals else 0.0
                return top, margin, probs
            return res.answer, res.margin, probs
        except Exception as e:
            if self.config.strict_no_fallback:
                raise RuntimeError(f"choice scoring failed for qid={item.qid}") from e
            return None, 0.0, {}

    def _pairwise_tournament(self, item: MCQItem, current: str | None, votes: dict[str, str], context: str, max_calls_override: int | None = None) -> dict[str, str]:
        max_calls = self.config.max_pairwise_calls if max_calls_override is None else max_calls_override
        if not self.config.use_pairwise_judge or max_calls <= 0:
            return {}
        # Compare the current weighted winner against strong alternatives; cheap and often fixes close calls.
        order = []
        if current:
            order.append(current)
        for ans in votes.values():
            if ans and ans not in order:
                order.append(ans)
        for ans in "ABCD":
            if ans not in order:
                order.append(ans)
        champion = order[0]
        out: dict[str, str] = {}
        calls = 0
        for challenger in order[1:]:
            if calls >= max_calls:
                break
            if challenger == champion:
                continue
            winner = self._ask(pairwise_prompt(item, champion, challenger, context), item, allowed={champion, challenger})
            calls += 1
            if winner:
                out[f"pair_{champion}_vs_{challenger}"] = winner
                champion = winner
        return out

    def _permutation_votes(self, item: MCQItem, current: str | None, context: str) -> tuple[dict[str, str], bool | None, str]:
        if not getattr(self.config, "use_permutation_check", True):
            return {}, None, "perm=off"
        def ask_perm(pitem: MCQItem) -> str | None:
            return self._ask(direct_prompt(pitem, context), pitem)
        res = run_permutation_check(item, ask_perm, current_answer=current, max_checks=getattr(self.config, "permutation_checks", 2))
        return res.votes, res.consistent, res.notes

    def _solve_cpu_portable(self, item: MCQItem, difficulty: float) -> SolverResult:
        """One constrained call for easy rows, plus one verifier for risky rows."""
        risk = assess_risk(item)
        high_risk = bool(risk.should_deepen or difficulty >= 0.55 or has_negation(item))
        evidence = self._evidence(item, difficulty, k=min(2, self.config.rag_k)) if self.config.enable_rag else []
        context = self._context_from_evidence(item, evidence, max_chars=min(1000, self.config.rag_max_chars))
        direct = self._ask(cpu_choice_prompt(item, context), item)
        if not direct:
            raise RuntimeError(f"CPU constrained direct call produced no valid answer for qid={item.qid}")
        votes = {"direct": direct}
        final = direct
        strategy = "cpu_direct"
        if high_risk:
            verified = self._ask(cpu_verifier_prompt(item, direct, context), item)
            if not verified:
                raise RuntimeError(f"CPU verifier produced no valid answer for qid={item.qid}")
            votes["verifier"] = verified
            final = verified
            strategy = "cpu_verified"
        return self._finish(item, SolverResult(
            item.qid, final, 0.72 if high_risk else 0.66, strategy, votes, {final: 1.0},
            f"cpu_portable=1;model_calls={2 if high_risk else 1};difficulty={difficulty:.3f};risk={risk.score:.3f};rag_hits={len(evidence)}",
        ))

    def solve(self, item: MCQItem, index: int = 0) -> SolverResult:
        if not self.start_time:
            self.start_time = time.time()
        elapsed = time.time() - self.start_time
        difficulty = estimate_difficulty(item)
        if getattr(self.config, "cpu_portable", False):
            return self._solve_cpu_portable(item, difficulty)
        base_mode = choose_solver_mode(difficulty, self.config.mode, elapsed, self.config.time_budget_seconds, index, self.total_rows)
        time_decision = TimeBudgetController(self.total_rows, self.config.time_budget_seconds).decide(
            self.config.mode, base_mode, elapsed, max(1, index), difficulty
        ) if getattr(self.config, "enable_time_controller", True) else None
        mode = time_decision.mode if time_decision else base_mode
        time_pressure = bool(time_decision.time_pressure) if time_decision else False

        # Reuse exact/option-permutation duplicates inside the same 2000-row run.
        # This improves stability and speed without relying on any external data.
        if self.question_memory:
            hit = self.question_memory.lookup(item)
            if hit and hit.answer:
                return self._finish(item, SolverResult(item.qid, hit.answer, hit.confidence, "memory_reuse", {"memory": hit.answer}, {hit.answer: hit.confidence}, hit.reason))

        pre_risk = assess_risk(item) if getattr(self.config, "use_risk_gate", True) else None
        if pre_risk and pre_risk.should_deepen and not time_pressure and mode == "direct":
            mode = "vote"
        elif pre_risk and pre_risk.score >= 0.62 and not time_pressure and mode in {"adaptive", "vote"}:
            mode = "judge"

        speed_plan = build_speed_plan(
            profile=getattr(self.config, "speed_profile", "balanced"),
            configured_mode=self.config.mode,
            base_mode=mode,
            difficulty=difficulty,
            risk_score=(pre_risk.score if pre_risk else 0.0),
            time_pressure=time_pressure,
            rag_k=self.config.rag_k,
            rag_max_chars=self.config.rag_max_chars,
            max_pairwise_calls=self.config.max_pairwise_calls,
        )
        mode = speed_plan.effective_mode

        evidence = self._evidence(item, difficulty, k=speed_plan.rag_k, skip=speed_plan.skip_rag)
        if (not evidence) and pre_risk and pre_risk.should_force_evidence and self.rag and self.config.enable_rag and not speed_plan.skip_rag:
            try:
                evidence = self.rag.search(item, k=max(1, speed_plan.rag_k))
            except AttributeError:
                evidence = self.rag.search(item.text_for_retrieval(), k=max(1, speed_plan.rag_k))
        context = self._context_from_evidence(item, evidence, max_chars=speed_plan.max_context_chars)
        votes: dict[str, str] = {}
        score_map: dict[str, float] = {}

        # Option-aware retrieval is a strong low-cost upgrade for weak models:
        # it retrieves evidence separately for A/B/C/D and gives the LLM an
        # explicit support matrix instead of one generic context blob.
        option_matrix_notes = ""
        if getattr(self.config, "use_option_evidence_matrix", True) and (not speed_plan.skip_option_matrix) and self.rag and (evidence or (pre_risk and pre_risk.should_deepen) or mode in {"judge", "max_accuracy"}):
            try:
                matrix = build_option_evidence_matrix(item, self.rag, k_per_option=min(getattr(self.config, "option_evidence_k", 2), max(1, speed_plan.rag_k or 1)), max_context_chars=max(600, int(speed_plan.max_context_chars * 0.60)))
                if matrix.context:
                    context = compact_context((context + "\n\n" + matrix.context).strip(), max_chars=speed_plan.max_context_chars + 600)
                mscores = matrix.score_map()
                for k, v in mscores.items():
                    score_map[k] = score_map.get(k, 0.0) + getattr(self.config, "option_evidence_weight", 0.22) * float(v)
                if matrix.best and matrix.margin >= 0.035:
                    votes["option_evidence_matrix"] = matrix.best
                option_matrix_notes = f";matrix_best={matrix.best};matrix_margin={matrix.margin:.3f}"
            except Exception:
                if self.config.strict_no_fallback and getattr(self.config, "require_feature_graph", True):
                    raise

        # 0) Offline knowledge prior: a soft evidence-based signal from the self-loaded corpus.
        # This is not a heuristic fallback; it is fused with the legal model/vote pipeline.
        if getattr(self.config, "use_knowledge_prior", True) and evidence:
            kp = score_options_from_evidence(item, evidence, min_confidence=getattr(self.config, "knowledge_prior_min_confidence", 0.34))
            if kp.answer:
                votes["knowledge_prior"] = kp.answer
                for k, v in kp.scores.items():
                    score_map[k] = score_map.get(k, 0.0) + 0.35 * kp.confidence * float(v)
            # Evidence consensus is a second, independent soft signal. It is weak
            # enough to avoid overriding the model, but strong enough to stabilize
            # weak/local models on factual MCQ questions.
            try:
                consensus = evaluate_evidence_consensus(item, evidence)
                c_scores = consensus_score_map(item, evidence) if consensus.ok else {}
                if c_scores:
                    best = max(c_scores, key=c_scores.get)
                    votes["evidence_consensus"] = best
                    for k, v in c_scores.items():
                        score_map[k] = score_map.get(k, 0.0) + 0.18 * float(v)
            except Exception:
                if self.config.strict_no_fallback and getattr(self.config, "require_feature_graph", True):
                    raise

        # 1) Fast/high-value path: next-token scoring.
        score_ans, margin, probs = self._score_choice(item, context)
        if score_ans:
            votes["token_score"] = score_ans
            score_map.update(probs)
            if getattr(self.config, "token_fast_exit", True) and speed_plan.allow_fast_token_exit and margin >= speed_plan.token_exit_margin:
                state = weighted_vote(votes, score_map)
                conf = max(state.confidence, 0.55 + min(0.35, margin))
                guard = evaluate_early_exit(
                    item,
                    answer=state.answer or score_ans,
                    strategy="token_fast_exit",
                    difficulty=difficulty,
                    risk_score=(pre_risk.score if pre_risk else 0.0),
                    token_margin=margin,
                    vote_margin=state.margin,
                    confidence=conf,
                    votes=state.votes,
                    score_map=state.weighted or score_map,
                    mode=mode,
                    time_pressure=time_pressure,
                    profile=getattr(self.config, "speed_profile", "balanced"),
                )
                if guard.allow:
                    return self._finish(item, SolverResult(item.qid, state.answer or score_ans, conf, "token_fast_exit", state.votes, state.weighted, f"difficulty={difficulty:.2f};margin={margin:.3f};risk={(pre_risk.score if pre_risk else 0.0):.2f};{speed_plan.notes}" + option_matrix_notes + guard_note(guard)))
                votes["token_guarded"] = state.answer or score_ans
            if mode == "direct" and margin >= self.config.uncertain_margin and (speed_plan.profile == "turbo" or time_pressure):
                state = weighted_vote(votes, score_map)
                return self._finish(item, SolverResult(item.qid, state.answer or score_ans, state.confidence, "token_score", state.votes, state.weighted, f"difficulty={difficulty:.2f};margin={margin:.3f};risk={(pre_risk.score if pre_risk else 0.0):.2f};{speed_plan.notes}"))

        # 2) Direct generation. Always run unless score margin is extremely strong in time pressure.
        direct = self._ask(direct_prompt(item, context), item)
        if direct:
            votes["direct"] = direct

        state = weighted_vote(votes, score_map)
        arb = arbitrate_decision(item, votes, score_map, token_margin=margin, risk_score=(pre_risk.score if pre_risk else 0.0), time_pressure=time_pressure) if getattr(self.config, "use_decision_arbitrator", True) else None
        if mode == "direct" and state.answer and not (arb and arb.should_deepen):
            cal = calibrate_confidence(item, state, token_margin=margin, time_pressure=time_pressure, threshold=self.config.confidence_threshold) if getattr(self.config, "confidence_calibration", True) else None
            chosen = arb.answer if arb and arb.answer else state.answer
            conf = max((cal.confidence if cal else max(state.confidence, 0.45 + margin)), (arb.confidence if arb else 0.0))
            weighted = arb.scores if arb and arb.scores else state.weighted
            guard = evaluate_early_exit(
                item,
                answer=chosen,
                strategy="direct",
                difficulty=difficulty,
                risk_score=(pre_risk.score if pre_risk else 0.0),
                token_margin=margin,
                vote_margin=state.margin,
                confidence=conf,
                votes=state.votes,
                score_map=weighted,
                mode=mode,
                time_pressure=time_pressure,
                profile=getattr(self.config, "speed_profile", "balanced"),
            )
            if guard.allow:
                notes = f"difficulty={difficulty:.2f};margin={margin:.3f};{cal.notes if cal else ''};{time_decision.notes if time_decision else ''};{arb.notes if arb else ''}"
                return self._finish(item, SolverResult(item.qid, chosen, conf, "direct", state.votes, weighted, notes + option_matrix_notes + f";{speed_plan.notes}" + guard_note(guard) + (f";risk={pre_risk.score:.2f};risk_reasons={'|'.join(pre_risk.reasons[:4])}" if pre_risk else "")))
            # Direct signal was not strong enough; keep it as a vote and continue to verifier/vote/judge.
            votes["direct_guarded"] = chosen

        # 3) Vote mode: use different reasoning frames.
        elimination = self._ask(elimination_prompt(item, context), item)
        if elimination:
            votes["elimination"] = elimination
        scoring = self._ask(scoring_prompt(item, context), item)
        if scoring:
            votes["scoring"] = scoring
        if has_negation(item):
            neg = self._ask(negation_guard_prompt(item, context), item)
            if neg:
                votes["negation_guard"] = neg

        # Non-Vietnamese and mixed-language rows often fail because a Vietnamese-only prompt
        # assumes the wrong negation/domain cues. Add one cheap sanity pass only where it matters.
        lang_sig = analyze_multilingual(item.text_for_retrieval())
        if (not speed_plan.skip_multilingual_sanity) and (lang_sig.language not in {"vi", "en", "unknown"} or lang_sig.is_mixed_language):
            trans = self._ask(translation_sanity_prompt(item, context), item)
            if trans:
                votes["multilingual_sanity"] = trans

        state = weighted_vote(votes, score_map)
        perm_consistent = None
        perm_notes = ""
        # Option permutation is relatively cheap and catches A/B/C positional bias.
        # Run it only when the row is uncertain or explicitly in judge/max modes.
        if (not speed_plan.skip_permutation) and (mode in {"judge", "max_accuracy"} or (state.margin < 0.22 and not time_pressure)):
            perm_votes, perm_consistent, perm_notes = self._permutation_votes(item, state.answer, context)
            if perm_votes:
                votes.update(perm_votes)
                state = weighted_vote(votes, score_map)
        arb = arbitrate_decision(item, votes, score_map, token_margin=margin, risk_score=(pre_risk.score if pre_risk else 0.0), time_pressure=time_pressure, permutation_consistent=perm_consistent) if getattr(self.config, "use_decision_arbitrator", True) else None
        cal = calibrate_confidence(item, state, token_margin=margin, permutation_consistent=perm_consistent, time_pressure=time_pressure, threshold=self.config.confidence_threshold) if getattr(self.config, "confidence_calibration", True) else None
        route_conf = max((cal.confidence if cal else state.confidence), (arb.confidence if arb else 0.0))
        if state.answer and mode == "vote" and not (arb and arb.should_deepen) and (state.margin >= 0.18 or route_conf >= self.config.confidence_threshold):
            chosen = arb.answer if arb and arb.answer else state.answer
            conf = route_conf
            weighted = arb.scores if arb and arb.scores else state.weighted
            guard = evaluate_early_exit(
                item,
                answer=chosen,
                strategy="vote",
                difficulty=difficulty,
                risk_score=(pre_risk.score if pre_risk else 0.0),
                token_margin=margin,
                vote_margin=state.margin,
                confidence=conf,
                votes=state.votes,
                score_map=weighted,
                mode=mode,
                time_pressure=time_pressure,
                profile=getattr(self.config, "speed_profile", "balanced"),
            )
            if guard.allow:
                return self._finish(item, SolverResult(item.qid, chosen, conf, "vote", state.votes, weighted, f"difficulty={difficulty:.2f};margin={margin:.3f};wm={state.margin:.3f};{perm_notes};{cal.notes if cal else ''};{arb.notes if arb else ''};{time_decision.notes if time_decision else ''}" + option_matrix_notes + f";{speed_plan.notes}" + guard_note(guard) + (f";risk={pre_risk.score:.2f};risk_reasons={'|'.join(pre_risk.reasons[:4])}" if pre_risk else "")))
            votes["vote_guarded"] = chosen

        # 4) Verifier catches direct/vote mistakes. Use heavily in max_accuracy/judge mode.
        if self.config.use_verifier and (not speed_plan.skip_verifier) and state.answer and mode in {"judge", "max_accuracy"}:
            ver = self._ask(verifier_prompt(item, state.answer, context), item)
            if ver:
                votes["verifier"] = ver
                state = weighted_vote(votes, score_map)

        # 5) Pairwise tournament on unresolved conflicts.
        if (not speed_plan.skip_pairwise) and mode in {"judge", "max_accuracy"} and (state.margin < 0.30 or self.config.mode == "max_accuracy"):
            pair_votes = self._pairwise_tournament(item, state.answer, votes, context, max_calls_override=speed_plan.max_pairwise_calls)
            votes.update(pair_votes)
            state = weighted_vote(votes, score_map)

        # 6) Final judge only when still hard/conflicted, or always in max_accuracy.
        if mode in {"judge", "max_accuracy"} and (state.margin < 0.34 or self.config.mode == "max_accuracy"):
            judge = self._ask(judge_prompt(item, votes, context), item)
            if judge:
                votes["judge"] = judge
                state = weighted_vote(votes, score_map)

        arb_final = arbitrate_decision(item, votes, score_map, token_margin=margin, risk_score=(pre_risk.score if pre_risk else 0.0), time_pressure=time_pressure, permutation_consistent=perm_consistent) if getattr(self.config, "use_decision_arbitrator", True) else None
        final = (arb_final.answer if arb_final and arb_final.answer else None) or state.answer or score_ans or direct
        if not final:
            # Strict mode forbids weak heuristic fallback, but still allows same-model
            # repair with a more constrained prompt. This improves reliability without
            # violating the official backend/model boundary.
            final = repair_with_same_backend(self.backend, item, context=context, votes=votes)
        if not final:
            if self.config.strict_no_fallback:
                raise RuntimeError(f"No valid A/B/C/D answer produced for qid={item.qid}; strict mode forbids deterministic fallback.")
            final = deterministic_fallback(item.qid, item.question)
        strategy = "max_accuracy" if self.config.mode == "max_accuracy" else ("judge" if mode == "judge" else "vote")
        cal = calibrate_confidence(item, state, token_margin=margin, permutation_consistent=perm_consistent, time_pressure=time_pressure, threshold=self.config.confidence_threshold) if getattr(self.config, "confidence_calibration", True) else None
        conf = max((cal.confidence if cal and state.answer else (state.confidence if state.answer else 0.20)), (arb_final.confidence if arb_final else 0.0))
        weighted = (arb_final.scores if arb_final and arb_final.scores else (state.weighted or score_map))
        return self._finish(item, SolverResult(item.qid, final, conf, strategy, state.votes, weighted, f"difficulty={difficulty:.2f};token_margin={margin:.3f};vote_margin={state.margin:.3f};{perm_notes};{cal.notes if cal else ''};{arb_final.notes if arb_final else ''};{time_decision.notes if time_decision else ''}" + option_matrix_notes + f";{speed_plan.notes}" + (f";risk={pre_risk.score:.2f};risk_reasons={'|'.join(pre_risk.reasons[:4])}" if pre_risk else "")))
