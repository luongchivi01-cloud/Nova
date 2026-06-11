from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from .config import RuntimeConfig, from_env
from .auto_integrations import compact_status_line, prepare_auto_integrations
from .strict_dependencies import assert_strict_runtime_ready
from .io_utils import read_items, validate_predictions, write_predictions, write_trace
from .official_contract import (
    ContractReport,
    official_model_probe,
    runtime_fingerprint,
    validate_input_contract,
    validate_output_contract,
    validate_pred_file_contract,
    write_contract_report,
)
from .runtime_supervisor import RuntimeSupervisor
from .runtime_invariant_guard import RuntimeInvariantGuard
from .resource_guard import ResourceGuard
from .result_ledger import ResultLedger
from .retrieval_cache import CachedRAG
from .preflight_stress_probe import run_preflight_probe
from .model_output_watchdog import OutputWatchdog
from .speed_report import SpeedReport, write_speed_report
from .official_autorun import compact_autorun_line, validate_autorun_contract, write_autorun_report, apply_official_autorun_defaults
from .runtime_auto_optimizer import optimize_runtime_config, write_auto_optimize_report
from .wiring_integrity import apply_wiring_integrity_guard, compact_wiring_line, write_wiring_report
from .model_backends import create_backend, detect_vram_gb
from .cli_batch_patch import run_batch_solve_loop, batch_backend_available, compute_optimal_batch_size
from .rag import create_rag
from .knowledge_engine import create_knowledge_engine_from_env
from .calibration import AnswerPriorCalibrator
from .cache import PromptCache
from .question_memory import QuestionMemory
from .safeguards import assert_no_browser_automation, assert_no_sensitive_files, optional_network_block
from .schema import RunStats
from .solver import AdaptiveSolver
from .stability_orchestrator import (
    build_stability_report,
    compact_report_line,
    configure_determinism,
    write_stability_report,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HackAIthon C MCQ solver")
    parser.add_argument("--input", default=None, help="CSV input. Defaults to /data/private_test.csv or /data/public_test.csv")
    parser.add_argument("--output", default=None, help="Output CSV. Defaults to /output/pred.csv")
    parser.add_argument("--backend", default=None, choices=["auto", "heuristic", "llama_cpp", "transformers", "vllm"], help="Inference backend")
    parser.add_argument("--model-path", default=None, help="Local model path, e.g. /models/model.gguf")
    parser.add_argument("--mode", default=None, choices=["adaptive", "direct", "vote", "judge", "max_accuracy"], help="Solver strategy")
    parser.add_argument("--validate-only", action="store_true", help="Only validate an existing output file")
    parser.add_argument("--self-check", action="store_true", help="Check official submission safety")
    return parser.parse_args(argv)


def merge_config(args: argparse.Namespace) -> RuntimeConfig:
    cfg = from_env()
    return RuntimeConfig(
        input_path=Path(args.input) if args.input else cfg.input_path,
        output_path=Path(args.output) if args.output else cfg.output_path,
        backend=args.backend or cfg.backend,
        model_path=args.model_path or cfg.model_path,
        mode=args.mode or cfg.mode,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        seed=cfg.seed,
        enable_network=cfg.enable_network,
        enable_rag=cfg.enable_rag,
        rag_corpus=cfg.rag_corpus,
        confidence_threshold=cfg.confidence_threshold,
        uncertain_margin=cfg.uncertain_margin,
        max_retries=cfg.max_retries,
        log_every=cfg.log_every,
        time_budget_seconds=cfg.time_budget_seconds,
        force_cpu=cfg.force_cpu,
        n_ctx=cfg.n_ctx,
        n_gpu_layers=cfg.n_gpu_layers,
        batch_size=cfg.batch_size,
        use_batch_inference=cfg.use_batch_inference,
        use_vllm=cfg.use_vllm,
        load_in_awq=cfg.load_in_awq,
        vllm_quantization=cfg.vllm_quantization,
        vllm_gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
        token_fast_exit_margin=cfg.token_fast_exit_margin,
        trace_path=cfg.trace_path,
        submission_strict=cfg.submission_strict,
        use_token_scoring=cfg.use_token_scoring,
        use_pairwise_judge=cfg.use_pairwise_judge,
        use_verifier=cfg.use_verifier,
        max_pairwise_calls=cfg.max_pairwise_calls,
        calibration_path=cfg.calibration_path,
        calibration_strength=cfg.calibration_strength,
        rag_k=cfg.rag_k,
        rag_max_chars=cfg.rag_max_chars,
        load_in_4bit=cfg.load_in_4bit,
        load_in_8bit=cfg.load_in_8bit,
        torch_dtype=cfg.torch_dtype,
        cache_path=cfg.cache_path,
        checkpoint_every=cfg.checkpoint_every,
        auto_integrations=cfg.auto_integrations,
        rag_backend=cfg.rag_backend,
        strict_no_fallback=cfg.strict_no_fallback,
        require_model=cfg.require_model,
        use_permutation_check=cfg.use_permutation_check,
        permutation_checks=cfg.permutation_checks,
        enable_time_controller=cfg.enable_time_controller,
        confidence_calibration=cfg.confidence_calibration,
        health_report_path=cfg.health_report_path,
        require_feature_graph=cfg.require_feature_graph,
        enable_model_probe=cfg.enable_model_probe,
        model_probe_questions=cfg.model_probe_questions,
        heartbeat_path=cfg.heartbeat_path,
        slow_row_seconds=cfg.slow_row_seconds,
        contract_report_path=cfg.contract_report_path,
        use_knowledge_prior=cfg.use_knowledge_prior,
        knowledge_prior_min_confidence=cfg.knowledge_prior_min_confidence,
        use_question_memory=cfg.use_question_memory,
        use_risk_gate=cfg.use_risk_gate,
        use_option_evidence_matrix=cfg.use_option_evidence_matrix,
        option_evidence_k=cfg.option_evidence_k,
        option_evidence_weight=cfg.option_evidence_weight,
        use_decision_arbitrator=cfg.use_decision_arbitrator,
        use_retrieval_cache=cfg.use_retrieval_cache,
        enable_preflight_probe=cfg.enable_preflight_probe,
        preflight_rows=cfg.preflight_rows,
        use_output_watchdog=cfg.use_output_watchdog,
        speed_profile=cfg.speed_profile,
        token_fast_exit=cfg.token_fast_exit,
        prompt_hard_limit_chars=cfg.prompt_hard_limit_chars,
        speed_report_path=cfg.speed_report_path,
        cpu_portable=cfg.cpu_portable,
    )


def main(argv: list[str] | None = None) -> int:
    # Apply one-command official profile before parsing env-backed config.
    autorun_report = apply_official_autorun_defaults()
    print(compact_autorun_line(autorun_report), flush=True)
    args = parse_args(argv)
    cfg = merge_config(args)

    configure_determinism(cfg.seed)
    if autorun_report.enabled:
        ok_auto, auto_errors = validate_autorun_contract()
        if not ok_auto:
            for err in auto_errors:
                print(f"[hackai] autorun contract error: {err}", file=sys.stderr, flush=True)
            return 11
        write_autorun_report(cfg.output_path.with_name("official_autorun.json"), autorun_report)

    if args.self_check:
        assert_no_sensitive_files(".")
        assert_no_browser_automation("src")
        print("OK: official source tree safety checks passed")
        return 0

    if args.validate_only:
        ok, errors = validate_predictions(cfg.output_path)
        if not ok:
            print("\n".join(errors), file=sys.stderr)
            return 2
        print(f"OK: {cfg.output_path}")
        return 0

    if cfg.submission_strict:
        assert_no_browser_automation("src")

    integration_state = prepare_auto_integrations() if cfg.auto_integrations else None
    if integration_state:
        print(compact_status_line(integration_state), flush=True)
    if cfg.strict_no_fallback:
        assert_strict_runtime_ready(cfg)

    start = time.time()
    items = read_items(cfg.input_path)
    cfg, auto_opt_report, dataset_profile = optimize_runtime_config(
        cfg,
        items,
        profile_path=cfg.output_path.with_name("dataset_profile.json"),
    )
    write_auto_optimize_report(auto_opt_report, cfg.output_path.with_name("runtime_auto_optimizer.json"))
    if auto_opt_report.enabled:
        print(f"[hackai] auto_optimizer=on profile={cfg.speed_profile} rows={dataset_profile.rows} risk={dataset_profile.high_risk_rows}/{dataset_profile.rows} changes={len(auto_opt_report.changed)}", flush=True)
        for note in auto_opt_report.notes[:4]:
            print(f"[hackai] optimizer note: {note}", flush=True)

    cfg, wiring_report = apply_wiring_integrity_guard(cfg, items)
    write_wiring_report(wiring_report, cfg.output_path.with_name("wiring_integrity.json"))
    print(compact_wiring_line(wiring_report), flush=True)
    for warn in wiring_report.warnings[:6]:
        print(f"[hackai] wiring warning: {warn}", flush=True)
    if not wiring_report.ok and cfg.require_feature_graph:
        for err in wiring_report.errors:
            print(f"[hackai] wiring error: {err}", file=sys.stderr, flush=True)
        return 13
    contract_report = ContractReport(ok=True, timestamp=time.time(), runtime=runtime_fingerprint(cfg.model_path))
    for check in validate_input_contract(items):
        contract_report.add(check.name, check.ok, check.required, check.detail)
    write_contract_report(contract_report, cfg.contract_report_path)
    if cfg.require_feature_graph and not contract_report.ok:
        for err in contract_report.errors:
            print(f"[hackai] contract error: {err}", file=sys.stderr, flush=True)
        return 6
    stability = build_stability_report(cfg, items, integration_state)
    write_stability_report(stability, cfg.health_report_path)
    print(compact_report_line(stability), flush=True)
    for warn in stability.warnings[:8]:
        print(f"[hackai] stability warning: {warn}", flush=True)
    if cfg.require_feature_graph and not stability.ok:
        for err in stability.errors:
            print(f"[hackai] stability error: {err}", file=sys.stderr, flush=True)
        return 5
    print(f"[hackai] input={cfg.input_path} rows={len(items)} output={cfg.output_path}", flush=True)
    print(f"[hackai] backend={cfg.backend} mode={cfg.mode} model_path={cfg.model_path or 'none'} vram={detect_vram_gb() or 'unknown'}", flush=True)

    with optional_network_block(cfg.enable_network):
        backend = create_backend(
            cfg.backend,
            cfg.model_path,
            cfg.max_new_tokens,
            cfg.temperature,
            cfg.top_p,
            n_ctx=cfg.n_ctx,
            n_gpu_layers=cfg.n_gpu_layers,
            force_cpu=cfg.force_cpu,
            load_in_4bit=cfg.load_in_4bit,
            load_in_8bit=cfg.load_in_8bit,
            load_in_awq=cfg.load_in_awq,
            torch_dtype=cfg.torch_dtype,
            batch_size=cfg.batch_size,
            use_vllm=cfg.use_vllm,
            vllm_quantization=cfg.vllm_quantization,
            vllm_gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
        )
        if cfg.enable_model_probe:
            probe = official_model_probe(backend, max_questions=cfg.model_probe_questions)
            # Merge probe checks into the same official contract report.
            for c in probe.checks:
                contract_report.add(c.name, c.ok, c.required, c.detail)
            contract_report.runtime.update(runtime_fingerprint(cfg.model_path))
            write_contract_report(contract_report, cfg.contract_report_path)
            if not probe.ok:
                for err in probe.errors:
                    print(f"[hackai] model probe error: {err}", file=sys.stderr, flush=True)
                return 7
            print("[hackai] model probe passed", flush=True)
        rag = None
        if cfg.enable_rag:
            if os.getenv("USE_KNOWLEDGE_ENGINE", "1").strip().lower() in {"1", "true", "yes", "on", "auto"}:
                rag = create_knowledge_engine_from_env()
                if rag is None and cfg.strict_no_fallback:
                    raise RuntimeError("ENABLE_RAG=1 but KnowledgeEngine could not load any offline corpus")
            else:
                rag = create_rag(cfg.rag_corpus, mode="auto")
            if rag and cfg.use_retrieval_cache:
                rag = CachedRAG(rag)
            if rag:
                print(f"[hackai] Knowledge/RAG enabled backend={getattr(rag, 'name', type(rag).__name__)} docs={len(getattr(rag, 'docs', []))}", flush=True)
        calibrator = AnswerPriorCalibrator.from_path(cfg.calibration_path, cfg.calibration_strength) if cfg.calibration_path else None
        if calibrator:
            print(f"[hackai] calibration enabled priors={calibrator.priors}", flush=True)
        cache = PromptCache(cfg.cache_path)
        qmem = QuestionMemory(enabled=cfg.use_question_memory) if cfg.use_question_memory else None
        solver = AdaptiveSolver(backend=backend, config=cfg, rag=rag, calibrator=calibrator, cache=cache, question_memory=qmem, start_time=start, total_rows=len(items))
        if cfg.enable_preflight_probe:
            preflight = run_preflight_probe(items, solver.solve, max_rows=cfg.preflight_rows, slow_row_seconds=cfg.slow_row_seconds)
            print(f"[hackai] preflight rows={preflight.sampled_rows} seconds={preflight.seconds:.2f} median={preflight.median_row_seconds:.2f}s warnings={len(preflight.warnings)} errors={len(preflight.errors)}", flush=True)
            for warn in preflight.warnings[:5]:
                print(f"[hackai] preflight warning: {warn}", flush=True)
            if not preflight.ok:
                for err in preflight.errors:
                    print(f"[hackai] preflight error: {err}", file=sys.stderr, flush=True)
                return 9

        results = []
        stats = RunStats(rows=len(items), backend=getattr(backend, "name", cfg.backend), mode=cfg.mode)
        speed_report = SpeedReport(profile=getattr(cfg, "speed_profile", "balanced"))
        supervisor = RuntimeSupervisor(items, heartbeat_path=cfg.heartbeat_path, slow_row_seconds=cfg.slow_row_seconds, heartbeat_every=cfg.log_every)
        invariant_guard = RuntimeInvariantGuard(items, strict=cfg.strict_no_fallback, report_path=cfg.output_path.with_name("runtime_invariants.json"))
        resource_guard = ResourceGuard(cfg.output_path.parent, report_path=cfg.output_path.with_name("resource_guard.json"), sample_every=max(1, cfg.log_every or 100))
        ledger = ResultLedger(cfg.output_path.with_name("result_ledger.jsonl"), enabled=cfg.submission_strict)
        watchdog = OutputWatchdog() if cfg.use_output_watchdog else None
        resource_guard.snapshot(0, force=True)

        def _record_result(idx: int, item, result, row_seconds: float) -> None:
            results.append(result)
            supervisor.end_row(result)
            invariant_guard.check_result(item, result, idx)
            ledger.append(idx, item, result, row_seconds)
            if watchdog:
                watchdog.observe(result, idx)
            resource_guard.snapshot(idx)
            strategy = result.strategy or ""
            if strategy == "direct":
                stats.direct_count += 1
            elif strategy == "vote":
                stats.vote_count += 1
            elif strategy == "judge":
                stats.judge_count += 1
            if "fallback" in strategy:
                stats.fallback_count += 1
            if strategy in {"token_fast_exit", "batch_token_fast_exit"}:
                speed_report.fast_token_exits += 1
            if strategy == "memory_reuse":
                speed_report.memory_reuse_rows += 1
            if strategy in {"direct", "token_score", "token_fast_exit", "batch_token_fast_exit", "batch_direct", "memory_reuse"}:
                speed_report.direct_rows += 1
            else:
                speed_report.deep_rows += 1

        use_batch = bool(getattr(cfg, "use_batch_inference", True)) and cfg.batch_size > 1 and batch_backend_available(backend)
        if use_batch:
            # V8 one-command speed path: the judge still runs the same container,
            # but supported backends process easy rows in batches before deep solve.
            try:
                batch_size = cfg.batch_size or compute_optimal_batch_size(detect_vram_gb(), n_ctx=cfg.n_ctx)
                batch_results = run_batch_solve_loop(
                    solver,
                    items,
                    batch_size=batch_size,
                    log_every=cfg.log_every,
                    strict=cfg.strict_no_fallback,
                )
                if len(batch_results) != len(items):
                    raise RuntimeError(f"batch loop returned {len(batch_results)} results for {len(items)} items")
                for idx, (item, result) in enumerate(zip(items, batch_results), start=1):
                    row_start = time.time()
                    supervisor.begin_row(item)
                    _record_result(idx, item, result, time.time() - row_start)
                    if cfg.log_every and (idx % cfg.log_every == 0 or idx == len(items)):
                        print(f"[hackai] processed={idx}/{len(items)} elapsed={time.time()-start:.1f}s batch=1", flush=True)
                    if cfg.checkpoint_every and idx % cfg.checkpoint_every == 0:
                        ckpt = cfg.output_path.with_suffix(cfg.output_path.suffix + ".partial")
                        write_predictions(results, ckpt)
            except Exception as e:
                stats.errors += 1
                print(f"[hackai] strict batch failure: {e}", file=sys.stderr, flush=True)
                if cfg.strict_no_fallback:
                    return 4
                # Non-strict local debugging path only. Official profile keeps strict enabled.
                for idx, item in enumerate(items[len(results):], start=len(results)+1):
                    row_start = time.time()
                    supervisor.begin_row(item)
                    try:
                        result = solver.solve(item, index=idx)
                    except Exception as row_e:
                        from .answer_parser import deterministic_fallback
                        from .schema import SolverResult
                        ans = deterministic_fallback(item.qid, item.question)
                        result = SolverResult(item.qid, ans, 0.0, "fatal_fallback", {}, {ans: 0.0}, str(row_e))
                    _record_result(idx, item, result, time.time() - row_start)
        else:
            for idx, item in enumerate(items, start=1):
                row_start = time.time()
                supervisor.begin_row(item)
                try:
                    result = solver.solve(item, index=idx)
                    _record_result(idx, item, result, time.time() - row_start)
                except Exception as e:
                    stats.errors += 1
                    if cfg.strict_no_fallback:
                        print(f"[hackai] strict row failure qid={item.qid}: {e}", file=sys.stderr, flush=True)
                        return 4
                    from .answer_parser import deterministic_fallback
                    from .schema import SolverResult
                    ans = deterministic_fallback(item.qid, item.question)
                    fb_result = SolverResult(item.qid, ans, 0.0, "fatal_fallback", {}, {ans: 0.0}, str(e))
                    _record_result(idx, item, fb_result, time.time() - row_start)

                if cfg.log_every and (idx % cfg.log_every == 0 or idx == len(items)):
                    print(f"[hackai] processed={idx}/{len(items)} elapsed={time.time()-start:.1f}s", flush=True)
                if cfg.checkpoint_every and idx % cfg.checkpoint_every == 0:
                    ckpt = cfg.output_path.with_suffix(cfg.output_path.suffix + ".partial")
                    write_predictions(results, ckpt)

    inv_report = invariant_guard.write_report()
    res_report = resource_guard.write_report()
    for check in validate_output_contract(results, items):
        contract_report.add(check.name, check.ok, check.required, check.detail)
    for warning in supervisor.final_checks(results):
        contract_report.add("runtime_supervisor_warning", False, False, warning)
    for warning in inv_report.warnings[:12]:
        contract_report.add("runtime_invariant_warning", False, False, warning)
    for error in inv_report.errors:
        contract_report.add("runtime_invariant_error", False, True, error)
    for warning in res_report.warnings[:12]:
        contract_report.add("resource_guard_warning", False, False, warning)
    for error in res_report.errors:
        contract_report.add("resource_guard_error", False, True, error)
    if 'watchdog' in locals() and watchdog:
        for warning in watchdog.warnings[:12]:
            contract_report.add("output_watchdog_warning", False, False, warning)
    write_predictions(results, cfg.output_path)
    for check in validate_pred_file_contract(cfg.output_path, items):
        contract_report.add(check.name, check.ok, check.required, check.detail)
    write_contract_report(contract_report, cfg.contract_report_path)
    if not contract_report.ok:
        for err in contract_report.errors:
            print(f"[hackai] final contract error: {err}", file=sys.stderr, flush=True)
        return 8
    if cfg.trace_path:
        write_trace(results, cfg.trace_path)
    ok, errors = validate_predictions(cfg.output_path, expected_count=len(items))
    if not ok:
        print("[hackai] output validation failed:", file=sys.stderr)
        print("\n".join(errors), file=sys.stderr)
        return 3
    stats.seconds = time.time() - start
    if 'speed_report' in locals():
        speed_report.rows = len(items)
        speed_report.seconds = stats.seconds
        if 'rag' in locals() and hasattr(rag, "stats"):
            try:
                rstats = rag.stats()
                speed_report.retrieval_cache_hits = int(rstats.get("hits", 0))
                speed_report.retrieval_cache_misses = int(rstats.get("misses", 0))
            except Exception:
                pass
        write_speed_report(getattr(cfg, "speed_report_path", None), speed_report)
    print(f"[hackai] done rows={len(items)} seconds={stats.seconds:.1f} backend={stats.backend} direct={stats.direct_count} vote={stats.vote_count} judge={stats.judge_count} errors={stats.errors}", flush=True)
    return 0


if __name__ == "__main__":
    # Some optional vendored research libraries may leave non-daemon helper
    # threads alive after the official CSV->pred.csv job is finished. Force a
    # clean process exit so Docker/BTC judges never hang after writing output.
    import json
    import traceback
    import os
    try:
        code = main()
    except Exception as exc:
        # Graceful hard-fail: no silent fallback, but also no uncontrolled stack-only crash.
        try:
            out_dir = Path(os.getenv("OUTPUT_PATH", "/output/pred.csv")).parent
            out_dir.mkdir(parents=True, exist_ok=True)
            failure = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "hint": "Official strict runtime failed before producing pred.csv. Check model mount, dependency install, and official_autorun.json.",
            }
            (out_dir / "runtime_failure.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        print(f"[hackai] fatal official runtime error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        if os.getenv("SHOW_TRACEBACK", "0").strip().lower() in {"1", "true", "yes", "on"}:
            traceback.print_exc()
        code = 12
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
