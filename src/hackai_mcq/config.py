from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class RuntimeConfig:
    input_path: Path
    output_path: Path
    backend: str = "auto"               # auto | heuristic | llama_cpp | transformers
    model_path: str | None = None
    mode: str = "adaptive"              # direct | vote | judge | max_accuracy | adaptive
    max_new_tokens: int = 12
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42
    enable_network: bool = False
    enable_rag: bool = True
    rag_corpus: str | None = None
    rag_k: int = 3
    rag_max_chars: int = 1800
    confidence_threshold: float = 0.62
    uncertain_margin: float = 0.08
    max_retries: int = 1
    log_every: int = 50
    time_budget_seconds: float = 0.0
    force_cpu: bool = False
    n_ctx: int = 2048
    n_gpu_layers: int = -1
    batch_size: int = 1
    use_batch_inference: bool = True
    use_vllm: bool = False
    load_in_awq: bool = False
    vllm_quantization: str = "awq"
    vllm_gpu_memory_utilization: float = 0.85
    token_fast_exit_margin: float = 0.35
    trace_path: Path | None = None
    submission_strict: bool = True
    # High-score knobs
    use_token_scoring: bool = True
    use_pairwise_judge: bool = True
    use_verifier: bool = True
    max_pairwise_calls: int = 3
    calibration_path: str | None = None
    calibration_strength: float = 0.04
    # Transformers loading knobs
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    torch_dtype: str = "auto"
    cache_path: str | None = None
    checkpoint_every: int = 0
    auto_integrations: bool = True
    rag_backend: str = "auto"
    strict_no_fallback: bool = True
    require_model: bool = True
    use_permutation_check: bool = True
    permutation_checks: int = 2
    enable_time_controller: bool = True
    confidence_calibration: bool = True
    health_report_path: Path | None = None
    require_feature_graph: bool = True
    enable_model_probe: bool = True
    model_probe_questions: int = 3
    heartbeat_path: Path | None = None
    slow_row_seconds: float = 20.0
    contract_report_path: Path | None = None
    use_knowledge_prior: bool = True
    knowledge_prior_min_confidence: float = 0.34
    use_question_memory: bool = True
    use_risk_gate: bool = True
    use_option_evidence_matrix: bool = True
    option_evidence_k: int = 2
    option_evidence_weight: float = 0.22
    use_decision_arbitrator: bool = True
    use_retrieval_cache: bool = True
    enable_preflight_probe: bool = False
    preflight_rows: int = 3
    use_output_watchdog: bool = True
    speed_profile: str = "balanced"  # turbo | fast | balanced | accuracy
    token_fast_exit: bool = True
    prompt_hard_limit_chars: int = 4500
    speed_report_path: Path | None = None
    cpu_portable: bool = False


def find_input_path(data_dir: str | Path = "/data") -> Path:
    d = Path(data_dir)
    candidates = [d / "private_test.csv", d / "public_test.csv"]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    csvs = sorted(d.glob("*.csv")) if d.exists() else []
    if csvs:
        return csvs[0]
    raise FileNotFoundError(f"No CSV found in {d}. Expected private_test.csv or public_test.csv")


def auto_model_path(models_dir: str | Path = "/models") -> str | None:
    d = Path(models_dir)
    if not d.exists():
        return None
    # GGUF first: best fit for 6GB VRAM and Docker reproducibility.
    for ext in [".gguf", ".safetensors", ".bin"]:
        files = sorted(d.rglob(f"*{ext}"))
        if files:
            if ext in {".safetensors", ".bin"}:
                # Prefer model folder when HF-style files are found.
                for parent in [files[0].parent, *files[0].parents]:
                    if (parent / "config.json").exists():
                        return str(parent)
            return str(files[0])
    configs = sorted(d.rglob("config.json"))
    if configs:
        return str(configs[0].parent)
    return None


def from_env() -> RuntimeConfig:
    # Official Docker run is one-command: feature flags are applied automatically.
    try:
        from .official_autorun import apply_official_autorun_defaults
        apply_official_autorun_defaults()
    except Exception:
        # Do not hide broken autorun logic in strict official mode.
        if _bool_env("STRICT_NO_FALLBACK", True):
            raise

    # Make official Docker run self-contained: vendored repos/corpus are auto-discovered.
    # Local validation can disable this to avoid importing heavy optional repos.
    # In strict official mode, integration errors are not swallowed.
    if _bool_env("AUTO_INTEGRATIONS", True):
        try:
            from .auto_integrations import prepare_auto_integrations
            prepare_auto_integrations()
        except Exception:
            if _bool_env("STRICT_NO_FALLBACK", True):
                raise
    explicit_input = os.getenv("INPUT_PATH")
    if explicit_input:
        input_path = Path(explicit_input)
    else:
        try:
            input_path = find_input_path(os.getenv("DATA_DIR", "/data"))
        except FileNotFoundError:
            input_path = Path(os.getenv("DATA_DIR", "/data")) / "private_test.csv"

    model_path = os.getenv("MODEL_PATH") or auto_model_path(os.getenv("MODELS_DIR", "/models"))
    trace_raw = os.getenv("TRACE_PATH", "")
    trace_path = Path(trace_raw) if trace_raw else None

    return RuntimeConfig(
        input_path=input_path,
        output_path=Path(os.getenv("OUTPUT_PATH", "/output/pred.csv")),
        backend=os.getenv("LLM_BACKEND", "auto"),
        model_path=model_path,
        mode=os.getenv("SOLVER_MODE", "adaptive"),
        max_new_tokens=_int_env("MAX_NEW_TOKENS", 12),
        temperature=_float_env("TEMPERATURE", 0.0),
        top_p=_float_env("TOP_P", 1.0),
        seed=_int_env("SEED", 42),
        enable_network=_bool_env("ENABLE_NETWORK", False),
        enable_rag=_bool_env("ENABLE_RAG", True),
        rag_corpus=os.getenv("RAG_CORPUS"),
        rag_k=_int_env("RAG_K", 3),
        rag_max_chars=_int_env("RAG_MAX_CHARS", 1800),
        confidence_threshold=_float_env("CONFIDENCE_THRESHOLD", 0.62),
        uncertain_margin=_float_env("UNCERTAIN_MARGIN", 0.08),
        max_retries=_int_env("MAX_RETRIES", 1),
        log_every=_int_env("LOG_EVERY", 50),
        time_budget_seconds=_float_env("TIME_BUDGET_SECONDS", 0.0),
        force_cpu=_bool_env("FORCE_CPU", False),
        n_ctx=_int_env("N_CTX", 2048),
        n_gpu_layers=_int_env("N_GPU_LAYERS", -1),
        batch_size=_int_env("BATCH_SIZE", 8),
        use_batch_inference=_bool_env("USE_BATCH_INFERENCE", True),
        use_vllm=_bool_env("USE_VLLM", False),
        load_in_awq=_bool_env("LOAD_IN_AWQ", False),
        vllm_quantization=os.getenv("VLLM_QUANTIZATION", "awq"),
        vllm_gpu_memory_utilization=_float_env("VLLM_GPU_MEMORY_UTILIZATION", 0.85),
        token_fast_exit_margin=_float_env("TOKEN_FAST_EXIT_MARGIN", 0.35),
        trace_path=trace_path,
        submission_strict=_bool_env("SUBMISSION_STRICT", True),
        use_token_scoring=_bool_env("USE_TOKEN_SCORING", True),
        use_pairwise_judge=_bool_env("USE_PAIRWISE_JUDGE", True),
        use_verifier=_bool_env("USE_VERIFIER", True),
        max_pairwise_calls=_int_env("MAX_PAIRWISE_CALLS", 3),
        calibration_path=os.getenv("CALIBRATION_PATH"),
        calibration_strength=_float_env("CALIBRATION_STRENGTH", 0.04),
        load_in_4bit=_bool_env("LOAD_IN_4BIT", False),
        load_in_8bit=_bool_env("LOAD_IN_8BIT", False),
        torch_dtype=os.getenv("TORCH_DTYPE", "auto"),
        cache_path=os.getenv("CACHE_PATH"),
        checkpoint_every=_int_env("CHECKPOINT_EVERY", 0),
        auto_integrations=_bool_env("AUTO_INTEGRATIONS", True),
        rag_backend=os.getenv("RAG_BACKEND", "bm25s"),
        strict_no_fallback=_bool_env("STRICT_NO_FALLBACK", True),
        require_model=_bool_env("REQUIRE_MODEL", True),
        use_permutation_check=_bool_env("USE_PERMUTATION_CHECK", True),
        permutation_checks=_int_env("PERMUTATION_CHECKS", 2),
        enable_time_controller=_bool_env("ENABLE_TIME_CONTROLLER", True),
        confidence_calibration=_bool_env("CONFIDENCE_CALIBRATION", True),
        health_report_path=Path(os.getenv("HEALTH_REPORT_PATH", "/output/runtime_health.json")) if _bool_env("WRITE_HEALTH_REPORT", True) else None,
        require_feature_graph=_bool_env("REQUIRE_FEATURE_GRAPH", True),
        enable_model_probe=_bool_env("ENABLE_MODEL_PROBE", True),
        model_probe_questions=_int_env("MODEL_PROBE_QUESTIONS", 3),
        heartbeat_path=Path(os.getenv("HEARTBEAT_PATH", "/output/runtime_heartbeat.json")) if _bool_env("WRITE_HEARTBEAT", True) else None,
        slow_row_seconds=_float_env("SLOW_ROW_SECONDS", 20.0),
        contract_report_path=Path(os.getenv("CONTRACT_REPORT_PATH", "/output/official_contract.json")) if _bool_env("WRITE_CONTRACT_REPORT", True) else None,
        use_knowledge_prior=_bool_env("USE_KNOWLEDGE_PRIOR", True),
        knowledge_prior_min_confidence=_float_env("KNOWLEDGE_PRIOR_MIN_CONFIDENCE", 0.34),
        use_question_memory=_bool_env("USE_QUESTION_MEMORY", True),
        use_risk_gate=_bool_env("USE_RISK_GATE", True),
        use_option_evidence_matrix=_bool_env("USE_OPTION_EVIDENCE_MATRIX", True),
        option_evidence_k=_int_env("OPTION_EVIDENCE_K", 2),
        option_evidence_weight=_float_env("OPTION_EVIDENCE_WEIGHT", 0.22),
        use_decision_arbitrator=_bool_env("USE_DECISION_ARBITRATOR", True),
        use_retrieval_cache=_bool_env("USE_RETRIEVAL_CACHE", True),
        enable_preflight_probe=_bool_env("ENABLE_PREFLIGHT_PROBE", False),
        preflight_rows=_int_env("PREFLIGHT_ROWS", 3),
        use_output_watchdog=_bool_env("USE_OUTPUT_WATCHDOG", True),
        speed_profile=os.getenv("SPEED_PROFILE", "balanced"),
        token_fast_exit=_bool_env("TOKEN_FAST_EXIT", True),
        prompt_hard_limit_chars=_int_env("PROMPT_HARD_LIMIT_CHARS", 4500),
        speed_report_path=Path(os.getenv("SPEED_REPORT_PATH", "/output/speed_report.json")) if _bool_env("WRITE_SPEED_REPORT", True) else None,
        cpu_portable=_bool_env("CPU_PORTABLE", False),
    )
