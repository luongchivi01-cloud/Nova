import os
from pathlib import Path

from hackai_mcq.official_autorun import apply_official_autorun_defaults, validate_autorun_contract
from hackai_mcq.config import RuntimeConfig
from hackai_mcq.stability_orchestrator import build_stability_report
from hackai_mcq.schema import MCQItem


def test_official_autorun_turns_on_core_flags(monkeypatch):
    monkeypatch.setenv("OFFICIAL_AUTORUN", "1")
    for key in [
        "AUTO_INTEGRATIONS", "ENABLE_RAG", "USE_KNOWLEDGE_ENGINE", "USE_TOKEN_SCORING",
        "USE_PAIRWISE_JUDGE", "USE_VERIFIER", "USE_PERMUTATION_CHECK", "USE_QUESTION_MEMORY",
        "USE_RISK_GATE", "USE_OPTION_EVIDENCE_MATRIX", "USE_DECISION_ARBITRATOR",
        "USE_RETRIEVAL_CACHE", "USE_OUTPUT_WATCHDOG", "ENABLE_NETWORK", "ALLOW_HEURISTIC",
        "STRICT_NO_FALLBACK", "REQUIRE_MODEL", "SUBMISSION_STRICT",
    ]:
        monkeypatch.delenv(key, raising=False)
    report = apply_official_autorun_defaults()
    assert report.enabled
    assert os.environ["AUTO_INTEGRATIONS"] == "1"
    assert os.environ["CPU_PORTABLE"] == "1"
    assert os.environ["ENABLE_RAG"] == "0"
    assert os.environ["USE_PAIRWISE_JUDGE"] == "0"
    assert os.environ["USE_PERMUTATION_CHECK"] == "0"
    assert os.environ["ENABLE_NETWORK"] == "0"
    assert os.environ["ALLOW_HEURISTIC"] == "0"
    ok, errors = validate_autorun_contract()
    assert ok, errors


def test_stability_report_includes_official_autorun_guard(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFICIAL_AUTORUN", "1")
    cfg = RuntimeConfig(
        input_path=tmp_path / "public_test.csv",
        output_path=tmp_path / "pred.csv",
        backend="auto",
        model_path=str(tmp_path / "model.gguf"),
        require_model=False,
    )
    item = MCQItem("1", "Question?", {"A": "one", "B": "two", "C": "three", "D": "four"})
    report = build_stability_report(cfg, [item])
    names = {f.name for f in report.features}
    assert "official_autorun_guard" in names
