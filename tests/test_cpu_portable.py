from pathlib import Path

from hackai_mcq.config import RuntimeConfig
from hackai_mcq.schema import MCQItem
from hackai_mcq.solver import AdaptiveSolver


class CountingBackend:
    name = "counting"

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0

    def generate(self, prompt, item):
        answer = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        return answer


def _cfg(tmp_path):
    return RuntimeConfig(
        input_path=Path("in.csv"),
        output_path=tmp_path / "pred.csv",
        cpu_portable=True,
        enable_rag=False,
        strict_no_fallback=True,
        require_model=False,
    )


def test_cpu_easy_row_uses_one_model_call(tmp_path):
    backend = CountingBackend(["B"])
    item = MCQItem("1", "Capital of Vietnam?", {"A": "Hue", "B": "Hanoi", "C": "Da Nang", "D": "Can Tho"})
    result = AdaptiveSolver(backend, _cfg(tmp_path)).solve(item)
    assert result.answer == "B"
    assert result.strategy == "cpu_direct"
    assert backend.calls == 1


def test_cpu_high_risk_row_uses_at_most_one_verifier(tmp_path):
    backend = CountingBackend(["A", "C"])
    item = MCQItem("2", "Which answer is NOT correct?", {"A": "one", "B": "two", "C": "three", "D": "four"})
    result = AdaptiveSolver(backend, _cfg(tmp_path)).solve(item)
    assert result.answer == "C"
    assert result.strategy == "cpu_verified"
    assert backend.calls == 2
