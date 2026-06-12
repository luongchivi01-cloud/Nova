import importlib.util
import json
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "round1_v2_runner.py"
SPEC = importlib.util.spec_from_file_location("round1_v2_runner", SCRIPT)
v2 = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = v2
SPEC.loader.exec_module(v2)


class Round1V2Tests(unittest.TestCase):
    def test_parser_only_accepts_final_line(self):
        self.assertEqual(v2.extract_final("Evidence: Option A is mentioned.\nFinal: C"), "C")
        self.assertIsNone(v2.extract_final("Evidence: A and B are possible."))
        self.assertIsNone(v2.extract_final("Final answer is B"))

    def test_long_context_keeps_question_relevant_paragraph_and_neighbor(self):
        raw = (
            "Đoạn đầu không liên quan.\n\n"
            "Đoạn trước nói về Paris.\n\n"
            "Paris là thủ đô của Pháp.\n\n"
            "Đoạn sau nói về châu Âu.\n"
            "Câu hỏi: Thủ đô của Pháp là gì?"
        )
        question, context = v2.split_question_and_context(raw)
        selected, meta = v2.select_relevant_context(context, question, ["Berlin", "Paris", "Rome", "Madrid"])
        self.assertEqual(question, "Thủ đô của Pháp là gì?")
        self.assertIn("Paris là thủ đô của Pháp", selected)
        self.assertGreaterEqual(meta["selected"], 1)

    def test_very_long_paragraph_is_split_before_relevance_selection(self):
        filler = "Thông tin không liên quan. " * 80
        context = filler + "Paris là thủ đô của Pháp. " + filler
        selected, meta = v2.select_relevant_context(context, "Thủ đô của Pháp là gì?", ["Berlin", "Paris", "Rome", "Madrid"])
        self.assertIn("Paris là thủ đô của Pháp", selected)
        self.assertGreater(meta["paragraphs"], 1)

    def test_midpoint_elasticity_exact_unique_match(self):
        result = v2.deterministic_solve(
            "Tại mức giá 5, lượng cầu là 150; tại mức giá 3, lượng cầu là 250. Độ co giãn của cầu theo giá giữa hai điểm là bao nhiêu?",
            ["0.5", "1.0", "2.0", "2.5"],
        )
        self.assertEqual(result.answer if result else None, "B")

    def test_cylinder_rate_exact_unique_match(self):
        result = v2.deterministic_solve(
            "Một bể hình trụ được đổ với tốc độ là 50 cm3/s. Bán kính của bể là 5 cm. Tốc độ tăng chiều cao là bao nhiêu?",
            ["0.2 cm/s", "0.4 cm/s", "0.6 cm/s", "0.8 cm/s"],
        )
        self.assertEqual(result.answer if result else None, "C")

    def test_exponential_decay_symbolic_match(self):
        result = v2.deterministic_solve(
            "Cho dB/dt = -k B và B(0)=B_0. Tìm B(t).",
            ["B(t)=B_0 e^{-kt}", "B(t)=B_0 e^{kt}", "B(t)=B_0(1-kt)", "B(t)=B_0/(1+kt)"],
        )
        self.assertEqual(result.answer if result else None, "A")

    def test_hess_direct_sum(self):
        result = v2.deterministic_solve(
            "Theo định luật Hess, delta H_1 = -80 kJ/mol và delta H_2 = -30 kJ/mol. Tính delta H_3 bằng tổng hai phản ứng.",
            ["-110", "-80", "-30", "0"],
        )
        self.assertEqual(result.answer if result else None, "A")

    def test_ambiguous_numeric_match_routes_to_model(self):
        self.assertIsNone(v2.match_numeric_option(["1.0", "1", "2", "3"], 1.0))

    def test_verifier_prompt_does_not_contain_primary_answer(self):
        prompt = v2.make_prompt("2+2=?", ["3", "4", "5", "6"], "", "", verifier=True)
        self.assertNotIn("proposed answer", prompt.lower())
        self.assertNotIn("primary answer", prompt.lower())

    def test_checkpoint_rejects_wrong_config_hash(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.jsonl"
            checkpoint.write_text(json.dumps({"qid": "1", "answer": "A", "model_calls": 1, "config_hash": "old"}) + "\n")
            with self.assertRaisesRegex(ValueError, "configuration hash"):
                v2.load_checkpoint(checkpoint, ["1"], "new")

    def test_checkpoint_rejects_more_than_two_model_calls(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.jsonl"
            checkpoint.write_text(json.dumps({"qid": "1", "answer": "A", "model_calls": 3, "config_hash": "same"}) + "\n")
            with self.assertRaisesRegex(ValueError, "call count"):
                v2.load_checkpoint(checkpoint, ["1"], "same")

    def test_eta_projection_uses_recorded_row_seconds(self):
        rows = [{"seconds": 10.0}, {"seconds": 20.0}, {"seconds": 30.0}]
        self.assertEqual(v2.projected_total_seconds(rows, 6), 120.0)


if __name__ == "__main__":
    unittest.main()
