from __future__ import annotations

"""Generate adversarial 2000-row local benchmark probes.

This is not a private-test substitute. It is designed to break weak pipelines:
negation, option-order bias, multilingual text, near-duplicate choices, and noisy
Unicode. Labels are deterministic so it can be used for regression tests.
"""

import csv
from pathlib import Path

TEMPLATES = [
    ("vi", "Câu nào sau đây KHÔNG phải là đặc điểm của hợp đồng mua bán hàng hóa?", ["Có chủ thể", "Có đối tượng hàng hóa", "Luôn vô hiệu", "Có thỏa thuận"], "C"),
    ("en", "Which option is NOT a renewable energy source?", ["Solar", "Wind", "Coal", "Hydro"], "C"),
    ("zh", "以下哪一项不是哺乳动物？", ["猫", "狗", "鲨鱼", "牛"], "C"),
    ("ja", "次のうち、最も水に関係するものはどれですか。", ["砂漠", "川", "火", "石"], "B"),
    ("ko", "다음 중 과일이 아닌 것은?", ["사과", "바나나", "자동차", "포도"], "C"),
    ("ru", "Что НЕ является планетой?", ["Марс", "Венера", "Солнце", "Юпитер"], "C"),
    ("ar", "أي مما يلي ليس حيوانًا؟", ["قطة", "كلب", "كرسي", "حصان"], "C"),
    ("th", "ข้อใดไม่ใช่สี", ["แดง", "น้ำเงิน", "โต๊ะ", "เขียว"], "C"),
    ("logic", "If all A are B and all B are C, which statement must be true?", ["All C are A", "All A are C", "No A are C", "Some C are not B"], "B"),
    ("numeric", "A student scores 8, 10, and 12. What is the average?", ["8", "9", "10", "12"], "C"),
]


def _rotate(options: list[str], label: str, shift: int) -> tuple[list[str], str]:
    labels = "ABCD"
    idx = labels.index(label)
    n = len(options)
    shift %= n
    new = options[shift:] + options[:shift]
    new_idx = (idx - shift) % n
    return new, labels[new_idx]


def create_adversarial_csv(path: str | Path, rows: int = 2000) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["qid", "question", "A", "B", "C", "D", "label", "case_type"])
        w.writeheader()
        for i in range(1, rows + 1):
            typ, q, opts, label = TEMPLATES[(i - 1) % len(TEMPLATES)]
            # Position-bias probe: rotate answers but keep content label correct.
            opts2, label2 = _rotate(list(opts), label, shift=(i // len(TEMPLATES)) % 4)
            noise = "" if i % 7 else "  ※ answer carefully; options may be reordered."
            w.writerow({"qid": str(i), "question": q + noise, "A": opts2[0], "B": opts2[1], "C": opts2[2], "D": opts2[3], "label": label2, "case_type": typ})
    return p


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/adversarial_2000.csv")
    p.add_argument("--rows", type=int, default=2000)
    args = p.parse_args(argv)
    out = create_adversarial_csv(args.out, args.rows)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
