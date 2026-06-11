from __future__ import annotations

"""Self-loading offline corpus for the official Docker path.

The contest runner should not need to know how to mount or enable a corpus.
This module guarantees that the KnowledgeEngine has a bundled, deterministic
base corpus even when the user has not mounted /knowledge. Stronger corpora can
still be mounted at /knowledge or /data/knowledge and will be merged on top.

No live web/API calls are made here.
"""

import os
from pathlib import Path
from textwrap import dedent

AUTO_SEED_DIR_ENV = "AUTO_SEED_KNOWLEDGE_DIR"

SEED_FILES: dict[str, str] = {
    "00_exam_reasoning_playbook.md": r"""
# Multiple-choice reasoning playbook

When a question asks for the best answer, compare each option against the exact wording of the question rather than choosing an option that is merely true.
When a question contains not, except, false, incorrect, không, sai, ngoại trừ, 不, ない, 제외, لا, ไม่, or не, the target is inverted: choose the false or excluded option.
When two choices are both partly true, prefer the more specific answer only if it fully satisfies the question. Prefer the general principle only when the question asks for a principle.
Definitions usually require necessary and sufficient features. Examples are not definitions unless the question asks for an example.
Cause questions ask why something happens; effect questions ask what happens afterward; method questions ask how to do it.
Questions with dates, laws, or current status should be treated cautiously. Use bundled legal/current corpus only when the source is explicitly provided or stable.

# English MCQ cues
not, except, least likely, false, incorrect, does not, cannot, never, only if, most likely, best describes, primary reason, consequence, advantage, disadvantage.

# Vietnamese MCQ cues
không đúng, sai, ngoại trừ, không phải, chưa chính xác, đúng nhất, phù hợp nhất, nguyên nhân, hệ quả, đặc điểm, bản chất, khái niệm, ví dụ.

# Multilingual negation cues
Chinese: 不, 不是, 除了, 错误. Japanese: ない, ではない, 除く, 誤り. Korean: 아니다, 제외, 틀린. Arabic: لا, ليس, باستثناء. Thai: ไม่, ยกเว้น. Russian: не, кроме, неверно.
""",
    "01_math_science_core.md": r"""
# Core mathematics and science facts

Arithmetic follows order of operations: parentheses, exponents, multiplication and division, addition and subtraction.
A percentage means a part per hundred. 25% equals 1/4; 50% equals 1/2; 75% equals 3/4.
The mean is the sum of values divided by the number of values. The median is the middle value after sorting. The mode is the most frequent value.
Probability of an event lies between 0 and 1. Independent events multiply: P(A and B)=P(A)P(B). For mutually exclusive events, P(A or B)=P(A)+P(B).
Area of a rectangle equals length times width. Area of a triangle equals one half times base times height. Circumference of a circle equals 2πr. Area of a circle equals πr².
The Pythagorean theorem states that in a right triangle, a² + b² = c² where c is the hypotenuse.
Speed equals distance divided by time. Density equals mass divided by volume. Force equals mass times acceleration. Kinetic energy equals one half times mass times velocity squared.
Water has chemical formula H2O. Carbon dioxide has formula CO2. Sodium chloride is NaCl.
An atom contains protons, neutrons, and electrons. Protons are positively charged, electrons are negatively charged, and neutrons are neutral.
Photosynthesis uses light energy, carbon dioxide, and water to produce glucose and oxygen. Cellular respiration releases energy from glucose.
DNA stores genetic information. RNA participates in protein synthesis.
The Earth orbits the Sun. The Moon orbits the Earth. Gravity attracts masses toward each other.
""",
    "02_computing_ai_core.md": r"""
# Computing, AI, and data core facts

A Docker container packages code, runtime libraries, dependencies, and execution settings so the same program can run reproducibly on another machine.
A Docker image is the packaged template; a container is a running instance of that image.
A CSV file stores tabular data as comma-separated values. A JSON object stores key-value pairs.
An API is an interface that lets software components communicate.
A database stores structured data. An index speeds up retrieval by mapping keys or terms to locations.
BM25 is a lexical retrieval ranking function based on term frequency, document frequency, and document length normalization.
Retrieval-Augmented Generation, or RAG, retrieves relevant documents from a knowledge base and gives them to a language model as context before answering.
An embedding maps text into numeric vectors so semantic similarity can be computed.
A reranker reorders retrieved candidates using a stronger relevance model.
A language model predicts tokens based on context. A small model may need retrieval context to answer knowledge-heavy questions reliably.
Quantization reduces model precision to lower memory usage, often allowing larger models to run on limited VRAM.
Inference means running a trained model to produce outputs. Training adjusts model weights using data.
Overfitting occurs when a system performs well on known examples but generalizes poorly to unseen examples.
A deterministic pipeline gives the same output for the same input and configuration.
""",
    "03_world_history_geography_core.md": r"""
# Stable world history and geography facts

Vietnam is in Southeast Asia. Hanoi is the capital of Vietnam. Ho Chi Minh City is a major economic center of Vietnam.
Thailand, Laos, Cambodia, Malaysia, Singapore, Indonesia, the Philippines, Brunei, Myanmar, Vietnam, and Timor-Leste are in Southeast Asia.
The United Nations is an international organization founded in 1945 after World War II.
World War I lasted from 1914 to 1918. World War II lasted from 1939 to 1945.
The Industrial Revolution involved a shift from hand production to machine production and factory systems.
Democracy is a political system where citizens participate directly or indirectly in choosing leaders and policies.
A market economy allocates resources mainly through supply, demand, and prices. A command economy allocates resources mainly through central planning.
Inflation is a general increase in prices and a decrease in purchasing power. GDP measures the value of final goods and services produced in an economy over a period.
The equator divides the Earth into Northern and Southern Hemispheres. Lines of longitude run north-south and measure east-west position.
The Pacific Ocean is the largest ocean. The Sahara is a large desert in North Africa. The Amazon is a major river system and rainforest region in South America.
""",
    "04_business_law_economics_core.md": r"""
# Business, law, and economics core concepts

A contract is an agreement that creates, changes, or ends rights and obligations between parties.
Offer and acceptance are basic elements of contract formation. Capacity, lawful purpose, and voluntary consent are important for validity in many legal systems.
A breach of contract occurs when a party fails to perform as agreed without lawful excuse.
Civil liability generally aims to compensate for damage caused by unlawful conduct or breach of obligation.
Insurance transfers financial risk from the insured to the insurer in exchange for a premium, subject to policy terms.
Property insurance covers loss or damage to property. Life insurance concerns human life or longevity. Liability insurance covers legal responsibility to third parties.
A company is a legal or organizational form used to conduct business. Limited liability means owners are not personally liable beyond their capital contribution except in special cases.
Revenue is income from business activities. Cost is the expense incurred to produce or obtain goods or services. Profit equals revenue minus cost.
Supply generally rises when price rises, holding other factors constant. Demand generally falls when price rises, holding other factors constant.
Opportunity cost is the value of the best alternative forgone.
""",
    "05_language_logic_core.md": r"""
# Language, logic, and reading comprehension cues

A statement and its negation cannot both be true at the same time in classical logic.
If all A are B and x is A, then x is B. If no A are B and x is A, then x is not B.
Correlation does not by itself prove causation.
An analogy compares relationships; the correct answer should preserve the same relation.
A synonym has similar meaning. An antonym has opposite meaning.
The main idea of a passage is the central point, not a minor detail.
An inference is a conclusion supported by evidence but not directly stated.
A necessary condition must be present for something to happen. A sufficient condition guarantees the result when present.
In English grammar, a noun names a person, place, thing, or idea. A verb expresses an action or state. An adjective describes a noun. An adverb modifies a verb, adjective, or another adverb.
In Vietnamese, dấu câu and context are important for identifying negation, comparison, and cause-effect relations.
""",
    "06_contest_official_contract.md": r"""
# Official local-inference contract reminders

The official prediction path must read public_test.csv or private_test.csv from /data and write pred.csv to /output.
The output pred.csv must contain exactly two columns: qid and answer. The answer must be one of A, B, C, or D.
The runtime must not require live web search, hidden browser automation, external API keys, or manual post-processing.
For stability, a valid system should load its legal local model once, process all rows, validate every answer, and atomically write the output file.
The knowledge engine should use offline corpus paths such as /knowledge, /data/knowledge, /data/corpus, /data/docs, and /app/knowledge.
""",
}


def _write_seed_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text(encoding="utf-8", errors="ignore").strip() != dedent(content).strip():
        path.write_text(dedent(content).strip() + "\n", encoding="utf-8")


def ensure_embedded_seed(seed_dir: str | Path | None = None) -> Path:
    """Create a deterministic seed corpus and return its directory.

    The directory is tiny and safe to create even in strict official mode. It is
    used only as an offline local corpus, not as a heuristic answer fallback.
    """
    raw = seed_dir or os.getenv(AUTO_SEED_DIR_ENV) or "/tmp/hackai_autoseed_knowledge"
    root = Path(raw)
    for name, content in SEED_FILES.items():
        _write_seed_file(root / name, content)
    return root


def official_knowledge_paths(raw: str | None = None) -> list[str]:
    """Return ordered knowledge paths with auto-seed included.

    Mounted corpora come first, bundled /app/knowledge next, deterministic seed
    last. Duplicate paths are removed while preserving order.
    """
    base_raw = raw if raw is not None else os.getenv("KNOWLEDGE_PATHS")
    if base_raw:
        pieces = [p.strip() for p in base_raw.replace(";", ":").split(":") if p.strip()]
    else:
        pieces = [
            "/knowledge", "/data/knowledge", "/data/corpus", "/data/docs", "/corpus",
            "/app/knowledge", "./knowledge", "./data/knowledge", "./data/corpus", "./data/docs",
        ]
    if os.getenv("AUTO_SEED_KNOWLEDGE", "1").strip().lower() in {"1", "true", "yes", "on", "auto"}:
        pieces.append(str(ensure_embedded_seed()))
        try:
            from .knowledge_corpus_expander import ensure_all_extra_corpus
            pieces.extend(ensure_all_extra_corpus())
        except Exception as e:
            if os.getenv("STRICT_NO_FALLBACK", "1").strip().lower() in {"1", "true", "yes", "on"} and os.getenv("REQUIRE_EXPANDED_CORPUS", "0").strip().lower() in {"1", "true", "yes", "on"}:
                raise RuntimeError(f"expanded corpus seeding failed: {e}") from e
    out: list[str] = []
    seen: set[str] = set()
    for p in pieces:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out
