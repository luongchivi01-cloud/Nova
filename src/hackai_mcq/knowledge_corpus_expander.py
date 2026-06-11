from __future__ import annotations

"""Extra self-loading offline corpus and repo-doc corpus ingestion.

This module does not call the web. It creates a wider deterministic corpus from
bundled knowledge packs and safely extracts documentation from vendored repos.
The goal is to give weak local models more high-signal evidence while keeping
the official Docker path reproducible and offline.
"""

import os
import re
from pathlib import Path
from textwrap import dedent

EXPANDED_DIR_ENV = "EXPANDED_KNOWLEDGE_DIR"
REPO_DOC_DIR_ENV = "REPO_DOC_KNOWLEDGE_DIR"

EXPANDED_FILES: dict[str, str] = {
    "10_math_formula_reference.md": """
# Math formula reference for multiple-choice reasoning

Numbers and operations: order of operations is parentheses, exponents, multiplication/division, addition/subtraction. A fraction represents division; 0.25 equals one fourth; 0.2 equals one fifth. A ratio compares quantities. A rate compares quantities with different units.
Percent change equals (new value - old value) divided by old value times 100%. Simple interest equals principal times rate times time. Compound interest uses repeated multiplication by (1+r) per period.
Algebra: a linear equation has variables of degree one. A quadratic equation has degree two. The quadratic formula solves ax^2+bx+c=0 by x=(-b plus or minus sqrt(b^2-4ac))/(2a). Slope is rise over run. A proportional relationship has equation y=kx.
Geometry: perimeter is total boundary length. Rectangle area is length times width. Triangle area is base times height divided by two. Circle circumference is 2*pi*r and area is pi*r^2. Cylinder volume is pi*r^2*h. Sphere volume is 4/3*pi*r^3.
Statistics: mean is average, median is middle value after sorting, mode is most frequent value, range is maximum minus minimum. Variance measures average squared deviation; standard deviation is square root of variance. A z-score measures distance from mean in standard deviations.
Probability: probability ranges from 0 to 1. Complement probability P(not A)=1-P(A). Independent event probability multiplies. Mutually exclusive event probability adds. Conditional probability P(A|B)=P(A and B)/P(B).
Logic traps: if a question asks for least, not, except, false, impossible, or incorrect, invert the target. Units must match before arithmetic. Approximate answers should be checked by order of magnitude.
""",
    "11_physics_chemistry_biology_core.md": """
# Physics, chemistry, biology core facts

Physics: Speed equals distance divided by time. Velocity has direction; speed is scalar. Acceleration is change in velocity over time. Newton's second law states F=ma. Weight equals mass times gravitational acceleration. Work equals force times displacement in the direction of force. Power equals work divided by time. Kinetic energy is 1/2 mv^2. Potential energy near Earth is mgh.
Waves: frequency is cycles per second; period is the reciprocal of frequency. Wave speed equals frequency times wavelength. Sound needs a medium; light can travel through vacuum. Reflection is bouncing back; refraction is bending when entering a different medium.
Electricity: current is flow of charge. Voltage is electric potential difference. Resistance opposes current. Ohm's law is V=IR. Series resistors add directly; parallel resistors have reciprocal sum.
Chemistry: Atoms contain protons, neutrons, and electrons. Atomic number equals number of protons. Isotopes have same protons but different neutrons. Ionic bonds involve electron transfer; covalent bonds involve electron sharing. Acids donate protons or produce H+ in water; bases accept protons or produce OH-. pH below 7 is acidic, 7 neutral, above 7 basic.
Biology: Cells are basic units of life. Prokaryotes lack a nucleus; eukaryotes have a nucleus. DNA stores hereditary information. Genes encode functional products. Photosynthesis converts carbon dioxide and water into glucose and oxygen using light. Cellular respiration releases energy from glucose. Evolution by natural selection favors heritable traits that improve reproductive success.
Human body: The heart pumps blood. Lungs exchange oxygen and carbon dioxide. The nervous system transmits signals. The immune system defends against pathogens. Vaccines train immune response without causing the full disease.
""",
    "12_computing_data_ai_core.md": """
# Computing, data, cybersecurity, and AI core facts

Programming: A variable stores a value. A function groups reusable logic. A loop repeats instructions. A condition chooses between branches. A data structure organizes data. Arrays/lists store ordered items. Dictionaries/maps store key-value pairs. Recursion is a function calling itself with a base case.
Complexity: Big-O describes how runtime or memory grows with input size. O(1) is constant, O(log n) logarithmic, O(n) linear, O(n log n) common for efficient sorting, O(n^2) quadratic.
Databases: SQL databases store structured tables. Primary keys uniquely identify rows. Foreign keys connect tables. Indexes speed lookups but cost storage and write overhead. Transactions use ACID properties: atomicity, consistency, isolation, durability.
Networking: HTTP is a request-response protocol. HTTPS encrypts HTTP using TLS. DNS maps domain names to IP addresses. TCP is reliable and connection-oriented; UDP is faster but does not guarantee delivery.
Cybersecurity: Authentication verifies identity. Authorization grants permissions. Encryption protects confidentiality. Hashing maps data to fixed-size digests. Salting helps protect password hashes. Phishing tricks users into revealing secrets. Principle of least privilege reduces risk.
AI/ML: Supervised learning uses labeled data. Unsupervised learning finds patterns without labels. Reinforcement learning learns by rewards and actions. Classification predicts categories; regression predicts continuous values. Precision, recall, F1, and accuracy evaluate models. A confusion matrix summarizes predictions vs true labels.
LLM systems: Retrieval augmented generation retrieves external evidence before generation. Prompting controls behavior but does not change model weights. Fine-tuning changes weights. Quantization lowers memory use at possible quality cost. Constrained decoding restricts possible outputs. For multiple-choice tasks, scoring A/B/C/D tokens can be more stable than free-form generation.
""",
    "13_world_culture_geography_history_core.md": """
# World geography, history, culture, and civics core facts

Continents commonly listed are Africa, Antarctica, Asia, Europe, North America, Australia/Oceania, and South America. The Pacific is the largest ocean. The equator is 0 degrees latitude. The prime meridian is 0 degrees longitude. Latitude measures north-south position; longitude measures east-west position.
Capital examples: Vietnam-Hanoi, Thailand-Bangkok, Japan-Tokyo, China-Beijing, South Korea-Seoul, France-Paris, Germany-Berlin, Italy-Rome, United Kingdom-London, United States-Washington DC, Canada-Ottawa, Australia-Canberra, India-New Delhi, Russia-Moscow.
Vietnam: Vietnam is in Southeast Asia. Hanoi is the capital. Ho Chi Minh City is a major economic city. The Red River and Mekong River are important river systems. ASEAN is a regional organization in Southeast Asia.
History: Ancient civilizations developed around river valleys such as Nile, Tigris-Euphrates, Indus, and Yellow River. The Renaissance emphasized renewed interest in classical learning, art, and humanism. The Industrial Revolution shifted production toward machines, factories, and urbanization. World War I occurred 1914-1918. World War II occurred 1939-1945. The United Nations was founded in 1945.
Civics: A constitution establishes basic principles and institutions of a state. Democracy involves citizen participation and accountability. Rule of law means government and citizens are subject to law. Separation of powers divides governmental authority to prevent abuse.
Culture/language: A primary source is direct evidence from the period or event; a secondary source interprets primary sources. Bias is a systematic slant in presentation or interpretation. Context is necessary to interpret historical events fairly.
""",
    "14_business_law_finance_core.md": """
# Business, law, accounting, finance, and economics core facts

Economics: Scarcity means resources are limited relative to wants. Opportunity cost is the value of the best forgone alternative. Supply tends to rise with price; demand tends to fall with price. Equilibrium occurs where supply equals demand. Inflation is a general rise in prices. GDP measures market value of final goods and services produced in an economy.
Finance: Simple interest is principal times rate times time. Compound interest earns interest on previous interest. Risk-return tradeoff means higher expected returns usually require higher risk. Diversification reduces unsystematic risk by spreading investments. Liquidity means how easily an asset can be converted to cash.
Accounting: Assets are resources controlled by an entity. Liabilities are present obligations. Equity is residual interest after liabilities. Revenue increases economic benefits from ordinary activities. Expenses decrease economic benefits. Profit equals revenue minus expenses. Double-entry accounting records debits and credits.
Contract law: A contract is an agreement creating rights and obligations. Offer, acceptance, capacity, consent, lawful purpose, and consideration/causa may matter depending on legal system. Breach occurs when a party fails to perform an obligation. Damages compensate loss. Force majeure can excuse performance when an unforeseeable external event prevents performance under agreed conditions.
Business entities: Limited liability protects owners from personal responsibility beyond their investment in many situations. Sole proprietorship usually has unlimited liability. Partnerships involve two or more persons carrying on business together. Corporations/companies can have separate legal personality.
Insurance: Insurance transfers risk to insurer for premium. Insurable interest means the insured would suffer loss from the event. Indemnity restores the insured to financial position before loss, not to profit. Deductible is the portion borne by insured. Reinsurance is insurance purchased by an insurer to transfer part of its risk. Co-insurance can involve sharing risk between insurers or insured participation depending on context.
""",
    "15_health_education_public_admin_core.md": """
# Health, education, public administration, and service design core facts

Health: Diagnosis identifies disease or condition. Treatment aims to cure, manage, or reduce symptoms. Prevention reduces likelihood of disease. Public health focuses on populations, not only individuals. Triage prioritizes patients based on urgency. Informed consent requires adequate information and voluntary agreement. Privacy and confidentiality protect patient data.
Evidence-based medicine combines clinical expertise, patient values, and best available evidence. Screening tests detect possible disease before symptoms; diagnostic tests confirm or identify disease. Sensitivity measures true positive rate; specificity measures true negative rate.
Education: Personalized learning adapts pace, content, or support to learner needs. Formative assessment guides learning during instruction; summative assessment evaluates learning after instruction. Feedback should be timely, specific, and actionable. Accessibility means designing so people with disabilities can use the service.
Public administration: Public services should be transparent, efficient, accessible, and accountable. Digital transformation changes processes, data flows, and service delivery, not only user interface. A one-stop service center aims to reduce citizen effort. User satisfaction can be measured by surveys, completion rate, waiting time, and complaint rate.
AI governance: Data minimization collects only necessary data. Consent, security, fairness, explainability, and accountability are important when AI affects people. Human oversight is important in high-stakes decisions.
""",
    "16_english_vietnamese_language_core.md": """
# English and Vietnamese language knowledge for MCQ

English parts of speech: A noun names a person, place, thing, or idea. A pronoun replaces a noun. A verb expresses action or state. An adjective describes a noun. An adverb modifies a verb, adjective, or adverb. A preposition shows relationship such as in, on, at, by, with. A conjunction connects words or clauses.
English grammar: Subject-verb agreement means singular subjects usually take singular verbs and plural subjects take plural verbs. Present perfect connects past action to present relevance. Passive voice uses be plus past participle. Comparative forms compare two; superlative forms compare three or more. Articles a/an are indefinite; the is definite.
Reading: Main idea is the central point. Supporting details explain or prove the main idea. Tone is the writer's attitude. Inference is a reasonable conclusion from evidence. A conclusion should not go beyond the passage.
Vietnamese: Từ đồng nghĩa have similar meanings; từ trái nghĩa have opposite meanings. Chủ ngữ is subject; vị ngữ is predicate. Biện pháp tu từ includes so sánh, nhân hóa, ẩn dụ, hoán dụ, điệp ngữ, nói quá, nói giảm nói tránh. Câu nghi vấn asks, câu cầu khiến requests/orders, câu cảm thán expresses emotion, câu trần thuật states information.
Translation traps: False friends, negation, quantifiers, and idioms can change meaning. Always preserve whether the sentence is affirmative or negative.
""",
    "17_reasoning_traps_adversarial_core.md": """
# Reasoning traps and adversarial MCQ cues

Negation trap: In questions with not, except, false, incorrect, không đúng, sai, ngoại trừ, choose the option that fails the statement, not the generally correct statement.
Best-answer trap: If all options are partly true, choose the one that directly answers the exact question and is most complete.
Scope trap: Words like always, never, all, only, must, cannot are stronger than words like usually, often, may, can. Strong absolute claims are often false unless supported.
Causation trap: Correlation alone does not prove causation. Temporal order alone does not prove causation.
Definition trap: A correct definition must include essential features, not just an example or consequence.
Calculation trap: Check units, signs, and whether the question asks increase, decrease, ratio, percentage point, or percent change.
Vocabulary trap: A term can have domain-specific meaning. In law, consideration/capacity/liability may differ from everyday language. In computing, class/object/interface have technical meanings.
Evidence trap: Retrieved evidence can be related but not sufficient. Prefer evidence that directly mentions the entity and relation asked in the question.
""",
    "18_multilingual_cues_expanded.md": """
# Multilingual cue words for offline MCQ routing

English negation/exclusion: not, except, false, incorrect, least likely, does not, cannot, never, without, excluding.
Vietnamese negation/exclusion: không, không phải, không đúng, sai, ngoại trừ, trừ, chưa chính xác, ít có khả năng nhất.
Chinese negation/exclusion: 不, 不是, 没有, 错误, 除了, 不正确. Cues: 最可能 means most likely; 主要原因 means main reason; 结果 means result.
Japanese negation/exclusion: ない, ではない, 誤り, 除く, 以外. Cues: 最も means most; 原因 means cause; 結果 means result.
Korean negation/exclusion: 아니다, 않은, 틀린, 제외, 없다. Cues: 가장 means most; 원인 means cause; 결과 means result.
Arabic negation/exclusion: لا, ليس, غير, باستثناء, خاطئ. Cues: السبب means cause; النتيجة means result; الأكثر means most.
Thai negation/exclusion: ไม่, ยกเว้น, ผิด, ไม่ใช่. Cues: สาเหตุ means cause; ผล means result.
Russian negation/exclusion: не, неверно, кроме, исключая, ложный. Cues: причина means cause; результат means result; наиболее means most.
For mixed-language questions, keep named entities unchanged and translate only instruction cues mentally. The final answer must still be A/B/C/D.
""",
    "19_exam_domain_router_reference.md": """
# Domain router reference

If the question asks about formula, calculate, percent, probability, mean, speed, force, area, volume, pH, atom, cell, DNA, photosynthesis, classify it as math/science.
If it asks about algorithm, Docker, API, database, network, HTTP, encryption, AI, model, embedding, retrieval, classify it as computing/AI.
If it asks about contract, liability, insurance, company, revenue, cost, market, inflation, GDP, classify it as business/law/economics.
If it asks about constitution, state, public service, administration, citizen, policy, governance, classify it as civics/public administration.
If it asks about grammar, synonym, antonym, main idea, inference, tone, translation, classify it as language/reading.
If it asks about country, capital, river, ocean, war, revolution, organization, classify it as history/geography.
Domain classification should route retrieval and prompts, not override the model answer by itself.
""",
}

REPO_DOC_PATTERNS = ("README", "readme", "docs", "examples", "tutorial", "guide", "paper")
REPO_DOC_SUFFIXES = {".md", ".rst", ".txt"}
MAX_REPO_DOC_FILES = 120
MAX_REPO_DOC_CHARS = 16000


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = dedent(text).strip() + "\n"
    if not path.exists() or path.read_text(encoding="utf-8", errors="ignore") != cleaned:
        path.write_text(cleaned, encoding="utf-8")


def ensure_expanded_seed(seed_dir: str | Path | None = None) -> Path:
    root = Path(seed_dir or os.getenv(EXPANDED_DIR_ENV) or "/tmp/hackai_expanded_knowledge")
    for name, content in EXPANDED_FILES.items():
        _write(root / name, content)
    return root


def _safe_read(path: Path, limit: int = MAX_REPO_DOC_CHARS) -> str:
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")[:limit]
    except Exception:
        return ""
    raw = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", raw)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def ensure_repo_doc_seed(repo_root: str | Path | None = None, out_dir: str | Path | None = None) -> Path | None:
    """Extract high-signal docs from vendored repos into a searchable corpus.

    This is intentionally conservative: it indexes documentation/readmes only,
    not entire source trees, to avoid polluting the MCQ knowledge index with code.
    """
    base = Path(repo_root or os.getenv("THIRD_PARTY_DIR", "third_party"))
    if not base.exists():
        base = Path("/app/third_party")
    if not base.exists():
        return None
    out = Path(out_dir or os.getenv(REPO_DOC_DIR_ENV) or "/tmp/hackai_repo_doc_knowledge")
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in sorted(base.rglob("*")):
        if count >= MAX_REPO_DOC_FILES:
            break
        if not f.is_file() or f.suffix.lower() not in REPO_DOC_SUFFIXES:
            continue
        rel = str(f.relative_to(base)).replace("\\", "/")
        low = rel.lower()
        if not any(p.lower() in low for p in REPO_DOC_PATTERNS):
            continue
        txt = _safe_read(f)
        if len(txt) < 300:
            continue
        header = f"# Vendored repo documentation: {rel}\n\nSource path: {rel}\n\n"
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", rel)[:180] + ".md"
        _write(out / safe_name, header + txt)
        count += 1
    _write(out / "00_REPO_DOC_CORPUS_README.md", f"""
# Vendored repository documentation corpus

This directory is generated offline from documentation files already bundled in third_party.
It is used for questions about AI systems, retrieval, evaluation, containers, and tool behavior.
It does not call the internet and does not index private credentials or runtime logs.
Files extracted: {count}.
""")
    return out


def ensure_all_extra_corpus() -> list[str]:
    paths = [str(ensure_expanded_seed())]
    if os.getenv("AUTO_INDEX_REPO_DOCS", "1").strip().lower() in {"1", "true", "yes", "on", "auto"}:
        repo = ensure_repo_doc_seed()
        if repo is not None:
            paths.append(str(repo))
    return paths
