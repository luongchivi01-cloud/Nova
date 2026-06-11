from __future__ import annotations

"""Vietnamese-specific preprocessing and optional VnCoreNLP integration.

VnCoreNLP is vendored under third_party/ and can be enabled without network.
The official path falls back to regex/Unicode normalization when Java is absent.
"""

import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .third_party_registry import vncorenlp_home

NEGATION_TERMS = (
    "không", "khong", "sai", "ngoại trừ", "ngoai tru", "không đúng", "khong dung",
    "không phải", "khong phai", "chưa chính xác", "chua chinh xac",
)

DOMAIN_TERMS = {
    "law": ("hợp đồng", "pháp luật", "thương mại", "điều khoản", "chủ thể", "vi phạm"),
    "finance": ("ngân hàng", "lãi suất", "tài chính", "doanh thu", "chi phí", "tỷ giá"),
    "health": ("bệnh", "khám", "chữa", "y tế", "bệnh nhân", "sức khỏe"),
    "education": ("học", "sinh viên", "giáo viên", "nhà trường", "đào tạo"),
}


@dataclass(slots=True)
class VietnameseSignals:
    normalized: str
    has_negation: bool
    domains: list[str]
    token_count: int
    has_number: bool
    segmented: str = ""
    vncorenlp_used: bool = False


def normalize_vi(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\u200b", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


class VnCoreNLPCommandSegmenter:
    """Tiny command-line wrapper around the vendored VnCoreNLP jar.

    Requires Java. It is intentionally optional because the Docker official path
    must still run on CPU-only/minimal environments.
    """

    def __init__(self, home: str | Path | None = None, java_bin: str = "java", timeout: float = 8.0):
        self.home = Path(home) if home else vncorenlp_home()
        self.java_bin = java_bin
        self.timeout = timeout
        if self.home is None:
            raise RuntimeError("VnCoreNLP vendored repo not found")
        self.jar = self.home / "VnCoreNLP-1.2.jar"
        if not self.jar.exists():
            raise FileNotFoundError(self.jar)
        if shutil.which(java_bin) is None:
            raise RuntimeError("Java runtime not found; install openjdk or disable USE_VNCORENLP")

    def segment(self, text: str) -> str:
        text = normalize_vi(text)
        if not text:
            return text
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            fin = td_path / "input.txt"
            fout = td_path / "output.txt"
            fin.write_text(text, encoding="utf-8")
            cmd = [self.java_bin, "-Xmx1g", "-jar", str(self.jar), "-fin", str(fin), "-fout", str(fout), "-annotators", "wseg"]
            subprocess.run(cmd, cwd=str(self.home), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=self.timeout, check=True)
            raw = fout.read_text(encoding="utf-8", errors="ignore") if fout.exists() else ""
        toks: list[str] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2 and parts[0].isdigit():
                toks.append(parts[1])
            elif len(parts) >= 1:
                toks.extend(parts[0].split())
        return " ".join(toks) if toks else text


def word_segment_optional(text: str) -> str:
    """Best available Vietnamese word segmentation.

    Order:
    1. VnCoreNLP when USE_VNCORENLP=1 and Java is available.
    2. underthesea if installed.
    3. normalized text fallback.
    """
    text = normalize_vi(text)
    mode = os.getenv("USE_VNCORENLP", "0").strip().lower()
    if mode in {"1", "true", "yes", "on"}:
        try:
            return VnCoreNLPCommandSegmenter().segment(text)
        except Exception:
            pass
    try:
        from underthesea import word_tokenize  # type: ignore
        return str(word_tokenize(text, format="text"))
    except Exception:
        return text


def analyze_vi(text: str) -> VietnameseSignals:
    norm = normalize_vi(text)
    segmented = word_segment_optional(norm)
    lower = norm.lower()
    domains = [name for name, terms in DOMAIN_TERMS.items() if any(t in lower for t in terms)]
    return VietnameseSignals(
        normalized=norm,
        has_negation=any(t in lower for t in NEGATION_TERMS),
        domains=domains,
        token_count=len(segmented.lower().split()),
        has_number=bool(re.search(r"\d", lower)),
        segmented=segmented,
        vncorenlp_used=(segmented != norm and "_" in segmented),
    )
