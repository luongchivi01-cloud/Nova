
from __future__ import annotations

"""Offline corpus loader for /knowledge, /data/knowledge, /corpus and ZIM metadata."""

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .knowledge_manifest import KnowledgeManifest, KnowledgeSource, hash_file_sample
from .normalization import truncate

TEXT_SUFFIXES = {".txt", ".md", ".rst", ".html", ".htm"}
STRUCTURED_SUFFIXES = {".csv", ".jsonl", ".json"}
ZIM_SUFFIXES = {".zim"}

@dataclass(slots=True)
class CorpusDoc:
    doc_id: str
    text: str
    source: str = "corpus"
    meta: dict[str, str] | None = None


def _clean_text(text: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", text, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_blocks(text: str, max_block_chars: int = 1600) -> list[str]:
    text = _clean_text(text)
    if not text:
        return []
    rough = [b.strip() for b in re.split(r"\n\s*\n|(?<=\.)\s+(?=[A-ZÀ-Ỵ])", text) if b.strip()]
    blocks: list[str] = []
    current = ""
    for b in rough:
        if len(current) + len(b) + 1 <= max_block_chars:
            current = (current + " " + b).strip()
        else:
            if current:
                blocks.append(current)
            if len(b) > max_block_chars:
                blocks.extend(b[i:i+max_block_chars] for i in range(0, len(b), max_block_chars))
                current = ""
            else:
                current = b
    if current:
        blocks.append(current)
    return blocks or [truncate(text, max_block_chars)]


def load_file(path: Path, source_label: str | None = None) -> list[CorpusDoc]:
    suffix = path.suffix.lower()
    label = source_label or path.name
    docs: list[CorpusDoc] = []
    try:
        if suffix in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for i, block in enumerate(_split_blocks(text)):
                docs.append(CorpusDoc(f"{path.name}:{i}", block, label))
        elif suffix == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    text = " | ".join(str(v) for v in row.values() if v)
                    if text.strip():
                        docs.append(CorpusDoc(f"{path.name}:{i}", text, label))
        elif suffix == ".jsonl":
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                        text = str(obj.get("text") or obj.get("contents") or obj.get("content") or obj)
                    except Exception:
                        text = line
                    if text.strip():
                        docs.append(CorpusDoc(f"{path.name}:{i}", text, label))
        elif suffix == ".json":
            obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            rows = obj if isinstance(obj, list) else [obj]
            for i, row in enumerate(rows):
                if isinstance(row, dict):
                    text = str(row.get("text") or row.get("contents") or row.get("content") or row)
                else:
                    text = str(row)
                if text.strip():
                    docs.append(CorpusDoc(f"{path.name}:{i}", text, label))
        elif suffix in ZIM_SUFFIXES:
            # We cannot parse binary ZIM without libzim/kiwix binary. Keep a
            # manifest doc so the operator sees the archive was found; actual
            # ZIM search is handled by KiwixZimSearchAdapter if kiwix-search exists.
            docs.append(CorpusDoc(f"{path.name}:zim", f"Offline ZIM archive available at {path.name}. Use Kiwix adapter for article search.", "zim"))
    except Exception:
        return []
    return docs


def discover_files(paths: Iterable[str | Path]) -> list[Path]:
    files: list[Path] = []
    allowed = TEXT_SUFFIXES | STRUCTURED_SUFFIXES | ZIM_SUFFIXES
    for raw in paths:
        if not raw:
            continue
        p = Path(raw)
        if not p.exists():
            continue
        if p.is_file() and p.suffix.lower() in allowed:
            files.append(p)
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in allowed:
                    files.append(f)
    return files


def build_corpus(paths: Iterable[str | Path]) -> tuple[list[CorpusDoc], KnowledgeManifest]:
    manifest = KnowledgeManifest(ok=True)
    docs: list[CorpusDoc] = []
    by_root: dict[str, KnowledgeSource] = {}
    for f in discover_files(paths):
        loaded = load_file(f)
        docs.extend(loaded)
        root = str(f.parent)
        source = by_root.setdefault(root, KnowledgeSource(path=root, kind="offline_folder"))
        source.files += 1
        source.docs += len(loaded)
        try:
            source.bytes += f.stat().st_size
        except Exception:
            pass
        if not source.sha256_sample:
            source.sha256_sample = hash_file_sample(f)
    manifest.sources = list(by_root.values())
    manifest.total_docs = len(docs)
    return docs, manifest


def export_jsonl(docs: list[CorpusDoc], out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"id": d.doc_id, "text": d.text, "source": d.source, "meta": d.meta or {}}, ensure_ascii=False) + "\n")
