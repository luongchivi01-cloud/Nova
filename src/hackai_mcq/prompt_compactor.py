from __future__ import annotations

"""Prompt/context compaction utilities.

Long evidence is a major latency source for 6GB VRAM.  This module keeps the
context concise without changing the input/output contract.
"""

import re

_SENT_SPLIT = re.compile(r"(?<=[.!?。！？])\s+|\n+")


def compact_context(context: str, max_chars: int = 1200, keep_head: int = 240) -> str:
    text = (context or "").strip()
    if not text or len(text) <= max_chars:
        return text
    max_chars = max(256, int(max_chars))
    # Prefer complete evidence lines/sentences with question-answer keywords.
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    if not parts:
        return text[:max_chars]
    kept: list[str] = []
    total = 0
    for p in parts:
        # Keep short, information-dense snippets first.  Very long paragraphs
        # consume context and slow scoring/generation disproportionately.
        if len(p) > max_chars * 0.55:
            p = p[: int(max_chars * 0.55)].rstrip() + "…"
        if total + len(p) + 1 > max_chars:
            continue
        kept.append(p)
        total += len(p) + 1
        if total >= max_chars * 0.92:
            break
    out = "\n".join(kept).strip()
    if len(out) < min(keep_head, max_chars // 2):
        out = text[:max_chars]
    return out[:max_chars]


def trim_prompt(prompt: str, hard_limit_chars: int = 4500) -> str:
    prompt = prompt or ""
    if len(prompt) <= hard_limit_chars:
        return prompt
    # Preserve the end, where the final output instruction usually lives, and a
    # short header that contains role/task framing.
    head = prompt[:900]
    tail = prompt[-max(1200, hard_limit_chars - len(head) - 40):]
    return head.rstrip() + "\n...[context compacted for speed]...\n" + tail.lstrip()
