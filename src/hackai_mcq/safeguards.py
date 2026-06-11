from __future__ import annotations

import ast
import os
from contextlib import contextmanager
from pathlib import Path


BANNED_FILES = (
    ".env", "cookies.txt", "cookie.json", "session.json", "playwright/.auth", "storage_state.json",
)
BANNED_MODULES = {"playwright", "selenium", "pyautogui", "undetected_chromedriver"}


def assert_no_sensitive_files(root: str | Path = ".") -> None:
    r = Path(root)
    for p in r.rglob("*"):
        rel = str(p.relative_to(r)).replace("\\", "/").lower()
        if any(b in rel for b in BANNED_FILES):
            raise RuntimeError(f"Sensitive/session file must not be included: {p}")


def _imports_banned_module(source: str) -> str | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_name = alias.name.split(".")[0]
                if root_name in BANNED_MODULES:
                    return alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_name = node.module.split(".")[0]
                if root_name in BANNED_MODULES:
                    return node.module
    return None


def assert_no_browser_automation(root: str | Path = ".") -> None:
    """Keep official submission source tree clean and reproducible."""
    r = Path(root)
    if not r.exists():
        return
    for p in r.rglob("*.py"):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        banned = _imports_banned_module(text)
        if banned:
            raise RuntimeError(f"Browser automation import found in official path: {p} imports {banned}")


@contextmanager
def optional_network_block(enable_network: bool):
    # Docker should run with --network none for strict validation.
    old = os.environ.get("HACKAI_NETWORK_ENABLED")
    os.environ["HACKAI_NETWORK_ENABLED"] = "1" if enable_network else "0"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("HACKAI_NETWORK_ENABLED", None)
        else:
            os.environ["HACKAI_NETWORK_ENABLED"] = old
