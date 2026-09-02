from __future__ import annotations

from pathlib import Path

from .safety import ensure_workspace, redact_secrets, safe_path


MEMORY_FILES = ("TRACEGRAPH.md", "AGENTS.md")


def load_project_memory(workspace: str | Path, max_chars: int = 6000) -> str:
    root = ensure_workspace(workspace)
    sections: list[str] = []
    remaining = max_chars
    for name in MEMORY_FILES:
        target = safe_path(root, name)
        if not target.exists() or not target.is_file():
            continue
        try:
            text = target.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if not text:
            continue
        text = redact_secrets(text)
        if len(text) > remaining:
            text = text[:remaining] + "\n...[project memory truncated]"
        sections.append(f"## {name}\n{text}")
        remaining -= len(text)
        if remaining <= 0:
            break
    return "\n\n".join(sections)
