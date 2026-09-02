from __future__ import annotations

import difflib
import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .repo_graph import filter_files
from .safety import ensure_workspace, redact_secrets, safe_path


MAX_DIFF_FILE_BYTES = 200_000
MAX_DIFF_CHARS = 60_000


@dataclass(frozen=True, slots=True)
class _CachedFileState:
    size: int
    mtime_ns: int
    sha256: str
    text: str | None


_SNAPSHOT_CACHE: dict[str, dict[str, _CachedFileState]] = {}


@dataclass(frozen=True, slots=True)
class FileState:
    path: str
    sha256: str
    size: int
    mtime_ns: int
    text: str | None


@dataclass(frozen=True, slots=True)
class WorkspaceSnapshot:
    root: str
    files: dict[str, FileState]

    @classmethod
    def capture(cls, root: str | Path) -> "WorkspaceSnapshot":
        workspace = ensure_workspace(root)
        cache_key = str(workspace)
        previous_cache = _SNAPSHOT_CACHE.get(cache_key, {})
        next_cache: dict[str, _CachedFileState] = {}
        files: dict[str, FileState] = {}
        for rel in filter_files(workspace, pattern="*", limit=20_000):
            target = safe_path(workspace, rel)
            try:
                stat_result = target.stat()
            except OSError:
                continue
            cached = previous_cache.get(rel)
            if cached is not None and cached.size == stat_result.st_size and cached.mtime_ns == stat_result.st_mtime_ns:
                state = FileState(
                    path=rel,
                    sha256=cached.sha256,
                    size=cached.size,
                    mtime_ns=cached.mtime_ns,
                    text=cached.text,
                )
            else:
                try:
                    raw = target.read_bytes()
                except OSError:
                    continue
                text = None
                if len(raw) <= MAX_DIFF_FILE_BYTES:
                    text = raw.decode("utf-8", errors="replace")
                state = FileState(
                    path=rel,
                    sha256=hashlib.sha256(raw).hexdigest(),
                    size=len(raw),
                    mtime_ns=stat_result.st_mtime_ns,
                    text=text,
                )
            files[rel] = state
            next_cache[rel] = _CachedFileState(
                size=state.size,
                mtime_ns=state.mtime_ns,
                sha256=state.sha256,
                text=state.text,
            )
        _SNAPSHOT_CACHE[cache_key] = next_cache
        return cls(root=str(workspace), files=files)

    def changed_from(self, previous: "WorkspaceSnapshot") -> bool:
        return {k: (v.sha256, v.size) for k, v in self.files.items()} != {
            k: (v.sha256, v.size) for k, v in previous.files.items()
        }

    def changed_paths_from(self, previous: "WorkspaceSnapshot") -> list[str]:
        paths = sorted(set(self.files) | set(previous.files))
        changed: list[str] = []
        for path in paths:
            current = self.files.get(path)
            old = previous.files.get(path)
            if current is None or old is None or (current.sha256, current.size) != (old.sha256, old.size):
                changed.append(path)
        return changed


def workspace_diff(root: str | Path, baseline: WorkspaceSnapshot | None = None) -> str:
    workspace = ensure_workspace(root)
    if baseline is not None:
        return snapshot_diff(baseline, WorkspaceSnapshot.capture(workspace))
    git = _git_diff(workspace)
    if git is not None:
        return git
    return "(git diff unavailable and no baseline snapshot was captured)"


def snapshot_diff(before: WorkspaceSnapshot, after: WorkspaceSnapshot) -> str:
    chunks: list[str] = []
    paths = sorted(set(before.files) | set(after.files))
    for rel in paths:
        old = before.files.get(rel)
        new = after.files.get(rel)
        if old is None and new is not None:
            chunks.append(f"created: {rel} ({new.size} bytes)\n")
            chunks.extend(_text_diff("", new.text, f"a/{rel}", f"b/{rel}"))
        elif old is not None and new is None:
            chunks.append(f"deleted: {rel} ({old.size} bytes)\n")
            chunks.extend(_text_diff(old.text, "", f"a/{rel}", f"b/{rel}"))
        elif old is not None and new is not None and old.sha256 != new.sha256:
            chunks.append(f"modified: {rel} ({old.size} -> {new.size} bytes)\n")
            chunks.extend(_text_diff(old.text, new.text, f"a/{rel}", f"b/{rel}"))
        if sum(len(chunk) for chunk in chunks) > MAX_DIFF_CHARS:
            chunks.append("\n...[diff truncated]\n")
            break
    return "".join(chunks).strip() or "(empty)"


def _text_diff(old_text: str | None, new_text: str | None, old_name: str, new_name: str) -> list[str]:
    if old_text is None or new_text is None:
        return ["(content omitted because a file is too large for snapshot diff)\n"]
    return list(
        difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=old_name,
            tofile=new_name,
            n=3,
        )
    )


def _git_diff(workspace: Path) -> str | None:
    try:
        inside = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=workspace,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
    except Exception:
        return None
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return None
    proc = subprocess.run(
        ["git", "diff", "--", "."],
        cwd=workspace,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )
    out = redact_secrets(proc.stdout or "").strip()
    if proc.returncode != 0:
        return f"(git diff failed: {out or 'no output'})"
    return out or "(empty)"
