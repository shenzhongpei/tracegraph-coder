from __future__ import annotations

import os
import re
import shlex
from pathlib import Path


SECRET_PATTERNS = [
    re.compile(r"(sk-[A-Za-z0-9_\-]{16,})"),
    re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*['\"]?([^'\"\s]+)"),
]

BLOCKED_COMMAND_PATTERNS = [
    r"\brm\s+-rf\s+[/~.]",
    r"\bgit\s+reset\s+--hard\b",
    r"\bgit\s+clean\s+-[fdx]+",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bmkfs\b",
    r"\bformat\b",
    r"\bdel\s+/[sq]\b",
    r"\brd\s+/s\b",
    r"\bRemove-Item\b.*\b-Recurse\b",
    r":\(\)\s*\{\s*:\|:",
]

BLOCKED_EXECUTABLES = {
    "del",
    "erase",
    "format",
    "mkfs",
    "rd",
    "reboot",
    "rm",
    "rmdir",
    "shutdown",
}


class SafetyError(ValueError):
    """Raised when a path or command violates the local safety policy."""


def ensure_workspace(root: str | Path) -> Path:
    path = Path(root).expanduser().resolve()
    if not path.exists():
        raise SafetyError(f"Workspace does not exist: {path}")
    if not path.is_dir():
        raise SafetyError(f"Workspace is not a directory: {path}")
    return path


def safe_path(root: str | Path, path: str | Path) -> Path:
    workspace = ensure_workspace(root)
    raw = Path(path)
    target = (raw if raw.is_absolute() else workspace / raw).resolve()
    try:
        target.relative_to(workspace)
    except ValueError as exc:
        raise SafetyError(f"Path escapes workspace: {path}") from exc
    return target


def assert_safe_command(command: str) -> None:
    normalized = command.strip()
    if not normalized:
        raise SafetyError("Command must not be empty.")
    if _has_unquoted_shell_operator(normalized):
        raise SafetyError("Shell operators are not allowed in agent commands. Run one command at a time.")
    for pattern in BLOCKED_COMMAND_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            raise SafetyError(f"Blocked dangerous command: {command}")
    args = split_safe_command(command)
    executable = Path(args[0]).name.lower()
    if executable.endswith((".exe", ".cmd", ".bat", ".ps1")):
        executable = Path(executable).stem.lower()
    if executable in BLOCKED_EXECUTABLES:
        raise SafetyError(f"Blocked dangerous executable: {args[0]}")


def split_safe_command(command: str) -> list[str]:
    try:
        args = shlex.split(command, posix=True)
    except ValueError as exc:
        raise SafetyError(f"Could not parse command safely: {exc}") from exc
    if not args:
        raise SafetyError("Command must not be empty.")
    return args


def _has_unquoted_shell_operator(command: str) -> bool:
    quote: str | None = None
    escaped = False
    for char in command:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char in {"|", "&", ";", "<", ">", "`"}:
            return True
    return False


def redact_secrets(text: str) -> str:
    redacted = text
    for pattern in SECRET_PATTERNS:
        if pattern.groups >= 2:
            redacted = pattern.sub(lambda m: f"{m.group(1)}=[REDACTED]", redacted)
        else:
            redacted = pattern.sub("[REDACTED]", redacted)
    for key, value in os.environ.items():
        if not value or len(value) < 12:
            continue
        if any(word in key.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            redacted = redacted.replace(value, "[REDACTED]")
    return redacted
