from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .safety import redact_secrets


@dataclass(slots=True)
class EvidenceStep:
    iteration: int
    stage: str
    action: str
    args: dict[str, Any]
    ok: bool
    observation: str
    timestamp: float = field(default_factory=time.time)


class EvidenceLog:
    def __init__(self, workspace: str | Path, session_id: str | None = None):
        self.workspace = Path(workspace).resolve()
        self.session_id = _safe_name(session_id) if session_id else ""
        self.dir = self.workspace / ".tracegraph"
        if self.session_id:
            self.dir = self.dir / "sessions" / self.session_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "evidence.jsonl"

    def append(self, step: EvidenceStep) -> None:
        payload = asdict(step)
        payload["observation"] = redact_secrets(str(payload["observation"]))[:12000]
        payload["args"] = _redact_obj(payload["args"])
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def reset(self) -> None:
        self.path.write_text("", encoding="utf-8")

    def read_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines()[-limit:]:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    def format_chain(self, limit: int = 50) -> str:
        rows = self.read_recent(limit)
        if not rows:
            return "(no evidence recorded)"
        lines: list[str] = []
        for row in rows:
            args = json.dumps(row.get("args", {}), ensure_ascii=False)
            observation = str(row.get("observation", "")).replace("\n", " ")
            lines.append(
                (
                    f"[{row.get('iteration')}] {row.get('stage')}:{row.get('action')} "
                    f"ok={row.get('ok')} args={args[:260]}"
                )
            )
            if observation:
                lines.append(f"    {observation[:500]}")
        return "\n".join(lines)

    def write_report(
        self,
        final_text: str,
        verifier_text: str,
        diff_text: str,
        working_memory_text: str = "",
    ) -> Path:
        report = self.dir / "final_report.md"
        rows = self.read_recent(50)
        lines = ["# TraceGraph Coder Report", "", "## Final Answer", "", final_text.strip() or "(empty)", ""]
        lines.extend(["## Verification", "", verifier_text.strip() or "(not run)", ""])
        if working_memory_text:
            lines.extend(["## Working Memory", "", "```text", working_memory_text.strip(), "```", ""])
        lines.extend(["## Workspace Diff", "", "```diff", diff_text.strip()[:20000], "```", ""])
        lines.append("## Evidence")
        for row in rows:
            lines.append(
                f"- step {row.get('iteration')}: {row.get('action')} ok={row.get('ok')} stage={row.get('stage')}"
            )
        lines.extend(["", "## Evidence Chain", "", "```text", self.format_chain(50), "```", ""])
        report.write_text("\n".join(lines), encoding="utf-8")
        return report


def _redact_obj(obj: Any) -> Any:
    if isinstance(obj, str):
        return redact_secrets(obj)
    if isinstance(obj, dict):
        return {k: _redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact_obj(v) for v in obj]
    return obj


def _safe_name(value: str | None) -> str:
    clean = "".join(ch for ch in str(value or "") if ch.isalnum() or ch in "._-")
    if not clean:
        raise ValueError("invalid evidence session id")
    return clean[:120]
