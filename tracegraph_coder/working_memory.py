from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .safety import ensure_workspace, redact_secrets, safe_path
from .verifier import VerificationResult


PHASES = {"exploring", "modifying", "verifying", "ready"}
TASK_TYPES = {"analysis", "coding", "conversation", "docs", "test", "ui"}
MAX_FILES = 12
MAX_CANDIDATE_FILES = 12
MAX_MODIFIED_PATHS = 64
MAX_OBSERVATIONS = 8
MAX_CHANGES = 8
MAX_VERIFICATION = 5
MAX_TEXT = 420
MAX_TARGET_PATHS = 8
MAX_EXPERIENCE_TERMS = 16
MAX_FAILURE_TERMS = 16
MAX_FAILURE_EVENTS = 6
MAX_TERM_TEXT = 80


@dataclass(frozen=True, slots=True)
class FileMemory:
    path: str
    action: str
    step: int
    size: int | None = None
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class FailureEvent:
    step: int
    source: str
    category: str
    summary: str
    terms: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)


@dataclass
class WorkingMemory:
    """Deterministic, bounded state for one agent run.

    The model can suggest progress through record_progress, but file evidence,
    mutations, and verification state are updated only from local tool results.
    """

    task: str
    files: list[FileMemory] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    changes: list[str] = field(default_factory=list)
    verification: list[str] = field(default_factory=list)
    stale_files: list[str] = field(default_factory=list)
    task_type: str = "coding"
    candidate_files: list[str] = field(default_factory=list)
    phase: str = "exploring"
    hypothesis: str = ""
    next_step: str = ""
    target_paths: list[str] = field(default_factory=list)
    modified_paths: list[str] = field(default_factory=list)
    modified_paths_overflow: bool = False
    last_mutation_step: int | None = None
    last_verification_step: int | None = None
    verification_command: str = ""
    verification_summary: str = ""
    verified_files: list[FileMemory] = field(default_factory=list)
    experience_hint: str = ""
    experience_id: str = ""
    experience_searches: int = 0
    experience_terms: list[str] = field(default_factory=list)
    failure_terms: list[str] = field(default_factory=list)
    failure_events: list[FailureEvent] = field(default_factory=list)

    def observe_progress(self, arguments: dict[str, Any], step: int) -> None:
        _validate_step(step)
        allowed = {
            "phase",
            "hypothesis",
            "next_step",
            "target_paths",
            "experience_hint",
            "experience_id",
            "experience_searches",
            "experience_terms",
            "failure_terms",
        }
        unknown = sorted(set(arguments) - allowed)
        if unknown:
            raise ValueError(f"unknown progress field(s): {', '.join(unknown)}")
        if "phase" in arguments:
            phase = str(arguments["phase"])
            if phase not in PHASES:
                raise ValueError(f"phase must be one of {', '.join(sorted(PHASES))}")
            self.phase = phase
        if "hypothesis" in arguments:
            self.hypothesis = _clean(arguments["hypothesis"])
        if "next_step" in arguments:
            self.next_step = _clean(arguments["next_step"])
        if "target_paths" in arguments:
            targets = arguments["target_paths"]
            if not isinstance(targets, list):
                raise ValueError("target_paths must be a list")
            if len(targets) > MAX_TARGET_PATHS:
                raise ValueError(f"target_paths must contain at most {MAX_TARGET_PATHS} items")
            self.target_paths = [_clean(item, limit=240) for item in targets if str(item).strip()]
        if "experience_hint" in arguments:
            self.experience_hint = _clean(arguments["experience_hint"], limit=1000)
        if "experience_id" in arguments:
            self.experience_id = _clean(arguments["experience_id"], limit=120)
        if "experience_searches" in arguments:
            value = arguments["experience_searches"]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("experience_searches must be a non-negative integer")
            self.experience_searches = min(value, 2)
        if "experience_terms" in arguments:
            self.experience_terms = _clean_terms(arguments["experience_terms"], limit=MAX_EXPERIENCE_TERMS)
        if "failure_terms" in arguments:
            self.failure_terms = _clean_terms(arguments["failure_terms"], limit=MAX_FAILURE_TERMS)

    def observe_failure_event(
        self,
        *,
        source: str,
        category: str,
        summary: str,
        step: int,
        terms: list[str] | None = None,
        paths: list[str] | None = None,
    ) -> None:
        _validate_step(step)
        clean_terms = _clean_terms(terms or [], limit=MAX_FAILURE_TERMS)
        clean_paths = _clean_paths(paths or [], limit=MAX_TARGET_PATHS)
        event = FailureEvent(
            step=step,
            source=_clean(source, limit=80) or "unknown",
            category=_clean(category, limit=120) or "unknown",
            summary=_clean(summary, limit=620),
            terms=clean_terms,
            paths=clean_paths,
        )
        self.failure_events.append(event)
        del self.failure_events[:-MAX_FAILURE_EVENTS]
        if clean_terms:
            self.failure_terms = _clean_terms([*self.failure_terms, *clean_terms], limit=MAX_FAILURE_TERMS)

    def set_task_profile(self, task_type: str, candidate_files: list[str]) -> None:
        if task_type not in TASK_TYPES:
            task_type = "coding"
        self.task_type = task_type
        cleaned: list[str] = []
        seen: set[str] = set()
        for path in candidate_files:
            clean = _clean_path(path)
            if clean and clean not in seen:
                seen.add(clean)
                cleaned.append(clean)
            if len(cleaned) >= MAX_CANDIDATE_FILES:
                break
        self.candidate_files = cleaned

    def observe_tool(
        self,
        *,
        name: str,
        arguments: dict[str, Any],
        ok: bool,
        observation: str,
        step: int,
        workspace: str | Path,
    ) -> None:
        _validate_step(step)
        root = ensure_workspace(workspace)
        summary = _clean(observation)
        if not ok:
            self.phase = "verifying" if name in {"verify", "auto_verify"} else "exploring"
            _remember(self.observations, f"step {step}: {name} failed: {summary}", MAX_OBSERVATIONS)
            if name in {"verify", "auto_verify"}:
                self._clear_verification()
            return

        path = arguments.get("path")
        if name == "read_file" and isinstance(path, str):
            self._remember_file(_fingerprint(root, path, "read", step))
            self.phase = "exploring"
        elif name == "read_many":
            requests = arguments.get("requests")
            if isinstance(requests, list):
                for item in requests:
                    if isinstance(item, dict) and isinstance(item.get("path"), str):
                        self._remember_file(_fingerprint(root, item["path"], "read", step))
            self.phase = "exploring"
        elif name in {"write_file", "apply_patch"} and isinstance(path, str):
            had_read_evidence = self._has_read_evidence(path)
            record = _fingerprint(root, path, "modified", step)
            self._remember_file(record)
            if had_read_evidence:
                self._mark_stale(record.path)
            self._remember_modified_path(record.path)
            self.last_mutation_step = step
            self.phase = "modifying"
            self._clear_verification()
            _remember(self.changes, f"step {step}: {name} {record.path}", MAX_CHANGES)
        elif name in {
            "read_conversation_memory",
            "repo_graph_query",
            "repo_graph_neighborhood",
            "list_files",
            "search_text",
            "git_status",
            "git_diff",
        }:
            self.phase = "exploring"
            _remember(self.observations, f"step {step}: {name}: {summary}", MAX_OBSERVATIONS)
        elif name == "run_command":
            self.phase = "modifying"
            _remember(self.observations, f"step {step}: run_command: {summary}", MAX_OBSERVATIONS)

    def observe_workspace_changes(self, paths: list[str], step: int, workspace: str | Path) -> None:
        _validate_step(step)
        if not paths:
            return
        root = ensure_workspace(workspace)
        read_paths = {item.path for item in self.files if item.action == "read"}
        for path in sorted(set(paths)):
            record = _fingerprint(root, path, "modified", step)
            self._remember_file(record)
            if record.path in read_paths:
                self._mark_stale(record.path)
            self._remember_modified_path(record.path)
        self.last_mutation_step = step
        self.phase = "modifying"
        self._clear_verification()
        _remember(
            self.changes,
            f"step {step}: workspace changed {len(set(paths))} file(s): {', '.join(sorted(set(paths))[:8])}",
            MAX_CHANGES,
        )

    def observe_verification(
        self,
        result: VerificationResult | None,
        step: int,
        workspace: str | Path,
    ) -> None:
        _validate_step(step)
        if result is None:
            return
        summary = _clean(result.output)
        if result.ok:
            self.last_verification_step = step
            self.verification_command = _clean(result.command or "(auto-detect found no command)", limit=360)
            self.verification_summary = summary
            self.verified_files = [_fingerprint(workspace, path, "modified", step) for path in self.modified_paths]
            self.phase = "ready"
            _remember(
                self.verification,
                f"step {step}: {self.verification_command} -> {summary}",
                MAX_VERIFICATION,
            )
        else:
            self.phase = "verifying"
            self._clear_verification()
            _remember(
                self.verification,
                f"step {step}: verification failed: {summary}",
                MAX_VERIFICATION,
            )

    def completion_blockers(self, workspace: str | Path) -> list[str]:
        ensure_workspace(workspace)
        if self.modified_paths_overflow:
            return ["Modified path ledger overflow prevents complete verification."]
        if not self.modified_paths:
            return []
        if (
            self.last_mutation_step is None
            or self.last_verification_step is None
            or self.last_verification_step < self.last_mutation_step
        ):
            return ["A successful verification is required after the latest mutation."]
        current = [_fingerprint(workspace, path, "modified", self.last_verification_step) for path in self.modified_paths]
        if _fingerprint_values(current) != _fingerprint_values(self.verified_files):
            return ["Verified file fingerprints no longer match the current modified files."]
        return []

    def refresh_files(self, workspace: str | Path) -> None:
        root = ensure_workspace(workspace)
        for item in list(self.files):
            current = _fingerprint(root, item.path, item.action, item.step)
            if _single_fingerprint_value(current) != _single_fingerprint_value(item):
                self._mark_stale(item.path)

    def render(self, *, include_experience: bool = True) -> str:
        sections = [
            f"Task: {_clean(self.task, limit=1000)}",
            f"Task type: {self.task_type}",
            f"Phase: {self.phase}",
            f"Hypothesis: {self.hypothesis or '(none)'}",
            f"Next step: {self.next_step or '(none)'}",
            "Candidate files: " + (", ".join(self.candidate_files) or "(none)"),
            "Target paths: " + (", ".join(self.target_paths) or "(none)"),
            "Recent files: " + (", ".join(f"{item.path} ({item.action})" for item in self.files) or "(none)"),
            "Stale files: " + (", ".join(self.stale_files) or "(none)"),
            "Modified paths: "
            + (", ".join(self.modified_paths) or "(none)")
            + (" (overflow)" if self.modified_paths_overflow else ""),
            "Experience terms: " + (", ".join(self.experience_terms) or "(none)"),
            "Failure terms: " + (", ".join(self.failure_terms) or "(none)"),
            _render_failure_events(self.failure_events),
            _render_list("Changes", self.changes),
            _render_list("Observations", self.observations),
            _render_list("Verification", self.verification),
            (
                "Evidence gate: mutation step "
                f"{self.last_mutation_step if self.last_mutation_step is not None else '(none)'}, "
                "verification step "
                f"{self.last_verification_step if self.last_verification_step is not None else '(none)'}"
            ),
            f"Verification command: {self.verification_command or '(none)'}",
            f"Verification summary: {self.verification_summary or '(none)'}",
        ]
        if include_experience:
            sections.extend(
                [
                    f"Experience hint: {self.experience_hint or '(none)'}",
                    f"Experience ID: {self.experience_id or '(none)'}",
                    f"Experience searches: {self.experience_searches}",
                ]
            )
        return "\n".join(sections)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkingMemory":
        memory = cls(task=_clean(data.get("task", "")))
        memory.files = [FileMemory(**item) for item in data.get("files", [])[:MAX_FILES]]
        memory.observations = [_clean(item) for item in data.get("observations", [])[:MAX_OBSERVATIONS]]
        memory.changes = [_clean(item) for item in data.get("changes", [])[:MAX_CHANGES]]
        memory.verification = [_clean(item) for item in data.get("verification", [])[:MAX_VERIFICATION]]
        memory.stale_files = [_clean(item, limit=240) for item in data.get("stale_files", [])[:MAX_FILES]]
        task_type = data.get("task_type", "coding")
        memory.task_type = task_type if task_type in TASK_TYPES else "coding"
        memory.candidate_files = [
            _clean_path(item) for item in data.get("candidate_files", [])[:MAX_CANDIDATE_FILES]
        ]
        memory.candidate_files = [item for item in memory.candidate_files if item]
        memory.phase = data.get("phase", "exploring") if data.get("phase") in PHASES else "exploring"
        memory.hypothesis = _clean(data.get("hypothesis", ""))
        memory.next_step = _clean(data.get("next_step", ""))
        memory.target_paths = [_clean(item, limit=240) for item in data.get("target_paths", [])[:MAX_TARGET_PATHS]]
        memory.modified_paths = [_clean(item, limit=240) for item in data.get("modified_paths", [])[:MAX_MODIFIED_PATHS]]
        memory.modified_paths_overflow = bool(data.get("modified_paths_overflow", False))
        memory.last_mutation_step = _optional_int(data.get("last_mutation_step"))
        memory.last_verification_step = _optional_int(data.get("last_verification_step"))
        memory.verification_command = _clean(data.get("verification_command", ""), limit=360)
        memory.verification_summary = _clean(data.get("verification_summary", ""))
        memory.verified_files = [
            FileMemory(**item) for item in data.get("verified_files", [])[:MAX_MODIFIED_PATHS]
        ]
        memory.experience_hint = _clean(data.get("experience_hint", ""), limit=1000)
        memory.experience_id = _clean(data.get("experience_id", ""), limit=120)
        memory.experience_searches = min(max(_optional_int(data.get("experience_searches")) or 0, 0), 2)
        memory.experience_terms = _clean_terms(data.get("experience_terms", []), limit=MAX_EXPERIENCE_TERMS)
        memory.failure_terms = _clean_terms(data.get("failure_terms", []), limit=MAX_FAILURE_TERMS)
        memory.failure_events = _clean_failure_events(data.get("failure_events", []))
        return memory

    def _remember_file(self, record: FileMemory) -> None:
        self.files = [item for item in self.files if item.path != record.path]
        self.files.append(record)
        del self.files[:-MAX_FILES]
        if record.action == "read":
            self.stale_files = [item for item in self.stale_files if item != record.path]

    def _has_read_evidence(self, path: str) -> bool:
        clean = str(path).replace("\\", "/").strip()
        if clean.startswith("./"):
            clean = clean[2:]
        return any(item.path == clean and item.action == "read" for item in self.files)

    def _mark_stale(self, path: str) -> None:
        clean = _clean(path, limit=240)
        if clean and clean not in self.stale_files:
            self.stale_files.append(clean)
            del self.stale_files[:-MAX_FILES]

    def _remember_modified_path(self, path: str) -> None:
        if path in self.modified_paths:
            self.modified_paths.remove(path)
            self.modified_paths.append(path)
        elif len(self.modified_paths) < MAX_MODIFIED_PATHS:
            self.modified_paths.append(path)
        else:
            self.modified_paths_overflow = True

    def _clear_verification(self) -> None:
        self.last_verification_step = None
        self.verification_command = ""
        self.verification_summary = ""
        self.verified_files = []


def _fingerprint(workspace: str | Path, path: str, action: str, step: int) -> FileMemory:
    root = ensure_workspace(workspace)
    target = safe_path(root, path)
    rel = target.relative_to(root).as_posix()
    if not target.exists() or target.is_dir():
        return FileMemory(rel, action, step, size=None, sha256=None)
    raw = target.read_bytes()
    return FileMemory(rel, action, step, size=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def _fingerprint_values(items: list[FileMemory]) -> list[tuple[str, int | None, str | None]]:
    return sorted((item.path, item.size, item.sha256) for item in items)


def _single_fingerprint_value(item: FileMemory) -> tuple[str, int | None, str | None]:
    return item.path, item.size, item.sha256


def _remember(items: list[str], value: str, limit: int) -> None:
    clean = _clean(value)
    if not clean:
        return
    key = clean.casefold()
    existing = [item for item in items if item.casefold() != key]
    existing.append(clean)
    del existing[:-limit]
    items[:] = existing


def _render_list(title: str, values: list[str]) -> str:
    if not values:
        return f"{title}: (none)"
    return title + ":\n" + "\n".join(f"- {value}" for value in values)


def _render_failure_events(values: list[FailureEvent]) -> str:
    if not values:
        return "Failure events: (none)"
    lines = ["Failure events:"]
    for event in values:
        details: list[str] = []
        if event.paths:
            details.append("paths=" + ", ".join(event.paths[:4]))
        if event.terms:
            details.append("terms=" + ", ".join(event.terms[:4]))
        suffix = f" ({'; '.join(details)})" if details else ""
        summary = event.summary or "(no summary)"
        lines.append(f"- step {event.step}: [{event.source}/{event.category}]{suffix} {summary}")
    return "\n".join(lines)


def _clean_terms(values: Any, *, limit: int) -> list[str]:
    if not isinstance(values, list):
        raise ValueError("term fields must be a list")
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _clean(item, limit=MAX_TERM_TEXT)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _clean_paths(values: Any, *, limit: int) -> list[str]:
    if not isinstance(values, list):
        raise ValueError("path fields must be a list")
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        path = _clean_path(item)
        if not path:
            continue
        key = path.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
        if len(out) >= limit:
            break
    return out


def _clean_failure_events(values: Any) -> list[FailureEvent]:
    if not isinstance(values, list):
        return []
    events: list[FailureEvent] = []
    for item in values[-MAX_FAILURE_EVENTS:]:
        if not isinstance(item, dict):
            continue
        step = _optional_int(item.get("step"))
        if step is None:
            continue
        try:
            terms = _clean_terms(item.get("terms", []), limit=MAX_FAILURE_TERMS)
            paths = _clean_paths(item.get("paths", []), limit=MAX_TARGET_PATHS)
        except ValueError:
            terms = []
            paths = []
        events.append(
            FailureEvent(
                step=step,
                source=_clean(item.get("source", ""), limit=80) or "unknown",
                category=_clean(item.get("category", ""), limit=120) or "unknown",
                summary=_clean(item.get("summary", ""), limit=620),
                terms=terms,
                paths=paths,
            )
        )
    return events


def _clean(value: Any, *, limit: int = MAX_TEXT) -> str:
    text = redact_secrets(str(value)).strip()
    if len(text) <= limit:
        return text
    marker = "... omitted ..."
    keep = max(0, limit - len(marker))
    head = (keep + 1) // 2
    tail = keep - head
    return f"{text[:head]}{marker}{text[-tail:] if tail else ''}"


def _clean_path(value: Any) -> str:
    text = _clean(value, limit=240).replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    return text


def _validate_step(step: int) -> None:
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a non-negative integer")


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def memory_to_json(memory: WorkingMemory) -> str:
    return json.dumps(memory.to_dict(), ensure_ascii=False, indent=2)
