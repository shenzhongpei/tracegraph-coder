from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from .models import Message
from .safety import ensure_workspace, redact_secrets
from .working_memory import WorkingMemory


SCHEMA_VERSION = 5
SUPPORTED_SCHEMA_VERSIONS = {1, 2, 3, 4, SCHEMA_VERSION}
MESSAGE_BLOB_THRESHOLD_CHARS = 8_000
MESSAGE_BLOB_METADATA_KEY = "content_blob"


@dataclass
class RunSession:
    session_id: str
    workspace_root: str
    created_at: str
    updated_at: str
    status: str
    task: str
    model: str
    final_text: str
    report_path: str
    verification: str
    working_memory: dict[str, Any]
    messages: list[dict[str, Any]]
    iterations: int = 0
    tree_id: str = ""
    parent_id: str | None = None
    event_type: str = "checkpoint"
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = SCHEMA_VERSION
        return _redact_obj(payload)

    @classmethod
    def from_dict(cls, data: Any) -> "RunSession":
        if not isinstance(data, dict):
            raise ValueError("session payload must be an object")
        version = data.get("schema_version")
        if version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"unsupported session schema: {version!r}")
        required = {
            "session_id",
            "workspace_root",
            "created_at",
            "updated_at",
            "status",
            "task",
            "model",
            "final_text",
            "report_path",
            "verification",
            "working_memory",
        }
        missing = sorted(required - set(data))
        if missing:
            raise ValueError("session is missing field(s): " + ", ".join(missing))
        memory = data["working_memory"]
        if not isinstance(memory, dict):
            raise ValueError("working_memory must be an object")
        messages = data.get("messages", [])
        if messages is None:
            messages = []
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        for item in messages:
            Message.from_dict(item)
        session_id = _string(data["session_id"])
        status = _string(data["status"])
        event_type = _optional_string(data.get("event_type")) or ("completed" if status == "completed" else "checkpoint")
        return cls(
            session_id=session_id,
            workspace_root=_string(data["workspace_root"]),
            created_at=_string(data["created_at"]),
            updated_at=_string(data["updated_at"]),
            status=status,
            task=_string(data["task"]),
            model=_string(data["model"]),
            final_text=_string(data["final_text"]),
            report_path=_string(data["report_path"]),
            verification=_string(data["verification"]),
            working_memory=memory,
            messages=messages,
            iterations=_optional_int(data.get("iterations")) or 0,
            tree_id=_optional_string(data.get("tree_id")) or session_id,
            parent_id=_optional_string(data.get("parent_id")),
            event_type=event_type,
            summary=_optional_string(data.get("summary")) or "",
        )


class SessionStore:
    def __init__(self, workspace: str | Path, root: str | Path | None = None):
        self.workspace = ensure_workspace(workspace)
        base = Path(root).expanduser().resolve() if root is not None else _default_state_root()
        self.directory = base / _workspace_id(self.workspace) / "sessions"

    def save_run(
        self,
        *,
        task: str,
        model: str,
        final_text: str,
        report_path: str | Path,
        verification: str,
        working_memory: WorkingMemory,
        messages: list[Message] | None = None,
        iterations: int = 0,
        status: str = "completed",
    ) -> RunSession:
        return self.create_checkpoint(
            task=task,
            model=model,
            working_memory=working_memory,
            messages=messages or [],
            status=status,
            final_text=final_text,
            report_path=report_path,
            verification=verification,
            iterations=iterations,
        )

    def create_checkpoint(
        self,
        *,
        task: str,
        model: str,
        working_memory: WorkingMemory,
        messages: list[Message],
        status: str = "running",
        final_text: str = "",
        report_path: str | Path = "",
        verification: str = "",
        iterations: int = 0,
        tree_id: str | None = None,
        parent_id: str | None = None,
        event_type: str = "root",
        summary: str = "",
    ) -> RunSession:
        now = _now()
        session_id = _new_session_id()
        session = RunSession(
            session_id=session_id,
            workspace_root=str(self.workspace),
            created_at=now,
            updated_at=now,
            status=status,
            task=task,
            model=model,
            final_text=final_text,
            report_path=str(report_path),
            verification=verification,
            working_memory=working_memory.to_dict(),
            messages=_messages_to_dict(messages),
            iterations=max(0, iterations),
            tree_id=_safe_session_id(tree_id) if tree_id else session_id,
            parent_id=_safe_session_id(parent_id) if parent_id else None,
            event_type=_clean_event_type(event_type),
            summary=redact_secrets(str(summary)),
        )
        self.save(session)
        return session

    def update_checkpoint(
        self,
        session_id: str,
        *,
        task: str,
        model: str,
        working_memory: WorkingMemory,
        messages: list[Message],
        status: str = "running",
        final_text: str = "",
        report_path: str | Path = "",
        verification: str = "",
        iterations: int = 0,
    ) -> RunSession:
        previous = self.load(session_id, include_blobs=False)
        now = _now()
        session = RunSession(
            session_id=previous.session_id,
            workspace_root=str(self.workspace),
            created_at=previous.created_at,
            updated_at=now,
            status=status,
            task=task,
            model=model,
            final_text=final_text,
            report_path=str(report_path),
            verification=verification,
            working_memory=working_memory.to_dict(),
            messages=_messages_to_dict(messages),
            iterations=max(0, iterations),
            tree_id=previous.tree_id or previous.session_id,
            parent_id=previous.parent_id,
            event_type=previous.event_type,
            summary=_checkpoint_summary(status=status, iterations=iterations, messages=messages),
        )
        self.save(session)
        return session

    def save(self, session: RunSession) -> None:
        if Path(session.workspace_root).resolve() != self.workspace:
            raise ValueError("session belongs to a different workspace")
        self.directory.mkdir(parents=True, exist_ok=True)
        session_id = _safe_session_id(session.session_id)
        session_dir = self._session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        data = session.to_dict()
        referenced_blobs = _externalize_large_message_contents(data, session_dir)
        payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
        target = self.directory / f"{session_id}.json"
        temporary_name = ""
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self.directory,
                prefix=f".{session_id}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_name = stream.name
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, target)
            _remove_unreferenced_blobs(session_dir / "message_blobs", referenced_blobs)
        finally:
            if temporary_name and Path(temporary_name).exists():
                try:
                    Path(temporary_name).unlink()
                except OSError:
                    pass

    def load(self, selector: str = "latest", *, include_blobs: bool = True) -> RunSession:
        path = self._path_for_selector(selector)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"session could not be loaded: {exc}") from exc
        session = RunSession.from_dict(data)
        if Path(session.workspace_root).resolve() != self.workspace:
            raise ValueError("session belongs to a different workspace")
        if include_blobs:
            _hydrate_message_blobs(session, self._session_dir(session.session_id))
        WorkingMemory.from_dict(session.working_memory)
        return session

    def list_recent(self, limit: int = 10, *, include_blobs: bool = False) -> list[RunSession]:
        if not self.directory.exists():
            return []
        sessions: list[RunSession] = []
        candidates = sorted(
            self.directory.glob("*.json"),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
            reverse=True,
        )
        for path in candidates:
            try:
                sessions.append(self.load(path.stem, include_blobs=include_blobs))
            except ValueError:
                continue
            if len(sessions) >= limit:
                break
        return sessions

    def list_heads(self, limit: int = 10, *, include_blobs: bool = False) -> list[RunSession]:
        sessions = self.list_recent(limit=10_000, include_blobs=False)
        child_parent_ids = {session.parent_id for session in sessions if session.parent_id}
        heads = [session for session in sessions if session.session_id not in child_parent_ids]
        heads = heads[: max(0, limit)]
        if not include_blobs:
            return heads
        hydrated: list[RunSession] = []
        for session in heads:
            try:
                hydrated.append(self.load(session.session_id, include_blobs=True))
            except ValueError:
                continue
        return hydrated

    def list_children(self, session_id: str, *, include_blobs: bool = False) -> list[RunSession]:
        clean_id = _safe_session_id(session_id)
        children = [
            session
            for session in self.list_recent(limit=10_000, include_blobs=False)
            if session.parent_id == clean_id
        ]
        children.sort(key=lambda session: (session.created_at, session.session_id))
        if not include_blobs:
            return children
        hydrated: list[RunSession] = []
        for child in children:
            try:
                hydrated.append(self.load(child.session_id, include_blobs=True))
            except ValueError:
                continue
        return hydrated

    def lineage(self, session_id: str, *, include_blobs: bool = False) -> list[RunSession]:
        lineage: list[RunSession] = []
        seen: set[str] = set()
        current = self.load(session_id, include_blobs=include_blobs)
        while current.session_id not in seen:
            lineage.append(current)
            seen.add(current.session_id)
            if not current.parent_id:
                break
            try:
                current = self.load(current.parent_id, include_blobs=include_blobs)
            except ValueError:
                break
        return list(reversed(lineage))

    def fork_checkpoint(
        self,
        session_id: str,
        *,
        follow_up: str = "",
        status: str = "running",
        event_type: str = "fork",
    ) -> RunSession:
        source = self.load(session_id)
        messages = [Message.from_dict(item) for item in source.messages]
        if follow_up.strip():
            messages.append(Message(role="user", content=f"Follow-up request after branching:\n{follow_up.strip()}"))
        return self.create_checkpoint(
            task=source.task,
            model=source.model,
            working_memory=WorkingMemory.from_dict(source.working_memory),
            messages=messages,
            status=status,
            final_text=source.final_text,
            report_path=source.report_path,
            verification=source.verification,
            iterations=source.iterations,
            tree_id=source.tree_id or source.session_id,
            parent_id=source.session_id,
            event_type=event_type,
            summary=redact_secrets(follow_up.strip()),
        )

    def latest_resumable(self, *, include_blobs: bool = True) -> RunSession | None:
        for session in self.list_heads(limit=20, include_blobs=False):
            if is_resumable_session(session):
                return self.load(session.session_id, include_blobs=include_blobs) if include_blobs else session
        return None

    def _path_for_selector(self, selector: str) -> Path:
        if selector == "latest":
            candidates = sorted(
                self.directory.glob("*.json"),
                key=lambda path: (path.stat().st_mtime_ns, path.name),
            )
            if not candidates:
                raise ValueError("no saved sessions for this workspace")
            return candidates[-1]
        return self.directory / f"{_safe_session_id(selector)}.json"

    def _session_dir(self, session_id: str) -> Path:
        return self.directory / _safe_session_id(session_id)


def workspace_state_dir(workspace: str | Path, root: str | Path | None = None) -> Path:
    resolved = ensure_workspace(workspace)
    base = Path(root).expanduser().resolve() if root is not None else _default_state_root()
    return base / _workspace_id(resolved)


def is_resumable_session(session: RunSession) -> bool:
    return bool(session.messages)


def _default_state_root() -> Path:
    if os.name == "nt":
        base = os.getenv("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "TraceGraphCoder"
    base = os.getenv("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "tracegraph-coder"


def _workspace_id(workspace: Path) -> str:
    normalized = os.path.normcase(str(workspace.resolve()))
    return sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _safe_session_id(value: str) -> str:
    clean = "".join(ch for ch in str(value) if ch.isalnum() or ch in "._-")
    if not clean:
        raise ValueError("invalid session id")
    return clean[:120]


def _new_session_id() -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _string(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("session string field has invalid type")
    return redact_secrets(value)


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("session optional string field has invalid type")
    clean = redact_secrets(value).strip()
    return clean or None


def _clean_event_type(value: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value).strip().lower()).strip("._-")
    return clean[:80] or "checkpoint"


def _checkpoint_summary(*, status: str, iterations: int, messages: list[Message]) -> str:
    role = messages[-1].role if messages else "empty"
    return f"{status} conversation saved at iteration {max(0, iterations)} after {role} message"


def _messages_to_dict(messages: list[Message]) -> list[dict[str, Any]]:
    return [message.to_dict() for message in messages]


def _externalize_large_message_contents(
    data: dict[str, Any],
    session_dir: Path,
) -> set[str]:
    messages = data.get("messages")
    if not isinstance(messages, list):
        return set()
    blob_dir = session_dir / "message_blobs"
    referenced: set[str] = set()
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        metadata = message.get("metadata")
        if metadata is None:
            if len(content) <= MESSAGE_BLOB_THRESHOLD_CHARS:
                continue
            metadata = {}
            message["metadata"] = metadata
        elif not isinstance(metadata, dict):
            continue
        else:
            metadata.pop(MESSAGE_BLOB_METADATA_KEY, None)
        if len(content) <= MESSAGE_BLOB_THRESHOLD_CHARS:
            continue
        digest = sha256(content.encode("utf-8")).hexdigest()
        blob_name = f"message-{index:04d}-{digest[:16]}.txt"
        rel_path = f"message_blobs/{blob_name}"
        target = blob_dir / blob_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        metadata[MESSAGE_BLOB_METADATA_KEY] = {
            "path": rel_path,
            "sha256": digest,
            "chars": len(content),
        }
        message["content"] = _blob_placeholder(len(content))
        referenced.add(blob_name)
    return referenced


def _hydrate_message_blobs(session: RunSession, session_dir: Path) -> None:
    blob_root = (session_dir / "message_blobs").resolve()
    for message in session.messages:
        if not isinstance(message, dict):
            continue
        metadata = message.get("metadata")
        if not isinstance(metadata, dict):
            continue
        blob_spec = metadata.get(MESSAGE_BLOB_METADATA_KEY)
        if not isinstance(blob_spec, dict):
            continue
        rel_path = blob_spec.get("path")
        if not isinstance(rel_path, str) or not rel_path:
            raise ValueError("session message blob reference is invalid")
        target = (session_dir / rel_path).resolve()
        try:
            target.relative_to(blob_root)
        except ValueError as exc:
            raise ValueError("session message blob path escapes its directory") from exc
        if not target.exists():
            raise ValueError(f"session message blob is missing: {rel_path}")
        content = target.read_text(encoding="utf-8")
        expected_sha = blob_spec.get("sha256")
        if isinstance(expected_sha, str):
            actual_sha = sha256(content.encode("utf-8")).hexdigest()
            if actual_sha != expected_sha:
                raise ValueError(f"session message blob checksum mismatch: {rel_path}")
        message["content"] = content
        metadata.pop(MESSAGE_BLOB_METADATA_KEY, None)


def _remove_unreferenced_blobs(blob_dir: Path, referenced: set[str]) -> None:
    if not blob_dir.exists():
        return
    for path in blob_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(blob_dir).as_posix()
        if rel not in referenced:
            try:
                path.unlink()
            except OSError:
                pass


def _blob_placeholder(length: int) -> str:
    return f"[externalized message content: {length} chars]"


def _redact_obj(value: Any) -> Any:
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, list):
        return [_redact_obj(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_obj(item) for key, item in value.items()}
    return value
