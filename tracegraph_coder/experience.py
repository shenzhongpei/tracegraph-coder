from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .safety import ensure_workspace, redact_secrets, safe_path
from .session import workspace_state_dir
from .working_memory import FileMemory, WorkingMemory


SCHEMA_VERSION = 1
MAX_CARDS = 50
MAX_TEXT = 420
MAX_TERMS = 24
MAX_FILES = 64
GENERIC_TERMS = {
    "add",
    "bug",
    "change",
    "fix",
    "issue",
    "problem",
    "run",
    "test",
    "tests",
    "update",
    "verify",
}
GENERIC_CHINESE_BIGRAMS = {"修复", "测试", "验证", "运行", "问题", "错误", "失败"}


@dataclass(frozen=True, slots=True)
class ExperienceFile:
    path: str
    size: int | None
    sha256: str | None


@dataclass(frozen=True, slots=True)
class ExperienceCard:
    experience_id: str
    created_at: str
    task: str
    strategy: str
    trigger_terms: list[str]
    files: list[ExperienceFile]
    verification_command: str
    verification_summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> "ExperienceCard":
        if not isinstance(data, dict):
            raise ValueError("experience card must be an object")
        files = data.get("files")
        if not isinstance(files, list):
            raise ValueError("experience card files must be a list")
        return cls(
            experience_id=_string(data.get("experience_id")),
            created_at=_string(data.get("created_at")),
            task=_string(data.get("task")),
            strategy=_string(data.get("strategy")),
            trigger_terms=[_string(item) for item in data.get("trigger_terms", [])[:MAX_TERMS]],
            files=[ExperienceFile(**item) for item in files[:MAX_FILES] if isinstance(item, dict)],
            verification_command=_string(data.get("verification_command")),
            verification_summary=_string(data.get("verification_summary")),
        )


@dataclass(frozen=True, slots=True)
class ExperienceMatch:
    card: ExperienceCard
    score: float
    shared_terms: list[str]

    @property
    def experience_id(self) -> str:
        return self.card.experience_id

    def render_hint(self) -> str:
        files = ", ".join(item.path for item in self.card.files)
        priority_terms = ", ".join(self.shared_terms[:12]) or "(none)"
        return "\n".join(
            [
                f"Verified experience reference: {self.card.experience_id}",
                "Use it only as a hint; re-read current files and verify again before finishing.",
                f"Strategy: {self.card.strategy}",
                f"Priority terms: {priority_terms}",
                f"Files: {files}",
                f"Verification: {self.card.verification_command}",
                f"Verification summary: {self.card.verification_summary}",
            ]
        )


class ExperienceStore:
    def __init__(self, workspace: str | Path, root: str | Path | None = None):
        self.workspace = ensure_workspace(workspace)
        self.directory = workspace_state_dir(self.workspace, root) / "experiences"

    def add_verified(self, memory: WorkingMemory, strategy: str) -> ExperienceCard:
        blockers = memory.completion_blockers(self.workspace)
        if blockers or not memory.modified_paths or not memory.verified_files:
            raise ValueError("experience requires verified modified files")
        files = [_experience_file(self.workspace, item) for item in memory.verified_files]
        terms = _terms(memory.task, strategy, *(item.path for item in files))
        if len(terms) < 2:
            raise ValueError("experience needs at least two useful trigger terms")
        card = ExperienceCard(
            experience_id=f"exp-{uuid.uuid4().hex}",
            created_at=datetime.now(timezone.utc).isoformat(),
            task=_clip(memory.task),
            strategy=_clip(strategy),
            trigger_terms=terms[:MAX_TERMS],
            files=files,
            verification_command=_clip(memory.verification_command),
            verification_summary=_clip(memory.verification_summary),
        )
        cards = [item for item in self._read_cards() if _dedupe_key(item) != _dedupe_key(card)]
        cards.append(card)
        self._write_cards(cards[-MAX_CARDS:])
        return card

    def retrieve(self, query: str) -> ExperienceMatch | None:
        query_terms = _terms(query)
        if len(query_terms) < 2:
            return None
        eligible = [card for card in self._read_cards() if not self._stale_reason(card)]
        if not eligible:
            return None
        documents = [set(card.trigger_terms) for card in eligible]
        idf = _idf(documents)
        query_set = set(query_terms)
        query_weight = sum(idf.get(term, 1.0) for term in query_set)
        if query_weight <= 0:
            return None
        matches: list[ExperienceMatch] = []
        for card, doc in zip(eligible, documents):
            shared = sorted(query_set & doc)
            if len(shared) < 2:
                continue
            score = sum(idf.get(term, 1.0) for term in shared) / query_weight
            if score >= 0.45:
                matches.append(ExperienceMatch(card, score, shared))
        if not matches:
            return None
        return max(matches, key=lambda item: (item.score, item.card.created_at, item.card.experience_id))

    def list_status(self) -> list[dict[str, Any]]:
        return [
            {"card": card.to_dict(), "stale": bool(reason), "reason": reason}
            for card in self._read_cards()
            for reason in [self._stale_reason(card)]
        ]

    def _stale_reason(self, card: ExperienceCard) -> str:
        for item in card.files:
            try:
                current = _fingerprint_file(self.workspace, item.path)
            except OSError:
                return f"file is missing: {item.path}"
            if current.size != item.size or current.sha256 != item.sha256:
                return f"file fingerprint changed: {item.path}"
        return ""

    def _read_cards(self) -> list[ExperienceCard]:
        path = self.directory / "cards.json"
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"experience cards are corrupt: {exc}") from exc
        if payload.get("schema_version") != SCHEMA_VERSION or not isinstance(payload.get("cards"), list):
            raise ValueError("experience cards schema is incompatible")
        return [ExperienceCard.from_dict(item) for item in payload["cards"][:MAX_CARDS]]

    def _write_cards(self, cards: list[ExperienceCard]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"schema_version": SCHEMA_VERSION, "cards": [card.to_dict() for card in cards]},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        temporary_name = ""
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self.directory,
                prefix=".cards.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_name = stream.name
                stream.write(redact_secrets(payload))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, self.directory / "cards.json")
        finally:
            if temporary_name and Path(temporary_name).exists():
                try:
                    Path(temporary_name).unlink()
                except OSError:
                    pass


def _experience_file(workspace: Path, record: FileMemory) -> ExperienceFile:
    current = _fingerprint_file(workspace, record.path)
    if current.size != record.size or current.sha256 != record.sha256:
        raise ValueError(f"verified file fingerprint is stale: {record.path}")
    return current


def _fingerprint_file(workspace: Path, path: str) -> ExperienceFile:
    target = safe_path(workspace, path)
    raw = target.read_bytes()
    return ExperienceFile(
        path=target.relative_to(workspace).as_posix(),
        size=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _terms(*values: str) -> list[str]:
    terms: list[str] = []
    for value in values:
        text = str(value)
        terms.extend(token.casefold() for token in re.findall(r"[A-Za-z0-9]+(?:[_-][A-Za-z0-9]+)*", text))
        for run in re.findall(r"[\u4e00-\u9fff]+", text):
            if len(run) == 1:
                terms.append(run)
            else:
                terms.extend(run[index : index + 2] for index in range(len(run) - 1))
    out: list[str] = []
    for term in terms:
        if term in GENERIC_TERMS or term in GENERIC_CHINESE_BIGRAMS or len(term) > 80:
            continue
        if term not in out:
            out.append(term)
    return out


def extract_experience_terms(*values: str, limit: int = MAX_TERMS) -> list[str]:
    terms = _terms(*values)
    try:
        max_terms = max(0, int(limit))
    except (TypeError, ValueError):
        max_terms = MAX_TERMS
    return terms[:max_terms]


def _idf(documents: list[set[str]]) -> dict[str, float]:
    all_terms = sorted(set().union(*documents)) if documents else []
    out: dict[str, float] = {}
    for term in all_terms:
        count = sum(1 for doc in documents if term in doc)
        out[term] = math.log((1 + len(documents)) / (1 + count)) + 1
    return out


def _dedupe_key(card: ExperienceCard) -> tuple[str, tuple[str, ...]]:
    return (" ".join(card.trigger_terms[:8]), tuple(sorted(item.path for item in card.files)))


def _clip(value: Any, limit: int = MAX_TEXT) -> str:
    text = redact_secrets(str(value)).strip()
    if len(text) <= limit:
        return text
    marker = "... omitted ..."
    keep = max(0, limit - len(marker))
    head = (keep + 1) // 2
    tail = keep - head
    return f"{text[:head]}{marker}{text[-tail:] if tail else ''}"


def _string(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("experience string field has invalid type")
    return _clip(value)
