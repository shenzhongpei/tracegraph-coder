from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal


Role = Literal["system", "user", "assistant", "tool"]
ROLES = {"system", "user", "assistant", "tool"}


@dataclass(slots=True)
class Message:
    role: Role
    content: str
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "metadata": deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "Message":
        if not isinstance(data, dict):
            raise ValueError("message payload must be an object")
        role = data.get("role")
        if role not in ROLES:
            raise ValueError(f"invalid message role: {role!r}")
        content = data.get("content")
        if not isinstance(content, str):
            raise ValueError("message content must be a string")
        metadata = data.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("message metadata must be an object")
        return cls(role=role, content=content, metadata=deepcopy(metadata))

    def to_wire(self) -> dict[str, Any]:
        if self.role == "assistant" and self.metadata and self.metadata.get("tool_calls"):
            return {
                "role": "assistant",
                "content": self.content or None,
                "tool_calls": self.metadata["tool_calls"],
            }
        if self.role == "tool" and self.metadata:
            return {
                "role": "tool",
                "content": self.content,
                "tool_call_id": self.metadata.get("tool_call_id", ""),
            }
        return {"role": self.role, "content": self.content}


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    @classmethod
    def from_wire(cls, item: dict[str, Any], fallback_index: int = 0) -> "ToolCall":
        fn = item.get("function", {})
        raw_args = fn.get("arguments", "{}")
        if isinstance(raw_args, str):
            args = json.loads(raw_args or "{}")
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            raise ValueError(f"Tool arguments must be JSON object, got {type(raw_args).__name__}")
        if not isinstance(args, dict):
            raise ValueError("Tool arguments must decode to a JSON object")
        return cls(
            id=str(item.get("id") or f"call_{fallback_index}"),
            name=str(fn.get("name") or item.get("name") or ""),
            arguments=args,
        )


@dataclass(slots=True)
class LLMResponse:
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    raw: dict[str, Any] | None = None


@dataclass(slots=True)
class ToolResult:
    ok: bool
    data: Any = None
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def serialize(self, limit: int = 50000) -> str:
        payload = {
            "ok": self.ok,
            "data": self.data if self.ok else None,
            "error": self.error if not self.ok else None,
            "meta": self.meta,
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if len(text) > limit:
            text = text[:limit] + "\n...[truncated]"
        return text


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Any
    read_only: bool = True
    timeout: float | None = None

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
