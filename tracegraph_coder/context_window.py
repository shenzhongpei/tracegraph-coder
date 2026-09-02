from __future__ import annotations

import copy
import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from .models import Message
from .safety import redact_secrets


CHARS_PER_TOKEN = 4
DEFAULT_CONTEXT_CHAR_BUDGET = 120_000
DEFAULT_COMPACT_TRIGGER_RATIO = 0.82
DEFAULT_KEEP_RECENT_GROUPS = 8
DEFAULT_MAX_TOOL_RESULT_CHARS = 1_600
MAX_ANCHOR_GROUPS = 6
MIN_TOOL_RESULT_CHARS = 160
MAX_COMPACTION_SUMMARY_CHARS = 6_000
MAX_TOOL_DIGEST_PREVIEW_CHARS = 520
IMPORTANT_TOOL_NAMES = {
    "apply_patch",
    "write_file",
    "run_command",
    "verify",
    "finish_task",
    "record_progress",
}
DEFAULT_CONTEXT_MIN_CHAR_BUDGET = 12_000
MIN_CALIBRATION_TOKENS = 1_000


@dataclass(frozen=True, slots=True)
class ContextWindowReport:
    original_chars: int
    final_chars: int
    compacted_tool_results: int = 0
    dropped_groups: int = 0
    original_tokens: int = 0
    final_tokens: int = 0
    preserved_anchor_groups: int = 0
    hard_compacted_messages: int = 0
    strategy: str = "none"
    trigger_chars: int = 0
    soft_compaction: bool = False

    @property
    def summarized_groups(self) -> int:
        return self.dropped_groups

    @property
    def compacted(self) -> bool:
        return bool(
            self.compacted_tool_results
            or self.dropped_groups
            or self.hard_compacted_messages
        )

    def format(self) -> str:
        before = self.original_tokens or estimate_tokens_from_chars(self.original_chars)
        after = self.final_tokens or estimate_tokens_from_chars(self.final_chars)
        return (
            f"context compacted: {before} -> {after} estimated tokens, "
            f"strategy={self.strategy}, tool_results={self.compacted_tool_results}, "
            f"summarized_groups={self.dropped_groups}, "
            f"anchors={self.preserved_anchor_groups}, "
            f"mode={'soft' if self.soft_compaction else 'limit'}"
        )


@dataclass(slots=True)
class ContextBudgetTuner:
    """Adapt the character budget using provider-reported prompt token usage."""

    base_max_chars: int = DEFAULT_CONTEXT_CHAR_BUDGET
    min_chars: int = DEFAULT_CONTEXT_MIN_CHAR_BUDGET
    calibration: float = 1.0
    observations: int = 0
    last_observed_prompt_tokens: int | None = None
    last_estimated_prompt_tokens: int | None = None

    def current_max_chars(self) -> int:
        base = max(1, int(self.base_max_chars))
        floor = min(base, max(1, int(self.min_chars)))
        adjusted = int(base / max(0.75, min(3.0, self.calibration)))
        return max(floor, min(base, adjusted))

    def observe(
        self,
        report: ContextWindowReport,
        usage: dict[str, Any] | None,
        *,
        tool_schema_tokens: int = 0,
    ) -> str | None:
        observed = prompt_tokens_from_usage(usage)
        if observed is None:
            return None
        estimated = max(1, report.final_tokens + max(0, int(tool_schema_tokens)))
        self.last_observed_prompt_tokens = observed
        self.last_estimated_prompt_tokens = estimated
        if estimated < MIN_CALIBRATION_TOKENS:
            return (
                "context usage: "
                f"observed_prompt_tokens={observed}, estimated_prompt_tokens={estimated}, "
                "calibration=skipped-small-sample"
            )
        ratio = max(0.5, min(3.0, observed / estimated))
        alpha = 0.35 if self.observations == 0 else 0.2
        self.calibration = (self.calibration * (1 - alpha)) + (ratio * alpha)
        self.calibration = max(0.75, min(3.0, self.calibration))
        self.observations += 1
        return (
            "context usage: "
            f"observed_prompt_tokens={observed}, estimated_prompt_tokens={estimated}, "
            f"calibration={self.calibration:.2f}, next_max_chars={self.current_max_chars()}"
        )


def prepare_context_messages(
    messages: list[Message],
    *,
    max_chars: int = DEFAULT_CONTEXT_CHAR_BUDGET,
    max_tokens: int | None = None,
    compact_trigger_ratio: float = DEFAULT_COMPACT_TRIGGER_RATIO,
    keep_recent_groups: int = DEFAULT_KEEP_RECENT_GROUPS,
    max_tool_result_chars: int = DEFAULT_MAX_TOOL_RESULT_CHARS,
    focus_terms: Iterable[str] | None = None,
) -> tuple[list[Message], ContextWindowReport]:
    """Build a bounded model view without mutating the durable transcript.

    The policy follows a virtual-memory pattern used by long-running agents:
    the immutable task contract stays hot, recent tool trajectory stays detailed,
    important older groups and semantic anchors from experience/failure markers
    are retained, and the remaining history is represented by a structured
    summary. Exact old messages remain in the session checkpoint and evidence log
    for recovery or later retrieval.
    """

    max_chars = max(1, int(max_chars))
    if max_tokens is not None:
        max_chars = min(max_chars, _chars_from_token_budget(max_tokens))
    compact_trigger_ratio = max(0.1, min(1.0, float(compact_trigger_ratio)))
    trigger_chars = max(1, int(max_chars * compact_trigger_ratio))
    keep_recent_groups = max(1, int(keep_recent_groups))
    max_tool_result_chars = max(MIN_TOOL_RESULT_CHARS, int(max_tool_result_chars))
    normalized_focus_terms = _normalize_focus_terms(focus_terms)
    original_chars = estimate_context_chars(messages)
    original_tokens = estimate_context_tokens(messages)
    if original_chars <= trigger_chars:
        view = [_clone_message(message) for message in messages]
        return view, ContextWindowReport(
            original_chars,
            original_chars,
            original_tokens=original_tokens,
            final_tokens=original_tokens,
            trigger_chars=trigger_chars,
        )

    prefix, groups = _split_prefix_and_groups(messages)
    if not groups:
        view, hard_count = _fit_message_contents(prefix, max_chars)
        return view, ContextWindowReport(
            original_chars,
            estimate_context_chars(view),
            original_tokens=original_tokens,
            final_tokens=estimate_context_tokens(view),
            hard_compacted_messages=hard_count,
            strategy="hard-fit",
            trigger_chars=trigger_chars,
            soft_compaction=original_chars <= max_chars,
        )

    recent_start = max(0, len(groups) - keep_recent_groups)
    recent_indices = set(range(recent_start, len(groups)))
    soft_only = original_chars <= max_chars
    anchor_indices = _anchor_indices(groups, recent_indices, normalized_focus_terms)
    selected_indices = (
        list(range(len(groups)))
        if soft_only
        else sorted(recent_indices | anchor_indices)
    )
    summary_limit = max(480, min(MAX_COMPACTION_SUMMARY_CHARS, max_chars // 4))

    def render(*, compact_recent: bool, tool_limit: int, include_summary: bool) -> tuple[list[Message], int, int]:
        omitted = [group for index, group in enumerate(groups) if index not in selected_indices]
        body: list[Message] = []
        compacted_tools = 0
        for index in _ordered_indices(selected_indices, recent_indices):
            group = groups[index]
            should_compact = compact_recent or index < recent_start
            if should_compact:
                compacted_group, count = _compact_group(group, tool_limit, normalized_focus_terms)
                body.extend(compacted_group)
                compacted_tools += count
            else:
                body.extend(_clone_message(message) for message in group)
        view = _clone_message_list(prefix)
        if include_summary and omitted:
            view.append(_summary_message(omitted, limit=summary_limit, focus_terms=normalized_focus_terms))
        view.extend(body)
        return view, compacted_tools, len(omitted) if include_summary else 0

    view, compacted_tools, dropped_groups = render(
        compact_recent=False,
        tool_limit=max_tool_result_chars,
        include_summary=not soft_only,
    )
    strategy_parts = ["semantic-anchors", "recent"]
    if original_chars <= max_chars:
        strategy_parts.append("soft-tool-digest")

    if estimate_context_chars(view) > max_chars:
        view, compacted_tools, dropped_groups = render(
            compact_recent=True,
            tool_limit=max_tool_result_chars,
            include_summary=True,
        )
        strategy_parts.append("tool-digest")

    # Prefer the recent window, but retain older mutation/verification/error
    # anchors while there is enough room for them.
    removable_anchors = sorted(anchor_indices - recent_indices)
    while estimate_context_chars(view) > max_chars and removable_anchors:
        selected_indices.remove(removable_anchors.pop(0))
        view, compacted_tools, dropped_groups = render(
            compact_recent=True,
            tool_limit=max_tool_result_chars,
            include_summary=True,
        )
        strategy_parts.append("anchor-prune")

    # If the recent window itself is large, progressively keep only its newest
    # groups. The last group is always retained so a pending tool exchange is not
    # discarded completely.
    recent_to_prune = sorted(recent_indices)[:-1]
    while estimate_context_chars(view) > max_chars and recent_to_prune:
        selected_indices.remove(recent_to_prune.pop(0))
        view, compacted_tools, dropped_groups = render(
            compact_recent=True,
            tool_limit=max_tool_result_chars,
            include_summary=True,
        )
        strategy_parts.append("recent-prune")

    # A second, more aggressive observation pass is cheaper and safer than
    # discarding the latest tool exchange outright.
    for tool_limit in _compression_limits(max_tool_result_chars):
        if estimate_context_chars(view) <= max_chars:
            break
        view, compacted_tools, dropped_groups = render(
            compact_recent=True,
            tool_limit=tool_limit,
            include_summary=True,
        )
        strategy_parts.append("deep-tool-digest")

    hard_count = 0
    if estimate_context_chars(view) > max_chars:
        view, hard_count = _fit_message_contents(view, max_chars)
        strategy_parts.append("hard-fit")

    final_chars = estimate_context_chars(view)
    return view, ContextWindowReport(
        original_chars=original_chars,
        final_chars=final_chars,
        compacted_tool_results=compacted_tools,
        dropped_groups=dropped_groups,
        original_tokens=original_tokens,
        final_tokens=estimate_context_tokens(view),
        preserved_anchor_groups=len(set(selected_indices) & anchor_indices),
        hard_compacted_messages=hard_count,
        strategy="+".join(dict.fromkeys(strategy_parts)),
        trigger_chars=trigger_chars,
        soft_compaction=original_chars <= max_chars,
    )


def estimate_context_chars(messages: list[Message]) -> int:
    payload = [message.to_wire() for message in messages]
    return len(json.dumps(payload, ensure_ascii=False, default=str))


def estimate_tokens_from_chars(chars: int) -> int:
    return max(1, math.ceil(max(0, chars) / CHARS_PER_TOKEN))


def _chars_from_token_budget(tokens: int) -> int:
    return max(1, int(max(1, tokens) * CHARS_PER_TOKEN))


def estimate_context_tokens(messages: list[Message]) -> int:
    """Estimate tokens without requiring a provider-specific tokenizer."""

    payload = json.dumps(
        [message.to_wire() for message in messages],
        ensure_ascii=False,
        default=str,
    )
    return _estimate_text_tokens(payload)


def estimate_tool_schema_tokens(tools: list[dict[str, Any]]) -> int:
    if not tools:
        return 0
    payload = json.dumps(tools, ensure_ascii=False, default=str)
    return _estimate_text_tokens(payload)


def prompt_tokens_from_usage(usage: dict[str, Any] | None) -> int | None:
    if not isinstance(usage, dict):
        return None
    for key in ("prompt_tokens", "input_tokens"):
        value = usage.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _estimate_text_tokens(text: str) -> int:
    weighted_chars = 0.0
    for character in text:
        weighted_chars += 1 / 1.5 if ord(character) > 127 else 1 / CHARS_PER_TOKEN
    return max(1, math.ceil(weighted_chars))


def _split_prefix_and_groups(messages: list[Message]) -> tuple[list[Message], list[list[Message]]]:
    prefix_end = 0
    while prefix_end < len(messages) and messages[prefix_end].role in {"system", "user"}:
        prefix_end += 1
    prefix = _clone_message_list(messages[:prefix_end])
    groups: list[list[Message]] = []
    index = prefix_end
    while index < len(messages):
        current = messages[index]
        group = [_clone_message(current)]
        index += 1
        if current.role == "assistant" and _has_tool_calls(current):
            expected_tools = len(current.metadata.get("tool_calls", [])) if current.metadata else 0
            while index < len(messages) and messages[index].role == "tool" and len(group) <= expected_tools:
                group.append(_clone_message(messages[index]))
                index += 1
        groups.append(group)
    return prefix, groups


def _anchor_indices(
    groups: list[list[Message]],
    recent_indices: set[int],
    focus_terms: tuple[str, ...],
) -> set[int]:
    scored: list[tuple[int, int]] = []
    for index, group in enumerate(groups):
        if index in recent_indices:
            continue
        names = set(_group_tool_names(group))
        score = 0
        if names & IMPORTANT_TOOL_NAMES:
            score += 100
        if names & {"apply_patch", "write_file", "run_command"}:
            score += 30
        if names & {"verify", "finish_task"}:
            score += 26
        if any(_looks_like_failure(message.content) for message in group):
            score += 24
        if any(message.role == "user" for message in group):
            score += 8
        if focus_terms and _contains_focus_term(group, focus_terms):
            score += 48
        if score:
            scored.append((score, index))
    scored.sort(key=lambda item: (-item[0], -item[1]))
    return {index for _score, index in scored[:MAX_ANCHOR_GROUPS]}


def _group_tool_names(group: list[Message]) -> list[str]:
    return [name for name, _arguments in _group_tool_call_specs(group)]


def _group_tool_call_specs(group: list[Message]) -> list[tuple[str, dict[str, Any]]]:
    specs: list[tuple[str, dict[str, Any]]] = []
    for message in group:
        if not message.metadata:
            continue
        raw_calls = message.metadata.get("tool_calls")
        if not isinstance(raw_calls, list):
            continue
        for raw_call in raw_calls:
            if not isinstance(raw_call, dict):
                continue
            function = raw_call.get("function")
            if isinstance(function, dict):
                name = str(function.get("name") or "").strip()
                arguments = _parse_tool_arguments(function.get("arguments"))
            else:
                name = str(raw_call.get("name") or "").strip()
                arguments = _parse_tool_arguments(raw_call.get("arguments"))
            if name:
                specs.append((name, arguments))
    return specs


def _parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return dict(arguments)
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _tool_call_label(name: str, arguments: dict[str, Any]) -> str:
    pieces = [name]
    for key in ("path", "file_path", "file_pattern", "query", "command"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            pieces.append(f"{key}={_clip_middle(value.strip(), 80)}")
            break
    paths = arguments.get("paths")
    if len(pieces) == 1 and isinstance(paths, list):
        clean_paths = [str(item).strip() for item in paths if str(item).strip()]
        if clean_paths:
            pieces.append("paths=" + ", ".join(_clip_middle(path, 48) for path in clean_paths[:3]))
    return " ".join(redact_secrets(piece) for piece in pieces)


def _tool_result_digest(
    content: str,
    *,
    limit: int,
    focus_terms: tuple[str, ...],
    tool_name: str,
    tool_args: dict[str, Any],
) -> str:
    ok, body, meta = _parse_tool_result(content)
    context = _tool_call_label(tool_name or "tool", tool_args) if tool_name or tool_args else "tool"
    body = redact_secrets(body)
    keep_preview = bool(_matched_focus_terms(body, focus_terms)) or ok is False or _looks_like_failure(body)
    preview = ""
    if keep_preview and body:
        preview_limit = min(max(MIN_TOOL_RESULT_CHARS, limit), MAX_TOOL_DIGEST_PREVIEW_CHARS)
        preview = _focus_preview(body, preview_limit, focus_terms)
        preview = _clip_middle(" ".join(preview.split()), preview_limit)
    else:
        preview = _tool_result_shape_hint(body)
    omitted = max(0, len(content) - len(preview))
    lines = [
        "[Tool result compacted]",
        f"{context}",
        f"ok={ok if ok is not None else 'unknown'} omitted_chars={omitted}",
    ]
    if isinstance(meta, dict) and meta:
        lines.append("meta=" + _clip_middle(json.dumps(meta, ensure_ascii=False, default=str), 220))
    if preview:
        label = "focused_preview" if keep_preview else "result_shape"
        lines.append(f"{label}: {preview}")
    lines.append("Full result remains in durable transcript/evidence log; re-read exact file/range if needed.")
    return _clip_middle("\n".join(lines), max(limit + 320, MIN_TOOL_RESULT_CHARS))


def _tool_result_brief(content: str, focus_terms: tuple[str, ...], limit: int) -> str:
    ok, body, _meta = _parse_tool_result(redact_secrets(content))
    matched = _matched_focus_terms(body, focus_terms)
    if matched or ok is False or _looks_like_failure(body):
        preview = _focus_preview(body, min(limit, MAX_TOOL_DIGEST_PREVIEW_CHARS), focus_terms)
    else:
        preview = _tool_result_shape_hint(body)
    text = f"ok={ok if ok is not None else 'unknown'}; {preview}"
    return _clip_middle(" ".join(text.split()), limit)


def _parse_tool_result(content: str) -> tuple[bool | None, str, dict[str, Any]]:
    text = str(content or "")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None, text, {}
    if not isinstance(payload, dict):
        return None, text, {}
    ok = payload.get("ok")
    ok_value = ok if isinstance(ok, bool) else None
    if ok_value is False:
        body = payload.get("error", payload.get("data", ""))
    else:
        body = payload.get("data", payload.get("error", ""))
    meta = payload.get("meta")
    return ok_value, str(body or ""), meta if isinstance(meta, dict) else {}


def _matched_focus_terms(content: str, focus_terms: tuple[str, ...]) -> list[str]:
    if not focus_terms:
        return []
    lowered = str(content or "").lower()
    return [term for term in focus_terms if term and term in lowered]


def _tool_result_shape_hint(content: str) -> str:
    text = str(content or "")
    if not text.strip():
        return "(empty result)"
    lines = text.splitlines()
    first_line = next((line.strip() for line in lines if line.strip()), "")
    hint = f"{len(lines)} line(s), {len(text)} chars"
    if first_line:
        hint += f"; starts: {_clip_middle(first_line, 140)}"
    return hint


def _has_tool_calls(message: Message) -> bool:
    return bool(message.metadata and message.metadata.get("tool_calls"))


def _compact_group(
    group: list[Message],
    limit: int,
    focus_terms: tuple[str, ...],
) -> tuple[list[Message], int]:
    rewritten: list[Message] = []
    count = 0
    tool_specs = _group_tool_call_specs(group)
    tool_index = 0
    for message in group:
        if message.role == "tool" and len(message.content) > limit:
            tool_name = ""
            tool_args: dict[str, Any] = {}
            if tool_specs:
                tool_name, tool_args = tool_specs[min(tool_index, len(tool_specs) - 1)]
            rewritten.append(_compact_tool_message(message, limit, focus_terms, tool_name, tool_args))
            count += 1
        else:
            rewritten.append(_clone_message(message))
        if message.role == "tool":
            tool_index += 1
    return rewritten, count


def _compact_tool_message(
    message: Message,
    limit: int,
    focus_terms: tuple[str, ...] = (),
    tool_name: str = "",
    tool_args: dict[str, Any] | None = None,
) -> Message:
    content = redact_secrets(message.content)
    replacement = _tool_result_digest(
        content,
        limit=limit,
        focus_terms=focus_terms,
        tool_name=tool_name,
        tool_args=tool_args or {},
    )
    return _clone_message(message, content=replacement)


def _summary_message(
    groups: list[list[Message]],
    *,
    limit: int = MAX_COMPACTION_SUMMARY_CHARS,
    focus_terms: tuple[str, ...] = (),
) -> Message:
    actions: Counter[str] = Counter()
    paths: list[str] = []
    path_set: set[str] = set()
    rows: list[str] = []
    for group in groups:
        names = _group_tool_names(group)
        actions.update(names)
        path = _path_from_group(group)
        if path and path not in path_set:
            path_set.add(path)
            paths.append(path)
        row = _summarize_group(group, focus_terms)
        if row:
            rows.append(row)

    action_text = ", ".join(f"{name} x{count}" for name, count in actions.most_common(12)) or "(none)"
    path_text = ", ".join(paths[:16]) or "(none)"
    lines = [
        "[Context compaction packet: compacted history]",
        "Layer: compressed conversation history",
        f"- omitted trajectory groups: {len(groups)}",
        f"- older tool activity: {action_text}",
        f"- paths mentioned by older groups: {path_text}",
        "- exact old messages remain in the durable session transcript and evidence log",
        "- trust working memory plus this packet for routing; re-read exact file ranges before editing",
    ]
    if focus_terms:
        lines.append("- semantic anchors: " + ", ".join(focus_terms[:16]))
    if rows:
        lines.extend(["", "Older trajectory highlights:", *rows[:18]])
    content = _clip_middle(redact_secrets("\n".join(lines)), limit)
    return Message(role="system", content=content)


def _summarize_group(group: list[Message], focus_terms: tuple[str, ...] = ()) -> str:
    specs = _group_tool_call_specs(group)
    if specs:
        label = ", ".join(_tool_call_label(name, args) for name, args in specs[:5])
        result = next((message.content for message in group if message.role == "tool"), "")
        result_brief = _tool_result_brief(result, focus_terms, 260) if result else "no tool output"
        return f"- {label}: {result_brief}"
    for message in reversed(group):
        if message.content.strip():
            return f"- {message.role}: {_clip_middle(' '.join(message.content.split()), 260)}"
    return ""


def _path_from_group(group: list[Message]) -> str:
    for _name, arguments in _group_tool_call_specs(group):
        for key in ("path", "file_path", "file_pattern"):
            value = arguments.get(key)
            if isinstance(value, str) and value.strip():
                return redact_secrets(value.strip())
        value = arguments.get("paths")
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    return redact_secrets(item.strip())
    return ""


def _looks_like_failure(content: str) -> bool:
    lowered = content.lower()
    return any(
        token in lowered
        for token in (
            "error",
            "failed",
            "failure",
            "traceback",
            "rejected",
            "exception",
            "错误",
            "失败",
            "拒绝",
            "异常",
        )
    )


def _normalize_focus_terms(values: Iterable[str] | None) -> tuple[str, ...]:
    terms: list[str] = []
    for value in values or ():
        for part in str(value).replace("\\", "/").replace("\n", " ").split():
            clean = part.strip("`'\"()[]{}:;,.").lower()
            if len(clean) >= 2 and clean not in terms:
                terms.append(clean)
    return tuple(terms[:48])


def _contains_focus_term(group: list[Message], focus_terms: tuple[str, ...]) -> bool:
    text = " ".join(message.content for message in group).lower()
    return any(term in text for term in focus_terms)


def _focus_preview(content: str, limit: int, focus_terms: tuple[str, ...]) -> str:
    """Keep matching lines plus the beginning/end of a tool observation."""

    if len(content) <= limit:
        return content
    lines = content.splitlines()
    if len(lines) < 3 or not focus_terms:
        return _clip_middle(content, limit)
    normalized_terms = tuple(term.lower() for term in focus_terms)
    matching = [
        index
        for index, line in enumerate(lines)
        if any(term in line.lower() for term in normalized_terms)
    ]
    if not matching:
        return _clip_middle(content, limit)
    important = matching[:6]
    line_limit = max(32, limit // max(2, len(important) + 2))
    selected: list[int] = []
    for index in [0, *important, len(lines) - 1]:
        if index not in selected:
            selected.append(index)
    rows = [
        _focus_line_preview(lines[index], normalized_terms, line_limit)
        for index in selected
    ]
    focused = "\n".join(rows)
    if len(focused) <= limit:
        return focused
    # Keep the first matching line as a last-resort semantic anchor.
    return _focus_line_preview(lines[important[0]], normalized_terms, max(32, limit - 8))


def _focus_line_preview(line: str, focus_terms: tuple[str, ...], limit: int) -> str:
    if len(line) <= limit:
        return line
    lowered = line.lower()
    for term in focus_terms:
        position = lowered.find(term)
        if position < 0:
            continue
        left = max(0, position - limit // 3)
        right = min(len(line), left + limit)
        snippet = line[left:right]
        prefix = "..." if left else ""
        suffix = "..." if right < len(line) else ""
        return f"{prefix}{snippet}{suffix}"[:limit]
    return _clip_middle(line, limit)


def _ordered_indices(selected_indices: list[int], recent_indices: set[int]) -> list[int]:
    anchors = sorted(index for index in selected_indices if index not in recent_indices)
    recent = sorted(index for index in selected_indices if index in recent_indices)
    return [*anchors, *recent]


def _compression_limits(initial: int) -> list[int]:
    limits: list[int] = []
    value = initial
    while value > MIN_TOOL_RESULT_CHARS:
        value = max(MIN_TOOL_RESULT_CHARS, value // 2)
        if value not in limits:
            limits.append(value)
        if value == MIN_TOOL_RESULT_CHARS:
            break
    return limits


def _fit_message_contents(messages: list[Message], max_chars: int) -> tuple[list[Message], int]:
    view = [_clone_message(message) for message in messages]
    count = 0
    while estimate_context_chars(view) > max_chars:
        candidates = sorted(
            (
                index,
                message,
            )
            for index, message in enumerate(view)
            if message.role in {"tool", "assistant", "user"} and message.content
        )
        changed = False
        for index, message in candidates:
            if len(message.content) <= 120:
                continue
            new_limit = max(120, len(message.content) // 2)
            view[index] = _clone_message(message, content=_clip_middle(message.content, new_limit))
            count += 1
            changed = True
            break
        if not changed:
            break
    return view, count


def _clip_middle(value: Any, limit: int) -> str:
    text = redact_secrets(str(value))
    limit = max(0, int(limit))
    if len(text) <= limit:
        return text
    marker = "... omitted ..."
    keep = max(0, limit - len(marker))
    head = (keep + 1) // 2
    tail = keep - head
    return f"{text[:head]}{marker}{text[-tail:] if tail else ''}"


def _clone_message(message: Message, *, content: str | None = None) -> Message:
    return Message(
        role=message.role,
        content=message.content if content is None else content,
        metadata=copy.deepcopy(message.metadata),
    )


def _clone_message_list(messages: list[Message]) -> list[Message]:
    return [_clone_message(message) for message in messages]
