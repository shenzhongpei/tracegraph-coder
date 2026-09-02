from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from .models import Tool, ToolCall, ToolResult
from .repo_graph import RepoGraph, filter_files, read_text_region
from .safety import assert_safe_command, ensure_workspace, redact_secrets, safe_path, split_safe_command
from .verifier import Verifier
from .workspace_state import WorkspaceSnapshot


class ToolRegistry:
    def __init__(self, environment: Any | None = None):
        self.environment = environment
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def current_repo_graph(self) -> RepoGraph | None:
        graph = getattr(self.environment, "repo_graph", None)
        return graph if isinstance(graph, RepoGraph) else None

    def execute(self, call: ToolCall) -> ToolResult:
        tool = self.get(call.name)
        if tool is None:
            return ToolResult(False, error=f"Unknown tool '{call.name}'. Available tools: {', '.join(self.names())}")
        validation_error = _validate_tool_arguments(tool.parameters, call.arguments)
        if validation_error:
            return ToolResult(False, error=f"Invalid arguments for tool '{call.name}': {validation_error}")
        try:
            data = tool.func(**call.arguments)
            if isinstance(data, ToolResult):
                return data
            if isinstance(data, str) and data.startswith("Error:"):
                return ToolResult(False, error=data)
            return ToolResult(True, data=data)
        except Exception as exc:
            return ToolResult(False, error=f"Error executing tool '{call.name}': {exc}")


class ToolEnvironment:
    def __init__(
        self,
        workspace: str | Path,
        repo_graph: RepoGraph,
        verifier: Verifier,
        conversation_memory: str = "",
        repo_graph_ready: bool = True,
    ):
        self.workspace = ensure_workspace(workspace)
        self.repo_graph = repo_graph
        self.verifier = verifier
        self.conversation_memory = redact_secrets(str(conversation_memory or "")).strip()
        self.repo_graph_ready = bool(repo_graph_ready)

    def ensure_graph(self) -> RepoGraph:
        if not self.repo_graph_ready:
            return self.refresh_graph()
        return self.repo_graph

    def refresh_graph(self) -> RepoGraph:
        from .repo_graph import build_repo_graph

        self.repo_graph = build_repo_graph(self.workspace)
        self.repo_graph.save(self.workspace / ".tracegraph" / "repo_graph.json")
        self.repo_graph_ready = True
        return self.repo_graph


def build_default_registry(env: ToolEnvironment) -> ToolRegistry:
    registry = ToolRegistry(env)
    tools = []
    if env.conversation_memory:
        tools.append(_read_conversation_memory_tool(env))
    tools.extend(
        [
            _list_files_tool(env),
            _read_file_tool(env),
            _read_many_tool(env),
            _search_text_tool(env),
            _repo_graph_query_tool(env),
            _repo_graph_neighborhood_tool(env),
            _write_file_tool(env),
            _apply_patch_tool(env),
            _run_command_tool(env),
            _verify_tool(env),
            _git_status_tool(env),
            _git_diff_tool(env),
        ]
    )
    for tool in tools:
        registry.register(tool)
    return registry


def _object_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def _validate_tool_arguments(schema: dict[str, Any], args: dict[str, Any]) -> str | None:
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    if not isinstance(args, dict):
        return "arguments must be an object"
    for name in required:
        if name not in args:
            return f"missing required argument '{name}'"
    if schema.get("additionalProperties") is False:
        extra = sorted(set(args) - set(properties))
        if extra:
            return f"unexpected argument(s): {', '.join(extra)}"
    for name, value in args.items():
        prop = properties.get(name)
        if prop is None:
            continue
        error = _validate_value(name, value, prop)
        if error:
            return error
    return None


def _validate_value(name: str, value: Any, schema: dict[str, Any]) -> str | None:
    expected = schema.get("type")
    allowed = expected if isinstance(expected, list) else [expected]
    if value is None:
        return None if "null" in allowed else f"argument '{name}' must not be null"
    if "string" in allowed and isinstance(value, str):
        return None
    if "boolean" in allowed and isinstance(value, bool):
        return None
    if "integer" in allowed and isinstance(value, int) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and value < minimum:
            return f"argument '{name}' must be >= {minimum}"
        if maximum is not None and value > maximum:
            return f"argument '{name}' must be <= {maximum}"
        return None
    if "array" in allowed and isinstance(value, list):
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if min_items is not None and len(value) < min_items:
            return f"argument '{name}' must contain at least {min_items} item(s)"
        if max_items is not None and len(value) > max_items:
            return f"argument '{name}' must contain at most {max_items} item(s)"
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                error = _validate_value(f"{name}[{index}]", item, item_schema)
                if error:
                    return error
        return None
    if "object" in allowed and isinstance(value, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if isinstance(required, list):
            for required_name in required:
                if required_name not in value:
                    return f"argument '{name}' missing required field '{required_name}'"
        if schema.get("additionalProperties") is False and isinstance(properties, dict):
            extra = sorted(set(value) - set(properties))
            if extra:
                return f"argument '{name}' has unexpected field(s): {', '.join(extra)}"
        if isinstance(properties, dict):
            for field_name, field_value in value.items():
                field_schema = properties.get(field_name)
                if isinstance(field_schema, dict):
                    error = _validate_value(f"{name}.{field_name}", field_value, field_schema)
                    if error:
                        return error
        return None
    return f"argument '{name}' has invalid type {type(value).__name__}"


def _read_conversation_memory_tool(env: ToolEnvironment) -> Tool:
    def read_conversation_memory(
        question: str,
        focus: str = "full_summary",
        limit: int = 40,
    ) -> str:
        if not question.strip():
            return "Error: question must not be empty."
        focus = focus or "full_summary"
        allowed = {"recent_messages", "final_answer", "working_memory", "full_summary"}
        if focus not in allowed:
            return "Error: focus must be one of: " + ", ".join(sorted(allowed))
        memory = env.conversation_memory.strip()
        if not memory:
            return "(no saved conversation memory is available for this run)"
        selected = _select_conversation_memory(memory, focus)
        char_limit = max(1500, min(limit, 80) * 700)
        header = f"Conversation memory lookup\nQuestion: {question.strip()}\nFocus: {focus}"
        return _truncate_text(redact_secrets(header + "\n\n" + selected), char_limit)

    return Tool(
        name="read_conversation_memory",
        description=(
            "Read saved conversation memory, prior user/assistant messages, previous final answers, "
            "and deterministic working memory. Use this for questions about what was discussed before, "
            "what the current session remembers, or how to continue from earlier conversation state."
        ),
        parameters=_object_schema(
            {
                "question": {"type": "string"},
                "focus": {
                    "type": "string",
                    "description": "Which part of the saved conversation to read.",
                    "enum": ["recent_messages", "final_answer", "working_memory", "full_summary"],
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 80,
                    "description": "Approximate amount of memory to return; larger values return more text.",
                },
            },
            ["question"],
        ),
        func=read_conversation_memory,
        read_only=True,
    )


def _select_conversation_memory(memory: str, focus: str) -> str:
    if focus == "full_summary":
        return memory
    heading_by_focus = {
        "recent_messages": "Recent conversation:",
        "final_answer": "Previous final answer:",
        "working_memory": "Working memory:",
    }
    heading = heading_by_focus.get(focus)
    if not heading:
        return memory
    sections = [section.strip() for section in memory.split("\n\n---\n\n")]
    for section in sections:
        if section.startswith(heading):
            return section
    return memory


def _truncate_text(text: str, limit: int) -> str:
    clean = text.strip()
    if len(clean) <= limit:
        return clean
    return clean[:limit].rstrip() + "\n...[truncated]"


def _list_files_tool(env: ToolEnvironment) -> Tool:
    def list_files(pattern: str = "*", limit: int = 200) -> str:
        files = filter_files(env.workspace, pattern=pattern, limit=limit)
        return "\n".join(files) or "(no files matched)"

    return Tool(
        name="list_files",
        description="List source-like files in the workspace. Use before reading unknown paths.",
        parameters=_object_schema(
            {
                "pattern": {"type": "string", "description": "Glob pattern, for example '*.py' or 'src/*.ts'."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 500},
            }
        ),
        func=list_files,
        read_only=True,
    )


def _read_file_tool(env: ToolEnvironment) -> Tool:
    def read_file(path: str, start: int = 1, end: int | None = None) -> str:
        return read_text_region(env.workspace, path, start=start, end=end)

    return Tool(
        name="read_file",
        description="Read a text file with 1-based line numbers. Prefer focused line ranges.",
        parameters=_object_schema(
            {
                "path": {"type": "string"},
                "start": {"type": "integer", "minimum": 1},
                "end": {"type": ["integer", "null"], "minimum": 1},
            },
            ["path"],
        ),
        func=read_file,
        read_only=True,
    )


def _read_many_tool(env: ToolEnvironment) -> Tool:
    def read_many(requests: list[dict[str, Any]], max_chars: int = 30000) -> str:
        if not requests:
            return "Error: requests must not be empty."
        limit = max(1000, min(int(max_chars), 60000))
        chunks: list[str] = []
        for index, item in enumerate(requests[:8], start=1):
            path = str(item.get("path") or "").strip()
            if not path:
                return f"Error: request {index} is missing path."
            start = item.get("start", 1)
            end = item.get("end")
            if not isinstance(start, int) or isinstance(start, bool) or start < 1:
                return f"Error: request {index} start must be a positive integer."
            if end is not None and (not isinstance(end, int) or isinstance(end, bool) or end < 1):
                return f"Error: request {index} end must be a positive integer or null."
            label = f"--- {path}:{start}-{end if end is not None else 'end'} ---"
            chunks.append(label + "\n" + read_text_region(env.workspace, path, start=start, end=end))
            rendered = "\n\n".join(chunks)
            if len(rendered) > limit:
                return _truncate_text(rendered, limit)
        return "\n\n".join(chunks)

    return Tool(
        name="read_many",
        description=(
            "Read several focused file ranges in one tool call. Use this instead of many read_file calls "
            "when the likely target files are already known."
        ),
        parameters=_object_schema(
            {
                "requests": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "start": {"type": "integer", "minimum": 1},
                            "end": {"type": ["integer", "null"], "minimum": 1},
                        },
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
                "max_chars": {"type": "integer", "minimum": 1000, "maximum": 60000},
            },
            ["requests"],
        ),
        func=read_many,
        read_only=True,
    )


def _search_text_tool(env: ToolEnvironment) -> Tool:
    def search_text(pattern: str, file_pattern: str = "*", regex: bool = False, limit: int = 50) -> str:
        if not pattern:
            return "Error: pattern must not be empty."
        flags = 0
        use_regex = bool(regex) or _looks_like_regex(pattern)
        if use_regex:
            try:
                compiled = re.compile(pattern, flags)
            except re.error as exc:
                if regex:
                    return f"Error: invalid regex pattern: {exc}"
                compiled = re.compile(re.escape(pattern), flags)
        else:
            compiled = re.compile(re.escape(pattern), flags)

        results: list[str] = []
        for rel in filter_files(env.workspace, pattern=file_pattern, limit=5000):
            target = safe_path(env.workspace, rel)
            try:
                lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for lineno, line in enumerate(lines, start=1):
                if compiled.search(line):
                    results.append(f"{rel}:{lineno}: {line[:240]}")
                    if len(results) >= limit:
                        return "\n".join(results)
        return "\n".join(results) if results else "(no matches)"

    return Tool(
        name="search_text",
        description="Search text in workspace files. Returns path:line snippets.",
        parameters=_object_schema(
            {
                "pattern": {"type": "string"},
                "file_pattern": {"type": "string"},
                "regex": {"type": "boolean"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200},
            },
            ["pattern"],
        ),
        func=search_text,
        read_only=True,
    )


def _looks_like_regex(pattern: str) -> bool:
    regex_markers = ("|", ".*", ".+", "\\d", "\\w", "\\s", "\\.", "^", "$", "[", "]", "(", ")", "?")
    return any(marker in pattern for marker in regex_markers)


def _repo_graph_query_tool(env: ToolEnvironment) -> Tool:
    def repo_graph_query(query: str, limit: int = 12) -> str:
        graph = env.ensure_graph()
        hits = graph.query_with_reasons(query, limit=limit)
        if not hits:
            return "(no graph hits)"
        rows = []
        for hit in hits:
            node = hit.node
            symbols = ", ".join(f"{s.kind}:{s.name}@{s.line}" for s in node.symbols[:8])
            imports = ", ".join(node.imports[:6])
            imports_local = ", ".join(node.imports_local[:6])
            imported_by = ", ".join(node.imported_by[:6])
            calls = ", ".join(node.calls[:8])
            rows.append(
                json.dumps(
                    {
                        "path": node.path,
                        "score": hit.score,
                        "matched_tokens": hit.matched_tokens,
                        "match_reasons": hit.reasons,
                        "language": node.language,
                        "lines": node.lines,
                        "role": node.role,
                        "is_test": node.is_test,
                        "symbols": symbols,
                        "imports": imports,
                        "imports_local": imports_local,
                        "imported_by": imported_by,
                        "calls": calls,
                        "related_tests": node.related_tests[:6],
                        "related_sources": node.related_sources[:6],
                    },
                    ensure_ascii=False,
                )
            )
        return "\n".join(rows)

    return Tool(
        name="repo_graph_query",
        description=(
            "Query the repository evidence graph for relevant files and symbols. "
            "Builds the graph lazily if it has not been loaded yet."
        ),
        parameters=_object_schema(
            {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            ["query"],
        ),
        func=repo_graph_query,
        read_only=True,
    )


def _repo_graph_neighborhood_tool(env: ToolEnvironment) -> Tool:
    def repo_graph_neighborhood(path: str, limit: int = 24) -> str:
        try:
            payload = env.ensure_graph().neighborhood(path, limit=limit)
        except KeyError:
            return f"Error: path is not indexed in repository graph: {path}"
        return json.dumps(payload, ensure_ascii=False, indent=2)

    return Tool(
        name="repo_graph_neighborhood",
        description=(
            "Show local imports, reverse imports, related tests, and related sources for one indexed file. "
            "Use before patching to estimate impact and choose verification targets."
        ),
        parameters=_object_schema(
            {
                "path": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 80},
            },
            ["path"],
        ),
        func=repo_graph_neighborhood,
        read_only=True,
    )


def _write_file_tool(env: ToolEnvironment) -> Tool:
    def write_file(path: str, content: str) -> str:
        target = safe_path(env.workspace, path)
        if target.exists():
            return "Error: write_file only creates new files. Use apply_patch for existing files."
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(target, content)
        env.refresh_graph()
        return f"Wrote {len(content.encode('utf-8'))} bytes to {path}."

    return Tool(
        name="write_file",
        description="Create a new text file inside the workspace. Use apply_patch for existing files.",
        parameters=_object_schema(
            {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            ["path", "content"],
        ),
        func=write_file,
        read_only=False,
    )


def _apply_patch_tool(env: ToolEnvironment) -> Tool:
    def apply_patch(path: str, old_text: str, new_text: str) -> str:
        target = safe_path(env.workspace, path)
        if not target.exists():
            return f"Error: file does not exist: {path}"
        content = target.read_text(encoding="utf-8", errors="replace")
        count = content.count(old_text)
        if count == 0:
            return "Error: old_text not found. Read the file again and include exact whitespace."
        if count > 1:
            return f"Error: old_text matched {count} locations. Include more surrounding context."
        updated = content.replace(old_text, new_text, 1)
        _atomic_write_text(target, updated)
        env.refresh_graph()
        return f"Patched {path}: replaced {len(old_text)} chars with {len(new_text)} chars."

    return Tool(
        name="apply_patch",
        description="Apply an exact one-location text replacement. Safer than overwriting whole files.",
        parameters=_object_schema(
            {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            ["path", "old_text", "new_text"],
        ),
        func=apply_patch,
        read_only=False,
    )


def _run_command_tool(env: ToolEnvironment) -> Tool:
    def run_command(command: str, timeout: int = 120) -> ToolResult:
        assert_safe_command(command)
        args = split_safe_command(command)
        before = WorkspaceSnapshot.capture(env.workspace)
        proc = subprocess.run(
            args,
            cwd=env.workspace,
            shell=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=max(1, min(timeout, 300)),
        )
        after = WorkspaceSnapshot.capture(env.workspace)
        if after.changed_from(before):
            env.refresh_graph()
        out = redact_secrets(proc.stdout or "")
        prefix = f"exit_code={proc.returncode}\n"
        payload = prefix + (out[-50000:] if out else "(no output)")
        return ToolResult(
            ok=proc.returncode == 0,
            data=payload if proc.returncode == 0 else None,
            error=payload if proc.returncode != 0 else None,
            meta={"exit_code": proc.returncode},
        )

    return Tool(
        name="run_command",
        description="Run a non-interactive shell command in the workspace. Dangerous commands are blocked.",
        parameters=_object_schema(
            {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "minimum": 1, "maximum": 300},
            },
            ["command"],
        ),
        func=run_command,
        read_only=False,
        timeout=300,
    )


def _verify_tool(env: ToolEnvironment) -> Tool:
    def verify(command: str | None = None) -> ToolResult:
        result = env.verifier.run(command)
        payload = json.dumps(
            {"ok": result.ok, "command": result.command, "output": result.output[-50000:]},
            ensure_ascii=False,
            indent=2,
        )
        return ToolResult(
            ok=result.ok,
            data=payload if result.ok else None,
            error=payload if not result.ok else None,
            meta={"command": result.command},
        )

    return Tool(
        name="verify",
        description="Run the detected or provided verification command. Use after code changes.",
        parameters=_object_schema(
            {
                "command": {"type": ["string", "null"], "description": "Optional explicit command."},
            }
        ),
        func=verify,
        read_only=False,
    )


def _git_status_tool(env: ToolEnvironment) -> Tool:
    def git_status() -> str:
        return _git(env.workspace, ["status", "--short"])

    return Tool(
        name="git_status",
        description="Show concise git status. Useful before final answer.",
        parameters=_object_schema({}),
        func=git_status,
        read_only=True,
    )


def _git_diff_tool(env: ToolEnvironment) -> Tool:
    def git_diff() -> str:
        return _git(env.workspace, ["diff", "--", "."])

    return Tool(
        name="git_diff",
        description="Show current workspace diff. Use to audit modifications.",
        parameters=_object_schema({}),
        func=git_diff,
        read_only=True,
    )


def _git(workspace: Path, args: list[str]) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=workspace,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )
    out = redact_secrets(proc.stdout or "").strip()
    if proc.returncode != 0:
        return f"Error: git {' '.join(args)} failed: {out or '(no output)'}"
    return out or "(empty)"


def _atomic_write_text(path: Path, content: str) -> None:
    temp = path.with_name(f".{path.name}.tracegraph-tmp")
    temp.write_text(content, encoding="utf-8")
    temp.replace(path)
