from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .coding_context import DEFAULT_CODING_CONTEXT_BUDGET, compile_coding_context
from .context_window import (
    DEFAULT_CONTEXT_CHAR_BUDGET,
    ContextBudgetTuner,
    estimate_tool_schema_tokens,
    prepare_context_messages,
)
from .evidence import EvidenceLog, EvidenceStep
from .experience import extract_experience_terms
from .models import LLMResponse, Message, ToolCall, ToolResult
from .repo_graph import RepoGraph
from .safety import redact_secrets
from .tools import ToolRegistry
from .verifier import VerificationResult, Verifier
from .working_memory import WorkingMemory
from .workspace_state import WorkspaceSnapshot, workspace_diff


SYSTEM_PROMPT = """You are TraceGraph Coder, an evidence-driven coding agent.

You can inspect and change the local repository only through the provided tools.
Your operating policy:
1. Decide which evidence source fits the user's request before acting; do not inspect files just because tools exist.
2. Use read_conversation_memory for questions about prior discussion, saved context, previous final answers, or how to continue the current conversation.
3. Use repo_graph_query, repo_graph_neighborhood, list_files, read_file, or search_text for repository/code questions.
4. Follow project memory when it is provided, but treat actual tool observations as the source of truth.
5. Work through phases as needed: PLAN -> LOCATE -> READ -> PATCH -> VERIFY -> REPORT.
6. Keep exploration proportional to the task. For a narrow UI/code change, identify a few likely files, use read_many or exact ranges, then patch.
7. Before changing code, make the smallest possible change and base it on observed code evidence.
8. When verification or harness feedback fails, preserve the failure terms and reuse them as evidence markers instead of restarting broad exploration.
9. Use repo_graph_neighborhood on likely target files when impact or related tests are unclear.
10. Prefer apply_patch over write_file for existing files.
11. After any code change, call verify or let the controller run automatic verification.
12. Before final answer, inspect git_diff when possible; if Git is unavailable, rely on the final workspace diff report.
13. Never fabricate tool results. If verification cannot run, explain why.
14. Use record_progress before meaningful edits to state phase, hypothesis, next step, and target paths.
15. Prefer finish_task for completion. If files changed, completion is accepted only after successful verification after the latest mutation.
16. The local harness may reject repeated, low-novelty, or ill-timed exploration. Treat those rejections as execution feedback: narrow the next tool call, patch from existing evidence, verify, or report.

Final-answer style:
- Match the user's language. If the user writes Chinese, answer in natural Chinese.
- Lead with the actual answer or result. Do not start with process preambles such as "I now have...", "Let me...", or "This is a read-only analysis task".
- Keep the tone like a helpful coding partner: warm, natural, and conversational.
- Be a little more expansive than a terse log, but avoid sounding like a code walkthrough or a formal report.
- Use simple transitions and human phrasing, as if you are explaining progress to a teammate.
- When the answer is nuanced, give the conclusion first and then 2-4 short supporting points.
- When explaining execution, summarize tool calls and their outcomes in plain language instead of echoing raw JSON or full result blobs.
- Do not expose internal control text such as harness messages, exploration budgets, or context compaction details unless the user explicitly asks about them.
- For code-changing tasks, include what changed, why it is supported by observed evidence, and the verification result, but keep it short.
- For analysis or conversation-continuation tasks, answer the user's question directly and mention saved context/tool observations only when it helps the user trust the answer.
"""

CONTROL_TOOL_NAMES = {"record_progress", "finish_task"}
CONTROL_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "record_progress",
            "description": "Record current bounded progress before changing code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "phase": {
                        "type": "string",
                        "enum": ["exploring", "modifying", "verifying", "ready"],
                    },
                    "hypothesis": {"type": "string"},
                    "next_step": {"type": "string"},
                    "target_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 8,
                    },
                },
                "required": ["phase", "hypothesis", "next_step", "target_paths"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish_task",
            "description": "Request evidence-gated completion of the coding task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": (
                            "User-facing final answer. Match the user's language, lead with the result, "
                            "and avoid internal process preambles or harness/tool-control wording."
                        ),
                    },
                    "strategy": {"type": "string"},
                    "no_changes_reason": {"type": "string"},
                },
                "required": ["summary", "strategy"],
                "additionalProperties": False,
            },
        },
    },
]


@dataclass(slots=True)
class RunResult:
    final_text: str
    verification: VerificationResult | None
    report_path: Path
    iterations: int
    working_memory: WorkingMemory | None = None
    messages: list[Message] | None = None


class AgentController:
    def __init__(
        self,
        *,
        llm: Any,
        registry: ToolRegistry,
        repo_graph: RepoGraph,
        evidence: EvidenceLog,
        verifier: Verifier,
        project_memory: str | None = None,
        max_steps: int = 20,
        auto_verify: bool = True,
        max_repair_attempts: int = 1,
        tool_repetition_limit: int = 3,
        context_budget_chars: int = DEFAULT_CONTEXT_CHAR_BUDGET,
        coding_context_budget_chars: int = DEFAULT_CODING_CONTEXT_BUDGET,
        experience_store: Any | None = None,
        on_event: Callable[[str], None] | None = None,
    ):
        self.llm = llm
        self.registry = registry
        self.repo_graph = repo_graph
        self.evidence = evidence
        self.verifier = verifier
        self.project_memory = project_memory or ""
        self.max_steps = max_steps
        self.auto_verify = auto_verify
        self.max_repair_attempts = max_repair_attempts
        self.tool_repetition_limit = max(1, int(tool_repetition_limit))
        self.context_budget_chars = context_budget_chars
        self.context_budget_tuner = ContextBudgetTuner(base_max_chars=context_budget_chars)
        self.coding_context_budget_chars = coding_context_budget_chars
        self.experience_store = experience_store
        self.on_event = on_event or (lambda _msg: None)

    def run(
        self,
        task: str,
        *,
        initial_messages: list[Message] | None = None,
        initial_memory: WorkingMemory | None = None,
        initial_iteration: int = 0,
        reset_evidence: bool = True,
        on_checkpoint: Callable[[list[Message], WorkingMemory, int, str], None] | None = None,
    ) -> RunResult:
        if reset_evidence:
            self.evidence.reset()
        memory = initial_memory or WorkingMemory(task)
        effective_task = memory.task or task
        self._refresh_task_profile(memory, effective_task)
        if initial_memory is None:
            self._load_experience_hint(memory)
        messages = (
            [Message.from_dict(message.to_dict()) for message in initial_messages]
            if initial_messages
            else self._initial_messages(effective_task, memory)
        )
        baseline: WorkspaceSnapshot | None = None
        changed = bool(initial_memory and initial_memory.modified_paths)
        repair_attempts = 0
        last_verification: VerificationResult | None = None
        final_text = ""
        evidence_gathered = bool(initial_memory and (initial_memory.files or initial_memory.observations))
        last_iteration = max(0, initial_iteration)
        completed_by_finish_tool = False
        tool_call_counts = _seed_tool_signature_counts(messages)
        progress_guard = _ExplorationProgressGuard()
        harness = _AgentHarness()
        self._checkpoint(on_checkpoint, messages, memory, last_iteration, "running")

        for local_iteration in range(1, self.max_steps + 1):
            iteration = max(0, initial_iteration) + local_iteration
            last_iteration = iteration
            self._emit_phase(iteration, "PLAN")
            self.on_event(f"[{iteration}] calling model")
            if self._sync_repo_graph():
                self.on_event(f"[{iteration}] repo graph synchronized")
            stale_before = set(memory.stale_files)
            memory.refresh_files(self.repo_graph.root)
            new_stale = sorted(set(memory.stale_files) - stale_before)
            if new_stale:
                self.on_event(f"[{iteration}] stale file evidence: {', '.join(new_stale[:8])}")
            tool_schemas = self._tool_schemas()
            prompt_view, context_report = prepare_context_messages(
                self._messages_with_memory(messages, memory),
                max_chars=self.context_budget_tuner.current_max_chars(),
                focus_terms=[
                    effective_task,
                    memory.task_type,
                    memory.hypothesis,
                    memory.next_step,
                    *memory.experience_terms,
                    *memory.failure_terms,
                    *memory.candidate_files,
                    *memory.target_paths,
                    *memory.modified_paths,
                    *(item.path for item in memory.files),
                ],
            )
            if context_report.compacted:
                self.on_event(f"[{iteration}] {context_report.format()}")
            response = self.llm.chat(prompt_view, tool_schemas)
            usage_note = self.context_budget_tuner.observe(
                context_report,
                response.usage,
                tool_schema_tokens=estimate_tool_schema_tokens(tool_schemas),
            )
            if usage_note:
                self.on_event(f"[{iteration}] {usage_note}")
            assistant_content = response.content.strip()
            if not response.tool_calls:
                assistant_content = _normalize_final_text(assistant_content)
            messages.append(
                Message(
                    role="assistant",
                    content=assistant_content if assistant_content else response.content,
                    metadata={"tool_calls": response.tool_calls} if response.tool_calls else None,
                )
            )
            self._checkpoint(on_checkpoint, messages, memory, iteration, "running")

            tool_calls = self._tool_calls_from_response(response)
            if tool_calls:
                iteration_tool_names: list[str] = []
                iteration_changed_paths: list[str] = []
                for call in tool_calls:
                    iteration_tool_names.append(call.name)
                    phase = _phase_for_tool(call.name)
                    self._emit_phase(iteration, phase)
                    self.on_event(f"[{iteration}] tool {call.name}({self._compact_args(call.arguments)})")
                    signature = _tool_signature(call.name, call.arguments)
                    tool_call_counts[signature] = tool_call_counts.get(signature, 0) + 1
                    if tool_call_counts[signature] > self.tool_repetition_limit:
                        result = ToolResult(
                            False,
                            error=(
                                "Repeated tool call blocked: "
                                f"{call.name} with the same arguments exceeded "
                                f"the limit of {self.tool_repetition_limit}. "
                                "Choose a different search range, inspect different evidence, or revise the plan."
                            ),
                            meta={"phase": phase},
                        )
                        self._record_tool_step(iteration, call, result, phase)
                        memory.observe_tool(
                            name=call.name,
                            arguments=call.arguments,
                            ok=False,
                            observation=result.serialize(limit=4000),
                            step=iteration,
                            workspace=self.repo_graph.root,
                        )
                        _record_failure_signal(
                            memory,
                            step=iteration,
                            source="harness",
                            category="repeated_tool_call",
                            summary=result.error or "",
                            paths=_paths_from_tool_call(call),
                            term_sources=(memory.hypothesis, memory.next_step),
                        )
                        messages.append(
                            Message(
                                role="tool",
                                content=result.serialize(),
                                metadata={"tool_call_id": call.id},
                            )
                        )
                        self._checkpoint(on_checkpoint, messages, memory, iteration, "running")
                        continue
                    harness_result = harness.before_tool_call(
                        call,
                        memory=memory,
                        evidence_gathered=evidence_gathered,
                    )
                    if harness_result is not None:
                        self._record_tool_step(iteration, call, harness_result, phase)
                        harness_meta = harness_result.meta if isinstance(harness_result.meta, dict) else {}
                        harness_category = str(harness_meta.get("harness") or "exploration_guard")
                        failure_terms = extract_experience_terms(
                            harness_result.error or "",
                            memory.hypothesis,
                            memory.next_step,
                            *memory.target_paths,
                            *memory.candidate_files,
                        )
                        memory.observe_failure_event(
                            source="harness",
                            category=harness_category,
                            summary=harness_result.error or "",
                            step=iteration,
                            terms=failure_terms,
                            paths=_known_harness_paths(memory)[:8],
                        )
                        messages.append(
                            Message(
                                role="tool",
                                content=harness_result.serialize(),
                                metadata={"tool_call_id": call.id},
                            )
                        )
                        self._checkpoint(on_checkpoint, messages, memory, iteration, "running")
                        continue
                    if call.name in CONTROL_TOOL_NAMES:
                        result, accepted_summary = self._execute_control_tool(
                            call,
                            memory,
                            iteration,
                            allow_finish=len(tool_calls) == 1,
                        )
                        self._record_tool_step(iteration, call, result, phase)
                        if not result.ok:
                            _record_failure_signal(
                                memory,
                                step=iteration,
                                source="control_tool",
                                category=f"{call.name}_rejected",
                                summary=result.error or "",
                                paths=memory.modified_paths or memory.target_paths,
                                term_sources=(memory.hypothesis, memory.next_step),
                            )
                        messages.append(
                            Message(
                                role="tool",
                                content=result.serialize(),
                                metadata={"tool_call_id": call.id},
                            )
                        )
                        self._checkpoint(on_checkpoint, messages, memory, iteration, "running")
                        if accepted_summary is not None:
                            final_text = accepted_summary
                            completed_by_finish_tool = True
                            break
                        continue
                    if _is_mutating_tool(call.name) and not evidence_gathered:
                        result = ToolResult(
                            False,
                            error=(
                                "Mutation blocked: gather code evidence first with repo_graph_query, "
                                "repo_graph_neighborhood, list_files, read_file, or search_text. "
                                "Conversation memory alone is not sufficient evidence for code changes."
                            ),
                            meta={"phase": phase},
                        )
                        self._record_tool_step(iteration, call, result, phase)
                        memory.observe_tool(
                            name=call.name,
                            arguments=call.arguments,
                            ok=False,
                            observation=result.serialize(limit=4000),
                            step=iteration,
                            workspace=self.repo_graph.root,
                        )
                        _record_failure_signal(
                            memory,
                            step=iteration,
                            source="harness",
                            category="mutation_without_evidence",
                            summary=result.error or "",
                            paths=_paths_from_tool_call(call),
                            term_sources=(memory.hypothesis, memory.next_step),
                        )
                        messages.append(
                            Message(
                                role="tool",
                                content=result.serialize(),
                                metadata={"tool_call_id": call.id},
                            )
                        )
                        self._checkpoint(on_checkpoint, messages, memory, iteration, "running")
                        continue
                    tool_def = self.registry.get(call.name)
                    tracks_workspace = bool(tool_def and not tool_def.read_only)
                    before_tool = WorkspaceSnapshot.capture(self.repo_graph.root) if tracks_workspace else None
                    if baseline is None and before_tool is not None:
                        baseline = before_tool
                    result = self.registry.execute(call)
                    after_tool = WorkspaceSnapshot.capture(self.repo_graph.root) if tracks_workspace else None
                    if self._sync_repo_graph():
                        self.on_event(f"[{iteration}] repo graph synchronized")
                    changed_paths = (
                        after_tool.changed_paths_from(before_tool)
                        if before_tool is not None and after_tool is not None
                        else []
                    )
                    if changed_paths:
                        changed = True
                        iteration_changed_paths.extend(changed_paths)
                        memory.observe_workspace_changes(changed_paths, iteration, self.repo_graph.root)
                    if call.name == "verify":
                        last_verification = _parse_verification_tool_result(result)
                        memory.observe_verification(last_verification, iteration, self.repo_graph.root)
                        if last_verification is not None and not last_verification.ok:
                            _record_failure_signal(
                                memory,
                                step=iteration,
                                source="verification",
                                category="manual_verify_failed",
                                summary=f"{last_verification.command}\n{last_verification.output}",
                                paths=memory.modified_paths,
                                term_sources=(memory.hypothesis, memory.next_step),
                            )
                    if result.ok and _is_code_evidence_tool(call.name):
                        evidence_gathered = True
                    memory.observe_tool(
                        name=call.name,
                        arguments=call.arguments,
                        ok=result.ok,
                        observation=result.serialize(limit=4000),
                        step=iteration,
                        workspace=self.repo_graph.root,
                    )
                    if not result.ok and call.name != "verify":
                        _record_failure_signal(
                            memory,
                            step=iteration,
                            source="tool",
                            category=f"{call.name}_failed",
                            summary=result.error or result.serialize(limit=1200),
                            paths=_paths_from_tool_call(call),
                            term_sources=(memory.hypothesis, memory.next_step),
                        )
                    harness.after_tool_call(
                        call,
                        result,
                        memory=memory,
                        changed_paths=changed_paths,
                    )
                    self._record_tool_step(iteration, call, result, phase)
                    tool_metadata = {"tool_call_id": call.id}
                    if changed_paths:
                        tool_metadata["changed_paths"] = changed_paths
                    messages.append(
                        Message(
                            role="tool",
                            content=result.serialize(),
                            metadata=tool_metadata,
                        )
                    )
                    self._checkpoint(on_checkpoint, messages, memory, iteration, "running")
                if completed_by_finish_tool:
                    break
                progress_nudge = progress_guard.after_tool_turn(
                    iteration=iteration,
                    tool_names=iteration_tool_names,
                    changed_paths=iteration_changed_paths,
                    memory=memory,
                    evidence_gathered=evidence_gathered,
                )
                if progress_nudge:
                    self.on_event(f"[{iteration}] exploration guard nudged model")
                    self.evidence.append(
                        EvidenceStep(
                            iteration=iteration,
                            stage="plan",
                            action="exploration_guard",
                            args={"tool_names": iteration_tool_names},
                            ok=True,
                            observation=progress_nudge,
                        )
                    )
                    messages.append(
                        Message(
                            role="user",
                            content=progress_nudge,
                            metadata={"control": "exploration_guard"},
                        )
                    )
                    self._checkpoint(on_checkpoint, messages, memory, iteration, "running")
                continue

            final_text = assistant_content if not response.tool_calls else response.content.strip()
            if changed and self.auto_verify:
                self._emit_phase(iteration, "VERIFY")
                last_verification = self._auto_verify(iteration)
                memory.observe_verification(last_verification, iteration, self.repo_graph.root)
                if not last_verification.ok and repair_attempts < self.max_repair_attempts:
                    _record_failure_signal(
                        memory,
                        step=iteration,
                        source="verification",
                        category="auto_verify_failed",
                        summary=f"{last_verification.command}\n{last_verification.output}",
                        paths=memory.modified_paths,
                        term_sources=(memory.hypothesis, memory.next_step),
                    )
                    repair_attempts += 1
                    messages.append(
                        Message(
                            role="user",
                            content=(
                                "Automatic verification failed. Use the failure output as evidence, "
                                "inspect the relevant code, and make the smallest repair.\n\n"
                                f"Command: {last_verification.command}\n"
                                f"Output:\n{last_verification.output[-12000:]}"
                            ),
                        )
                    )
                    self._checkpoint(on_checkpoint, messages, memory, iteration, "running")
                    continue
            blockers = memory.completion_blockers(self.repo_graph.root)
            if final_text and blockers:
                _record_failure_signal(
                    memory,
                    step=iteration,
                    source="completion_gate",
                    category="completion_rejected",
                    summary=" ".join(blockers),
                    paths=memory.modified_paths or memory.target_paths,
                    term_sources=(memory.hypothesis, memory.next_step),
                )
                messages.append(
                    Message(
                        role="user",
                        content=(
                            "Completion rejected by the local evidence gate: "
                            + " ".join(blockers)
                            + " Continue working, verify after the latest mutation, then call finish_task."
                        ),
                    )
                )
                self._checkpoint(on_checkpoint, messages, memory, iteration, "running")
                continue
            break
        else:
            final_text = final_text or "Stopped because max_steps was reached."
            if changed and self.auto_verify:
                self._emit_phase(last_iteration, "VERIFY")
                last_verification = self._auto_verify(last_iteration)
                memory.observe_verification(last_verification, last_iteration, self.repo_graph.root)
                self._checkpoint(on_checkpoint, messages, memory, last_iteration, "running")

        final_text = _normalize_final_text(final_text)
        _ensure_final_answer_message(messages, final_text)
        self._checkpoint(on_checkpoint, messages, memory, last_iteration, "running")
        self._emit_phase(max(1, last_iteration), "REPORT")
        diff_text = self._workspace_diff(baseline)
        verification_text = _format_verification(last_verification)
        working_memory_text = memory.render()
        report_path = self.evidence.write_report(
            final_text,
            verification_text,
            diff_text,
            working_memory_text,
        )
        return RunResult(
            final_text=final_text,
            verification=last_verification,
            report_path=report_path,
            iterations=len([m for m in messages if m.role == "assistant"]),
            working_memory=memory,
            messages=messages,
        )

    def _initial_messages(self, task: str, memory: WorkingMemory) -> list[Message]:
        project_memory_text = (
            f"Project memory:\n{self.project_memory}\n\n"
            if self.project_memory
            else "Project memory: (none)\n\n"
        )
        repo_summary = self._initial_repository_guidance(task, memory)
        return [
            Message(role="system", content=SYSTEM_PROMPT),
            Message(
                role="user",
                content=(
                    f"Task:\n{task}\n\n"
                    f"{project_memory_text}"
                    "Initial repository guidance:\n"
                    f"{repo_summary}"
                ),
            ),
        ]

    def _refresh_task_profile(self, memory: WorkingMemory, task: str) -> None:
        task_type = _classify_task(task)
        candidates = _candidate_files_for_task(task, self.repo_graph, task_type)
        memory.set_task_profile(task_type, candidates)

    def _initial_repository_guidance(self, task: str, memory: WorkingMemory) -> str:
        if not self.repo_graph.files:
            return (
                "Repository evidence graph: (not loaded yet)\n"
                "- Use repo_graph_query or repo_graph_neighborhood when repository/code evidence is needed.\n"
                "- Use read_conversation_memory for follow-up or history questions."
            )
        graph_by_path = {node.path: node for node in self.repo_graph.files}
        candidates = list(memory.candidate_files)
        for node in self.repo_graph.query(task, limit=8):
            if node.path not in candidates:
                candidates.append(node.path)
            if len(candidates) >= 8:
                break

        lines = [
            "Task profile:",
            f"- type: {memory.task_type}",
            "- candidate files: " + (", ".join(candidates[:8]) or "(none)"),
            "",
            "Initial repository candidates:",
        ]
        for path in candidates[:8]:
            node = graph_by_path.get(path)
            if node is None:
                continue
            symbols = ", ".join(f"{symbol.kind}:{symbol.name}@{symbol.line}" for symbol in node.symbols[:5])
            detail = f"{node.path} ({node.language}, {node.lines} lines, role={node.role})"
            if symbols:
                detail += f"; symbols=[{symbols}]"
            if node.related_tests:
                detail += f"; related_tests=[{', '.join(node.related_tests[:3])}]"
            lines.append("- " + detail)
        if len(lines) == 5:
            lines.append("- (no strong candidate; use one targeted repo_graph_query or search_text call)")
        lines.extend(
            [
                "",
                "Use these as starting points only. For small changes, read the smallest exact range that confirms the target, then patch and verify.",
            ]
        )
        return "\n".join(lines)

    def _messages_with_memory(self, messages: list[Message], memory: WorkingMemory) -> list[Message]:
        if not messages:
            return messages
        memory_message = Message(
            role="system",
            content=(
                "Current working memory (deterministic local state; re-check stale evidence before editing):\n"
                f"{memory.render()}"
            ),
        )
        coding_context = compile_coding_context(
            task=memory.task,
            repo_graph=self.repo_graph,
            evidence=self.evidence,
            memory=memory,
            project_memory=self.project_memory,
            budget_chars=self.coding_context_budget_chars,
        )
        context_message = Message(
            role="system",
            content=coding_context.render(),
        )
        if len(messages) >= 2 and messages[1].role == "user":
            return [messages[0], messages[1], memory_message, context_message, *messages[2:]]
        return [messages[0], memory_message, context_message, *messages[1:]]

    def _tool_schemas(self) -> list[dict[str, Any]]:
        registered = self.registry.schemas()
        names = {
            item.get("function", {}).get("name")
            for item in registered
            if isinstance(item, dict) and isinstance(item.get("function"), dict)
        }
        conflicts = CONTROL_TOOL_NAMES.intersection(name for name in names if isinstance(name, str))
        if conflicts:
            raise ValueError("Tool registry conflicts with control tool(s): " + ", ".join(sorted(conflicts)))
        return [*registered, *CONTROL_TOOL_SCHEMAS]

    def _execute_control_tool(
        self,
        call: ToolCall,
        memory: WorkingMemory,
        iteration: int,
        *,
        allow_finish: bool,
    ) -> tuple[ToolResult, str | None]:
        if call.name == "record_progress":
            required = {"phase", "hypothesis", "next_step", "target_paths"}
            error = _validate_control_arguments(call.arguments, allowed=required, required=required)
            if error is None:
                try:
                    memory.observe_progress(call.arguments, iteration)
                except ValueError as exc:
                    error = str(exc)
            if error:
                return ToolResult(False, error=f"Invalid record_progress arguments: {error}"), None
            return ToolResult(True, data="Progress recorded."), None

        allowed = {"summary", "strategy", "no_changes_reason"}
        error = _validate_control_arguments(call.arguments, allowed=allowed, required={"summary", "strategy"})
        if error is None and not allow_finish:
            error = "finish_task must be the only tool call in its assistant message"
        summary = str(call.arguments.get("summary", "")).strip()
        if error is None and not summary:
            error = "summary must be non-empty"
        strategy = str(call.arguments.get("strategy", "")).strip()
        if error is None and not strategy:
            error = "strategy must be non-empty"
        if error:
            return ToolResult(False, error=f"Invalid finish_task arguments: {error}"), None

        blockers = memory.completion_blockers(self.repo_graph.root)
        if not memory.modified_paths and not str(call.arguments.get("no_changes_reason", "")).strip():
            blockers.append("No workspace changes were recorded; provide no_changes_reason for analysis-only tasks.")
        if blockers:
            return ToolResult(False, error="Completion rejected: " + " ".join(blockers)), None
        if self.experience_store is not None and memory.modified_paths:
            try:
                card = self.experience_store.add_verified(memory, strategy)
                memory.observe_progress(
                    {
                        "experience_hint": memory.experience_hint,
                        "experience_id": card.experience_id,
                        "experience_searches": memory.experience_searches,
                    },
                    iteration,
                )
                return (
                    ToolResult(True, data=f"Completion accepted. Experience stored: {card.experience_id}"),
                    redact_secrets(summary),
                )
            except Exception as exc:
                return ToolResult(False, error=f"Completion accepted but experience storage failed: {exc}"), None
        return ToolResult(True, data="Completion accepted."), redact_secrets(summary)

    def _load_experience_hint(self, memory: WorkingMemory) -> None:
        if self.experience_store is None:
            return
        try:
            match = self.experience_store.retrieve(memory.task)
            update: dict[str, Any] = {"experience_searches": memory.experience_searches + 1}
            if match is not None:
                update["experience_hint"] = match.render_hint()
                update["experience_id"] = match.experience_id
                update["experience_terms"] = extract_experience_terms(
                    match.card.task,
                    match.card.strategy,
                    match.card.verification_summary,
                    match.card.verification_command,
                    *match.shared_terms,
                )
                self.on_event(f"[0] experience {match.experience_id} injected")
            memory.observe_progress(update, 0)
        except Exception as exc:
            self.on_event(f"[0] experience lookup skipped: {redact_secrets(str(exc))}")

    def _tool_calls_from_response(self, response: LLMResponse) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for idx, raw in enumerate(response.tool_calls or []):
            try:
                calls.append(ToolCall.from_wire(raw, idx))
            except Exception as exc:
                calls.append(ToolCall(id=f"bad_{idx}", name="__invalid__", arguments={"error": str(exc)}))
        if calls:
            return calls
        parsed = _parse_json_action(response.content)
        return [parsed] if parsed else []

    def _record_tool_step(self, iteration: int, call: ToolCall, result: ToolResult, phase: str) -> None:
        self.evidence.append(
            EvidenceStep(
                iteration=iteration,
                stage=phase.lower(),
                action=call.name,
                args=call.arguments,
                ok=result.ok,
                observation=result.serialize(limit=12000),
            )
        )

    def _auto_verify(self, iteration: int) -> VerificationResult:
        self.on_event(f"[{iteration}] auto verification")
        result = self.verifier.run()
        self.evidence.append(
            EvidenceStep(
                iteration=iteration,
                stage="verify",
                action="auto_verify",
                args={"command": result.command},
                ok=result.ok,
                observation=result.output,
            )
        )
        return result

    def _workspace_diff(self, baseline: WorkspaceSnapshot | None) -> str:
        if baseline is None:
            return "(no workspace changes observed)"
        return workspace_diff(self.repo_graph.root, baseline)

    def _sync_repo_graph(self) -> bool:
        graph = self.registry.current_repo_graph()
        if graph is None or graph is self.repo_graph:
            return False
        if Path(graph.root).resolve() != Path(self.repo_graph.root).resolve():
            return False
        self.repo_graph = graph
        return True

    def _emit_phase(self, iteration: int, phase: str) -> None:
        self.on_event(f"[{iteration}] phase {phase}")

    @staticmethod
    def _checkpoint(
        callback: Callable[[list[Message], WorkingMemory, int, str], None] | None,
        messages: list[Message],
        memory: WorkingMemory,
        iteration: int,
        status: str,
    ) -> None:
        if callback is not None:
            callback(messages, memory, iteration, status)

    @staticmethod
    def _compact_args(args: dict[str, Any]) -> str:
        text = json.dumps(args, ensure_ascii=False)
        text = redact_secrets(text)
        return text[:180] + ("..." if len(text) > 180 else "")


def _parse_json_action(content: str) -> ToolCall | None:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    if data.get("type") not in {"tool", "action"}:
        return None
    name = data.get("name") or data.get("tool")
    args = data.get("args") or data.get("arguments") or {}
    if not isinstance(name, str) or not isinstance(args, dict):
        return None
    return ToolCall(id="json_action", name=name, arguments=args)


def _normalize_final_text(text: str) -> str:
    cleaned = _strip_stiff_final_preamble(text.strip())
    if not cleaned:
        return cleaned
    blocks = [block.strip() for block in re.split(r"\n{2,}", cleaned) if block.strip()]
    if len(blocks) < 6:
        return cleaned
    repeated = _collapse_repeating_blocks(blocks)
    if repeated is None:
        return cleaned
    prefix, cycle = repeated
    normalized = "\n\n".join([*prefix, *cycle])
    if len(prefix) + len(cycle) < len(blocks):
        normalized += "\n\n[重复内容已自动折叠]"
    return normalized


def _strip_stiff_final_preamble(text: str) -> str:
    blocks = [block.strip() for block in re.split(r"\n{2,}", text.strip()) if block.strip()]
    while len(blocks) > 1 and _looks_like_stiff_final_preamble(blocks[0]):
        blocks.pop(0)
    return "\n\n".join(blocks) if blocks else text.strip()


def _looks_like_stiff_final_preamble(block: str) -> bool:
    if len(block) > 900:
        return False
    lowered = block.lower()
    markers = [
        "i now have",
        "i have a complete understanding",
        "i have a comprehensive understanding",
        "let me provide",
        "let me give",
        "let me explain",
        "this is a read-only analysis task",
        "no code changes are needed",
    ]
    return any(marker in lowered for marker in markers)


def _collapse_repeating_blocks(blocks: list[str]) -> tuple[list[str], list[str]] | None:
    n = len(blocks)
    for start in range(0, n - 5):
        tail = blocks[start:]
        tail_len = len(tail)
        for cycle_len in range(1, min(4, tail_len // 2) + 1):
            cycle = tail[:cycle_len]
            if tail_len < cycle_len * 3:
                continue
            if all(tail[i] == cycle[i % cycle_len] for i in range(tail_len)):
                return blocks[:start], cycle
    return None


def _ensure_final_answer_message(messages: list[Message], final_text: str) -> None:
    final = final_text.strip()
    if not final:
        return
    normalized_final = _normalize_comparable_text(final)
    latest_user_index = -1
    for index, message in enumerate(messages):
        if message.role == "user" and not _is_internal_control_message(message):
            latest_user_index = index

    for index in range(len(messages) - 1, latest_user_index, -1):
        message = messages[index]
        if message.role != "assistant":
            continue
        metadata = message.metadata if isinstance(message.metadata, dict) else {}
        if metadata.get("tool_calls"):
            continue
        if _normalize_comparable_text(message.content) != normalized_final:
            continue
        if not metadata.get("final_answer"):
            message.metadata = {**metadata, "final_answer": True}
        return

    messages.append(Message(role="assistant", content=final, metadata={"final_answer": True}))


def _normalize_comparable_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _is_internal_control_message(message: Message) -> bool:
    metadata = message.metadata if isinstance(message.metadata, dict) else {}
    return bool(metadata.get("control") or metadata.get("harness") or metadata.get("internal"))


def _record_failure_signal(
    memory: WorkingMemory,
    *,
    step: int,
    source: str,
    category: str,
    summary: str,
    paths: list[str] | None = None,
    term_sources: tuple[Any, ...] = (),
) -> None:
    event_paths = _merge_failure_paths(paths or [], _known_harness_paths(memory))
    terms = extract_experience_terms(summary, *term_sources, *event_paths)
    memory.observe_failure_event(
        source=source,
        category=category,
        summary=summary,
        step=step,
        terms=terms,
        paths=event_paths[:8],
    )


def _paths_from_tool_call(call: ToolCall) -> list[str]:
    paths: list[str] = []
    for key in ("path", "file_pattern"):
        value = call.arguments.get(key)
        if isinstance(value, str):
            paths.append(value)
    requests = call.arguments.get("requests")
    if isinstance(requests, list):
        for item in requests:
            if isinstance(item, dict) and isinstance(item.get("path"), str):
                paths.append(item["path"])
    return _merge_failure_paths(paths)


def _merge_failure_paths(*groups: list[str]) -> list[str]:
    merged: list[str] = []
    for group in groups:
        for path in group:
            clean = _normalize_repo_path(path)
            if clean and clean not in merged:
                merged.append(clean)
    return merged


class _AgentHarness:
    """Enforce adaptive execution discipline around model-selected tools.

    The harness does not prescribe a fixed workflow. It watches for evidence
    maturity and low-novelty exploration, then blocks only calls that are very
    unlikely to improve the next decision.
    """

    def __init__(self) -> None:
        self.exploration_attempts = 0
        self.broad_attempts = 0
        self.targeted_escape_used = 0
        self.low_gain_streak = 0
        self.seen_exploration_signatures: set[str] = set()
        self.observed_paths: set[str] = set()

    def before_tool_call(
        self,
        call: ToolCall,
        *,
        memory: WorkingMemory,
        evidence_gathered: bool,
    ) -> ToolResult | None:
        if not _is_exploration_tool(call.name):
            return None
        if memory.task_type in {"analysis", "conversation"}:
            return None
        if call.name in {"read_conversation_memory", "git_status", "git_diff"}:
            return None

        known_paths = _known_harness_paths(memory)
        if not known_paths:
            self._count_attempt(broad=False)
            return None

        broad = _is_broad_exploration_call(call, known_paths)
        signature = _semantic_exploration_signature(call)
        evidence_ready = _evidence_is_actionable(memory, evidence_gathered)
        if signature in self.seen_exploration_signatures and not _is_targeted_escape_call(call, known_paths):
            return _harness_blocked_result(
                memory,
                reason=(
                    "a semantically equivalent locate/read call has already been tried; "
                    "repeating the same evidence request is unlikely to add new information"
                ),
                category="redundant_exploration",
            )

        if evidence_ready and self.low_gain_streak >= 2 and broad:
            return _harness_blocked_result(
                memory,
                reason=(
                    "recent exploration produced little new evidence and the task already has "
                    "candidate or target files"
                ),
                category="low_novelty_exploration",
            )

        if memory.modified_paths and broad:
            return _harness_blocked_result(
                memory,
                reason=(
                    "the workspace has already changed; broad exploration at this point risks "
                    "losing the repair thread"
                ),
                category="post_mutation_exploration",
            )

        if memory.phase in {"modifying", "verifying", "ready"} and broad:
            return _harness_blocked_result(
                memory,
                reason=(
                    f"the task is already in {memory.phase} phase and target files are known; "
                    "broad locate/read calls are no longer useful"
                ),
                category="phase_mismatch",
            )

        if broad and self.broad_attempts >= _broad_exploration_budget(memory.task_type):
            return _harness_blocked_result(
                memory,
                reason="the broad exploration budget has been exhausted",
                category="broad_budget",
            )

        if evidence_gathered and self.exploration_attempts >= _exploration_budget(memory.task_type):
            if _is_targeted_escape_call(call, known_paths) and self.targeted_escape_used < 1:
                self.targeted_escape_used += 1
                self._count_attempt(broad=broad)
                self.seen_exploration_signatures.add(signature)
                return None
            return _harness_blocked_result(
                memory,
                reason="the locate/read budget has been exhausted after actionable evidence was gathered",
                category="read_budget",
            )

        self._count_attempt(broad=broad)
        self.seen_exploration_signatures.add(signature)
        return None

    def after_tool_call(
        self,
        call: ToolCall,
        result: ToolResult,
        *,
        memory: WorkingMemory,
        changed_paths: list[str],
    ) -> None:
        if changed_paths or _is_mutating_tool(call.name) or call.name in {"verify", "finish_task"}:
            self.low_gain_streak = 0
            self.observed_paths.update(_normalize_repo_path(path) for path in changed_paths if path)
            return
        if not _is_exploration_tool(call.name) or call.name in {"read_conversation_memory", "git_status", "git_diff"}:
            return

        text = _tool_result_text(result)
        discovered = _extract_repo_like_paths(text)
        new_paths = [path for path in discovered if path not in self.observed_paths]
        self.observed_paths.update(discovered)
        known_paths = _known_harness_paths(memory)
        targeted = _is_targeted_escape_call(call, known_paths) if known_paths else False
        if result.ok and (new_paths or targeted) and not _result_is_empty_or_negative(text):
            self.low_gain_streak = 0
            return
        self.low_gain_streak += 1

    def _count_attempt(self, *, broad: bool) -> None:
        self.exploration_attempts += 1
        if broad:
            self.broad_attempts += 1


def _harness_blocked_result(
    memory: WorkingMemory,
    *,
    reason: str,
    category: str = "exploration_budget",
) -> ToolResult:
    paths = ", ".join(_known_harness_paths(memory)[:8]) or "(none)"
    return ToolResult(
        False,
        error=(
            f"Harness blocked low-yield exploration ({category}): "
            f"{reason}. Known target/candidate files: {paths}. "
            "Use the already gathered evidence to choose the next action. Good next moves are: "
            "apply_patch if the target is clear, verify if files changed, finish_task for analysis-only work, "
            "or one exact read_file/read_many call on a known target path if a precise line range is still missing."
        ),
        meta={"harness": category, "task_type": memory.task_type},
    )


def _exploration_budget(task_type: str) -> int:
    return {
        "ui": 8,
        "docs": 7,
        "test": 10,
        "coding": 12,
    }.get(task_type, 12)


def _broad_exploration_budget(task_type: str) -> int:
    return {
        "ui": 6,
        "docs": 5,
        "test": 7,
        "coding": 8,
    }.get(task_type, 8)


def _evidence_is_actionable(memory: WorkingMemory, evidence_gathered: bool) -> bool:
    return bool(
        evidence_gathered
        and (
            memory.target_paths
            or memory.modified_paths
            or memory.files
            or memory.candidate_files
        )
    )


def _known_harness_paths(memory: WorkingMemory) -> list[str]:
    paths: list[str] = []
    for path in [
        *memory.target_paths,
        *memory.candidate_files,
        *memory.modified_paths,
        *(item.path for item in memory.files),
    ]:
        normalized = _normalize_repo_path(path)
        if normalized and normalized not in paths:
            paths.append(normalized)
    return paths


def _is_broad_exploration_call(call: ToolCall, known_paths: list[str]) -> bool:
    if call.name == "list_files":
        return True
    if call.name == "repo_graph_query":
        query = str(call.arguments.get("query") or "")
        return not _text_targets_known_path(query, known_paths)
    if call.name == "repo_graph_neighborhood":
        return not _argument_path_is_known(call.arguments.get("path"), known_paths)
    if call.name == "search_text":
        file_pattern = _normalize_repo_path(call.arguments.get("file_pattern") or "*")
        if file_pattern in {"", "*", "**/*"}:
            return True
        return not _path_pattern_targets_known(file_pattern, known_paths)
    if call.name == "read_file":
        return not _read_request_is_targeted(call.arguments, known_paths)
    if call.name == "read_many":
        requests = call.arguments.get("requests")
        if not isinstance(requests, list) or not requests:
            return True
        return any(not isinstance(item, dict) or not _read_request_is_targeted(item, known_paths) for item in requests)
    return False


def _is_targeted_escape_call(call: ToolCall, known_paths: list[str]) -> bool:
    if call.name in {"read_file", "repo_graph_neighborhood", "search_text"}:
        return not _is_broad_exploration_call(call, known_paths)
    if call.name == "repo_graph_query":
        return not _is_broad_exploration_call(call, known_paths)
    if call.name == "read_many":
        return not _is_broad_exploration_call(call, known_paths)
    return False


def _semantic_exploration_signature(call: ToolCall) -> str:
    if call.name == "read_many":
        requests = call.arguments.get("requests")
        if isinstance(requests, list):
            parts = []
            for item in requests:
                if not isinstance(item, dict):
                    continue
                parts.append(
                    ":".join(
                        [
                            _normalize_repo_path(item.get("path")),
                            str(item.get("start", 1)),
                            str(item.get("end", "")),
                        ]
                    )
                )
            return "read_many:" + "|".join(sorted(parts))
    if call.name == "read_file":
        return "read_file:" + ":".join(
            [
                _normalize_repo_path(call.arguments.get("path")),
                str(call.arguments.get("start", 1)),
                str(call.arguments.get("end", "")),
            ]
        )
    if call.name == "search_text":
        return "search_text:" + ":".join(
            [
                _normalize_search_text(call.arguments.get("pattern")),
                _normalize_repo_path(call.arguments.get("file_pattern") or "*"),
                str(bool(call.arguments.get("regex"))),
            ]
        )
    if call.name == "repo_graph_query":
        return "repo_graph_query:" + _normalize_search_text(call.arguments.get("query"))
    if call.name == "repo_graph_neighborhood":
        return "repo_graph_neighborhood:" + _normalize_repo_path(call.arguments.get("path"))
    if call.name == "list_files":
        return "list_files:" + _normalize_repo_path(call.arguments.get("pattern") or "*")
    return call.name + ":" + _tool_signature(call.name, call.arguments)


def _normalize_search_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").casefold()).strip()
    return text[:240]


def _read_request_is_targeted(arguments: dict[str, Any], known_paths: list[str]) -> bool:
    if not _argument_path_is_known(arguments.get("path"), known_paths):
        return False
    start = arguments.get("start", 1)
    end = arguments.get("end")
    if not isinstance(start, int) or isinstance(start, bool) or start < 1:
        return False
    if end is None:
        return False
    if not isinstance(end, int) or isinstance(end, bool) or end < start:
        return False
    return (end - start) <= 260


def _argument_path_is_known(value: Any, known_paths: list[str]) -> bool:
    path = _normalize_repo_path(value)
    return bool(path and any(path == known or path.endswith("/" + known) or known.endswith("/" + path) for known in known_paths))


def _text_targets_known_path(text: str, known_paths: list[str]) -> bool:
    lowered = _normalize_repo_path(text).casefold()
    if not lowered:
        return False
    for known in known_paths:
        normalized = _normalize_repo_path(known).casefold()
        if not normalized:
            continue
        basename = Path(normalized).name
        if normalized in lowered or (basename and basename in lowered):
            return True
    return False


def _path_pattern_targets_known(pattern: str, known_paths: list[str]) -> bool:
    normalized = _normalize_repo_path(pattern).strip("*")
    if not normalized:
        return False
    return any(normalized in known or known in normalized for known in known_paths)


def _normalize_repo_path(value: Any) -> str:
    text = str(value or "").replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    return text


def _tool_result_text(result: ToolResult) -> str:
    value = result.data if result.ok else result.error
    if value is None:
        return ""
    return redact_secrets(str(value))


def _result_is_empty_or_negative(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(text or "").casefold()).strip()
    if not normalized:
        return True
    negative_markers = {
        "(no matches)",
        "(no files matched)",
        "(no graph hits)",
        "no matches",
        "not found",
        "file does not exist",
        "path is not indexed",
        "error:",
        "未找到",
        "没有匹配",
        "不存在",
    }
    return any(marker in normalized for marker in negative_markers)


def _extract_repo_like_paths(text: str) -> set[str]:
    results: set[str] = set()
    pattern = re.compile(
        r"(?P<path>(?:[A-Za-z0-9_.@~+-]+/)*[A-Za-z0-9_.@~+-]+"
        r"\.(?:py|pyw|js|jsx|ts|tsx|css|html|md|json|toml|yaml|yml|txt))"
    )
    for match in pattern.finditer(str(text or "")):
        path = _normalize_repo_path(match.group("path").strip("`'\"()[]{}:,;"))
        if path and not path.startswith("../"):
            results.add(path)
        if len(results) >= 80:
            break
    return results


class _ExplorationProgressGuard:
    def __init__(self, *, max_nudges: int = 2):
        self.read_only_turns = 0
        self.stagnant_turns = 0
        self.max_nudges = max(0, max_nudges)
        self.nudges = 0
        self.last_signal = ""

    def after_tool_turn(
        self,
        *,
        iteration: int,
        tool_names: list[str],
        changed_paths: list[str],
        memory: WorkingMemory,
        evidence_gathered: bool,
    ) -> str:
        if not tool_names:
            return ""
        if changed_paths or any(_is_mutating_tool(name) or name in {"verify", "finish_task"} for name in tool_names):
            self._reset()
            return ""
        if not all(_is_exploration_tool(name) for name in tool_names):
            return ""

        self.read_only_turns += 1
        signal = _semantic_progress_signal(memory)
        if signal == self.last_signal:
            self.stagnant_turns += 1
        else:
            self.stagnant_turns = 0
            self.last_signal = signal

        has_actionable_evidence = evidence_gathered or bool(memory.files or memory.target_paths or memory.candidate_files)
        threshold = 3 if has_actionable_evidence else 4
        if self.nudges >= self.max_nudges:
            return ""
        if self.read_only_turns < threshold and self.stagnant_turns < 2:
            return ""

        self.nudges += 1
        self.read_only_turns = 0
        self.stagnant_turns = 0
        candidates = ", ".join(memory.target_paths or memory.candidate_files or [item.path for item in memory.files])
        candidates = candidates or "(none)"
        return (
            "Exploration guard: recent locate/read tool calls have not produced a workspace change. "
            f"Current candidate files: {candidates}. "
            "If one candidate has enough evidence, stop broad searching and make the smallest patch. "
            "If evidence is still missing, use exactly one targeted tool call to answer the missing fact, then patch, verify, or finish."
        )

    def _reset(self) -> None:
        self.read_only_turns = 0
        self.stagnant_turns = 0
        self.last_signal = ""


def _phase_for_tool(tool_name: str) -> str:
    if tool_name == "record_progress":
        return "PLAN"
    if tool_name == "finish_task":
        return "REPORT"
    if tool_name in {"repo_graph_query", "repo_graph_neighborhood", "list_files", "search_text", "git_status", "git_diff"}:
        return "LOCATE"
    if tool_name in {"read_file", "read_many", "read_conversation_memory"}:
        return "READ"
    if tool_name in {"write_file", "apply_patch", "run_command"}:
        return "PATCH"
    if tool_name == "verify":
        return "VERIFY"
    return "PLAN"


def _is_code_evidence_tool(tool_name: str) -> bool:
    return tool_name in {"repo_graph_query", "repo_graph_neighborhood", "list_files", "read_file", "read_many", "search_text"}


def _is_exploration_tool(tool_name: str) -> bool:
    return tool_name in {
        "repo_graph_query",
        "repo_graph_neighborhood",
        "list_files",
        "read_file",
        "read_many",
        "read_conversation_memory",
        "search_text",
        "git_status",
        "git_diff",
    }


def _is_mutating_tool(tool_name: str) -> bool:
    return tool_name in {"write_file", "apply_patch", "run_command"}


def _semantic_progress_signal(memory: WorkingMemory) -> str:
    parts = [
        memory.phase,
        memory.hypothesis,
        memory.next_step,
        ",".join(sorted(memory.target_paths)),
        ",".join(sorted(memory.modified_paths)),
        ",".join(sorted(item.path for item in memory.files)),
    ]
    return "\n".join(parts)


def _classify_task(task: str) -> str:
    text = task.casefold()
    mutation_terms = {
        "add",
        "change",
        "delete",
        "fix",
        "implement",
        "modify",
        "refactor",
        "remove",
        "修改",
        "优化",
        "改进",
        "实现",
        "添加",
        "删除",
        "修复",
        "重构",
        "美化",
        "功能",
    }
    if _contains_any(text, {"界面", "页面", "按钮", "前端", "样式", "美化", "ui", "html", "css", "browser", "web"}):
        return "ui"
    if _contains_any(text, {"测试", "验证", "unittest", "pytest", "coverage"}):
        return "test"
    if _contains_any(text, {"readme", "文档", "说明", "教程", "doc"}):
        return "docs"
    if _contains_any(text, {"之前聊", "历史对话", "继续会话", "会话记忆", "conversation memory"}):
        return "conversation"
    if _contains_any(text, {"讲解", "分析", "介绍", "是什么", "为什么", "如何"}) and not _contains_any(text, mutation_terms):
        return "analysis"
    return "coding"


def _candidate_files_for_task(task: str, repo_graph: RepoGraph, task_type: str) -> list[str]:
    candidates: list[str] = []
    paths = [node.path for node in repo_graph.files]
    path_set = set(paths)

    def add(path: str) -> None:
        clean = path.replace("\\", "/").strip()
        if clean and clean not in candidates and (clean in path_set or _workspace_file_exists(repo_graph.root, clean)):
            candidates.append(clean)

    if task_type == "ui":
        for path in [
            "tracegraph_coder/web/index.html",
            "tracegraph_coder/web/assets/app.js",
            "tracegraph_coder/web/assets/styles.css",
            "tracegraph_coder/web_app.py",
        ]:
            add(path)
        for path in paths:
            if "/web/" in path or path.endswith("/web_app.py") or path.endswith("web_app.py"):
                add(path)
    elif task_type == "docs":
        for path in paths:
            name = Path(path).name.casefold()
            if name in {"readme.md", "tracegraph.md", "agents.md"} or path.startswith("docs/"):
                add(path)
    elif task_type == "test":
        for path in paths:
            if "/test" in path.casefold() or Path(path).name.casefold().startswith("test_"):
                add(path)

    for node in repo_graph.query(task, limit=12):
        add(node.path)
        if len(candidates) >= 12:
            break
    return candidates[:12]


def _contains_any(text: str, terms: set[str]) -> bool:
    return any(term in text for term in terms)


def _workspace_file_exists(root: str | Path, path: str) -> bool:
    clean = _normalize_repo_path(path)
    if not clean or clean.startswith("../") or Path(clean).is_absolute():
        return False
    try:
        root_path = Path(root).resolve()
        target = (root_path / clean).resolve()
        if target == root_path or root_path not in target.parents:
            return False
        return target.is_file()
    except OSError:
        return False


def _seed_tool_signature_counts(messages: list[Message]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for message in messages:
        if message.role != "assistant" or not message.metadata:
            continue
        raw_calls = message.metadata.get("tool_calls")
        if not isinstance(raw_calls, list):
            continue
        for index, raw_call in enumerate(raw_calls):
            if not isinstance(raw_call, dict):
                continue
            try:
                call = ToolCall.from_wire(raw_call, index)
            except Exception:
                continue
            signature = _tool_signature(call.name, call.arguments)
            counts[signature] = counts.get(signature, 0) + 1
    return counts


def _tool_signature(tool_name: str, arguments: dict[str, Any]) -> str:
    payload = json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
    return f"{tool_name}:{payload}"


def _parse_verification_tool_result(result: ToolResult) -> VerificationResult | None:
    text = result.data if result.ok else result.error
    if not isinstance(text, str):
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return VerificationResult(
        ok=bool(payload.get("ok")),
        command=payload.get("command"),
        output=str(payload.get("output", "")),
    )


def _format_verification(result: VerificationResult | None) -> str:
    if result is None:
        return "Verification was not run."
    status = "passed" if result.ok else "failed"
    cmd = result.command or "(auto-detect found no command)"
    return f"status: {status}\ncommand: {cmd}\n\n{result.output}"


def _validate_control_arguments(
    arguments: dict[str, Any],
    *,
    allowed: set[str],
    required: set[str],
) -> str | None:
    if not isinstance(arguments, dict):
        return "arguments must be an object"
    missing = sorted(required - set(arguments))
    if missing:
        return "missing required argument(s): " + ", ".join(missing)
    extra = sorted(set(arguments) - allowed)
    if extra:
        return "unexpected argument(s): " + ", ".join(extra)
    for name, value in arguments.items():
        if name == "target_paths":
            if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                return "target_paths must be a list of strings"
        elif name == "experience_searches":
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                return "experience_searches must be a non-negative integer"
        elif not isinstance(value, str):
            return f"{name} must be a string"
    return None
