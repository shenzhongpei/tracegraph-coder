from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .evidence import EvidenceLog
from .repo_graph import FileNode, RepoGraph
from .safety import redact_secrets
from .working_memory import WorkingMemory


DEFAULT_CODING_CONTEXT_BUDGET = 12_000
MAX_ATOM_CHARS = 1_200


@dataclass(frozen=True, slots=True)
class CodingContextAtom:
    kind: str
    content: str
    priority: int
    source_path: str = ""
    reason: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def cost(self) -> int:
        return len(self.render())

    def render(self) -> str:
        header = f"- [{self.kind}]"
        if self.source_path:
            header += f" {self.source_path}"
        if self.reason:
            header += f" ({self.reason})"
        body = _clip(self.content, MAX_ATOM_CHARS)
        return f"{header}\n  {body}"


@dataclass(frozen=True, slots=True)
class CompiledCodingContext:
    phase: str
    atoms: list[CodingContextAtom]
    omitted_atoms: int
    budget_chars: int

    def render(self) -> str:
        lines = [
            "Coding context packet:",
            f"- phase: {self.phase}",
            f"- selected_atoms: {len(self.atoms)}",
            f"- omitted_atoms: {self.omitted_atoms}",
            "",
            "Use this packet to choose the next local tool call. Treat it as routing context, not as a substitute for reading exact code before editing.",
            "",
        ]
        lines.extend(atom.render() for atom in self.atoms)
        lines.extend(
            [
                "",
                "This-turn guidance:",
                "- Locate with repo_graph_query/search_text when paths are uncertain.",
                "- If candidate or target files are already known, avoid broad searches and use read_many or exact ranges.",
                "- Use repo_graph_neighborhood before patching likely target files only when impact or related tests are unclear.",
                "- Read exact line ranges before patching.",
                "- Prefer apply_patch for existing files.",
                "- If the harness blocks exploration, treat it as private execution feedback: stop searching and patch, verify, or finish unless one exact missing range is needed.",
                "- Treat experience markers and failure terms as high-priority anchors when deciding which evidence must stay in view.",
                "- Do not quote harness/control messages in the user-facing final answer.",
                "- Explain tool calls and results in plain language, like a teammate talking through progress; do not paste raw JSON or full result blobs unless the user explicitly asks for them.",
                "- Verify after changes, then use finish_task only when the evidence gate is clear.",
            ]
        )
        return "\n".join(lines)


def compile_coding_context(
    *,
    task: str,
    repo_graph: RepoGraph,
    evidence: EvidenceLog,
    memory: WorkingMemory,
    project_memory: str = "",
    budget_chars: int = DEFAULT_CODING_CONTEXT_BUDGET,
) -> CompiledCodingContext:
    phase = _phase_from_memory(memory)
    atoms = _required_atoms(memory, phase, repo_graph.root)
    atoms.extend(_failure_control_atoms(memory))
    atoms.extend(_project_rule_atoms(project_memory))
    atoms.extend(_experience_marker_atoms(memory))
    atoms.extend(_repo_atoms(task, repo_graph, memory, phase))
    atoms.extend(_evidence_atoms(evidence, memory, phase))
    atoms.extend(_experience_atoms(memory))
    selected, omitted = _select_atoms(atoms, budget_chars)
    return CompiledCodingContext(phase=phase, atoms=selected, omitted_atoms=omitted, budget_chars=budget_chars)


def _required_atoms(memory: WorkingMemory, phase: str, workspace: str) -> list[CodingContextAtom]:
    blockers = memory.completion_blockers(workspace)
    gate_lines = [
        f"working phase: {memory.phase}",
        f"hypothesis: {memory.hypothesis or '(none)'}",
        f"next step: {memory.next_step or '(none)'}",
        "target paths: " + (", ".join(memory.target_paths) or "(none)"),
        "stale files: " + (", ".join(memory.stale_files) or "(none)"),
        "modified paths: " + (", ".join(memory.modified_paths) or "(none)"),
        f"latest mutation step: {memory.last_mutation_step if memory.last_mutation_step is not None else '(none)'}",
        f"latest verification step: {memory.last_verification_step if memory.last_verification_step is not None else '(none)'}",
        f"verification command: {memory.verification_command or '(none)'}",
        f"verification summary: {memory.verification_summary or '(none)'}",
    ]
    if blockers:
        gate_lines.append("completion blockers: " + "; ".join(blockers))
    return [
        CodingContextAtom(
            kind="working_memory",
            priority=100,
            content="\n".join(gate_lines),
            reason="hot state for current coding task",
            tags=(phase,),
        )
    ]


def _project_rule_atoms(project_memory: str) -> list[CodingContextAtom]:
    if not project_memory.strip():
        return []
    return [
        CodingContextAtom(
            kind="project_rules",
            priority=95,
            content=project_memory,
            reason="persistent repository instructions",
            tags=("PLAN", "PATCH", "VERIFY"),
        )
    ]


def _failure_control_atoms(memory: WorkingMemory) -> list[CodingContextAtom]:
    if not memory.failure_terms and not memory.failure_events:
        return []
    lines = [
        "Failure control packet:",
        "failure_terms: " + (", ".join(memory.failure_terms) or "(none)"),
        "recent_events:",
    ]
    if memory.failure_events:
        for event in memory.failure_events[-5:]:
            details: list[str] = []
            if event.paths:
                details.append("paths=" + ", ".join(event.paths[:4]))
            if event.terms:
                details.append("terms=" + ", ".join(event.terms[:4]))
            suffix = f" ({'; '.join(details)})" if details else ""
            lines.append(f"- step {event.step} [{event.source}/{event.category}]{suffix}: {event.summary or '(no summary)'}")
    else:
        lines.append("- (none)")
    lines.extend(
        [
            "control_policy:",
            "- Treat these events as private execution feedback, not user-facing content.",
            "- Avoid repeating the same failed exploration category unless a new exact target is available.",
            "- If post-mutation exploration was blocked, verify or do one exact targeted read on a known file.",
            "- Preserve evidence matching failure terms during context compression and next-step planning.",
        ]
    )
    return [
        CodingContextAtom(
            kind="failure_control",
            priority=98 if memory.failure_events else 94,
            content="\n".join(lines),
            reason="failure-aware control signals for planning and compression",
            tags=("PLAN", "LOCATE", "PATCH", "VERIFY"),
        )
    ]


def _repo_atoms(
    task: str,
    repo_graph: RepoGraph,
    memory: WorkingMemory,
    phase: str,
) -> list[CodingContextAtom]:
    query = " ".join(
        [
            task,
            memory.hypothesis,
            memory.next_step,
            " ".join(memory.target_paths),
            " ".join(memory.modified_paths),
        ]
    )
    query_hits = repo_graph.query(query, limit=12)
    nodes_by_path: dict[str, FileNode] = {node.path: node for node in query_hits}
    graph_by_path = {node.path: node for node in repo_graph.files}
    for node in query_hits[:6]:
        _add_neighbor_nodes(nodes_by_path, graph_by_path, node, per_relation=2)
    for path in [*memory.target_paths, *memory.modified_paths, *(item.path for item in memory.files)]:
        node = graph_by_path.get(path)
        if node is not None:
            nodes_by_path[path] = node
            _add_neighbor_nodes(nodes_by_path, graph_by_path, node, per_relation=4)

    atoms: list[CodingContextAtom] = []
    for node in nodes_by_path.values():
        atoms.append(
            CodingContextAtom(
                kind="repo_node",
                source_path=node.path,
                priority=_repo_priority(node, memory, phase),
                reason=_repo_reason(node, memory),
                content=_format_repo_node(node),
                tags=("LOCATE", "READ", "PATCH", "VERIFY"),
            )
        )
    return atoms


def _add_neighbor_nodes(
    nodes_by_path: dict[str, FileNode],
    graph_by_path: dict[str, FileNode],
    node: FileNode,
    *,
    per_relation: int,
) -> None:
    for related in [
        *node.imports_local[:per_relation],
        *node.imported_by[:per_relation],
        *node.related_tests[:per_relation],
        *node.related_sources[:per_relation],
    ]:
        related_node = graph_by_path.get(related)
        if related_node is not None:
            nodes_by_path[related] = related_node


def _repo_priority(node: FileNode, memory: WorkingMemory, phase: str) -> int:
    score = 60
    if node.path in memory.target_paths:
        score += 25
    if node.path in memory.modified_paths:
        score += 30
    if any(item.path == node.path for item in memory.files):
        score += 18
    if node.is_test and phase in {"VERIFY", "REPORT"}:
        score += 18
    if node.related_tests:
        score += 8
    return min(score, 99)


def _repo_reason(node: FileNode, memory: WorkingMemory) -> str:
    if node.path in memory.modified_paths:
        return "modified file"
    if node.path in memory.target_paths:
        return "target path"
    if any(item.path == node.path for item in memory.files):
        return "recently observed file"
    if node.is_test:
        return "related test candidate"
    return "task/query match"


def _format_repo_node(node: FileNode) -> str:
    symbols = ", ".join(f"{symbol.kind}:{symbol.name}@{symbol.line}" for symbol in node.symbols[:8]) or "(none)"
    imports = ", ".join(node.imports[:6]) or "(none)"
    imports_local = ", ".join(node.imports_local[:6]) or "(none)"
    imported_by = ", ".join(node.imported_by[:6]) or "(none)"
    calls = ", ".join(node.calls[:10]) or "(none)"
    related = ", ".join(node.related_tests[:5]) or "(none)"
    related_sources = ", ".join(node.related_sources[:5]) or "(none)"
    return "\n".join(
        [
            f"path: {node.path}",
            f"language: {node.language}",
            f"role: {node.role}",
            f"lines: {node.lines}",
            f"is_test: {node.is_test}",
            f"symbols: {symbols}",
            f"imports: {imports}",
            f"imports_local: {imports_local}",
            f"imported_by: {imported_by}",
            f"calls: {calls}",
            f"related_tests: {related}",
            f"related_sources: {related_sources}",
        ]
    )


def _evidence_atoms(evidence: EvidenceLog, memory: WorkingMemory, phase: str) -> list[CodingContextAtom]:
    rows = evidence.read_recent(24)
    if not rows:
        return []
    target_paths = set(memory.target_paths) | set(memory.modified_paths) | {item.path for item in memory.files}
    semantic_terms = _semantic_terms(memory)
    atoms: list[CodingContextAtom] = []
    for row in rows:
        action = str(row.get("action") or "")
        args = row.get("args") if isinstance(row.get("args"), dict) else {}
        path = str(args.get("path") or args.get("file_pattern") or "")
        relevant_path = path and (path in target_paths or any(path.endswith(target) for target in target_paths))
        row_text = _format_evidence_row(row).casefold()
        matched_terms = [term for term in semantic_terms if term and term.casefold() in row_text]
        priority = 52
        if action in {"verify", "auto_verify"}:
            priority = 88 if phase in {"VERIFY", "REPORT"} else 70
        elif action in {"apply_patch", "write_file", "run_command"}:
            priority = 78
        elif relevant_path:
            priority = 74
        elif action in {"read_file", "repo_graph_query", "search_text"}:
            priority = 62
        if matched_terms:
            priority = min(99, priority + 18)
            if action in {"verify", "auto_verify"} and not bool(row.get("ok", True)):
                priority = min(99, priority + 8)
        atoms.append(
            CodingContextAtom(
                kind="evidence",
                source_path=path,
                priority=priority,
                reason=(
                    f"step {row.get('iteration')} {action} ok={row.get('ok')}"
                    + (f" matched_terms={', '.join(matched_terms[:4])}" if matched_terms else "")
                ),
                content=_format_evidence_row(row),
                tags=(str(row.get("stage") or ""), phase, *tuple(matched_terms[:4])),
            )
        )
    return atoms


def _format_evidence_row(row: dict[str, Any]) -> str:
    args = json.dumps(row.get("args", {}), ensure_ascii=False)
    observation = str(row.get("observation") or "").replace("\n", " ")
    return "\n".join(
        [
            f"iteration: {row.get('iteration')}",
            f"stage: {row.get('stage')}",
            f"action: {row.get('action')}",
            f"ok: {row.get('ok')}",
            f"args: {_clip(args, 360)}",
            f"observation: {_clip(observation, 720)}",
        ]
    )


def _experience_atoms(memory: WorkingMemory) -> list[CodingContextAtom]:
    if not memory.experience_hint.strip():
        return []
    return [
        CodingContextAtom(
            kind="experience",
            priority=72,
            reason="verified but potentially stale historical hint",
            content=memory.experience_hint,
            tags=("PLAN", "LOCATE"),
        )
    ]


def _experience_marker_atoms(memory: WorkingMemory) -> list[CodingContextAtom]:
    experience_terms = _semantic_terms(memory)
    if not experience_terms:
        return []
    priority = 92 if memory.failure_terms else 88
    content_lines = [
        "Experience markers:",
        "experience_terms: " + (", ".join(memory.experience_terms) or "(none)"),
        "failure_terms: " + (", ".join(memory.failure_terms) or "(none)"),
        "Use these markers to keep failure-related evidence visible during compaction.",
    ]
    return [
        CodingContextAtom(
            kind="experience_markers",
            priority=priority,
            content="\n".join(content_lines),
            reason="semantic anchors from history and recent failure signals",
            tags=("LOCATE", "READ", "PATCH", "VERIFY"),
        )
    ]


def _select_atoms(
    atoms: list[CodingContextAtom],
    budget_chars: int,
) -> tuple[list[CodingContextAtom], int]:
    unique = _dedupe_atoms(atoms)
    unique.sort(key=lambda atom: (-atom.priority, atom.kind, atom.source_path))
    selected: list[CodingContextAtom] = []
    used = 0
    for atom in unique:
        cost = atom.cost + 1
        if selected and used + cost > budget_chars:
            continue
        selected.append(atom)
        used += cost
    return selected, max(0, len(unique) - len(selected))


def _semantic_terms(memory: WorkingMemory) -> list[str]:
    values = [*memory.experience_terms, *memory.failure_terms]
    seen: set[str] = set()
    terms: list[str] = []
    for value in values:
        text = str(value or "").strip().casefold()
        if len(text) < 2 or text in seen:
            continue
        seen.add(text)
        terms.append(text)
    return terms


def _dedupe_atoms(atoms: list[CodingContextAtom]) -> list[CodingContextAtom]:
    by_key: dict[tuple[str, str, str], CodingContextAtom] = {}
    for atom in atoms:
        key = (atom.kind, atom.source_path, atom.content[:180])
        current = by_key.get(key)
        if current is None or atom.priority > current.priority:
            by_key[key] = atom
    return list(by_key.values())


def _phase_from_memory(memory: WorkingMemory) -> str:
    return {
        "exploring": "LOCATE",
        "modifying": "PATCH",
        "verifying": "VERIFY",
        "ready": "REPORT",
    }.get(memory.phase, "PLAN")


def _clip(value: Any, limit: int = MAX_ATOM_CHARS) -> str:
    text = redact_secrets(str(value)).strip()
    if len(text) <= limit:
        return text
    marker = "... omitted ..."
    keep = max(0, limit - len(marker))
    head = (keep + 1) // 2
    tail = keep - head
    return f"{text[:head]}{marker}{text[-tail:] if tail else ''}"
