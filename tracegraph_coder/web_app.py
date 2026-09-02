from __future__ import annotations

import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
import textwrap
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .controller import AgentController
from .evidence import EvidenceLog
from .experience import ExperienceStore
from .llm import LLMConfig, OpenAICompatibleLLM
from .memory import load_project_memory
from .models import Message
from .repo_graph import RepoGraph, build_repo_graph
from .safety import ensure_workspace, redact_secrets
from .session import SessionStore, is_resumable_session
from .tools import ToolEnvironment, build_default_registry
from .verifier import VerificationResult, Verifier
from .working_memory import WorkingMemory


STATIC_ROOT = Path(__file__).with_name("web")
MAX_REQUEST_BYTES = 1_000_000
MAX_LOG_LINES = 1000


@dataclass
class RunState:
    lock: threading.RLock = field(default_factory=threading.RLock)
    running: bool = False
    run_id: str | None = None
    status: str = "就绪"
    phase: str = "IDLE"
    logs: list[str] = field(default_factory=list)
    final: str = ""
    verification: str = ""
    graph: str = ""
    evidence_chain: str = ""
    working_memory: str = ""
    report_path: str = ""
    session_id: str = ""
    error: str = ""
    workspace: str = ""
    files_indexed: int = 0
    iterations: int = 0
    started_at: float | None = None
    finished_at: float | None = None

    def start(self, run_id: str, workspace: Path) -> None:
        with self.lock:
            self.running = True
            self.run_id = run_id
            self.status = "准备运行"
            self.phase = "PLAN"
            self.logs = []
            self.final = ""
            self.verification = ""
            self.evidence_chain = ""
            self.working_memory = ""
            self.report_path = ""
            self.session_id = ""
            self.error = ""
            self.workspace = str(workspace)
            self.files_indexed = 0
            self.iterations = 0
            self.started_at = time.time()
            self.finished_at = None
        self.append_log(f"run {run_id} started in {workspace}")

    def clear(self) -> bool:
        with self.lock:
            if self.running:
                return False
            self.status = "就绪"
            self.phase = "IDLE"
            self.logs = []
            self.final = ""
            self.verification = ""
            self.graph = ""
            self.evidence_chain = ""
            self.working_memory = ""
            self.report_path = ""
            self.session_id = ""
            self.error = ""
            self.files_indexed = 0
            self.iterations = 0
            self.started_at = None
            self.finished_at = None
            return True

    def append_log(self, message: str) -> None:
        clean = redact_secrets(str(message))
        stamp = time.strftime("%H:%M:%S")
        with self.lock:
            phase = _phase_from_event(clean)
            if phase:
                self.phase = phase
                self.status = _status_for_phase(phase)
            self.logs.append(f"{stamp}  {clean}")
            if len(self.logs) > MAX_LOG_LINES:
                self.logs = self.logs[-MAX_LOG_LINES:]

    def update(self, **kwargs: Any) -> None:
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def finish(self, *, status: str, phase: str, error: str = "") -> None:
        with self.lock:
            self.running = False
            self.status = status
            self.phase = phase
            self.error = redact_secrets(error)
            self.finished_at = time.time()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            elapsed = _elapsed_seconds(self.started_at, self.finished_at)
            return {
                "running": self.running,
                "runId": self.run_id,
                "status": self.status,
                "phase": self.phase,
                "logs": list(self.logs),
                "final": self.final,
                "verification": self.verification,
                "graph": self.graph,
                "evidenceChain": self.evidence_chain,
                "workingMemory": self.working_memory,
                "reportPath": self.report_path,
                "sessionId": self.session_id,
                "error": self.error,
                "workspace": self.workspace,
                "filesIndexed": self.files_indexed,
                "iterations": self.iterations,
                "elapsedSeconds": elapsed,
            }


class TraceGraphHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        default_workspace: Path,
        state: RunState | None = None,
    ):
        super().__init__(server_address, handler_class)
        self.default_workspace = default_workspace
        self.state = state or RunState()


class TraceGraphRequestHandler(BaseHTTPRequestHandler):
    server: TraceGraphHTTPServer

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/api/defaults":
            self._send_json(_default_payload(self.server.default_workspace))
            return
        if route == "/api/state":
            self._send_json({"ok": True, "state": self.server.state.snapshot()})
            return
        if route == "/api/browse":
            self._handle_browse()
            return
        self._serve_static(route)

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        try:
            if route == "/api/run":
                self._handle_run()
                return
            if route == "/api/graph":
                self._handle_graph()
                return
            if route == "/api/clear":
                self._handle_clear()
                return
            if route == "/api/session/latest":
                self._handle_session_latest()
                return
            if route == "/api/sessions":
                self._handle_sessions()
                return
            if route == "/api/session/detail":
                self._handle_session_detail()
                return
            if route == "/api/resume":
                self._handle_resume()
                return
            if route == "/api/open-report":
                self._handle_open_report()
                return
            if route == "/api/shutdown":
                self._handle_shutdown()
                return
            self._send_json({"ok": False, "error": "Unknown API route."}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"ok": False, "error": redact_secrets(str(exc))}, HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _handle_browse(self) -> None:
        """Open a native folder selection dialog and return the chosen path."""
        try:
            selected = _choose_workspace_folder()
        except Exception as exc:
            self._send_json({"ok": False, "error": redact_secrets(str(exc))}, HTTPStatus.BAD_REQUEST)
            return

        if not selected:
            self._send_json({"ok": True, "cancelled": True})
            return

        self._send_json({"ok": True, "path": selected})

    def _handle_run(self) -> None:
        payload = _read_json(self)
        task = str(payload.get("task") or "").strip()
        if not task:
            self._send_json({"ok": False, "error": "请先输入任务描述。"}, HTTPStatus.BAD_REQUEST)
            return
        workspace = _workspace_from_payload(payload, self.server.default_workspace)
        with self.server.state.lock:
            if self.server.state.running:
                self._send_json({"ok": False, "error": "已有任务正在运行。"}, HTTPStatus.CONFLICT)
                return
        run_id = uuid.uuid4().hex[:12]
        payload["workspace"] = str(workspace)
        self.server.state.start(run_id, workspace)
        thread = threading.Thread(
            target=_run_agent_job,
            args=(self.server.state, payload),
            name=f"tracegraph-run-{run_id}",
            daemon=True,
        )
        thread.start()
        self._send_json({"ok": True, "runId": run_id})

    def _handle_resume(self) -> None:
        payload = _read_json(self)
        workspace = _workspace_from_payload(payload, self.server.default_workspace)
        with self.server.state.lock:
            if self.server.state.running:
                self._send_json({"ok": False, "error": "已有任务正在运行。"}, HTTPStatus.CONFLICT)
                return
        store = SessionStore(workspace)
        selector = str(payload.get("sessionId") or "latest").strip() or "latest"
        session = store.latest_resumable() if selector == "latest" else store.load(selector)
        if session is None:
            self._send_json({"ok": False, "error": "没有可继续的历史对话。"}, HTTPStatus.NOT_FOUND)
            return
        if not session.messages:
            self._send_json({"ok": False, "error": "该对话没有可恢复的消息历史。"}, HTTPStatus.BAD_REQUEST)
            return
        follow_up = str(payload.get("followUp") or "").strip()
        if session.status == "completed" and not follow_up:
            self._send_json({"ok": False, "error": "该对话上一次已经完成，请在任务框输入新的追问后继续。"}, HTTPStatus.BAD_REQUEST)
            return

        run_id = uuid.uuid4().hex[:12]
        payload["workspace"] = str(workspace)
        payload["task"] = session.task
        payload["resumeSessionId"] = session.session_id
        self.server.state.start(run_id, workspace)
        self.server.state.update(
            status="恢复会话",
            phase="PLAN",
            session_id=session.session_id,
            iterations=session.iterations,
            working_memory=WorkingMemory.from_dict(session.working_memory).render(),
        )
        self.server.state.append_log(f"resuming session {session.session_id}")
        thread = threading.Thread(
            target=_run_agent_job,
            args=(self.server.state, payload),
            name=f"tracegraph-resume-{run_id}",
            daemon=True,
        )
        thread.start()
        self._send_json({"ok": True, "runId": run_id, "sessionId": session.session_id})

    def _handle_graph(self) -> None:
        payload = _read_json(self)
        with self.server.state.lock:
            if self.server.state.running:
                self._send_json({"ok": False, "error": "运行中不能重建仓库图。"}, HTTPStatus.CONFLICT)
                return
        workspace = _workspace_from_payload(payload, self.server.default_workspace)
        graph = _build_and_save_graph(workspace)
        graph_text = graph.format_for_prompt(max_files=220)
        self.server.state.update(
            status="仓库图已更新",
            phase="LOCATE",
            graph=graph_text,
            workspace=str(workspace),
            files_indexed=len(graph.files),
        )
        self.server.state.append_log(f"repo graph indexed {len(graph.files)} files")
        self._send_json({"ok": True, "filesIndexed": len(graph.files), "graph": graph_text})

    def _handle_clear(self) -> None:
        if not self.server.state.clear():
            self._send_json({"ok": False, "error": "运行中不能清空输出。"}, HTTPStatus.CONFLICT)
            return
        self._send_json({"ok": True})

    def _handle_session_latest(self) -> None:
        payload = _read_json(self)
        workspace = _workspace_from_payload(payload, self.server.default_workspace)
        session = _latest_resumable_session(workspace)
        self._send_json({"ok": True, "session": _session_payload(session) if session else None})

    def _handle_sessions(self) -> None:
        payload = _read_json(self)
        workspace = _workspace_from_payload(payload, self.server.default_workspace)
        sessions = _recent_sessions(workspace)
        self._send_json(
            {
                "ok": True,
                "sessions": [_session_payload(session) for session in sessions],
                "tree": _session_tree_payload(workspace),
            }
        )

    def _handle_session_detail(self) -> None:
        payload = _read_json(self)
        workspace = _workspace_from_payload(payload, self.server.default_workspace)
        session_id = str(payload.get("sessionId") or "").strip()
        if not session_id:
            self._send_json({"ok": False, "error": "缺少会话 ID。"}, HTTPStatus.BAD_REQUEST)
            return
        try:
            session = SessionStore(workspace).load(session_id, include_blobs=True)
        except ValueError:
            self._send_json({"ok": False, "error": "会话不存在或已损坏。"}, HTTPStatus.NOT_FOUND)
            return
        self._send_json({"ok": True, "session": _session_detail_payload(session)})

    def _handle_open_report(self) -> None:
        snapshot = self.server.state.snapshot()
        report_path = snapshot.get("reportPath")
        if not report_path:
            self._send_json({"ok": False, "error": "还没有可打开的报告。"}, HTTPStatus.NOT_FOUND)
            return
        path = Path(str(report_path)).resolve()
        if not path.exists():
            self._send_json({"ok": False, "error": f"报告不存在: {path}"}, HTTPStatus.NOT_FOUND)
            return
        _open_local_path(path)
        self._send_json({"ok": True, "path": str(path)})

    def _handle_shutdown(self) -> None:
        self._send_json({"ok": True})
        threading.Thread(target=self.server.shutdown, daemon=True).start()

    def _serve_static(self, route: str) -> None:
        path = _static_path(route)
        if path is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix == ".js":
            content_type = "text/javascript"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, Any], status: int | HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def _run_agent_job(state: RunState, payload: dict[str, Any]) -> None:
    evidence: EvidenceLog | None = None
    graph: RepoGraph | None = None
    active_session = None
    try:
        workspace = ensure_workspace(str(payload["workspace"]))
        session_store = SessionStore(workspace)
        resume_session_id = str(payload.get("resumeSessionId") or "").strip()
        resume_session = session_store.load(resume_session_id) if resume_session_id else None
        follow_up = str(payload.get("followUp") or "").strip()
        active_session = resume_session
        task = active_session.task if active_session else str(payload["task"]).strip()
        config = _llm_config_from_payload(payload)
        llm = OpenAICompatibleLLM(config)

        graph = RepoGraph(root=str(workspace), files=[])
        state.update(
            status="准备 Agent",
            phase="PLAN",
            graph=_lazy_graph_notice(workspace),
            files_indexed=0,
            workspace=str(workspace),
        )
        state.append_log("repository graph deferred until a repository graph tool is selected")

        verifier = Verifier(workspace)
        project_memory = load_project_memory(workspace)
        conversation_memory = (
            _conversation_memory_text(active_session, follow_up=follow_up)
            if active_session is not None
            else ""
        )
        env = ToolEnvironment(
            workspace,
            graph,
            verifier,
            conversation_memory=conversation_memory,
            repo_graph_ready=False,
        )
        registry = build_default_registry(env)
        initial_memory = WorkingMemory.from_dict(active_session.working_memory) if active_session else None
        initial_messages = [Message.from_dict(item) for item in active_session.messages] if active_session else None
        initial_iteration = active_session.iterations if active_session else 0
        is_follow_up_run = bool(resume_session and follow_up)
        if resume_session and follow_up and initial_messages is not None and initial_memory is not None:
            _append_saved_final_answer_if_missing(initial_messages, active_session.final_text)
            initial_messages.append(Message(role="user", content=follow_up))
            _prepare_continuation_memory(initial_memory, base_task=active_session.task, follow_up=follow_up)
            state.append_log("follow-up appended to the same conversation")
        elif resume_session:
            state.append_log("resuming conversation without a new user message")
        if active_session is not None:
            state.update(
                session_id=active_session.session_id,
                iterations=active_session.iterations,
                working_memory=WorkingMemory.from_dict(active_session.working_memory).render(),
            )

        if active_session is None:
            active_session = session_store.create_checkpoint(
                task=task,
                model=config.model,
                working_memory=WorkingMemory(task),
                messages=[],
                status="running",
                iterations=0,
            )
            state.update(session_id=active_session.session_id)

        evidence = EvidenceLog(workspace, active_session.tree_id or active_session.session_id)
        experience_store = ExperienceStore(workspace)

        def save_checkpoint(messages: list[Message], memory: WorkingMemory, iteration: int, status: str) -> None:
            nonlocal active_session
            try:
                if active_session is None:
                    active_session = session_store.create_checkpoint(
                        task=task,
                        model=config.model,
                        working_memory=memory,
                        messages=messages,
                        status=status,
                        iterations=iteration,
                    )
                else:
                    checkpoint_final_text = ""
                    checkpoint_report_path = ""
                    checkpoint_verification = ""
                    if not is_follow_up_run:
                        checkpoint_final_text = active_session.final_text
                        checkpoint_report_path = active_session.report_path
                        checkpoint_verification = active_session.verification
                    active_session = session_store.update_checkpoint(
                        active_session.session_id,
                        task=task,
                        model=config.model,
                        working_memory=memory,
                        messages=messages,
                        status=status,
                        final_text=checkpoint_final_text,
                        report_path=checkpoint_report_path,
                        verification=checkpoint_verification,
                        iterations=iteration,
                    )
                state.update(
                    session_id=active_session.session_id,
                    iterations=iteration,
                    working_memory=memory.render(),
                )
            except Exception as exc:
                state.append_log(f"checkpoint skipped: {redact_secrets(str(exc))}")

        controller = AgentController(
            llm=llm,
            registry=registry,
            repo_graph=graph,
            evidence=evidence,
            verifier=verifier,
            project_memory=project_memory,
            max_steps=_bounded_int(payload.get("maxSteps"), default=20, minimum=1, maximum=80),
            auto_verify=bool(payload.get("autoVerify", True)),
            experience_store=experience_store,
            on_event=state.append_log,
        )
        state.append_log("agent controller started")
        result = controller.run(
            task,
            initial_messages=initial_messages,
            initial_memory=initial_memory,
            initial_iteration=initial_iteration,
            reset_evidence=resume_session is None,
            on_checkpoint=save_checkpoint,
        )
        verification = _format_verification(result.verification)
        evidence_chain = evidence.format_chain(80)
        working_memory = result.working_memory.render() if result.working_memory else ""
        graph = controller.repo_graph
        final_iterations = max(result.iterations, active_session.iterations if active_session else 0)
        if active_session is None:
            session = session_store.save_run(
                task=task,
                model=config.model,
                final_text=result.final_text.strip(),
                report_path=result.report_path,
                verification=verification,
                working_memory=result.working_memory,
                messages=result.messages or [],
                iterations=final_iterations,
            )
        else:
            session = session_store.update_checkpoint(
                active_session.session_id,
                task=task,
                model=config.model,
                final_text=result.final_text.strip(),
                report_path=result.report_path,
                verification=verification,
                working_memory=result.working_memory,
                messages=result.messages or [],
                status="completed",
                iterations=final_iterations,
            )
        state.update(
            final=result.final_text.strip(),
            verification=verification,
            evidence_chain=evidence_chain,
            working_memory=working_memory,
            report_path=str(result.report_path),
            session_id=session.session_id,
            iterations=final_iterations,
            graph=graph.format_for_prompt(max_files=220) if graph.files else _lazy_graph_notice(workspace),
            files_indexed=len(graph.files),
        )
        state.append_log(f"final report written: {result.report_path}")
        state.finish(status="完成", phase="REPORT")
    except Exception as exc:
        message = redact_secrets(str(exc))
        if evidence:
            try:
                state.update(evidence_chain=evidence.format_chain(80))
            except Exception:
                pass
        if graph:
            state.update(graph=graph.format_for_prompt(max_files=220), files_indexed=len(graph.files))
        state.append_log(f"error: {message}")
        state.finish(status="失败", phase="ERROR", error=message)


def _conversation_memory_text(
    session: Any,
    *,
    follow_up: str = "",
    limit: int = 40,
    char_budget: int = 18000,
) -> str:
    try:
        memory = WorkingMemory.from_dict(session.working_memory)
        memory_text = memory.render()
    except Exception:
        memory_text = "(working memory could not be loaded)"
    messages: list[Message] = []
    for item in session.messages:
        try:
            messages.append(Message.from_dict(item))
        except Exception:
            continue
    sections = [
        "Original task:\n" + _compact_context_text(getattr(session, "task", ""), 1600),
        "Latest user request:\n" + (_compact_context_text(follow_up, 1600) if follow_up else "(none)"),
        "Previous final answer:\n" + _compact_context_text(getattr(session, "final_text", "") or "(none)", 5000),
        "Working memory:\n" + _compact_context_text(memory_text, 5000),
        "Recent conversation:\n" + _conversation_context_excerpt(messages, limit=limit, char_budget=9000),
    ]
    return _compact_context_text(redact_secrets("\n\n---\n\n".join(sections)), char_budget)


def _conversation_context_excerpt(messages: list[Message], *, limit: int, char_budget: int) -> str:
    rows: list[str] = []
    remaining = max(1000, char_budget)
    for message in messages[-limit:]:
        excerpt = _message_context_excerpt(message)
        if not excerpt:
            continue
        if len(excerpt) > remaining:
            excerpt = excerpt[:remaining].rstrip() + "\n...[truncated]"
        rows.append(excerpt)
        remaining -= len(excerpt)
        if remaining <= 0:
            break
    return "\n\n".join(rows) or "(no saved conversation messages)"


def _message_context_excerpt(message: Message) -> str:
    role = message.role
    content = _clean_saved_user_content(message.content) if role == "user" else message.content
    metadata = message.metadata or {}
    if role == "system":
        return ""
    if role == "assistant":
        tool_calls = metadata.get("tool_calls") if isinstance(metadata, dict) else None
        tool_names = []
        if isinstance(tool_calls, list):
            for call in tool_calls:
                fn = call.get("function", {}) if isinstance(call, dict) else {}
                if isinstance(fn, dict):
                    tool_names.append(str(fn.get("name") or "tool"))
        label = "模型"
        suffix = f" 工具调用: {', '.join(tool_names[:6])}" if tool_names else ""
        return f"{label}{suffix}:\n{redact_secrets(_compact_context_text(content, 1200))}"
    if role == "tool":
        call_id = metadata.get("tool_call_id", "") if isinstance(metadata, dict) else ""
        return f"工具结果 {call_id}:\n{redact_secrets(_compact_tool_result(content, 900))}"
    return f"用户:\n{redact_secrets(_compact_context_text(content, 1200))}"


def _compact_tool_result(content: str, limit: int) -> str:
    text = str(content or "")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return _compact_context_text(text, limit)
    ok = payload.get("ok")
    body = payload.get("data") if ok else payload.get("error")
    meta = payload.get("meta") or {}
    prefix = f"ok={ok}"
    if isinstance(meta, dict) and meta:
        prefix += f" meta={json.dumps(meta, ensure_ascii=False)[:240]}"
    return prefix + "\n" + _compact_context_text(str(body or ""), limit)


def _compact_context_text(content: str, limit: int) -> str:
    text = _clean_saved_user_content(str(content or "")).replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]"


def _clean_saved_user_content(content: str) -> str:
    text = str(content or "")
    prefixes = [
        "Follow-up request after branching:\n",
        "Follow-up request:\n",
    ]
    for prefix in prefixes:
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return text.strip()


def _append_saved_final_answer_if_missing(messages: list[Message], final_text: str) -> None:
    final = str(final_text or "").strip()
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
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _is_internal_control_message(message: Message) -> bool:
    metadata = message.metadata if isinstance(message.metadata, dict) else {}
    return bool(metadata.get("control") or metadata.get("harness") or metadata.get("internal"))


def _build_and_save_graph(workspace: Path) -> RepoGraph:
    graph = build_repo_graph(workspace)
    graph.save(workspace / ".tracegraph" / "repo_graph.json")
    return graph


def _lazy_graph_notice(workspace: Path) -> str:
    return (
        "Repository evidence graph is lazy for this run.\n"
        f"- root: {workspace}\n"
        "- files indexed: 0\n"
        "- Use repo_graph_query or repo_graph_neighborhood when repository/code evidence is needed.\n"
        "- Conversation-only follow-ups can be answered from read_conversation_memory without scanning files."
    )


def _prepare_continuation_memory(memory: WorkingMemory, *, base_task: str, follow_up: str) -> None:
    follow_up = follow_up.strip()
    if not follow_up:
        return
    memory.task = f"{base_task.strip()}\n\nLatest user request:\n{follow_up}"
    memory.phase = "exploring"
    memory.hypothesis = ""
    memory.next_step = ""


def _default_payload(default_workspace: Path) -> dict[str, Any]:
    env_key_name = _first_env_key_name()
    latest_session = _latest_resumable_session(default_workspace)
    return {
        "workspace": str(default_workspace.resolve()),
        "model": _default_model(),
        "baseUrl": _default_base_url(),
        "hasEnvKey": env_key_name is not None,
        "envKeyName": env_key_name or "",
        "latestSession": _session_payload(latest_session) if latest_session else None,
        "maxSteps": 20,
        "autoVerify": True,
    }


def _latest_resumable_session(workspace: Path):
    try:
        return SessionStore(workspace).latest_resumable(include_blobs=False)
    except Exception:
        return None


def _recent_sessions(workspace: Path):
    try:
        return [
            session
            for session in SessionStore(workspace).list_heads(limit=20, include_blobs=False)
            if is_resumable_session(session)
        ]
    except Exception:
        return []


def _session_tree_payload(workspace: Path) -> dict[str, Any]:
    try:
        store = SessionStore(workspace)
        nodes = store.list_recent(limit=500, include_blobs=False)
    except Exception:
        return {"nodes": [], "heads": [], "roots": []}
    visible_nodes, display_parent_by_id = _conversation_tree_nodes(nodes)
    child_parent_ids = {display_parent_by_id.get(session.session_id) for session in visible_nodes}
    child_parent_ids.discard("")
    child_parent_ids.discard(None)
    root_ids = [session.session_id for session in visible_nodes if not display_parent_by_id.get(session.session_id)]
    head_ids = [session.session_id for session in visible_nodes if session.session_id not in child_parent_ids]
    return {
        "nodes": [
            _session_payload_for_tree(session, parent_id=display_parent_by_id.get(session.session_id, ""))
            for session in visible_nodes
        ],
        "heads": head_ids,
        "roots": root_ids,
    }


def _conversation_tree_nodes(sessions: list[Any]) -> tuple[list[Any], dict[str, str]]:
    by_id = {session.session_id: session for session in sessions}
    child_parent_ids = {session.parent_id for session in sessions if session.parent_id}
    raw_heads = {session.session_id for session in sessions if session.session_id not in child_parent_ids}
    synthetic_roots = {
        session.session_id
        for session in sessions
        if session.event_type == "root"
        and not session.messages
        and any(
            child.parent_id == session.session_id and child.event_type in {"checkpoint", "completed"}
            for child in sessions
        )
    }
    visible_ids: set[str] = set()
    for session in sessions:
        if session.session_id in synthetic_roots:
            continue
        if session.event_type in {"checkpoint", "completed"}:
            if session.session_id in raw_heads:
                visible_ids.add(session.session_id)
            continue
        visible_ids.add(session.session_id)

    parent_by_id = {
        session.session_id: _nearest_visible_parent(session, by_id, visible_ids)
        for session in sessions
        if session.session_id in visible_ids
    }
    return [session for session in sessions if session.session_id in visible_ids], parent_by_id


def _nearest_visible_parent(session: Any, by_id: dict[str, Any], visible_ids: set[str]) -> str:
    parent_id = session.parent_id or ""
    seen: set[str] = set()
    while parent_id and parent_id not in seen:
        if parent_id in visible_ids:
            return parent_id
        seen.add(parent_id)
        parent = by_id.get(parent_id)
        if parent is None:
            break
        parent_id = parent.parent_id or ""
    return ""


def _session_payload_for_tree(session: Any, *, parent_id: str) -> dict[str, Any]:
    payload = _session_payload(session)
    payload["parentId"] = parent_id
    if session.event_type in {"checkpoint", "completed"}:
        payload["eventType"] = "conversation"
        payload["summary"] = payload["summary"].replace("checkpoint", "conversation")
    return payload


def _session_payload(session) -> dict[str, Any]:
    continuable = is_resumable_session(session)
    return {
        "sessionId": session.session_id,
        "treeId": session.tree_id or session.session_id,
        "parentId": session.parent_id or "",
        "workspaceRoot": redact_secrets(session.workspace_root),
        "eventType": session.event_type,
        "summary": redact_secrets(session.summary),
        "status": session.status,
        "resumable": continuable,
        "continuable": continuable,
        "task": redact_secrets(session.task),
        "model": redact_secrets(session.model),
        "createdAt": session.created_at,
        "updatedAt": session.updated_at,
        "iterations": session.iterations,
        "hasMessages": bool(session.messages),
        "reportPath": redact_secrets(session.report_path),
        "finalText": redact_secrets(session.final_text),
    }


def _session_detail_payload(session) -> dict[str, Any]:
    payload = _session_payload(session)
    payload["messageCount"] = len(session.messages)
    payload["messages"] = [_message_detail_payload(message) for message in session.messages]
    return payload


def _message_detail_payload(message: Any) -> dict[str, Any]:
    if isinstance(message, Message):
        payload = message.to_dict()
    elif isinstance(message, dict):
        payload = json.loads(json.dumps(message, ensure_ascii=False))
    else:
        payload = json.loads(json.dumps(getattr(message, "__dict__", {}), ensure_ascii=False))
    payload["content"] = redact_secrets(str(payload.get("content", "")))
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        payload["metadata"] = _redact_nested_value(metadata)
    elif metadata is not None:
        payload["metadata"] = _redact_nested_value(metadata)
    return payload


def _redact_nested_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, list):
        return [_redact_nested_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _redact_nested_value(item) for key, item in value.items()}
    return value


def _llm_config_from_payload(payload: dict[str, Any]) -> LLMConfig:
    api_key = str(payload.get("apiKey") or "").strip()
    model = str(payload.get("model") or "").strip()
    base_url = str(payload.get("baseUrl") or "").strip()
    use_env_key = bool(payload.get("useEnvKey", True))
    if api_key:
        return LLMConfig(
            api_key=api_key,
            model=model or _default_model(),
            base_url=base_url or _base_url_for_model(model),
        )
    if not use_env_key:
        raise RuntimeError("请填写 API Key，或启用环境变量密钥。")
    config = LLMConfig.from_env()
    if model:
        config.model = model
    if base_url:
        config.base_url = base_url
    return config


def _workspace_from_payload(payload: dict[str, Any], default_workspace: Path) -> Path:
    raw = str(payload.get("workspace") or "").strip()
    return ensure_workspace(raw or default_workspace)


def _choose_workspace_folder() -> str | None:
    script = textwrap.dedent(
        """
        import json
        from pathlib import Path
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            selected = filedialog.askdirectory(title="选择工作区文件夹")
        finally:
            root.destroy()

        selected_path = str(Path(selected).resolve()) if selected else ""
        print(json.dumps({"selected": selected_path}, ensure_ascii=False))
        """
    )
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        creationflags=creationflags,
    )
    if result.returncode != 0:
        stderr = redact_secrets((result.stderr or "").strip() or "文件夹选择窗口打开失败。")
        raise RuntimeError(stderr)
    stdout = (result.stdout or "").strip()
    if not stdout:
        return None
    data = json.loads(stdout)
    if not isinstance(data, dict):
        raise RuntimeError("文件夹选择窗口返回了无效结果。")
    selected = str(data.get("selected") or "").strip()
    return selected or None


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or "0")
    if length > MAX_REQUEST_BYTES:
        raise ValueError("Request body is too large.")
    raw = handler.rfile.read(length) if length else b"{}"
    if not raw:
        return {}
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("JSON body must be an object.")
    return data


def _static_path(route: str) -> Path | None:
    rel = "index.html" if route in {"", "/"} else route.lstrip("/")
    if not rel or rel.startswith("api/"):
        return None
    root = STATIC_ROOT.resolve()
    target = (root / rel).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return None
    if not target.is_file():
        return None
    return target


def _format_verification(result: VerificationResult | None) -> str:
    if result is None:
        return "Verification was not run."
    status = "passed" if result.ok else "failed"
    command = result.command or "(auto-detect found no command)"
    return f"status: {status}\ncommand: {command}\n\n{result.output}"


def _phase_from_event(message: str) -> str | None:
    match = re.search(r"\bphase\s+([A-Z]+)\b", message)
    return match.group(1) if match else None


def _status_for_phase(phase: str) -> str:
    return {
        "PLAN": "规划中",
        "LOCATE": "定位代码",
        "READ": "读取证据",
        "PATCH": "修改中",
        "VERIFY": "验证中",
        "REPORT": "生成报告",
        "ERROR": "失败",
        "IDLE": "就绪",
    }.get(phase, phase)


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _first_env_key_name() -> str | None:
    for name in ("TRACEGRAPH_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
        if os.getenv(name):
            return name
    return None


def _default_model() -> str:
    if os.getenv("TRACEGRAPH_MODEL"):
        return str(os.getenv("TRACEGRAPH_MODEL"))
    if os.getenv("DEEPSEEK_API_KEY") and not os.getenv("TRACEGRAPH_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        return "deepseek-chat"
    return "gpt-4o-mini"


def _default_base_url() -> str:
    if os.getenv("TRACEGRAPH_BASE_URL"):
        return str(os.getenv("TRACEGRAPH_BASE_URL"))
    if os.getenv("DEEPSEEK_API_KEY") and not os.getenv("TRACEGRAPH_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        return "https://api.deepseek.com/v1"
    return "https://api.openai.com/v1"


def _base_url_for_model(model: str) -> str:
    if "deepseek" in model.lower():
        return "https://api.deepseek.com/v1"
    return _default_base_url()


def _elapsed_seconds(started_at: float | None, finished_at: float | None) -> int:
    if started_at is None:
        return 0
    end = finished_at or time.time()
    return max(0, int(end - started_at))


def _open_local_path(path: Path) -> None:
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
    else:
        webbrowser.open(path.as_uri())


def launch(*, host: str = "127.0.0.1", port: int = 0, open_browser: bool = True) -> None:
    default_workspace = Path.cwd().resolve()
    server = TraceGraphHTTPServer(
        (host, port),
        TraceGraphRequestHandler,
        default_workspace=default_workspace,
    )
    actual_host, actual_port = server.server_address
    url = f"http://{actual_host}:{actual_port}/"
    if open_browser:
        threading.Timer(0.25, lambda: webbrowser.open(url)).start()
    print(f"TraceGraph Coder Web UI: {url}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="TraceGraph Coder local Web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args(argv)
    launch(host=args.host, port=args.port, open_browser=not args.no_open)


if __name__ == "__main__":
    main()
