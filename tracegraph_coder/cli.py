from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .controller import AgentController
from .evidence import EvidenceLog
from .experience import ExperienceStore
from .llm import LLMConfig, OpenAICompatibleLLM
from .memory import load_project_memory
from .models import Message
from .repo_graph import build_repo_graph
from .safety import ensure_workspace
from .session import SessionStore
from .tools import ToolEnvironment, build_default_registry
from .verifier import Verifier
from .working_memory import WorkingMemory


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="TraceGraph Coder: evidence-driven coding agent.")
    parser.add_argument("task", nargs="*", help="Programming task for the agent.")
    parser.add_argument("--workspace", default=".", help="Workspace directory. Defaults to current directory.")
    parser.add_argument("--max-steps", type=int, default=20, help="Maximum model/tool iterations.")
    parser.add_argument("--model", help="Override TRACEGRAPH_MODEL.")
    parser.add_argument("--base-url", help="Override TRACEGRAPH_BASE_URL.")
    parser.add_argument("--no-auto-verify", action="store_true", help="Disable controller-level verification.")
    parser.add_argument("--no-experience", action="store_true", help="Disable verified experience memory.")
    parser.add_argument("--show-session", nargs="?", const="latest", help="Show saved session summary without API.")
    parser.add_argument("--resume-session", nargs="?", const="latest", help="Continue a saved unfinished session.")
    parser.add_argument("--show-experiences", action="store_true", help="Show verified experience cards without API.")
    parser.add_argument("--graph-only", action="store_true", help="Only build and print the repository graph.")
    args = parser.parse_args(argv)

    workspace = ensure_workspace(args.workspace)
    session_store = SessionStore(workspace)

    if args.show_session is not None:
        session = session_store.load(args.show_session)
        print(f"Session: {session.session_id}")
        print(f"Tree: {session.tree_id or session.session_id}")
        print(f"Parent: {session.parent_id or '(root)'}")
        print(f"Status: {session.status}")
        print(f"Task: {session.task}")
        print(f"Model: {session.model}")
        print(f"Report: {session.report_path}")
        print("\n=== Final ===")
        print(session.final_text)
        return

    if args.show_experiences:
        store = ExperienceStore(workspace)
        statuses = store.list_status()
        print("=== Experiences ===")
        if not statuses:
            print("No stored experiences.")
        for item in statuses:
            card = item["card"]
            state = "stale" if item["stale"] else "valid"
            suffix = f" ({item['reason']})" if item["reason"] else ""
            print(f"- {card['experience_id']} [{state}]{suffix}: {card['strategy']}")
        return

    graph = build_repo_graph(workspace)
    graph_path = workspace / ".tracegraph" / "repo_graph.json"
    graph.save(graph_path)

    if args.graph_only:
        print(graph.format_for_prompt(max_files=200))
        print(f"\nSaved graph: {graph_path}")
        return

    resume_session = None
    if args.resume_session is not None:
        if args.resume_session == "latest":
            resume_session = session_store.latest_resumable()
            if resume_session is None:
                raise SystemExit("No continuable conversation found.")
        else:
            resume_session = session_store.load(args.resume_session)
        if not resume_session.messages:
            raise SystemExit("Selected conversation has no saved message history.")

    resume_follow_up = " ".join(args.task).strip() if resume_session else ""
    if resume_session and resume_session.status == "completed" and not resume_follow_up:
        raise SystemExit("Selected conversation is completed. Provide a follow-up task to continue it.")
    task = resume_session.task if resume_session else " ".join(args.task).strip()
    if not task and resume_session is None:
        task = input("Task: ").strip()
    if not task:
        raise SystemExit("No task provided.")

    config = LLMConfig.from_env()
    if args.model:
        config.model = args.model
    if args.base_url:
        config.base_url = args.base_url

    llm = OpenAICompatibleLLM(config)
    verifier = Verifier(workspace)
    project_memory = load_project_memory(workspace)
    env = ToolEnvironment(workspace, graph, verifier)
    registry = build_default_registry(env)
    active_session = resume_session
    if active_session is None:
        active_session = session_store.create_checkpoint(
            task=task,
            model=config.model,
            working_memory=WorkingMemory(task),
            messages=[],
            status="running",
            iterations=0,
        )
    evidence = EvidenceLog(workspace, active_session.tree_id or active_session.session_id)

    controller = AgentController(
        llm=llm,
        registry=registry,
        repo_graph=graph,
        evidence=evidence,
        verifier=verifier,
        project_memory=project_memory,
        max_steps=args.max_steps,
        auto_verify=not args.no_auto_verify,
        experience_store=None if args.no_experience else ExperienceStore(workspace),
        on_event=lambda msg: print(msg, file=sys.stderr),
    )
    def save_checkpoint(messages, memory, iteration, status):
        nonlocal active_session
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
            active_session = session_store.update_checkpoint(
                active_session.session_id,
                task=task,
                model=config.model,
                working_memory=memory,
                messages=messages,
                status=status,
                final_text=active_session.final_text,
                report_path=active_session.report_path,
                verification=active_session.verification,
                iterations=iteration,
            )

    initial_messages = [Message.from_dict(item) for item in active_session.messages] if active_session and resume_session else None
    initial_memory = WorkingMemory.from_dict(active_session.working_memory) if active_session and resume_session else None
    if initial_messages is not None and initial_memory is not None and resume_follow_up:
        initial_messages.append(Message(role="user", content=f"Follow-up request:\n{resume_follow_up}"))
        _prepare_continuation_memory(initial_memory, base_task=active_session.task, follow_up=resume_follow_up)

    result = controller.run(
        task,
        initial_messages=initial_messages,
        initial_memory=initial_memory,
        initial_iteration=active_session.iterations if active_session and resume_session else 0,
        reset_evidence=resume_session is None,
        on_checkpoint=save_checkpoint,
    )
    verification_text = _format_verification_for_session(result.verification)
    final_iterations = max(result.iterations, active_session.iterations if active_session else 0)
    if active_session is None:
        session = session_store.save_run(
            task=task,
            model=config.model,
            final_text=result.final_text,
            report_path=result.report_path,
            verification=verification_text,
            working_memory=result.working_memory,
            messages=result.messages or [],
            iterations=final_iterations,
        )
    else:
        session = session_store.update_checkpoint(
            active_session.session_id,
            task=task,
            model=config.model,
            final_text=result.final_text,
            report_path=result.report_path,
            verification=verification_text,
            working_memory=result.working_memory,
            messages=result.messages or [],
            status="completed",
            iterations=final_iterations,
        )
    print("\n=== Final ===")
    print(result.final_text)
    if result.verification:
        print("\n=== Verification ===")
        print(f"ok={result.verification.ok} command={result.verification.command}")
    print(f"\nReport: {result.report_path}")
    print(f"Session: {session.session_id}")
    print(f"Tree: {session.tree_id or session.session_id}")


def _format_verification_for_session(result) -> str:
    if result is None:
        return "Verification was not run."
    status = "passed" if result.ok else "failed"
    command = result.command or "(auto-detect found no command)"
    return f"status: {status}\ncommand: {command}\n\n{result.output}"


def _prepare_continuation_memory(memory: WorkingMemory, *, base_task: str, follow_up: str) -> None:
    follow_up = follow_up.strip()
    if not follow_up:
        return
    memory.task = f"{base_task.strip()}\n\nLatest user request:\n{follow_up}"
    memory.phase = "exploring"
    memory.hypothesis = ""
    memory.next_step = ""


if __name__ == "__main__":
    main()
