from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tracegraph_coder.coding_context import compile_coding_context
from tracegraph_coder.context_window import (
    ContextBudgetTuner,
    estimate_context_chars,
    estimate_context_tokens,
    estimate_tool_schema_tokens,
    prepare_context_messages,
    prompt_tokens_from_usage,
)
from tracegraph_coder.controller import AgentController
from tracegraph_coder.evidence import EvidenceLog, EvidenceStep
from tracegraph_coder.experience import ExperienceStore
from tracegraph_coder.models import LLMResponse, Message
from tracegraph_coder.repo_graph import build_repo_graph
from tracegraph_coder import repo_graph as repo_graph_module
from tracegraph_coder.session import SessionStore
from tracegraph_coder import workspace_state as workspace_state_module
from tracegraph_coder.tools import ToolEnvironment, build_default_registry
from tracegraph_coder.verifier import VerificationResult, Verifier
from tracegraph_coder.working_memory import WorkingMemory


class FakeLLM:
    def __init__(self, responses: list[LLMResponse]):
        self.responses = responses
        self.calls = []

    def chat(self, messages, tools):
        self.calls.append((messages, tools))
        if not self.responses:
            return LLMResponse(content="done")
        return self.responses.pop(0)


def tool_response(name: str, args: dict, call_id: str) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ],
    )


class IntegratedStateTests(unittest.TestCase):
    def test_coding_context_routes_repo_nodes_evidence_and_tests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            (root / "app.py").write_text("def answer():\n    return 1\n", encoding="utf-8")
            (root / "tests" / "test_app.py").write_text(
                "from app import answer\n\n"
                "def test_answer():\n"
                "    assert answer() == 1\n",
                encoding="utf-8",
            )
            graph = build_repo_graph(root)
            evidence = EvidenceLog(root)
            memory = WorkingMemory("fix answer behavior")
            memory.observe_progress(
                {
                    "phase": "modifying",
                    "hypothesis": "answer() needs a targeted patch",
                    "next_step": "read app.py and verify tests",
                    "target_paths": ["app.py"],
                },
                1,
            )
            evidence.append(
                EvidenceStep(
                    iteration=1,
                    stage="read",
                    action="read_file",
                    args={"path": "app.py", "start": 1, "end": 2},
                    ok=True,
                    observation="1 | def answer(): 2 |     return 1",
                )
            )

            packet = compile_coding_context(
                task="fix answer behavior",
                repo_graph=graph,
                evidence=evidence,
                memory=memory,
                budget_chars=8_000,
            ).render()

            self.assertIn("Coding context packet", packet)
            self.assertIn("app.py", packet)
            self.assertIn("tests/test_app.py", packet)
            self.assertIn("read_file", packet)
            self.assertIn("Read exact line ranges before patching", packet)

    def test_coding_context_prioritizes_experience_markers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("def answer():\n    return 1\n", encoding="utf-8")
            graph = build_repo_graph(root)
            evidence = EvidenceLog(root)
            evidence.append(
                EvidenceStep(
                    iteration=3,
                    stage="verify",
                    action="auto_verify",
                    args={"command": "python -m unittest"},
                    ok=False,
                    observation="Traceback: timeout while checking app.py",
                )
            )
            memory = WorkingMemory("fix flaky timeout")
            memory.observe_progress(
                {
                    "experience_terms": ["timeout", "retry"],
                    "failure_terms": ["timeout"],
                },
                1,
            )

            packet = compile_coding_context(
                task="fix flaky timeout",
                repo_graph=graph,
                evidence=evidence,
                memory=memory,
                budget_chars=4_000,
            ).render()

            self.assertIn("Experience markers", packet)
            self.assertIn("failure_terms: timeout", packet)
            self.assertIn("matched_terms=timeout", packet)

    def test_coding_context_emits_failure_control_packet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("def answer():\n    return 1\n", encoding="utf-8")
            graph = build_repo_graph(root)
            evidence = EvidenceLog(root)
            memory = WorkingMemory("stop repeated exploration")
            memory.observe_failure_event(
                source="harness",
                category="low_novelty_exploration",
                summary="Broad search was blocked after useful candidates were already known.",
                step=2,
                terms=["low_novelty", "app"],
                paths=["app.py"],
            )

            packet = compile_coding_context(
                task="stop repeated exploration",
                repo_graph=graph,
                evidence=evidence,
                memory=memory,
                budget_chars=4_000,
            ).render()

            self.assertIn("Failure control packet", packet)
            self.assertIn("low_novelty_exploration", packet)
            self.assertIn("low_novelty", packet)
            self.assertIn("app.py", packet)
            self.assertIn("Avoid repeating", packet)

    def test_controller_injects_coding_context_packet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("def answer():\n    return 1\n", encoding="utf-8")
            graph = build_repo_graph(root)
            verifier = Verifier(root)
            env = ToolEnvironment(root, graph, verifier)
            registry = build_default_registry(env)
            evidence = EvidenceLog(root)
            llm = FakeLLM([LLMResponse(content="done")])
            controller = AgentController(
                llm=llm,
                registry=registry,
                repo_graph=graph,
                evidence=evidence,
                verifier=verifier,
                auto_verify=False,
            )

            controller.run("make answer return 42")
            first_messages = llm.calls[0][0]

            self.assertTrue(any("Coding context packet" in message.content for message in first_messages))
            self.assertTrue(any("app.py" in message.content for message in first_messages))

    def test_context_window_compacts_old_tool_results_without_mutating_history(self) -> None:
        original_big_result = "x" * 5000
        source = [
            Message(role="system", content="system"),
            Message(role="user", content="task"),
            Message(role="system", content="memory"),
        ]
        for index in range(12):
            call_id = f"call_{index}"
            source.append(
                Message(
                    role="assistant",
                    content="",
                    metadata={
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {"name": "read_file", "arguments": "{}"},
                            }
                        ]
                    },
                )
            )
            source.append(Message(role="tool", content=original_big_result, metadata={"tool_call_id": call_id}))

        view, report = prepare_context_messages(
            source,
            max_chars=18_000,
            keep_recent_groups=2,
            max_tool_result_chars=120,
        )

        self.assertTrue(report.compacted)
        self.assertGreater(report.compacted_tool_results + report.dropped_groups, 0)
        self.assertLess(estimate_context_chars(view), estimate_context_chars(source))
        self.assertEqual(source[4].content, original_big_result)
        self.assertIn("compacted", "\n".join(message.content for message in view).lower())

    def test_context_window_soft_compacts_old_tool_results_before_overflow(self) -> None:
        source = [
            Message(role="system", content="system"),
            Message(role="user", content="task"),
        ]
        for index in range(6):
            call_id = f"call_{index}"
            source.extend(
                [
                    Message(
                        role="assistant",
                        content="",
                        metadata={
                            "tool_calls": [
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": json.dumps({"path": f"src/{index}.py"}),
                                    },
                                }
                            ]
                        },
                    ),
                    Message(role="tool", content="noise line\n" * 220, metadata={"tool_call_id": call_id}),
                ]
            )

        view, report = prepare_context_messages(
            source,
            max_chars=20_000,
            compact_trigger_ratio=0.55,
            keep_recent_groups=2,
            max_tool_result_chars=220,
        )

        self.assertTrue(report.compacted)
        self.assertTrue(report.soft_compaction)
        self.assertEqual(report.dropped_groups, 0)
        self.assertIn("soft-tool-digest", report.strategy)
        self.assertIn("result_shape", "\n".join(message.content for message in view))
        self.assertEqual(sum(1 for message in view if message.role == "tool"), 6)

    def test_context_window_preserves_change_anchor_and_reports_token_estimate(self) -> None:
        source = [
            Message(role="system", content="system"),
            Message(role="user", content="fix the endpoint"),
        ]
        for index in range(4):
            call_id = f"read_{index}"
            source.extend(
                [
                    Message(
                        role="assistant",
                        content="",
                        metadata={
                            "tool_calls": [
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": json.dumps({"path": f"src/{index}.py"}),
                                    },
                                }
                            ]
                        },
                    ),
                    Message(role="tool", content=("x" * 1400), metadata={"tool_call_id": call_id}),
                ]
            )
        source.extend(
            [
                Message(
                    role="assistant",
                    content="",
                    metadata={
                        "tool_calls": [
                            {
                                "id": "patch_1",
                                "type": "function",
                                "function": {
                                    "name": "apply_patch",
                                    "arguments": json.dumps({"path": "src/api.py"}),
                                },
                            }
                        ]
                    },
                ),
                Message(role="tool", content="patch applied", metadata={"tool_call_id": "patch_1"}),
            ]
        )
        for index in range(4, 8):
            call_id = f"read_{index}"
            source.extend(
                [
                    Message(
                        role="assistant",
                        content="",
                        metadata={
                            "tool_calls": [
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": json.dumps({"path": f"src/{index}.py"}),
                                    },
                                }
                            ]
                        },
                    ),
                    Message(role="tool", content=("x" * 1400), metadata={"tool_call_id": call_id}),
                ]
            )

        view, report = prepare_context_messages(
            source,
            max_chars=2_400,
            keep_recent_groups=2,
            max_tool_result_chars=220,
        )

        wire = [json.dumps(message.to_wire(), ensure_ascii=False) for message in view]
        self.assertTrue(any("apply_patch" in item and "src/api.py" in item for item in wire))
        self.assertGreaterEqual(report.preserved_anchor_groups, 1)
        self.assertEqual(report.original_tokens, estimate_context_tokens(source))
        self.assertEqual(report.final_tokens, estimate_context_tokens(view))
        self.assertLessEqual(report.final_chars, 2_400)

    def test_context_window_promotes_focus_lines_from_old_observations(self) -> None:
        source = [
            Message(role="system", content="system"),
            Message(role="user", content="fix target.py"),
        ]
        for index in range(5):
            call_id = f"read_{index}"
            observation = "\n".join(
                [
                    f"line {line}: unrelated implementation detail"
                    for line in range(1, 35)
                ]
            )
            if index == 1:
                observation = observation.replace(
                    "line 18: unrelated implementation detail",
                    "line 18: target.py contains needle() and must remain compatible",
                )
            source.extend(
                [
                    Message(
                        role="assistant",
                        content="",
                        metadata={
                            "tool_calls": [
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": json.dumps({"path": f"src/{index}.py"}),
                                    },
                                }
                            ]
                        },
                    ),
                    Message(role="tool", content=observation, metadata={"tool_call_id": call_id}),
                ]
            )

        view, report = prepare_context_messages(
            source,
            max_chars=2_000,
            keep_recent_groups=1,
            max_tool_result_chars=180,
            focus_terms=["target.py", "needle"],
        )

        rendered = "\n".join(message.content for message in view)
        self.assertIn("target.py", rendered)
        self.assertIn("needle", rendered)
        self.assertGreaterEqual(report.preserved_anchor_groups, 1)
        self.assertLessEqual(report.final_chars, 2_000)

    def test_context_window_summarizes_unfocused_tool_noise_as_digest(self) -> None:
        repeated_result = json.dumps(
            {
                "ok": True,
                "data": "\n".join(f"unfocused-noise-{index}" for index in range(400)),
            }
        )
        source = [
            Message(role="system", content="system"),
            Message(role="user", content="fix target.py"),
        ]
        for index in range(8):
            call_id = f"read_{index}"
            source.extend(
                [
                    Message(
                        role="assistant",
                        content="",
                        metadata={
                            "tool_calls": [
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": json.dumps({"path": f"src/{index}.py"}),
                                    },
                                }
                            ]
                        },
                    ),
                    Message(role="tool", content=repeated_result, metadata={"tool_call_id": call_id}),
                ]
            )

        view, report = prepare_context_messages(
            source,
            max_chars=4_000,
            keep_recent_groups=1,
            max_tool_result_chars=240,
            focus_terms=["target.py"],
        )

        rendered = "\n".join(message.content for message in view)
        self.assertIn("Context compaction packet", rendered)
        self.assertIn("Layer: compressed conversation history", rendered)
        self.assertIn("result_shape", rendered)
        self.assertNotIn("unfocused-noise-399", rendered)
        self.assertLessEqual(report.final_chars, 4_000)

    def test_context_budget_tuner_calibrates_from_real_prompt_usage(self) -> None:
        source = [
            Message(role="system", content="system"),
            Message(role="user", content="fix app.py"),
            Message(role="assistant", content="read result " + ("x" * 12_000)),
        ]
        view, report = prepare_context_messages(source, max_chars=8_000)
        tool_tokens = estimate_tool_schema_tokens(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a file.",
                        "parameters": {"type": "object"},
                    },
                }
            ]
        )
        estimated = report.final_tokens + tool_tokens
        tuner = ContextBudgetTuner(base_max_chars=80_000, min_chars=10_000)

        note = tuner.observe(
            report,
            {"prompt_tokens": estimated * 2},
            tool_schema_tokens=tool_tokens,
        )

        self.assertIsNotNone(note)
        self.assertIn("context usage", note or "")
        self.assertLess(tuner.current_max_chars(), 80_000)
        self.assertEqual(prompt_tokens_from_usage({"input_tokens": 17}), 17)

    def test_controller_updates_context_budget_from_response_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("def answer():\n    return 1\n", encoding="utf-8")
            graph = build_repo_graph(root)
            verifier = Verifier(root)
            env = ToolEnvironment(root, graph, verifier)
            registry = build_default_registry(env)
            evidence = EvidenceLog(root)
            events: list[str] = []
            llm = FakeLLM([LLMResponse(content="done", usage={"prompt_tokens": 40_000})])
            controller = AgentController(
                llm=llm,
                registry=registry,
                repo_graph=graph,
                evidence=evidence,
                verifier=verifier,
                auto_verify=False,
                context_budget_chars=80_000,
                on_event=events.append,
            )

            controller.run(
                "inspect answer",
                initial_messages=[
                    Message(role="system", content="system"),
                    Message(role="user", content="Task:\ninspect answer"),
                    Message(role="assistant", content="older reasoning " + ("x" * 20_000)),
                ],
            )

            self.assertLess(controller.context_budget_tuner.current_max_chars(), 80_000)
            self.assertTrue(any("context usage:" in event for event in events))

    def test_working_memory_requires_verification_after_latest_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "app.py"
            target.write_text("def answer():\n    return 1\n", encoding="utf-8")
            memory = WorkingMemory("make answer return 42")

            memory.observe_workspace_changes(["app.py"], 1, root)
            self.assertIn("verification", " ".join(memory.completion_blockers(root)).lower())

            memory.observe_verification(VerificationResult(True, None, "ok"), 2, root)
            self.assertEqual(memory.completion_blockers(root), [])

            target.write_text("def answer():\n    return 99\n", encoding="utf-8")
            self.assertIn("fingerprints", " ".join(memory.completion_blockers(root)))

    def test_finish_task_is_rejected_until_verified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("def answer():\n    return 1\n", encoding="utf-8")
            graph = build_repo_graph(root)
            verifier = Verifier(root)
            env = ToolEnvironment(root, graph, verifier)
            registry = build_default_registry(env)
            evidence = EvidenceLog(root)
            llm = FakeLLM(
                [
                    tool_response("read_file", {"path": "app.py", "start": 1, "end": 3}, "call_1"),
                    tool_response(
                        "apply_patch",
                        {
                            "path": "app.py",
                            "old_text": "def answer():\n    return 1\n",
                            "new_text": "def answer():\n    return 42\n",
                        },
                        "call_2",
                    ),
                    tool_response(
                        "finish_task",
                        {
                            "summary": "Changed answer() to return 42.",
                            "strategy": "Read app.py, patch answer(), then finish.",
                        },
                        "call_3",
                    ),
                    tool_response("verify", {}, "call_4"),
                    tool_response(
                        "finish_task",
                        {
                            "summary": "Changed answer() to return 42 after verification.",
                            "strategy": "Read app.py, patch answer(), verify, then finish.",
                        },
                        "call_5",
                    ),
                ]
            )
            controller = AgentController(
                llm=llm,
                registry=registry,
                repo_graph=graph,
                evidence=evidence,
                verifier=verifier,
                auto_verify=False,
            )

            result = controller.run("make answer return 42")

            self.assertIn("after verification", result.final_text)
            self.assertEqual(result.working_memory.completion_blockers(root), [])
            chain = evidence.format_chain(20)
            self.assertIn("Completion rejected", chain)
            self.assertIn("Completion accepted", chain)
            self.assertIn("Working Memory", result.report_path.read_text(encoding="utf-8"))

    def test_session_store_redacts_and_loads_working_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            state_root = Path(tmp) / "state"
            root.mkdir()
            memory = WorkingMemory("record summary")
            memory.observe_progress(
                {
                    "phase": "exploring",
                    "hypothesis": "token=unit-secret-token-123456789",
                    "next_step": "read files",
                    "target_paths": ["app.py"],
                },
                1,
            )

            store = SessionStore(root, root=state_root)
            saved = store.save_run(
                task="record summary",
                model="unit-model",
                final_text="secret token=unit-secret-token-123456789",
                report_path=root / ".tracegraph" / "report.md",
                verification="ok",
                working_memory=memory,
            )
            loaded = store.load(saved.session_id)

            self.assertEqual(loaded.session_id, saved.session_id)
            self.assertIn("[REDACTED]", loaded.final_text)
            self.assertNotIn("unit-secret-token", json.dumps(loaded.to_dict()))
            restored = WorkingMemory.from_dict(loaded.working_memory)
            self.assertEqual(restored.target_paths, ["app.py"])

    def test_session_store_persists_resumable_message_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            state_root = Path(tmp) / "state"
            root.mkdir()
            memory = WorkingMemory("continue previous task")
            messages = [
                Message(role="system", content="system"),
                Message(role="user", content="continue previous task"),
                Message(
                    role="assistant",
                    content="",
                    metadata={
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": "{}"},
                            }
                        ]
                    },
                ),
                Message(role="tool", content='{"ok": true}', metadata={"tool_call_id": "call_1"}),
            ]

            store = SessionStore(root, root=state_root)
            saved = store.create_checkpoint(
                task="continue previous task",
                model="unit-model",
                working_memory=memory,
                messages=messages,
                iterations=1,
            )
            loaded = store.load(saved.session_id)
            latest = store.latest_resumable()

            self.assertEqual(loaded.status, "running")
            self.assertEqual(loaded.iterations, 1)
            self.assertEqual(Message.from_dict(loaded.messages[3]).metadata["tool_call_id"], "call_1")
            self.assertIsNotNone(latest)
            self.assertEqual(latest.session_id, saved.session_id)
            self.assertEqual(saved.tree_id, saved.session_id)
            self.assertIsNone(saved.parent_id)

    def test_session_store_updates_conversation_node_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            state_root = Path(tmp) / "state"
            root.mkdir()
            memory = WorkingMemory("branchable task")
            store = SessionStore(root, root=state_root)
            root_node = store.create_checkpoint(
                task="branchable task",
                model="unit-model",
                working_memory=memory,
                messages=[Message(role="user", content="branchable task")],
                iterations=0,
            )

            child = store.update_checkpoint(
                root_node.session_id,
                task="branchable task",
                model="unit-model",
                working_memory=memory,
                messages=[
                    Message(role="user", content="branchable task"),
                    Message(role="assistant", content="inspected repository"),
                ],
                iterations=1,
            )

            self.assertEqual(child.session_id, root_node.session_id)
            self.assertIsNone(child.parent_id)
            self.assertEqual(child.tree_id, root_node.session_id)
            self.assertEqual(child.iterations, 1)
            self.assertEqual(child.messages[-1]["content"], "inspected repository")
            self.assertEqual(store.load(root_node.session_id).messages[0]["content"], "branchable task")
            self.assertEqual([item.session_id for item in store.lineage(child.session_id)], [root_node.session_id])
            self.assertEqual(store.list_children(root_node.session_id), [])
            self.assertEqual([item.session_id for item in store.list_heads(limit=10)], [root_node.session_id])

    def test_session_store_forks_from_existing_node_with_follow_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            state_root = Path(tmp) / "state"
            root.mkdir()
            memory = WorkingMemory("try another approach")
            store = SessionStore(root, root=state_root)
            root_node = store.create_checkpoint(
                task="try another approach",
                model="unit-model",
                working_memory=memory,
                messages=[Message(role="user", content="try another approach")],
                iterations=0,
            )

            fork = store.fork_checkpoint(root_node.session_id, follow_up="use the safer patch")

            self.assertEqual(fork.parent_id, root_node.session_id)
            self.assertEqual(fork.tree_id, root_node.session_id)
            self.assertEqual(fork.event_type, "fork")
            self.assertIn("safer patch", fork.messages[-1]["content"])

            follow_up = store.fork_checkpoint(root_node.session_id, follow_up="continue this dialogue", event_type="follow_up")
            self.assertEqual(follow_up.parent_id, root_node.session_id)
            self.assertEqual(follow_up.tree_id, root_node.session_id)
            self.assertEqual(follow_up.event_type, "follow_up")
            self.assertIn("continue this dialogue", follow_up.messages[-1]["content"])

    def test_completed_session_remains_continuable_when_it_has_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            state_root = Path(tmp) / "state"
            root.mkdir()
            store = SessionStore(root, root=state_root)
            saved = store.save_run(
                task="completed conversation",
                model="unit-model",
                final_text="done",
                report_path="report.md",
                verification="ok",
                working_memory=WorkingMemory("completed conversation"),
                messages=[
                    Message(role="user", content="completed conversation"),
                    Message(role="assistant", content="done"),
                ],
                iterations=1,
            )

            latest = store.latest_resumable()

            self.assertIsNotNone(latest)
            self.assertEqual(latest.session_id, saved.session_id)
            self.assertEqual(latest.status, "completed")

    def test_session_store_externalizes_large_message_contents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            state_root = Path(tmp) / "state"
            root.mkdir()
            memory = WorkingMemory("large message session")
            large_content = "log line\n" * 2000
            messages = [
                Message(role="system", content="system"),
                Message(role="user", content="large message session"),
                Message(role="assistant", content=large_content),
            ]

            store = SessionStore(root, root=state_root)
            saved = store.create_checkpoint(
                task="large message session",
                model="unit-model",
                working_memory=memory,
                messages=messages,
                iterations=1,
            )

            session_dir = store.directory / saved.session_id
            session_file = session_dir.with_suffix(".json")
            self.assertTrue(session_file.exists())
            payload = json.loads(session_file.read_text(encoding="utf-8"))
            blob_info = payload["messages"][2]["metadata"]["content_blob"]
            blob_path = session_dir / blob_info["path"]
            self.assertTrue(blob_path.exists())
            self.assertIn("externalized message content", payload["messages"][2]["content"])
            self.assertLess(session_file.stat().st_size, len(large_content))

            loaded = store.load(saved.session_id)
            self.assertEqual(loaded.messages[2]["content"], large_content)
            self.assertNotIn("content_blob", loaded.messages[2]["metadata"])

    def test_web_defaults_and_recent_sessions_use_lightweight_session_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            state_root = Path(tmp) / "state"
            root.mkdir()
            memory = WorkingMemory("index only")
            store = SessionStore(root, root=state_root)
            saved = store.create_checkpoint(
                task="index only",
                model="unit-model",
                working_memory=memory,
                messages=[
                    Message(role="system", content="system"),
                    Message(role="user", content="index only"),
                    Message(role="assistant", content="x" * 10_000),
                ],
                iterations=1,
            )

            latest = store.latest_resumable(include_blobs=False)
            recent = store.list_recent(limit=1, include_blobs=False)

            self.assertIsNotNone(latest)
            self.assertEqual(latest.session_id, saved.session_id)
            self.assertEqual(len(recent), 1)
            self.assertIn("content_blob", json.dumps(recent[0].to_dict()))

    def test_workspace_snapshot_reuses_unchanged_file_metadata_between_captures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.py").write_text("print('a')\n", encoding="utf-8")
            (root / "b.py").write_text("print('b')\n", encoding="utf-8")

            workspace_state_module._SNAPSHOT_CACHE.clear()
            original_read_bytes = Path.read_bytes
            call_count = 0

            def counting_read_bytes(self: Path):
                nonlocal call_count
                call_count += 1
                return original_read_bytes(self)

            with patch.object(Path, "read_bytes", counting_read_bytes):
                first = workspace_state_module.WorkspaceSnapshot.capture(root)
                second = workspace_state_module.WorkspaceSnapshot.capture(root)

            self.assertEqual(first.files["a.py"].sha256, second.files["a.py"].sha256)
            self.assertEqual(first.files["b.py"].sha256, second.files["b.py"].sha256)
            self.assertEqual(call_count, 2)

    def test_repo_graph_reuses_unchanged_file_summaries_between_builds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("def answer():\n    return 1\n", encoding="utf-8")
            (root / "util.py").write_text("def helper():\n    return 2\n", encoding="utf-8")

            repo_graph_module._GRAPH_CACHE.clear()
            original_read_text = Path.read_text
            call_count = 0

            def counting_read_text(self: Path, *args, **kwargs):
                nonlocal call_count
                call_count += 1
                return original_read_text(self, *args, **kwargs)

            with patch.object(Path, "read_text", counting_read_text):
                first = repo_graph_module.build_repo_graph(root)
                second = repo_graph_module.build_repo_graph(root)

            self.assertEqual({node.path for node in first.files}, {node.path for node in second.files})
            self.assertEqual(call_count, 2)

    def test_session_scoped_evidence_logs_do_not_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            one = EvidenceLog(root, "session-one")
            two = EvidenceLog(root, "session-two")

            one.reset()
            two.reset()
            one.append(
                EvidenceStep(
                    iteration=1,
                    stage="locate",
                    action="repo_graph_query",
                    args={"query": "alpha"},
                    ok=True,
                    observation="alpha hit",
                )
            )
            two.append(
                EvidenceStep(
                    iteration=1,
                    stage="locate",
                    action="repo_graph_query",
                    args={"query": "beta"},
                    ok=True,
                    observation="beta hit",
                )
            )

            self.assertIn(".tracegraph", one.path.as_posix())
            self.assertIn("session-one", one.path.as_posix())
            self.assertIn("alpha hit", one.format_chain())
            self.assertNotIn("beta hit", one.format_chain())
            self.assertIn("beta hit", two.format_chain())
            self.assertNotEqual(one.write_report("done", "ok", "", "").parent, two.write_report("done", "ok", "", "").parent)

    def test_controller_resumes_from_checkpoint_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("def answer():\n    return 1\n", encoding="utf-8")
            graph = build_repo_graph(root)
            verifier = Verifier(root)
            env = ToolEnvironment(root, graph, verifier)
            registry = build_default_registry(env)
            evidence = EvidenceLog(root)
            memory = WorkingMemory("inspect answer")
            memory.observe_tool(
                name="read_file",
                arguments={"path": "app.py", "start": 1, "end": 2},
                ok=True,
                observation="1 | def answer():\n2 |     return 1",
                step=1,
                workspace=root,
            )
            messages = [
                Message(role="system", content="system"),
                Message(role="user", content="Task:\ninspect answer"),
                Message(
                    role="assistant",
                    content="",
                    metadata={
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": json.dumps({"path": "app.py", "start": 1, "end": 2}),
                                },
                            }
                        ]
                    },
                ),
                Message(
                    role="tool",
                    content='{"ok": true, "data": "1 | def answer():\\n2 |     return 1"}',
                    metadata={"tool_call_id": "call_1"},
                ),
            ]
            llm = FakeLLM([LLMResponse(content="resumed final")])
            controller = AgentController(
                llm=llm,
                registry=registry,
                repo_graph=graph,
                evidence=evidence,
                verifier=verifier,
                auto_verify=False,
            )

            result = controller.run(
                "inspect answer",
                initial_messages=messages,
                initial_memory=memory,
                initial_iteration=1,
                reset_evidence=False,
            )

            self.assertEqual(result.final_text, "resumed final")
            self.assertTrue(any("return 1" in message.content for message in llm.calls[0][0]))
            self.assertGreaterEqual(result.iterations, 2)

    def test_experience_store_retrieves_only_current_verified_cards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            state_root = Path(tmp) / "state"
            root.mkdir()
            target = root / "app.py"
            target.write_text("def answer():\n    return 42\n", encoding="utf-8")
            memory = WorkingMemory("修复 answer 函数返回值")
            memory.observe_workspace_changes(["app.py"], 1, root)
            memory.observe_verification(VerificationResult(True, "python -m compileall .", "ok"), 2, root)

            store = ExperienceStore(root, root=state_root)
            card = store.add_verified(memory, "定位 answer 函数并用 apply_patch 改返回值")
            match = store.retrieve("继续修复 answer 函数返回值")

            self.assertIsNotNone(match)
            self.assertEqual(match.experience_id, card.experience_id)

            target.write_text("def answer():\n    return 99\n", encoding="utf-8")
            self.assertIsNone(store.retrieve("继续修复 answer 函数返回值"))


if __name__ == "__main__":
    unittest.main()
