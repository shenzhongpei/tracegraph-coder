from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tracegraph_coder.web_app import (
    _choose_workspace_folder,
    RunState,
    _append_saved_final_answer_if_missing,
    _default_payload,
    _llm_config_from_payload,
    _prepare_continuation_memory,
    _session_detail_payload,
    _session_payload,
    _session_tree_payload,
    _static_path,
)
from tracegraph_coder.models import Message
from tracegraph_coder.session import RunSession
from tracegraph_coder.working_memory import WorkingMemory


class WebAppTests(unittest.TestCase):
    def test_defaults_do_not_expose_api_key(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TRACEGRAPH_API_KEY": "unit-secret-token-123456789",
                "TRACEGRAPH_MODEL": "unit-test-model",
            },
            clear=True,
        ):
            payload = _default_payload(Path.cwd())

        self.assertTrue(payload["hasEnvKey"])
        self.assertEqual(payload["envKeyName"], "TRACEGRAPH_API_KEY")
        self.assertEqual(payload["model"], "unit-test-model")
        self.assertNotIn("apiKey", payload)
        self.assertNotIn("unit-secret-token", json.dumps(payload))

    def test_static_path_blocks_traversal(self) -> None:
        self.assertIsNone(_static_path("/../pyproject.toml"))
        self.assertIsNotNone(_static_path("/assets/app.js"))

    def test_choose_workspace_folder_parses_selected_path(self) -> None:
        completed = SimpleNamespace(returncode=0, stdout='{"selected": "E:/Desktop/南软项目"}\n', stderr="")
        with patch("tracegraph_coder.web_app.subprocess.run", return_value=completed):
            selected = _choose_workspace_folder()

        self.assertEqual(selected, "E:/Desktop/南软项目")

    def test_choose_workspace_folder_returns_none_on_cancel(self) -> None:
        completed = SimpleNamespace(returncode=0, stdout='{"selected": ""}\n', stderr="")
        with patch("tracegraph_coder.web_app.subprocess.run", return_value=completed):
            selected = _choose_workspace_folder()

        self.assertIsNone(selected)

    def test_run_state_redacts_log_entries(self) -> None:
        state = RunState()
        state.start("run-1", Path.cwd())
        state.append_log("token=unit-secret-token-123456789")
        snapshot = state.snapshot()
        self.assertIn("[REDACTED]", "\n".join(snapshot["logs"]))
        self.assertNotIn("unit-secret-token", "\n".join(snapshot["logs"]))

    def test_explicit_config_uses_provided_key(self) -> None:
        config = _llm_config_from_payload(
            {
                "apiKey": "unit-key",
                "model": "unit-model",
                "baseUrl": "http://localhost:9999/v1",
                "useEnvKey": False,
            }
        )
        self.assertEqual(config.api_key, "unit-key")
        self.assertEqual(config.model, "unit-model")
        self.assertEqual(config.base_url, "http://localhost:9999/v1")

    def test_session_payload_marks_resumable_and_redacts(self) -> None:
        session = RunSession(
            session_id="session-1",
            workspace_root=str(Path.cwd()),
            created_at="2026-08-31T00:00:00+00:00",
            updated_at="2026-08-31T00:00:01+00:00",
            status="running",
            task="fix bug with token=unit-secret-token-123456789",
            model="unit-model",
            final_text="",
            report_path="",
            verification="",
            working_memory={},
            messages=[{"role": "user", "content": "hello", "metadata": None}],
            iterations=3,
        )

        payload = _session_payload(session)

        self.assertTrue(payload["resumable"])
        self.assertTrue(payload["continuable"])
        self.assertEqual(payload["iterations"], 3)
        self.assertEqual(payload["workspaceRoot"], str(Path.cwd()))
        self.assertIn("[REDACTED]", payload["task"])
        self.assertNotIn("unit-secret-token", json.dumps(payload))

    def test_session_detail_payload_includes_redacted_transcript(self) -> None:
        session = RunSession(
            session_id="session-1",
            workspace_root=str(Path.cwd()),
            created_at="2026-08-31T00:00:00+00:00",
            updated_at="2026-08-31T00:00:01+00:00",
            status="completed",
            task="fix bug",
            model="unit-model",
            final_text="done",
            report_path="",
            verification="",
            working_memory={},
            messages=[
                {
                    "role": "user",
                    "content": "please use token=unit-secret-token-123456789",
                    "metadata": None,
                },
                {
                    "role": "assistant",
                    "content": "I will inspect the file.",
                    "metadata": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path":"app.py","apiKey":"token=unit-secret-token-123456789"}',
                                },
                            }
                        ]
                    },
                },
                {
                    "role": "tool",
                    "content": "result contains token=unit-secret-token-123456789",
                    "metadata": {"tool_call_id": "call_1"},
                },
            ],
            iterations=2,
        )

        payload = _session_detail_payload(session)

        self.assertEqual(payload["messageCount"], 3)
        self.assertEqual(payload["messages"][1]["metadata"]["tool_calls"][0]["function"]["name"], "read_file")
        self.assertIn("[REDACTED]", payload["messages"][0]["content"])
        self.assertIn("[REDACTED]", payload["messages"][2]["content"])
        self.assertNotIn("unit-secret-token", json.dumps(payload))

    def test_prepare_continuation_memory_preserves_context_but_focuses_follow_up(self) -> None:
        memory = WorkingMemory("original task")
        memory.phase = "ready"
        memory.hypothesis = "old hypothesis"
        memory.next_step = "old next step"
        memory.observations.append("older useful context")

        _prepare_continuation_memory(memory, base_task="original task", follow_up="now answer this")

        self.assertIn("original task", memory.task)
        self.assertIn("Latest user request:\nnow answer this", memory.task)
        self.assertEqual(memory.phase, "exploring")
        self.assertEqual(memory.hypothesis, "")
        self.assertEqual(memory.next_step, "")
        self.assertEqual(memory.observations, ["older useful context"])

    def test_saved_final_answer_is_inserted_before_follow_up(self) -> None:
        messages = [{"role": "user", "content": "first question", "metadata": None}]
        typed = [Message.from_dict(item) for item in messages]

        _append_saved_final_answer_if_missing(typed, "first answer")
        typed.append(Message(role="user", content="second question"))

        self.assertEqual([message.role for message in typed], ["user", "assistant", "user"])
        self.assertEqual(typed[1].content, "first answer")
        self.assertEqual(typed[1].metadata, {"final_answer": True})

    def test_saved_final_answer_helper_marks_existing_answer_without_duplication(self) -> None:
        messages = [
            Message(role="user", content="first question"),
            Message(role="assistant", content="first answer"),
        ]

        _append_saved_final_answer_if_missing(messages, "first answer")

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[1].metadata, {"final_answer": True})

    def test_session_tree_payload_builds_roots_and_heads(self) -> None:
        root = RunSession(
            session_id="root",
            workspace_root=str(Path.cwd()),
            created_at="2026-08-31T00:00:00+00:00",
            updated_at="2026-08-31T00:00:01+00:00",
            status="running",
            task="root task",
            model="unit-model",
            final_text="",
            report_path="",
            verification="",
            working_memory={},
            messages=[{"role": "user", "content": "hello", "metadata": None}],
            iterations=1,
            tree_id="root",
            parent_id=None,
            event_type="root",
        )
        completed = RunSession(
            session_id="completed",
            workspace_root=str(Path.cwd()),
            created_at="2026-08-31T00:01:00+00:00",
            updated_at="2026-08-31T00:01:01+00:00",
            status="completed",
            task="root task",
            model="unit-model",
            final_text="done",
            report_path="report.md",
            verification="ok",
            working_memory={},
            messages=[{"role": "user", "content": "hello", "metadata": None}],
            iterations=2,
            tree_id="root",
            parent_id="root",
            event_type="completed",
        )
        branch = RunSession(
            session_id="branch",
            workspace_root=str(Path.cwd()),
            created_at="2026-08-31T00:02:00+00:00",
            updated_at="2026-08-31T00:02:01+00:00",
            status="running",
            task="root task",
            model="unit-model",
            final_text="",
            report_path="",
            verification="",
            working_memory={},
            messages=[{"role": "user", "content": "try another path", "metadata": None}],
            iterations=2,
            tree_id="root",
            parent_id="root",
            event_type="fork",
        )

        class FakeStore:
            def __init__(self, workspace: Path):
                self.workspace = workspace

            def list_recent(self, limit: int = 500, *, include_blobs: bool = False) -> list[RunSession]:
                return [branch, completed, root]

        with patch("tracegraph_coder.web_app.SessionStore", FakeStore):
            payload = _session_tree_payload(Path.cwd())

        self.assertEqual([node["sessionId"] for node in payload["nodes"]], ["branch", "completed", "root"])
        self.assertEqual(set(payload["roots"]), {"root"})
        self.assertEqual(set(payload["heads"]), {"branch", "completed"})
        self.assertEqual(payload["nodes"][0]["treeId"], "root")
        self.assertEqual(payload["nodes"][0]["parentId"], "root")
        self.assertEqual(payload["nodes"][0]["eventType"], "fork")
        self.assertTrue(payload["nodes"][1]["resumable"])

    def test_session_tree_payload_collapses_legacy_tool_checkpoints(self) -> None:
        root = RunSession(
            session_id="legacy-root",
            workspace_root=str(Path.cwd()),
            created_at="2026-08-31T00:00:00+00:00",
            updated_at="2026-08-31T00:00:01+00:00",
            status="running",
            task="legacy task",
            model="unit-model",
            final_text="",
            report_path="",
            verification="",
            working_memory={},
            messages=[],
            iterations=0,
            tree_id="legacy-root",
            parent_id=None,
            event_type="root",
        )
        middle = RunSession(
            session_id="legacy-middle",
            workspace_root=str(Path.cwd()),
            created_at="2026-08-31T00:00:02+00:00",
            updated_at="2026-08-31T00:00:03+00:00",
            status="running",
            task="legacy task",
            model="unit-model",
            final_text="",
            report_path="",
            verification="",
            working_memory={},
            messages=[{"role": "user", "content": "legacy task", "metadata": None}],
            iterations=1,
            tree_id="legacy-root",
            parent_id="legacy-root",
            event_type="checkpoint",
            summary="running checkpoint at iteration 1 after tool message",
        )
        head = RunSession(
            session_id="legacy-head",
            workspace_root=str(Path.cwd()),
            created_at="2026-08-31T00:00:04+00:00",
            updated_at="2026-08-31T00:00:05+00:00",
            status="completed",
            task="legacy task",
            model="unit-model",
            final_text="done",
            report_path="report.md",
            verification="ok",
            working_memory={},
            messages=[
                {"role": "user", "content": "legacy task", "metadata": None},
                {"role": "assistant", "content": "done", "metadata": None},
            ],
            iterations=2,
            tree_id="legacy-root",
            parent_id="legacy-middle",
            event_type="completed",
            summary="completed checkpoint at iteration 2 after assistant message",
        )

        class FakeStore:
            def __init__(self, workspace: Path):
                self.workspace = workspace

            def list_recent(self, limit: int = 500, *, include_blobs: bool = False) -> list[RunSession]:
                return [head, middle, root]

        with patch("tracegraph_coder.web_app.SessionStore", FakeStore):
            payload = _session_tree_payload(Path.cwd())

        self.assertEqual([node["sessionId"] for node in payload["nodes"]], ["legacy-head"])
        self.assertEqual(payload["roots"], ["legacy-head"])
        self.assertEqual(payload["heads"], ["legacy-head"])
        self.assertEqual(payload["nodes"][0]["parentId"], "")
        self.assertEqual(payload["nodes"][0]["eventType"], "conversation")
        self.assertIn("conversation", payload["nodes"][0]["summary"])


if __name__ == "__main__":
    unittest.main()
