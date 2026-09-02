from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path

from tracegraph_coder.models import ToolCall
from tracegraph_coder.repo_graph import build_repo_graph
from tracegraph_coder.safety import SafetyError, safe_path
from tracegraph_coder.tools import ToolEnvironment, build_default_registry
from tracegraph_coder.verifier import Verifier


class ToolTests(unittest.TestCase):
    def test_safe_path_blocks_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SafetyError):
                safe_path(tmp, "../outside.txt")

    def test_apply_patch_requires_unique_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
            env = ToolEnvironment(root, build_repo_graph(root), Verifier(root))
            registry = build_default_registry(env)
            result = registry.execute(
                ToolCall("1", "apply_patch", {"path": "a.py", "old_text": "x = 1", "new_text": "x = 2"})
            )
            self.assertFalse(result.ok)
            self.assertIn("matched 2", result.error or result.data)

    def test_search_text_finds_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.py").write_text("def hello():\n    return 'world'\n", encoding="utf-8")
            env = ToolEnvironment(root, build_repo_graph(root), Verifier(root))
            registry = build_default_registry(env)
            result = registry.execute(ToolCall("1", "search_text", {"pattern": "hello"}))
            self.assertTrue(result.ok)
            self.assertIn("a.py:1", result.data)

    def test_search_text_auto_detects_regex_alternation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.js").write_text(
                "function apiGet() {}\nfunction persistSettings() {}\n",
                encoding="utf-8",
            )
            env = ToolEnvironment(root, build_repo_graph(root), Verifier(root))
            registry = build_default_registry(env)

            result = registry.execute(ToolCall("1", "search_text", {"pattern": "apiGet|persistSettings"}))

            self.assertTrue(result.ok)
            self.assertIn("app.js:1", result.data)
            self.assertIn("app.js:2", result.data)

    def test_read_many_reads_multiple_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.py").write_text("a1\na2\na3\n", encoding="utf-8")
            (root / "b.py").write_text("b1\nb2\nb3\n", encoding="utf-8")
            env = ToolEnvironment(root, build_repo_graph(root), Verifier(root))
            registry = build_default_registry(env)

            result = registry.execute(
                ToolCall(
                    "1",
                    "read_many",
                    {
                        "requests": [
                            {"path": "a.py", "start": 1, "end": 2},
                            {"path": "b.py", "start": 2, "end": 3},
                        ]
                    },
                )
            )

            self.assertTrue(result.ok)
            self.assertIn("--- a.py:1-2 ---", result.data)
            self.assertIn("1 | a1", result.data)
            self.assertIn("--- b.py:2-3 ---", result.data)
            self.assertIn("2 | b2", result.data)

    def test_repo_graph_neighborhood_tool_returns_impact_edges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("from helper import value\n\nanswer = value\n", encoding="utf-8")
            (root / "helper.py").write_text("value = 42\n", encoding="utf-8")
            env = ToolEnvironment(root, build_repo_graph(root), Verifier(root))
            registry = build_default_registry(env)

            result = registry.execute(ToolCall("1", "repo_graph_neighborhood", {"path": "app.py"}))

            self.assertTrue(result.ok)
            self.assertIn("helper.py", result.data)
            self.assertIn("imports_local", result.data)

    def test_repo_graph_query_tool_explains_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "user_service.py").write_text(
                "class UserService:\n"
                "    def create_user(self):\n"
                "        return True\n",
                encoding="utf-8",
            )
            env = ToolEnvironment(root, build_repo_graph(root), Verifier(root))
            registry = build_default_registry(env)

            result = registry.execute(ToolCall("1", "repo_graph_query", {"query": "create user"}))

            self.assertTrue(result.ok)
            self.assertIn('"score"', result.data)
            self.assertIn('"matched_tokens"', result.data)
            self.assertIn('"match_reasons"', result.data)

    def test_conversation_memory_tool_is_available_when_memory_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory = "Previous final answer:\n我们讨论了仓库图、证据链和会话树。"
            env = ToolEnvironment(root, build_repo_graph(root), Verifier(root), conversation_memory=memory)
            registry = build_default_registry(env)

            self.assertIn("read_conversation_memory", registry.names())
            result = registry.execute(
                ToolCall(
                    "1",
                    "read_conversation_memory",
                    {"question": "之前聊了什么", "focus": "full_summary", "limit": 4},
                )
            )

            self.assertTrue(result.ok)
            self.assertIn("仓库图", result.data)
            self.assertIn("之前聊了什么", result.data)

    def test_write_file_does_not_overwrite_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.py").write_text("x = 1\n", encoding="utf-8")
            env = ToolEnvironment(root, build_repo_graph(root), Verifier(root))
            registry = build_default_registry(env)
            result = registry.execute(ToolCall("1", "write_file", {"path": "a.py", "content": "x = 2\n"}))
            self.assertFalse(result.ok)
            self.assertIn("only creates new files", result.error or "")
            self.assertEqual((root / "a.py").read_text(encoding="utf-8"), "x = 1\n")

    def test_tool_validation_rejects_extra_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.py").write_text("x = 1\n", encoding="utf-8")
            env = ToolEnvironment(root, build_repo_graph(root), Verifier(root))
            registry = build_default_registry(env)
            result = registry.execute(ToolCall("1", "read_file", {"path": "a.py", "surprise": True}))
            self.assertFalse(result.ok)
            self.assertIn("unexpected argument", result.error or "")

    def test_run_command_nonzero_exit_is_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = ToolEnvironment(root, build_repo_graph(root), Verifier(root))
            registry = build_default_registry(env)
            command = f'"{sys.executable}" -c "import sys; sys.exit(3)"'
            result = registry.execute(ToolCall("1", "run_command", {"command": command}))
            self.assertFalse(result.ok)
            self.assertIn("exit_code=3", result.error or "")

    def test_run_command_blocks_shell_chaining(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = ToolEnvironment(root, build_repo_graph(root), Verifier(root))
            registry = build_default_registry(env)
            command = f'"{sys.executable}" -c "print(1)" && echo bad'
            result = registry.execute(ToolCall("1", "run_command", {"command": command}))
            self.assertFalse(result.ok)
            self.assertIn("Shell operators", result.error or "")


if __name__ == "__main__":
    unittest.main()
