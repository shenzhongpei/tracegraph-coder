from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from tracegraph_coder.controller import CONTROL_TOOL_SCHEMAS, AgentController
from tracegraph_coder.evidence import EvidenceLog
from tracegraph_coder.models import LLMResponse
from tracegraph_coder.repo_graph import RepoGraph, build_repo_graph
from tracegraph_coder.tools import ToolEnvironment, build_default_registry
from tracegraph_coder.verifier import Verifier


class FakeLLM:
    def __init__(self, responses: list[LLMResponse]):
        self.responses = responses
        self.calls = []

    def chat(self, messages, tools):
        self.calls.append((messages, tools))
        if not self.responses:
            return LLMResponse(content="done")
        return self.responses.pop(0)


def tool_response(name: str, args: dict, call_id: str = "call_1") -> LLMResponse:
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


class ControllerTests(unittest.TestCase):
    def test_finish_task_schema_requests_user_facing_summary(self) -> None:
        finish_schema = next(
            schema
            for schema in CONTROL_TOOL_SCHEMAS
            if schema["function"]["name"] == "finish_task"
        )

        description = finish_schema["function"]["parameters"]["properties"]["summary"]["description"]

        self.assertIn("User-facing final answer", description)
        self.assertIn("avoid internal process preambles", description)

    def test_controller_executes_patch_and_writes_report(self) -> None:
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
                    tool_response("read_file", {"path": "app.py", "start": 1, "end": 3}),
                    tool_response(
                        "apply_patch",
                        {
                            "path": "app.py",
                            "old_text": "def answer():\n    return 1\n",
                            "new_text": "def answer():\n    return 42\n",
                        },
                        call_id="call_2",
                    ),
                    LLMResponse(content="Changed answer() to return 42."),
                ]
            )
            controller = AgentController(
                llm=llm,
                registry=registry,
                repo_graph=graph,
                evidence=evidence,
                verifier=verifier,
                auto_verify=True,
            )
            result = controller.run("make answer return 42")
            self.assertIn("42", (root / "app.py").read_text(encoding="utf-8"))
            self.assertTrue(result.report_path.exists())
            self.assertTrue(result.verification and result.verification.ok)
            self.assertTrue(result.messages and result.messages[-1].metadata)
            self.assertTrue(result.messages[-1].metadata.get("final_answer"))

    def test_finish_task_summary_is_persisted_as_final_answer_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = build_repo_graph(root)
            verifier = Verifier(root)
            env = ToolEnvironment(root, graph, verifier)
            registry = build_default_registry(env)
            evidence = EvidenceLog(root)
            llm = FakeLLM(
                [
                    tool_response(
                        "finish_task",
                        {
                            "summary": "这是本轮对话的最终结果。",
                            "strategy": "直接回答用户问题并收尾。",
                            "no_changes_reason": "这是分析问答，不需要修改文件。",
                        },
                        call_id="finish_1",
                    )
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

            result = controller.run("解释一下当前项目")

            self.assertEqual(result.final_text, "这是本轮对话的最终结果。")
            self.assertTrue(result.messages)
            self.assertEqual(result.messages[-1].role, "assistant")
            self.assertEqual(result.messages[-1].content, "这是本轮对话的最终结果。")
            self.assertEqual(result.messages[-1].metadata, {"final_answer": True})

    def test_controller_strips_stiff_final_preamble(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = build_repo_graph(root)
            verifier = Verifier(root)
            env = ToolEnvironment(root, graph, verifier)
            registry = build_default_registry(env)
            evidence = EvidenceLog(root)
            llm = FakeLLM(
                [
                    LLMResponse(
                        content=(
                            "I now have a comprehensive understanding. Let me provide a clear explanation.\n\n"
                            "当前项目的核心是一个本地编程 Agent。"
                        )
                    )
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

            result = controller.run("介绍一下项目")

            self.assertNotIn("I now have", result.final_text)
            self.assertEqual(result.final_text, "当前项目的核心是一个本地编程 Agent。")

    def test_run_command_file_change_triggers_auto_verify(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = build_repo_graph(root)
            verifier = Verifier(root)
            env = ToolEnvironment(root, graph, verifier)
            registry = build_default_registry(env)
            evidence = EvidenceLog(root)
            command = (
                f'"{sys.executable}" -c "'
                "open('made.py','w',encoding='utf-8').write('x = 1\\n')"
                '"'
            )
            llm = FakeLLM(
                [
                    tool_response("list_files", {"pattern": "*"}, call_id="call_0"),
                    tool_response("run_command", {"command": command}, call_id="call_1"),
                    LLMResponse(content="Created made.py."),
                ]
            )
            controller = AgentController(
                llm=llm,
                registry=registry,
                repo_graph=graph,
                evidence=evidence,
                verifier=verifier,
                auto_verify=True,
            )
            result = controller.run("create a small python file")
            self.assertTrue((root / "made.py").exists())
            self.assertTrue(result.verification and result.verification.ok)
            run_command_message = next(
                message
                for message in result.messages or []
                if message.metadata and message.metadata.get("tool_call_id") == "call_1"
            )
            self.assertEqual(["made.py"], run_command_message.metadata.get("changed_paths"))
            report = result.report_path.read_text(encoding="utf-8")
            self.assertIn("created: made.py", report)
            self.assertIn("Working Memory", report)

    def test_mutation_before_evidence_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = build_repo_graph(root)
            verifier = Verifier(root)
            env = ToolEnvironment(root, graph, verifier)
            registry = build_default_registry(env)
            evidence = EvidenceLog(root)
            llm = FakeLLM(
                [
                    tool_response("write_file", {"path": "unsafe.py", "content": "x = 1\n"}),
                    LLMResponse(content="Could not write without evidence."),
                ]
            )
            controller = AgentController(
                llm=llm,
                registry=registry,
                repo_graph=graph,
                evidence=evidence,
                verifier=verifier,
                auto_verify=True,
            )
            result = controller.run("create unsafe.py immediately")
            self.assertFalse((root / "unsafe.py").exists())
            chain = evidence.format_chain()
            self.assertIn("Mutation blocked", chain)
            self.assertTrue(result.report_path.exists())

    def test_model_can_choose_conversation_memory_without_file_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = build_repo_graph(root)
            verifier = Verifier(root)
            env = ToolEnvironment(
                root,
                graph,
                verifier,
                conversation_memory="Previous final answer:\n我们之前讨论了仓库图和会话树。",
            )
            registry = build_default_registry(env)
            evidence = EvidenceLog(root)
            llm = FakeLLM(
                [
                    tool_response(
                        "read_conversation_memory",
                        {"question": "之前聊了什么", "focus": "full_summary", "limit": 4},
                        call_id="memory_1",
                    ),
                    LLMResponse(content="我们之前讨论了仓库图和会话树。"),
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

            result = controller.run("之前聊了什么")

            self.assertIn("会话树", result.final_text)
            self.assertIn("read_conversation_memory", evidence.format_chain())
            self.assertNotIn("read_file", evidence.format_chain())
            tool_message = next(
                message
                for message in result.messages or []
                if message.metadata and message.metadata.get("tool_call_id") == "memory_1"
            )
            self.assertIn("仓库图", tool_message.content)

    def test_controller_collapses_repeating_final_answer_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = build_repo_graph(root)
            verifier = Verifier(root)
            env = ToolEnvironment(root, graph, verifier)
            registry = build_default_registry(env)
            evidence = EvidenceLog(root)
            repeated = (
                "第一段。\n\n"
                "第二段。\n\n"
                "第一段。\n\n"
                "第二段。\n\n"
                "第一段。\n\n"
                "第二段。"
            )
            llm = FakeLLM([LLMResponse(content=repeated)])
            controller = AgentController(
                llm=llm,
                registry=registry,
                repo_graph=graph,
                evidence=evidence,
                verifier=verifier,
                auto_verify=False,
            )

            result = controller.run("summarize the project")

            self.assertIn("重复内容已自动折叠", result.final_text)
            self.assertEqual(result.final_text.count("第一段。"), 1)
            self.assertEqual(result.final_text.count("第二段。"), 1)

    def test_failed_evidence_does_not_unlock_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = build_repo_graph(root)
            verifier = Verifier(root)
            env = ToolEnvironment(root, graph, verifier)
            registry = build_default_registry(env)
            evidence = EvidenceLog(root)
            llm = FakeLLM(
                [
                    tool_response("read_file", {"path": "missing.py"}),
                    tool_response("write_file", {"path": "unsafe.py", "content": "x = 1\n"}, call_id="call_2"),
                    LLMResponse(content="done"),
                ]
            )
            controller = AgentController(
                llm=llm,
                registry=registry,
                repo_graph=graph,
                evidence=evidence,
                verifier=verifier,
                auto_verify=True,
            )
            controller.run("try failed evidence then write")
            self.assertFalse((root / "unsafe.py").exists())
            self.assertIn("Mutation blocked", evidence.format_chain())

    def test_controller_uses_refreshed_repo_graph_after_new_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = build_repo_graph(root)
            verifier = Verifier(root)
            env = ToolEnvironment(root, graph, verifier)
            registry = build_default_registry(env)
            evidence = EvidenceLog(root)
            llm = FakeLLM(
                [
                    tool_response("list_files", {"pattern": "*"}, call_id="call_0"),
                    tool_response(
                        "write_file",
                        {
                            "path": "new_feature.py",
                            "content": "def created_symbol():\n    return 42\n",
                        },
                        call_id="call_1",
                    ),
                    LLMResponse(content="done"),
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

            controller.run("create new_feature.py with created_symbol")

            third_prompt = "\n".join(message.content for message in llm.calls[2][0])
            self.assertIn("new_feature.py", third_prompt)
            self.assertIn("created_symbol", third_prompt)

    def test_controller_marks_read_evidence_stale_after_patch(self) -> None:
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
                    tool_response("read_file", {"path": "app.py", "start": 1, "end": 2}, call_id="call_0"),
                    tool_response(
                        "apply_patch",
                        {
                            "path": "app.py",
                            "old_text": "def answer():\n    return 1\n",
                            "new_text": "def answer():\n    return 42\n",
                        },
                        call_id="call_1",
                    ),
                    LLMResponse(content="done"),
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

            controller.run("make answer return 42")

            third_prompt = "\n".join(message.content for message in llm.calls[2][0])
            self.assertIn("Stale files: app.py", third_prompt)

    def test_repeated_tool_call_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("x = 1\n", encoding="utf-8")
            graph = build_repo_graph(root)
            verifier = Verifier(root)
            env = ToolEnvironment(root, graph, verifier)
            registry = build_default_registry(env)
            evidence = EvidenceLog(root)
            repeated = {"path": "app.py", "start": 1, "end": 1}
            llm = FakeLLM(
                [
                    tool_response("read_file", repeated, call_id="call_0"),
                    tool_response("read_file", repeated, call_id="call_1"),
                    tool_response("read_file", repeated, call_id="call_2"),
                    LLMResponse(content="done"),
                ]
            )
            controller = AgentController(
                llm=llm,
                registry=registry,
                repo_graph=graph,
                evidence=evidence,
                verifier=verifier,
                auto_verify=False,
                tool_repetition_limit=2,
            )

            controller.run("inspect app.py repeatedly")

            self.assertIn("Repeated tool call blocked", evidence.format_chain(20))

    def test_initial_guidance_is_task_profiled_for_ui_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tracegraph_coder" / "web" / "assets").mkdir(parents=True)
            (root / "tracegraph_coder" / "web" / "index.html").write_text(
                '<input id="workspaceInput" />\n',
                encoding="utf-8",
            )
            (root / "tracegraph_coder" / "web" / "assets" / "app.js").write_text(
                'const workspaceInput = document.querySelector("#workspaceInput");\n',
                encoding="utf-8",
            )
            (root / "tracegraph_coder" / "web" / "assets" / "styles.css").write_text(
                ".field { display: grid; }\n",
                encoding="utf-8",
            )
            (root / "tracegraph_coder" / "web_app.py").write_text("def launch():\n    pass\n", encoding="utf-8")
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

            result = controller.run("在工作区那里添加一个按钮，点击就可以选择文件夹")

            first_prompt = "\n".join(message.content for message in llm.calls[0][0])
            self.assertIn("Task profile:", first_prompt)
            self.assertIn("- type: ui", first_prompt)
            self.assertIn("tracegraph_coder/web/index.html", first_prompt)
            self.assertIn("tracegraph_coder/web/assets/app.js", first_prompt)
            self.assertEqual(result.working_memory.task_type, "ui")

    def test_initial_guidance_uses_filesystem_candidates_when_graph_is_lazy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tracegraph_coder" / "web" / "assets").mkdir(parents=True)
            (root / "tracegraph_coder" / "web" / "index.html").write_text("<main></main>\n", encoding="utf-8")
            (root / "tracegraph_coder" / "web" / "assets" / "app.js").write_text("console.log('ui')\n", encoding="utf-8")
            (root / "tracegraph_coder" / "web" / "assets" / "styles.css").write_text("body {}\n", encoding="utf-8")
            (root / "tracegraph_coder" / "web_app.py").write_text("def launch():\n    pass\n", encoding="utf-8")
            graph = RepoGraph(root=str(root), files=[])
            verifier = Verifier(root)
            env = ToolEnvironment(root, graph, verifier, repo_graph_ready=False)
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

            result = controller.run("在工作区那里添加一个按钮，点击就可以选择文件夹")

            first_prompt = "\n".join(message.content for message in llm.calls[0][0])
            self.assertIn("tracegraph_coder/web/index.html", first_prompt)
            self.assertIn("tracegraph_coder/web/assets/app.js", first_prompt)
            self.assertIn("tracegraph_coder/web/assets/styles.css", result.working_memory.candidate_files)

    def test_harness_blocks_low_yield_ui_exploration_after_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tracegraph_coder" / "web" / "assets").mkdir(parents=True)
            (root / "tracegraph_coder" / "web" / "index.html").write_text("<main></main>\n", encoding="utf-8")
            (root / "tracegraph_coder" / "web" / "assets" / "app.js").write_text("console.log('ui')\n", encoding="utf-8")
            (root / "tracegraph_coder" / "web" / "assets" / "styles.css").write_text("body {}\n", encoding="utf-8")
            (root / "tracegraph_coder" / "web_app.py").write_text("def launch():\n    pass\n", encoding="utf-8")
            graph = RepoGraph(root=str(root), files=[])
            verifier = Verifier(root)
            env = ToolEnvironment(root, graph, verifier, repo_graph_ready=False)
            registry = build_default_registry(env)
            evidence = EvidenceLog(root)
            responses = [
                tool_response("search_text", {"pattern": f"missing_{index}"}, call_id=f"call_{index}")
                for index in range(9)
            ]
            responses.append(LLMResponse(content="done"))
            llm = FakeLLM(responses)
            controller = AgentController(
                llm=llm,
                registry=registry,
                repo_graph=graph,
                evidence=evidence,
                verifier=verifier,
                auto_verify=False,
                max_steps=12,
            )

            controller.run("在工作区那里添加一个按钮，点击就可以选择文件夹")

            chain = evidence.format_chain(80)
            self.assertIn("Harness blocked low-yield exploration", chain)
            self.assertLessEqual(chain.count("missing_8"), 1)

    def test_harness_blocks_low_novelty_broad_search_after_candidates_are_known(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tracegraph_coder" / "web" / "assets").mkdir(parents=True)
            (root / "tracegraph_coder" / "web" / "index.html").write_text("<main></main>\n", encoding="utf-8")
            (root / "tracegraph_coder" / "web" / "assets" / "app.js").write_text("console.log('ui')\n", encoding="utf-8")
            (root / "tracegraph_coder" / "web" / "assets" / "styles.css").write_text("body {}\n", encoding="utf-8")
            (root / "tracegraph_coder" / "web_app.py").write_text("def launch():\n    pass\n", encoding="utf-8")
            graph = build_repo_graph(root)
            verifier = Verifier(root)
            env = ToolEnvironment(root, graph, verifier)
            registry = build_default_registry(env)
            evidence = EvidenceLog(root)
            llm = FakeLLM(
                [
                    tool_response("search_text", {"pattern": "definitely_missing_a"}, call_id="call_0"),
                    tool_response("search_text", {"pattern": "definitely_missing_b"}, call_id="call_1"),
                    tool_response("search_text", {"pattern": "definitely_missing_c"}, call_id="call_2"),
                    LLMResponse(content="done"),
                ]
            )
            controller = AgentController(
                llm=llm,
                registry=registry,
                repo_graph=graph,
                evidence=evidence,
                verifier=verifier,
                auto_verify=False,
                max_steps=6,
            )

            result = controller.run("在工作区那里添加一个按钮，点击就可以选择文件夹")

            chain = evidence.format_chain(80)
            self.assertIn("low_novelty_exploration", chain)
            self.assertIn("definitely_missing_c", chain)
            self.assertTrue(
                any(
                    event.category == "low_novelty_exploration"
                    for event in result.working_memory.failure_events
                )
            )

    def test_harness_blocks_broad_exploration_after_mutation(self) -> None:
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
                    tool_response("read_file", {"path": "app.py", "start": 1, "end": 2}, call_id="call_0"),
                    tool_response(
                        "apply_patch",
                        {
                            "path": "app.py",
                            "old_text": "def answer():\n    return 1\n",
                            "new_text": "def answer():\n    return 42\n",
                        },
                        call_id="call_1",
                    ),
                    tool_response("repo_graph_query", {"query": "everything about this project"}, call_id="call_2"),
                    LLMResponse(content="done"),
                ]
            )
            controller = AgentController(
                llm=llm,
                registry=registry,
                repo_graph=graph,
                evidence=evidence,
                verifier=verifier,
                auto_verify=False,
                max_steps=6,
            )

            result = controller.run("make answer return 42")

            self.assertIn("42", (root / "app.py").read_text(encoding="utf-8"))
            chain = evidence.format_chain(80)
            self.assertIn("post_mutation_exploration", chain)
            self.assertIn("Failure terms", result.working_memory.render())
            self.assertTrue(
                any(
                    event.category == "post_mutation_exploration"
                    for event in result.working_memory.failure_events
                )
            )
            self.assertIn("Failure events", result.working_memory.render())

    def test_exploration_guard_nudges_after_low_gain_searching(self) -> None:
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
                    tool_response("read_file", {"path": "app.py", "start": 1, "end": 2}, call_id="call_0"),
                    tool_response("search_text", {"pattern": "answer"}, call_id="call_1"),
                    tool_response("search_text", {"pattern": "return"}, call_id="call_2"),
                    LLMResponse(content="done"),
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

            controller.run("make answer return 42")

            fourth_prompt = "\n".join(message.content for message in llm.calls[3][0])
            self.assertIn("Exploration guard", fourth_prompt)
            self.assertIn("smallest patch", fourth_prompt)
            self.assertIn("exploration_guard", evidence.format_chain(20))


if __name__ == "__main__":
    unittest.main()
