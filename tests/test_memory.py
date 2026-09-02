from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tracegraph_coder.controller import AgentController
from tracegraph_coder.evidence import EvidenceLog, EvidenceStep
from tracegraph_coder.memory import load_project_memory
from tracegraph_coder.models import LLMResponse
from tracegraph_coder.repo_graph import build_repo_graph
from tracegraph_coder.tools import ToolEnvironment, build_default_registry
from tracegraph_coder.verifier import Verifier
from tracegraph_coder.working_memory import WorkingMemory


class FakeLLM:
    def __init__(self):
        self.calls = []

    def chat(self, messages, tools):
        self.calls.append((messages, tools))
        return LLMResponse(content="done")


class MemoryAndEvidenceTests(unittest.TestCase):
    def test_project_memory_is_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "TRACEGRAPH.md").write_text("Use unittest for verification.", encoding="utf-8")
            memory = load_project_memory(root)
            self.assertIn("TRACEGRAPH.md", memory)
            self.assertIn("Use unittest", memory)

    def test_controller_includes_project_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = build_repo_graph(root)
            verifier = Verifier(root)
            env = ToolEnvironment(root, graph, verifier)
            registry = build_default_registry(env)
            evidence = EvidenceLog(root)
            llm = FakeLLM()
            controller = AgentController(
                llm=llm,
                registry=registry,
                repo_graph=graph,
                evidence=evidence,
                verifier=verifier,
                project_memory="Always run unittest.",
                auto_verify=False,
            )
            controller.run("say done")
            self.assertIn("Always run unittest.", llm.calls[0][0][1].content)

    def test_evidence_chain_formats_recent_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evidence = EvidenceLog(tmp)
            evidence.append(
                EvidenceStep(
                    iteration=1,
                    stage="tool",
                    action="read_file",
                    args={"path": "app.py"},
                    ok=True,
                    observation="read app.py",
                )
            )
            chain = evidence.format_chain()
            self.assertIn("tool:read_file", chain)
            self.assertIn("app.py", chain)

    def test_working_memory_tracks_experience_and_failure_terms(self) -> None:
        memory = WorkingMemory("fix flaky timeout")
        memory.observe_progress(
            {
                "experience_terms": ["timeout", "retry", "timeout"],
                "failure_terms": ["traceback", "timeout"],
            },
            1,
        )

        rendered = memory.render()
        restored = WorkingMemory.from_dict(memory.to_dict())

        self.assertIn("Experience terms: timeout, retry", rendered)
        self.assertIn("Failure terms: traceback, timeout", rendered)
        self.assertEqual(restored.experience_terms, ["timeout", "retry"])
        self.assertEqual(restored.failure_terms, ["traceback", "timeout"])

    def test_working_memory_tracks_structured_failure_events(self) -> None:
        memory = WorkingMemory("fix repeated search loop")
        memory.observe_failure_event(
            source="harness",
            category="low_novelty_exploration",
            summary="Broad search repeated after target files were known.",
            step=3,
            terms=["search", "target"],
            paths=["tracegraph_coder/web/assets/app.js"],
        )

        rendered = memory.render()
        restored = WorkingMemory.from_dict(memory.to_dict())

        self.assertIn("Failure events:", rendered)
        self.assertIn("low_novelty_exploration", rendered)
        self.assertIn("tracegraph_coder/web/assets/app.js", rendered)
        self.assertEqual(memory.failure_terms, ["search", "target"])
        self.assertEqual(restored.failure_events[0].category, "low_novelty_exploration")
        self.assertEqual(restored.failure_events[0].paths, ["tracegraph_coder/web/assets/app.js"])


if __name__ == "__main__":
    unittest.main()
