from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tracegraph_coder.repo_graph import build_repo_graph, read_text_region


class RepoGraphTests(unittest.TestCase):
    def test_python_symbols_are_indexed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text(
                "import os\n\nclass Service:\n    pass\n\ndef run():\n    return os.getcwd()\n",
                encoding="utf-8",
            )
            graph = build_repo_graph(root)
            node = next(f for f in graph.files if f.path == "app.py")
            names = {s.name for s in node.symbols}
            self.assertIn("Service", names)
            self.assertIn("run", names)
            self.assertIn("os", node.imports)

    def test_pyw_launcher_is_indexed_as_python(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "launcher.pyw").write_text("def main():\n    return 0\n", encoding="utf-8")
            graph = build_repo_graph(root)
            node = next(f for f in graph.files if f.path == "launcher.pyw")
            self.assertIn("main", {s.name for s in node.symbols})

    def test_python_calls_and_related_tests_are_indexed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text(
                "def helper():\n    return 41\n\n"
                "def answer():\n    return helper() + 1\n",
                encoding="utf-8",
            )
            tests = root / "tests"
            tests.mkdir()
            (tests / "test_app.py").write_text(
                "from app import answer\n\n"
                "def test_answer():\n    assert answer() == 42\n",
                encoding="utf-8",
            )
            graph = build_repo_graph(root)
            node = next(f for f in graph.files if f.path == "app.py")
            self.assertIn("helper", node.calls)
            self.assertIn("tests/test_app.py", node.related_tests)
            test_node = next(f for f in graph.files if f.path == "tests/test_app.py")
            self.assertIn("app.py", test_node.imports_local)
            self.assertIn("app.py", test_node.related_sources)

    def test_local_imports_reverse_edges_and_neighborhood_are_indexed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "pkg"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "service.py").write_text(
                "from .storage import load_user\n\n"
                "def get_user():\n"
                "    return load_user()\n",
                encoding="utf-8",
            )
            (package / "storage.py").write_text(
                "def load_user():\n"
                "    return {'name': 'Ada'}\n",
                encoding="utf-8",
            )
            tests = root / "tests"
            tests.mkdir()
            (tests / "test_service.py").write_text(
                "from pkg.service import get_user\n\n"
                "def test_get_user():\n"
                "    assert get_user()['name'] == 'Ada'\n",
                encoding="utf-8",
            )

            graph = build_repo_graph(root)
            service = next(f for f in graph.files if f.path == "pkg/service.py")
            storage = next(f for f in graph.files if f.path == "pkg/storage.py")
            test = next(f for f in graph.files if f.path == "tests/test_service.py")

            self.assertEqual(service.role, "source")
            self.assertIn("pkg/storage.py", service.imports_local)
            self.assertIn("pkg/service.py", storage.imported_by)
            self.assertIn("tests/test_service.py", service.related_tests)
            self.assertIn("pkg/service.py", test.related_sources)

            neighborhood = graph.neighborhood("pkg/service.py")
            neighbor_paths = {item["path"] for item in neighborhood["neighbors"]}
            self.assertIn("pkg/storage.py", neighbor_paths)
            self.assertIn("tests/test_service.py", neighbor_paths)

    def test_query_splits_identifier_names_and_prioritizes_tests_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "user_service.py").write_text(
                "class CreateUserService:\n"
                "    def create_user(self):\n"
                "        return True\n",
                encoding="utf-8",
            )
            tests = root / "tests"
            tests.mkdir()
            (tests / "test_user_service.py").write_text(
                "from user_service import CreateUserService\n\n"
                "def test_create_user():\n"
                "    assert CreateUserService().create_user()\n",
                encoding="utf-8",
            )

            graph = build_repo_graph(root)

            self.assertEqual(graph.query("create user service")[0].path, "user_service.py")
            self.assertEqual(graph.query("test create user service")[0].path, "tests/test_user_service.py")

    def test_javascript_relative_imports_are_linked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src"
            src.mkdir()
            (src / "app.ts").write_text(
                "import { helper } from './helper'\n"
                "export function runApp() { return helper() }\n",
                encoding="utf-8",
            )
            (src / "helper.ts").write_text(
                "export function helper() { return 42 }\n",
                encoding="utf-8",
            )

            graph = build_repo_graph(root)
            app = next(f for f in graph.files if f.path == "src/app.ts")
            helper = next(f for f in graph.files if f.path == "src/helper.ts")

            self.assertIn("src/helper.ts", app.imports_local)
            self.assertIn("src/app.ts", helper.imported_by)

    def test_python_from_package_import_alias_links_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "pkg"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "storage.py").write_text("def load():\n    return 1\n", encoding="utf-8")
            (package / "service.py").write_text(
                "from pkg import storage\n\n"
                "def get_value():\n"
                "    return storage.load()\n",
                encoding="utf-8",
            )

            graph = build_repo_graph(root)
            service = next(f for f in graph.files if f.path == "pkg/service.py")
            storage = next(f for f in graph.files if f.path == "pkg/storage.py")

            self.assertIn("pkg/storage.py", service.imports_local)
            self.assertIn("pkg/service.py", storage.imported_by)

    def test_javascript_alias_imports_are_linked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            components = root / "src" / "components"
            components.mkdir(parents=True)
            (root / "src" / "app.ts").write_text(
                "import { Button } from '@/components/Button'\n"
                "export function renderApp() { return Button() }\n",
                encoding="utf-8",
            )
            (components / "Button.ts").write_text(
                "export function Button() { return 'ok' }\n",
                encoding="utf-8",
            )

            graph = build_repo_graph(root)
            app = next(f for f in graph.files if f.path == "src/app.ts")
            button = next(f for f in graph.files if f.path == "src/components/Button.ts")

            self.assertIn("src/components/Button.ts", app.imports_local)
            self.assertIn("src/app.ts", button.imported_by)

    def test_neighborhood_accepts_windows_style_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "pkg"
            package.mkdir()
            (package / "service.py").write_text("from .storage import load\n", encoding="utf-8")
            (package / "storage.py").write_text("def load():\n    return 1\n", encoding="utf-8")

            graph = build_repo_graph(root)
            neighborhood = graph.neighborhood("pkg\\service.py")

            self.assertEqual(neighborhood["path"], "pkg/service.py")
            self.assertIn("pkg/storage.py", neighborhood["imports_local"])

    def test_read_text_region_has_line_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
            self.assertEqual(read_text_region(root, "a.txt", 2, 3), "2 | two\n3 | three")

    def test_read_text_region_handles_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "empty.txt").write_text("", encoding="utf-8")
            self.assertEqual(read_text_region(root, "empty.txt"), "(empty file)")

    def test_query_matches_chinese_path_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "用户服务.py").write_text("def create_user():\n    return True\n", encoding="utf-8")
            graph = build_repo_graph(root)
            hits = graph.query("用户注册服务")
            self.assertEqual(hits[0].path, "用户服务.py")

    def test_read_text_region_truncates_large_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("\n".join(str(i) for i in range(300)), encoding="utf-8")
            output = read_text_region(root, "a.txt")
            self.assertIn("truncated", output)
            self.assertIn("continue with start=241", output)


if __name__ == "__main__":
    unittest.main()
