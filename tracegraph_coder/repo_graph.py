from __future__ import annotations

import ast
import fnmatch
import json
import posixpath
import re
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .safety import ensure_workspace, safe_path


IGNORE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    ".tracegraph",
}

TEXT_EXTENSIONS = {
    ".py",
    ".pyw",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cs",
    ".rb",
    ".php",
    ".md",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".ini",
    ".cfg",
    ".html",
    ".css",
}


@dataclass(slots=True)
class Symbol:
    name: str
    kind: str
    line: int


@dataclass(frozen=True, slots=True)
class GraphHit:
    node: "FileNode"
    score: int
    matched_tokens: list[str]
    reasons: list[str]


@dataclass(slots=True)
class FileNode:
    path: str
    language: str
    lines: int
    role: str = "source"
    symbols: list[Symbol] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    imports_local: list[str] = field(default_factory=list)
    imported_by: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    related_tests: list[str] = field(default_factory=list)
    related_sources: list[str] = field(default_factory=list)
    is_test: bool = False


@dataclass(frozen=True, slots=True)
class _CachedFileNodeState:
    size: int
    mtime_ns: int
    path: str
    language: str
    lines: int
    role: str
    symbols: tuple[tuple[str, str, int], ...]
    imports: tuple[str, ...]
    calls: tuple[str, ...]
    is_test: bool


_GRAPH_CACHE: dict[str, dict[str, _CachedFileNodeState]] = {}


@dataclass(slots=True)
class RepoGraph:
    root: str
    files: list[FileNode]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")

    def format_for_prompt(self, max_files: int = 80) -> str:
        lines = [
            "Repository evidence graph:",
            f"- root: {self.root}",
            f"- files indexed: {len(self.files)}",
        ]
        for node in self.files[:max_files]:
            symbols = ", ".join(f"{s.kind}:{s.name}@{s.line}" for s in node.symbols[:8])
            imports = ", ".join(node.imports[:6])
            imports_local = ", ".join(node.imports_local[:6])
            imported_by = ", ".join(node.imported_by[:6])
            calls = ", ".join(node.calls[:8])
            related_tests = ", ".join(node.related_tests[:5])
            related_sources = ", ".join(node.related_sources[:5])
            parts = [f"{node.path} ({node.language}, {node.lines} lines, role={node.role})"]
            if node.is_test:
                parts.append("test")
            if symbols:
                parts.append(f"symbols=[{symbols}]")
            if imports:
                parts.append(f"imports=[{imports}]")
            if imports_local:
                parts.append(f"imports_local=[{imports_local}]")
            if imported_by:
                parts.append(f"imported_by=[{imported_by}]")
            if calls:
                parts.append(f"calls=[{calls}]")
            if related_tests:
                parts.append(f"related_tests=[{related_tests}]")
            if related_sources:
                parts.append(f"related_sources=[{related_sources}]")
            lines.append("- " + "; ".join(parts))
        if len(self.files) > max_files:
            lines.append(f"- ... {len(self.files) - max_files} more files omitted")
        return "\n".join(lines)

    def query(self, text: str, limit: int = 12) -> list[FileNode]:
        return [hit.node for hit in self.query_with_reasons(text, limit=limit)]

    def query_with_reasons(self, text: str, limit: int = 12) -> list[GraphHit]:
        tokens = _query_tokens(text)
        if not tokens:
            return [
                GraphHit(
                    node=node,
                    score=_node_importance(node),
                    matched_tokens=[],
                    reasons=["fallback:repository-order"],
                )
                for node in self.files[:limit]
            ]
        ranked: list[GraphHit] = []
        wants_tests = _query_wants_tests(tokens)
        for node in self.files:
            hit = _score_node(node, tokens, wants_tests)
            if hit.score:
                ranked.append(hit)
        ranked.sort(
            key=lambda hit: (
                -hit.score,
                _test_sort_penalty(hit.node, wants_tests),
                -_node_importance(hit.node),
                hit.node.path,
            )
        )
        return ranked[:limit]

    def neighborhood(self, path: str, limit: int = 24) -> dict[str, object]:
        by_path = {node.path: node for node in self.files}
        clean_path = _normalize_graph_path(path)
        node = by_path.get(clean_path)
        if node is None:
            raise KeyError(path)
        rows: list[dict[str, object]] = []
        for relation, rel_paths in [
            ("imports_local", node.imports_local),
            ("imported_by", node.imported_by),
            ("related_tests", node.related_tests),
            ("related_sources", node.related_sources),
        ]:
            for rel_path in rel_paths:
                neighbor = by_path.get(rel_path)
                if neighbor is None:
                    continue
                rows.append(
                    {
                        "relation": relation,
                        "path": neighbor.path,
                        "role": neighbor.role,
                        "language": neighbor.language,
                        "lines": neighbor.lines,
                        "symbols": [f"{symbol.kind}:{symbol.name}@{symbol.line}" for symbol in neighbor.symbols[:8]],
                    }
                )
                if len(rows) >= limit:
                    break
            if len(rows) >= limit:
                break
        return {
            "path": node.path,
            "role": node.role,
            "language": node.language,
            "lines": node.lines,
            "impact_score": _node_importance(node),
            "symbols": [f"{symbol.kind}:{symbol.name}@{symbol.line}" for symbol in node.symbols[:12]],
            "imports_local": node.imports_local,
            "imported_by": node.imported_by,
            "related_tests": node.related_tests,
            "related_sources": node.related_sources,
            "neighbors": rows,
        }


def build_repo_graph(root: str | Path) -> RepoGraph:
    workspace = ensure_workspace(root)
    cache_key = str(workspace)
    previous_cache = _GRAPH_CACHE.get(cache_key, {})
    next_cache: dict[str, _CachedFileNodeState] = {}
    files: list[FileNode] = []
    for path in _iter_source_files(workspace):
        rel = path.relative_to(workspace).as_posix()
        try:
            stat_result = path.stat()
        except OSError:
            continue
        cached = previous_cache.get(rel)
        if cached is not None and cached.size == stat_result.st_size and cached.mtime_ns == stat_result.st_mtime_ns:
            node = _node_from_cache(cached)
        else:
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            node = _summarize_file(rel, content)
        files.append(node)
        next_cache[rel] = _cache_node_state(node, stat_result.st_size, stat_result.st_mtime_ns)
    _GRAPH_CACHE[cache_key] = next_cache
    _link_local_imports(files)
    _link_related_tests(files)
    files.sort(key=lambda f: (not f.is_test, f.path))
    return RepoGraph(root=str(workspace), files=files)


def _iter_source_files(root: Path):
    git_paths = _git_list_source_files(root)
    if git_paths is not None:
        yield from git_paths
        return
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        if _is_source_candidate(root, path):
            yield path


def _git_list_source_files(root: Path) -> list[Path] | None:
    try:
        proc = subprocess.run(
            ["git", "ls-files", "-co", "--exclude-standard"],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    out: list[Path] = []
    seen: set[Path] = set()
    for raw in proc.stdout.splitlines():
        if not raw.strip():
            continue
        try:
            path = safe_path(root, raw)
        except Exception:
            continue
        if path in seen or not _is_source_candidate(root, path):
            continue
        seen.add(path)
        out.append(path)
    return sorted(out, key=lambda p: p.relative_to(root).as_posix())


def _is_source_candidate(root: Path, path: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return False
    if any(part in IGNORE_DIRS for part in rel.parts):
        return False
    if path.is_dir():
        return False
    if path.suffix.lower() not in TEXT_EXTENSIONS:
        return False
    try:
        if path.stat().st_size > 512_000:
            return False
    except OSError:
        return False
    return True


def _summarize_file(path: str, content: str) -> FileNode:
    suffix = Path(path).suffix.lower()
    language = suffix[1:] if suffix else "text"
    lines = len(content.splitlines())
    is_test = _looks_like_test(path)
    if suffix in {".py", ".pyw"}:
        symbols, imports, calls = _python_summary(content)
    else:
        symbols, imports, calls = _regex_summary(content, suffix)
    return FileNode(
        path=path,
        language=language,
        lines=lines,
        role=_file_role(path, is_test),
        symbols=symbols,
        imports=imports,
        calls=calls,
        is_test=is_test,
    )


def _looks_like_test(path: str) -> bool:
    name = Path(path).name.lower()
    parts = [p.lower() for p in Path(path).parts]
    return name.startswith("test_") or name.endswith("_test.py") or "tests" in parts or "__tests__" in parts


def _python_summary(content: str) -> tuple[list[Symbol], list[str], list[str]]:
    symbols: list[Symbol] = []
    imports: list[str] = []
    calls: list[str] = []
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return _regex_summary(content, ".py")
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            symbols.append(Symbol(node.name, "class", node.lineno))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
            symbols.append(Symbol(node.name, kind, node.lineno))
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = ("." * node.level) + (node.module or "")
            if module:
                imports.append(module)
            for alias in node.names:
                if alias.name == "*":
                    continue
                if node.module:
                    imports.append(f"{module}.{alias.name}")
                elif node.level:
                    imports.append(f"{'.' * node.level}{alias.name}")
        elif isinstance(node, ast.Call):
            name = _python_call_name(node.func)
            if name:
                calls.append(name)
    symbols.sort(key=lambda s: s.line)
    return symbols[:50], sorted(set(imports))[:50], _unique(calls)[:80]


def _regex_summary(content: str, suffix: str) -> tuple[list[Symbol], list[str], list[str]]:
    symbols: list[Symbol] = []
    imports: list[str] = []
    calls: list[str] = []
    for lineno, line in enumerate(content.splitlines(), start=1):
        stripped = line.strip()
        if suffix in {".js", ".jsx", ".ts", ".tsx"}:
            m = re.match(r"(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)", stripped)
            if m:
                symbols.append(Symbol(m.group(1), "function", lineno))
            m = re.match(r"(?:export\s+)?class\s+([A-Za-z_$][\w$]*)", stripped)
            if m:
                symbols.append(Symbol(m.group(1), "class", lineno))
            imports.extend(_js_imports(stripped))
            calls.extend(_regex_calls(stripped))
        elif suffix == ".java":
            m = re.match(r"(?:public|private|protected)?\s*(?:class|interface|enum)\s+(\w+)", stripped)
            if m:
                symbols.append(Symbol(m.group(1), "type", lineno))
            calls.extend(_regex_calls(stripped))
        elif suffix == ".go":
            m = re.match(r"func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)", stripped)
            if m:
                symbols.append(Symbol(m.group(1), "function", lineno))
            calls.extend(_regex_calls(stripped))
        elif suffix == ".rs":
            m = re.match(r"(?:pub\s+)?(?:async\s+)?fn\s+(\w+)", stripped)
            if m:
                symbols.append(Symbol(m.group(1), "function", lineno))
            calls.extend(_regex_calls(stripped))
    return symbols[:50], sorted(set(imports))[:50], _unique(calls)[:80]


def _js_imports(line: str) -> list[str]:
    imports: list[str] = []
    for pattern in (
        r"\bimport\s+.*?\s+from\s+['\"]([^'\"]+)['\"]",
        r"\bimport\s+['\"]([^'\"]+)['\"]",
        r"\bexport\s+.*?\s+from\s+['\"]([^'\"]+)['\"]",
        r"\brequire\(\s*['\"]([^'\"]+)['\"]\s*\)",
    ):
        imports.extend(match.group(1) for match in re.finditer(pattern, line))
    return imports


def _python_call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _regex_calls(line: str) -> list[str]:
    ignored = {
        "class",
        "def",
        "for",
        "function",
        "if",
        "import",
        "return",
        "switch",
        "while",
    }
    out: list[str] = []
    for match in re.finditer(r"\b([A-Za-z_$][\w$]*)\s*\(", line):
        name = match.group(1)
        if name.lower() not in ignored:
            out.append(name)
    return out


def _cache_node_state(node: FileNode, size: int, mtime_ns: int) -> _CachedFileNodeState:
    return _CachedFileNodeState(
        size=size,
        mtime_ns=mtime_ns,
        path=node.path,
        language=node.language,
        lines=node.lines,
        role=node.role,
        symbols=tuple((symbol.name, symbol.kind, symbol.line) for symbol in node.symbols),
        imports=tuple(node.imports),
        calls=tuple(node.calls),
        is_test=node.is_test,
    )


def _node_from_cache(state: _CachedFileNodeState) -> FileNode:
    return FileNode(
        path=state.path,
        language=state.language,
        lines=state.lines,
        role=state.role,
        symbols=[Symbol(name, kind, line) for name, kind, line in state.symbols],
        imports=list(state.imports),
        calls=list(state.calls),
        is_test=state.is_test,
    )


def _link_local_imports(files: list[FileNode]) -> None:
    by_path = {node.path: node for node in files}
    module_to_path: dict[str, str] = {}
    for node in files:
        _add_path_aliases(module_to_path, node.path)

    for node in files:
        local: list[str] = []
        for raw in node.imports:
            local.extend(_resolve_local_import(node.path, raw, by_path, module_to_path))
        node.imports_local = _unique(local)
        node.imported_by = []

    for node in files:
        for target_path in node.imports_local:
            target = by_path.get(target_path)
            if target and node.path not in target.imported_by:
                target.imported_by.append(node.path)

    for node in files:
        node.imported_by.sort()


def _link_related_tests(files: list[FileNode]) -> None:
    source_nodes = [node for node in files if not node.is_test]
    test_nodes = [node for node in files if node.is_test]
    by_path = {node.path: node for node in files}
    for source in source_nodes:
        scored: list[tuple[int, str]] = []
        source_stem = Path(source.path).stem.lower()
        source_module = _module_name(source.path)
        source_symbols = {symbol.name.lower() for symbol in source.symbols}
        source_path_tokens = set(_path_tokens(source.path))
        for test in test_nodes:
            score = 0
            test_path = test.path.lower()
            test_imports = {item.lower() for item in test.imports}
            test_local_imports = set(test.imports_local)
            test_calls = {item.lower() for item in test.calls}
            test_hay = " ".join(
                [
                    test.path,
                    " ".join(symbol.name for symbol in test.symbols),
                    " ".join(test.imports),
                    " ".join(test.imports_local),
                    " ".join(test.calls),
                ]
            ).lower()
            if source.path in test_local_imports:
                score += 8
            if source_stem and source_stem in test_path:
                score += 6
            if source_module and source_module in test_imports:
                score += 5
            if source_symbols & test_calls:
                score += 3
            if source_symbols and any(symbol in test_hay for symbol in source_symbols):
                score += 2
            if source_path_tokens & set(_path_tokens(test.path)):
                score += 1
            if score:
                scored.append((score, test.path))
        scored.sort(key=lambda item: (-item[0], item[1]))
        source.related_tests = [path for _, path in scored[:8]]
        for test_path in source.related_tests:
            test = by_path.get(test_path)
            if test and source.path not in test.related_sources:
                test.related_sources.append(source.path)

    for test in test_nodes:
        test.related_sources.sort()


def _resolve_local_import(
    importer_path: str,
    raw_import: str,
    by_path: dict[str, FileNode],
    module_to_path: dict[str, str],
) -> list[str]:
    raw = raw_import.strip()
    if not raw:
        return []
    if raw.startswith("."):
        return _resolve_relative_import(importer_path, raw, by_path, module_to_path)
    if raw.startswith(("/", "\\")):
        return []
    resolved = module_to_path.get(raw.lower())
    if resolved:
        return [resolved]
    path_matches: list[str] = []
    if raw.startswith(("@/", "~/")):
        tail = raw[2:].lstrip("/")
        path_matches.extend(_resolve_path_like_import(f"src/{tail}", by_path))
        path_matches.extend(_resolve_path_like_import(tail, by_path))
    elif "/" in raw or "\\" in raw:
        path_matches.extend(_resolve_path_like_import(raw, by_path))
    return _unique(path_matches)


def _resolve_relative_import(
    importer_path: str,
    raw_import: str,
    by_path: dict[str, FileNode],
    module_to_path: dict[str, str],
) -> list[str]:
    if raw_import.startswith(".") and not raw_import.startswith("./") and not raw_import.startswith("../"):
        return _resolve_python_relative_import(importer_path, raw_import, module_to_path)
    importer_dir = Path(importer_path).parent
    current = posixpath.normpath((importer_dir / raw_import).as_posix())
    return _resolve_path_like_import(current, by_path)


def _resolve_path_like_import(raw_path: str, by_path: dict[str, FileNode]) -> list[str]:
    current = posixpath.normpath(raw_path.replace("\\", "/").strip("/"))
    if current in {"", "."}:
        return []
    candidates = [current]
    for suffix in (".py", ".pyw", ".js", ".jsx", ".ts", ".tsx", ".json", ".css"):
        candidates.append(current + suffix)
    for suffix in (".py", ".pyw", ".js", ".jsx", ".ts", ".tsx"):
        candidates.append(f"{current}/index{suffix}")
        if suffix in {".py", ".pyw"}:
            candidates.append(f"{current}/__init__{suffix}")
    return _unique(path for path in candidates if path in by_path)


def _resolve_python_relative_import(
    importer_path: str,
    raw_import: str,
    module_to_path: dict[str, str],
) -> list[str]:
    dots = len(raw_import) - len(raw_import.lstrip("."))
    module_tail = raw_import[dots:]
    package_parts = Path(importer_path).with_suffix("").parts[:-1]
    if Path(importer_path).name == "__init__.py":
        package_parts = Path(importer_path).with_suffix("").parts[:-1]
    keep = max(0, len(package_parts) - max(0, dots - 1))
    base = ".".join(package_parts[:keep])
    module = ".".join(part for part in [base, module_tail] if part)
    resolved = module_to_path.get(module.lower())
    return [resolved] if resolved else []


def _add_path_aliases(module_to_path: dict[str, str], path: str) -> None:
    suffix = Path(path).suffix.lower()
    if suffix not in {".py", ".pyw", ".js", ".jsx", ".ts", ".tsx", ".json", ".css"}:
        return
    module = _module_name(path)
    without_suffix = Path(path).with_suffix("").as_posix()
    aliases = [module, without_suffix]
    if module.endswith(".__init__"):
        aliases.append(module[: -len(".__init__")])
    if module.endswith(".index"):
        aliases.append(module[: -len(".index")])
    if without_suffix.endswith("/__init__"):
        aliases.append(without_suffix[: -len("/__init__")])
    if without_suffix.endswith("/index"):
        aliases.append(without_suffix[: -len("/index")])
    for alias in aliases:
        if alias:
            module_to_path.setdefault(alias.lower(), path)


def _module_name(path: str) -> str:
    without_suffix = Path(path).with_suffix("").as_posix()
    if without_suffix.endswith("/__init__"):
        without_suffix = without_suffix[: -len("/__init__")]
    return without_suffix.replace("/", ".").lower()


def _file_role(path: str, is_test: bool) -> str:
    if is_test:
        return "test"
    name = Path(path).name.lower()
    suffix = Path(path).suffix.lower()
    if name in {"pyproject.toml", "package.json", "cargo.toml", "go.mod", "pom.xml"}:
        return "manifest"
    if suffix in {".toml", ".yaml", ".yml", ".json", ".ini", ".cfg"}:
        return "config"
    if suffix in {".md", ".txt"}:
        return "doc"
    return "source"


def _path_tokens(path: str) -> list[str]:
    generic = {
        "cfg",
        "css",
        "html",
        "ini",
        "java",
        "json",
        "jsx",
        "md",
        "py",
        "rs",
        "src",
        "test",
        "tests",
        "toml",
        "tsx",
        "txt",
        "yaml",
        "yml",
    }
    without_suffix = Path(path).with_suffix("").as_posix()
    tokens = _identifier_tokens(without_suffix)
    return [token for token in tokens if token and token not in generic]


def _node_token_map(node: FileNode) -> dict[str, set[str]]:
    path_tokens = set(_path_tokens(node.path))
    symbol_tokens: set[str] = set()
    for symbol in node.symbols:
        symbol_tokens.update(_identifier_tokens(symbol.name))
        symbol_tokens.add(symbol.name.lower())
    call_tokens: set[str] = set()
    for call in node.calls:
        call_tokens.update(_identifier_tokens(call))
        call_tokens.add(call.lower())
    relation_tokens: set[str] = set()
    for value in [*node.imports_local, *node.imported_by, *node.related_tests, *node.related_sources]:
        relation_tokens.update(_path_tokens(value))
    return {
        "path": path_tokens,
        "symbols": symbol_tokens,
        "calls": call_tokens,
        "local_relations": relation_tokens,
    }


def _score_node(node: FileNode, tokens: list[str], wants_tests: bool) -> GraphHit:
    token_map = _node_token_map(node)
    hay = " ".join(
        [
            node.path,
            node.role,
            node.language,
            " ".join(s.name for s in node.symbols),
            " ".join(node.imports),
            " ".join(node.imports_local),
            " ".join(node.imported_by),
            " ".join(node.calls),
            " ".join(node.related_tests),
            " ".join(node.related_sources),
        ]
    ).lower()
    score = 0
    matched_tokens: list[str] = []
    reasons: list[str] = []
    for token in tokens:
        reason = ""
        if token in token_map["path"]:
            score += 6
            reason = "path"
        elif token in token_map["symbols"]:
            score += 5
            reason = "symbol"
        elif token in token_map["local_relations"]:
            score += 4
            reason = "graph-relation"
        elif token in node.path.lower():
            score += 4
            reason = "path-substring"
        elif token in token_map["calls"]:
            score += 2
            reason = "call"
        elif token in hay:
            score += 1
            reason = "metadata"
        if reason:
            matched_tokens.append(token)
            reasons.append(reason)
    if node.is_test and wants_tests:
        score += 4
        reasons.append("test-intent")
    if score:
        importance_bonus = min(4, _node_importance(node))
        if importance_bonus:
            score += importance_bonus
            reasons.append("graph-importance")
    return GraphHit(
        node=node,
        score=score,
        matched_tokens=_unique(matched_tokens),
        reasons=_unique(reasons),
    )


def _node_importance(node: FileNode) -> int:
    return min(
        24,
        len(node.imported_by) * 3
        + len(node.related_tests) * 2
        + len(node.imports_local)
        + min(6, len(node.symbols) // 4),
    )


def _query_wants_tests(tokens: list[str]) -> bool:
    return any(token in {"test", "tests", "pytest", "unittest", "verify", "验证", "测试"} for token in tokens)


def _test_sort_penalty(node: FileNode, wants_tests: bool) -> int:
    if wants_tests:
        return 0 if node.is_test else 1
    return 1 if node.is_test else 0


def _identifier_tokens(text: str) -> list[str]:
    raw_tokens: list[str] = []
    for part in re.split(r"[^A-Za-z0-9\u4e00-\u9fff]+", text):
        if not part:
            continue
        raw_tokens.append(part)
        raw_tokens.extend(re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", part))
    tokens: list[str] = []
    for raw in raw_tokens:
        if re.search(r"[\u4e00-\u9fff]", raw):
            tokens.append(raw)
            if len(raw) > 1:
                tokens.extend(raw[index : index + 2] for index in range(len(raw) - 1))
        else:
            tokens.append(raw.lower())
    return _unique([token for token in tokens if token])


def _normalize_graph_path(path: str) -> str:
    clean = path.replace("\\", "/").strip()
    if clean.startswith("./"):
        clean = clean[2:]
    return posixpath.normpath(clean)


def _unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def filter_files(root: str | Path, pattern: str = "*", limit: int = 200) -> list[str]:
    workspace = ensure_workspace(root)
    out: list[str] = []
    for path in _iter_source_files(workspace):
        rel = path.relative_to(workspace).as_posix()
        if fnmatch.fnmatch(rel, pattern):
            out.append(rel)
        if len(out) >= limit:
            break
    return sorted(out)


def read_text_region(root: str | Path, path: str, start: int = 1, end: int | None = None) -> str:
    target = safe_path(root, path)
    if not target.exists():
        raise FileNotFoundError(path)
    if target.is_dir():
        raise IsADirectoryError(path)
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return "(empty file)"
    start = max(start, 1)
    end = len(lines) if end is None else min(end, len(lines))
    if start > len(lines):
        raise ValueError(f"start line {start} is beyond file length {len(lines)}")
    max_lines = 240
    truncated = end - start + 1 > max_lines
    if truncated:
        end = start + max_lines - 1
    width = len(str(end))
    rows = [f"{i:>{width}} | {lines[i - 1]}" for i in range(start, end + 1)]
    if truncated:
        rows.append(f"... truncated; continue with start={end + 1}")
    return "\n".join(rows)


def _query_tokens(text: str) -> list[str]:
    tokens = _identifier_tokens(text)
    seen: set[str] = set()
    out: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out
