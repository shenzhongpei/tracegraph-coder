from __future__ import annotations

import json
import shlex
import subprocess
import sys
import importlib.util
from dataclasses import dataclass
from pathlib import Path

from .safety import assert_safe_command, ensure_workspace, redact_secrets, split_safe_command


@dataclass(slots=True)
class VerificationResult:
    ok: bool
    command: str | None
    output: str


class Verifier:
    def __init__(self, workspace: str | Path, timeout: int = 120):
        self.workspace = ensure_workspace(workspace)
        self.timeout = timeout

    def detect_command(self) -> str | None:
        root = self.workspace
        if (root / "pyproject.toml").exists() or (root / "pytest.ini").exists() or (root / "tests").exists():
            quoted_python = shlex.quote(sys.executable)
            if (root / "pytest.ini").exists() or importlib.util.find_spec("pytest") is not None:
                return f"{quoted_python} -m pytest -q"
            if (root / "tests").exists():
                return f"{quoted_python} -m unittest discover -s tests -v"
            return f"{quoted_python} -m compileall ."
        if (root / "package.json").exists():
            try:
                data = json.loads((root / "package.json").read_text(encoding="utf-8"))
                scripts = data.get("scripts", {})
                if "test" in scripts:
                    return "npm test -- --runInBand"
            except Exception:
                return "npm test"
        if (root / "Cargo.toml").exists():
            return "cargo test"
        if (root / "go.mod").exists():
            return "go test ./..."
        if (root / "pom.xml").exists():
            return "mvn test"
        return None

    def run(self, command: str | None = None) -> VerificationResult:
        cmd = command or self.detect_command()
        if not cmd:
            return VerificationResult(True, None, "No verifier command detected.")
        try:
            assert_safe_command(cmd)
            args = split_safe_command(cmd)
            proc = subprocess.run(
                args,
                cwd=self.workspace,
                shell=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=self.timeout,
            )
            out = redact_secrets(proc.stdout or "")
            return VerificationResult(proc.returncode == 0, cmd, out[-50000:])
        except subprocess.TimeoutExpired as exc:
            partial = ((exc.stdout or "") + (exc.stderr or "")) if isinstance(exc.stdout, str) else ""
            return VerificationResult(False, cmd, f"Verifier timed out after {self.timeout}s.\n{partial[-5000:]}")
        except Exception as exc:
            return VerificationResult(False, cmd, f"Verifier failed: {exc}")
