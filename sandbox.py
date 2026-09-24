"""
sandbox.py
----------
Isolated execution environment for LLM-generated Pandas/Plotly code.

Design contract with the LLM (see agent.py's prompts):
  - The dataframe is already loaded into a variable called `df`.
  - The final Plotly figure MUST be assigned to a variable called `fig`.
  - The code must NOT read/write files, hit the network, or shell out.

Safety layers:
  1. Static denylist scan of the generated source (defense in depth --
     this is NOT a sandbox by itself, it just catches obviously bad code
     before we even bother spinning up a subprocess).
  2. Execution happens in a *separate* `python -I` (isolated mode)
     subprocess, not in-process -- a crash or infinite loop cannot take
     down the Streamlit app.
  3. Hard wall-clock timeout (default 30s) via subprocess.run(timeout=...).
  4. Hard memory ceiling (default 512MB) via resource.setrlimit on
     POSIX systems (best-effort no-op on Windows).
  5. Output artifacts are only ever written to a designated directory
     that we control (the LLM never chooses the path).
"""

from __future__ import annotations

import os
import re
import sys
import uuid
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MEMORY_LIMIT_MB = 512

# Tokens that should never appear in LLM-generated analysis code.
# This is a coarse, defense-in-depth denylist -- not a substitute for
# running in an isolated subprocess, which is the real boundary.
FORBIDDEN_PATTERNS = [
    r"\bos\.system\b",
    r"\bos\.popen\b",
    r"\bos\.remove\b",
    r"\bos\.rmdir\b",
    r"\bos\.unlink\b",
    r"\bshutil\.rmtree\b",
    r"\bsubprocess\b",
    r"\bsocket\b",
    r"\b__import__\s*\(",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bopen\s*\([^)]*[\"'](w|a|wb|ab)[\"']",  # writing/appending to files
    r"\bsys\.exit\b",
    r"\bpip\b",
    r"\bimport\s+ctypes\b",
    r"\bimport\s+socket\b",
]


class UnsafeCodeError(Exception):
    """Raised when generated code trips the static safety denylist."""


@dataclass
class ExecutionResult:
    success: bool
    stdout: str = ""
    stderr: str = ""
    error_trace: str = ""
    artifact_path: str | None = None       # HTML (interactive) artifact
    artifact_png_path: str | None = None   # optional static PNG artifact
    timed_out: bool = False
    blocked_reason: str | None = None


def static_safety_check(code: str) -> None:
    """Raise UnsafeCodeError if the code matches a forbidden pattern."""
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, code):
            raise UnsafeCodeError(f"Blocked forbidden pattern: `{pattern}`")


def _loader_snippet() -> str:
    """Returns the source of the dataframe loader, inlined into the runner
    script so the subprocess has zero dependency on this module's import
    path."""
    return '''
def _load_df(path):
    import pandas as pd
    lower = path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    if lower.endswith(".json"):
        return pd.read_json(path)
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    if lower.endswith((".db", ".sqlite", ".sqlite3")):
        import sqlite3
        conn = sqlite3.connect(path)
        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table'", conn
        )
        if tables.empty:
            raise ValueError("SQLite file has no tables")
        table_name = tables.iloc[0]["name"]
        return pd.read_sql(f"SELECT * FROM {table_name}", conn)
    raise ValueError(f"Unsupported dataset type: {path}")
'''


def _build_runner_script(
    dataset_path: str,
    user_code: str,
    artifact_html_path: str,
    artifact_png_path: str,
) -> str:
    """Assembles the full standalone Python script that will be executed
    in the child process."""
    # Indent the user's code so it can be dropped inside a try/except block.
    indented_user_code = "\n".join(
        "    " + line if line.strip() else "" for line in user_code.splitlines()
    )

    return f'''
import sys
import traceback
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

{_loader_snippet()}

DATASET_PATH = {dataset_path!r}
ARTIFACT_HTML = {artifact_html_path!r}
ARTIFACT_PNG = {artifact_png_path!r}

df = _load_df(DATASET_PATH)
fig = None

try:
{indented_user_code}
except Exception:
    traceback.print_exc()
    sys.exit(1)

if fig is None:
    print("SANDBOX_ERROR: no `fig` (Plotly figure) variable was produced.", file=sys.stderr)
    sys.exit(2)

fig.write_html(ARTIFACT_HTML, include_plotlyjs="cdn")
print(f"ARTIFACT_HTML_SAVED::{{ARTIFACT_HTML}}")

try:
    fig.write_image(ARTIFACT_PNG, width=1000, height=600, scale=2)
    print(f"ARTIFACT_PNG_SAVED::{{ARTIFACT_PNG}}")
except Exception as png_err:
    # PNG export needs kaleido; treat as non-fatal since HTML already saved.
    print(f"PNG export skipped: {{png_err}}", file=sys.stderr)
'''


def _preexec_memory_limit(memory_limit_mb: int):
    """Returns a preexec_fn that caps virtual memory on POSIX systems."""
    if platform.system() == "Windows":
        return None

    def _limit():
        import resource

        limit_bytes = memory_limit_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        except (ValueError, resource.error):
            # Some sandboxed/containerized environments disallow lowering
            # RLIMIT_AS further -- fail open rather than crash the run.
            pass

    return _limit


def run_generated_code(
    code: str,
    dataset_path: str,
    artifacts_dir: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
) -> ExecutionResult:
    """Execute `code` in an isolated subprocess against `dataset_path`.

    Returns an ExecutionResult with captured stdout/stderr and, on success,
    the path to the rendered chart artifact(s).
    """
    try:
        static_safety_check(code)
    except UnsafeCodeError as e:
        return ExecutionResult(
            success=False,
            blocked_reason=str(e),
            error_trace=f"UnsafeCodeError: {e}",
        )

    artifacts_dir_path = Path(artifacts_dir).resolve()
    artifacts_dir_path.mkdir(parents=True, exist_ok=True)
    dataset_abs_path = str(Path(dataset_path).resolve())

    run_id = uuid.uuid4().hex[:10]
    artifact_html = str(artifacts_dir_path / f"chart_{run_id}.html")
    artifact_png = str(artifacts_dir_path / f"chart_{run_id}.png")

    runner_script = _build_runner_script(
        dataset_path=dataset_abs_path,
        user_code=code,
        artifact_html_path=artifact_html,
        artifact_png_path=artifact_png,
    )

    script_path = artifacts_dir_path / f"_runner_{run_id}.py"
    script_path.write_text(runner_script, encoding="utf-8")

    try:
        proc = subprocess.run(
            [sys.executable, "-I", str(script_path.resolve())],  # -I = isolated mode
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            preexec_fn=_preexec_memory_limit(memory_limit_mb),
            cwd=str(artifacts_dir_path),
        )
    except subprocess.TimeoutExpired as e:
        return ExecutionResult(
            success=False,
            timed_out=True,
            stdout=e.stdout or "",
            stderr=e.stderr or "",
            error_trace=f"Execution exceeded {timeout_seconds}s timeout and was killed.",
        )
    finally:
        # Clean up the transient runner script; keep chart artifacts.
        try:
            script_path.unlink(missing_ok=True)
        except Exception:
            pass

    success = proc.returncode == 0 and "ARTIFACT_HTML_SAVED::" in proc.stdout

    return ExecutionResult(
        success=success,
        stdout=proc.stdout,
        stderr=proc.stderr,
        error_trace="" if success else (proc.stderr or proc.stdout),
        artifact_path=artifact_html if success else None,
        artifact_png_path=artifact_png if (success and Path(artifact_png).exists()) else None,
    )
