"""Code check (step 9.5): does the Python code in the notes actually run?

Code blocks are copied from what the lecture showed, so a transcription slip (a dropped
colon, `0` read as `O`, lost indentation) makes them useless to a student. Two levels:

- **compiles** — always: `compile()` only parses, nothing is executed.
- **runs** — only when a runner is given. The code is transcribed from someone else's video,
  so it is untrusted: it is never run by the app, only here, preferably in Docker
  (`DockerRunner`: no network, read-only filesystem, memory/CPU/pid limits, nobody user).
  `LocalRunner` (isolated interpreter, empty environment so no secrets, temp directory,
  timeout, rlimits on POSIX) is for a reviewed local check only.

Lecture snippets often build on earlier ones (a list defined in one block, used in the
next), so a block that fails with NameError alone is retried with the video's earlier
blocks in front of it. Blocks that need keyboard input or a third-party package can't be
judged here and are reported separately, not counted as failures.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from typing import Literal, Protocol

from lecnotes.domain import CodeBlock, KnowledgeUnit

Status = Literal[
    "ok", "syntax_error", "runtime_error", "timeout", "needs_input", "missing_module", "skipped"
]
PYTHON = {"", "python", "python3", "py"}
_ERROR_LINE = re.compile(r"^(\w+(?:Error|Exception|Exit|Interrupt))\b", re.M)


@dataclass(frozen=True)
class RunOutcome:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


class Runner(Protocol):
    name: str

    def run(self, source: str, timeout_s: float) -> RunOutcome: ...


@dataclass(frozen=True)
class BlockResult:
    unit_id: str
    index: int
    status: Status
    error: str | None = None
    in_context: bool = False
    output_matches: bool | None = None
    """For a block that ran and shows its output in the notes: did it print exactly that?"""


@dataclass(frozen=True)
class CodeScore:
    results: list[BlockResult] = field(default_factory=list)
    executed: bool = False

    def count(self, *statuses: Status) -> int:
        return sum(r.status in statuses for r in self.results)

    @property
    def python_blocks(self) -> int:
        return len(self.results) - self.count("skipped")

    @property
    def compile_rate(self) -> float | None:
        n = self.python_blocks
        return (n - self.count("syntax_error")) / n if n else None

    @property
    def run_rate(self) -> float | None:
        """Share of judgeable blocks that ran; None if nothing was executed."""
        if not self.executed:
            return None
        judgeable = self.python_blocks - self.count("needs_input", "missing_module")
        return self.count("ok") / judgeable if judgeable else None

    @property
    def output_match_rate(self) -> float | None:
        """Of the blocks that ran and show an output, the share that printed exactly it —
        catches transcription slips in code that still runs."""
        checked = [r.output_matches for r in self.results if r.output_matches is not None]
        return sum(checked) / len(checked) if checked else None


def is_python(block: CodeBlock) -> bool:
    return (block.language or "").strip().lower() in PYTHON


def syntax_error(source: str) -> str | None:
    try:
        compile(source, "<lecture>", "exec")
    except SyntaxError as exc:
        return f"SyntaxError: {exc.msg} (line {exc.lineno})"
    except ValueError as exc:  # e.g. null bytes
        return f"ValueError: {exc}"
    return None


def classify(outcome: RunOutcome) -> tuple[Status, str | None]:
    if outcome.timed_out:
        return "timeout", None
    if outcome.returncode == 0:
        return "ok", None
    found = _ERROR_LINE.findall(outcome.stderr)
    error = found[-1] if found else f"exit code {outcome.returncode}"
    last = outcome.stderr.strip().splitlines()[-1] if outcome.stderr.strip() else error
    if error == "EOFError":
        return "needs_input", last
    if error == "ModuleNotFoundError":
        return "missing_module", last
    return "runtime_error", last


def check_code(
    units: list[KnowledgeUnit], runner: Runner | None = None, *, timeout_s: float = 10.0
) -> CodeScore:
    """Check every code block of the notes, in lecture order."""
    results: list[BlockResult] = []
    earlier: list[str] = []  # compiling Python blocks seen so far (context for later ones)
    for unit in sorted(units, key=lambda u: (u.start_s, u.id)):
        for i, block in enumerate(unit.code):
            if not is_python(block):
                results.append(BlockResult(unit.id, i, "skipped"))
                continue
            bad = syntax_error(block.source)
            if bad:
                results.append(BlockResult(unit.id, i, "syntax_error", bad))
                continue
            if runner is None:
                results.append(BlockResult(unit.id, i, "ok"))  # compiles; not executed
            else:
                results.append(_run(unit.id, i, block, earlier, runner, timeout_s))
            earlier.append(block.source)
    return CodeScore(results=results, executed=runner is not None)


def _lines(text: str) -> list[str]:
    return [" ".join(line.split()) for line in text.strip().splitlines() if line.strip()]


def output_matches(expected: str | None, stdout: str) -> bool | None:
    """None when the notes show no output. With earlier blocks run first (context), their
    prints come first — so the block's own output must be the END of what was printed."""
    if not expected or not expected.strip():
        return None
    want, got = _lines(expected), _lines(stdout)
    return got[-len(want) :] == want


def _run(
    unit_id: str,
    index: int,
    block: CodeBlock,
    earlier: list[str],
    runner: Runner,
    timeout_s: float,
) -> BlockResult:
    outcome = runner.run(block.source, timeout_s)
    status, error = classify(outcome)
    if status == "ok":
        return BlockResult(
            unit_id, index, "ok", output_matches=output_matches(block.output, outcome.stdout)
        )
    if status == "runtime_error" and error and error.startswith("NameError") and earlier:
        context = runner.run("\n\n".join([*earlier, block.source]), timeout_s)
        status2, error2 = classify(context)
        if status2 == "ok":
            return BlockResult(
                unit_id,
                index,
                "ok",
                in_context=True,
                output_matches=output_matches(block.output, context.stdout),
            )
        if status2 in ("needs_input", "missing_module"):
            return BlockResult(unit_id, index, status2, error2, in_context=True)
    return BlockResult(unit_id, index, status, error)


class DockerRunner:
    """Each block in a fresh, locked-down container (CI)."""

    name = "docker"

    def __init__(self, image: str = "python:3.12-slim") -> None:
        self.image = image

    def command(self, container: str) -> list[str]:
        return [
            "docker", "run", "--rm", "-i", "--name", container,
            "--network", "none", "--read-only", "--tmpfs", "/tmp:size=16m",
            "--memory", "256m", "--cpus", "1", "--pids-limit", "64",
            "--user", "65534:65534", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            self.image, "python", "-I", "-c",
            "import sys; exec(compile(sys.stdin.read(), '<lecture>', 'exec'))",
        ]  # fmt: skip

    def run(self, source: str, timeout_s: float) -> RunOutcome:
        container = f"code-check-{uuid.uuid4().hex[:12]}"
        # The snippet arrives on stdin and is exec'd; `input()` then sees EOF (needs_input).
        try:
            done = subprocess.run(
                self.command(container),
                input=source,
                capture_output=True,
                text=True,
                timeout=timeout_s + 20,  # + container start
            )
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", container], capture_output=True)
            return RunOutcome(-1, "", "", timed_out=True)
        return RunOutcome(done.returncode, done.stdout, done.stderr)


class LocalRunner:
    """A child interpreter in isolated mode (-I: no user site, no env vars, no cwd on the
    path), with an empty environment (no secrets), in a throwaway directory, killed after the
    timeout; on POSIX also CPU, memory and file-size limits. Not a security boundary — use it
    only for code you have read."""

    name = "local"

    def run(self, source: str, timeout_s: float) -> RunOutcome:
        env = {"PYTHONIOENCODING": "utf-8"}
        if sys.platform == "win32":
            env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
        # The base interpreter, not the venv's: on Windows the venv python.exe is a launcher
        # that starts the real interpreter as a child, and killing only the launcher left a
        # `while True` snippet spinning (and holding the pipes, so the check hung). -I needs
        # no venv packages anyway.
        python = getattr(sys, "_base_executable", None) or sys.executable
        with tempfile.TemporaryDirectory(prefix="code-check-") as cwd:
            proc = subprocess.Popen(
                [python, "-I", "-c", source],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=cwd,
                env=env,
                # own process group/session, so a timeout can kill everything it started
                start_new_session=sys.platform != "win32",
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
                preexec_fn=_limits if sys.platform != "win32" else None,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                _kill_tree(proc)
                proc.communicate()
                return RunOutcome(-1, "", "", timed_out=True)
        return RunOutcome(proc.returncode, stdout, stderr)


def _kill_tree(proc: subprocess.Popen[str]) -> None:
    """Kill the snippet and anything it started."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        import signal

        os.killpg(proc.pid, signal.SIGKILL)


def _limits() -> None:  # pragma: no cover - POSIX only, runs in the child
    if sys.platform != "win32":
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
        resource.setrlimit(resource.RLIMIT_AS, (512 * 2**20, 512 * 2**20))
        resource.setrlimit(resource.RLIMIT_FSIZE, (2**20, 2**20))
