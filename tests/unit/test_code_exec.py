"""Code check. Only snippets written here are executed (LocalRunner); lecture code is not."""

import pytest

from evals.metrics.code_exec import (
    CodeScore,
    DockerRunner,
    LocalRunner,
    RunOutcome,
    check_code,
    classify,
    syntax_error,
)
from lecnotes.domain import CodeBlock, Evidence, KnowledgeUnit, VideoSpan

VID = "Ek49cGiAOwo"


def unit(id: str, start: float, *sources: str, language: str | None = "python") -> KnowledgeUnit:
    return KnowledgeUnit(
        id=id,
        concept=id,
        explanation="x",
        code=[CodeBlock(source=s, language=language) for s in sources],
        spans=[VideoSpan(video_id=VID, start_s=start, end_s=start + 10)],
        evidence=[Evidence(video_id=VID, segment_ids=["s1"])],
    )


def test_syntax_errors_are_found_without_running_anything() -> None:
    assert syntax_error("for i in range(3):\n    print(i)") is None
    assert "SyntaxError" in (syntax_error("for i in range(3)\n    print(i)") or "")


def test_compile_only_needs_no_runner() -> None:
    score = check_code(
        [
            unit("u1", 0, "names = ['a']\nprint(names)"),
            unit("u2", 10, "for name in names\n    print(name)"),  # missing colon
            unit("u3", 20, "SELECT 1;", language="sql"),
        ]
    )
    assert [r.status for r in score.results] == ["ok", "syntax_error", "skipped"]
    assert (score.python_blocks, score.compile_rate, score.run_rate) == (2, 0.5, None)


@pytest.mark.parametrize(
    ("stderr", "status"),
    [
        ("Traceback...\nNameError: name 'names' is not defined", "runtime_error"),
        ("Traceback...\nEOFError: EOF when reading a line", "needs_input"),
        ("Traceback...\nModuleNotFoundError: No module named 'numpy'", "missing_module"),
    ],
)
def test_classify(stderr: str, status: str) -> None:
    assert classify(RunOutcome(1, "", stderr))[0] == status
    assert classify(RunOutcome(0, "ok", ""))[0] == "ok"
    assert classify(RunOutcome(-1, "", "", timed_out=True))[0] == "timeout"


def test_local_runner_runs_with_earlier_blocks_as_context() -> None:
    score = check_code(
        [
            unit("u1", 0, "names = ['ada', 'alan']"),
            unit("u2", 10, "for i, n in enumerate(names):\n    print(i, n)"),  # uses u1
            unit("u3", 20, "print(1 / 0)"),
            unit("u4", 30, "x = input('name? ')"),
            unit("u5", 40, "import numpy_not_installed_here"),
        ],
        LocalRunner(),
    )
    by_unit = {r.unit_id: r for r in score.results}
    assert by_unit["u1"].status == "ok"
    assert (by_unit["u2"].status, by_unit["u2"].in_context) == ("ok", True)
    assert by_unit["u3"].status == "runtime_error"
    assert "ZeroDivisionError" in (by_unit["u3"].error or "")
    assert by_unit["u4"].status == "needs_input"
    assert by_unit["u5"].status == "missing_module"
    # judgeable = 5 - needs_input - missing_module = 3; ok = 2
    assert score.run_rate == pytest.approx(2 / 3)


def test_output_is_compared_with_what_the_code_prints() -> None:
    def block_unit(id: str, start: float, source: str, output: str) -> KnowledgeUnit:
        return unit(id, start).model_copy(
            update={"code": [CodeBlock(source=source, language="python", output=output)]}
        )

    score = check_code(
        [
            block_unit("u1", 0, "names = ['tom', 'jerry']\nprint(len(names))", "2"),
            # runs with u1 as context: u1 prints "2" first, so only the end has to match
            block_unit("u2", 10, "for n in names:\n    print(n)", "tom\njerry"),
            block_unit("u3", 20, "print(sum([1, 2]))", "4"),  # a transcription slip
        ],
        LocalRunner(),
    )
    assert [r.output_matches for r in score.results] == [True, True, False]
    assert score.output_match_rate == pytest.approx(2 / 3)


def test_local_runner_times_out_and_sees_no_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "must-not-leak")
    runner = LocalRunner()
    assert runner.run("while True: pass", timeout_s=1).timed_out
    leaked = runner.run("import os; print(os.environ.get('GEMINI_API_KEY'))", timeout_s=10)
    assert leaked.stdout.strip() == "None"


def test_docker_runner_is_locked_down() -> None:
    cmd = DockerRunner().command("c1")
    for flag in ("--network", "--read-only", "--memory", "--pids-limit", "--cap-drop"):
        assert flag in cmd
    assert cmd[cmd.index("--network") + 1] == "none"
    assert cmd[cmd.index("--user") + 1] == "65534:65534"


def test_empty_notes() -> None:
    assert CodeScore().compile_rate is None
