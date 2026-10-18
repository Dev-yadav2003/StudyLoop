"""Tracing tests run fully offline: spans go to an in-memory OTel exporter, not Langfuse."""

import io
import json
import logging

import pytest
import structlog

from lecnotes.observability import tracing
from lecnotes.observability.logging import configure_logging, get_logger
from lecnotes.observability.tracing import (
    TracingNotInitialized,
    traced_job,
    traced_stage,
)
from tests.support import SpanRecorder


def test_stage_spans_nest_under_job(span_recorder: SpanRecorder) -> None:
    with traced_job("job-1", video_id="vid-1"):
        with traced_stage("resolve", version=1):
            pass
        with traced_stage("perceive", version=2, prompt_version=3):
            pass

    spans = span_recorder.by_name()
    job, resolve, perceive = spans["job"], spans["stage:resolve"], spans["stage:perceive"]

    assert job.parent is None
    assert job.context is not None
    for child in (resolve, perceive):
        assert child.parent is not None
        assert child.parent.span_id == job.context.span_id
        assert child.context is not None
        assert child.context.trace_id == job.context.trace_id

    attrs = perceive.attributes or {}
    assert attrs["langfuse.version"] == "v2"
    assert attrs["langfuse.observation.metadata.stage"] == '"perceive"'
    assert attrs["langfuse.observation.metadata.prompt_version"] == "3"
    assert attrs["langfuse.trace.name"] == "lecture-notes"
    assert attrs["session.id"] == "job-1"


def test_stage_error_is_recorded_and_reraised(span_recorder: SpanRecorder) -> None:
    with (
        pytest.raises(ValueError, match="bad"),
        traced_job("job-2"),
        traced_stage("extract", version=1),
    ):
        raise ValueError("bad")

    attrs = span_recorder.by_name()["stage:extract"].attributes or {}
    assert attrs["langfuse.observation.level"] == "ERROR"
    assert attrs["langfuse.observation.status_message"] == "ValueError: bad"


@pytest.mark.usefixtures("span_recorder")
def test_log_lines_carry_trace_and_stage() -> None:
    buf = io.StringIO()
    configure_logging(stream=buf)
    try:
        with traced_job("job-3", video_id="vid-3"), traced_stage("validate", version=1) as span:
            get_logger().info("inside")
            trace_id = span.trace_id
    finally:
        structlog.reset_defaults()
        logging.getLogger().handlers.clear()

    record = json.loads(buf.getvalue().splitlines()[-1])
    assert record["trace_id"] == trace_id
    assert record["job_id"] == "job-3"
    assert record["video_id"] == "vid-3"
    assert record["stage"] == "validate"
    assert record["stage_version"] == 1


def test_tracing_off_yields_noop_spans() -> None:
    tracing.shutdown_tracing()
    assert not tracing.tracing_enabled()
    with traced_job("job-x") as job, traced_stage("resolve", version=1) as stage:
        stage.update(output={"ok": True})  # accepted, ignored
    assert job.trace_id is None
    assert stage.trace_id is None


def test_get_langfuse_before_init_fails_clearly() -> None:
    tracing.shutdown_tracing()
    with pytest.raises(TracingNotInitialized):
        tracing.get_langfuse()


class _ScoreRecorder:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[dict[str, object]] = []
        self.fail = fail

    def create_score(self, **kwargs: object) -> None:
        if self.fail:
            raise RuntimeError("langfuse down")
        self.calls.append(kwargs)


def test_score_goes_to_the_trace_or_falls_back_to_the_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = _ScoreRecorder()
    monkeypatch.setattr(tracing, "_client", rec)
    assert tracing.score("user_feedback", -1, trace_id="t1", session_id="job", comment="why")
    assert tracing.score("user_feedback", 1, session_id="job")
    assert [(c["trace_id"], c["session_id"], c["value"]) for c in rec.calls] == [
        ("t1", None, -1),
        (None, "job", 1),
    ]
    assert rec.calls[0]["comment"] == "why"
    assert not tracing.score("user_feedback", 1)  # nothing to attach it to


def test_score_is_best_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    assert not tracing.score("user_feedback", 1, trace_id="t1")  # tracing off
    monkeypatch.setattr(tracing, "_client", _ScoreRecorder(fail=True))
    assert not tracing.score("user_feedback", 1, trace_id="t1")  # Langfuse error swallowed
