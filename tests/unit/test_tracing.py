"""Tracing tests run fully offline: spans go to an in-memory OTel exporter, not Langfuse."""

import io
import json
import logging
import uuid
from collections.abc import Iterator

import pytest
import structlog
from langfuse import Langfuse
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lecnotes.observability import tracing
from lecnotes.observability.logging import configure_logging, get_logger
from lecnotes.observability.tracing import (
    TracingNotInitialized,
    init_tracing,
    traced_job,
    traced_stage,
)


@pytest.fixture
def exporter() -> Iterator[InMemorySpanExporter]:
    exp = InMemorySpanExporter()
    client = Langfuse(
        # Langfuse keeps one resource manager per public key; unique key = isolated test.
        public_key=f"pk-test-{uuid.uuid4()}",
        secret_key="sk-test",
        base_url="http://localhost:9",
        tracer_provider=TracerProvider(),
        span_exporter=exp,
    )
    init_tracing(client=client)
    yield exp
    tracing.shutdown_tracing()


def finished(exp: InMemorySpanExporter) -> dict[str, ReadableSpan]:
    tracing.get_langfuse().flush()
    return {span.name: span for span in exp.get_finished_spans()}


def test_stage_spans_nest_under_job(exporter: InMemorySpanExporter) -> None:
    with traced_job("job-1", video_id="vid-1"):
        with traced_stage("resolve", version=1):
            pass
        with traced_stage("perceive", version=2, prompt_version=3):
            pass

    spans = finished(exporter)
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


def test_stage_error_is_recorded_and_reraised(exporter: InMemorySpanExporter) -> None:
    with (
        pytest.raises(ValueError, match="bad"),
        traced_job("job-2"),
        traced_stage("extract", version=1),
    ):
        raise ValueError("bad")

    attrs = finished(exporter)["stage:extract"].attributes or {}
    assert attrs["langfuse.observation.level"] == "ERROR"
    assert attrs["langfuse.observation.status_message"] == "ValueError: bad"


@pytest.mark.usefixtures("exporter")
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


def test_using_tracing_before_init_fails_clearly() -> None:
    tracing.shutdown_tracing()
    with pytest.raises(TracingNotInitialized), traced_stage("resolve", version=1):
        pass
