import uuid
from collections.abc import Iterator

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lecnotes.observability import tracing
from tests.support import SpanRecorder


@pytest.fixture
def span_recorder() -> Iterator[SpanRecorder]:
    exporter = InMemorySpanExporter()
    tracing.init_tracing(
        # Langfuse keeps one resource manager per public key; unique key = isolated test.
        public_key=f"pk-test-{uuid.uuid4()}",
        secret_key="sk-test",
        base_url="http://localhost:9",
        environment="test",
        tracer_provider=TracerProvider(),
        span_exporter=exporter,
    )
    yield SpanRecorder(exporter)
    tracing.shutdown_tracing()
