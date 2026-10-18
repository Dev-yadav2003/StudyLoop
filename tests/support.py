"""Test helpers shared across suites (not fixtures: those live in conftest.py)."""

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lecnotes.observability import tracing


class SpanRecorder:
    """In-memory Langfuse tracing: real span tree, no network."""

    def __init__(self, exporter: InMemorySpanExporter) -> None:
        self._exporter = exporter

    def spans(self) -> list[ReadableSpan]:
        tracing.get_langfuse().flush()
        return list(self._exporter.get_finished_spans())

    def by_name(self) -> dict[str, ReadableSpan]:
        return {span.name: span for span in self.spans()}
