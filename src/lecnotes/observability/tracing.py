"""Langfuse tracing.

One trace per job, one span per pipeline stage. LLM generations created by the LangChain
callback handler (Phase 2) nest under whichever stage span is current, because Langfuse v5
is OpenTelemetry-based and tracks the active span in context.

    init_tracing(settings)
    with traced_job("job-123", video_id="abc"):
        with traced_stage("perceive", version=1, prompt_version=1) as span:
            ...
            span.update(output={"segments": 42})

Each context manager also binds the matching logging fields, so log lines and spans can be
correlated by `trace_id`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Protocol

from langchain_core.callbacks import BaseCallbackHandler
from langfuse import Langfuse, propagate_attributes
from langfuse.langchain import CallbackHandler

from lecnotes import __version__
from lecnotes.config import Settings
from lecnotes.observability.logging import bind_context

TRACE_NAME = "lecture-notes"

_client: Langfuse | None = None
_public_key: str | None = None


class TracingNotInitialized(RuntimeError):
    pass


class Span(Protocol):
    """What callers may use from a span; satisfied by Langfuse spans and `NoopSpan`."""

    @property
    def trace_id(self) -> str | None: ...

    def update(self, **kwargs: Any) -> Any: ...


class NoopSpan:
    """Used when tracing is off (unit tests, local CLI without Langfuse keys). Production
    initialises tracing at startup, where missing keys already fail fast."""

    trace_id: str | None = None

    def update(self, **kwargs: Any) -> NoopSpan:
        return self


def tracing_enabled() -> bool:
    return _client is not None


def init_tracing(
    *,
    public_key: str,
    secret_key: str,
    base_url: str,
    environment: str = "dev",
    **client_kwargs: Any,
) -> Langfuse:
    """Create the process-wide Langfuse client.

    `client_kwargs` go straight to `Langfuse(...)`; tests use them to inject an in-memory
    OpenTelemetry `tracer_provider` + `span_exporter`.
    """
    global _client, _public_key
    _client = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=base_url,
        environment=environment,
        release=__version__,
        **client_kwargs,
    )
    _public_key = public_key
    return _client


def init_tracing_from_settings(settings: Settings) -> Langfuse:
    return init_tracing(
        public_key=settings.langfuse_public_key.get_secret_value(),
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        base_url=settings.langfuse_base_url,
        environment=settings.env,
    )


def get_langfuse() -> Langfuse:
    if _client is None:
        raise TracingNotInitialized("call init_tracing() at startup before tracing")
    return _client


def langchain_callbacks() -> list[BaseCallbackHandler]:
    """Callbacks that trace LangChain runs as Langfuse generations under the current span.

    Empty when tracing is not initialised, so the LLM gateway works in scripts and tests
    without a Langfuse project.
    """
    if _client is None:
        return []
    return [CallbackHandler(public_key=_public_key)]


def shutdown_tracing() -> None:
    """Flush pending spans and release the client. Call on app shutdown / end of script."""
    global _client, _public_key
    if _client is not None:
        _client.shutdown()
    _client = None
    _public_key = None


@contextmanager
def traced_job(job_id: str, *, video_id: str | None = None) -> Iterator[Span]:
    """Root span for one job. All stage spans opened inside become its children."""
    attrs = {"job_id": job_id} | ({"video_id": video_id} if video_id else {})
    if _client is None:
        with bind_context(**attrs):
            yield NoopSpan()
        return
    client = _client
    with (
        client.start_as_current_observation(as_type="span", name="job", metadata=attrs) as span,
        propagate_attributes(trace_name=TRACE_NAME, session_id=job_id, metadata=attrs),
        bind_context(trace_id=span.trace_id, **attrs),
    ):
        yield span


@contextmanager
def traced_stage(
    name: str,
    version: int,
    *,
    prompt_version: int | None = None,
    input: Any = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[Span]:
    """Span for one pipeline stage. Records errors on the span, then re-raises."""
    if _client is None:
        with bind_context(stage=name, stage_version=version):
            yield NoopSpan()
        return
    client = _client
    stage_meta: dict[str, Any] = {"stage": name, "stage_version": version}
    if prompt_version is not None:
        stage_meta["prompt_version"] = prompt_version
    with (
        client.start_as_current_observation(
            as_type="span",
            name=f"stage:{name}",
            version=f"v{version}",
            input=input,
            metadata=stage_meta | (metadata or {}),
        ) as span,
        bind_context(stage=name, stage_version=version, trace_id=span.trace_id),
    ):
        try:
            yield span
        except Exception as exc:
            span.update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}")
            raise
