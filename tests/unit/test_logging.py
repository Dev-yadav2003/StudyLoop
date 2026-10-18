import asyncio
import io
import json
import logging
from collections.abc import Iterator

import pytest
import structlog

from lecnotes.observability.logging import bind_context, configure_logging, get_logger


@pytest.fixture
def stream() -> Iterator[io.StringIO]:
    buf = io.StringIO()
    configure_logging("DEBUG", stream=buf)
    yield buf
    structlog.contextvars.clear_contextvars()
    structlog.reset_defaults()
    logging.getLogger().handlers.clear()


def lines(buf: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in buf.getvalue().splitlines()]


def test_line_is_json_with_bound_context(stream: io.StringIO) -> None:
    with bind_context(job_id="job-1", stage="perceive"):
        get_logger("test").info("stage.started", segments=3)

    [record] = lines(stream)
    assert record["event"] == "stage.started"
    assert record["job_id"] == "job-1"
    assert record["stage"] == "perceive"
    assert record["segments"] == 3
    assert record["level"] == "info"
    assert "timestamp" in record


def test_context_is_removed_after_block(stream: io.StringIO) -> None:
    with bind_context(job_id="job-1"):
        pass
    get_logger().info("after")

    [record] = lines(stream)
    assert "job_id" not in record


def test_nested_context_restores_outer_value(stream: io.StringIO) -> None:
    with bind_context(job_id="job-1", stage="resolve"):
        with bind_context(stage="perceive"):
            get_logger().info("inner")
        get_logger().info("outer")

    inner, outer = lines(stream)
    assert inner["stage"] == "perceive"
    assert outer["stage"] == "resolve"
    assert outer["job_id"] == "job-1"


def test_stdlib_loggers_get_context_too(stream: io.StringIO) -> None:
    with bind_context(job_id="job-2"):
        logging.getLogger("httpx").warning("third-party line")

    [record] = lines(stream)
    assert record["event"] == "third-party line"
    assert record["job_id"] == "job-2"
    assert record["logger"] == "httpx"


async def test_context_isolated_between_tasks(stream: io.StringIO) -> None:
    async def run(job_id: str) -> None:
        with bind_context(job_id=job_id):
            await asyncio.sleep(0)
            get_logger().info("tick")

    await asyncio.gather(run("a"), run("b"))

    # Filter: asyncio itself logs at DEBUG (e.g. "Using proactor" on Windows).
    ticks = [r for r in lines(stream) if r["event"] == "tick"]
    assert sorted(str(r["job_id"]) for r in ticks) == ["a", "b"]


def test_exceptions_are_structured(stream: io.StringIO) -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        get_logger().exception("failed")

    [record] = lines(stream)
    assert record["exception"][0]["exc_type"] == "ValueError"  # type: ignore[index]
