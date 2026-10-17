"""Structured JSON logging.

Every log line is a JSON object. Context such as `job_id` and `stage` is bound once with
`bind_context(...)` and then appears on every line emitted inside that block, including
lines from third-party libraries that use stdlib `logging` (uvicorn, httpx, sqlalchemy).

Context is stored in contextvars, so it is safe across `await` and per asyncio task.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import IO, Any

import structlog
from structlog.typing import Processor

# Fields every pipeline log line should carry (plan §4.7). Not enforced, but documented here
# so call sites use consistent names.
CONTEXT_FIELDS = ("job_id", "video_id", "stage", "stage_version", "trace_id")

_SHARED_PROCESSORS: list[Processor] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_logger_name,
    structlog.stdlib.add_log_level,
    structlog.processors.TimeStamper(fmt="iso", utc=True),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.dict_tracebacks,
]


def configure_logging(
    level: str = "INFO",
    *,
    json: bool = True,
    stream: IO[str] | None = None,
) -> None:
    """Configure structlog and the stdlib root logger. Safe to call more than once."""
    renderer: Processor = (
        structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
    )

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=_SHARED_PROCESSORS,
            processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
        )
    )

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *_SHARED_PROCESSORS,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.stdlib.get_logger(name)


@contextmanager
def bind_context(**fields: Any) -> Iterator[None]:
    """Attach fields to every log line emitted inside the block; restores previous values after."""
    with structlog.contextvars.bound_contextvars(**fields):
        yield
