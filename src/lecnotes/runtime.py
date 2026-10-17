"""Process entrypoint helpers."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable, Coroutine
from typing import Any


def loop_factory() -> Callable[[], asyncio.AbstractEventLoop] | None:
    """psycopg 3's async mode does not support Windows' default ProactorEventLoop."""
    return asyncio.SelectorEventLoop if sys.platform == "win32" else None


def run[T](main: Coroutine[Any, Any, T]) -> T:
    """`asyncio.run` with an event loop that works for our DB driver on every OS."""
    return asyncio.run(main, loop_factory=loop_factory())
