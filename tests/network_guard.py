"""Block real network access in tests (plan rule §4.9: tests never call real APIs).

Loopback stays allowed: on Windows the asyncio event loop talks to itself over a
127.0.0.1 socketpair, so a blanket "no sockets" ban (pytest-socket style) breaks every
async test. Everything else fails fast at DNS resolution or connect().
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0", ""})


class NetworkBlocked(RuntimeError):
    pass


def _host_of(address: Any) -> str:
    host = address[0] if isinstance(address, tuple) else address
    return host.decode() if isinstance(host, bytes) else str(host)


def _check(host: str) -> None:
    if host not in LOOPBACK:
        raise NetworkBlocked(
            f"test tried to reach {host!r}; mark it @pytest.mark.network if intended"
        )


@contextmanager
def block_network() -> Iterator[None]:
    real_getaddrinfo = socket.getaddrinfo
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if host is not None:
            _check(_host_of(host))
        return real_getaddrinfo(host, *args, **kwargs)

    def connect(self: socket.socket, address: Any) -> None:
        if self.family in (socket.AF_INET, socket.AF_INET6):
            _check(_host_of(address))
        return real_connect(self, address)

    def connect_ex(self: socket.socket, address: Any) -> int:
        if self.family in (socket.AF_INET, socket.AF_INET6):
            _check(_host_of(address))
        return real_connect_ex(self, address)

    socket.getaddrinfo = getaddrinfo
    socket.socket.connect = connect  # type: ignore[method-assign]
    socket.socket.connect_ex = connect_ex  # type: ignore[method-assign]
    try:
        yield
    finally:
        socket.getaddrinfo = real_getaddrinfo
        socket.socket.connect = real_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = real_connect_ex  # type: ignore[method-assign]
