from collections.abc import Awaitable, Callable

import pytest

from lecnotes.adapters.cache.memory import InMemoryStageCache
from lecnotes.ports.cache import StageCache
from tests.cache_contract import CONTRACT, make_key


@pytest.mark.parametrize("check", CONTRACT, ids=lambda f: f.__name__)
async def test_contract(check: Callable[[StageCache], Awaitable[None]]) -> None:
    await check(InMemoryStageCache())


async def test_hits_are_counted() -> None:
    cache = InMemoryStageCache()
    key = make_key()
    await cache.get(key)  # a miss is not a hit
    await cache.put(key, {"v": 1})
    await cache.get(key)
    await cache.get(key)
    assert cache.hits[key.digest] == 2
