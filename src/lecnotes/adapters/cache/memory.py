from __future__ import annotations

import copy
from typing import Any

from lecnotes.cache.keys import StageKey


class InMemoryStageCache:
    """Process-local cache for tests and the dev CLI. Same contract as the Postgres cache."""

    def __init__(self) -> None:
        self._items: dict[str, dict[str, Any]] = {}
        self.hits: dict[str, int] = {}

    async def get(self, key: StageKey) -> dict[str, Any] | None:
        payload = self._items.get(key.digest)
        if payload is None:
            return None
        self.hits[key.digest] = self.hits.get(key.digest, 0) + 1
        return copy.deepcopy(payload)  # callers must not mutate the cached object

    async def put(self, key: StageKey, payload: dict[str, Any]) -> None:
        self._items[key.digest] = copy.deepcopy(payload)

    def __len__(self) -> int:
        return len(self._items)
