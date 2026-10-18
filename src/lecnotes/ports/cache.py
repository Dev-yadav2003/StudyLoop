from typing import Any, Protocol

from lecnotes.cache.keys import StageKey


class StageCache(Protocol):
    """Whole-stage output cache. Payloads are JSON objects (a Pydantic `model_dump`)."""

    async def get(self, key: StageKey) -> dict[str, Any] | None:
        """Payload on hit (and count the hit), None on miss."""
        ...

    async def put(self, key: StageKey, payload: dict[str, Any]) -> None:
        """Store, replacing any previous payload (a forced re-run recomputes the same key)."""
        ...
