"""Behaviour every StageCache must have. Run against the in-memory cache (unit) and the
Postgres cache (integration), so the test double can't drift from the real thing."""

from lecnotes.cache.keys import StageKey
from lecnotes.ports.cache import StageCache


def make_key(stage_version: int = 1, input: object = "x") -> StageKey:
    return StageKey.build(
        stage="extract", stage_version=stage_version, input=input, video_id="dQw4w9WgXcQ"
    )


PAYLOAD = {
    "units": [{"concept": "enumerate()", "code": "for i, x in enumerate(xs):\n    ..."}],
    "note": "unicode ✓ — ünïcödé",
    "n": 3,
    "ratio": 0.25,
    "nested": {"empty": [], "none": None},
}


async def miss_then_hit(cache: StageCache) -> None:
    key = make_key()
    assert await cache.get(key) is None
    await cache.put(key, PAYLOAD)
    assert await cache.get(key) == PAYLOAD


async def keys_are_isolated(cache: StageCache) -> None:
    await cache.put(make_key(1), {"v": 1})
    await cache.put(make_key(2), {"v": 2})
    assert await cache.get(make_key(1)) == {"v": 1}
    assert await cache.get(make_key(2)) == {"v": 2}
    assert await cache.get(make_key(3)) is None


async def put_overwrites(cache: StageCache) -> None:
    key = make_key()
    await cache.put(key, {"v": "old"})
    await cache.put(key, {"v": "new"})  # forced re-run of the same stage
    assert await cache.get(key) == {"v": "new"}


async def returned_payload_is_not_shared(cache: StageCache) -> None:
    key = make_key()
    await cache.put(key, {"items": [1]})
    first = await cache.get(key)
    assert first is not None
    first["items"].append(2)
    assert await cache.get(key) == {"items": [1]}


CONTRACT = [miss_then_hit, keys_are_isolated, put_overwrites, returned_payload_is_not_shared]
