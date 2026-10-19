"""Quota-aware routing: least-used key first, spent keys skipped before calling, limits
learned from 429s, overloaded models skipped, and a retry time at the daily reset."""

from datetime import UTC, date, datetime

import pytest
from langchain_core.exceptions import ModelAPIError, ModelRateLimitError
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from lecnotes.adapters.llm.fake import FakeLLM
from lecnotes.errors import QuotaExceeded
from lecnotes.llm.gateway import LLMGateway
from lecnotes.llm.keys import OVERLOAD_BENCH_S, KeyPool, ModelBreaker, is_overloaded
from lecnotes.llm.quota import (
    RESET_MARGIN_S,
    MemoryUsageStore,
    daily_limit_from,
    key_id,
    quota_day,
    seconds_until_reset,
)
from lecnotes.prompts.loader import PromptSpec, RenderedPrompt

FLASH, LITE = "gemini-2.5-flash", "gemini-2.5-flash-lite"
# 10:00 UTC on 2026-10-19 is 03:00 PDT: 21 h before the Pacific midnight reset.
NOW = datetime(2026, 10, 19, 10, 0, tzinfo=UTC)
TODAY = date(2026, 10, 19)
DAILY_20 = ModelRateLimitError(
    "429 RESOURCE_EXHAUSTED {'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', "
    "'quotaValue': '20'}"
)
OVERLOADED = ModelAPIError("503 UNAVAILABLE. This model is currently experiencing high demand.")


class Answer(BaseModel):
    text: str


OK = Answer(text="ok")


def prompt() -> RenderedPrompt:
    spec = PromptSpec(
        id="qa", version=1, model=None, temperature=0.0, response_schema="Answer",
        changelog="t", system="s", user="u",
    )  # fmt: skip
    return RenderedPrompt(ref=spec.ref, spec=spec, messages=[SystemMessage("s"), HumanMessage("u")])


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def gateway(
    fake: FakeLLM,
    *,
    keys: int = 3,
    store: MemoryUsageStore | None = None,
    breaker: ModelBreaker | None = None,
) -> LLMGateway:
    return LLMGateway(
        fake.factory,
        default_model=FLASH,
        fallback_models=[LITE],
        retry_backoff=False,
        key_pool=KeyPool(keys),
        key_ids=[f"k{i}" for i in range(keys)],
        usage_store=store or MemoryUsageStore(),
        breaker=breaker,
        now=lambda: NOW,
    )


# --- helpers ---------------------------------------------------------------------------


def test_quota_day_and_reset_follow_pacific_time() -> None:
    assert quota_day(NOW) == TODAY
    # 06:30 UTC on the 20th is still the 19th in California (23:30 PDT).
    assert quota_day(datetime(2026, 10, 20, 6, 30, tzinfo=UTC)) == TODAY
    assert seconds_until_reset(NOW) == 21 * 3600


def test_daily_limit_is_read_from_the_429() -> None:
    assert daily_limit_from(str(DAILY_20)) == 20
    assert daily_limit_from("429 RESOURCE_EXHAUSTED GenerateRequestsPerMinute") is None


def test_key_id_is_stable_and_does_not_reveal_the_key() -> None:
    assert key_id("AIza-secret") == key_id("AIza-secret")
    assert "secret" not in key_id("AIza-secret")
    assert len(key_id("AIza-secret")) == 12


def test_overload_is_recognised() -> None:
    assert is_overloaded(OVERLOADED)
    assert not is_overloaded(ModelAPIError("500 INTERNAL"))


# --- routing ---------------------------------------------------------------------------


async def test_least_used_key_goes_first() -> None:
    fake = FakeLLM({"qa": OK})
    gw = gateway(fake)
    for _ in range(4):
        await gw.generate_structured(prompt(), Answer)
    assert [c.key for c in fake.calls] == [0, 1, 2, 0]


async def test_spent_key_is_skipped_without_a_request() -> None:
    store = MemoryUsageStore()
    await store.mark_exhausted(TODAY, "k0", FLASH, limit=None)
    fake = FakeLLM({"qa": OK})
    await gateway(fake, store=store).generate_structured(prompt(), Answer)
    assert [c.key for c in fake.calls] == [1]


async def test_a_daily_429_teaches_the_limit_and_is_remembered_after_a_restart() -> None:
    store = MemoryUsageStore()
    fake = FakeLLM({"qa": [DAILY_20, OK, OK]})
    await gateway(fake, store=store).generate_structured(prompt(), Answer)
    assert await store.daily_limit(FLASH) == 20

    # A new gateway (e.g. after a redeploy) shares the store: key 0 is not tried again today.
    await gateway(fake, store=store).generate_structured(prompt(), Answer)
    assert [c.key for c in fake.calls] == [0, 1, 2]


async def test_key_at_its_learned_limit_is_skipped_before_it_fails() -> None:
    store = MemoryUsageStore()
    await store.mark_exhausted(TODAY, "k9", FLASH, limit=2)  # another key taught: 2/day
    for _ in range(2):
        await store.record(TODAY, "k0", FLASH, tokens_in=1, tokens_out=1)
    fake = FakeLLM({"qa": OK})
    await gateway(fake, store=store).generate_structured(prompt(), Answer)
    assert fake.calls[0].key != 0  # k0 used its 2 requests: never called


async def test_all_keys_spent_on_top_model_then_next_model_on_every_key() -> None:
    store = MemoryUsageStore()
    for k in ("k0", "k1", "k2"):
        await store.mark_exhausted(TODAY, k, FLASH, limit=20)
    fake = FakeLLM({"qa": OK})
    result = await gateway(fake, store=store).generate_structured(prompt(), Answer)
    assert [(c.model, c.key) for c in fake.calls] == [(LITE, 0)]
    assert result.usage.model == LITE


async def test_everything_spent_waits_for_the_daily_reset() -> None:
    store = MemoryUsageStore()
    for model in (FLASH, LITE):
        for k in ("k0", "k1"):
            await store.mark_exhausted(TODAY, k, model, limit=20)
    fake = FakeLLM({"qa": OK})
    with pytest.raises(QuotaExceeded) as exc_info:
        await gateway(fake, keys=2, store=store).generate_structured(prompt(), Answer)
    assert fake.calls == []  # not one wasted request
    assert exc_info.value.retry_after_s == 21 * 3600 + RESET_MARGIN_S


async def test_per_minute_limits_everywhere_retry_in_minutes() -> None:
    fake = FakeLLM({"qa": ModelRateLimitError("429 RESOURCE_EXHAUSTED PerMinute")})
    with pytest.raises(QuotaExceeded) as exc_info:
        await gateway(fake, keys=1).generate_structured(prompt(), Answer)
    assert exc_info.value.retry_after_s is not None
    assert exc_info.value.retry_after_s < 600


async def test_overloaded_model_gets_one_retry_then_the_next_model() -> None:
    fake = FakeLLM({"qa": [OVERLOADED, OVERLOADED, OK]})
    result = await gateway(fake).generate_structured(prompt(), Answer)
    assert [(c.model, c.key) for c in fake.calls] == [(FLASH, 0), (FLASH, 0), (LITE, 0)]
    assert result.usage.model == LITE


async def test_overloaded_model_is_skipped_by_later_calls_then_probed_again() -> None:
    clock = Clock()
    breaker = ModelBreaker(clock=clock)
    fake = FakeLLM({"qa": [OVERLOADED, OVERLOADED, OK, OK, OK]})
    gw = gateway(fake, breaker=breaker)
    await gw.generate_structured(prompt(), Answer)  # Flash overloaded -> Lite
    await gw.generate_structured(prompt(), Answer)  # Flash still benched: straight to Lite
    clock.now += OVERLOAD_BENCH_S + 1
    await gw.generate_structured(prompt(), Answer)  # bench over: Flash probed again
    assert [c.model for c in fake.calls] == [FLASH, FLASH, LITE, LITE, FLASH]


async def test_requests_are_counted_per_key_and_model() -> None:
    store = MemoryUsageStore()
    fake = FakeLLM({"qa": OK})
    gw = gateway(fake, store=store)
    for _ in range(3):
        await gw.generate_structured(prompt(), Answer)
    usage = await store.usage(TODAY, FLASH)
    assert {k: u.requests for k, u in usage.items()} == {"k0": 1, "k1": 1, "k2": 1}
