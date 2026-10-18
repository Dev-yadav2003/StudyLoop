"""Several API keys: failover on a key's own problems (invalid, out of quota), same model
first; not on server errors. Offline (FakeLLM records which key each call used)."""

import pytest
from langchain_core.exceptions import (
    ModelAPIError,
    ModelInvalidRequestError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
)
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from lecnotes.adapters.llm.fake import FakeLLM
from lecnotes.config import load_settings
from lecnotes.errors import QuotaExceeded
from lecnotes.llm.gateway import LLMGateway
from lecnotes.llm.keys import DAILY_BENCH_S, PER_MINUTE_BENCH_S, KeyPool, key_problem
from lecnotes.prompts.loader import PromptSpec, RenderedPrompt

FLASH, LITE = "gemini-2.5-flash", "gemini-2.5-flash-lite"
DAILY = ModelRateLimitError("429 RESOURCE_EXHAUSTED quotaId GenerateRequestsPerDayPerProject")
PER_MINUTE = ModelRateLimitError("429 RESOURCE_EXHAUSTED GenerateRequestsPerMinute")
# what langchain-google-genai raises for an invalid key (HTTP 400)
INVALID = ModelInvalidRequestError("400 INVALID_ARGUMENT: API key not valid. API_KEY_INVALID")


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


def gateway(fake: FakeLLM, keys: int = 3, clock: Clock | None = None) -> LLMGateway:
    return LLMGateway(
        fake.factory,
        default_model=FLASH,
        fallback_models=[LITE],
        retry_backoff=False,
        key_pool=KeyPool(keys, clock=clock or Clock()),
    )


async def test_out_of_quota_key_fails_over_to_the_next_key_same_model() -> None:
    fake = FakeLLM({"qa": [DAILY, OK]})
    result = await gateway(fake).generate_structured(prompt(), Answer)
    assert [(c.model, c.key) for c in fake.calls] == [(FLASH, 0), (FLASH, 1)]  # no 429 retry
    assert result.usage.model == FLASH  # quality kept


async def test_benched_key_is_skipped_on_later_calls_until_the_bench_ends() -> None:
    clock = Clock()
    fake = FakeLLM({"qa": [PER_MINUTE, OK, OK, OK]})
    gw = gateway(fake, clock=clock)
    await gw.generate_structured(prompt(), Answer)  # key 0 → per-minute 429 → key 1
    await gw.generate_structured(prompt(), Answer)  # starts at key 1
    clock.now += PER_MINUTE_BENCH_S + 1
    await gw.generate_structured(prompt(), Answer)  # key 0 is back
    assert [c.key for c in fake.calls] == [0, 1, 1, 0]


async def test_invalid_key_is_benched_for_every_model() -> None:
    fake = FakeLLM({"qa": [INVALID, OK, OK]})
    gw = gateway(fake, keys=2)
    await gw.generate_structured(prompt(), Answer)
    await gw.generate_structured(prompt(), Answer)
    assert [c.key for c in fake.calls] == [0, 1, 1]


async def test_all_keys_out_of_quota_for_the_model_falls_back_to_the_next_model() -> None:
    # quotas are per key AND per model: keys out of Flash quota still serve Flash-Lite
    fake = FakeLLM({"qa": [DAILY, DAILY, OK]})
    result = await gateway(fake, keys=2).generate_structured(prompt(), Answer)
    assert [(c.model, c.key) for c in fake.calls] == [(FLASH, 0), (FLASH, 1), (LITE, 0)]
    assert result.usage.model == LITE


async def test_server_errors_are_retried_on_the_same_key_not_failed_over() -> None:
    fake = FakeLLM({"qa": [ModelAPIError("503 UNAVAILABLE"), OK]})
    await gateway(fake).generate_structured(prompt(), Answer)
    assert [c.key for c in fake.calls] == [0, 0]


async def test_every_key_and_model_exhausted_is_quota_exceeded() -> None:
    fake = FakeLLM({"qa": DAILY})
    with pytest.raises(QuotaExceeded) as exc:
        await gateway(fake, keys=2).generate_structured(prompt(), Answer)
    assert exc.value.retryable  # the job runner retries later
    assert len(fake.calls) == 4  # 2 keys x 2 models, no wasted retries on 429


def test_key_problem_classification() -> None:
    for broken in (INVALID, ModelPermissionDeniedError("403 PERMISSION_DENIED")):
        problem = key_problem(broken)
        assert problem is not None
        assert problem.all_models
    daily, minute = key_problem(DAILY), key_problem(PER_MINUTE)
    assert daily is not None
    assert minute is not None
    assert (daily.seconds, daily.all_models) == (DAILY_BENCH_S, False)
    assert minute.seconds == PER_MINUTE_BENCH_S
    assert key_problem(ModelAPIError("503 UNAVAILABLE")) is None


def test_settings_keys_and_models(monkeypatch: pytest.MonkeyPatch) -> None:
    from lecnotes.config import Settings

    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for name in ("YOUTUBE_API_KEY", "DATABASE_URL", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.setenv(name, "x")
    monkeypatch.setenv("GEMINI_API_KEY", "k1")
    monkeypatch.setenv("GEMINI_API_KEYS", "k2, k1 ,k3")
    monkeypatch.setenv(
        "GEMINI_MODELS", "gemini-2.5-flash,gemini-2.5-flash-lite,gemini-3.1-flash-lite"
    )
    s = load_settings()
    assert s.gemini_keys == ["k1", "k2", "k3"]  # GEMINI_API_KEY first, duplicates dropped
    assert (s.text_model, s.perceive_model) == (FLASH, FLASH)
    assert s.fallback_models == [LITE, "gemini-3.1-flash-lite"]
    assert "k2" not in repr(s)  # secrets never in logs/reprs


async def test_our_own_bugs_pass_through_unchanged_and_without_a_cause_cycle() -> None:
    """A non-provider error (a bug) must not be disguised as 'the AI model failed' — and must
    not be its own __cause__ (that cycle once made a test walk the chain forever)."""
    fake = FakeLLM({"qa": RuntimeError("bug in our code")})
    with pytest.raises(RuntimeError, match="bug in our code") as exc:
        await gateway(fake).generate_structured(prompt(), Answer)
    assert exc.value.__cause__ is not exc.value
