import json
from decimal import Decimal

import httpx
import pytest
import respx
from google.genai.types import MediaResolution
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAPIError,
    ModelInvalidRequestError,
    ModelRateLimitError,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, SecretStr

from lecnotes.adapters.llm.fake import FakeLLM
from lecnotes.config import Settings
from lecnotes.errors import (
    LLMError,
    OutputTruncated,
    QuotaExceeded,
    StructuredOutputError,
    UnknownModel,
)
from lecnotes.llm.gateway import LLMGateway, ModelOptions, gemini_factory
from lecnotes.llm.pricing import cost_usd
from lecnotes.llm.rate_limit import rate_limiter, requests_per_minute
from lecnotes.observability.tracing import traced_stage
from lecnotes.prompts.loader import PromptSpec, RenderedPrompt
from tests.support import SpanRecorder

FLASH, LITE, PRO = "gemini-2.5-flash", "gemini-3.1-flash-lite", "gemini-3-flash-preview"


class Answer(BaseModel):
    text: str
    confidence: float


ANSWER = Answer(text="enumerate yields (index, item)", confidence=0.9)


def make_prompt(prompt_id: str = "qa", *, model: str | None = None) -> RenderedPrompt:
    spec = PromptSpec(
        id=prompt_id,
        version=2,
        model=model,
        temperature=0.0,
        response_schema="Answer",
        changelog="test",
        system="sys",
        user="usr",
    )
    return RenderedPrompt(
        ref=spec.ref, spec=spec, messages=[SystemMessage("sys"), HumanMessage("usr")]
    )


def make_gateway(fake: FakeLLM, *, fallback: str | None = LITE) -> LLMGateway:
    return LLMGateway(
        fake.factory,
        default_model=FLASH,
        fallback_model=fallback,
        retry_attempts=3,
        retry_backoff=False,  # no sleeping in unit tests
    )


async def test_returns_validated_output_with_usage() -> None:
    fake = FakeLLM({"qa": ANSWER}, tokens_in=10_000, tokens_out=2_000)

    result = await make_gateway(fake).generate_structured(make_prompt(), Answer)

    assert result.output == ANSWER
    assert result.usage.model == FLASH
    assert (result.usage.tokens_in, result.usage.tokens_out) == (10_000, 2_000)
    assert result.usage.cost_usd == Decimal("0.008")
    assert (result.usage.prompt_id, result.usage.prompt_version) == ("qa", 2)
    assert result.usage.latency_ms >= 0
    assert [(c.prompt_id, c.model) for c in fake.calls] == [("qa", FLASH)]


async def test_prompt_model_overrides_default() -> None:
    fake = FakeLLM({"qa": ANSWER})
    result = await make_gateway(fake).generate_structured(make_prompt(model=PRO), Answer)
    assert result.usage.model == PRO


async def test_transient_error_is_retried() -> None:
    fake = FakeLLM({"qa": [ModelAPIError("503"), ANSWER]})
    result = await make_gateway(fake).generate_structured(make_prompt(), Answer)
    assert result.output == ANSWER
    assert [c.model for c in fake.calls] == [FLASH, FLASH]


async def test_malformed_output_is_retried() -> None:
    fake = FakeLLM({"qa": ['{"text": "missing confidence"}', ANSWER]})
    result = await make_gateway(fake).generate_structured(make_prompt(), Answer)
    assert result.output == ANSWER
    assert len(fake.calls) == 2


async def test_persistent_bad_output_raises_typed_error() -> None:
    fake = FakeLLM({"qa": "not json at all"})
    with pytest.raises(StructuredOutputError, match="did not match schema"):
        await make_gateway(fake, fallback=None).generate_structured(make_prompt(), Answer)
    assert len(fake.calls) == 3


async def test_non_retryable_error_fails_fast() -> None:
    fake = FakeLLM({"qa": ModelInvalidRequestError("400 bad schema")})
    with pytest.raises(LLMError) as exc_info:
        await make_gateway(fake).generate_structured(make_prompt(), Answer)
    assert exc_info.value.retryable is False
    assert len(fake.calls) == 1  # no retry, no fallback


async def test_context_overflow_is_not_retried() -> None:
    fake = FakeLLM({"qa": ContextOverflowError("too long")})
    with pytest.raises(LLMError, match="context window"):
        await make_gateway(fake).generate_structured(make_prompt(), Answer)
    assert len(fake.calls) == 1


async def test_quota_on_primary_falls_back_to_cheaper_model() -> None:
    quota = ModelRateLimitError("429 RESOURCE_EXHAUSTED")
    fake = FakeLLM({"qa": [quota, quota, quota, ANSWER]})

    result = await make_gateway(fake).generate_structured(make_prompt(), Answer)

    assert [c.model for c in fake.calls] == [FLASH, FLASH, FLASH, LITE]
    assert result.usage.model == LITE
    assert result.usage.cost_usd == cost_usd(LITE, tokens_in=100, tokens_out=50)


async def test_quota_everywhere_raises_quota_exceeded() -> None:
    fake = FakeLLM({"qa": ModelRateLimitError("429")})
    with pytest.raises(QuotaExceeded) as exc_info:
        await make_gateway(fake).generate_structured(make_prompt(), Answer)
    assert exc_info.value.retryable is True
    assert len(fake.calls) == 6  # 3 attempts x (primary + fallback)


async def test_no_fallback_when_prompt_pins_the_fallback_model() -> None:
    fake = FakeLLM({"qa": ModelRateLimitError("429")})
    with pytest.raises(QuotaExceeded):
        await make_gateway(fake).generate_structured(make_prompt(model=LITE), Answer)
    assert len(fake.calls) == 3


def test_unknown_model_fails_at_construction() -> None:
    with pytest.raises(UnknownModel):
        LLMGateway(FakeLLM({}).factory, default_model="gemini-typo")


async def test_generation_is_traced_under_stage_span(span_recorder: SpanRecorder) -> None:
    fake = FakeLLM({"qa": ANSWER})
    with traced_stage("qa", version=1):
        await make_gateway(fake).generate_structured(
            make_prompt(), Answer, metadata={"video_id": "dQw4w9WgXcQ"}
        )

    spans = span_recorder.spans()
    stage = next(s for s in spans if s.name == "stage:qa")
    generations = [
        s for s in spans if (s.attributes or {}).get("langfuse.observation.type") == "generation"
    ]
    assert len(generations) == 1
    gen = generations[0]
    assert stage.context is not None
    assert gen.context is not None
    assert gen.context.trace_id == stage.context.trace_id

    # One observation per LLM call: the generation hangs directly off the stage span, with
    # no LangChain-internal CHAIN spans in between (15 of them per call before).
    assert gen.parent is not None
    assert gen.parent.span_id == stage.context.span_id
    types = [(s.attributes or {}).get("langfuse.observation.type") for s in spans]
    assert "chain" not in types
    assert sorted(s.name for s in spans) == sorted(["stage:qa", gen.name])
    assert gen.name == "gemini-2.5-flash:qa@v2"

    gen_attrs = gen.attributes or {}
    assert any("prompt_version" in k for k in gen_attrs)
    assert any("dQw4w9WgXcQ" in str(v) for v in gen_attrs.values())


def test_rate_limiter_shared_per_model_and_sized_from_registry() -> None:
    rpm = requests_per_minute(FLASH)
    assert rpm == 10
    assert requests_per_minute(FLASH, override=4) == 4
    limiter = rate_limiter(FLASH, rpm)
    assert limiter is rate_limiter(FLASH, rpm)
    assert limiter.requests_per_second == pytest.approx(10 / 60)


def test_gemini_factory_disables_sdk_retries_and_shares_limiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Construction is offline. Guards the "SDK retries x our retries" quota burn.
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    settings = Settings(
        gemini_api_key=SecretStr("k"),
        youtube_api_key=SecretStr("k"),
        database_url=SecretStr("postgresql://x"),
        langfuse_public_key=SecretStr("k"),
        langfuse_secret_key=SecretStr("k"),
    )
    model = gemini_factory(settings)(FLASH, 0.3, ModelOptions(media_resolution="low"))

    assert isinstance(model, ChatGoogleGenerativeAI)
    assert model.max_retries == 1
    assert model.temperature == 0.3
    assert model.rate_limiter is rate_limiter(FLASH, requests_per_minute(FLASH))
    assert model.media_resolution == MediaResolution.MEDIA_RESOLUTION_LOW


async def test_truncated_output_fails_fast_without_retry_or_fallback() -> None:
    cut = AIMessage(
        content='{"text": "enumerate yie',
        response_metadata={"finish_reason": "MAX_TOKENS"},
        usage_metadata={"input_tokens": 9, "output_tokens": 65_536, "total_tokens": 65_545},
    )
    fake = FakeLLM({"qa": cut})
    with pytest.raises(OutputTruncated):
        await make_gateway(fake).generate_structured(make_prompt(), Answer)
    assert len(fake.calls) == 1


async def test_model_precedence_and_options_reach_the_factory() -> None:
    fake = FakeLLM({"qa": ANSWER})
    gateway = make_gateway(fake)
    opts = ModelOptions(media_resolution="low", max_output_tokens=1000)

    await gateway.generate_structured(make_prompt(), Answer, model=PRO, options=opts)
    await gateway.generate_structured(make_prompt(model=LITE), Answer, model=PRO)

    assert [(c.model, c.options) for c in fake.calls] == [(PRO, opts), (LITE, ModelOptions())]


async def test_unknown_call_time_model_rejected_before_any_request() -> None:
    fake = FakeLLM({"qa": ANSWER})
    with pytest.raises(UnknownModel):
        await make_gateway(fake).generate_structured(make_prompt(), Answer, model="gemini-typo")
    assert fake.calls == []


@respx.mock
async def test_thinking_level_reaches_the_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wire-level, like media_resolution: a setting LangChain accepts may never be sent."""
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    secret = SecretStr("k")
    settings = Settings(
        gemini_api_key=secret,
        youtube_api_key=secret,
        database_url=secret,
        langfuse_public_key=secret,
        langfuse_secret_key=secret,
        llm_rpm_override=60_000,
    )
    route = respx.post(url__regex=r".*:generateContent.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": ANSWER.model_dump_json()}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 5},
            },
        )
    )
    gateway = LLMGateway(gemini_factory(settings), default_model=PRO, callbacks=list)

    await gateway.generate_structured(
        make_prompt(), Answer, options=ModelOptions(thinking_level="low")
    )

    body = json.loads(route.calls.last.request.content)
    thinking = body["generationConfig"]["thinkingConfig"]
    assert str(thinking.get("thinkingLevel", thinking.get("thinking_level"))).lower() == "low"
