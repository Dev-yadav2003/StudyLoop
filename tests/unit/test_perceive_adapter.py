import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from langchain_core.exceptions import ModelInvalidRequestError, ModelRateLimitError
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import SecretStr

from lecnotes.adapters.llm.fake import FakeLLM
from lecnotes.adapters.perception.gemini_url import MAX_OUTPUT_TOKENS, GeminiUrlPerceiver
from lecnotes.adapters.perception.schema import PerceiveOutput
from lecnotes.config import Settings
from lecnotes.domain import VideoRef
from lecnotes.errors import PerceptionError, QuotaExceeded
from lecnotes.llm.gateway import LLMGateway, ModelOptions, gemini_factory
from lecnotes.ports.perception import Perceiver

FIXTURE = Path(__file__).parents[1] / "fixtures" / "perceive_output_sample.json"
VIDEO = VideoRef(video_id="dQw4w9WgXcQ")
FLASH = "gemini-2.5-flash"


def sample() -> PerceiveOutput:
    return PerceiveOutput.model_validate_json(FIXTURE.read_text(encoding="utf-8"))


def perceiver(fake: FakeLLM, **kwargs: Any) -> GeminiUrlPerceiver:
    gateway = LLMGateway(
        fake.factory, default_model=FLASH, fallback_model=None, retry_backoff=False
    )
    return GeminiUrlPerceiver(
        gateway, model=FLASH, fps=0.5, media_resolution="low", timeout_s=600, **kwargs
    )


def test_satisfies_port() -> None:
    p: Perceiver = perceiver(FakeLLM({}))
    assert p.fingerprint["prompt_version"] == 1
    assert p.fingerprint["fps"] == 0.5


async def test_perceive_returns_run_with_usage_and_report() -> None:
    fake = FakeLLM({"perceive": sample()}, tokens_in=50_000, tokens_out=4_000)

    run = await perceiver(fake).perceive(VIDEO, duration_s=754)

    assert len(run.result.board_states) == 3
    assert len(run.result.segments) == 4
    assert run.result.video_id == VIDEO.video_id
    assert len(run.dropped) == 4
    assert len(run.repaired) == 1
    [usage] = run.usage
    assert (usage.prompt_id, usage.prompt_version, usage.model) == ("perceive", 1, FLASH)
    assert usage.tokens_in == 50_000


async def test_video_part_options_and_duration_sent_to_model() -> None:
    fake = FakeLLM({"perceive": sample()})
    await perceiver(fake).perceive(VIDEO, duration_s=754)

    [call] = fake.calls
    assert call.options == ModelOptions(
        media_resolution="low", max_output_tokens=MAX_OUTPUT_TOKENS, timeout_s=600
    )
    user = call.messages[-1]
    assert isinstance(user, HumanMessage)
    assert isinstance(user.content, list)
    video_part, text_part = user.content
    assert video_part == {
        "type": "media",
        "file_uri": VIDEO.url,
        "video_metadata": {"fps": 0.5},
    }
    assert isinstance(text_part, dict)
    assert "(duration 12:34)" in text_part["text"]


async def test_truncation_becomes_perception_error() -> None:
    cut = AIMessage(content='{"board_states": [', response_metadata={"finish_reason": "MAX_TOKENS"})
    fake = FakeLLM({"perceive": cut})
    with pytest.raises(PerceptionError, match="truncated"):
        await perceiver(fake).perceive(VIDEO)
    assert len(fake.calls) == 1


async def test_quota_propagates_unchanged() -> None:
    fake = FakeLLM({"perceive": ModelRateLimitError("429")})
    with pytest.raises(QuotaExceeded):
        await perceiver(fake).perceive(VIDEO)


async def test_rejected_request_becomes_perception_error() -> None:
    fake = FakeLLM({"perceive": ModelInvalidRequestError("400 video is private")})
    with pytest.raises(PerceptionError) as exc_info:
        await perceiver(fake).perceive(VIDEO)
    assert exc_info.value.retryable is False
    assert "private" in exc_info.value.user_message


async def test_empty_perception_is_an_error() -> None:
    fake = FakeLLM({"perceive": PerceiveOutput(board_states=[], segments=[])})
    with pytest.raises(PerceptionError, match="no speech and no visuals"):
        await perceiver(fake).perceive(VIDEO)


# --- wire level: real ChatGoogleGenerativeAI + google-genai SDK, HTTP mocked by respx ---------


def gemini_response(text: str, finish_reason: str = "STOP") -> dict[str, Any]:
    return {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": text}]},
                "finishReason": finish_reason,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 51_234,
            "candidatesTokenCount": 3_210,
            "totalTokenCount": 54_444,
        },
        "modelVersion": FLASH,
    }


@pytest.fixture
def real_gateway(monkeypatch: pytest.MonkeyPatch) -> LLMGateway:
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    secret = SecretStr("test-key")
    settings = Settings(
        gemini_api_key=secret,
        youtube_api_key=secret,
        database_url=secret,
        langfuse_public_key=secret,
        langfuse_secret_key=secret,
        llm_rpm_override=60_000,  # the limiter's bucket starts empty; don't wait in tests
    )
    return LLMGateway(
        gemini_factory(settings), default_model=FLASH, retry_backoff=False, callbacks=list
    )


@respx.mock
async def test_wire_request_carries_video_url_fps_and_resolution(
    real_gateway: LLMGateway,
) -> None:
    route = respx.post(url__regex=r".*/models/gemini-2\.5-flash:generateContent.*").mock(
        return_value=httpx.Response(200, json=gemini_response(FIXTURE.read_text("utf-8")))
    )
    p = GeminiUrlPerceiver(real_gateway, model=FLASH, fps=0.5, media_resolution="low")

    run = await p.perceive(VIDEO, duration_s=754)

    body = json.loads(route.calls.last.request.content)
    parts = body["contents"][-1]["parts"]
    video = next(part for part in parts if "fileData" in part)
    # The SDK forwards FileData with its proto field name (`file_uri`); proto3 JSON parsers
    # accept both spellings, so either is a valid request.
    file_data = video["fileData"]
    assert (file_data.get("fileUri") or file_data.get("file_uri")) == VIDEO.url
    assert video["videoMetadata"]["fps"] == 0.5
    config = body["generationConfig"]
    assert config["mediaResolution"] == "MEDIA_RESOLUTION_LOW"
    assert config["maxOutputTokens"] == MAX_OUTPUT_TOKENS
    assert config["responseMimeType"] == "application/json"
    assert list(config["responseJsonSchema"]["properties"]) == ["board_states", "segments"]
    assert "VERBATIM" in json.dumps(body["systemInstruction"])

    assert len(run.result.board_states) == 3
    assert run.usage[0].tokens_in == 51_234
    assert run.usage[0].tokens_out == 3_210


@respx.mock
async def test_wire_max_tokens_is_not_retried(real_gateway: LLMGateway) -> None:
    route = respx.post(url__regex=r".*:generateContent.*").mock(
        return_value=httpx.Response(
            200, json=gemini_response('{"board_states": [{"first', finish_reason="MAX_TOKENS")
        )
    )
    p = GeminiUrlPerceiver(real_gateway, model=FLASH, fps=1, media_resolution="low")

    with pytest.raises(PerceptionError, match="truncated"):
        await p.perceive(VIDEO)
    assert route.call_count == 1  # no retries, no fallback: each attempt costs video quota
