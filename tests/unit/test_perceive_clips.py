"""Long videos are perceived in clips and stitched on the full video's timeline."""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from lecnotes.adapters.llm.fake import FakeCall, FakeLLM
from lecnotes.adapters.perception.gemini_url import (
    CLIP_OVERLAP_S,
    MAX_OUTPUT_TOKENS,
    RETRY_TEMPERATURE,
    Clip,
    GeminiUrlPerceiver,
    halve,
    plan_clips,
)
from lecnotes.adapters.perception.schema import (
    PerceivedBoardState,
    PerceivedSegment,
    PerceiveOutput,
)
from lecnotes.domain import VideoRef
from lecnotes.errors import VideoTooDense
from lecnotes.llm.gateway import LLMGateway

VIDEO = VideoRef(video_id="D1PcZaeQ2eg")
FLASH = "gemini-2.5-flash"
TRUNCATED = AIMessage(
    content='{"board_states": [', response_metadata={"finish_reason": "MAX_TOKENS"}
)


def output(*items: tuple[str, str, str]) -> PerceiveOutput:
    """One code board and one spoken segment per (start, end, text)."""
    return PerceiveOutput(
        board_states=[
            PerceivedBoardState(first_seen=a, last_seen=b, type="code", text=t, description="")
            for a, b, t in items
        ],
        segments=[PerceivedSegment(start=a, end=b, text=t) for a, b, t in items],
    )


def perceiver(fake: FakeLLM, **kwargs: Any) -> GeminiUrlPerceiver:
    gateway = LLMGateway(fake.factory, default_model=FLASH, retry_backoff=False)
    return GeminiUrlPerceiver(gateway, model=FLASH, fps=1, media_resolution="low", **kwargs)


def offsets(call: FakeCall) -> dict[str, Any]:
    user = call.messages[-1]
    assert isinstance(user, HumanMessage)
    assert isinstance(user.content, list)
    video_part = user.content[0]
    assert isinstance(video_part, dict)
    return video_part["video_metadata"]


def user_text(call: FakeCall) -> str:
    user = call.messages[-1]
    assert isinstance(user, HumanMessage)
    assert isinstance(user.content, list)
    text_part = user.content[1]
    assert isinstance(text_part, dict)
    return text_part["text"]


def test_plan_clips() -> None:
    assert plan_clips(None, 600) == [None]
    assert plan_clips(500, 600) == [Clip(0, 500, whole=True)]
    # 25 min -> three equal clips, each overlapping the next; the last ends at the end.
    assert plan_clips(1500, 600) == [
        Clip(0, 500 + CLIP_OVERLAP_S),
        Clip(500, 1000 + CLIP_OVERLAP_S),
        Clip(1000, 1500),
    ]


def test_halve_overlaps_in_the_middle() -> None:
    assert halve(Clip(600, 1000)) == [Clip(600, 800 + CLIP_OVERLAP_S), Clip(800, 1000)]


async def test_short_video_is_one_call_without_offsets() -> None:
    fake = FakeLLM({"perceive": output(("00:05", "00:20", "intro"))})
    await perceiver(fake).perceive(VIDEO, duration_s=500)

    [call] = fake.calls
    assert offsets(call) == {"fps": 1}
    assert "part from" not in user_text(call)


async def test_long_video_is_stitched_on_the_full_timeline() -> None:
    fake = FakeLLM(
        {
            "perceive": [
                output(("00:10", "00:30", "first part")),
                output(("08:30", "09:00", "second part, absolute times")),
                # Third clip starts at 16:40 but the model counted from the clip's start.
                output(("00:30", "01:00", "third part, relative times")),
            ]
        },
        tokens_out=40_000,
    )
    run = await perceiver(fake).perceive(VIDEO, duration_s=1500)

    assert [offsets(c) for c in fake.calls] == [
        {"fps": 1, "start_offset": "0s", "end_offset": "510s"},
        {"fps": 1, "start_offset": "500s", "end_offset": "1010s"},
        {"fps": 1, "start_offset": "1000s", "end_offset": "1500s"},
    ]
    assert "part from 08:20 to 16:50" in user_text(fake.calls[1])

    segs = run.result.segments
    assert [s.id for s in segs] == ["s0001", "s0002", "s0003"]  # unique across clips
    assert [(s.start_s, s.end_s) for s in segs] == [(10, 30), (510, 540), (1030, 1060)]
    boards = run.result.board_states
    assert [b.id for b in boards] == ["b0001", "b0002", "b0003"]
    assert [b.t_first_seen for b in boards] == [10, 510, 1030]
    assert len(run.usage) == 3
    assert any("shifted by 1000 s" in r for r in run.repaired)


async def test_items_far_outside_their_clip_are_dropped() -> None:
    fake = FakeLLM(
        {
            "perceive": [
                output(("00:10", "00:30", "ok")),
                output(("09:00", "09:10", "ok"), ("24:00", "24:10", "belongs to clip 3")),
                output(("17:00", "17:10", "ok")),
            ]
        }
    )
    run = await perceiver(fake).perceive(VIDEO, duration_s=1500)

    assert [s.text for s in run.result.segments] == ["ok", "ok", "ok"]
    assert sum("outside the clip" in d for d in run.dropped) == 2  # its segment and board


async def test_truncated_clip_is_halved() -> None:
    fake = FakeLLM(
        {
            "perceive": [
                TRUNCATED,
                output(("00:10", "00:20", "first half")),
                output(("02:40", "02:50", "second half")),
            ]
        }
    )
    run = await perceiver(fake).perceive(VIDEO, duration_s=300)

    assert [offsets(c) for c in fake.calls] == [
        {"fps": 1},
        {"fps": 1, "start_offset": "0s", "end_offset": "160s"},
        {"fps": 1, "start_offset": "150s", "end_offset": "300s"},
    ]
    assert [s.text for s in run.result.segments] == ["first half", "second half"]


async def test_looping_short_clip_is_retried_with_some_randomness() -> None:
    fake = FakeLLM({"perceive": [TRUNCATED, output(("00:10", "00:20", "unstuck"))]})
    run = await perceiver(fake).perceive(VIDEO, duration_s=200)  # < 2 x MIN_CLIP_S: no split

    assert [c.options.temperature for c in fake.calls] == [None, RETRY_TEMPERATURE]
    assert [s.text for s in run.result.segments] == ["unstuck"]


async def test_too_dense_when_the_shortest_clip_truncates_twice() -> None:
    fake = FakeLLM({"perceive": TRUNCATED})
    with pytest.raises(VideoTooDense) as exc_info:
        await perceiver(fake).perceive(VIDEO, duration_s=200)
    assert len(fake.calls) == 2  # once as usual, once at RETRY_TEMPERATURE
    assert "too much speech and on-screen text" in exc_info.value.user_message


async def test_output_cap_scales_with_clip_length() -> None:
    fake = FakeLLM({"perceive": output(("00:10", "00:20", "x"))})
    await perceiver(fake, thinking_budget=4096).perceive(VIDEO, duration_s=1500)

    # Three clips of 510, 510 and 500 s: 4,000 answer tokens per minute + the thinking budget.
    caps = [c.options.max_output_tokens for c in fake.calls]
    assert caps == [34_000 + 4096, 34_000 + 4096, 33_333 + 4096]
    assert all(cap is not None and cap < MAX_OUTPUT_TOKENS for cap in caps)


async def test_prompt_v1_still_works_with_clips() -> None:
    fake = FakeLLM({"perceive": output(("00:10", "00:30", "x"))})
    run = await perceiver(fake, prompt_version=1).perceive(VIDEO, duration_s=1500)

    assert len(fake.calls) == 3
    assert all("part from" not in user_text(c) for c in fake.calls)
    # v1 never mentions the clip, so the model counts from the clip's start: shifted back.
    assert [s.start_s for s in run.result.segments] == [10, 510, 1010]
