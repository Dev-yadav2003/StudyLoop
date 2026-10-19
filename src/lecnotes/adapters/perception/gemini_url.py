"""Perception via Gemini reading the YouTube URL directly (the ONLY call that sees video).

Our server never downloads the video: the URL goes to Gemini as a `media` part, and Gemini
fetches it. Frame rate is set per part (`video_metadata.fps`); media resolution is set on
the model (per-part resolution is ignored by Gemini 2.5). Both are reachable through
LangChain, so this adapter needs no direct SDK calls and inherits the gateway's retries,
rate limiting, tracing and costing.

Long videos are perceived in clips. One reply is capped at MAX_OUTPUT_TOKENS, and a dense
lecture (code on screen, every change a full snapshot) passes that in ~15-20 minutes. So a
video longer than `window_s` is cut into equal clips (`video_metadata.start_offset` /
`end_offset`) that overlap by CLIP_OVERLAP_S, perceived one after another, and stitched on
the full video's timeline. A clip that still truncates is halved, down to MIN_CLIP_S. The
overlap's duplicates are removed by the validate stage, like any repeated speech or slide.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from statistics import median

from langchain_core.messages import HumanMessage

from lecnotes.adapters.perception.schema import Conversion, PerceiveOutput, to_perception_result
from lecnotes.config import MediaResolution, Settings
from lecnotes.domain import (
    BoardState,
    LLMUsage,
    PerceptionResult,
    PerceptionRun,
    TranscriptSegment,
    VideoRef,
)
from lecnotes.errors import (
    LLMError,
    OutputTruncated,
    PerceptionError,
    QuotaExceeded,
    VideoTooDense,
)
from lecnotes.llm.gateway import LLMGateway, ModelOptions
from lecnotes.observability.logging import get_logger
from lecnotes.prompts.loader import RenderedPrompt, load
from lecnotes.timefmt import format_timestamp

PROMPT_ID = "perceive"
# Gemini 2.5 Flash / Flash-Lite maximum. A long lecture's transcript + board snapshots is
# large; truncation is detected and handled by perceiving shorter clips.
MAX_OUTPUT_TOKENS = 65_536
# Consecutive clips share this much video, so a sentence or slide cut at a boundary is
# still seen whole by one of them.
CLIP_OVERLAP_S = 10.0
# A truncated clip is halved until it is this short; then the video is too dense to read.
MIN_CLIP_S = 120.0
# Items timed this far outside their clip are model mistakes and are dropped.
CLIP_TOLERANCE_S = 30.0
# First prompt version that declares clip_start / clip_end (v1 renders without them).
CLIP_PROMPT_VERSION = 2

log = get_logger(__name__)


@dataclass(frozen=True)
class Clip:
    """A part of the video, in seconds. `whole` = the full video: sent without offsets."""

    start: float
    end: float
    whole: bool = False

    @property
    def label(self) -> str:
        return f"{format_timestamp(self.start)}-{format_timestamp(self.end)}"


@dataclass(frozen=True)
class _Part:
    clip: Clip | None  # None: duration unknown, the whole video in one call
    conv: Conversion
    usage: LLMUsage


def plan_clips(duration_s: float | None, window_s: float) -> list[Clip | None]:
    """Equal clips of at most `window_s` (plus overlap); one call when the video fits."""
    if not duration_s:
        return [None]
    if duration_s <= window_s:
        return [Clip(0.0, duration_s, whole=True)]
    n = math.ceil(duration_s / window_s)
    size = duration_s / n
    return [Clip(i * size, min((i + 1) * size + CLIP_OVERLAP_S, duration_s)) for i in range(n)]


def halve(clip: Clip) -> list[Clip]:
    mid = (clip.start + clip.end) / 2
    return [Clip(clip.start, min(mid + CLIP_OVERLAP_S, clip.end)), Clip(mid, clip.end)]


class GeminiUrlPerceiver:
    def __init__(
        self,
        gateway: LLMGateway,
        *,
        model: str,
        fps: float,
        media_resolution: MediaResolution,
        prompt_version: int = CLIP_PROMPT_VERSION,
        thinking_budget: int | None = None,
        timeout_s: float | None = None,
        window_s: float = 600.0,
    ) -> None:
        self._gateway = gateway
        self._model = model
        self._fps = fps
        self._media_resolution: MediaResolution = media_resolution
        self._prompt_version = prompt_version
        self._window_s = window_s
        self._options = ModelOptions(
            media_resolution=media_resolution,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_budget=thinking_budget,
            timeout_s=timeout_s,
        )

    @classmethod
    def from_settings(cls, settings: Settings, gateway: LLMGateway) -> GeminiUrlPerceiver:
        return cls(
            gateway,
            model=settings.perceive_model,
            fps=settings.perceive_fps,
            media_resolution=settings.perceive_media_resolution,
            prompt_version=settings.perceive_prompt_version,
            timeout_s=settings.perceive_timeout_s,
            window_s=settings.perceive_window_s,
        )

    @property
    def fingerprint(self) -> dict[str, str | int | float]:
        return {
            "adapter": "gemini_url",
            "prompt_id": PROMPT_ID,
            "prompt_version": self._prompt_version,
            "model": self._model,
            "fps": self._fps,
            "media_resolution": self._media_resolution,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "window_s": self._window_s,
        }

    async def perceive(self, video: VideoRef, *, duration_s: float | None = None) -> PerceptionRun:
        clips = plan_clips(duration_s, self._window_s)
        parts: list[_Part] = []
        # One clip at a time: each is a full-price video request, and running them together
        # would trip the free tier's tokens-per-minute limit on long lectures.
        for clip in clips:
            parts += await self._perceive_clip(video, clip, duration_s)

        run = stitch(parts, video.video_id)
        if not run.result.segments and not run.result.board_states:
            raise PerceptionError(f"{video.video_id}: model returned no speech and no visuals")

        log.info(
            "perceive.done",
            video_id=video.video_id,
            clips=len(parts),
            segments=len(run.result.segments),
            board_states=len(run.result.board_states),
            dropped=len(run.dropped),
            repaired=len(run.repaired),
            tokens_out=sum(u.tokens_out for u in run.usage),
        )
        for reason in run.dropped:
            log.warning("perceive.item_dropped", video_id=video.video_id, reason=reason)
        return run

    async def _perceive_clip(
        self, video: VideoRef, clip: Clip | None, duration_s: float | None
    ) -> list[_Part]:
        prompt = with_video(self._prompt(clip, duration_s), video.url, self._fps, clip)
        try:
            result = await self._gateway.generate_structured(
                prompt,
                PerceiveOutput,
                model=self._model,
                options=self._options,
                metadata={
                    "video_id": video.video_id,
                    "fps": self._fps,
                    "clip": clip.label if clip else "all",
                },
            )
        except QuotaExceeded:
            raise  # the job runner waits for the quota to reset
        except OutputTruncated as exc:
            if clip is None or clip.end - clip.start < 2 * MIN_CLIP_S:
                raise VideoTooDense(
                    f"{video.video_id}: perception output truncated at {MAX_OUTPUT_TOKENS} "
                    f"tokens for clip {clip.label if clip else 'all'}"
                ) from exc
            log.warning("perceive.clip_truncated", video_id=video.video_id, clip=clip.label)
            parts: list[_Part] = []
            for half in halve(clip):
                parts += await self._perceive_clip(video, half, duration_s)
            return parts
        except LLMError as exc:
            raise PerceptionError(f"{video.video_id}: {exc}", retryable=exc.retryable) from exc

        return [_Part(clip, to_perception_result(result.output, video.video_id), result.usage)]

    def _prompt(self, clip: Clip | None, duration_s: float | None) -> RenderedPrompt:
        variables: dict[str, str | None] = {
            "duration": format_timestamp(duration_s) if duration_s else None
        }
        if self._prompt_version >= CLIP_PROMPT_VERSION:
            part = clip if clip is not None and not clip.whole else None
            variables["clip_start"] = format_timestamp(part.start) if part else None
            variables["clip_end"] = format_timestamp(part.end) if part else None
        return load(PROMPT_ID, version=self._prompt_version, **variables)


def stitch(parts: list[_Part], video_id: str) -> PerceptionRun:
    """Put every clip's items on the full video's timeline and renumber their IDs."""
    segments: list[TranscriptSegment] = []
    boards: list[BoardState] = []
    dropped: list[str] = []
    repaired: list[str] = []
    for part in parts:
        clip, result = part.clip, part.conv.result
        tag = f"[clip {clip.label}] " if clip and not clip.whole else ""
        dropped += [tag + d for d in part.conv.dropped]
        repaired += [tag + r for r in part.conv.repaired]
        shift = clip_shift(result, clip)
        if shift:
            repaired.append(f"{tag}times were relative to the clip; shifted by {shift:.0f} s")
        lo = clip.start - CLIP_TOLERANCE_S if clip else -math.inf
        hi = clip.end + CLIP_TOLERANCE_S if clip else math.inf
        for s in result.segments:
            start, end = s.start_s + shift, s.end_s + shift
            if not lo <= start <= hi:
                dropped.append(f"{tag}segment at {format_timestamp(start)}: outside the clip")
                continue
            segments.append(
                s.model_copy(
                    update={"id": f"s{len(segments) + 1:04d}", "start_s": start, "end_s": end}
                )
            )
        for b in result.board_states:
            first, last = b.t_first_seen + shift, b.t_last_seen + shift
            if not lo <= first <= hi:
                dropped.append(f"{tag}board at {format_timestamp(first)}: outside the clip")
                continue
            boards.append(
                b.model_copy(
                    update={
                        "id": f"b{len(boards) + 1:04d}",
                        "t_first_seen": first,
                        "t_last_seen": last,
                    }
                )
            )
    return PerceptionRun(
        result=PerceptionResult(video_id=video_id, segments=segments, board_states=boards),
        usage=[p.usage for p in parts],
        dropped=dropped,
        repaired=repaired,
    )


def clip_shift(result: PerceptionResult, clip: Clip | None) -> float:
    """Seconds to add to the model's times. Asked for full-video times, models sometimes
    count from the clip's start instead. The two readings don't overlap for a clip that
    starts at `start` > 0 (relative times are mostly < start, absolute ones >= start), so the
    median start time tells them apart."""
    if clip is None or clip.start <= 0:
        return 0.0
    starts = [s.start_s for s in result.segments] + [b.t_first_seen for b in result.board_states]
    if not starts or median(starts) >= clip.start:
        return 0.0
    return clip.start


def with_video(
    prompt: RenderedPrompt, url: str, fps: float, clip: Clip | None = None
) -> RenderedPrompt:
    """Put the video in front of the user text (Gemini recommends media before the question)."""
    *head, user = prompt.messages
    if not isinstance(user, HumanMessage) or not isinstance(user.content, str):
        raise ValueError(f"{prompt.ref}: last message must be a plain-text HumanMessage")
    metadata: dict[str, float | str] = {"fps": fps}
    if clip is not None and not clip.whole:
        metadata["start_offset"] = f"{clip.start:.0f}s"
        metadata["end_offset"] = f"{clip.end:.0f}s"
    video_part = {"type": "media", "file_uri": url, "video_metadata": metadata}
    message = HumanMessage(content=[video_part, {"type": "text", "text": user.content}])
    return replace(prompt, messages=[*head, message])
