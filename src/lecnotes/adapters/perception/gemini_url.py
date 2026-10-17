"""Perception via Gemini reading the YouTube URL directly (plan: the ONLY video call).

Our server never downloads the video: the URL goes to Gemini as a `media` part, and Gemini
fetches it. Frame rate is set per part (`video_metadata.fps`); media resolution is set on
the model (per-part resolution is ignored by Gemini 2.5). Both are reachable through
LangChain, so this adapter needs no direct SDK calls and inherits the gateway's retries,
rate limiting, tracing and costing.
"""

from __future__ import annotations

from dataclasses import replace

from langchain_core.messages import HumanMessage

from lecnotes.adapters.perception.schema import PerceiveOutput, to_perception_result
from lecnotes.config import MediaResolution, Settings
from lecnotes.domain import PerceptionRun, VideoRef
from lecnotes.errors import LLMError, OutputTruncated, PerceptionError, QuotaExceeded
from lecnotes.llm.gateway import LLMGateway, ModelOptions
from lecnotes.observability.logging import get_logger
from lecnotes.prompts.loader import RenderedPrompt, load
from lecnotes.timefmt import format_timestamp

PROMPT_ID = "perceive"
# Gemini 2.5 Flash / Flash-Lite maximum. A long lecture's transcript + board snapshots is
# large; truncation is detected and reported rather than silently retried.
MAX_OUTPUT_TOKENS = 65_536

log = get_logger(__name__)


class GeminiUrlPerceiver:
    def __init__(
        self,
        gateway: LLMGateway,
        *,
        model: str,
        fps: float,
        media_resolution: MediaResolution,
        prompt_version: int = 1,
        thinking_budget: int | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self._gateway = gateway
        self._model = model
        self._fps = fps
        self._media_resolution: MediaResolution = media_resolution
        self._prompt_version = prompt_version
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
        }

    async def perceive(self, video: VideoRef, *, duration_s: float | None = None) -> PerceptionRun:
        duration = format_timestamp(duration_s) if duration_s else None
        prompt = with_video(
            load(PROMPT_ID, version=self._prompt_version, duration=duration), video.url, self._fps
        )
        try:
            result = await self._gateway.generate_structured(
                prompt,
                PerceiveOutput,
                model=self._model,
                options=self._options,
                metadata={"video_id": video.video_id, "fps": self._fps},
            )
        except QuotaExceeded:
            raise  # the job runner waits for the quota to reset
        except OutputTruncated as exc:
            raise PerceptionError(
                f"{video.video_id}: perception output truncated at {MAX_OUTPUT_TOKENS} tokens "
                "(video too long or too dense for one call)"
            ) from exc
        except LLMError as exc:
            raise PerceptionError(f"{video.video_id}: {exc}", retryable=exc.retryable) from exc

        conv = to_perception_result(result.output, video.video_id)
        perceived = conv.result
        if not perceived.segments and not perceived.board_states:
            raise PerceptionError(f"{video.video_id}: model returned no speech and no visuals")

        log.info(
            "perceive.done",
            video_id=video.video_id,
            segments=len(perceived.segments),
            board_states=len(perceived.board_states),
            dropped=len(conv.dropped),
            repaired=len(conv.repaired),
            tokens_out=result.usage.tokens_out,
        )
        for reason in conv.dropped:
            log.warning("perceive.item_dropped", video_id=video.video_id, reason=reason)

        return PerceptionRun(
            result=perceived, usage=[result.usage], dropped=conv.dropped, repaired=conv.repaired
        )


def with_video(prompt: RenderedPrompt, url: str, fps: float) -> RenderedPrompt:
    """Put the video in front of the user text (Gemini recommends media before the question)."""
    *head, user = prompt.messages
    if not isinstance(user, HumanMessage) or not isinstance(user.content, str):
        raise ValueError(f"{prompt.ref}: last message must be a plain-text HumanMessage")
    video_part = {"type": "media", "file_uri": url, "video_metadata": {"fps": fps}}
    message = HumanMessage(content=[video_part, {"type": "text", "text": user.content}])
    return replace(prompt, messages=[*head, message])
