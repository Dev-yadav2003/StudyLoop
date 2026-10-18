"""Guest allowance (lifetime, per account): GUEST_VIDEO_MINUTES of video and a token budget
for everything the AI does for them (notes + practice questions). The owner is unlimited but
still recorded, so usage can be reported for everyone.

Charges are made BEFORE the work, from the video's real length (YouTube metadata) and the
measured cost per minute (settings.tokens_per_video_minute), so a guest can never start a
job that would exceed the allowance. Adding a video the account already has costs nothing.
"""

from __future__ import annotations

from fastapi import HTTPException

from apps.api.auth import User
from lecnotes.bootstrap import ApiContainer
from lecnotes.domain import VideoMeta
from lecnotes.storage.repositories.usage import UsageTotals


def _remaining(c: ApiContainer, used: UsageTotals) -> tuple[float, int]:
    s = c.settings
    return s.guest_video_minutes - used.video_minutes, s.guest_token_budget - used.tokens


async def charge_video(c: ApiContainer, user: User, video: VideoMeta) -> None:
    """Charge (or refuse) a new lesson's video."""
    if await c.usage.has_video(user.id, video.video_id):
        return  # already paid for by this account
    minutes = video.duration_s / 60
    tokens = round(minutes * c.settings.tokens_per_video_minute)
    if not user.is_owner:
        left_min, left_tokens = _remaining(c, await c.usage.totals(user.id))
        if minutes > left_min or tokens > left_tokens:
            raise HTTPException(
                429,
                f"This video is {minutes:.0f} min; your account has {max(left_min, 0):.0f} min "
                f"of video left (guests get {c.settings.guest_video_minutes:.0f} min in total).",
            )
    await c.usage.charge(user.id, "video", video.video_id, video_minutes=minutes, tokens=tokens)


async def charge_questions(c: ApiContainer, user: User, notes_ref: str) -> None:
    tokens = c.settings.qa_token_cost
    if not user.is_owner:
        _, left_tokens = _remaining(c, await c.usage.totals(user.id))
        if tokens > left_tokens:
            raise HTTPException(
                429, "Your account has used its AI allowance, so no new practice questions."
            )
    await c.usage.charge(user.id, "qa", notes_ref, tokens=tokens)
