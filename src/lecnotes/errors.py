"""Typed errors raised by adapters and the LLM gateway (plan §4.6).

The orchestrator records `type(exc).__name__` as `error_type` in `stage_runs` and shows
`user_message` to the user. `retryable` tells the job runner whether a later attempt
(e.g. after the quota resets) could succeed.
"""

from __future__ import annotations


class LecNotesError(Exception):
    user_message = "Something went wrong while processing this video."
    retryable = False


class UnknownModel(LecNotesError, LookupError):
    def __init__(self, name: str, *, known: list[str]) -> None:
        super().__init__(f"unknown model {name!r}; known: {known}")
        self.name = name


class LLMError(LecNotesError):
    user_message = "The AI model failed to process this video. Please try again later."

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class QuotaExceeded(LLMError):
    user_message = "The free AI quota is used up for now. Your job will resume when it resets."

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=True)


class StructuredOutputError(LLMError):
    """The model answered, but not in the requested schema (after retries)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=True)


class OutputTruncated(LLMError):
    """The model hit its output-token limit; the JSON is cut off. Retrying the same request
    would burn quota for the same result, so this is not retryable."""


class PerceptionError(LecNotesError):
    user_message = (
        "We could not read this video. It may be private, age-restricted, region-locked, "
        "or too long."
    )

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class InvalidLink(LecNotesError):
    user_message = "That doesn't look like a link to a YouTube video."


class PlaylistNotSupported(InvalidLink):
    user_message = (
        "Playlists aren't supported. Open the video you want from the playlist and paste its link."
    )


class VideoUnavailable(LecNotesError):
    user_message = (
        "This video can't be processed: it is private, deleted, live, or not yet processed "
        "by YouTube. Only public videos are supported."
    )


class CatalogError(LecNotesError):
    user_message = "We couldn't look up this video on YouTube. Please try again later."

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class CatalogQuotaExceeded(CatalogError):
    user_message = "The YouTube lookup quota is used up for today. Your job will resume later."

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=True)
