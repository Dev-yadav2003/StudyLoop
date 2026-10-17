from typing import Protocol

from lecnotes.domain import ResolvedVideos


class VideoCatalog(Protocol):
    """Expands a user's link into video metadata. Official APIs only, never scraping:
    YouTube blocks cloud IPs (plan: the server never downloads from YouTube)."""

    async def resolve(self, link: str) -> ResolvedVideos:
        """Raises InvalidLink, VideoUnavailable, CatalogError, CatalogQuotaExceeded."""
        ...
