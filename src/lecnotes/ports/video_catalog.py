from typing import Protocol

from lecnotes.domain import VideoMeta


class VideoCatalog(Protocol):
    """Turns a user's link into the metadata of the one video it points at (ADR 0007).

    Official APIs only, never scraping: YouTube blocks cloud IPs (by design, the server never
    downloads from YouTube).
    """

    async def resolve(self, link: str) -> VideoMeta:
        """Raises InvalidLink (incl. PlaylistNotSupported), VideoUnavailable, CatalogError,
        CatalogQuotaExceeded."""
        ...
