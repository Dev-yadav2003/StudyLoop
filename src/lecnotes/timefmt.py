"""Video timestamps: parse "SS", "MM:SS", "H:MM:SS" (optional fractional seconds) and format."""

from __future__ import annotations

import re

_TIMESTAMP = re.compile(r"^\s*(?:(?:(\d+):)?(\d{1,2}):)?(\d+(?:\.\d+)?)\s*$")


def parse_timestamp(value: str | float | int) -> float:
    """Return seconds. Raises ValueError on anything that is not a clean timestamp."""
    if isinstance(value, int | float):
        if value < 0:
            raise ValueError(f"negative timestamp: {value}")
        return float(value)
    match = _TIMESTAMP.match(value)
    if not match:
        raise ValueError(f"not a timestamp: {value!r}")
    raw_h, raw_m, raw_s = match.groups()
    h, m, s = int(raw_h or 0), int(raw_m or 0), float(raw_s)
    if raw_m is not None and s >= 60:
        raise ValueError(f"seconds out of range in {value!r}")
    if raw_h is not None and m >= 60:
        raise ValueError(f"minutes out of range in {value!r}")
    return h * 3600 + m * 60 + s


def format_timestamp(seconds: float) -> str:
    """`mm:ss`, or `h:mm:ss` from one hour on. Truncates fractions (matches YouTube links)."""
    if seconds < 0:
        raise ValueError(f"negative timestamp: {seconds}")
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
