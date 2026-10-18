import pytest

from lecnotes.timefmt import format_timestamp, parse_timestamp


@pytest.mark.parametrize(
    ("raw", "seconds"),
    [
        ("0", 0),
        ("45", 45),
        ("12.5", 12.5),
        ("01:05", 65),
        ("1:05", 65),
        ("10:00.5", 600.5),
        ("1:02:03", 3723),
        (" 02:03 ", 123),
        (90, 90.0),
        (1.5, 1.5),
    ],
)
def test_parse(raw: str | float, seconds: float) -> None:
    assert parse_timestamp(raw) == seconds


@pytest.mark.parametrize("raw", ["", "abc", "1:2:3:4", "01:75", "1:60:00", "-5", "5s", -1])
def test_parse_rejects(raw: str | float) -> None:
    with pytest.raises(ValueError, match=r"timestamp|range"):
        parse_timestamp(raw)


@pytest.mark.parametrize(
    ("seconds", "text"), [(0, "00:00"), (65.9, "01:05"), (3599, "59:59"), (3723, "1:02:03")]
)
def test_format(seconds: float, text: str) -> None:
    assert format_timestamp(seconds) == text


def test_round_trip() -> None:
    for s in (0, 59, 61, 3600, 7322):
        assert parse_timestamp(format_timestamp(s)) == s
