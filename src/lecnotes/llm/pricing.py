"""List prices per 1M tokens, USD. Edit here when Google changes pricing.

We run on the free tier, so the billed cost is $0. These are paid-tier list prices, used to
report what each stage *would* cost — the number that matters when comparing prompt and
model versions. Source: https://ai.google.dev/gemini-api/docs/pricing
(text/image/video input; prompts <= 200k tokens).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from lecnotes.errors import UnknownModel

_PER_TOKEN = Decimal(1_000_000)


@dataclass(frozen=True)
class Price:
    input_per_m: Decimal
    output_per_m: Decimal


PRICES: dict[str, Price] = {
    "gemini-2.5-flash": Price(input_per_m=Decimal("0.30"), output_per_m=Decimal("2.50")),
    "gemini-2.5-flash-lite": Price(input_per_m=Decimal("0.10"), output_per_m=Decimal("0.40")),
    "gemini-2.5-pro": Price(input_per_m=Decimal("1.25"), output_per_m=Decimal("10.00")),
}


def get_price(model: str) -> Price:
    try:
        return PRICES[model]
    except KeyError:
        raise UnknownModel(model, known=sorted(PRICES)) from None


def cost_usd(model: str, *, tokens_in: int, tokens_out: int) -> Decimal:
    """Exact cost (Decimal, no float drift). Output tokens include any thinking tokens."""
    if tokens_in < 0 or tokens_out < 0:
        raise ValueError("token counts must be non-negative")
    price = get_price(model)
    return (tokens_in * price.input_per_m + tokens_out * price.output_per_m) / _PER_TOKEN
