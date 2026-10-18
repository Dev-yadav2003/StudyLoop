from decimal import Decimal

import pytest

from lecnotes.errors import UnknownModel
from lecnotes.llm.models import MODELS, get_model
from lecnotes.llm.pricing import PRICES, cost_usd


def test_known_token_counts() -> None:
    # 10k in @ $0.30/M = $0.003 ; 2k out @ $2.50/M = $0.005
    assert cost_usd("gemini-2.5-flash", tokens_in=10_000, tokens_out=2_000) == Decimal("0.008")
    # 1M in @ $0.25 + 1M out @ $1.50
    lite = cost_usd("gemini-3.1-flash-lite", tokens_in=1_000_000, tokens_out=1_000_000)
    assert lite == Decimal("1.75")


def test_zero_tokens_cost_nothing() -> None:
    assert cost_usd("gemini-3-flash-preview", tokens_in=0, tokens_out=0) == 0


def test_negative_tokens_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        cost_usd("gemini-2.5-flash", tokens_in=-1, tokens_out=0)


def test_unknown_model_raises() -> None:
    with pytest.raises(UnknownModel, match="gpt-9"):
        cost_usd("gpt-9", tokens_in=1, tokens_out=1)
    with pytest.raises(UnknownModel):
        get_model("gpt-9")


def test_every_registered_model_has_a_price() -> None:
    # Otherwise a successful LLM call would fail when we try to cost it.
    assert set(MODELS) <= set(PRICES)


def test_registry_lookup() -> None:
    m = get_model("gemini-2.5-flash")
    assert m.name == "gemini-2.5-flash"
    assert m.requests_per_minute > 0
