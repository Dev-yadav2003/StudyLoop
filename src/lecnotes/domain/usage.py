from decimal import Decimal

from pydantic import Field

from lecnotes.domain.base import DomainModel, NonEmptyStr


class LLMUsage(DomainModel):
    """Cost and size of one successful LLM call. Stages sum these into `StageRun` rows."""

    prompt_id: NonEmptyStr
    prompt_version: int = Field(ge=1)
    model: NonEmptyStr
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    cost_usd: Decimal = Field(ge=0, description="Paid-tier list price; free tier bills $0")
    latency_ms: float = Field(ge=0)


class KeyUsage(DomainModel):
    """One API key's use of one model on one quota day (Pacific)."""

    requests: int = Field(default=0, ge=0)
    exhausted: bool = False
    """A 429 said the daily budget is spent."""
