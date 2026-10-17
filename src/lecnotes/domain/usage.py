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
