"""Typed application settings.

All behaviour that may change between environments (models, thresholds, feature flags)
lives here, never as literals in code. Secrets are required and validated on first access,
so a missing key fails at startup rather than in the middle of a job.

Usage:
    from lecnotes.config import settings      # validated, cached singleton
    from lecnotes.config import get_settings  # same object; handy for FastAPI Depends
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    SecretStr,
    ValidationError,
    ValidationInfo,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

MediaResolution = Literal["low", "medium", "high"]


class ConfigError(RuntimeError):
    """Raised when settings are missing or invalid. Lists every offending env var."""


class FeatureFlags(BaseModel):
    """Toggled with env vars like `FEATURES__QA=true`."""

    merge: bool = True
    verify: bool = True
    qa: bool = False


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        frozen=True,
    )

    # --- runtime ---
    env: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # --- secrets (required) ---
    gemini_api_key: SecretStr
    youtube_api_key: SecretStr
    database_url: SecretStr
    langfuse_public_key: SecretStr
    langfuse_secret_key: SecretStr
    langfuse_host: str = "https://cloud.langfuse.com"

    # --- models ---
    perceive_model: str = "gemini-2.5-flash"
    text_model: str = "gemini-2.5-flash"
    fallback_model: str = "gemini-2.5-flash-lite"
    llm_requests_per_minute: float = Field(default=10, gt=0)

    # --- perception ---
    perceive_fps: float = Field(default=1.0, gt=0, le=24)
    perceive_media_resolution: MediaResolution = "low"

    # --- chunking (seconds) ---
    chunk_min_s: int = Field(default=120, gt=0)
    chunk_max_s: int = Field(default=300, gt=0)

    features: FeatureFlags = FeatureFlags()

    @field_validator(
        "gemini_api_key",
        "youtube_api_key",
        "database_url",
        "langfuse_public_key",
        "langfuse_secret_key",
    )
    @classmethod
    def _secret_not_blank(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("must not be empty")
        return value

    @field_validator("chunk_max_s")
    @classmethod
    def _chunk_bounds(cls, value: int, info: ValidationInfo) -> int:
        min_s = info.data.get("chunk_min_s")
        if min_s is not None and value < min_s:
            raise ValueError(f"must be >= chunk_min_s ({min_s})")
        return value


def load_settings(**overrides: object) -> Settings:
    """Build settings from env/.env, converting validation errors into a readable ConfigError."""
    try:
        return Settings(**overrides)  # type: ignore[arg-type]
    except ValidationError as exc:
        problems = []
        for err in exc.errors():
            env_var = "__".join(str(part) for part in err["loc"]).upper()
            problems.append(f"  {env_var}: {err['msg']}")
        raise ConfigError("Invalid configuration:\n" + "\n".join(problems)) from None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def __getattr__(name: str) -> Settings:
    # Lazy module attribute: importing this module never fails, accessing `settings` validates.
    if name == "settings":
        return get_settings()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
