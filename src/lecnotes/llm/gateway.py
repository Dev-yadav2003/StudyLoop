"""The only place that calls an LLM (plan §4.3).

Every call goes through `LLMGateway.generate_structured`, which guarantees it is:
  - validated  — output parsed into the requested Pydantic schema, or a typed error;
  - rate-limited — per-model token bucket sized to the free tier;
  - retried    — only on errors that can succeed on a second attempt;
  - degraded   — falls back to a cheaper model when the primary is out of quota;
  - traced     — Langfuse generation nested under the current stage span;
  - costed     — tokens + list-price cost returned in `LLMResult.usage`.

The chat model is built by an injected factory, so tests swap in `adapters/llm/fake.py`
and exercise the exact same retry / fallback / tracing path offline.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from functools import partial
from typing import Any, cast

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAPIError,
    ModelConnectionError,
    ModelError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from lecnotes.config import Settings
from lecnotes.errors import LLMError, QuotaExceeded, StructuredOutputError
from lecnotes.llm.models import get_model
from lecnotes.llm.pricing import cost_usd
from lecnotes.llm.rate_limit import rate_limiter, requests_per_minute
from lecnotes.observability.tracing import langchain_callbacks
from lecnotes.prompts.loader import PromptRef, RenderedPrompt

ModelFactory = Callable[[str, float], BaseChatModel]
"""(model name, temperature) -> chat model."""

# Errors where another attempt (same model after backoff, or the fallback model) may succeed.
TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    ModelRateLimitError,
    ModelAPIError,
    ModelConnectionError,
    ModelTimeoutError,
    StructuredOutputError,
)


@dataclass(frozen=True)
class LLMUsage:
    prompt: PromptRef
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: Decimal
    latency_ms: float


@dataclass(frozen=True)
class LLMResult[T: BaseModel]:
    output: T
    usage: LLMUsage


@dataclass(frozen=True)
class _Attempt:
    parsed: BaseModel
    raw: AIMessage
    model: str


def gemini_factory(settings: Settings) -> ModelFactory:
    """Production factory: Gemini via LangChain, with SDK retries off (we retry one level up)."""

    def build(model: str, temperature: float) -> BaseChatModel:
        rpm = requests_per_minute(model, settings.llm_rpm_override)
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            api_key=settings.gemini_api_key,
            rate_limiter=rate_limiter(model, rpm),
            # The SDK defaults to 6 internal retries; combined with ours that would be 18
            # requests per failure. 1 = "no SDK retries" (0 means "SDK default").
            max_retries=1,
            timeout=settings.llm_timeout_s,
        )

    return build


class LLMGateway:
    def __init__(
        self,
        model_factory: ModelFactory,
        *,
        default_model: str,
        fallback_model: str | None = None,
        retry_attempts: int = 3,
        retry_backoff: bool = True,
        callbacks: Callable[[], list[BaseCallbackHandler]] = langchain_callbacks,
    ) -> None:
        for name in filter(None, (default_model, fallback_model)):
            get_model(name)  # fail at startup on a typo, not on the first call
        self._factory = model_factory
        self._default_model = default_model
        self._fallback_model = fallback_model if fallback_model != default_model else None
        self._retry_attempts = retry_attempts
        self._retry_backoff = retry_backoff
        self._callbacks = callbacks

    @classmethod
    def from_settings(cls, settings: Settings) -> LLMGateway:
        return cls(
            gemini_factory(settings),
            default_model=settings.text_model,
            fallback_model=settings.fallback_model,
            retry_attempts=settings.llm_retry_attempts,
        )

    async def generate_structured[T: BaseModel](
        self,
        prompt: RenderedPrompt,
        schema: type[T],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> LLMResult[T]:
        primary = prompt.spec.model or self._default_model
        chain = self._branch(primary, prompt.spec.temperature, schema)
        if self._fallback_model and self._fallback_model != primary:
            fallback = self._branch(self._fallback_model, prompt.spec.temperature, schema)
            chain = chain.with_fallbacks([fallback], exceptions_to_handle=TRANSIENT_ERRORS)

        config: RunnableConfig = {
            "callbacks": self._callbacks(),
            "run_name": f"llm:{prompt.ref}",
            "metadata": {
                "prompt_id": prompt.ref.id,
                "prompt_version": prompt.ref.version,
                "response_schema": schema.__name__,
                **(metadata or {}),
            },
        }

        started = time.perf_counter()
        try:
            attempt = await chain.ainvoke(prompt.messages, config=config)
        except StructuredOutputError:
            raise
        except ModelRateLimitError as exc:
            raise QuotaExceeded(f"{prompt.ref}: rate limit / quota exhausted: {exc}") from exc
        except ContextOverflowError as exc:
            raise LLMError(f"{prompt.ref}: input exceeds context window: {exc}") from exc
        except ModelError as exc:
            raise LLMError(f"{prompt.ref}: {exc}", retryable=exc.is_retryable) from exc
        latency_ms = (time.perf_counter() - started) * 1000

        usage = attempt.raw.usage_metadata or {}
        tokens_in = usage.get("input_tokens", 0)
        tokens_out = usage.get("output_tokens", 0)
        return LLMResult(
            output=cast(T, attempt.parsed),
            usage=LLMUsage(
                prompt=prompt.ref,
                model=attempt.model,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost_usd(attempt.model, tokens_in=tokens_in, tokens_out=tokens_out),
                latency_ms=latency_ms,
            ),
        )

    def _branch(
        self, model: str, temperature: float, schema: type[BaseModel]
    ) -> Runnable[Any, _Attempt]:
        """model → structured output (with raw message) → validate, retried as one unit."""
        llm = self._factory(model, temperature)
        structured = llm.with_structured_output(schema, include_raw=True)
        unwrap = RunnableLambda(partial(_unwrap, model=model, schema=schema))
        return structured.pipe(unwrap).with_retry(
            retry_if_exception_type=TRANSIENT_ERRORS,
            stop_after_attempt=self._retry_attempts,
            wait_exponential_jitter=self._retry_backoff,
        )


def _unwrap(result: dict[str, Any], *, model: str, schema: type[BaseModel]) -> _Attempt:
    parsed, raw, error = result.get("parsed"), result.get("raw"), result.get("parsing_error")
    if error is not None or not isinstance(parsed, schema):
        reason = error or f"expected {schema.__name__}, got {type(parsed).__name__}"
        raise StructuredOutputError(f"{model}: output did not match schema: {reason}")
    if not isinstance(raw, AIMessage):
        raise LLMError(f"{model}: unexpected raw response type {type(raw).__name__}")
    return _Attempt(parsed=parsed, raw=raw, model=model)
