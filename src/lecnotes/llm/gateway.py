"""The only place that calls an LLM (plan §4.3).

Every call goes through `LLMGateway.generate_structured`, which guarantees it is:
  - validated  — output parsed into the requested Pydantic schema, or a typed error;
  - rate-limited — per-model token bucket sized to the free tier;
  - retried    — only on errors that can succeed on a second attempt;
  - failed over — to the next API key when a key is invalid or out of quota (same model
                 first, so quality doesn't drop), then to the next model in the list;
  - traced     — Langfuse generation nested under the current stage span;
  - costed     — tokens + list-price cost returned in `LLMResult.usage`.

The chat model is built by an injected factory, so tests swap in `adapters/llm/fake.py`
and exercise the exact same retry / fallback / tracing path offline.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, NoReturn, cast

import httpx
from google.genai.types import MediaResolution
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    BaseCallbackHandler,
    CallbackManagerForLLMRun,
)
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAPIError,
    ModelConnectionError,
    ModelError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, SecretStr

from lecnotes.config import Settings
from lecnotes.domain import LLMUsage
from lecnotes.errors import LLMError, OutputTruncated, QuotaExceeded, StructuredOutputError
from lecnotes.llm.keys import KeyPool, key_problem
from lecnotes.llm.models import get_model
from lecnotes.llm.pricing import cost_usd
from lecnotes.llm.rate_limit import rate_limiter, requests_per_minute
from lecnotes.observability.logging import get_logger
from lecnotes.observability.tracing import langchain_callbacks
from lecnotes.prompts.loader import RenderedPrompt

log = get_logger(__name__)


@dataclass(frozen=True)
class ModelOptions:
    """Per-call model settings beyond the prompt's model + temperature. None = model default."""

    media_resolution: Literal["low", "medium", "high"] | None = None
    max_output_tokens: int | None = None
    thinking_budget: int | None = None
    """Gemini 2.5 thinking control (tokens)."""
    thinking_level: Literal["minimal", "low", "medium", "high"] | None = None
    """Gemini 3+ thinking control; replaces thinking_budget there."""
    timeout_s: float | None = None


ModelFactory = Callable[[str, float, ModelOptions, int], BaseChatModel]
"""(model name, temperature, options, API-key index) -> chat model."""

_MEDIA_RESOLUTION = {
    "low": MediaResolution.MEDIA_RESOLUTION_LOW,
    "medium": MediaResolution.MEDIA_RESOLUTION_MEDIUM,
    "high": MediaResolution.MEDIA_RESOLUTION_HIGH,
}

# Errors where another attempt (same model after backoff, or the fallback model) may succeed.
# httpx.TransportError: the SDK sometimes surfaces raw timeouts / connection errors that
# LangChain doesn't classify (seen: a 180 s ReadTimeout when Gemini stalled once).
TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    ModelRateLimitError,
    ModelAPIError,
    ModelConnectionError,
    ModelTimeoutError,
    StructuredOutputError,
    httpx.TransportError,
)


@dataclass(frozen=True)
class LLMResult[T: BaseModel]:
    output: T
    usage: LLMUsage


@dataclass(frozen=True)
class _Attempt:
    parsed: BaseModel
    raw: AIMessage
    model: str


class _GeminiChat(ChatGoogleGenerativeAI):
    """Workaround: langchain-google-genai 4.4.2 accepts a model-level `media_resolution` but
    never puts it in the request's generationConfig (it only appears in identifying params).
    Found by the respx wire test. Injected per call via the supported `generation_config`
    kwarg; `setdefault` keeps this harmless once upstream sends it itself."""

    def _with_media_resolution(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        if self.media_resolution is None:
            return kwargs
        generation_config = dict(kwargs.get("generation_config") or {})
        generation_config.setdefault("media_resolution", self.media_resolution)
        return {**kwargs, "generation_config": generation_config}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return super()._generate(messages, stop, run_manager, **self._with_media_resolution(kwargs))

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return await super()._agenerate(
            messages, stop, run_manager, **self._with_media_resolution(kwargs)
        )


def gemini_factory(settings: Settings) -> ModelFactory:
    """Production factory: Gemini via LangChain, with SDK retries off (we retry one level up)."""

    keys = settings.gemini_keys

    def build(model: str, temperature: float, options: ModelOptions, key: int) -> BaseChatModel:
        rpm = requests_per_minute(model, settings.llm_rpm_override)
        resolution = options.media_resolution
        # Field aliases (retries / request_timeout / max_tokens): pyright types a pydantic
        # subclass's constructor from aliases.
        return _GeminiChat(
            model=model,
            temperature=temperature,
            api_key=SecretStr(keys[key]),
            rate_limiter=rate_limiter(model, rpm),  # per model, shared by every key
            # The SDK defaults to 6 internal retries; combined with ours that would be 18
            # requests per failure. 1 = "no SDK retries" (0 means "SDK default").
            retries=1,
            request_timeout=options.timeout_s or settings.llm_timeout_s,
            # Model-level (per-part media_resolution is ignored on Gemini 2.5). Actually
            # sent thanks to _GeminiChat; verified on the wire by test_perceive_adapter.
            media_resolution=_MEDIA_RESOLUTION[resolution] if resolution else None,
            max_tokens=options.max_output_tokens,
            thinking_budget=options.thinking_budget,
            thinking_level=options.thinking_level,
        )

    return build


class LLMGateway:
    def __init__(
        self,
        model_factory: ModelFactory,
        *,
        default_model: str,
        fallback_models: Sequence[str] = (),
        keys: int = 1,
        retry_attempts: int = 3,
        retry_backoff: bool = True,
        callbacks: Callable[[], list[BaseCallbackHandler]] = langchain_callbacks,
        key_pool: KeyPool | None = None,
    ) -> None:
        for name in (default_model, *fallback_models):
            get_model(name)  # fail at startup on a typo, not on the first call
        self._factory = model_factory
        self._default_model = default_model
        self._fallbacks = [m for m in dict.fromkeys(fallback_models) if m != default_model]
        self._keys = key_pool or KeyPool(keys)
        self._retry_attempts = retry_attempts
        self._retry_backoff = retry_backoff
        self._callbacks = callbacks

    @classmethod
    def from_settings(cls, settings: Settings) -> LLMGateway:
        return cls(
            gemini_factory(settings),
            default_model=settings.text_model,
            fallback_models=settings.fallback_models,
            keys=len(settings.gemini_keys),
            retry_attempts=settings.llm_retry_attempts,
        )

    async def generate_structured[T: BaseModel](
        self,
        prompt: RenderedPrompt,
        schema: type[T],
        *,
        model: str | None = None,
        options: ModelOptions | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LLMResult[T]:
        """Model precedence: the prompt file's pinned model (part of the prompt version, so
        reproducible) > `model` (caller config, e.g. settings.perceive_model) > default."""
        primary = prompt.spec.model or model or self._default_model
        get_model(primary)
        opts = options or ModelOptions()
        # A prompt that pins its model is reproducible by design: no model fallback for it.
        models = [primary] if prompt.spec.model else [primary, *self._fallbacks]
        models = list(dict.fromkeys(models))
        base_metadata = {
            "prompt_id": prompt.ref.id,
            "prompt_version": prompt.ref.version,
            "response_schema": schema.__name__,
            **(metadata or {}),
        }

        started = time.perf_counter()
        attempt, last_error = None, None
        for name in models:
            keys = self._keys.available(name)
            for key in keys:
                config: RunnableConfig = {
                    "callbacks": self._callbacks(),
                    "run_name": f"llm:{prompt.ref}",
                    "metadata": base_metadata | {"api_key_index": key},
                }
                chain = self._branch(name, key, prompt, opts, schema)
                try:
                    attempt = await chain.ainvoke(prompt.messages, config=config)
                    break
                except OutputTruncated:
                    raise  # another key or model would truncate the same way
                except Exception as exc:
                    last_error = exc
                    problem = key_problem(exc)
                    if problem is not None:  # this key is broken or out of quota: next key
                        self._keys.bench(key, name, problem)
                        log.warning(
                            "llm.key_benched", model=name, key=key, seconds=problem.seconds,
                            all_models=problem.all_models, error=type(exc).__name__,
                        )  # fmt: skip
                        continue
                    if isinstance(exc, TRANSIENT_ERRORS):
                        break  # not the key's fault and retries are used up: next model
                    _raise_mapped(prompt, exc)
            if attempt is not None:
                break
        if attempt is None:
            if last_error is None:
                raise QuotaExceeded(f"{prompt.ref}: every API key is benched for {models}")
            _raise_mapped(prompt, last_error)
        latency_ms = (time.perf_counter() - started) * 1000

        usage = attempt.raw.usage_metadata or {}
        tokens_in = usage.get("input_tokens", 0)
        tokens_out = usage.get("output_tokens", 0)
        return LLMResult(
            output=cast(T, attempt.parsed),
            usage=LLMUsage(
                prompt_id=prompt.ref.id,
                prompt_version=prompt.ref.version,
                model=attempt.model,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost_usd(attempt.model, tokens_in=tokens_in, tokens_out=tokens_out),
                latency_ms=latency_ms,
            ),
        )

    def _branch(
        self,
        model: str,
        key: int,
        prompt: RenderedPrompt,
        options: ModelOptions,
        schema: type[BaseModel],
    ) -> Runnable[Any, _Attempt]:
        """model → structured output (with raw message) → validate, retried as one unit.
        With several keys, a 429 isn't retried on the same key: the next key is used instead."""
        llm = self._factory(model, prompt.spec.temperature, options, key)
        llm.name = f"{model}:{prompt.ref}"  # the Langfuse generation's name; llm is per-call
        structured = llm.with_structured_output(schema, include_raw=True)
        unwrap = RunnableLambda(partial(_unwrap, model=model, schema=schema))
        retry_on = (
            TRANSIENT_ERRORS
            if self._keys.size == 1
            else tuple(e for e in TRANSIENT_ERRORS if e is not ModelRateLimitError)
        )
        return structured.pipe(unwrap).with_retry(
            retry_if_exception_type=retry_on,
            stop_after_attempt=self._retry_attempts,
            wait_exponential_jitter=self._retry_backoff,
        )


def _unwrap(result: dict[str, Any], *, model: str, schema: type[BaseModel]) -> _Attempt:
    parsed, raw, error = result.get("parsed"), result.get("raw"), result.get("parsing_error")
    if isinstance(raw, AIMessage) and raw.response_metadata.get("finish_reason") == "MAX_TOKENS":
        # Checked before parsing: cut-off JSON would otherwise look like a retryable
        # schema error and re-send the (expensive) request for the same truncated result.
        raise OutputTruncated(f"{model}: output hit the max output token limit")
    if error is not None or not isinstance(parsed, schema):
        reason = error or f"expected {schema.__name__}, got {type(parsed).__name__}"
        raise StructuredOutputError(f"{model}: output did not match schema: {reason}")
    if not isinstance(raw, AIMessage):
        raise LLMError(f"{model}: unexpected raw response type {type(raw).__name__}")
    return _Attempt(parsed=parsed, raw=raw, model=model)


def _mapped(prompt: RenderedPrompt, exc: BaseException) -> Exception:
    """A provider/LangChain error as one of our typed errors (what the job runner understands);
    anything else unchanged."""
    if isinstance(exc, (StructuredOutputError, OutputTruncated, LLMError)):
        return exc
    if isinstance(exc, ModelRateLimitError) or key_problem(exc) is not None:
        return QuotaExceeded(f"{prompt.ref}: every API key is out of quota or invalid: {exc}")
    if isinstance(exc, ContextOverflowError):
        return LLMError(f"{prompt.ref}: input exceeds context window: {exc}")
    if isinstance(exc, ModelError):
        return LLMError(f"{prompt.ref}: {exc}", retryable=exc.is_retryable)
    if isinstance(exc, httpx.TransportError):
        return LLMError(f"{prompt.ref}: {type(exc).__name__}: {exc}", retryable=True)
    if isinstance(exc, Exception):
        return exc  # not a provider error (e.g. a bug in our code): don't disguise it
    return LLMError(f"{prompt.ref}: {type(exc).__name__}: {exc}")


def _raise_mapped(prompt: RenderedPrompt, exc: BaseException) -> NoReturn:
    mapped = _mapped(prompt, exc)
    if mapped is exc:
        raise exc  # unchanged: `raise exc from exc` would make it its own cause (a cycle)
    raise mapped from exc
