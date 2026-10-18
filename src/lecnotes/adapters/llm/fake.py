"""Scripted chat model for tests: canned Pydantic outputs keyed by prompt ID, no network.

    fake = FakeLLM({"extract": ExtractOutput(...)})          # same answer every call
    fake = FakeLLM({"extract": [rate_limit_error, answer]})  # consumed in order
    gateway = LLMGateway(fake.factory, default_model="gemini-2.5-flash")

`FakeChatModel` is a real `BaseChatModel`, so LangChain callbacks (Langfuse), retries and
fallbacks run exactly as in production. The prompt ID is read from the run metadata the
gateway attaches to every call.

A scripted item is one of:
  - a Pydantic model → returned as JSON (parses into the requested schema);
  - a `str`          → returned verbatim (use it to simulate malformed output);
  - an `Exception`   → raised (simulate 429s, server errors, ...);
  - an `AIMessage`    → returned as-is (full control, e.g. finish_reason="MAX_TOKENS").
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, ConfigDict

from lecnotes.llm.gateway import ModelOptions

Scripted = BaseModel | str | Exception | AIMessage


@dataclass(frozen=True)
class FakeCall:
    prompt_id: str
    model: str
    options: ModelOptions
    messages: list[BaseMessage]
    key: int = 0


class FakeScriptExhausted(AssertionError):
    pass


class FakeLLM:
    def __init__(
        self,
        responses: Mapping[str, Scripted | Sequence[Scripted]],
        *,
        tokens_in: int = 100,
        tokens_out: int = 50,
    ) -> None:
        self._queues: dict[str, deque[Scripted]] = {}
        self._repeat: dict[str, Scripted] = {}
        for prompt_id, item in responses.items():
            if isinstance(item, Sequence) and not isinstance(item, str):
                self._queues[prompt_id] = deque(item)
            else:
                self._repeat[prompt_id] = item
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.calls: list[FakeCall] = []

    @classmethod
    def from_recordings(
        cls,
        recordings: Mapping[str, tuple[Path, type[BaseModel]]],
        **kwargs: Any,
    ) -> FakeLLM:
        """Load real outputs saved as JSON (e.g. by scripts/). Validated on load, so a schema
        change makes stale recordings fail loudly instead of drifting silently."""
        return cls(
            {
                prompt_id: schema.model_validate_json(path.read_text(encoding="utf-8"))
                for prompt_id, (path, schema) in recordings.items()
            },
            **kwargs,
        )

    def factory(
        self, model: str, temperature: float, options: ModelOptions, key: int = 0
    ) -> FakeChatModel:
        """Matches the gateway's `ModelFactory` signature."""
        return FakeChatModel(
            script=self, model_name=model, temperature=temperature, options=options, key=key
        )

    def next(self, prompt_id: str, model: FakeChatModel, messages: list[BaseMessage]) -> Scripted:
        self.calls.append(
            FakeCall(
                prompt_id=prompt_id,
                model=model.model_name,
                options=model.options,
                messages=messages,
                key=model.key,
            )
        )
        if prompt_id in self._repeat:
            return self._repeat[prompt_id]
        queue = self._queues.get(prompt_id)
        if not queue:
            raise FakeScriptExhausted(f"no scripted response left for prompt {prompt_id!r}")
        return queue.popleft()


class FakeChatModel(BaseChatModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    script: FakeLLM
    model_name: str
    temperature: float = 0.0
    options: ModelOptions = ModelOptions()
    key: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt_id = str(run_manager.metadata.get("prompt_id", "")) if run_manager else ""
        item = self.script.next(prompt_id, self, messages)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, AIMessage):
            return ChatResult(generations=[ChatGeneration(message=item)])
        content = item if isinstance(item, str) else item.model_dump_json()
        message = AIMessage(
            content=content,
            usage_metadata={
                "input_tokens": self.script.tokens_in,
                "output_tokens": self.script.tokens_out,
                "total_tokens": self.script.tokens_in + self.script.tokens_out,
            },
            response_metadata={"model_name": self.model_name},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def with_structured_output(  # type: ignore[override]
        self,
        schema: type[BaseModel],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Any]:
        """Mirror ChatGoogleGenerativeAI's json_schema mode: parse the message content."""
        parser = PydanticOutputParser(pydantic_object=schema)
        if not include_raw:
            return self | parser

        def parse(raw: AIMessage) -> dict[str, Any]:
            try:
                return {"raw": raw, "parsed": parser.invoke(raw), "parsing_error": None}
            except OutputParserException as exc:
                return {"raw": raw, "parsed": None, "parsing_error": exc}

        return self | RunnableLambda(parse)
