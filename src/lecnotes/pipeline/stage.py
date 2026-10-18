"""Stage contract: one stage = one file = one responsibility.

A stage is a typed function `In -> Out` (both Pydantic models) plus the metadata that makes
its output cacheable and measurable. It never calls another stage, never touches the cache,
spans or metrics — the orchestrator does that.

Bump `version` on any logic change. Prompt changes go in a new prompt file (its version is
part of the cache key through `prompt_version`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel

from lecnotes.pipeline.context import RunContext


class Stage[In: BaseModel, Out: BaseModel](ABC):
    name: ClassVar[str]
    version: ClassVar[int]
    input_model: ClassVar[type[BaseModel]]
    output_model: ClassVar[type[BaseModel]]
    cacheable: ClassVar[bool] = True

    @property
    def prompt_version(self) -> int | None:
        return None

    @property
    def model(self) -> str | None:
        return None

    def config(self, ctx: RunContext) -> dict[str, Any]:
        """Settings/options that change this stage's output, and nothing else (cache key)."""
        return {}

    @abstractmethod
    async def run(self, inp: In, ctx: RunContext) -> Out: ...
