"""Versioned prompts stored as YAML in git.

Layout: `prompts/<id>/v<N>.yaml`, where `<id>` may be nested (e.g. `judge/faithfulness`).
Never edit a prompt file in place once it has been evaluated: add `v<N+1>.yaml` instead.
The (id, version) pair is part of every stage cache key and eval report.

    prompt = load("extract", version=1, subject_preset="stem", segments=[...], board_states=[...])
    prompt.messages  # [SystemMessage, HumanMessage] ready for the LLM gateway
    prompt.ref       # PromptRef(id="extract", version=1)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined, meta
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError

PROMPTS_DIR = Path(__file__).parent
_VERSION_FILE = re.compile(r"^v(\d+)\.yaml$")


class PromptError(Exception):
    pass


class PromptNotFound(PromptError):
    pass


class PromptRenderError(PromptError):
    pass


class PromptRef(BaseModel):
    """Identifies an exact prompt revision. Hashable; used in cache keys and trace metadata."""

    model_config = ConfigDict(frozen=True)

    id: str
    version: int = Field(ge=1)

    def __str__(self) -> str:
        return f"{self.id}@v{self.version}"


class PromptSpec(BaseModel):
    """Schema of a prompt YAML file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    version: int = Field(ge=1)
    model: str | None = Field(default=None, description="Override; None = configured default")
    temperature: float = Field(default=0.0, ge=0, le=2)
    response_schema: str = Field(description="Name of the Pydantic model the output must match")
    changelog: str
    system: str
    user: str

    @property
    def ref(self) -> PromptRef:
        return PromptRef(id=self.id, version=self.version)


@dataclass(frozen=True)
class RenderedPrompt:
    ref: PromptRef
    spec: PromptSpec
    messages: list[BaseMessage]


class PromptLoader:
    def __init__(self, root: Path = PROMPTS_DIR) -> None:
        self._root = root
        self._env = Environment(
            undefined=StrictUndefined,
            autoescape=False,  # output goes to an LLM, not a browser
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=False,
        )
        # Per-instance caches: specs and parsed templates are immutable once read.
        self._spec = lru_cache(maxsize=None)(self._read_spec)
        self._template = lru_cache(maxsize=None)(self._env.from_string)
        self._declared_in = lru_cache(maxsize=None)(self._find_variables)

    def versions(self, prompt_id: str) -> list[int]:
        folder = self._root / prompt_id
        if not folder.is_dir():
            return []
        return sorted(
            int(m.group(1)) for p in folder.iterdir() if (m := _VERSION_FILE.match(p.name))
        )

    def spec(self, prompt_id: str, version: int) -> PromptSpec:
        return self._spec(prompt_id, version)

    def load(self, prompt_id: str, /, *, version: int, **variables: Any) -> RenderedPrompt:
        spec = self.spec(prompt_id, version)
        system = self._render(spec, "system", variables)
        user = self._render(spec, "user", variables)
        unused = set(variables) - self._declared(spec, "system") - self._declared(spec, "user")
        if unused:
            raise PromptRenderError(f"{spec.ref}: unexpected variables {sorted(unused)}")
        return RenderedPrompt(
            ref=spec.ref,
            spec=spec,
            messages=[SystemMessage(content=system), HumanMessage(content=user)],
        )

    # --- internals ---------------------------------------------------------------------

    def _read_spec(self, prompt_id: str, version: int) -> PromptSpec:
        path = self._root / prompt_id / f"v{version}.yaml"
        if not path.is_file():
            known = self.versions(prompt_id)
            hint = f"known versions: {known}" if known else "unknown prompt id"
            raise PromptNotFound(f"{prompt_id}@v{version} not found ({hint})")
        try:
            spec = PromptSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        except (yaml.YAMLError, ValidationError) as exc:
            raise PromptError(f"{path}: invalid prompt file\n{exc}") from exc
        if (spec.id, spec.version) != (prompt_id, version):
            raise PromptError(
                f"{path}: header says {spec.ref} but path says {prompt_id}@v{version}"
            )
        return spec

    def _find_variables(self, source: str) -> frozenset[str]:
        return frozenset(meta.find_undeclared_variables(self._env.parse(source)))

    def _declared(self, spec: PromptSpec, field: str) -> frozenset[str]:
        return self._declared_in(getattr(spec, field))

    def _render(self, spec: PromptSpec, field: str, variables: dict[str, Any]) -> str:
        try:
            return self._template(getattr(spec, field)).render(**variables).strip()
        except Exception as exc:  # jinja raises UndefinedError, TypeError, etc.
            raise PromptRenderError(f"{spec.ref} ({field}): {exc}") from exc


_default = PromptLoader()


def load(prompt_id: str, /, *, version: int, **variables: Any) -> RenderedPrompt:
    """Render a prompt from the package's prompt directory."""
    return _default.load(prompt_id, version=version, **variables)
