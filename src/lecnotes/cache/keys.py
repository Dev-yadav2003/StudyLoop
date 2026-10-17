"""Content-addressed cache keys for whole stage outputs.

    key = sha256(video_id, stage, stage_version, prompt_version, model, input_hash, config_hash)

Invalidation is by construction: bump a stage or prompt version, change the model, the
input, or a setting the stage *declares* relevant (`Stage.config`) → new key → miss.
Settings a stage does not depend on are never hashed, so changing them keeps the hit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from pydantic import BaseModel

KEY_FORMAT = 1  # bump if the key recipe itself changes (invalidates everything)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def canonical_json(value: Any) -> bytes:
    """Deterministic JSON: sorted keys, no whitespace. Pydantic models are dumped first."""
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class StageKey:
    stage: str
    stage_version: int
    video_id: str | None
    prompt_version: int | None
    model: str | None
    input_hash: str
    config_hash: str

    @classmethod
    def build(
        cls,
        *,
        stage: str,
        stage_version: int,
        input: Any,
        config: Any = None,
        video_id: str | None = None,
        prompt_version: int | None = None,
        model: str | None = None,
    ) -> StageKey:
        return cls(
            stage=stage,
            stage_version=stage_version,
            video_id=video_id,
            prompt_version=prompt_version,
            model=model,
            input_hash=content_hash(input),
            config_hash=content_hash(config if config is not None else {}),
        )

    @property
    def digest(self) -> str:
        return content_hash({"format": KEY_FORMAT, **asdict(self)})
