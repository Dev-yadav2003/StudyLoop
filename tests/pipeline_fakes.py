"""A complete pipeline with fakes only at the edges: static catalog, recorded perception,
scripted LLM. Shared by the runner and API tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from pydantic import SecretStr

from lecnotes.adapters.catalog.static import StaticCatalog
from lecnotes.adapters.llm.fake import FakeLLM, Scripted
from lecnotes.adapters.perception.fake import FakePerceiver
from lecnotes.adapters.perception.schema import PerceiveOutput, to_perception_result
from lecnotes.bootstrap import Services, assemble
from lecnotes.config import Settings
from lecnotes.domain import VideoMeta
from lecnotes.llm.gateway import LLMGateway
from lecnotes.pipeline.stages.extract import ExtractedCode, ExtractedUnit, ExtractOutput
from lecnotes.pipeline.stages.verify import VerifyOutput
from lecnotes.ports.cache import StageCache
from lecnotes.ports.metrics import StageRunSink

SAMPLE = Path(__file__).parent / "fixtures" / "perceive_output_sample.json"
VID = "dQw4w9WgXcQ"
URL = f"https://www.youtube.com/watch?v={VID}"
VIDEO = VideoMeta(video_id=VID, title="Python Iteration Tools", duration_s=754)
CODE = 'names = ["ada", "alan"]\nfor i, name in enumerate(names):\n    print(i, name)'


def make_settings(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> Settings:
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    secret = SecretStr("unused")
    fields: dict[str, object] = {
        "gemini_api_key": secret,
        "youtube_api_key": secret,
        "database_url": secret,
        "langfuse_public_key": secret,
        "langfuse_secret_key": secret,
    }
    return Settings(**(fields | overrides))  # type: ignore[arg-type]


def default_script() -> dict[str, Scripted | Sequence[Scripted]]:
    unit = ExtractedUnit(
        section="Iteration tools",
        concept="enumerate()",
        explanation="enumerate gives you the index and the item at the same time.",
        examples=[],
        board_text="",
        code=[ExtractedCode(language="python", source=CODE)],
        segment_ids=["s0002"],
        board_state_ids=["b0002"],
    )
    return {"extract": ExtractOutput(units=[unit]), "verify": VerifyOutput(results=[])}


def fake_services(
    settings: Settings,
    cache: StageCache,
    *,
    llm: FakeLLM | None = None,
    script: Mapping[str, Scripted | Sequence[Scripted]] | None = None,
    sinks: Sequence[StageRunSink] = (),
) -> tuple[Services, FakeLLM]:
    raw = PerceiveOutput.model_validate_json(SAMPLE.read_text(encoding="utf-8"))
    fake_llm = llm or FakeLLM(dict(script or default_script()))
    services = assemble(
        settings=settings,
        catalog=StaticCatalog([VIDEO]),
        perceiver=FakePerceiver({VID: to_perception_result(raw, VID).result}),
        gateway=LLMGateway(fake_llm.factory, default_model="gemini-2.5-flash", retry_backoff=False),
        cache=cache,
        sinks=sinks,
    )
    return services, fake_llm
