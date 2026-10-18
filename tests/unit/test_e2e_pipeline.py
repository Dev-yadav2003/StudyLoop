"""Milestone B, offline: one link → notes through every real stage and the orchestrator.

Only the edges are fakes: a static catalog, recorded perception, a scripted LLM.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import SecretStr

from lecnotes.adapters.cache.memory import InMemoryStageCache
from lecnotes.adapters.catalog.static import StaticCatalog
from lecnotes.adapters.llm.fake import FakeLLM
from lecnotes.adapters.perception.fake import FakePerceiver
from lecnotes.adapters.perception.schema import PerceiveOutput, to_perception_result
from lecnotes.bootstrap import Services, assemble
from lecnotes.cli import ServicesFactory, main
from lecnotes.config import Settings
from lecnotes.domain import VideoMeta
from lecnotes.llm.gateway import LLMGateway
from lecnotes.pipeline.registry import VIDEO_STAGE_ORDER
from lecnotes.pipeline.stages.extract import ExtractedCode, ExtractedUnit, ExtractOutput
from lecnotes.pipeline.stages.verify import UnitVerdict, VerifyOutput

FIXTURE = Path(__file__).parents[1] / "fixtures" / "perceive_output_sample.json"
VID = "dQw4w9WgXcQ"
URL = f"https://youtu.be/{VID}"
VIDEO = VideoMeta(video_id=VID, title="Python Iteration Tools", duration_s=754)
CODE = 'names = ["ada", "alan"]\nfor i, name in enumerate(names):\n    print(i, name)'
FLASH = "gemini-2.5-flash"


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    secret = SecretStr("unused")
    return Settings(
        gemini_api_key=secret,
        youtube_api_key=secret,
        database_url=secret,
        langfuse_public_key=secret,
        langfuse_secret_key=secret,
    )


def recorded_perception() -> FakePerceiver:
    raw = PerceiveOutput.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    return FakePerceiver({VID: to_perception_result(raw, VID).result})


def scripted_llm() -> FakeLLM:
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
    verified = UnitVerdict(
        unit_id=f"{VID}-c001-u01",
        verdict="supported",
        revised_explanation="",
        supported_examples=[],
        unsupported_claims=[],
    )
    return FakeLLM(
        {"extract": ExtractOutput(units=[unit]), "verify": VerifyOutput(results=[verified])}
    )


def services(settings: Settings, llm: FakeLLM) -> Services:
    return assemble(
        settings=settings,
        catalog=StaticCatalog([VIDEO]),
        perceiver=recorded_perception(),
        gateway=LLMGateway(llm.factory, default_model=FLASH, retry_backoff=False),
        cache=InMemoryStageCache(),
    )


async def test_one_link_to_notes_then_cache_then_rerun(settings: Settings) -> None:
    llm = scripted_llm()
    svc = services(settings, llm)

    doc = await svc.pipeline.run(URL, job_id=uuid4())

    assert doc.title == VIDEO.title
    assert [u.concept for u in doc.units] == ["enumerate()"]
    assert "### enumerate()\n\n<sub>▶ [00:42](https://youtu.be/dQw4w9WgXcQ?t=42)" in doc.markdown
    assert "print(i, name)" in doc.markdown
    first = svc.runs.runs
    assert [r.stage for r in first] == ["resolve", *VIDEO_STAGE_ORDER]
    assert not any(r.cache_hit for r in first)
    extract_calls = len(llm.calls)
    assert extract_calls >= 1

    # Second run: every cacheable stage is a hit, no new LLM calls.
    svc.runs.runs.clear()
    again = await svc.pipeline.run(URL, job_id=uuid4())
    assert again == doc
    assert [r.stage for r in svc.runs.runs if not r.cache_hit] == ["resolve"]  # never cached
    assert len(llm.calls) == extract_calls

    # rerun_from=extract recomputes extract and render only.
    svc.runs.runs.clear()
    await svc.pipeline.run(URL, job_id=uuid4(), rerun_from="extract")
    assert [r.stage for r in svc.runs.runs if not r.cache_hit] == [
        "resolve",
        "extract",
        "merge",
        "verify",
        "render",
    ]
    assert len(llm.calls) == 2 * extract_calls


def cli_factory(svc: Services) -> ServicesFactory:
    @asynccontextmanager
    async def factory(settings: Settings, use_db: bool) -> AsyncIterator[Services]:
        yield svc

    return factory


def test_cli_writes_notes_and_prints_stage_summary(
    settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    svc = services(settings, scripted_llm())

    code = main(
        ["run", URL, "--out", str(tmp_path), "--no-db", "--preset", "stem"],
        settings=settings,
        services=cli_factory(svc),
    )

    assert code == 0
    notes = (tmp_path / f"{VID}.md").read_text(encoding="utf-8")
    assert notes.startswith("# Python Iteration Tools")
    out = capsys.readouterr().out
    for stage in ["resolve", *VIDEO_STAGE_ORDER]:
        assert stage in out
    assert "1 units" in out


def test_cli_reports_user_facing_failure(
    settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    svc = services(settings, scripted_llm())
    code = main(
        ["run", "https://youtu.be/aaaaaaaaaaa", "--out", str(tmp_path), "--no-db"],
        settings=settings,
        services=cli_factory(svc),
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "Failed in stage 'resolve'" in err
    assert "Only public videos are supported" in err
    assert not list(tmp_path.iterdir())
