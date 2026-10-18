from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from lecnotes.adapters.cache.memory import InMemoryStageCache
from lecnotes.adapters.llm.fake import FakeLLM
from lecnotes.domain import (
    AlignedChunk,
    BoardState,
    BoardType,
    Chunks,
    ExtractedUnits,
    JobOptions,
    SubjectPreset,
    TranscriptSegment,
    VideoMeta,
)
from lecnotes.llm.gateway import LLMGateway
from lecnotes.observability.metrics import InMemoryStageRunSink
from lecnotes.pipeline.context import RunContext
from lecnotes.pipeline.orchestrator import Orchestrator
from lecnotes.pipeline.stages.extract import (
    ExtractedCode,
    ExtractedUnit,
    ExtractOutput,
    ExtractStage,
    appears_on,
)

VID = "dQw4w9WgXcQ"
VIDEO = VideoMeta(video_id=VID, title="Python iteration", duration_s=600)
FLASH = "gemini-2.5-flash"
CODE = "for i, name in enumerate(names):\n    print(i, name)"

CHUNK = AlignedChunk(
    id="c001",
    video_id=VID,
    start_s=0,
    end_s=180,
    segments=[
        TranscriptSegment(id="s1", start_s=40, end_s=48, text="enumerate gives index and item"),
        TranscriptSegment(id="s2", start_s=50, end_s=58, text="it starts at zero by default"),
        TranscriptSegment(id="s3", start_s=120, end_s=125, text="see you next week"),
    ],
    board_states=[
        BoardState(id="b1", t_first_seen=30, t_last_seen=150, type=BoardType.CODE, text=CODE)
    ],
)


def unit(**overrides: object) -> ExtractedUnit:
    fields: dict[str, object] = {
        "section": "Iteration tools",
        "concept": "enumerate()",
        "explanation": "Yields (index, item) pairs, starting at zero.",
        "examples": ["for i, name in enumerate(names)"],
        "board_text": "",
        "code": [ExtractedCode(language="python", source=CODE)],
        "segment_ids": ["s1", "s2"],
        "board_state_ids": ["b1"],
    }
    return ExtractedUnit.model_validate(fields | overrides)


def ctx(preset: SubjectPreset = SubjectPreset.STEM) -> RunContext:
    return RunContext(job_id=uuid4(), video=VIDEO, options=JobOptions(subject_preset=preset))


async def extract(
    output: ExtractOutput, chunks: list[AlignedChunk] | None = None
) -> tuple[ExtractedUnits, FakeLLM]:
    fake = FakeLLM({"extract": output})
    stage = ExtractStage(
        LLMGateway(fake.factory, default_model=FLASH, retry_backoff=False), model=FLASH
    )
    return await stage.run(Chunks(chunks=chunks or [CHUNK]), ctx()), fake


async def test_unit_with_span_computed_from_cited_speech() -> None:
    out, _ = await extract(ExtractOutput(units=[unit()]))

    [u] = out.units
    assert u.id == f"{VID}-c001-u01"
    assert (u.start_s, u.end_s) == (40, 58)  # from s1..s2, not from the model
    assert u.section == "Iteration tools"
    assert [c.source for c in u.code] == [CODE]
    assert u.code[0].language == "python"
    assert u.evidence[0].segment_ids == ["s1", "s2"]
    assert out.dropped == []


async def test_board_only_unit_spans_the_board() -> None:
    out, _ = await extract(ExtractOutput(units=[unit(segment_ids=[])]))
    assert (out.units[0].start_s, out.units[0].end_s) == (30, 150)


async def test_unknown_ids_removed_and_unit_without_evidence_dropped() -> None:
    out, _ = await extract(
        ExtractOutput(
            units=[
                unit(segment_ids=["s1", "s99"]),
                unit(concept="Invented", segment_ids=["s77"], board_state_ids=["b42"]),
            ]
        )
    )
    assert [u.concept for u in out.units] == ["enumerate()"]
    assert out.units[0].evidence[0].segment_ids == ["s1"]
    assert any("s99" in d for d in out.dropped)
    assert any("Invented" in d and "no valid evidence" in d for d in out.dropped)


async def test_code_not_on_the_board_is_removed() -> None:
    invented = ExtractedCode(language="python", source="list(zip(a, b))")
    reindented = ExtractedCode(
        language="", source="for i, name in enumerate(names): print(i, name)"
    )
    out, _ = await extract(ExtractOutput(units=[unit(code=[invented, reindented])]))

    [u] = out.units
    assert [c.source for c in u.code] == [reindented.source]  # whitespace-insensitive match
    assert u.code[0].language is None
    assert any("code not found" in d for d in out.dropped)


async def test_code_output_is_kept_apart_and_checked_on_the_board() -> None:
    """Real slide: code with its printed output underneath. v3 glued the output into the
    code (5 of 11 blocks didn't parse); now it is a separate, board-checked field."""
    shown = f"{CODE}\n0 tom\n1 jerry"
    chunk = CHUNK.model_copy(
        update={
            "board_states": [
                BoardState(
                    id="b1", t_first_seen=30, t_last_seen=150, type=BoardType.CODE, text=shown
                )
            ]
        }
    )
    good = ExtractedCode(language="python", source=CODE, output="0 tom\n1 jerry")
    invented = ExtractedCode(language="python", source=CODE, output="0 ada")
    out, _ = await extract(ExtractOutput(units=[unit(code=[good])]), [chunk])
    assert [(c.source, c.output) for c in out.units[0].code] == [(CODE, "0 tom\n1 jerry")]

    out, _ = await extract(ExtractOutput(units=[unit(code=[invented])]), [chunk])
    assert out.units[0].code[0].output is None  # code kept, invented output dropped
    assert any("output not found" in d for d in out.dropped)


async def test_same_code_shown_twice_is_one_block_with_its_output() -> None:
    shown = f"{CODE}\n0 tom"
    chunk = CHUNK.model_copy(
        update={
            "board_states": [
                BoardState(
                    id="b1", t_first_seen=30, t_last_seen=150, type=BoardType.CODE, text=shown
                )
            ]
        }
    )
    blocks = [
        ExtractedCode(language="python", source=CODE),
        ExtractedCode(language="python", source=CODE, output="0 tom"),
    ]
    out, _ = await extract(ExtractOutput(units=[unit(code=blocks)]), [chunk])
    assert [(c.source, c.output) for c in out.units[0].code] == [(CODE, "0 tom")]


async def test_board_text_must_be_verbatim_from_a_cited_board() -> None:
    out, _ = await extract(
        ExtractOutput(
            units=[
                unit(board_text="for i, name in enumerate(names):"),
                unit(concept="Other", board_text="while True: pass"),
            ]
        )
    )
    assert out.units[0].board_text == "for i, name in enumerate(names):"
    assert out.units[1].board_text is None


async def test_board_text_cited_from_an_uncited_board_is_removed() -> None:
    out, _ = await extract(ExtractOutput(units=[unit(board_text=CODE, board_state_ids=[])]))
    assert out.units[0].board_text is None
    assert out.units[0].code == []  # code must come from a *cited* board


async def test_one_llm_call_per_chunk_with_usage_reported() -> None:
    second = CHUNK.model_copy(update={"id": "c002"})
    fake = FakeLLM({"extract": ExtractOutput(units=[unit()])}, tokens_in=2000, tokens_out=300)
    stage = ExtractStage(
        LLMGateway(fake.factory, default_model=FLASH, retry_backoff=False), model=FLASH
    )
    c = ctx()
    out = await stage.run(Chunks(chunks=[CHUNK, second]), c)

    assert [u.id for u in out.units] == [f"{VID}-c001-u01", f"{VID}-c002-u01"]
    assert len(fake.calls) == 2
    assert sum(u.tokens_in for u in c.take_usage()) == 4000


async def test_prompt_carries_chunk_evidence_and_preset() -> None:
    _, fake = await extract(ExtractOutput(units=[]))
    system, user = fake.calls[0].messages
    assert isinstance(system, SystemMessage)
    assert isinstance(user, HumanMessage)
    assert "STEM lecture" in str(system.content)
    content = str(user.content)
    assert "Lecture: Python iteration" in content
    assert "[b1] code, visible 30s-150s" in content
    assert "print(i, name)" in content
    assert "[s3] 120s: see you next week" in content


async def test_subject_preset_is_part_of_the_cache_key() -> None:
    fake = FakeLLM({"extract": ExtractOutput(units=[unit()])})
    stage = ExtractStage(
        LLMGateway(fake.factory, default_model=FLASH, retry_backoff=False), model=FLASH
    )
    orch = Orchestrator(InMemoryStageCache(), InMemoryStageRunSink())
    chunks = Chunks(chunks=[CHUNK])

    await orch.run_stage(stage, chunks, ctx(SubjectPreset.STEM))
    await orch.run_stage(stage, chunks, ctx(SubjectPreset.STEM))
    await orch.run_stage(stage, chunks, ctx(SubjectPreset.THEORY))

    assert len(fake.calls) == 2


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("print(i, name)", True),
        ("for i, name in enumerate(names): print(i, name)", True),
        ("for i, nme in enumerate(names):", True),  # one OCR-level slip tolerated
        ("import numpy as np", False),
        ("", False),
    ],
)
def test_appears_on(text: str, expected: bool) -> None:
    assert appears_on(text, CHUNK.board_states) is expected


async def test_early_mention_does_not_drag_the_link_back() -> None:
    """Regression for the first real run: an intro sentence at 00:26 ("today: zip and
    enumerate") was cited together with the teaching at ~03:29, so the link pointed at 00:26."""
    chunk = AlignedChunk(
        id="c001",
        video_id=VID,
        start_s=0,
        end_s=300,
        segments=[
            TranscriptSegment(id="s_intro", start_s=26, end_s=31, text="today zip and enumerate"),
            TranscriptSegment(id="s1", start_s=209, end_s=220, text="wrap names in enumerate"),
            TranscriptSegment(id="s2", start_s=221, end_s=240, text="it gives index and item"),
        ],
        board_states=CHUNK.board_states,
    )
    out, _ = await extract(
        ExtractOutput(units=[unit(segment_ids=["s_intro", "s1", "s2"], code=[])]), [chunk]
    )
    [u] = out.units
    assert [(s.start_s, s.end_s) for s in u.spans] == [(209, 240), (26, 31)]
    assert u.start_s == 209  # primary moment drives ordering and the main link
