import json
from pathlib import Path

import pytest

from lecnotes.adapters.perception.schema import PerceiveOutput, to_perception_result
from lecnotes.domain import BoardType
from lecnotes.prompts.loader import PromptRef, PromptRenderError, load

FIXTURE = Path(__file__).parents[1] / "fixtures" / "perceive_output_sample.json"
VID = "dQw4w9WgXcQ"


def sample() -> PerceiveOutput:
    return PerceiveOutput.model_validate_json(FIXTURE.read_text(encoding="utf-8"))


def test_prompt_v1_renders() -> None:
    prompt = load("perceive", version=1, duration="12:30")
    assert prompt.ref == PromptRef(id="perceive", version=1)
    system, user = (str(m.content) for m in prompt.messages)
    assert "VERBATIM" in system
    assert "[unclear]" in system
    assert "(duration 12:30)" in user


def test_prompt_v1_without_duration() -> None:
    user = str(load("perceive", version=1, duration=None).messages[1].content)
    assert "duration" not in user


def test_prompt_v1_requires_duration_variable() -> None:
    with pytest.raises(PromptRenderError):
        load("perceive", version=1)


def test_schema_validates_fixture() -> None:
    raw = sample()
    assert len(raw.board_states) == 4
    assert len(raw.segments) == 7


def test_schema_puts_board_states_first() -> None:
    # Generation order = field order; board text must survive a truncated output.
    assert list(PerceiveOutput.model_json_schema()["properties"]) == ["board_states", "segments"]


def test_conversion_assigns_ids_and_repairs_or_drops() -> None:
    conv = to_perception_result(sample(), VID)
    result = conv.result

    assert [b.id for b in result.board_states] == ["b0001", "b0002", "b0003"]
    assert [s.id for s in result.segments] == ["s0001", "s0002", "s0003", "s0004"]

    code = result.board_states[1]
    assert code.type is BoardType.CODE
    assert (code.t_first_seen, code.t_last_seen) == (42, 130)
    assert "    print(i, name)" in code.text  # indentation preserved verbatim

    right = result.segments[2]
    assert (right.start_s, right.end_s) == (50, 51)  # zero-length repaired to 1 s
    assert result.segments[3].start_s == 133  # "2:13" parsed

    assert len(conv.repaired) == 1
    assert len(conv.dropped) == 4
    assert any("board_states[3]" in d for d in conv.dropped)
    assert any("empty text" in d for d in conv.dropped)
    assert any("'soon'" in d for d in conv.dropped)


def test_conversion_of_empty_output() -> None:
    conv = to_perception_result(PerceiveOutput(board_states=[], segments=[]), VID)
    assert conv.result.segments == []
    assert conv.dropped == []


def test_fixture_is_plausible_model_output() -> None:
    # Guard against the fixture drifting from what the schema allows.
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert set(data) == set(PerceiveOutput.model_fields)
