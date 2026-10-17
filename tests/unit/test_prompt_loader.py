from pathlib import Path
from textwrap import dedent

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from lecnotes.domain import BoardState, BoardType, TranscriptSegment
from lecnotes.prompts.loader import (
    PromptError,
    PromptLoader,
    PromptNotFound,
    PromptRef,
    PromptRenderError,
    load,
)


def write_prompt(root: Path, prompt_id: str, version: int, body: str) -> None:
    folder = root / prompt_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"v{version}.yaml").write_text(dedent(body), encoding="utf-8")


GREET = """\
    id: {id}
    version: {version}
    temperature: 0.1
    response_schema: Greeting
    changelog: test
    system: |
      You greet {{{{ audience }}}}.
    user: |
      Say hi to {{{{ name }}}}.
"""


@pytest.fixture
def loader(tmp_path: Path) -> PromptLoader:
    write_prompt(tmp_path, "greet", 1, GREET.format(id="greet", version=1))
    write_prompt(tmp_path, "greet", 2, GREET.format(id="greet", version=2))
    write_prompt(
        tmp_path, "judge/faithfulness", 1, GREET.format(id="judge/faithfulness", version=1)
    )
    return PromptLoader(tmp_path)


def test_renders_messages_and_ref(loader: PromptLoader) -> None:
    prompt = loader.load("greet", version=2, audience="students", name="Ada")

    assert prompt.ref == PromptRef(id="greet", version=2)
    assert str(prompt.ref) == "greet@v2"
    system, user = prompt.messages
    assert isinstance(system, SystemMessage)
    assert isinstance(user, HumanMessage)
    assert system.content == "You greet students."
    assert user.content == "Say hi to Ada."
    assert prompt.spec.temperature == 0.1
    assert prompt.spec.model is None


def test_nested_prompt_ids(loader: PromptLoader) -> None:
    prompt = loader.load("judge/faithfulness", version=1, audience="a", name="b")
    assert prompt.ref.id == "judge/faithfulness"


def test_missing_variable_raises(loader: PromptLoader) -> None:
    with pytest.raises(PromptRenderError, match="'name' is undefined"):
        loader.load("greet", version=1, audience="students")


def test_unexpected_variable_raises(loader: PromptLoader) -> None:
    with pytest.raises(PromptRenderError, match="nmae"):
        loader.load("greet", version=1, audience="students", name="Ada", nmae="typo")


def test_unknown_prompt_id_raises(loader: PromptLoader) -> None:
    with pytest.raises(PromptNotFound, match="unknown prompt id"):
        loader.load("nope", version=1)


def test_unknown_version_lists_known_versions(loader: PromptLoader) -> None:
    with pytest.raises(PromptNotFound, match=r"known versions: \[1, 2\]"):
        loader.load("greet", version=9, audience="a", name="b")


def test_header_must_match_path(tmp_path: Path) -> None:
    # The classic mistake: copy v1.yaml to v2.yaml and forget to bump `version`.
    write_prompt(tmp_path, "greet", 2, GREET.format(id="greet", version=1))
    with pytest.raises(PromptError, match="header says greet@v1"):
        PromptLoader(tmp_path).load("greet", version=2, audience="a", name="b")


def test_invalid_file_raises(tmp_path: Path) -> None:
    write_prompt(tmp_path, "bad", 1, "id: bad\nversion: 1\n")  # missing required fields
    with pytest.raises(PromptError, match="invalid prompt file"):
        PromptLoader(tmp_path).load("bad", version=1)


def test_versions_listing(loader: PromptLoader) -> None:
    assert loader.versions("greet") == [1, 2]
    assert loader.versions("missing") == []


# --- real prompt files shipped in the package --------------------------------------------


def test_extract_v1_renders_with_domain_objects() -> None:
    board = BoardState(
        id="b1",
        t_first_seen=10,
        t_last_seen=60,
        type=BoardType.CODE,
        text="for i, x in enumerate(xs):",
    )
    seg = TranscriptSegment(id="s1", start_s=12, end_s=20, text="enumerate gives you the index")

    prompt = load("extract", version=1, subject_preset="stem", board_states=[board], segments=[seg])

    assert prompt.ref == PromptRef(id="extract", version=1)
    system, user = (str(m.content) for m in prompt.messages)
    assert "STEM lecture" in system
    assert "theory lecture" not in system
    assert "[b1] (code, 10.0s-60.0s)" in user
    assert "for i, x in enumerate(xs):" in user
    assert "[s1] (12.0s-20.0s) enumerate gives you the index" in user
