import pytest
from pydantic import BaseModel, SecretStr

from lecnotes.cache.keys import StageKey, canonical_json, config_subset, content_hash
from lecnotes.config import Settings
from lecnotes.domain import TranscriptSegment

PERCEIVE_CONFIG = ["perceive_fps", "perceive_media_resolution", "perceive_model"]


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    s = SecretStr("x")
    return Settings(
        gemini_api_key=s,
        youtube_api_key=s,
        database_url=s,
        langfuse_public_key=s,
        langfuse_secret_key=s,
    )


def key(settings: Settings, **overrides: object) -> StageKey:
    fields: dict[str, object] = {
        "stage": "perceive",
        "stage_version": 1,
        "video_id": "dQw4w9WgXcQ",
        "prompt_version": 1,
        "model": "gemini-2.5-flash",
        "input": {"video_id": "dQw4w9WgXcQ"},
        "config": config_subset(settings, PERCEIVE_CONFIG),
    }
    return StageKey.build(**(fields | overrides))  # type: ignore[arg-type]


# --- the four cases from the plan --------------------------------------------------------


def test_same_inputs_hit(settings: Settings) -> None:
    assert key(settings).digest == key(settings).digest


def test_stage_version_bump_misses(settings: Settings) -> None:
    assert key(settings, stage_version=2).digest != key(settings).digest


def test_prompt_version_change_misses(settings: Settings) -> None:
    assert key(settings, prompt_version=2).digest != key(settings).digest


def test_unrelated_config_change_hits(settings: Settings) -> None:
    unrelated = settings.model_copy(
        update={"chunk_max_s": 240, "text_model": "gemini-3-flash-preview"}
    )
    relevant = settings.model_copy(update={"perceive_fps": 2.0})

    def with_config(s: Settings) -> str:
        return key(s, config=config_subset(s, PERCEIVE_CONFIG)).digest

    assert with_config(unrelated) == key(settings).digest
    assert with_config(relevant) != key(settings).digest


# --- determinism and sensitivity ---------------------------------------------------------


def test_model_input_and_video_change_miss(settings: Settings) -> None:
    base = key(settings).digest
    assert key(settings, model="gemini-2.5-flash-lite").digest != base
    assert key(settings, input={"video_id": "other"}).digest != base
    assert key(settings, video_id="aaaaaaaaaaa").digest != base


def test_canonical_json_ignores_dict_order() -> None:
    a = canonical_json({"b": 1, "a": [1, {"y": 2, "x": 1}]})
    b = canonical_json({"a": [1, {"x": 1, "y": 2}], "b": 1})
    assert a == b


def test_pydantic_input_hashes_like_its_dump() -> None:
    seg = TranscriptSegment(id="s1", start_s=0, end_s=1, text="hi")
    assert content_hash(seg) == content_hash(seg.model_dump(mode="json"))
    assert content_hash([seg]) == content_hash([seg.model_dump(mode="json")])


def test_digest_is_sha256_hex(settings: Settings) -> None:
    digest = key(settings).digest
    assert len(digest) == 64
    int(digest, 16)


def test_config_subset_is_sorted_and_limited() -> None:
    class Cfg(BaseModel):
        b: int = 2
        a: int = 1
        c: int = 3

    assert list(config_subset(Cfg(), ["b", "a"])) == ["a", "b"]
