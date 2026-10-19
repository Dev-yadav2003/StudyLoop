import pytest

from lecnotes.config import ConfigError, Settings, load_settings

REQUIRED = {
    "GEMINI_API_KEY": "g-key",
    "YOUTUBE_API_KEY": "yt-key",
    "DATABASE_URL": "postgresql+psycopg://u:p@localhost/db",
    "LANGFUSE_PUBLIC_KEY": "pk-lf",
    "LANGFUSE_SECRET_KEY": "sk-lf",
}


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip any real env vars and ignore .env so tests are hermetic."""
    for name in [*REQUIRED, "ENV", "PERCEIVE_FPS", "CHUNK_MIN_S", "CHUNK_MAX_S", "FEATURES__MERGE"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", None)


@pytest.fixture
def required_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in REQUIRED.items():
        monkeypatch.setenv(name, value)


@pytest.mark.usefixtures("required_env")
def test_defaults_load() -> None:
    s = load_settings()
    assert s.env == "dev"
    assert s.perceive_model == "gemini-2.5-flash"
    assert s.perceive_media_resolution == "low"
    assert s.chunk_min_s == 120
    assert s.chunk_max_s == 300
    assert s.features.verify is True
    assert s.features.merge is True
    assert s.gemini_keys == ["g-key"]


def test_missing_secrets_raise_clear_error() -> None:
    with pytest.raises(ConfigError) as exc_info:
        load_settings()
    message = str(exc_info.value)
    for name in REQUIRED:
        assert name in message


@pytest.mark.usefixtures("required_env")
def test_blank_secret_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        load_settings()


@pytest.mark.usefixtures("required_env")
def test_env_overrides_and_nested_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERCEIVE_FPS", "2")
    monkeypatch.setenv("FEATURES__MERGE", "false")
    s = load_settings()
    assert s.perceive_fps == 2.0
    assert s.features.merge is False


@pytest.mark.usefixtures("required_env")
@pytest.mark.parametrize(
    "raw",
    [
        '["https://a.vercel.app", "http://localhost:3000"]',
        "https://a.vercel.app, http://localhost:3000",
        "https://a.vercel.app/,http://localhost:3000/",
    ],
)
def test_cors_origins_json_or_comma_separated(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("CORS_ORIGINS", raw)
    assert load_settings().cors_origins == ["https://a.vercel.app", "http://localhost:3000"]


@pytest.mark.usefixtures("required_env")
def test_chunk_bounds_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_MIN_S", "300")
    monkeypatch.setenv("CHUNK_MAX_S", "100")
    with pytest.raises(ConfigError, match="CHUNK_MAX_S"):
        load_settings()


@pytest.mark.usefixtures("required_env")
def test_secrets_are_masked_in_repr() -> None:
    assert "g-key" not in repr(load_settings())


@pytest.mark.usefixtures("required_env")
def test_module_level_settings_attribute() -> None:
    from lecnotes import config

    config.get_settings.cache_clear()
    try:
        from lecnotes.config import settings

        assert settings is config.get_settings()
    finally:
        config.get_settings.cache_clear()
