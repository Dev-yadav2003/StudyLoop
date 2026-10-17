"""The unit suite must run with no network (step 2.3). These tests prove the guard holds."""

import asyncio
import json
import socket
from pathlib import Path

import httpx
import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, SecretStr, ValidationError

from lecnotes.adapters.llm.fake import FakeLLM
from lecnotes.config import Settings
from lecnotes.errors import LLMError
from lecnotes.llm.gateway import LLMGateway, gemini_factory
from lecnotes.prompts.loader import PromptSpec, RenderedPrompt
from tests.network_guard import NetworkBlocked


class Tiny(BaseModel):
    word: str


def tiny_prompt() -> RenderedPrompt:
    spec = PromptSpec(
        id="tiny", version=1, response_schema="Tiny", changelog="t", system="s", user="u"
    )
    return RenderedPrompt(ref=spec.ref, spec=spec, messages=[HumanMessage("u")])


def test_dns_to_external_host_blocked() -> None:
    with pytest.raises(NetworkBlocked, match="generativelanguage"):
        socket.getaddrinfo("generativelanguage.googleapis.com", 443)


def test_connect_to_external_ip_blocked() -> None:
    with socket.socket() as s, pytest.raises(NetworkBlocked):
        s.connect(("142.250.0.1", 443))


async def test_httpx_blocked() -> None:
    async with httpx.AsyncClient() as client:
        with pytest.raises(NetworkBlocked):
            await client.get("https://example.com")


async def test_loopback_and_event_loop_still_work() -> None:
    reader, writer = await asyncio.open_connection(*await _echo_server())
    writer.write(b"ping")
    await writer.drain()
    assert await reader.read(4) == b"ping"
    writer.close()


async def _echo_server() -> tuple[str, int]:
    async def echo(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        w.write(await r.read(4))
        await w.drain()
        w.close()

    server = await asyncio.start_server(echo, "127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]
    return host, port


async def test_real_gemini_model_cannot_leak(monkeypatch: pytest.MonkeyPatch) -> None:
    """If a test accidentally wires the production factory, it fails instead of calling Google."""
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    secret = SecretStr("not-a-real-key")
    settings = Settings(
        gemini_api_key=secret,
        youtube_api_key=secret,
        database_url=secret,
        langfuse_public_key=secret,
        langfuse_secret_key=secret,
        # The limiter's bucket starts empty: at the real 10 RPM the first call waits 6 s.
        llm_rpm_override=60_000,
    )
    gateway = LLMGateway(
        gemini_factory(settings),
        default_model="gemini-2.5-flash",
        retry_attempts=1,
        retry_backoff=False,
        callbacks=list,
    )
    with pytest.raises((NetworkBlocked, LLMError)) as exc_info:
        await gateway.generate_structured(tiny_prompt(), Tiny)

    chain = [exc_info.value]
    while chain[-1].__cause__ or chain[-1].__context__:
        chain.append(chain[-1].__cause__ or chain[-1].__context__)  # type: ignore[arg-type]
    assert any(isinstance(e, NetworkBlocked) for e in chain)


async def test_fake_from_recordings(tmp_path: Path) -> None:
    path = tmp_path / "tiny.json"
    path.write_text(json.dumps({"word": "recorded"}), encoding="utf-8")
    fake = FakeLLM.from_recordings({"tiny": (path, Tiny)})

    gateway = LLMGateway(fake.factory, default_model="gemini-2.5-flash", retry_backoff=False)
    result = await gateway.generate_structured(tiny_prompt(), Tiny)

    assert result.output == Tiny(word="recorded")


def test_stale_recording_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "tiny.json"
    path.write_text(json.dumps({"old_field": "x"}), encoding="utf-8")
    with pytest.raises(ValidationError):
        FakeLLM.from_recordings({"tiny": (path, Tiny)})
