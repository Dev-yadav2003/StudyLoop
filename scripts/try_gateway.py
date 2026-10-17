"""Step 2.2 check: one real structured Gemini call through the gateway, traced in Langfuse.

    uv run python scripts/try_gateway.py

Requires GEMINI_API_KEY and LANGFUSE_* in .env. Prints the validated object, usage and the
trace URL; in Langfuse, confirm the generation shows prompt, output, tokens and latency,
nested under the `stage:smoke` span.
"""

import asyncio
import uuid

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from lecnotes.config import get_settings
from lecnotes.llm.gateway import LLMGateway
from lecnotes.observability.logging import configure_logging, get_logger
from lecnotes.observability.tracing import (
    get_langfuse,
    init_tracing_from_settings,
    shutdown_tracing,
    traced_job,
    traced_stage,
)
from lecnotes.prompts.loader import PromptSpec, RenderedPrompt


class Concept(BaseModel):
    name: str = Field(description="Name of the concept")
    one_line_definition: str
    example: str


def smoke_prompt() -> RenderedPrompt:
    # Script-only prompt: deliberately not shipped in src/lecnotes/prompts/.
    spec = PromptSpec(
        id="smoke",
        version=1,
        temperature=0.0,
        response_schema="Concept",
        changelog="gateway smoke test",
        system="You explain programming concepts precisely and briefly.",
        user="Explain Python's built-in enumerate().",
    )
    return RenderedPrompt(
        ref=spec.ref,
        spec=spec,
        messages=[SystemMessage(spec.system), HumanMessage(spec.user)],
    )


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    log = get_logger(__name__)
    init_tracing_from_settings(settings)
    gateway = LLMGateway.from_settings(settings)

    try:
        with traced_job(f"smoke-{uuid.uuid4().hex[:8]}") as root, traced_stage("smoke", 1):
            result = await gateway.generate_structured(smoke_prompt(), Concept)
            trace_id = root.trace_id
        print(result.output.model_dump_json(indent=2))
        log.info(
            "gateway.smoke.done",
            model=result.usage.model,
            tokens_in=result.usage.tokens_in,
            tokens_out=result.usage.tokens_out,
            cost_usd=str(result.usage.cost_usd),
            latency_ms=round(result.usage.latency_ms),
        )
        print(f"\nOpen in Langfuse: {get_langfuse().get_trace_url(trace_id=trace_id)}\n")
    finally:
        shutdown_tracing()


if __name__ == "__main__":
    asyncio.run(main())
