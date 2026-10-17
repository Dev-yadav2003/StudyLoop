"""Step 0.4 check: send one real trace with two nested dummy stage spans to Langfuse.

    uv run python scripts/try_tracing.py

Requires LANGFUSE_* (and the other required settings) in .env. Prints the trace URL;
open it and confirm both stage spans and their metadata are visible.
"""

import time
import uuid

from lecnotes.config import get_settings
from lecnotes.observability.logging import configure_logging, get_logger
from lecnotes.observability.tracing import (
    init_tracing,
    shutdown_tracing,
    traced_job,
    traced_stage,
)


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    log = get_logger(__name__)
    client = init_tracing(settings)

    if not client.auth_check():
        raise SystemExit("Langfuse auth failed: check LANGFUSE_PUBLIC_KEY / SECRET_KEY / BASE_URL")

    job_id = f"smoke-{uuid.uuid4().hex[:8]}"
    try:
        with traced_job(job_id, video_id="dQw4w9WgXcQ") as root:
            with traced_stage("dummy_resolve", version=1, input={"url": "https://youtu.be/x"}) as s:
                time.sleep(0.2)
                s.update(output={"videos": 1})
                log.info("dummy.resolve.done")
            with traced_stage("dummy_perceive", version=1, prompt_version=1) as s:
                time.sleep(0.3)
                s.update(output={"segments": 12, "board_states": 4})
                log.info("dummy.perceive.done")
            trace_id = root.trace_id
        url = client.get_trace_url(trace_id=trace_id)
        log.info("trace.sent", trace_id=trace_id, url=url)
        print(f"\nOpen in Langfuse: {url}\n")
    finally:
        shutdown_tracing()


if __name__ == "__main__":
    main()
