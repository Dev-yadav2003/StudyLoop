from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from apps.api.deps import Container

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness: the process answers. Render's health check uses this."""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(c: Container) -> JSONResponse:
    """Readiness: the database answers and (if enabled) the job runner is alive."""
    checks: dict[str, str] = {}
    try:
        async with c.engine.connect() as conn:
            await conn.execute(text("select 1"))
        checks["database"] = "ok"
    except Exception as exc:  # report, don't raise: this endpoint must answer
        checks["database"] = f"error: {type(exc).__name__}"
    if c.settings.run_worker:
        task = c.runner_task
        checks["worker"] = "ok" if task is not None and not task.done() else "stopped"
    ready = all(v == "ok" for v in checks.values())
    return JSONResponse(
        {"status": "ok" if ready else "unavailable", "checks": checks},
        status_code=200 if ready else 503,
    )
