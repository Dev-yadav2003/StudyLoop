from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from apps.api.deps import Container
from lecnotes.storage.db import code_schema_revision, db_schema_revision

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness: the process answers. Render's health check uses this."""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(c: Container) -> JSONResponse:
    """Readiness: the database answers and is migrated to the schema this code expects, and
    (if enabled) the job runner is alive. A missing migration shows up here, not as 500s."""
    checks: dict[str, str] = {}
    try:
        async with c.engine.connect() as conn:
            found = await db_schema_revision(conn)
        expected = code_schema_revision()
        checks["database"] = "ok"
        checks["schema"] = (
            "ok" if found == expected else f"at {found}, needs {expected}: run alembic upgrade head"
        )
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
