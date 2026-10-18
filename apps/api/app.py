"""FastAPI application factory.

create_app()                    production: wiring built in the lifespan (fail fast on config)
create_app(container=...)       tests: inject a container built from fakes
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api import middleware
from apps.api.routers import admin, health, jobs, library, notes
from lecnotes import __version__
from lecnotes.bootstrap import ApiContainer, api_container
from lecnotes.config import Settings, get_settings
from lecnotes.observability.logging import configure_logging
from lecnotes.observability.tracing import init_tracing_from_settings, shutdown_tracing


def create_app(
    container: ApiContainer | None = None, *, settings: Settings | None = None
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if container is not None:  # injected (tests): attached below, at construction
            yield
            return
        prod = settings or get_settings()
        configure_logging(prod.log_level)
        init_tracing_from_settings(prod)
        try:
            async with api_container(prod) as built:
                app.state.container = built
                yield
        finally:
            shutdown_tracing()

    app = FastAPI(
        title="Lecture-to-Notes API",
        version=__version__,
        description="Turn one YouTube lecture into faithful, timestamped study notes.",
        lifespan=lifespan,
    )
    cors = (container.settings if container else settings or get_settings()).cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Content-Type", "Authorization", middleware.REQUEST_ID],
        expose_headers=[middleware.REQUEST_ID, "Location"],
    )
    if container is not None:
        app.state.container = container
    middleware.install(app)
    for router in (health.router, jobs.router, library.router, notes.router, admin.router):
        app.include_router(router)
    return app
