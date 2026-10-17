"""Request ID + timing + uniform JSON errors.

Every response carries `X-Request-ID` (echoed if the client sent one), and every log line
emitted while handling the request is bound to it. Errors always have the same JSON shape:
{"error": {"code", "message", "request_id"}}.
"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from apps.api.schemas import ErrorBody, ErrorOut
from lecnotes.errors import InvalidLink, LecNotesError
from lecnotes.observability.logging import bind_context, get_logger

log = get_logger("lecnotes.api")
REQUEST_ID = "X-Request-ID"
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def error_response(
    status: int, code: str, message: str, request: Request | None = None
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None) if request else None
    body = ErrorOut(error=ErrorBody(code=code, message=message, request_id=request_id))
    return JSONResponse(status_code=status, content=body.model_dump())


def install(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get(REQUEST_ID, "")
        request_id = incoming if _SAFE_ID.match(incoming) else uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        with bind_context(request_id=request_id):
            try:
                response = await call_next(request)
            except Exception:
                log.exception("http.unhandled_error", path=request.url.path)
                response = error_response(
                    500, "internal_error", LecNotesError.user_message, request
                )
            response.headers[REQUEST_ID] = request_id
            log.info(
                "http.request",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            return response

    @app.exception_handler(LecNotesError)
    async def domain_error(request: Request, exc: LecNotesError) -> JSONResponse:
        status = 422 if isinstance(exc, InvalidLink) else 503 if exc.retryable else 500
        return error_response(status, type(exc).__name__, exc.user_message, request)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {401: "unauthorized", 403: "forbidden", 404: "not_found", 409: "conflict"}.get(
            exc.status_code, "http_error"
        )
        return error_response(exc.status_code, code, str(exc.detail), request)
