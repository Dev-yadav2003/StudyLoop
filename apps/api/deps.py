from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request

from lecnotes.bootstrap import ApiContainer


def container(request: Request) -> ApiContainer:
    return request.app.state.container


Container = Annotated[ApiContainer, Depends(container)]


def require_admin(c: Container, authorization: Annotated[str | None, Header()] = None) -> None:
    """Bearer ADMIN_TOKEN. Without a configured token the admin API is disabled."""
    token = c.settings.admin_token
    if token is None:
        raise HTTPException(403, "The admin API is disabled (set ADMIN_TOKEN to enable it).")
    expected = f"Bearer {token.get_secret_value()}"
    if authorization is None or not secrets.compare_digest(authorization, expected):
        raise HTTPException(401, "Missing or invalid admin token.")


Admin = Depends(require_admin)
