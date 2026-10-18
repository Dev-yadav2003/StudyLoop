from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request

from apps.api.auth import Authenticator, InvalidToken, User, to_user
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


def current_user(
    request: Request, c: Container, authorization: Annotated[str | None, Header()] = None
) -> User:
    """The signed-in account (Supabase access token). 401 without a valid one."""
    auth: Authenticator | None = request.app.state.auth
    if auth is None:
        raise HTTPException(503, "Sign-in is not configured on this server (SUPABASE_URL).")
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Please sign in.")
    try:
        claims = auth.verify(authorization.removeprefix("Bearer ").strip())
        return to_user(claims, c.settings.owner_emails)
    except InvalidToken:
        raise HTTPException(401, "Your session has expired. Please sign in again.") from None


CurrentUser = Annotated[User, Depends(current_user)]


def require_owner(user: CurrentUser) -> User:
    if not user.is_owner:
        raise HTTPException(403, "Only the owner of this app can do that.")
    return user


Owner = Annotated[User, Depends(require_owner)]
