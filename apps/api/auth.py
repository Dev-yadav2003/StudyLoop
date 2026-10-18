"""Who is calling: Supabase Auth access tokens (JWT), verified locally.

The browser signs in with Supabase (email link) and sends `Authorization: Bearer <token>`.
The API verifies the token's signature with the project's PUBLIC keys (JWKS, ES256) — no
Supabase secret is needed or stored — plus expiry and audience. The owner is recognised by
email (`OWNER_EMAILS`) and has no limits; everyone else is a guest with a lifetime allowance.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Protocol

import jwt

AUDIENCE = "authenticated"


class InvalidToken(Exception):
    pass


@dataclass(frozen=True)
class User:
    id: uuid.UUID
    email: str
    is_owner: bool


class Authenticator(Protocol):
    def verify(self, token: str) -> dict[str, Any]: ...


class JwksAuthenticator:
    """Production: keys fetched from https://<ref>.supabase.co/auth/v1/.well-known/jwks.json
    and cached (PyJWKClient), so a key rotation is picked up without a deploy."""

    def __init__(self, supabase_url: str) -> None:
        self.issuer = f"{supabase_url}/auth/v1"
        self._keys = jwt.PyJWKClient(f"{self.issuer}/.well-known/jwks.json", cache_keys=True)

    def verify(self, token: str) -> dict[str, Any]:
        try:
            key = self._keys.get_signing_key_from_jwt(token)
            return jwt.decode(
                token,
                key.key,
                algorithms=["ES256", "RS256"],
                audience=AUDIENCE,
                issuer=self.issuer,
            )
        except jwt.PyJWTError as exc:
            raise InvalidToken(str(exc)) from exc


class StaticKeyAuthenticator:
    """Tests: tokens signed with a known key (HS256 secret or a PEM key pair)."""

    def __init__(self, key: str, algorithm: str = "HS256") -> None:
        self._key = key
        self._algorithm = algorithm

    def verify(self, token: str) -> dict[str, Any]:
        try:
            return jwt.decode(token, self._key, algorithms=[self._algorithm], audience=AUDIENCE)
        except jwt.PyJWTError as exc:
            raise InvalidToken(str(exc)) from exc


def to_user(claims: dict[str, Any], owner_emails: list[str]) -> User:
    try:
        user_id = uuid.UUID(str(claims["sub"]))
    except (KeyError, ValueError) as exc:
        raise InvalidToken("token has no valid subject") from exc
    email = str(claims.get("email") or "").strip().lower()
    return User(id=user_id, email=email, is_owner=bool(email) and email in owner_emails)
