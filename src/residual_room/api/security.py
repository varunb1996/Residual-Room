"""HTTP Basic auth for the deployed app.

Everything (screens and /api) needs the owner's login once ADMIN_PASSWORD is
set, except what must work for readers and machines: the health check, tracked
links (/r/), unsubscribe links (/u/), and the tick endpoint, which checks its
own bearer token. Locally, without ADMIN_PASSWORD, nothing is protected; on
Render (which sets RENDER) the app refuses to start without one.
"""

import base64
import binascii
import os
import secrets

from fastapi import Request
from fastapi.responses import Response

from residual_room.settings import get_settings

OPEN_PATHS = ("/health", "/r/", "/u/", "/api/internal/")


def check_deploy_safety() -> None:
    if os.environ.get("RENDER") and not get_settings().admin_password:
        raise RuntimeError("ADMIN_PASSWORD must be set when deployed: the screens can approve and send")


async def basic_auth(request: Request, call_next):
    password = get_settings().admin_password
    path = request.url.path
    if not password or path.startswith(OPEN_PATHS):
        return await call_next(request)
    if _authorized(request.headers.get("authorization", ""), get_settings().admin_user, password):
        return await call_next(request)
    return Response("Login required", status_code=401,
                    headers={"WWW-Authenticate": 'Basic realm="Residual Room", charset="UTF-8"'})


def _authorized(header: str, user: str, password: str) -> bool:
    scheme, _, encoded = header.partition(" ")
    if scheme.lower() != "basic":
        return False
    try:
        given_user, _, given_password = base64.b64decode(encoded).decode("utf-8").partition(":")
    except (binascii.Error, UnicodeDecodeError):
        return False
    # Constant-time comparisons, so timing doesn't leak how much matched.
    return secrets.compare_digest(given_user, user) & secrets.compare_digest(given_password, password)
