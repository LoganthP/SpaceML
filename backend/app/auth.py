"""Optional HTTP Basic authentication for persistent deployments; local default is unchanged.

Credentials stay in backend environment variables. The browser manages its own
same-origin HTTP authentication; no password or token is embedded in Vite.
"""
import base64
import hmac
import os

from fastapi import Request
from fastapi.responses import JSONResponse, RedirectResponse


def required():
    return os.environ.get("SMC_AUTH_REQUIRED", "false").lower() == "true"


def authenticated(request: Request):
    if not required():
        return True
    user, password = os.environ.get("SMC_AUTH_USER", ""), os.environ.get("SMC_AUTH_PASSWORD", "")
    if not user or not password:
        return False
    try:
        scheme, value = request.headers.get("authorization", "").split(" ", 1)
        if scheme.lower() != "basic":
            return False
        supplied = base64.b64decode(value, validate=True).decode("utf-8")
        return hmac.compare_digest(supplied.encode(), f"{user}:{password}".encode())
    except (ValueError, UnicodeError):
        return False


def install(app):
    @app.middleware("http")
    async def guard(request: Request, call_next):
        protected = request.url.path.startswith("/api/") and (
            request.method not in ("GET", "HEAD", "OPTIONS") or request.url.path == "/api/session")
        if protected and not authenticated(request):
            return JSONResponse({"detail": "Sign in to use the persistent backend"}, status_code=401,
                                headers={"WWW-Authenticate": 'Basic realm="Space ML", charset="UTF-8"',
                                         "Cache-Control": "no-store"})
        if protected and required() and request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            allowed = {s.strip().rstrip("/") for s in os.environ.get("SMC_PUBLIC_ORIGIN", "").split(",") if s.strip()}
            if origin and origin.rstrip("/") not in allowed:
                return JSONResponse({"detail": "Origin is not allowed"}, status_code=403, headers={"Cache-Control": "no-store"})
        response = await call_next(request)
        if required() and request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "private, no-store"
        return response

    @app.get("/api/session", include_in_schema=False)
    def session():
        return RedirectResponse("/", status_code=303, headers={"Cache-Control": "no-store"})
