"""Private Sandbox routing and short-lived, read-only large file downloads."""
import hashlib
import hmac
import os
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

HEADER = "x-smc-internal-token"


def message(path, query, expires):
    pairs = [(k, v) for k, v in parse_qsl(query, keep_blank_values=True) if k not in ("_smc_exp", "_smc_sig")]
    return f"GET\n{path}\n{urlencode(pairs)}\n{expires}".encode()


def signed_download(url, secret, lifetime=120):
    parsed = urlsplit(url)
    expires = int(time.time()) + min(300, max(1, lifetime))
    signature = hmac.new(secret.encode(), message(parsed.path, parsed.query, expires), hashlib.sha256).hexdigest()
    pairs = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if k not in ("_smc_exp", "_smc_sig")]
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(pairs + [("_smc_exp", str(expires)), ("_smc_sig", signature)]), ""))


def authorized(request):
    secret = os.environ.get("SMC_INTERNAL_TOKEN", "")
    if len(secret) < 32:
        return False
    if hmac.compare_digest(request.headers.get(HEADER, "").encode(), secret.encode()):
        return True
    if request.method != "GET":
        return False
    try:
        expires = int(request.query_params.get("_smc_exp", "0"))
    except ValueError:
        return False
    if not time.time() <= expires <= time.time() + 300:
        return False
    expected = hmac.new(secret.encode(), message(request.url.path, request.url.query, expires), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected.encode(), request.query_params.get("_smc_sig", "").encode())


def install(app):
    from fastapi.responses import JSONResponse
    @app.middleware("http")
    async def private_runtime(request, call_next):
        if not authorized(request):
            return JSONResponse({"detail": "Private Vercel runtime"}, status_code=401)
        response = await call_next(request)
        if not request.url.path.startswith("/api/live/"):
            response.headers["Cache-Control"] = "private, no-store"
        return response
