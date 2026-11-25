"""Vercel Function gateway to a project-owned Sandbox and durable Drive.

Training never runs inside an HTTP Function. The existing application/worker run
in one named Sandbox; all datasets, models and SQLite state live on its Drive.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response

from . import auth
from .managed_security import HEADER, signed_download

app = FastAPI(title="Space ML Mission Control", version="3.1.10")
auth.install(app)
MAX_BODY = 4_000_000
ROOT = Path(__file__).resolve().parent
logger = logging.getLogger("vercel-runtime")


def missing_configuration():
    missing = []
    if len(os.environ.get("SMC_INTERNAL_TOKEN", "")) < 32:
        missing.append("SMC_INTERNAL_TOKEN (at least 32 random characters)")
    if auth.required():
        missing += [name for name in ("SMC_AUTH_USER", "SMC_AUTH_PASSWORD") if not os.environ.get(name)]
    if not (ROOT / "runtime_bundle.json").is_file():
        missing.append("runtime source bundle: run scripts/build_vercel_runtime.py")
    if not (os.environ.get("VERCEL_URL") or os.environ.get("SMC_SOURCE_ORIGIN")):
        missing.append("SMC_SOURCE_ORIGIN or Vercel's deployment URL")
    return missing


def source_origin():
    domain = os.environ.get("VERCEL_PROJECT_PRODUCTION_URL") if os.environ.get("VERCEL_ENV") == "production" else None
    origin = os.environ.get("SMC_SOURCE_ORIGIN") or "https://" + (domain or os.environ.get("VERCEL_URL", ""))
    parsed = urlsplit(origin)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise ValueError("Training source must use an HTTPS deployment origin")
    return origin.rstrip("/")


class Runtime:
    def __init__(self):
        self.box = None
        self.refreshed = 0.0
        self.last_start = 0.0
        self.lock = asyncio.Lock()
        self.checks = {"sandbox": {"available": False}, "drive": {"mounted": False, "writable": None, "mount_path": "/smc"}}
        self.stage = "configuration"

    async def target(self):
        from vercel import sandbox
        async with self.lock:
            if not self.box or time.monotonic() - self.refreshed > 15:
                self.checks["sandbox"]["available"] = False
                self.checks["drive"].update(mounted=False, writable=None)
                environment = os.environ.get("VERCEL_ENV", "development")
                branch = os.environ.get("VERCEL_GIT_COMMIT_REF", environment)
                import hashlib
                suffix = "production" if environment == "production" else "preview-" + hashlib.sha256(branch.encode()).hexdigest()[:12]
                name = os.environ.get("SMC_SANDBOX_NAME", "space-ml-" + suffix)
                drive_name = os.environ.get("SMC_DRIVE_NAME", name + "-data")
                self.checks["drive"]["name"] = drive_name
                if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name + "") or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", drive_name):
                    raise ValueError("Invalid project-local runtime name")
                region = os.environ.get("SMC_REGION", "iad1")
                self.stage = "create_drive"
                drive, _ = await sandbox.get_or_create_drive(name=drive_name, region=region)
                seconds = int(os.environ.get("SMC_SANDBOX_SECONDS", "2700"))
                if not 300 <= seconds <= 86400:
                    raise ValueError("Sandbox session duration must be 300–86400 seconds, within your plan limit")
                self.stage = "create_sandbox"
                self.box, _ = await sandbox.get_or_create_sandbox(
                    name=name, image="vercel/sandbox/python:3.14", persistent=True,
                    ports=[8000], execution_time_limit=seconds,
                    resources=sandbox.SandboxResources(vcpus=2, memory=4096),
                    mounts={"/smc": drive}, region=region, snapshot_expiration=0,
                    snapshot_retention=sandbox.SnapshotRetention(count=1),
                )
                self.checks["sandbox"]["available"] = True
                self.stage = "verify_mount"
                mount = (self.box.mounts or {}).get("/smc")
                if mount is None or mount.mode != "read-write" or mount.drive not in (drive_name, getattr(drive, "id", None)):
                    self.box = None
                    raise RuntimeError("The Vercel runtime must mount its named Drive read-write at /smc")
                self.checks["drive"]["mounted"] = True
                self.refreshed = time.monotonic()
            route = next((r for r in self.box.routes if r.port == 8000), None)
            if route is None or not route.url.startswith("https://"):
                raise RuntimeError("Vercel Sandbox has not assigned its API route")
            return self.box, route.url.rstrip("/")

    async def start(self, box):
        async with self.lock:
            if time.monotonic() - self.last_start < 5:
                return
            bundle = json.loads((ROOT / "runtime_bundle.json").read_text(encoding="utf-8"))
            launcher = (ROOT / "managed_bootstrap.py").read_text(encoding="utf-8")
            self.stage = "start_worker"
            environment = {
                "SMC_BUNDLE_SHA256": bundle["sha256"], "SMC_SOURCE_URL": source_origin() + bundle["url"],
                "SMC_INTERNAL_TOKEN": os.environ["SMC_INTERNAL_TOKEN"],
                "SPACEML_TRAIN_CPUS": "2", "SPACEML_HEAVY_JOBS": "1", "PYTHONUNBUFFERED": "1",
            }
            for name in ("SMC_DEPLOYMENT_BYPASS", "SMC_PUBLIC_ORIGIN", "SMC_LIVE_MAX_AGE", "SPACEML_MARS_WEIGHTS_URL"):
                if os.environ.get(name):
                    environment[name] = os.environ[name]
            environment["SPACEML_CORS_ORIGINS"] = os.environ.get("SMC_PUBLIC_ORIGIN", source_origin())
            # Secrets are passed via environment, never command arguments.
            await box.create_command("python3", ["-c", launcher], env=environment)
            self.last_start = time.monotonic()

    async def probe(self):
        self.checks["drive"]["writable"] = None
        box, url = await self.target()
        self.stage = "worker_health"
        bundle = json.loads((ROOT / "runtime_bundle.json").read_text(encoding="utf-8"))
        try:
            async with httpx.AsyncClient(timeout=4) as client:
                response = await client.get(url + "/api/health", headers={HEADER: os.environ["SMC_INTERNAL_TOKEN"]})
                if response.status_code == 200:
                    body = response.json()
                    if body.get("bundle_sha256") == bundle["sha256"]:
                        storage = (body.get("checks") or {}).get("drive") or {}
                        self.checks["drive"]["writable"] = storage.get("writable") is True
                        if storage.get("paths_valid") is not True or storage.get("writable") is not True:
                            return url, body, "storage_error"
                        return url, body, "ready"
                    await self.start(box)
                    return url, body, "updating"
        except (httpx.HTTPError, ValueError):
            pass
        await self.start(box)
        try:
            status = json.loads(await box.fs.read_text("/smc/runtime.json"))
            return url, {"detail": status.get("detail")}, "error" if status.get("state") == "error" else "starting"
        except Exception:
            return url, {}, "starting"


runtime = Runtime()


def published(path, request):
    from .vercel_app import frozen, live, sky
    if path.startswith("live/"):
        parts = path.split("/")
        if len(parts) == 2:
            return live(parts[1])
        if len(parts) == 4 and parts[2] == "sky":
            try:
                fov = float(request.query_params.get("fov", "0.5"))
            except ValueError:
                return JSONResponse({"detail": "Invalid field of view"}, status_code=422)
            return sky(parts[1], parts[3], fov)
    return frozen(path, request)


@app.get("/api/health")
@app.get("/api/config")
async def health(request: Request):
    missing = missing_configuration()
    body, state = {}, "setup_required"
    startup_error = None
    if not missing:
        try:
            _, body, state = await runtime.probe()
        except Exception as error:
            logger.warning("Vercel runtime initialization failed: %s", type(error).__name__)
            state = "error"
            startup_error = {"stage": getattr(runtime, "stage", "startup"), "type": type(error).__name__}
            body = {"detail": "Vercel Sandbox or Drive could not start. Check project credentials, quotas and Function logs."}
    ready = state == "ready"
    if state in ("error", "storage_error") and startup_error is None:
        startup_error = {"stage": getattr(runtime, "stage", "worker_health"), "type": "WorkerStartupError" if state == "error" else "DriveWriteCheckFailed"}
    allowed = ready and auth.authenticated(request)
    observed = (body.get("checks") or {}).get("drive") or {}
    checks = getattr(runtime, "checks", {"sandbox": {"available": ready}, "drive": {"mounted": ready, "writable": None}})
    checks = {"sandbox": dict(checks["sandbox"]), "drive": {**checks["drive"], **observed}}
    if missing:
        checks = {"sandbox": {"available": False}, "drive": {"mounted": False, "writable": None, "mount_path": "/smc"}}
    checks["drive"]["name"] = os.environ.get("SMC_DRIVE_NAME", checks["drive"].get("name"))
    return JSONResponse({**body, "status": "ok" if ready else state, "state": state,
                         "checks": checks, "startup_error": startup_error,
                         "mode": "vercel", "execution": "managed", "version": "3.1.10",
                         "storage": "vercel-drive" if ready else "not_connected", "read_only": [] if allowed else ["authentication_required" if ready else "compute_not_ready"],
                         "authentication": {"required": auth.required(), "authenticated": auth.authenticated(request)},
                         "capabilities": {"write": allowed, "train": allowed, "live": True},
                         "setup_required": missing,
                         "hint": "Connect Vercel Sandbox and Drive" if missing else "Vercel Drive write check failed; inspect /api/health/drive" if state == "storage_error" else "Vercel compute could not start; check Sandbox permissions, quotas and project logs" if state == "error" else "Sign in to train" if ready and not allowed else "Preparing Vercel training compute" if not ready else None},
                        headers={"Cache-Control": "private, no-store"})


@app.get("/api/health/sandbox")
@app.get("/api/health/drive")
async def health_component(request: Request):
    report = json.loads((await health(request)).body)
    component = request.url.path.rsplit("/", 1)[-1]
    check = report["checks"][component]
    ok = check.get("available") is True if component == "sandbox" else check.get("mounted") is True and check.get("writable") is True
    return JSONResponse({"component": component, "ok": ok, "state": report["state"], **check,
                         "startup_error": report["startup_error"], "detail": report.get("detail")},
                        headers={"Cache-Control": "private, no-store"})


@app.api_route("/api/{path:path}", methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def proxy(path: str, request: Request):
    missing = missing_configuration()
    if missing:
        # Published data stays browsable while the account is being connected.
        # Never advertise a working trainer before real compute is reachable.
        if request.method == "GET":
            return published(path, request)
        return JSONResponse({"detail": "Connect Vercel Sandbox and Drive before loading or training data", "setup_required": missing}, status_code=409)
    try:
        url, _, state = await runtime.probe()
        if state != "ready" and not (state == "updating" and request.method == "GET"):
            if request.method == "GET":
                return published(path, request)
            return JSONResponse({"detail": "Vercel training compute is starting or updating. Please retry shortly.", "state": state}, status_code=409, headers={"Retry-After": "5"})
        body = bytearray()
        async for block in request.stream():
            body.extend(block)
            if len(body) > MAX_BODY:
                return JSONResponse({"detail": "Browser uploads are limited to 4 MB. Use the dataset Download action to fetch large source files directly into Vercel storage."}, status_code=413)
        target = url + "/api/" + path + ("?" + request.url.query if request.url.query else "")
        headers = {HEADER: os.environ["SMC_INTERNAL_TOKEN"]}
        for name in ("content-type", "accept", "if-none-match", "range"):
            if request.headers.get(name):
                headers[name] = request.headers[name]
        async with httpx.AsyncClient(timeout=55, follow_redirects=False) as client:
            async with client.stream(request.method, target, headers=headers, content=bytes(body)) as response:
                output = bytearray()
                async for block in response.aiter_bytes():
                    output.extend(block)
                    if len(output) > MAX_BODY:
                        if request.method == "GET":
                            return RedirectResponse(signed_download(target, os.environ["SMC_INTERNAL_TOKEN"]), status_code=307, headers={"Cache-Control": "private, no-store"})
                        return JSONResponse({"detail": "Choose a smaller page or use a result download"}, status_code=413)
                forwarded = {k: v for k, v in response.headers.items() if k.lower() in ("content-type", "content-disposition", "etag", "last-modified", "content-range", "accept-ranges")}
                forwarded["Cache-Control"] = "private, no-store"
                return Response(bytes(output), status_code=response.status_code, headers=forwarded)
    except httpx.HTTPError:
        return JSONResponse({"detail": "Vercel compute is restarting; persisted jobs and data are retained. Retry shortly."}, status_code=409, headers={"Retry-After": "5"})
    except Exception as error:
        logger.warning("Vercel runtime request failed: %s", type(error).__name__)
        return JSONResponse({"detail": "Vercel compute is unavailable. Check Sandbox/Drive configuration and project logs."}, status_code=409)
