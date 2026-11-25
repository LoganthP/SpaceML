import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from urllib.parse import urlsplit

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def gateway(monkeypatch):
    sys.path.insert(0, str(ROOT / "backend"))
    monkeypatch.setenv("SMC_MODE", "vercel")
    monkeypatch.delenv("SMC_EXECUTION", raising=False)
    monkeypatch.setenv("SMC_AUTH_REQUIRED", "false")
    monkeypatch.setenv("SMC_INTERNAL_TOKEN", "a" * 48)
    for name in list(sys.modules):
        if name.startswith(("app", "spaceml")):
            del sys.modules[name]
    module = importlib.import_module("app.managed")
    monkeypatch.setattr(module, "missing_configuration", lambda: [])
    class Runtime:
        calls = 0
        async def probe(self):
            self.calls += 1
            return "https://sandbox.example", {"checks": {"drive": {"paths_valid": True, "writable": True}}}, "ready"
    monkeypatch.setattr(module, "runtime", Runtime())
    return module, TestClient(module.app)


def test_default_gateway_requires_authentication_before_any_compute(gateway, monkeypatch):
    module, client = gateway
    monkeypatch.delenv("SMC_AUTH_REQUIRED")
    assert client.post("/api/jobs", json={"task": "telemetry.train"}).status_code == 401
    assert module.runtime.calls == 0
    assert client.get("/api/health").json()["capabilities"]["train"] is False


def test_missing_setup_is_honest_and_catalog_excludes_removed_missions(gateway, monkeypatch):
    module, client = gateway
    monkeypatch.setattr(module, "missing_configuration", lambda: ["SMC_INTERNAL_TOKEN"])
    health = client.get("/api/health").json()
    assert health["mode"] == "vercel" and health["execution"] == "managed"
    assert health["state"] == "setup_required" and not health["capabilities"]["train"]
    assert health["storage"] == "not_connected"
    assert module.runtime.calls == 0
    catalog = client.get("/api/catalog").json()
    assert catalog["free_bytes"] is None
    assert len(catalog["datasets"]) == 9
    assert {e["mission"] for e in catalog["datasets"] if e.get("mission")} == {"opssat", "hubble", "webb"}
    assert client.post("/api/jobs", json={}).status_code == 409
    assert "Docker" not in client.get("/api/catalog").text
    assert "Connect Vercel compute" not in client.get("/api/catalog").text
    assert client.get("/api/health/sandbox").json()["ok"] is False
    assert client.get("/api/health/drive").json()["ok"] is False


def test_ready_gateway_forwards_real_job_requests_and_bounds_bodies(gateway, monkeypatch):
    module, client = gateway
    upstream = FastAPI()
    received = []
    @upstream.post("/api/jobs")
    async def job(request: importlib.import_module("fastapi").Request):
        assert request.headers["x-smc-internal-token"] == "a" * 48
        body = await request.json()
        received.append(body)
        return {"id": "real-queue-id", "status": "queued", **body}
    @upstream.get("/api/large")
    def large():
        return Response(b"x" * (module.MAX_BODY + 1), media_type="application/octet-stream")
    original = httpx.AsyncClient
    monkeypatch.setattr(module.httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.ASGITransport(app=upstream), **kwargs))
    health = client.get("/api/health").json()
    assert health["capabilities"]["train"] is True and health["storage"] == "vercel-drive"
    assert client.get("/api/health/sandbox").json()["ok"] is True
    assert client.get("/api/health/drive").json()["ok"] is True
    result = client.post("/api/jobs", json={"task": "missions.opssat.train", "params": {"mode": "quick"}})
    assert result.status_code == 200 and result.json()["id"] == "real-queue-id"
    assert received[0]["task"] == "missions.opssat.train"
    assert client.post("/api/jobs", content=b"x" * (module.MAX_BODY + 1)).status_code == 413
    assert len(received) == 1
    download = client.get("/api/large", follow_redirects=False)
    assert download.status_code == 307 and "_smc_sig=" in download.headers["location"]
    assert "a" * 48 not in download.headers["location"]


def test_compute_is_not_advertised_until_ready(gateway, monkeypatch):
    module, client = gateway
    async def starting():
        return "https://sandbox.example", {}, "starting"
    monkeypatch.setattr(module.runtime, "probe", starting)
    body = client.get("/api/health").json()
    assert body["state"] == "starting" and not body["capabilities"]["write"]
    assert client.post("/api/jobs", json={}).status_code == 409


@pytest.mark.parametrize("writable,expected", [(False, "storage_error"), (True, "ready")])
def test_runtime_requires_worker_drive_write_proof(gateway, monkeypatch, tmp_path, writable, expected):
    import asyncio
    module, _ = gateway
    monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / "runtime_bundle.json").write_text(json.dumps({"sha256": "current"}))
    runtime = module.Runtime()
    async def target():
        return object(), "https://sandbox.example"
    monkeypatch.setattr(runtime, "target", target)
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={
        "bundle_sha256": "current", "checks": {"drive": {"paths_valid": True, "writable": writable}}
    }))
    original = httpx.AsyncClient
    monkeypatch.setattr(module.httpx, "AsyncClient", lambda **kwargs: original(transport=transport, **kwargs))
    assert asyncio.run(runtime.probe())[2] == expected
    assert runtime.checks["drive"]["writable"] is writable


def test_signed_download_is_read_only_bound_to_path_and_expires(gateway, monkeypatch):
    security = importlib.import_module("app.managed_security")
    app = FastAPI()
    security.install(app)
    @app.api_route("/api/file", methods=["GET", "POST"])
    def file():
        return {"ok": True}
    client = TestClient(app)
    monkeypatch.setattr(security.time, "time", lambda: 1000)
    signed = security.signed_download("https://sandbox.example/api/file?format=zip", "a" * 48)
    path = urlsplit(signed).path + "?" + urlsplit(signed).query
    assert client.get(path).status_code == 200
    assert client.post(path).status_code == 401
    assert client.get(path.replace("/file?", "/other?")).status_code == 401
    assert client.get("/api/file").status_code == 401
    monkeypatch.setattr(security.time, "time", lambda: 1201)
    assert client.get(path).status_code == 401


@pytest.mark.parametrize("drive_name", [None, "spaceml-data"])
def test_sdk_creates_one_project_owned_drive_and_persistent_runtime(gateway, monkeypatch, tmp_path, drive_name):
    import asyncio
    from vercel import sandbox
    module, _ = gateway
    monkeypatch.setenv("VERCEL_ENV", "production")
    monkeypatch.setenv("VERCEL_URL", "space.example.vercel.app")
    if drive_name:
        monkeypatch.setenv("SMC_DRIVE_NAME", drive_name)
    else:
        monkeypatch.delenv("SMC_DRIVE_NAME", raising=False)
    expected_name = drive_name or "space-ml-production-data"
    (tmp_path / "runtime_bundle.json").write_text(json.dumps({"sha256": "b" * 64, "url": "/runtime/source.tar.gz"}))
    (tmp_path / "managed_bootstrap.py").write_text("print('launcher')")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    calls = []
    drive = object()
    class Box:
        routes = [SimpleNamespace(port=8000, url="https://sandbox.example")]
        mounts = {"/smc": SimpleNamespace(drive=expected_name, mode="read-write")}
        async def create_command(self, *args, **kwargs):
            calls.append(("command", args, kwargs))
    async def get_drive(**kwargs):
        assert kwargs["name"] == expected_name
        calls.append(("drive", kwargs))
        return drive, True
    async def get_box(**kwargs):
        assert kwargs["mounts"] == {"/smc": drive}
        assert kwargs["persistent"] is True and kwargs["ports"] == [8000]
        calls.append(("sandbox", kwargs))
        return Box(), True
    monkeypatch.setattr(sandbox, "get_or_create_drive", get_drive)
    monkeypatch.setattr(sandbox, "get_or_create_sandbox", get_box)
    async def exercise():
        runtime = module.Runtime()
        box, url = await runtime.target()
        assert url == "https://sandbox.example"
        await runtime.start(box)
        await runtime.target()
        await runtime.start(box)
    asyncio.run(exercise())
    assert [c[0] for c in calls] == ["drive", "sandbox", "command"]
    command = calls[-1]
    assert "a" * 48 not in str(command[1])
    assert command[2]["env"]["SMC_INTERNAL_TOKEN"] == "a" * 48
