import base64
import importlib.util
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_auth_blocks_writes_and_checks_origin_without_affecting_public_reads(monkeypatch):
    auth = module("backend/app/auth.py", "auth_test")
    monkeypatch.setenv("SMC_AUTH_REQUIRED", "true")
    monkeypatch.setenv("SMC_AUTH_USER", "operator")
    monkeypatch.setenv("SMC_AUTH_PASSWORD", "test-password")
    monkeypatch.setenv("SMC_PUBLIC_ORIGIN", "https://space.example")
    app = FastAPI()
    auth.install(app)
    effects = []
    @app.post("/api/jobs")
    def write():
        effects.append(True)
        return {"ok": True}
    @app.get("/api/health")
    def read():
        return {"ok": True}
    client = TestClient(app)
    assert client.get("/api/health").status_code == 200
    assert client.post("/api/jobs").status_code == 401
    assert client.get("/api/session", follow_redirects=False).status_code == 401
    assert not effects
    authorization = "Basic " + base64.b64encode(b"operator:test-password").decode()
    assert client.post("/api/jobs", headers={"Authorization": authorization, "Origin": "https://evil.example"}).status_code == 403
    assert not effects
    result = client.post("/api/jobs", headers={"Authorization": authorization, "Origin": "https://space.example"})
    assert result.status_code == 200 and effects == [True]
    assert result.headers["cache-control"] == "private, no-store"
    monkeypatch.delenv("SMC_AUTH_PASSWORD")
    assert client.post("/api/jobs", headers={"Authorization": authorization}).status_code == 401
    monkeypatch.setenv("SMC_AUTH_REQUIRED", "false")
    assert client.post("/api/jobs").status_code == 200


def test_external_config_preserves_api_prefix_and_has_no_python_service():
    config = module("scripts/configure_persistent_backend.py", "deployment_test")
    body = config.configuration("https://api.example/")
    assert set(body["services"]) == {"frontend"}
    assert body["rewrites"][1]["destination"] == "https://api.example/api/:path*"
    for unsafe in ("http://api.example", "https://user:secret@api.example", "https://api.example/prefix"):
        try:
            config.configuration(unsafe)
        except ValueError:
            pass
        else:
            raise AssertionError(unsafe)
