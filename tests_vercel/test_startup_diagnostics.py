import hashlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import urllib.error
import urllib.request

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bootstrap = load("startup_bootstrap_test", "backend/app/managed_bootstrap.py")
doctor = load("startup_doctor_test", "scripts/check_vercel_setup.py")


class Response(io.BytesIO):
    def __init__(self, data, content_type):
        super().__init__(data)
        self.headers = {"Content-Type": content_type}


@pytest.mark.parametrize("corrupt", [False, True])
def test_runtime_source_checks_digest_and_forwards_bypass_privately(tmp_path, monkeypatch, corrupt):
    monkeypatch.setenv("SMC_INTERNAL_TOKEN", "internal-test-secret")
    monkeypatch.setenv("SMC_DEPLOYMENT_BYPASS", "automation-test-secret")
    payload = b"verified-runtime-archive"
    def open_request(request, **kwargs):
        headers = {key.lower(): value for key, value in request.header_items()}
        assert headers["x-smc-internal-token"] == "internal-test-secret"
        assert headers["x-vercel-protection-bypass"] == "automation-test-secret"
        assert "secret" not in request.full_url
        return Response(payload, "application/gzip")
    monkeypatch.setattr(bootstrap.urllib.request, "build_opener", lambda *args: SimpleNamespace(open=open_request))
    expected = "invalid" if corrupt else hashlib.sha256(payload).hexdigest()
    if corrupt:
        with pytest.raises(ValueError, match="checksum"):
            bootstrap.download_source("https://app.example/runtime/archive", tmp_path / "archive", expected)
    else:
        bootstrap.download_source("https://app.example/runtime/archive", tmp_path / "archive", expected)
        assert (tmp_path / "archive").read_bytes() == payload


@pytest.mark.parametrize("code,setting", [(401, "SMC_DEPLOYMENT_BYPASS"), (403, "SMC_INTERNAL_TOKEN"), (404, "SMC_SOURCE_ORIGIN")])
def test_runtime_source_denials_identify_the_setting(tmp_path, monkeypatch, code, setting):
    monkeypatch.setenv("SMC_INTERNAL_TOKEN", "not-printed")
    def denied(*args, **kwargs):
        raise urllib.error.HTTPError("https://app.example/runtime/archive", code, "denied", {}, None)
    monkeypatch.setattr(bootstrap.urllib.request, "build_opener", lambda *args: SimpleNamespace(open=denied))
    with pytest.raises(RuntimeError, match=setting) as raised:
        bootstrap.download_source("https://app.example/runtime/archive", tmp_path / "archive", "invalid")
    assert "not-printed" not in str(raised.value)


def test_runtime_source_html_is_not_misreported_as_a_checksum_error(tmp_path, monkeypatch):
    monkeypatch.setenv("SMC_INTERNAL_TOKEN", "not-printed")
    monkeypatch.setattr(bootstrap.urllib.request, "build_opener", lambda *args: SimpleNamespace(open=lambda *a, **k: Response(b"login", "text/html")))
    with pytest.raises(RuntimeError, match="SMC_DEPLOYMENT_BYPASS"):
        bootstrap.download_source("https://app.example/runtime/archive", tmp_path / "archive", "invalid")
    assert not (tmp_path / "archive").exists()


def test_health_doctor_rejects_login_html_even_with_http_200(monkeypatch):
    monkeypatch.setattr(doctor.urllib.request, "build_opener", lambda *args: SimpleNamespace(open=lambda *a, **k: Response(b"login", "text/html")))
    with pytest.raises(ValueError, match="HTML"):
        doctor.fetch_health("https://app.example", {})


@pytest.mark.parametrize("handler,error", [(bootstrap.SourceRedirect, RuntimeError), (doctor.HealthRedirect, ValueError)])
def test_redirects_never_forward_secrets_outside_the_deployment(handler, error):
    request = urllib.request.Request("https://app.example/api/health", headers={"Authorization": "Basic test-secret"})
    with pytest.raises(error, match="outside"):
        handler().redirect_request(request, None, 302, "redirect", {}, "https://vercel.com/login")


@pytest.mark.parametrize("state,allowed,expected", [("setup_required", False, 2), ("ready", False, 4), ("ready", True, 0)])
def test_doctor_separates_compute_setup_from_operator_access(monkeypatch, state, allowed, expected):
    body = {"state": state, "mode": "vercel", "execution": "managed",
            "setup_required": ["SMC_INTERNAL_TOKEN"] if state == "setup_required" else [],
            "checks": {"sandbox": {"available": True}, "drive": {"name": "test", "mounted": True, "paths_valid": True, "writable": True}},
            "capabilities": {"write": allowed, "train": allowed}}
    monkeypatch.setattr(doctor, "fetch_health", lambda *args: body)
    assert doctor.check("https://app.example", {}) == expected
