"""3.1.6: verified downloads (resume / Content-Range / MD5), ESA training subset via HTTP range reads, SatNOGS
auth / 429 / empty / undecodable / resume / host checks, FUNcube CSV validation, unified readiness.

A local HTTP server stands in for Zenodo and SatNOGS DB (this machine cannot reach them); it implements real
Range / Content-Range / Retry-After behaviour so the client code is exercised end to end."""
from __future__ import annotations

import hashlib
import importlib
import json
import re
import sys
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from tests import missions_fixtures as fx


@pytest.fixture()
def env(tmp_path, monkeypatch):
    for k in ("DATA", "MODELS", "STORAGE"):
        monkeypatch.setenv(f"SPACEML_{k}_DIR", str(tmp_path / k.lower()))
    monkeypatch.setenv("SPACEML_AUTOSETUP", "off")
    monkeypatch.setenv("SMC_AUTO_INSTALL", "off")
    monkeypatch.delenv("SATNOGS_TOKEN", raising=False)
    monkeypatch.setenv("SPACEML_BUNDLE_DIR", str(tmp_path / "no-bundle"))
    for m in [m for m in sys.modules if m.startswith(("spaceml", "app"))]:
        del sys.modules[m]
    return tmp_path


class Server:
    """Tiny HTTP server: files with optional Range support, plus scripted JSON responses."""

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.ranges = True
        self.ignore_range_once = False
        self.script: dict[str, list] = {}           # path -> list of (status, headers, body)
        self.log: list[tuple[str, dict]] = []
        srv = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):  # noqa: N802
                srv.log.append((self.path, dict(self.headers)))
                path = self.path.split("?")[0]
                if path in srv.script and srv.script[path]:
                    st, hd, body = srv.script[path].pop(0)
                    if callable(body):
                        body = body(self)
                    self.send_response(st)
                    for k, v in hd.items():
                        self.send_header(k, v)
                    self.end_headers()
                    self.wfile.write(body if isinstance(body, bytes) else body.encode())
                    return
                data = srv.files.get(path)
                if data is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                rng = self.headers.get("Range")
                if rng and srv.ranges and not srv.ignore_range_once:
                    a, b = re.match(r"bytes=(\d+)-(\d*)", rng).groups()
                    a, b = int(a), int(b) if b else len(data) - 1
                    chunk = data[a:b + 1]
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {a}-{a + len(chunk) - 1}/{len(data)}")
                    self.send_header("Content-Length", str(len(chunk)))
                    self.send_header("Content-Type", "application/octet-stream")
                    self.end_headers()
                    self.wfile.write(chunk)
                    return
                srv.ignore_range_once = False
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Content-Type", "application/octet-stream")
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture()
def server(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.delenv(k, raising=False)
    s = Server()
    yield s
    s.close()


# ------------------------------------------------------------------ verified, resumable downloads
def test_download_resumes_with_content_range_and_verifies_md5(env, server, tmp_path):
    fetch = importlib.import_module("spaceml.missions.fetch")
    data = bytes(range(256)) * 4000
    server.files["/f.zip"] = data
    dest = tmp_path / "cache" / "f.zip"
    dest.parent.mkdir(parents=True)
    (dest.parent / "f.zip.part").write_bytes(data[:300_000])          # an interrupted earlier attempt
    fetch.download(server.url + "/f.zip", dest, size=len(data), md5=hashlib.md5(data).hexdigest())
    assert dest.read_bytes() == data
    assert server.log[-1][1].get("Range") == "bytes=300000-"
    # cached and verified: a second call does not download again
    n = len(server.log)
    fetch.download(server.url + "/f.zip", dest, size=len(data), md5=hashlib.md5(data).hexdigest())
    assert len(server.log) == n


def test_download_restarts_when_server_ignores_range(env, server, tmp_path):
    fetch = importlib.import_module("spaceml.missions.fetch")
    data = b"abc" * 100_000
    server.files["/g.zip"] = data
    server.ignore_range_once = True
    dest = tmp_path / "g.zip"
    (tmp_path / "g.zip.part").write_bytes(b"XXXX")                     # garbage that must not be kept
    fetch.download(server.url + "/g.zip", dest, md5=hashlib.md5(data).hexdigest())
    assert dest.read_bytes() == data


def test_download_md5_mismatch_is_rejected(env, server, tmp_path):
    fetch = importlib.import_module("spaceml.missions.fetch")
    server.files["/h.zip"] = b"payload" * 1000
    with pytest.raises(fetch.FetchError) as e:
        fetch.download(server.url + "/h.zip", tmp_path / "h.zip", md5="0" * 32)
    assert e.value.kind == "integrity" and not (tmp_path / "h.zip").exists() and not (tmp_path / "h.zip.part").exists()


def test_download_rejects_metadata_instead_of_archive(env, server, tmp_path):
    fetch = importlib.import_module("spaceml.missions.fetch")
    server.script["/api/file"] = [(200, {"Content-Type": "application/json"}, '{"key": "ESA-Mission1.zip"}')]
    with pytest.raises(fetch.FetchError, match="instead of the file"):
        fetch.download(server.url + "/api/file", tmp_path / "x.zip")


# ------------------------------------------------------------------ ESA: subset via range reads -> import -> train -> evaluate
def _esa_archive(tmp_path: Path, n: int = 1) -> bytes:
    src = fx.esa(tmp_path / f"esa{n}", mission=n)
    import io
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                z.write(p, f"ESA-Mission{n}/{p.relative_to(src).as_posix()}")
        import os as _os
        for k in range(4):                                       # big channels the subset must skip
            z.writestr(f"ESA-Mission{n}/channels/channel_9{k}.zip", _os.urandom(6_000_000), zipfile.ZIP_STORED)
    return buf.getvalue()


def _point_esa(monkeypatch, inst, server, n, data):
    name = f"ESA-Mission{n}.zip"
    server.files[f"/records/15237121/files/{name}"] = data
    monkeypatch.setattr(inst, "ESA_URL", server.url + "/records/{record}/files/{name}?download=1")
    monkeypatch.setitem(inst.ESA_FILES, n, (name, hashlib.md5(data).hexdigest(), len(data)))


def test_esa_subset_reads_only_needed_members_then_trains_and_evaluates(env, server, tmp_path, monkeypatch):
    inst = importlib.import_module("spaceml.missions.install")
    data = _esa_archive(tmp_path, 1)
    _point_esa(monkeypatch, inst, server, 1, data)
    out = inst.esa_fetch("esa-m1", "subset", None)
    sel = json.loads((out / "selection.json").read_text())
    assert sel["channels"] == ["channel_41", "channel_42", "channel_43"]
    fetched = sum(min(int(h["Range"][6:].split("-")[1]), len(data) - 1) - int(h["Range"][6:].split("-")[0]) + 1
                  for p, h in server.log if "Range" in h)
    assert not (out / "channels" / "channel_90.zip").exists()
    assert fetched < len(data) * 0.5, f"the subset must not read the whole archive ({fetched} of {len(data)})"
    esa = importlib.import_module("spaceml.missions.esa")
    man = esa.prepare("esa-m1", out, sel["channels"], None, None)
    assert man["source"]["doi"] == "10.5281/zenodo.15237121" and man["source"]["install_mode"] == "subset"
    r = man["training_readiness"]
    assert r["trainable"] and r["test_events"] >= 1 and r["train_events"] >= 1
    assert man["split_at_utc"].startswith("2007-01-01")
    ch = importlib.import_module("spaceml.missions.channels")
    models = importlib.import_module("spaceml.missions.models")
    with models.Staging("esa-m1", True) as d:
        m = ch.train("esa-m1", d, quick=True)
    assert m["headline"] and (models.active("esa-m1") or {}).get("version")
    ev = m.get("evaluation") or m
    assert json.dumps(ev)                                        # metrics are JSON-serialisable and saved


def test_esa_subset_reports_blocker_when_ranges_unsupported(env, server, tmp_path, monkeypatch):
    inst = importlib.import_module("spaceml.missions.install")
    fetch = importlib.import_module("spaceml.missions.fetch")
    _point_esa(monkeypatch, inst, server, 2, _esa_archive(tmp_path, 2))
    server.ranges = False
    with pytest.raises(fetch.FetchError) as e:
        inst.esa_fetch("esa-m2", "subset", None)
    assert e.value.kind == "unsupported" and "Download full archive" in str(e.value)


def test_esa_full_archive_uses_stable_cache_and_md5(env, server, tmp_path, monkeypatch):
    inst = importlib.import_module("spaceml.missions.install")
    data = _esa_archive(tmp_path, 3)
    _point_esa(monkeypatch, inst, server, 3, data)
    out = inst.esa_fetch("esa-m3", "full", None)
    cached = inst.esa_cache("esa-m3") / "ESA-Mission3.zip"
    assert cached.exists() and hashlib.md5(cached.read_bytes()).hexdigest() == inst.ESA_FILES[3][1]
    assert (out / "labels.csv").exists()
    n = len(server.log)
    inst.esa_fetch("esa-m3", "full", None)                       # retry: reuses the verified archive
    assert len(server.log) == n


def test_esa_readiness_blocks_training_without_test_events(env):
    esa = importlib.import_module("spaceml.missions.esa")
    r = esa.readiness([{"start": 1, "end": 2, "category": "Anomaly"}], split_at=10, items=3)
    assert not r["trainable"] and "evaluation period" in r["blocked_reason"]


def test_disk_space_check(env, tmp_path):
    fetch = importlib.import_module("spaceml.missions.fetch")
    with pytest.raises(fetch.FetchError) as e:
        fetch.ensure_space(tmp_path, 10**18)
    assert e.value.kind == "space"


# ------------------------------------------------------------------ SatNOGS
def _sat(monkeypatch, server):
    sat = importlib.import_module("spaceml.missions.satnogs")
    monkeypatch.setattr(sat, "API_HOST", "127.0.0.1")
    monkeypatch.setattr(sat, "API", server.url + "/api/telemetry/")
    orig = sat._safe_next

    def safe(url):  # the test server is http://127.0.0.1 - same host check, scheme relaxed for the test only
        if not url:
            return None
        from urllib.parse import urlparse
        if urlparse(url).hostname != "127.0.0.1":
            return orig(url)
        return url
    monkeypatch.setattr(sat, "_safe_next", safe)
    return sat


def test_satnogs_requires_token_and_reports_auth(env, server, monkeypatch):
    sat = _sat(monkeypatch, server)
    fetch = importlib.import_module("spaceml.missions.fetch")
    with pytest.raises(fetch.FetchError) as e:
        sat.fetch_window(None, "2024-01-01", "2024-01-02")
    assert e.value.kind == "auth" and "SATNOGS_TOKEN" in str(e.value)
    monkeypatch.setenv("SATNOGS_TOKEN", "bad-token-123")
    server.script["/api/telemetry/"] = [(401, {"Content-Type": "application/json"}, '{"detail": "Invalid token."}')]
    with pytest.raises(fetch.FetchError) as e:
        sat.fetch_window(None, "2024-01-01", "2024-01-02")
    assert e.value.kind == "auth" and "DB token" in str(e.value)
    assert server.log[-1][1].get("Authorization") == "Token bad-token-123"


def test_satnogs_429_retry_after_then_resume_without_duplicates(env, server, monkeypatch):
    sat = _sat(monkeypatch, server)
    fetch = importlib.import_module("spaceml.missions.fetch")
    monkeypatch.setenv("SATNOGS_TOKEN", "good-token-456")
    sleeps = []
    monkeypatch.setattr(sat.time, "sleep", lambda s: sleeps.append(s))
    frames = [{"timestamp": f"2024-01-01T00:{i:02d}:00Z", "frame": f"{i:040x}", "norad_cat_id": 43880} for i in range(6)]
    nxt = server.url + "/api/telemetry/?cursor=2"
    server.script["/api/telemetry/"] = [
        (429, {"Retry-After": "7"}, "slow down"),
        (200, {"Link": f'<{nxt}>; rel="next"', "Content-Type": "application/json"}, json.dumps(frames[:3])),
        (429, {"Retry-After": "3600"}, "slow down"),               # long wait -> stop, keep page 1, tell the user when
    ]
    with pytest.raises(fetch.FetchError) as e:
        sat.fetch_window(None, "2024-01-01", "2024-01-02")
    assert e.value.kind == "rate_limited" and "Next attempt allowed after" in str(e.value)
    assert sleeps == [7.0]                                        # honoured Retry-After, bounded
    sat.record_failure(e.value)
    assert sat.blocked_until() and sat.blocked_until() > time.time() + 3000
    jobs = importlib.import_module("app.jobs")
    with pytest.raises(ValueError, match="next attempt is allowed after"):
        jobs.submit("missions.satnogs.install", {})                 # no duplicate fetch while throttled
    sat._write_status(kind="ok", message="", next_attempt_at=None)  # the wait is over
    server.script["/api/telemetry/"] = [(200, {"Content-Type": "application/json"}, json.dumps(frames[2:]))]  # page 2 overlaps page 1
    out = sat.fetch_window(None, "2024-01-01", "2024-01-02")
    got = json.loads(out.read_text())
    assert len(got) == 6 and len({(r["timestamp"], r["frame"]) for r in got}) == 6
    assert "cursor=2" in server.log[-1][0]                       # resumed at the saved cursor


def test_satnogs_refuses_to_forward_token_to_another_host(env, server, monkeypatch):
    sat = _sat(monkeypatch, server)
    fetch = importlib.import_module("spaceml.missions.fetch")
    monkeypatch.setenv("SATNOGS_TOKEN", "good-token-456")
    server.script["/api/telemetry/"] = [(200, {"Link": '<https://evil.example.com/api/telemetry/?cursor=2>; rel="next"'}, "[]")]
    with pytest.raises(fetch.FetchError, match="refusing to send the API token"):
        sat.fetch_window(None, "2024-02-01", "2024-02-02")


def test_satnogs_empty_window_and_undecodable_frames(env, server, monkeypatch, tmp_path):
    sat = _sat(monkeypatch, server)
    fetch = importlib.import_module("spaceml.missions.fetch")
    monkeypatch.setenv("SATNOGS_TOKEN", "good-token-456")
    server.script["/api/telemetry/"] = [(200, {"Content-Type": "application/json"}, "[]")]
    with pytest.raises(fetch.FetchError) as e:
        sat.fetch_window(None, "2024-03-01", "2024-03-02")
    assert e.value.kind == "no_frames" and "earlier window" in str(e.value)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"timestamp": "2024-01-01T00:00:00Z", "frame": "00" * 20, "norad_cat_id": 43880}] * 5))
    with pytest.raises(fetch.FetchError) as e:
        sat.prepare(bad)
    assert e.value.kind == "undecodable" and "nothing was installed" in str(e.value)
    assert not (importlib.import_module("spaceml.missions.store").installed("satnogs"))


def test_satnogs_token_never_in_api_or_job_log(env, monkeypatch):
    monkeypatch.setenv("SATNOGS_TOKEN", "s3cr3t-token-value")
    from fastapi.testclient import TestClient
    main = importlib.import_module("app.main")
    jobs = importlib.import_module("app.jobs")
    with TestClient(main.app) as c:
        st = c.get("/api/missions/satnogs").json()
        assert st["auto_install"]["token_configured"] is True and "s3cr3t" not in json.dumps(st)
        assert "s3cr3t" not in c.get("/api/catalog").text
    assert jobs._scrub("auth Token s3cr3t-token-value failed") == "auth Token *** failed"


def test_install_cmd_accepts_only_known_modes(env):
    jobs = importlib.import_module("app.jobs")
    assert jobs._install_cmd("esa-m1", {"mode": "subset"})[-2:] == ["--mode", "subset"]
    with pytest.raises(ValueError):
        jobs._install_cmd("esa-m1", {"mode": "everything"})
    with pytest.raises(ValueError):
        jobs._install_cmd("hubble", {"mode": "subset"})


# ------------------------------------------------------------------ FUNcube
def test_funcube_follows_downloads_page_and_validates_csv(env, server, monkeypatch, tmp_path):
    inst = importlib.import_module("spaceml.missions.install")
    csv = fx.funcube(tmp_path / "wod.csv").read_text()
    server.files["/wod.html"] = b'<p>see <a href="/downloads.html">the downloads page</a></p>'
    server.files["/downloads.html"] = (b'<a href="/files/WOD_2024-03-08.csv">week 10</a> <a href="/files/WOD_2024-03-01.csv">week 9</a>'
                                       b'<a href="/files/broken_2024-03-09.csv">x</a>')
    server.files["/files/WOD_2024-03-08.csv"] = csv.encode()
    server.files["/files/WOD_2024-03-01.csv"] = csv.encode()
    server.files["/files/broken_2024-03-09.csv"] = b"<!DOCTYPE html><html>not found</html>"
    monkeypatch.setattr(inst, "FUNCUBE_PAGES", (server.url + "/missing.html", server.url + "/wod.html"))
    out = inst._funcube_latest(tmp_path / "dl", 2)
    assert out.exists() and inst.looks_like_telemetry_csv(out.read_text()) is None
    fc = importlib.import_module("spaceml.missions.funcube")
    man = fc.prepare(out)
    assert man["items"] >= 3


def test_funcube_404_page_gives_actionable_error(env, server, monkeypatch, tmp_path):
    inst = importlib.import_module("spaceml.missions.install")
    monkeypatch.setattr(inst, "FUNCUBE_PAGES", (server.url + "/gone.html",))
    with pytest.raises(ValueError, match="HTTP 404.*Upload the source files"):
        inst._funcube_latest(tmp_path / "dl", 2)


# ------------------------------------------------------------------ readiness
def test_one_readiness_definition(env):
    cat = importlib.import_module("app.routers.catalog")
    st = lambda **kw: cat._status({"id": "x", "dataset": kw.get("ds", {"ready": True}), "model": kw.get("m", {"ready": False}),
                                   "train": [{"task": "t", "label": "Quick train"}]})["readiness"]
    assert st(ds={"ready": False}) == "missing"
    assert st(m=None) == "analysis_only"
    assert st(m={"ready": True}) == "model_available"
    assert st() == "ready_to_train"
    assert st(ds={"ready": True, "trainable": False}) == "installed"
    s = cat._summary([{"status": cat._status({"id": "a", "dataset": {"ready": False}, "model": None})}])
    assert s["readiness"]["missing"] == 1 and set(s["readiness_labels"].values()) == {
        "Missing", "Installed", "Ready to train", "Model available", "Analysis only"}


# ------------------------------------------------------------------ 3.1.7 regressions from real installs
def test_esa_subset_reads_deflate64_archives(env, server, tmp_path, monkeypatch):
    """The real ESA-AD zips use Deflate64 (ZIP method 9): 'That compression method is not supported' before 3.1.7."""
    import io
    inst = importlib.import_module("spaceml.missions.install")
    zs = importlib.import_module("spaceml.common.zipsupport")
    assert zs.enable_deflate64()
    src = fx.esa(tmp_path / "esa64", mission=2)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zs.ZIP_DEFLATED64) as z:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                z.write(p, f"ESA-Mission2/{p.relative_to(src).as_posix()}")
    data = buf.getvalue()
    assert all(i.compress_type == 9 for i in zipfile.ZipFile(io.BytesIO(data)).infolist())
    _point_esa(monkeypatch, inst, server, 2, data)
    out = inst.esa_fetch("esa-m2", "subset", None)
    esa = importlib.import_module("spaceml.missions.esa")
    man = esa.prepare("esa-m2", out, json.loads((out / "selection.json").read_text())["channels"], None, None)
    assert man["items"] == 3 and man["counts"]["events"] >= 3


def test_unsupported_method_is_named(env):
    import io
    zs = importlib.import_module("spaceml.common.zipsupport")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.txt", "x")
    zf = zipfile.ZipFile(io.BytesIO(buf.getvalue()))
    zf.getinfo("a.txt").compress_type = 93                       # zstandard: not readable here
    with pytest.raises(ValueError, match="zstandard"):
        zs.check_members(zf, ["a.txt"])


def test_funcube_without_csv_is_a_calm_unavailable(env, server, monkeypatch, tmp_path):
    inst = importlib.import_module("spaceml.missions.install")
    fetch = importlib.import_module("spaceml.missions.fetch")
    server.files["/wod.html"] = b'<p>Whole Orbit Data</p><a href="/rankings.html">rankings</a>'
    monkeypatch.setattr(inst, "FUNCUBE_PAGES", (server.url + "/wod.html",))
    with pytest.raises(fetch.FetchError) as e:
        inst._funcube_latest(tmp_path / "dl", 2)
    assert e.value.kind == "unavailable" and str(e.value).startswith("Not available right now")
    assert inst.auto_source("funcube")["automatic"] is False and inst.auto_source("funcube")["upload_first"]


def test_model_trained_on_other_version_stays_available(env):
    from fastapi.testclient import TestClient
    store = importlib.import_module("spaceml.missions.store")
    models = importlib.import_module("spaceml.missions.models")
    w = store.Writer("opssat")
    w.add({"id": "s1", "label": "s1"}, np.arange(30.0), {"v": np.ones(30)})
    w.publish({"scope": "x"})
    for name, fp in (("v1-quick", "other"), ):
        d = models.ROOT / "opssat" / name
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps({"dataset_fingerprint": fp, "run_kind": "quick"}))
    legacy = models.ROOT / "webb" / "v0-quick"
    legacy.mkdir(parents=True)
    (legacy / "manifest.json").write_text(json.dumps({"run_kind": "quick"}))          # no fingerprint recorded
    w2 = store.Writer("webb"); w2.add({"id": "v", "label": "v"}, np.arange(30.0), {"v": np.ones(30)}); w2.publish({"scope": "y"})
    assert models.active("webb")["stale"] is False
    main = importlib.import_module("app.main")
    with TestClient(main.app) as c:
        st = c.get("/api/missions/opssat").json()["model"]
        assert st["ready"] is True and st["stale"] is True
