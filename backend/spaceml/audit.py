"""End-to-end audit of a deployed Space ML Mission Control (run inside the app container).

    python -m spaceml.audit env              # torch/torchvision pair, pip check, network egress
    python -m spaceml.audit data             # inventory + validation of every local dataset (read-only)
    python -m spaceml.audit ui               # every UI workflow through the same HTTP API the buttons use
    python -m spaceml.audit snapshot         # record datasets / jobs / checkpoints before a restart
    python -m spaceml.audit verify-restart   # compare after `docker compose restart app`
    python -m spaceml.audit report           # merge everything into storage/audit/report.md

Evidence rules: a Ready badge or a queued job is never counted as success. A quick run passes only when its
job succeeded, a model version with artifacts exists, and an inference/evaluation run with that exact
version succeeded. Anything not exercised is written as UNTESTED with the reason.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from .config import DATA_DIR, MODELS_DIR, STORAGE_DIR

OUT = STORAGE_DIR / "audit"
BASE = os.environ.get("SPACEML_AUDIT_BASE", "http://127.0.0.1:8000")

# What the release bundle is meant to contain (release.py defaults + full source sizes), so a local
# subset is never mistaken for the full dataset.
SCOPE = {
    "telemetry": "82 SMAP/MSL channels in the NASA release (81 used by this project), 105 labeled anomaly sequences",
    "kepler": "release default: 1000 stars × 8 quarters (DR24 labeled TCEs on those stars); full DR24 has ~15k labeled TCEs",
    "ai4mars": "release default: 3000 train pairs + all 322 test_min3 images; full AI4Mars NAVCAM train has ~16k labeled images",
    "omni2": "release default: OMNI2 hourly 2003, 2010-2017, 2024",
    "traverse": "Curiosity waypoints (full MSL traverse, ~4k+ waypoints)",
    "flood": "2 GIBS before/after scenes (lake-mead, indus-flood)",
}


# ------------------------------------------------------------------ helpers
def _save(name: str, obj) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.json"
    p.write_text(json.dumps(obj, indent=2, default=str))
    return p


def _load(name: str):
    p = OUT / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _http(method: str, path: str, body: dict | None = None, timeout: float = 120):
    req = urllib.request.Request(BASE + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"content-type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # localhost: never via a proxy
    try:
        with opener.open(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw[:1] in (b"{", b"[") else raw.decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        txt = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(txt)
        except ValueError:
            return e.code, txt


def _say(msg: str) -> None:
    print(msg, flush=True)


# ------------------------------------------------------------------ env
def env() -> dict:
    out: dict = {}
    r = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "check_torch.py")],
                       capture_output=True, text=True)
    try:
        out["torch"] = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        out["torch"] = {"ok": False, "error": (r.stdout + r.stderr)[-600:]}
    pc = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    out["pip_check"] = {"ok": pc.returncode == 0, "output": (pc.stdout + pc.stderr).strip()[-1500:]}
    pf = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    out["packages"] = [l for l in pf.splitlines()
                       if l.lower().split("==")[0] in ("torch", "torchvision", "timm", "segmentation-models-pytorch",
                                                       "segmentation_models_pytorch", "numpy", "datasets")]
    out["pip_constraint"] = os.environ.get("PIP_CONSTRAINT")
    # egress: the audit expects an offline container
    blocked = {}
    for host in ("https://huggingface.co", "https://mast.stsci.edu", "https://download.pytorch.org"):
        try:
            urllib.request.urlopen(host, timeout=6)
            blocked[host] = False
        except Exception as e:  # noqa: BLE001
            blocked[host] = f"blocked ({type(e).__name__})"
    out["egress"] = blocked
    out["offline"] = all(v is not False for v in blocked.values())
    _save("env", out)
    _say(json.dumps(out, indent=2))
    return out


# ------------------------------------------------------------------ data
def _row(module, **kw):
    return {"module": module, "ok": kw.pop("ok", None), **kw}


def _telemetry() -> dict:
    from .telemetry import data
    if not data.is_downloaded():
        return _row("telemetry", ok=False, error="telemetry data not installed")
    ch = data.load_channels()
    by_sc: dict[str, int] = {}
    bad, n_seq, missing = [], 0, []
    for cid, c in ch.items():
        by_sc[c.spacecraft] = by_sc.get(c.spacecraft, 0) + 1
        if not data.has_channel(cid):
            missing.append(cid)
            continue
        tr, te = data.load_arrays(cid)
        if tr.ndim != 2 or te.ndim != 2 or tr.shape[1] != te.shape[1]:
            bad.append(f"{cid}: shapes {tr.shape}/{te.shape}")
        elif not (np.isfinite(tr).all() and np.isfinite(te).all()):
            bad.append(f"{cid}: non-finite values")
        for a, b in c.sequences:
            n_seq += 1
            if not (0 <= a <= b < len(te)):
                bad.append(f"{cid}: anomaly [{a},{b}] outside test length {len(te)}")
    return _row("telemetry", ok=not bad and not missing,
                counts={"channels": len(ch), "by_spacecraft": by_sc, "anomaly_sequences": n_seq,
                        "channels_without_arrays": missing},
                splits="per channel: train array / test array (labels on test)",
                problems=bad[:20], scope=SCOPE["telemetry"],
                subset=len(ch) < 81 or bool(missing))


def _kepler() -> dict:
    from .exoplanet import data
    if not data.TCE_PATH.exists():
        return _row("kepler", ok=False, error="Kepler catalog not installed")
    df = data.tce_table(labeled_only=True)
    views = data.cached_view_ids()
    keys = set(views)
    sel = df[[(int(k), int(p)) in keys for k, p in zip(df.kepid, df.tce_plnt_num)]]
    bad = []
    labels = set(sel.planet.unique().tolist())
    if not labels <= {0, 1}:
        bad.append(f"unexpected labels {labels}")
    nbad = 0
    for k, p in list(views)[:400]:
        try:
            v = data.get_views(int(k), int(p), allow_download=False)
            if v["global"].shape[0] != data.GLOBAL_BINS or v["local"].shape[0] != data.LOCAL_BINS \
                    or not (np.isfinite(v["global"]).all() and np.isfinite(v["local"]).all()):
                nbad += 1
        except Exception:  # noqa: BLE001
            nbad += 1
    if nbad:
        bad.append(f"{nbad} of {min(400, len(views))} checked views unreadable / wrong shape / non-finite")
    # star-disjoint split on the quick subset the UI quick run uses
    split = {}
    try:
        from .exoplanet import train as et
        sub, _, _ = et.load_dataset(80)
        tr, va, te = et.split_by_star(sub)
        s = [set(sub.kepid.iloc[i]) for i in (tr, va, te)]
        split = {"train_stars": len(s[0]), "val_stars": len(s[1]), "test_stars": len(s[2]),
                 "overlap": len((s[0] & s[1]) | (s[0] & s[2]) | (s[1] & s[2]))}
        if split["overlap"]:
            bad.append(f"{split['overlap']} stars appear in two splits")
    except SystemExit:
        bad.append("quick subset could not be split (too few planets / false positives)")
    lc = list((data.LC_DIR).glob("*.npz")) if data.LC_DIR.exists() else []
    targets = (DATA_DIR / "exoplanet" / "targets.json")
    return _row("kepler", ok=not bad,
                counts={"catalog_labeled_tces": int(len(df)), "stars_with_views": len({k for k, _ in views}),
                        "tces_with_views": len(views), "planet_candidates": int(sel.planet.sum()),
                        "false_positives": int((sel.planet == 0).sum()), "raw_lightcurve_files": len(lc),
                        "targets_manifest": targets.exists()},
                splits=split, problems=bad, scope=SCOPE["kepler"], subset=len({k for k, _ in views}) < 1000)


def _ai4mars() -> dict:
    from .mars import local, data as md
    idx = local.index()
    if not idx:
        return _row("ai4mars", ok=False, error="local AI4Mars pairs not installed")
    sp = local.splits()
    bad, classes, dims = [], set(), set()
    for split in sp:
        for p in local.pairs(split):
            try:
                img, lab = local.load_pair(p)
            except Exception as e:  # noqa: BLE001
                bad.append(f"{split}/{p['key']}: {e}")
                continue
            dims.add(img.size)
            if lab.shape != (img.height, img.width):
                bad.append(f"{split}/{p['key']}: image {img.size} vs mask {lab.shape[::-1]}")
            classes |= set(np.unique(lab).tolist())
    if classes - local.VALID:
        bad.append(f"invalid class ids {sorted(classes - local.VALID)}")
    v = local.validate()
    held = v["splits"].get("train", {}).get("held_out", 0)
    tr = {p.get("img_sha") for p in local.usable_pairs("train")}
    te = {p.get("img_sha") for s_ in local.expert_splits() for p in local.usable_pairs(s_)}
    if tr & te:
        bad.append(f"{len(tr & te)} expert test images (by pixel hash) used for training")
    samples = md.SAMPLES_DIR if hasattr(md, "SAMPLES_DIR") else DATA_DIR / "mars" / "samples"
    n_samples = len(list(Path(samples).glob("*/*.png"))) // 2 if Path(samples).exists() else 0
    try:
        n_samples = len(md.list_samples()) if hasattr(md, "list_samples") else n_samples
    except Exception:  # noqa: BLE001
        pass
    return _row("ai4mars", ok=not bad,
                counts={"splits": sp, "expert_samples": n_samples, "image_sizes": sorted(map(list, dims)),
                        "class_ids_present": sorted(classes), "build_stats": idx.get("splits")},
                field_mapping="image ← image; label ← label_mask (rover_mask / range_mask → 255 ignore); "
                              "rows with has_labels = False skipped",
                splits={"validation": v["splits"], "train_test_overlap_by_pixels": len(tr & te),
                        "duplicates_held_out": held, "root": v.get("root")}, problems=bad[:20],
                source={"dataset": idx.get("dataset"), "revision": idx.get("revision")},
                scope=SCOPE["ai4mars"], subset=sp.get("train", 0) < 3000)


def _omni() -> dict:
    from .spaceweather import data
    ys = data.cached_years()
    cov, bad = {}, []
    for y in ys:
        df = data.load(years=[y])
        col = next((c for c in df.columns if c.lower() == "dst"), None)
        dst = df[col] if col else None
        cov[y] = {"hours": int(len(df)), "dst_valid": float(dst.notna().mean()) if dst is not None else 0.0}
        if len(df) < 8000:
            bad.append(f"{y}: only {len(df)} hourly rows")
        if dst is None or dst.notna().mean() < 0.9:
            bad.append(f"{y}: Dst coverage {cov[y]['dst_valid']:.0%}")
    need = {2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2024}
    if need - set(ys):
        bad.append(f"missing years for train/test/replay: {sorted(need - set(ys))}")
    return _row("omni2", ok=not bad, counts={"years": ys, "coverage": cov},
                splits="train 2010-2016 · test 2017 · replay May 2024", problems=bad,
                scope=SCOPE["omni2"], subset=bool(need - set(ys)))


def _traverse() -> dict:
    from .mars import traverse
    st = traverse.status()
    if not st.get("downloaded"):
        return _row("traverse", ok=False, error="waypoints not installed", counts=st)
    wps = traverse.waypoints()
    bad = [] if wps and all(np.isfinite([w["lon"], w["lat"]]).all() for w in wps) else ["waypoints missing/invalid"]
    secs = traverse.sections(wps)
    return _row("traverse", ok=not bad, counts={"waypoints": len(wps), "sols": [wps[0]["sol"], wps[-1]["sol"]],
                                                "sections": len(secs)},
                splits="n/a (analysis only)", problems=bad, scope=SCOPE["traverse"], subset=len(wps) < 1000)


def _flood() -> dict:
    import zipfile
    d = DATA_DIR / "workbench_samples"
    zs = sorted(d.glob("*.zip")) if d.exists() else []
    bad, scenes = [], {}
    for z in zs:
        with zipfile.ZipFile(z) as f:
            imgs = [n for n in f.namelist() if n.lower().endswith((".png", ".jpg", ".tif", ".tiff"))]
        scenes[z.stem] = len(imgs)
        if len(imgs) < 2:
            bad.append(f"{z.name}: needs a before and an after image, has {len(imgs)}")
    if not zs:
        bad.append("no before/after scenes installed")
    return _row("flood", ok=not bad, counts={"scenes": scenes}, splits="n/a (analysis only)",
                problems=bad, scope=SCOPE["flood"], subset=len(zs) < 2)


def data_audit() -> dict:
    from . import release
    rows = []
    for f in (_telemetry, _kepler, _ai4mars, _omni, _traverse, _flood):
        t0 = time.time()
        try:
            r = f()
        except Exception as e:  # noqa: BLE001
            r = _row(f.__name__.strip("_"), ok=False, error=f"{type(e).__name__}: {e}")
        r["seconds"] = round(time.time() - t0, 1)
        rows.append(r)
        _say(f"[{'ok' if r.get('ok') else 'PROBLEM'}] {r['module']}: {json.dumps(r.get('counts', r.get('error')))[:300]}")
    st = release.status()
    man = {}
    if st.get("present"):
        try:
            m = json.loads((Path(st["bundle_dir"]) / "manifest.json").read_text())
            man = {d["id"]: {"files": d.get("file_count"), "bytes": d.get("bytes"), "details": d.get("details")}
                   for d in m.get("datasets", [])}
        except Exception as e:  # noqa: BLE001
            man = {"error": str(e)}
    out = {"rows": rows, "release": {k: st.get(k) for k in ("present", "bundle_version", "total_bytes", "installed")},
           "manifest": man}
    _save("data", out)
    return out


# ------------------------------------------------------------------ ui (through the HTTP API the buttons use)
TRAINING_MARKERS = ("epoch", "channels)", "rmse")   # messages that only appear once real training runs


def _wait_job(jid: str, limit_s: float, until_progress: bool = False) -> dict:
    t0, samples, last = time.time(), [], None
    while time.time() - t0 < limit_s:
        code, j = _http("GET", f"/api/jobs/{jid}?lines=40")
        if code != 200:
            return {"status": f"http {code}", "samples": samples}
        key = (round(float(j.get("progress") or 0), 3), j.get("message"))
        if key != last:
            samples.append({"t": round(time.time() - t0), "progress": key[0], "message": (key[1] or "")[:140]})
            last = key
        if j["status"] not in ("queued", "running"):
            break
        if until_progress and j["status"] == "running" and any(
                any(m in s["message"].lower() for m in TRAINING_MARKERS) for s in samples):
            break
        time.sleep(3)
    j["samples"] = samples
    j["elapsed_s"] = round(time.time() - t0)
    return j


def _start(task: str, params: dict) -> dict:
    code, j = _http("POST", "/api/jobs", {"task": task, "params": params})
    if code != 200:
        raise RuntimeError(f"start {task}: HTTP {code} {j}")
    return j


def _run(module: str, inp: dict, version: str | None = None, limit_s: float = 600) -> dict:
    body = {"module": module, "input": inp}
    if version:
        body["model_version"] = version
    code, r = _http("POST", "/api/runs", body)
    if code != 200:
        return {"status": f"http {code}", "error": r}
    rid, t0 = r["id"], time.time()
    while time.time() - t0 < limit_s:
        code, r = _http("GET", f"/api/runs/{rid}")
        if r.get("status") not in ("queued", "running"):
            break
        time.sleep(2)
    return r


def _versions(module: str) -> list[dict]:
    d = MODELS_DIR / module
    out = []
    for v in sorted(d.iterdir()) if d.exists() else []:
        if v.is_dir() and not v.name.startswith("."):
            out.append({"version": v.name, "files": sorted(p.name for p in v.iterdir())[:12],
                        "manifest": json.loads((v / "manifest.json").read_text()) if (v / "manifest.json").exists() else {},
                        "metrics": json.loads((v / "metrics.json").read_text()) if (v / "metrics.json").exists() else None})
    return out


def _new_version(module: str, before: set[str]) -> dict | None:
    new = [v for v in _versions(module) if v["version"] not in before]
    return new[-1] if new else None


def _brief_job(j: dict) -> dict:
    return {k: j.get(k) for k in ("id", "task", "status", "error", "elapsed_s", "command")} | {
        "progress_updates": len(j.get("samples", [])),
        "last_messages": [s["message"] for s in j.get("samples", [])[-4:]]}


def _quick_telemetry(ts: str) -> dict:
    from .telemetry import data
    ch = data.load_channels()
    pick = []
    for sc, pref in (("SMAP", ["P-1", "E-1", "A-1"]), ("MSL", ["M-6", "C-1", "T-8"])):
        cands = [c for c in pref if c in ch and data.has_channel(c)] or \
                [c for c, v in ch.items() if v.spacecraft == sc and v.sequences and data.has_channel(c)]
        if cands:
            pick.append(cands[0])
    ver = f"audit-{ts}"
    j = _wait_job(_start("telemetry.train", {"mode": "fast", "channels": pick, "version": ver})["id"], 60 * 60)
    vd = MODELS_DIR / "telemetry" / ver
    arts = sorted(p.name for p in vd.iterdir()) if vd.exists() else []
    inf = [_run("telemetry", {"chan_id": c}, ver) for c in pick] if j["status"] == "succeeded" else []
    ok = j["status"] == "succeeded" and all(f"{c}.pt" in arts for c in pick) and all(
        r.get("status") == "succeeded" for r in inf)
    return {"ok": ok, "channels": pick, "version": ver, "job": _brief_job(j), "artifacts": arts,
            "inference": [{"chan": c, "status": r.get("status"), "summary": r.get("summary"), "error": r.get("error")}
                          for c, r in zip(pick, inf)]}


def _quick_generic(module: str, task: str, params: dict, inference, limit_s: float) -> dict:
    before = {v["version"] for v in _versions(module)}
    j = _wait_job(_start(task, params)["id"], limit_s)
    v = _new_version(module, before)
    out = {"ok": False, "job": _brief_job(j), "version": v and v["version"],
           "artifacts": v and v["files"], "metrics": v and (v["metrics"] or {}).get("headline")}
    if j["status"] == "succeeded" and v:
        r = inference(v["version"])
        out["inference"] = {"status": r.get("status"), "summary": r.get("summary"), "error": r.get("error"),
                            "run_id": r.get("id")}
        out["ok"] = r.get("status") == "succeeded"
    return out


def _full(module: str, task: str, params: dict, wait_full: bool, limit_s: float, must_not: list[str]) -> dict:
    """Start the Full Training control. Pass = it trains on the full local split: it finished successfully,
    or (without --wait-full) it reached real training progress (an epoch / a trained channel) and was then
    cancelled. A download attempt, a failure or no training progress is a FAIL."""
    j0 = _start(task, params)
    cmd = (j0.get("command") or "")
    scope_ok = not any(m in cmd for m in must_not)
    downloads = any(d in cmd for d in ("spaceml.mars.local build", "spaceml.exoplanet.data", "spaceml.spaceweather.data"))
    j = _wait_job(j0["id"], limit_s if wait_full else 45 * 60, until_progress=not wait_full)
    trained = any(any(m in s["message"].lower() for m in TRAINING_MARKERS) for s in j["samples"])
    res = {"command": cmd, "uses_full_local_split": scope_ok and not downloads, "job": _brief_job(j)}
    if j["status"] in ("queued", "running"):
        _http("POST", f"/api/jobs/{j['id']}/cancel")
        res["result"] = ("started and reached training progress, then cancelled by the audit (full run NOT completed)"
                         if trained else "started; no training progress within the window (cancelled)")
        res["completed"] = False
        res["ok"] = trained and res["uses_full_local_split"]
    else:
        res["result"] = f"finished: {j['status']}" + (f" - {j.get('error')}" if j.get("error") else "")
        res["completed"] = j["status"] == "succeeded"
        res["ok"] = res["completed"] and res["uses_full_local_split"]
    return res


def ui_audit(wait_full: bool = False, skip_full: bool = False, only: list[str] | None = None,
             skip_quick: bool = False) -> dict:
    ts = time.strftime("%Y%m%d-%H%M%S")
    code, cat = _http("GET", "/api/catalog")
    if code != 200:
        raise SystemExit(f"API not reachable at {BASE}: {code} {cat}")
    res: dict = _load("ui") or {}
    res.update({"base": BASE, "started": ts,
                 "catalog": {e["id"]: {"dataset_ready": e["dataset"]["ready"],
                                       "model_ready": (e.get("model") or {}).get("ready"),
                                       "model_version": (e.get("model") or {}).get("version")}
                             for e in cat["datasets"]}})
    _save("ui", res)
    want = lambda m: not only or m in only  # noqa: E731

    def step(name, fn):
        _say(f"== {name}")
        t0 = time.time()
        try:
            r = fn()
        except Exception as e:  # noqa: BLE001
            r = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        r["seconds"] = round(time.time() - t0)
        res[name] = r
        _save("ui", res)
        _say(f"   -> {'PASS' if r.get('ok') else 'FAIL'} {json.dumps(r, default=str)[:400]}")

    if want("telemetry") and not skip_quick:
        step("telemetry.quick", lambda: _quick_telemetry(ts))
    if want("kepler") and not skip_quick:
        from .exoplanet import data as ed

        def exo_inf(ver):
            df = ed.tce_table(labeled_only=True)
            k, p = next(iter(sorted(ed.cached_view_ids())))
            return _run("exoplanet", {"kepid": int(k), "tce_plnt_num": int(p)}, ver)
        step("kepler.quick", lambda: _quick_generic("exoplanet", "exoplanet.train",
                                                    {"mode": "quick", "max_stars": 80, "epochs": 40}, exo_inf, 60 * 60))
    if want("ai4mars") and not skip_quick:
        def mars_inf(ver):
            code, s = _http("GET", "/api/mars/samples")
            items = s if isinstance(s, list) else (s or {}).get("samples", [])
            sid = items[0]["id"] if items else None
            if not sid:
                return {"status": "no samples", "error": s}
            r = _run("mars", {"sample_id": sid}, ver)
            return r

        def mars_quick():
            r = _quick_generic("mars", "mars.train", {"mode": "quick"}, mars_inf, 3 * 60 * 60)
            v = next((x for x in _versions("mars") if x["version"] == r.get("version")), None)
            if v:
                r["checkpoint"] = "model.pt" in v["files"]
                r["eval_on_expert_test"] = (v["metrics"] or {}).get("eval_split")
                r["encoder_init"] = v["manifest"].get("encoder_init", "random (pre-3.1 manifest)")
                r["train_test_overlap"] = v["manifest"].get("train_test_overlap")
                r["ok"] = bool(r["ok"] and r["checkpoint"])
            return r
        step("ai4mars.quick", mars_quick)
    if want("omni2") and not skip_quick:
        def sw():
            r = _quick_generic("spaceweather", "spaceweather.train", {},
                               lambda ver: _run("spaceweather", {"event_id": "2024-05"}, ver), 60 * 60)
            code, reps = _http("GET", "/api/spaceweather/replays")
            r["replays"] = reps if code == 200 else f"http {code}"
            m = r.get("metrics") or {}
            r["beats_persistence"] = (m.get("model", 0) > m.get("baseline", 0)) if m else None
            return r
        step("omni2.quick", sw)
    if want("traverse"):
        def tv():
            code, b = _http("GET", "/api/mars/traverse")
            n = len(((b or {}).get("waypoints") or {}).get("sol", [])) if code == 200 else 0
            c2, sec = _http("GET", "/api/mars/traverse/sections")
            ns = len(sec) if isinstance(sec, list) else len((sec or {}).get("sections", [])) if c2 == 200 else 0
            return {"ok": code == 200 and n > 0 and c2 == 200, "waypoints_rendered": n,
                    "sections": ns, "sections_note": None if ns else
                    "0 analysed raw images: section statistics need NASA raw images (fetched online when a "
                    "waypoint is opened) - UNTESTED offline",
                    "status": (b or {}).get("status")}
        step("traverse.render", tv)
    if want("flood"):
        def fl():
            d = DATA_DIR / "workbench_samples"
            zs = sorted(p.name for p in d.glob("*.zip")) if d.exists() else []
            if not zs:
                return {"ok": False, "error": "no bundled scene"}
            code, ds = _http("POST", f"/api/workbench/samples/{zs[0]}")
            if code != 200:
                return {"ok": False, "error": ds}
            code, meta = _http("GET", f"/api/workbench/datasets/{ds['id']}")
            imgs = [i.get("id") for i in (meta.get("images") or meta.get("files") or []) if isinstance(i, dict)]
            if len(imgs) < 2:
                return {"ok": False, "error": f"dataset has {len(imgs)} images", "meta_keys": list(meta)}
            r = _run("workbench", {"dataset_id": ds["id"], "analysis": "image.change",
                                   "params": {"before": imgs[0], "after": imgs[1], "pixel_size_m": 250}})
            res_ = r.get("result") or {}
            summ = r.get("summary") or {}
            area = res_.get("changed_km2") or res_.get("area_km2") or summ.get("changed_km2") or (
                summ.get("headline", "").split("·")[-1].strip() if "km" in summ.get("headline", "") else None)
            files = r.get("files") or res_.get("files")
            outs = r.get("files") or res_.get("files") or {}
            if r.get("status") == "succeeded":
                c3, _ = _http("GET", f"/api/runs/{r['id']}/files/{(outs or {}).get('change', 'change.png')}")
            else:
                c3 = None
            return {"ok": r.get("status") == "succeeded" and area is not None and c3 == 200, "scene": zs[0],
                    "run_id": r.get("id"), "overlay_http": c3,
                    "summary": r.get("summary"), "area_km2": area, "outputs": files, "error": r.get("error")}
        step("flood.change", fl)
    if want("review"):
        def rv():
            code, q = _http("GET", "/api/reviews/queue")
            items = (q or {}).get("items", []) if code == 200 else []
            if not items:
                return {"ok": False, "error": f"queue empty (HTTP {code})", "pending": (q or {}).get("pending")}
            it = items[0]
            c2, rv = _http("PUT", f"/api/runs/{it['run_id']}/reviews/{urllib.parse.quote(it['item'], safe='')}",
                           {"verdict": it.get("hint") or "unsure", "note": "audit"})
            c3, q2 = _http("GET", "/api/reviews/queue")
            return {"ok": code == 200 and c2 == 200 and (q2 or {}).get("pending", 0) == q["pending"] - 1,
                    "pending_before": q["pending"], "pending_after": (q2 or {}).get("pending"),
                    "reviewed": f"{it['module']} {it['item']}", "verdict_http": c2}
        step("review.queue", rv)

    if not skip_full:
        fulls = [("telemetry", "telemetry.retrain", {}, ["--channels"]),
                 ("exoplanet", "exoplanet.train", {"mode": "full", "epochs": 60}, ["--max-stars"]),
                 ("mars", "mars.train", {"mode": "full"}, ["--quick", "--max-train"]),
                 ("spaceweather", "spaceweather.train", {}, [])]
        for mod, task, params, must_not in fulls:
            if only and mod not in only and {"exoplanet": "kepler", "mars": "ai4mars", "spaceweather": "omni2"}.get(mod) not in only:
                continue
            step(f"{mod}.full", lambda mod=mod, task=task, params=params, must_not=must_not:
                 _full(mod, task, params, wait_full or mod in ("exoplanet", "spaceweather"), 12 * 60 * 60, must_not))
    _save("ui", res)
    return res


# ------------------------------------------------------------------ persistence
def _state() -> dict:
    code, cat = _http("GET", "/api/catalog")
    code2, jobs = _http("GET", "/api/jobs?limit=500")
    jl = jobs if isinstance(jobs, list) else (jobs or {}).get("jobs", [])
    ck = sorted(str(p.relative_to(MODELS_DIR)) for p in MODELS_DIR.rglob("*")
                if p.suffix in (".pt", ".joblib", ".json") and p.is_file())
    return {"datasets": {e["id"]: e["dataset"]["ready"] for e in cat.get("datasets", [])} if code == 200 else None,
            "models": {e["id"]: (e.get("model") or {}).get("version") for e in cat.get("datasets", [])} if code == 200 else None,
            "job_ids": sorted(j["id"] for j in jl), "artifacts": ck, "time": time.time()}


def snapshot() -> dict:
    s = _state()
    _save("snapshot", s)
    _say(f"snapshot: {len(s['job_ids'])} jobs, {len(s['artifacts'])} model files, datasets {s['datasets']}")
    return s


def verify_restart() -> dict:
    a, b = _load("snapshot"), _state()
    if not a:
        raise SystemExit("run `snapshot` before the restart")
    code, jobs = _http("GET", "/api/jobs?limit=500")
    jl = jobs if isinstance(jobs, list) else (jobs or {}).get("jobs", [])
    new = [j for j in jl if j["id"] not in set(a["job_ids"])]
    started_by_startup = [f"{j['task']} ({j['status']})" for j in new if j["task"] != "release.install"]
    out = {"datasets_same": a["datasets"] == b["datasets"], "models_same": a["models"] == b["models"],
           "jobs_kept": set(a["job_ids"]) <= set(b["job_ids"]),
           "artifacts_kept": set(a["artifacts"]) <= set(b["artifacts"]),
           "missing_artifacts": sorted(set(a["artifacts"]) - set(b["artifacts"]))[:20],
           "jobs_started_by_startup": started_by_startup,
           "failed_jobs_with_retry": [{"id": j["id"], "task": j["task"], "status": j["status"],
                                       "error": (j.get("error") or "")[:160]}
                                      for j in jl if j["status"] in ("failed", "interrupted", "source_unavailable")][:10]}
    out["ok"] = all(out[k] for k in ("datasets_same", "models_same", "jobs_kept", "artifacts_kept")) \
        and not started_by_startup
    _save("restart", out)
    _say(json.dumps(out, indent=2))
    return out


# ------------------------------------------------------------------ report
def _cell(x) -> str:
    return str(x).replace("|", "/").replace("\n", " ")[:260]


def report() -> str:
    e, d, u, r = _load("env") or {}, _load("data") or {}, _load("ui") or {}, _load("restart")
    rows = {x["module"]: x for x in d.get("rows", [])}
    qmap = {"telemetry": "telemetry.quick", "kepler": "kepler.quick", "ai4mars": "ai4mars.quick",
            "omni2": "omni2.quick", "traverse": "traverse.render", "flood": "flood.change"}
    fmap = {"telemetry": "telemetry.full", "kepler": "exoplanet.full", "ai4mars": "mars.full", "omni2": "spaceweather.full"}
    L = ["# Space ML Mission Control - audit report", ""]
    t = e.get("torch") or {}
    L += [f"**Runtime:** torch {t.get('torch')} · torchvision {t.get('torchvision')} · build types {t.get('build_types')} · "
          f"nms {t.get('nms')} · smp {t.get('smp')} · pip check {'OK' if (e.get('pip_check') or {}).get('ok') else 'FAILED'} · "
          f"offline {e.get('offline')} {e.get('egress')}", ""]
    L += ["| Module | Dataset (count · split) | Scope vs release | Offline load | UI workflow | Quick run | Full run | Artifacts / errors |",
          "|---|---|---|---|---|---|---|---|"]
    for m in ("telemetry", "kepler", "ai4mars", "omni2", "traverse", "flood"):
        x = rows.get(m, {})
        q = u.get(qmap[m]) if u else None
        f = u.get(fmap[m]) if (u and m in fmap) else None
        ds = _cell(json.dumps(x.get("counts", x.get("error", "UNTESTED"))))
        sc = ("LOCAL SUBSET - " if x.get("subset") else "") + _cell(x.get("scope", ""))
        load = "UNTESTED" if not x else ("OK" if x.get("ok") else "PROBLEM: " + _cell(x.get("problems") or x.get("error")))
        wf = "UNTESTED" if q is None else ("PASS" if q.get("ok") else "FAIL")
        if q is None:
            qr = "UNTESTED"
        elif m in ("traverse", "flood"):
            qr = "n/a (analysis only)"
        else:
            j = q.get("job") or {}
            qr = f"{j.get('status')} in {j.get('elapsed_s')} s, {j.get('progress_updates')} progress updates; " \
                 f"inference {((q.get('inference') or {}) if isinstance(q.get('inference'), dict) else q.get('inference'))}"
        if m not in fmap:
            fr = "n/a"
        elif f is None:
            fr = "UNTESTED"
        else:
            fr = (("PASS - " if f.get("ok") else "FAIL - ") + f"{f.get('result')}; full local split: "
                  f"{f.get('uses_full_local_split')}; completed: {f.get('completed')}")
        art = "UNTESTED" if q is None else (q.get("error") or q.get("artifacts") or q.get("outputs") or
                                            q.get("waypoints_rendered") or "")
        if q and q.get("version"):
            art = f"{q['version']}: {art}; metrics {q.get('metrics')}"
        L.append("| " + " | ".join(map(_cell, (m, ds, sc, load, wf, qr, fr, art))) + " |")
    L += ["", "**Restart persistence:** " + ("UNTESTED (run snapshot → restart → verify-restart)" if r is None else
                                             _cell(json.dumps(r))), ""]
    rv = (u or {}).get("review.queue")
    L += [f"**Review Studio queue:** {'UNTESTED' if rv is None else _cell(rv)}", ""]
    txt = "\n".join(L)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.md").write_text(txt)
    _say(txt)
    return txt


def main() -> None:
    global BASE
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["env", "data", "ui", "snapshot", "verify-restart", "report", "all"])
    p.add_argument("--base", default=BASE)
    p.add_argument("--wait-full", action="store_true", help="let every full training run finish (hours)")
    p.add_argument("--skip-full", action="store_true")
    p.add_argument("--skip-quick", action="store_true", help="only analysis workflows + full-training checks")
    p.add_argument("--only", nargs="+", help="telemetry kepler ai4mars omni2 traverse flood review")
    a = p.parse_args()
    BASE = a.base.rstrip("/")
    if a.cmd in ("env", "all"):
        env()
    if a.cmd in ("data", "all"):
        data_audit()
    if a.cmd in ("ui", "all"):
        ui_audit(a.wait_full, a.skip_full, a.only, a.skip_quick)
    if a.cmd == "snapshot":
        snapshot()
    if a.cmd == "verify-restart":
        verify_restart()
    if a.cmd in ("report", "all"):
        report()


if __name__ == "__main__":
    main()
