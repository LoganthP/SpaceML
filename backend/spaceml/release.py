"""Release bundle: every dataset the app uses, prepared once, installed offline.

Prepare (once, on a machine with internet - e.g. `docker compose --profile tools run --rm tools ...`):

    python -m spaceml.release build --out /release/bundle                # download + preprocess + package
    python -m spaceml.release build --out /release/bundle --from-data    # package what is already downloaded
    python -m spaceml.release verify --bundle /release/bundle

Install (automatic at startup when a bundle is mounted at /bundle, or Data & Training → Install):

    python -m spaceml.release install --bundle /bundle

Bundle layout:
    manifest.json                       bundle version, per dataset: source, revision, splits, date ranges,
                                        targets, file count, bytes, sha256 of every file, attribution, terms
    datasets/<id>/<path inside data/>   plain files (verifiable one by one; nothing to decompress)

Installing copies only files that are missing or differ (checksum) into the app's data volume;
the bundle itself is never modified. Datasets whose redistribution terms are not confirmed are
marked `redistribution: "verify"` in the manifest - resolve that before publishing the bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .common.progress import fail, report, step
from .common.utils import now_iso, read_json, write_json
from .config import DATA_DIR, MODELS_DIR, STORAGE_DIR

BUNDLE_DIR = Path(os.environ.get("SPACEML_BUNDLE_DIR", "/bundle"))
STATE = STORAGE_DIR / "release.json"
SCHEMA = 1

# what each dataset is, where it lives under data/, and its terms
DATASETS = {
    "telemetry": {
        "title": "SMAP & MSL spacecraft telemetry with expert-labeled anomalies",
        "module": "telemetry", "paths": ["telemetry"],
        "source": "NASA SMAP/MSL telemetry, Hundman et al. (KDD 2018), telemanom release",
        "source_url": "https://github.com/khundman/telemanom",
        "attribution": "Hundman, Constantinou, Laporte, Colwell, Soderstrom (2018). Detecting Spacecraft Anomalies "
                       "Using LSTMs and Nonparametric Dynamic Thresholding. KDD.",
        "terms": "Data released with telemanom (Apache-2.0 repository); NASA data", "redistribution": "yes"},
    "kepler": {
        "title": "Kepler Q1-Q17 DR24 TCE catalog + folded long-cadence light curves",
        "module": "exoplanet", "paths": ["exoplanet/q1_q17_dr24_tce.csv", "exoplanet/views", "exoplanet/targets.json"],
        "source": "NASA Exoplanet Archive (q1_q17_dr24_tce) + MAST Kepler long-cadence light curves",
        "source_url": "https://exoplanetarchive.ipac.caltech.edu",
        "attribution": "This research uses the NASA Exoplanet Archive (Caltech/IPAC, NASA) and data from MAST (STScI).",
        "terms": "NASA public data; acknowledge the archives", "redistribution": "yes"},
    "ai4mars": {
        "title": "AI4Mars terrain labels (Curiosity NAVCAM) - train pairs + expert test split",
        "module": "mars", "paths": ["mars/ai4mars", "mars/samples"],
        "source": "AI4Mars (NASA/JPL, Swan et al. 2021), Hugging Face mirror or data.nasa.gov release",
        "source_url": "https://data.nasa.gov/Space-Science/AI4MARS-A-Dataset-for-Terrain-Aware-Autonomous-Driv/cykx-2qix",
        "attribution": "Swan et al. (2021). AI4MARS: A Dataset for Terrain-Aware Autonomous Driving on Mars. CVPR Workshops.",
        "terms": "Check the license of the release you packaged (data.nasa.gov page / mirror card)",
        "redistribution": "verify"},
    "omni2": {
        "title": "OMNI2 hourly solar wind + geomagnetic indices",
        "module": "spaceweather", "paths": ["spaceweather"],
        "source": "NASA SPDF OMNIWeb low-resolution OMNI2", "source_url": "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/",
        "attribution": "OMNI data: J.H. King, N. Papitashvili, NASA/GSFC SPDF OMNIWeb.",
        "terms": "NASA public data; acknowledge OMNIWeb", "redistribution": "yes"},
    "traverse": {
        "title": "Curiosity traverse waypoints (+ cached Navcam image lists)",
        "module": "mars", "paths": ["mars/traverse"],
        "source": "NASA/JPL MMGIS mission map (MSL_waypoints) and raw-image API",
        "source_url": "https://mars.nasa.gov/maps/location/?mission=Curiosity",
        "attribution": "NASA/JPL-Caltech", "terms": "NASA/JPL public imagery and data", "redistribution": "yes"},
    "flood": {
        "title": "Before/after satellite scenes for change detection (Lake Mead, Indus flood)",
        "module": "workbench", "paths": ["workbench_samples"],
        "source": "NASA GIBS MODIS Terra corrected reflectance (WMS)", "source_url": "https://gibs.earthdata.nasa.gov",
        "attribution": "Imagery: NASA Worldview / GIBS, part of NASA EOSDIS.",
        "terms": "NASA EOSDIS imagery, no restrictions; acknowledge GIBS", "redistribution": "yes"},
}
# Multi-mission telemetry: one bundle entry per source; included only when prepared data is installed.
from .missions import catalog as _mcat  # noqa: E402
for _m in _mcat.ORDER:
    _src = _mcat.MISSIONS[_m]["source"]
    DATASETS[f"mission-{_m}"] = {
        "title": f"{_mcat.MISSIONS[_m]['name']} - prepared telemetry", "module": "missions",
        "paths": [f"missions/{_m}"], "source": _src["title"], "source_url": _src["url"],
        "attribution": _src.get("authors"), "terms": _src.get("license"), "redistribution": _src.get("redistribution"),
        "optional": True}


MODEL_SETS = {
    "models": {"title": "Trained models shipped with this release (active version per module)", "module": "all",
               "root": "models", "source": "trained by the project on the datasets in this bundle",
               "source_url": "", "attribution": "Space ML Mission Control", "terms": "project output",
               "redistribution": "yes"},
}


def _root(meta: dict) -> Path:
    return MODELS_DIR if meta.get("root") == "models" else DATA_DIR


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _files(root: Path, rels: list[str]) -> list[Path]:
    out = []
    for r in rels:
        p = root / r
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out += [f for f in sorted(p.rglob("*")) if f.is_file() and not f.name.endswith((".tmp", ".installing", ".part"))]
    return out


# ------------------------------------------------------------------ describing what is on disk
def _describe(ds: str) -> dict:
    """Splits, date ranges, targets... read from the data itself (not hard-coded)."""
    try:
        if ds == "telemetry":
            from .telemetry import data
            ch = data.load_channels()
            return {"channels": len(ch), "labeled_anomalies": sum(len(c.sequences) for c in ch.values()),
                    "spacecraft": sorted({c.spacecraft for c in ch.values()})}
        if ds == "kepler":
            from .exoplanet import data
            df = data.tce_table(labeled_only=False)
            views = data.cached_view_ids()
            targets = read_json(DATA_DIR / "exoplanet" / "targets.json", {}) or {}
            return {"tces_in_catalog": len(df), "tces_with_views": len(views),
                    "stars_with_views": len({k for k, _ in views}), "labels": "PC vs AFP/NTP (av_training_set)",
                    "quarters_per_star": targets.get("max_quarters"), "targets_file": "exoplanet/targets.json"}
        if ds == "ai4mars":
            from .mars import local
            idx = local.index() or {}
            return {"dataset": idx.get("dataset"), "revision": idx.get("revision"), "image_size": idx.get("size"),
                    "splits": local.splits(), "skipped": {k: v.get("skipped") for k, v in idx.get("splits", {}).items()}}
        if ds == "omni2":
            from .spaceweather import data
            ys = data.cached_years()
            return {"years": ys, "date_range": f"{min(ys)}-01-01 .. {max(ys)}-12-31" if ys else None, "cadence": "hourly"}
        if ds.startswith("mission-"):
            from .missions import store as mstore
            man = mstore.manifest(ds.removeprefix("mission-")) or {}
            return {"source": man.get("source", {}).get("name"), "source_version": man.get("source", {}).get("version"),
                    "doi": man.get("source", {}).get("doi"), "scope": man.get("scope"), "counts": man.get("counts"),
                    "date_range": man.get("date_range"), "time_basis": man.get("time_basis"),
                    "source_files": man.get("source", {}).get("files"), "split": man.get("split_protocol")}
        if ds == "traverse":
            from .mars import traverse
            w = read_json(traverse.WAYPOINTS_PATH, {}) or {}
            sols = [f["properties"].get("sol") for f in w.get("features", []) if f.get("properties")]
            sols = [s for s in sols if isinstance(s, (int, float))]
            return {"waypoints": len(sols), "sol_range": [min(sols), max(sols)] if sols else None}
        if ds == "flood":
            return {"scenes": sorted(p.name for p in (DATA_DIR / "workbench_samples").glob("*.zip"))}
    except Exception as e:  # noqa: BLE001
        return {"describe_error": str(e)[:200]}
    return {}


# ------------------------------------------------------------------ build
def _prepare_from_network(a) -> None:
    """Download and preprocess everything into data/ (each step is resumable)."""
    def run(*cmd):
        step(" ".join(cmd))
        r = subprocess.run([sys.executable, "-m", *cmd], cwd=Path(__file__).resolve().parents[1])
        if r.returncode != 0:
            fail(f"Release preparation step failed: python -m {' '.join(cmd)}")
    run("spaceml.telemetry.data", "download")
    run("spaceml.exoplanet.data", "tce", "--if-missing")
    run("spaceml.exoplanet.data", "lightcurves", "--n-stars", str(a.kepler_stars), "--workers", "6",
        "--max-quarters", str(a.kepler_quarters))
    write_json(DATA_DIR / "exoplanet" / "targets.json", _kepler_targets(a.kepler_quarters))
    run("spaceml.mars.local", "build", "--train", str(a.ai4mars_train), "--test", "all")
    run("spaceml.mars.data", "samples", "--n", "36")
    run("spaceml.spaceweather.data", "download", "--years", *[str(y) for y in a.omni_years])
    run("spaceml.mars.traverse", "download")
    from .workbench import samples
    samples.OUT = DATA_DIR / "workbench_samples"
    samples.OUT.mkdir(parents=True, exist_ok=True)
    for ev in samples.EVENTS:
        try:
            samples.imagery(ev)
        except Exception as e:  # noqa: BLE001
            print(f"(change-detection scene {ev} skipped: {e})")


def _kepler_targets(max_quarters: int | None) -> dict:
    from .exoplanet import data
    v = sorted(data.cached_view_ids())
    return {"created_at": now_iso(), "max_quarters": max_quarters, "stars": sorted({k for k, _ in v}),
            "tces": [f"{k}_{p}" for k, p in v]}


def build(out: Path, from_data: bool, a=None) -> dict:
    if not from_data:
        _prepare_from_network(a)
    elif (DATA_DIR / "exoplanet" / "views").exists() and not (DATA_DIR / "exoplanet" / "targets.json").exists():
        write_json(DATA_DIR / "exoplanet" / "targets.json", _kepler_targets(None))
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": SCHEMA, "bundle_version": datetime.now(timezone.utc).strftime("%Y.%m.%d-%H%M"),
                "created_at": now_iso(), "datasets": []}
    sets = dict(DATASETS)
    if a is not None and getattr(a, "include_models", False):
        from app import registry  # noqa: WPS433 - packaging helper, runs inside the backend
        act = [f"{m}/{v['version']}" for m in registry.MODULES if (v := registry.active(m))]
        from .missions import models as mmodels
        act += [f"missions/{m}/{v['version']}" for m in _mcat.ORDER if (v := mmodels.active(m))]
        sets["models"] = {**MODEL_SETS["models"], "paths": act}
    for ds, meta in sets.items():
        root = _root(meta)
        files = _files(root, meta["paths"])
        if not files:
            print(f"  {ds}: not present in {root} - left out of the bundle")
            continue
        step(f"Packaging {ds} ({len(files)} files)")
        recs = []
        for i, f in enumerate(files, 1):
            rel = f.relative_to(root).as_posix()
            dst = out / "datasets" / ds / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            digest = sha256(f)
            if not (dst.exists() and dst.stat().st_size == f.stat().st_size and sha256(dst) == digest):
                shutil.copy2(f, dst)
            recs.append({"path": rel, "bytes": f.stat().st_size, "sha256": digest})
            if i % 200 == 0 or i == len(files):
                report(i, len(files), f"{ds}: {i}/{len(files)} files")
        manifest["datasets"].append({"id": ds, **{k: v for k, v in meta.items() if k != "paths"},
                                     "details": _describe(ds) if ds != "models" else {"versions": meta["paths"]},
                                     "file_count": len(recs),
                                     "bytes": sum(r["bytes"] for r in recs), "files": recs})
    manifest["total_bytes"] = sum(d["bytes"] for d in manifest["datasets"])
    write_json(out / "manifest.json", manifest)
    print(f"Bundle {manifest['bundle_version']}: {len(manifest['datasets'])} datasets, "
          f"{manifest['total_bytes'] / 1e9:.2f} GB in {out}")
    unconfirmed = [d["id"] for d in manifest["datasets"] if d.get("redistribution") != "yes"]
    if unconfirmed:
        print(f"Before publishing: confirm redistribution terms for {', '.join(unconfirmed)} (see manifest.json).")
    return manifest


# ------------------------------------------------------------------ verify / install / status
def load_manifest(bundle: Path | None = None) -> dict | None:
    b = bundle or BUNDLE_DIR
    m = read_json(b / "manifest.json", None)
    return m if m and m.get("schema") == SCHEMA else None


def verify(bundle: Path, quick: bool = False) -> dict:
    m = load_manifest(bundle)
    if not m:
        raise FileNotFoundError(f"No release bundle manifest at {bundle / 'manifest.json'}")
    problems, n, total = [], 0, sum(d["file_count"] for d in m["datasets"])
    for d in m["datasets"]:
        for f in d["files"]:
            p = bundle / "datasets" / d["id"] / f["path"]
            n += 1
            if not p.exists():
                problems.append(f"missing: {d['id']}/{f['path']}")
            elif p.stat().st_size != f["bytes"] or (not quick and sha256(p) != f["sha256"]):
                problems.append(f"checksum mismatch: {d['id']}/{f['path']}")
            if n % 500 == 0:
                report(n, total, f"verified {n}/{total} files")
    return {"bundle_version": m["bundle_version"], "files": total, "problems": problems}


def _kept_local(b: Path, d: dict) -> bool:
    """A mission store that the user re-imported locally (different fingerprint from the bundled copy)
    is newer than the bundle - never overwrite it with the bundled files."""
    man = next((f["path"] for f in d["files"] if f["path"].endswith("manifest.json") and "/missions/" in "/" + f["path"]), None)
    if not d["id"].startswith("mission-") or not man:
        return False
    local = _root(d) / man
    if not local.exists():
        return False
    try:
        return (read_json(local, {}) or {}).get("fingerprint") != (read_json(b / "datasets" / d["id"] / man, {}) or {}).get("fingerprint")
    except Exception:  # noqa: BLE001
        return False


def install(bundle: Path | None = None) -> dict:
    b = bundle or BUNDLE_DIR
    m = load_manifest(b)
    if not m:
        raise FileNotFoundError(f"No release bundle at {b}")
    total = sum(d["file_count"] for d in m["datasets"])
    copied, ok, bad, n, kept = 0, 0, [], 0, []
    for d in m["datasets"]:
        if _kept_local(b, d):
            kept.append(d["id"])
            n += len(d["files"])
            continue
        step(f"Installing {d['title']}")
        for f in d["files"]:
            n += 1
            src, dst = b / "datasets" / d["id"] / f["path"], _root(d) / f["path"]
            if dst.exists() and dst.stat().st_size == f["bytes"] and sha256(dst) == f["sha256"]:
                ok += 1
            elif not src.exists() or src.stat().st_size != f["bytes"]:
                bad.append(f"{d['id']}/{f['path']}")
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_name(dst.name + ".installing")
                shutil.copyfile(src, tmp)
                if sha256(tmp) != f["sha256"]:
                    tmp.unlink(missing_ok=True)
                    bad.append(f"{d['id']}/{f['path']}")
                    continue
                tmp.replace(dst)
                copied += 1
            if n % 200 == 0 or n == total:
                report(n, total, f"{n}/{total} files checked, {copied} copied")
    state = {"bundle_version": m["bundle_version"], "installed_at": now_iso(), "copied": copied, "already_ok": ok,
             "problems": bad[:50], "n_problems": len(bad), "complete": not bad, "kept_local": kept,
             "datasets": [d["id"] for d in m["datasets"]]}
    write_json(STATE, state)
    print(f"Release {m['bundle_version']}: {copied} files installed, {ok} already present, {len(bad)} problems")
    return state


def install_if_needed() -> dict | None:
    """Startup hook: install a mounted bundle the first time (or when its version changes)."""
    m = load_manifest()
    if not m:
        return None
    st = read_json(STATE, {}) or {}
    if st.get("bundle_version") == m["bundle_version"] and st.get("complete"):
        return st
    return install()


def status() -> dict:
    m = load_manifest()
    st = read_json(STATE, {}) or {}
    return {"bundle_dir": str(BUNDLE_DIR), "present": m is not None,
            "bundle_version": (m or {}).get("bundle_version"), "total_bytes": (m or {}).get("total_bytes"),
            "installed": st, "datasets": [{k: v for k, v in d.items() if k != "files"} for d in (m or {}).get("datasets", [])]}


def main() -> None:
    p = argparse.ArgumentParser(description="Space ML release bundle")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", default=str(BUNDLE_DIR))
    b.add_argument("--from-data", action="store_true", help="package what is already in data/ (no network)")
    b.add_argument("--include-models", action="store_true", help="also ship the active trained model of each module")
    b.add_argument("--kepler-stars", type=int, default=1000)
    b.add_argument("--kepler-quarters", type=int, default=8)
    b.add_argument("--ai4mars-train", default="3000", help="train pairs, or 'all' (~18k, several GB)")
    b.add_argument("--omni-years", type=int, nargs="+", default=[2003, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2024])
    v = sub.add_parser("verify")
    v.add_argument("--bundle", default=str(BUNDLE_DIR))
    v.add_argument("--quick", action="store_true", help="sizes only, no hashing")
    i = sub.add_parser("install")
    i.add_argument("--bundle", default=str(BUNDLE_DIR))
    a = p.parse_args()
    if a.cmd == "build":
        build(Path(a.out), a.from_data, a)
    elif a.cmd == "verify":
        r = verify(Path(a.bundle), a.quick)
        print(json.dumps({k: v for k, v in r.items() if k != "problems"}), f"{len(r['problems'])} problem(s)")
        for x in r["problems"][:30]:
            print("  ", x)
        if r["problems"]:
            fail(f"{len(r['problems'])} file(s) missing or corrupt")
    else:
        try:
            st = install(Path(a.bundle))
        except FileNotFoundError as e:
            fail(str(e))
        if not st["complete"]:
            fail(f"{st['n_problems']} file(s) in the bundle are missing or corrupt")


if __name__ == "__main__":
    main()
