"""Module-specific run handlers executed by the background worker."""
from __future__ import annotations

from pathlib import Path

from spaceml.common.utils import read_json
from spaceml.config import UPLOADS_DIR

from .worker import register


@register("telemetry")
def telemetry_handler(inp: dict, version_dir: Path, run_dir: Path):
    from spaceml.telemetry import infer

    res = infer.run(inp["chan_id"], version_dir)
    n = len(res["flagged"])
    return res, {
        "headline": f"{n} interval{'s' if n != 1 else ''} flagged for review",
        "flags": n, **res["score"], "spacecraft": res["spacecraft"],
    }


@register("exoplanet")
def exoplanet_handler(inp: dict, version_dir: Path | None, run_dir: Path):
    from spaceml.exoplanet import infer, uploads

    if inp.get("dataset_id"):
        res = uploads.analyze(inp["dataset_id"], inp["target_id"], version_dir, inp.get("period"),
                              inp.get("t0"), inp.get("duration_hours"))
    else:
        res = infer.run(int(inp["kepid"]), int(inp.get("tce_plnt_num", 1)), version_dir)
    sc = res["candidate_score"]
    head = (f"Candidate score {sc:.2f}" if sc is not None else "Views + 3D ready (no CNN trained yet)")
    if res["source"] == "upload":
        head += f" · P = {res['tce']['period_days']:.3f} d"
    return res, {
        "headline": head, "candidate_score": sc, "baseline_score": res["baseline_score"],
        "catalog_label": res["catalog_label"], "split": res["split"], "source": res["source"],
    }


@register("mars")
def mars_handler(inp: dict, version_dir: Path, run_dir: Path):
    from spaceml.mars import data, infer

    location = None
    if inp.get("sample_id"):
        p = data.sample_paths(inp["sample_id"])
        res = infer.run(p["image"], version_dir, run_dir, p.get("label"), p.get("ignore"))
        location = _locate(data.sample_name(inp["sample_id"]))
    elif inp.get("raw_image_id"):
        from spaceml.mars import traverse
        img, meta = traverse.raw_image(inp["raw_image_id"])
        res = infer.run(img, version_dir, run_dir)
        location = traverse.locate(inp["raw_image_id"], meta)
        res["source_image"] = {k: meta.get(k) for k in ("imageid", "sol", "site", "drive", "instrument", "date_taken",
                                                         "url", "lmst", "mast_az", "mast_el")}
    else:
        img = UPLOADS_DIR / f"{inp['upload_id']}.png"
        if not img.exists():
            raise FileNotFoundError("Uploaded image not found - upload it again")
        res = infer.run(img, version_dir, run_dir)
    if location is None and inp.get("upload_id"):
        meta = read_json(UPLOADS_DIR / f"{inp['upload_id']}.json", {}) or {}
        location = _locate(meta.get("filename", ""))
    res["location"] = location
    top = max(res["class_shares"], key=res["class_shares"].get)
    return res, {
        "headline": f"Mostly {top} - sand {res['class_shares']['sand']:.0%}",
        "class_shares": res["class_shares"],
        "miou": res["evaluation"]["model"]["miou"] if res["evaluation"] else None,
        "sol": (location or {}).get("sol"), "site": (location or {}).get("site"), "drive": (location or {}).get("drive"),
    }


def _locate(name: str) -> dict | None:
    from spaceml.mars import traverse
    try:
        return traverse.locate(name)
    except Exception:  # noqa: BLE001 - location is optional context
        return None


@register("spaceweather")
def spaceweather_handler(inp: dict, version_dir: Path, run_dir: Path):
    from spaceml.spaceweather import data, replay

    ev = next((e for e in data.EVENTS if e["id"] == inp.get("event_id")), None)
    start, end = (ev["start"], ev["end"]) if ev else (inp["start"], inp["end"])
    res = replay.run(version_dir, start, end, inp.get("horizons"))
    res["event"] = ev
    mid = res["horizons"][len(res["horizons"]) // 2]
    ph = res["per_horizon"][str(mid)]
    head = (f"+{mid} h: RMSE {ph['rmse']:.0f} nT vs persistence {ph['rmse_persistence']:.0f} nT"
            if ph["rmse"] is not None else "No verifiable forecasts in this window")
    return res, {"headline": head, "skill": ph["skill"], "horizon": mid,
                 "better_hours": ph["better_hours"], "worse_hours": ph["worse_hours"],
                 "observed_min": ph["observed_min"], "event_id": inp.get("event_id")}


@register("workbench")
def workbench_handler(inp: dict, version_dir: Path | None, run_dir: Path):
    from spaceml.workbench import analyses

    from spaceml.workbench import insights

    res = analyses.runner(inp["analysis"])(inp["dataset_id"], inp["params"], run_dir, version_dir)
    res = {"analysis": inp["analysis"], "viewer": analyses.ANALYSES[inp["analysis"]]["output"], **res}
    if inp["analysis"] == "image.insights":
        sc = res["scene_counts"]
        top = max(sc, key=sc.get)
        head = f"{res['n_analysed']} images classified · mostly {res['scene_labels'][top].lower()}"
        if res["model_used"]:
            head += " · terrain model applied"
        return res, {"headline": head, "groups": len(res["groups"]), "scenes": sc}
    if inp["analysis"] == "samples.classify":
        m, b = res["model"], res["baseline"]
        head = (f"TSS {m['tss']:.2f} (model) vs {b['tss']:.2f} (single threshold) · "
                f"{m['tp']}/{m['tp'] + m['fn']} hits, {m['fp']} false alarms")
        return res, {"headline": head, "metrics": {"model_tss": m["tss"], "baseline_tss": b["tss"], "auc": m.get("auc")}}
    fn = {"timeline": insights.timeline, "change": insights.change, "segmentation": insights.segmentation}.get(res["viewer"])
    if fn:
        res["insights"] = insights.safe(fn, res)
    if inp["analysis"] == "timeseries.anomaly":
        m = res["metrics"]
        nz, nr = len(res["zscore"]["intervals"]), len(res["robust"]["intervals"])
        head = f"{nr} robust / {nz} z-score intervals flagged"
        if m:
            head += f" · robust F1 {m['robust']['f1']:.0%}"
        return res, {"headline": head, "flags": nr, "metrics": m and {"robust_f1": m["robust"]["f1"],
                                                                     "zscore_f1": m["zscore"]["f1"]}}
    if inp["analysis"] == "image.change":
        area = res["changed_area_km2"]
        head = f"{res['changed_fraction']:.1%} changed" + (f" · {area:.3g} km²" if area is not None else "")
        return res, {"headline": head, "regions": len(res["regions"])}
    top = max(res["class_shares"], key=res["class_shares"].get)
    return res, {"headline": f"Mostly {top} - sand {res['class_shares']['sand']:.0%}"}


@register("missions")
def missions_handler(inp: dict, version_dir: Path | None, run_dir: Path):
    from spaceml.missions import runs as mruns
    return mruns.handle(inp, version_dir, run_dir)
