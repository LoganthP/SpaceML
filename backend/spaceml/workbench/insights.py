"""Plain-language findings for workbench results, so every run explains what it found."""
from __future__ import annotations

import numpy as np


def _i(kind: str, title: str, text: str) -> dict:
    return {"kind": kind, "title": title, "text": text}


def timeline(r: dict) -> list[dict]:
    out = []
    n = r["n_rows"]
    rob, z = r["robust"]["intervals"], r["zscore"]["intervals"]
    cov = sum(iv["length"] for iv in rob) / max(n, 1)
    out.append(_i("classification", "Verdict",
                  ("No unusual intervals: the series looks nominal under both detectors."
                   if not rob and not z else
                   f"{len(rob)} interval(s) flagged by the robust detector ({cov:.1%} of the {n:,} rows) and "
                   f"{len(z)} by the z-score baseline.")))
    if rob:
        longest = max(rob, key=lambda iv: iv["length"])
        out.append(_i("warn", "Longest anomaly", f"{longest['length']:,} steps from {str(longest['x0'])[:19]} to "
                                                 f"{str(longest['x1'])[:19]}."))
        both = sum(any(not (a["end"] < b["start"] or a["start"] > b["end"]) for b in z) for a in rob)
        out.append(_i("info", "Detector agreement",
                      f"{both} of {len(rob)} robust flags are also found by the z-score baseline"
                      + (" - strong, unambiguous events." if both == len(rob) else
                         "; the others are local changes the global z-score misses (drifts, level shifts).")))
    ch = sorted(r["channels"], key=lambda c: -(c["flags_robust"] + c["flags_zscore"]))
    if ch and (ch[0]["flags_robust"] + ch[0]["flags_zscore"]):
        out.append(_i("classification", "Most anomalous sensor",
                      f"{ch[0]['name']}: {ch[0]['flags_robust']} robust / {ch[0]['flags_zscore']} z-score flags"
                      + (f"; quietest: {ch[-1]['name']}." if len(ch) > 1 else ".")))
    miss = [c for c in r["channels"] if c["missing_fraction"] > 0.05]
    if miss:
        out.append(_i("warn", "Data gaps", ", ".join(f"{c['name']} {c['missing_fraction']:.0%} missing" for c in miss)
                      + " - gaps were filled before detection."))
    m = r.get("metrics")
    if m:
        better = "robust detector" if m["robust"]["f1"] >= m["zscore"]["f1"] else "z-score baseline"
        out.append(_i("terrain", "Against your labels",
                      f"{m['n_labeled']} labeled event(s). Robust F1 {m['robust']['f1']:.0%} (precision "
                      f"{m['robust']['precision']:.0%}, recall {m['robust']['recall']:.0%}) vs z-score F1 "
                      f"{m['zscore']['f1']:.0%}. The {better} fits this data better."))
    else:
        out.append(_i("info", "Measure accuracy", "Add a 0/1 label column or labeled intervals to get precision, "
                                                  "recall and F1 for both detectors."))
    if rob:
        out.append(_i("info", "Next step", "Confirm or dismiss each flag below (or in Review Studio); verdicts are "
                                           "exported as labels for retraining."))
    return out


def change(r: dict) -> list[dict]:
    out = []
    frac, area = r["changed_fraction"], r.get("changed_area_km2")
    level = "little" if frac < 0.02 else "moderate" if frac < 0.15 else "extensive"
    out.append(_i("classification", "Verdict", f"{level.capitalize()} change: {frac:.1%} of the valid area"
                  + (f" ({area:.3g} km²)" if area is not None else "") + f" in {len(r['regions'])} region(s)."))
    if r["method"] == "water":
        c = r["counts"]
        net = c["gained"] - c["lost"]
        out.append(_i("terrain", "Surface water", f"Water {'expanded' if net > 0 else 'shrank'}: "
                      f"{c['gained']:,} px gained, {c['lost']:,} px lost ({c['water_before']:,} → {c['water_after']:,} px)."))
    if r["regions"]:
        g = r["regions"][0]
        tot = sum(x["pixels"] for x in r["regions"])
        out.append(_i("warn", "Largest region", f"{g['pixels'] / max(tot, 1):.0%} of all change is in one region "
                      f"(box {g['bbox']})" + (f", {g['area_km2']:.3g} km²." if g.get("area_km2") is not None else ".")))
    if r["valid_fraction"] < 0.8:
        out.append(_i("warn", "Partial overlap", f"Only {r['valid_fraction']:.0%} of pixels overlap in both images."))
    if not r.get("georeferenced"):
        out.append(_i("info", "Areas", "These images are not georeferenced, so areas are in pixels. Rover photos "
                                       "taken from different positions are not aligned - change detection is "
                                       "only meaningful for the same view at two times."))
    return out


def segmentation(r: dict) -> list[dict]:
    sh = r["class_shares"]
    top = max(sh, key=sh.get)
    out = [_i("classification", "Dominant terrain", f"{top} ({sh[top]:.0%}); "
              + ", ".join(f"{k} {v:.0%}" for k, v in sh.items() if k != top) + ".")]
    if sh.get("sand", 0) > 0.15:
        out.append(_i("warn", "Sand hazard", f"{sh['sand']:.0%} sand - soft sand is the main mobility risk."))
    ev = r.get("evaluation")
    if ev:
        out.append(_i("terrain", "Against the label mask", f"mIoU {ev['model']['miou']:.2f} vs all-soil baseline "
                                                           f"{ev['baseline']['miou']:.2f}."))
    return out


def safe(fn, r: dict) -> list[dict]:
    try:
        return fn(r)
    except Exception as e:  # noqa: BLE001 - findings are a convenience
        return [_i("info", "Findings unavailable", f"{type(e).__name__}: {e}")]


__all__ = ["timeline", "change", "segmentation", "safe", "np"]
