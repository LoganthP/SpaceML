"""Event-level evaluation on all labeled SMAP and MSL test files.

    python -m spaceml.telemetry.evaluate --version <version>   (default: latest)
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from ..common.progress import report
from ..common.utils import now_iso, read_json, write_json
from ..config import MODELS_DIR
from . import data
from .detection import ZSCORE_K, prf
from .infer import analyse_channel


def latest_version_dir():
    root = MODELS_DIR / "telemetry"
    dirs = sorted([d for d in root.glob("*") if (d / "manifest.json").exists()],
                  key=lambda d: read_json(d / "manifest.json")["created_at"])
    if not dirs:
        raise SystemExit("No telemetry model versions found. Train first.")
    return dirs[-1]


def evaluate(version_dir) -> dict:
    chans = data.load_channels()
    totals = defaultdict(lambda: {"model": [0, 0, 0], "baseline": [0, 0, 0]})
    by_class = defaultdict(lambda: {"model": [0, 0], "baseline": [0, 0]})  # found, total
    per_channel, missed = [], []
    for n_done, cid in enumerate(sorted(chans), 1):
        report(n_done, len(chans), f"evaluating {cid}")
        if not (version_dir / f"{cid}.pt").exists():
            print(f"  skipping {cid}: no checkpoint")
            continue
        a = analyse_channel(cid, version_dir)
        ch, ms, bs = a["channel"], a["model_score"], a["baseline_score"]
        for grp in (ch.spacecraft, "All"):
            for key, s in (("model", ms), ("baseline", bs)):
                t = totals[grp][key]
                t[0] += s["tp"]; t[1] += s["fp"]; t[2] += s["fn"]
        for cls, fm, fb in zip(ch.classes, ms["found"], bs["found"]):
            by_class[cls]["model"][0] += fm; by_class[cls]["model"][1] += 1
            by_class[cls]["baseline"][0] += fb; by_class[cls]["baseline"][1] += 1
            if not fm:
                missed.append({"chan_id": cid, "spacecraft": ch.spacecraft, "class": cls})
        per_channel.append({"chan_id": cid, "spacecraft": ch.spacecraft,
                            "model": {k: ms[k] for k in ("tp", "fp", "fn")},
                            "baseline": {k: bs[k] for k in ("tp", "fp", "fn")},
                            "threshold_z": a["det"]["z"]})
        print(f"  {cid:6s} {ch.spacecraft:4s} model tp/fp/fn={ms['tp']}/{ms['fp']}/{ms['fn']}  "
              f"z-score={bs['tp']}/{bs['fp']}/{bs['fn']}", flush=True)

    rows = []
    for grp in ("SMAP", "MSL", "All"):
        if grp not in totals:
            continue
        m, b = prf(*totals[grp]["model"]), prf(*totals[grp]["baseline"])
        rows.append({"group": grp, "metrics": [
            {"name": "Precision", "model": m["precision"], "baseline": b["precision"]},
            {"name": "Recall", "model": m["recall"], "baseline": b["recall"]},
            {"name": "F1", "model": m["f1"], "baseline": b["f1"]},
            {"name": "Found / labeled", "model": f"{m['tp']}/{m['tp'] + m['fn']}",
             "baseline": f"{b['tp']}/{b['tp'] + b['fn']}"},
            {"name": "False alarms", "model": m["fp"], "baseline": b["fp"], "lower_is_better": True},
        ]})
    for cls in sorted(by_class):
        fm, n = by_class[cls]["model"]
        fb, _ = by_class[cls]["baseline"]
        rows.append({"group": f"{cls.capitalize()} anomalies", "metrics": [
            {"name": "Recall", "model": fm / n, "baseline": fb / n},
            {"name": "Found / labeled", "model": f"{fm}/{n}", "baseline": f"{fb}/{n}"}]})

    manifest = read_json(version_dir / "manifest.json", {})
    allm = prf(*totals["All"]["model"])
    allb = prf(*totals["All"]["baseline"])
    result = {
        "module": "telemetry",
        "version": version_dir.name,
        "created_at": now_iso(),
        "model_name": manifest.get("model_name", "LSTM + dynamic threshold"),
        "baseline_name": f"z-score threshold (|x - mean| > {ZSCORE_K:g} std, train statistics)",
        "eval_split": f"Held-out test files of {len(per_channel)} of {len(chans)} channels "
                      f"({sum(len(chans[c['chan_id']].sequences) for c in per_channel)} labeled anomalies); "
                      "threshold chosen without labels",
        "headline": {"metric": "Event F1 (all)", "model": allm["f1"], "baseline": allb["f1"]},
        "rows": rows,
        "struggles": [f"Missed {m['class']} anomaly on {m['chan_id']} ({m['spacecraft']})" for m in missed],
        "per_channel": per_channel,
    }
    write_json(version_dir / "metrics.json", result)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--version", default=None)
    args = p.parse_args()
    vdir = MODELS_DIR / "telemetry" / args.version if args.version else latest_version_dir()
    res = evaluate(vdir)
    for r in res["rows"]:
        print(r["group"], {m["name"]: (m["model"], m["baseline"]) for m in r["metrics"]})
    print(f"Wrote {vdir / 'metrics.json'}")


if __name__ == "__main__":
    main()
