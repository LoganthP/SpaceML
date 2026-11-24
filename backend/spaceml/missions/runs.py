"""Runs of the `missions` module (saved runs, shown in Runs and Review Studio).

input_ref = {"mission": id, "action": "evaluate" | "score", "item": id?, "column": name?}
* evaluate  (trained missions)  held-out predictions of the pinned/active model version; every held-out
            prediction becomes a Review Studio item (errors first) with its signal, published label and prediction
* score     one item: rolling median/MAD baseline + the trained detector when one exists; unusual intervals
            become review candidates (guiding flags / annotations shown as context)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..common.utils import read_json
from . import catalog, store

ORDER = {"FN": 0, "FP": 1, "TP": 2, "TN": 3}


def handle(input_ref: dict, version_dir: Path | None, run_dir: Path) -> tuple[dict, dict]:
    mission = input_ref.get("mission")
    meta = catalog.get(mission)
    if not store.installed(mission):
        raise FileNotFoundError(f"{meta['name']} data is not installed - prepare it once: {meta['prepare']}")
    action = input_ref.get("action", "score")
    if action == "evaluate":
        return _evaluate(mission, meta, version_dir)
    return _score(mission, meta, input_ref, version_dir)


def _evaluate(mission, meta, vd):
    if vd is None:
        raise FileNotFoundError(f"No trained model for {meta['name']} yet - press Quick train first")
    m = read_json(Path(vd) / "metrics.json")
    preds = read_json(Path(vd) / "predictions.json")
    items = []
    if meta["family"] == "segments":
        for p in sorted(preds, key=lambda p: (ORDER[p["outcome"]], -p["prob"])):
            it = store.item(mission, str(p["segment"]))
            items.append({"item": f"seg:{p['segment']}", "kind": "mission_segment",
                          "title": f"Segment {p['segment']} · {p['channel']} · {p['outcome']}",
                          "detail": f"published label {'anomaly' if p['label'] else 'nominal'} ({it.get('published_label')}) · "
                                    f"predicted {'anomaly' if p['pred'] else 'nominal'} (p={p['prob']:.2f})",
                          "hint": "confirmed" if p["label"] == 1 else "dismissed",
                          "mission": mission, "target": str(p["segment"]), "label": p["label"], "pred": p["pred"],
                          "prob": p["prob"], "outcome": p["outcome"]})
    else:
        for p in preds:
            for iv in p.get("model_intervals", [])[:50]:
                items.append(_interval_item(mission, p["item"], iv, "model", p.get("context")))
    h = m.get("headline", {})
    summary = {"headline": f"{meta['name']}: {h.get('metric')} {_fmt(h.get('model'))} vs baseline {_fmt(h.get('baseline'))}",
               "review_items": len(items)}
    return {"action": "evaluate", "mission": mission, "metrics": m, "predictions": preds[:2000],
            "review_items": items}, summary


def _score(mission, meta, inp, vd):
    item_id = str(inp.get("item") or "")
    if not item_id:
        raise ValueError("choose an item (segment / exposure / visit / channel) to score")
    it = store.item(mission, item_id)
    d = store.load(mission, item_id)
    model = None
    if vd is not None and (Path(vd) / "model.joblib").exists():
        import joblib
        model = joblib.load(Path(vd) / "model.joblib")
    fam = meta["family"]
    res: dict = {"action": "score", "mission": mission, "item": it}
    items = []
    if fam == "segments":
        from . import opssat
        f = opssat.features()
        row = f[f.segment == int(item_id)]
        if model is None:
            raise FileNotFoundError("No trained OPS-SAT model yet - press Quick train first")
        prob = float(model["model"].predict_proba(row[model["features"]].to_numpy(np.float64))[0, 1])
        pred = int(prob >= model["threshold"])
        lab = int(row.anomaly.iloc[0])
        res.update(prob=prob, pred=pred, label=lab, split=it.get("split"))
        items.append({"item": f"seg:{item_id}", "kind": "mission_segment",
                      "title": f"Segment {item_id} · {it['channel']} ({it['split']} split)",
                      "detail": f"published label {'anomaly' if lab else 'nominal'} · predicted {'anomaly' if pred else 'nominal'} (p={prob:.2f})",
                      "hint": "confirmed" if lab else "dismissed", "mission": mission, "target": item_id,
                      "label": lab, "pred": pred, "prob": prob})
        summary = {"headline": f"Segment {item_id}: predicted {'anomaly' if pred else 'nominal'} (p={prob:.2f}), "
                               f"published {'anomaly' if lab else 'nominal'}"}
    else:
        from . import pointing
        r = pointing.score_item(mission, it, d, model)
        ctx = {"guiding": it.get("guiding"), "warnings": it.get("warnings"), "guiding_modes": it.get("guiding_modes")}
        evs = [e for e in store.events(mission) if e.get("channel") == item_id]
        res.update(model_intervals=r["model_intervals"], baseline_intervals=r["baseline_intervals"], events=evs,
                   context=ctx, detector="model + rolling MAD" if model else "rolling MAD only (no trained model)")
        src = r["model_intervals"] if r["model_intervals"] else r["baseline_intervals"]
        how = "model" if r["model_intervals"] else "baseline"
        for iv in src[:60]:
            items.append(_interval_item(mission, item_id, iv, how, ctx, evs))
        summary = {"headline": f"{it.get('label', item_id)}: {len(r['model_intervals'])} model / "
                               f"{len(r['baseline_intervals'])} baseline intervals"}
    res["review_items"] = items
    summary["review_items"] = len(items)
    return res, summary


def _interval_item(mission, item_id, iv, how, ctx=None, evs=None):
    over = [e for e in (evs or []) if e["start"] <= iv["end"] and e["end"] >= iv["start"]]
    return {"item": f"iv:{item_id}:{iv['start']:.3f}-{iv['end']:.3f}", "kind": "mission_interval",
            "title": f"{item_id} · unusual interval ({how})",
            "detail": f"{iv['n']} samples" + (f" · score {iv['score']}" if iv.get("score") is not None else "") +
                      (f" · overlaps annotated {over[0]['category']}" if over else "") +
                      (f" · guiding: {', '.join(ctx.get('warnings') or []) or 'no warnings'}" if ctx and ctx.get('warnings') is not None else ""),
            "hint": "confirmed" if over else None, "mission": mission, "target": item_id,
            "start": iv["start"], "end": iv["end"], "detector": how}


def _fmt(x):
    return "n/a" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))
