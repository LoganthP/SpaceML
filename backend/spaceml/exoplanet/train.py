"""Train the two-view CNN and the period/duration/depth baseline, then evaluate both
on held-out stars.

    python -m spaceml.exoplanet.train [--epochs 60] [--version v1]

Uses every labeled TCE whose views are cached (see spaceml.exoplanet.data).
Split is by star (kepid): ~80% train / 10% validation / 10% test.
"""
from __future__ import annotations

import argparse
import time

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

from ..common.progress import fail, report, step
from ..common.utils import new_version, now_iso, set_seed, write_json
from ..common.parallel import train_cpus
from ..config import MODELS_DIR, SEED
from . import data
from .model import TwoBranchCNN

BASELINE_FEATURES = ["tce_period", "tce_duration", "tce_depth"]


def baseline_matrix(df) -> np.ndarray:
    X = df[BASELINE_FEATURES].to_numpy(dtype=float)
    return np.log10(np.clip(X, 1e-6, None))


MIN_PER_CLASS = data.MIN_PER_CLASS


def load_dataset(max_stars: int = 0):
    try:
        df = data.cached_training_rows()
    except FileNotFoundError as e:
        fail(str(e))
    if max_stars:
        stars = np.random.default_rng(SEED).permutation(np.sort(df.kepid.unique())).tolist()
        pick = set()
        for label in (0, 1):
            counts = df.loc[df.planet == label, "kepid"].value_counts().to_dict()
            covered = sum(counts.get(star, 0) for star in pick)
            contributing = sum(star in counts for star in pick)
            for star in stars:
                if covered >= MIN_PER_CLASS and contributing >= 5:
                    break
                if star in counts and star not in pick and len(pick) < max_stars:
                    pick.add(star)
                    covered += counts[star]
                    contributing += 1
        for star in stars:
            if len(pick) >= max_stars:
                break
            pick.add(star)
        df = df[df.kepid.isin(pick)].reset_index(drop=True)
        print(f"Quick run: {len(pick)} of {len(stars)} stars (seeded subset)")
    n_pc, n_other = int(df.planet.sum()), int((df.planet == 0).sum())
    if n_pc < MIN_PER_CLASS or n_other < MIN_PER_CLASS:
        fail(f"Not enough cached light curves to train: {n_pc} planet candidates and {n_other} false "
             f"positives/non-transiting signals (need at least {MIN_PER_CLASS} of each, ideally 100+). "
             "Add the Kepler data from the release bundle, or download more light curves on Data & Training.")
    G = np.zeros((len(df), data.GLOBAL_BINS), np.float32)
    L = np.zeros((len(df), data.LOCAL_BINS), np.float32)
    keep = np.ones(len(df), bool)
    step(f"Loading {len(df)} folded views")
    for i, r in enumerate(df.itertuples()):
        try:
            v = data.get_views(int(r.kepid), int(r.tce_plnt_num), allow_download=False)
            G[i], L[i] = v["global"], v["local"]
        except Exception as e:  # noqa: BLE001 - a corrupt cache file should not stop training
            print(f"  skipping KIC {r.kepid} TCE {r.tce_plnt_num}: {e}")
            keep[i] = False
        report(i + 1, len(df), "loading views")
    df = df[keep].reset_index(drop=True)
    return df, np.nan_to_num(G[keep]), np.nan_to_num(L[keep])


def split_by_star(df):
    """Stratified by label, grouped by star: no star appears in two splits and every split
    contains both classes (plain random splits can leave a tiny test set with one class)."""
    idx = np.arange(len(df))
    y, groups = df.planet.to_numpy(), df.kepid.to_numpy()
    n_test = 10 if df.kepid.nunique() >= 100 else 5
    outer = StratifiedGroupKFold(n_splits=n_test, shuffle=True, random_state=SEED)
    trval, test = next(outer.split(idx, y, groups))
    inner = StratifiedGroupKFold(n_splits=max(n_test - 1, 4), shuffle=True, random_state=SEED)
    tr, va = next(inner.split(trval, y[trval], groups[trval]))
    tr, va = trval[tr], trval[va]
    for name, part in (("test", test), ("validation", va), ("training", tr)):
        if len(set(y[part])) < 2:
            fail(f"The {name} split has only one class. On Data & Training, press 'Download light curves' "
                 "on the Kepler card for more stars, then train again.")
    return tr, va, test


@torch.no_grad()
def cnn_logits(model, G, L, batch: int = 256) -> np.ndarray:
    model.eval()
    out = [model(torch.from_numpy(G[i:i + batch]), torch.from_numpy(L[i:i + batch])) for i in range(0, len(G), batch)]
    return torch.cat(out).numpy()


def apply_calibration(logits: np.ndarray, calib: dict | None) -> np.ndarray:
    a, b = (calib or {}).get("a", 1.0), (calib or {}).get("b", 0.0)
    return 1.0 / (1.0 + np.exp(-(a * np.asarray(logits, float) + b)))


def fit_calibration(logits: np.ndarray, y: np.ndarray) -> dict:
    """Platt scaling on held-out validation stars: turns raw logits (biased by the class weighting)
    into probabilities that mean what they say. Ranking (AUC) is unchanged."""
    from sklearn.linear_model import LogisticRegression
    if len(set(y)) < 2:
        return {"a": 1.0, "b": 0.0, "method": "none"}
    lr = LogisticRegression(C=10.0).fit(np.asarray(logits, float)[:, None], y)
    a = float(lr.coef_[0, 0])
    if a <= 0:  # validation set too small / degenerate: keep the raw scores
        return {"a": 1.0, "b": 0.0, "method": "none"}
    return {"a": a, "b": float(lr.intercept_[0]), "method": "platt"}


def cnn_scores(model, G, L, batch: int = 256, calib: dict | None = None) -> np.ndarray:
    return apply_calibration(cnn_logits(model, G, L, batch), calib)


def train_cnn(G, L, y, tr, va, epochs: int, verbose=True):
    set_seed(SEED)
    model = TwoBranchCNN()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pos = y[tr].mean()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / max(pos, 1e-6)))
    Gt, Lt, yt = torch.from_numpy(G[tr]), torch.from_numpy(L[tr]), torch.from_numpy(y[tr].astype(np.float32))
    best_auc, best_state, bad, patience = -1.0, None, 0, 15
    gen = torch.Generator().manual_seed(SEED)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(tr), generator=gen)
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            g, l = Gt[b], Lt[b]
            flip = torch.rand(len(b), generator=gen) < 0.5  # time-reversal augmentation
            g = torch.where(flip[:, None], g.flip(1), g)
            l = torch.where(flip[:, None], l.flip(1), l)
            opt.zero_grad()
            loss = loss_fn(model(g, l), yt[b])
            loss.backward()
            opt.step()
        auc = roc_auc_score(y[va], cnn_scores(model, G[va], L[va])) if len(set(y[va])) > 1 else 0.0
        if verbose:
            print(f"  epoch {ep + 1:3d}  loss={loss.item():.4f}  val AUC={auc:.4f}", flush=True)
        report(ep + 1, epochs, f"epoch {ep + 1}: validation AUC {auc:.3f}", every=0)
        if auc > best_auc + 1e-4:
            best_auc, bad = auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_auc


def preflight(G, L, y, tr) -> None:
    """Fail in seconds (before any model folder is written) if the packaged arrays can't train."""
    step("Preflight: checking the packaged Kepler views and running one training batch")
    if G.ndim != 2 or L.ndim != 2 or len(G) != len(L) or len(G) != len(y):
        fail(f"Kepler views have inconsistent shapes: global {G.shape}, local {L.shape}, labels {y.shape}")
    bad = int((~np.isfinite(G)).any(1).sum() + (~np.isfinite(L)).any(1).sum())
    if bad:
        fail(f"{bad} Kepler views contain NaN/inf values; rebuild them with task:exoplanet.lightcurves")
    if len(set(y[tr].tolist())) < 2:
        fail("The training split has only one class (planets or false positives); add more stars")
    set_seed(SEED)
    model = TwoBranchCNN()
    b = tr[:min(32, len(tr))]
    loss = nn.BCEWithLogitsLoss()(model(torch.from_numpy(G[b]), torch.from_numpy(L[b])),
                                  torch.from_numpy(y[b].astype(np.float32)))
    loss.backward()
    if not torch.isfinite(loss):
        fail(f"Preflight batch produced a non-finite loss ({loss.item()})")
    print(f"Preflight ok: {len(y)} views, global {G.shape[1]} / local {L.shape[1]} bins, "
          f"batch loss {loss.item():.4f}", flush=True)


def _curve(x, y, n=60):
    if len(x) <= n:
        return [[float(a), float(b)] for a, b in zip(x, y)]
    idx = np.linspace(0, len(x) - 1, n).astype(int)
    return [[float(x[i]), float(y[i])] for i in idx]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--version", default=None)
    p.add_argument("--max-stars", type=int, default=0, help="quick run on a seeded subset of stars (0 = all)")
    args = p.parse_args()
    t0 = time.time()
    torch.set_num_threads(train_cpus())

    df, G, L = load_dataset(args.max_stars)
    y = df.planet.to_numpy()
    tr, va, te = split_by_star(df)
    print(f"{len(df)} TCEs from {df.kepid.nunique()} stars | train {len(tr)}  val {len(va)}  test {len(te)} "
          f"| planet share {y.mean():.2%}")

    preflight(G, L, y, tr)
    quick = bool(args.max_stars)
    version = args.version or (new_version() + ("-quick" if quick else ""))
    out = MODELS_DIR / "exoplanet" / version
    out.mkdir(parents=True, exist_ok=True)

    # Baseline: knows how deep and long the dip is, but not its shape.
    base = HistGradientBoostingClassifier(random_state=SEED).fit(baseline_matrix(df.iloc[tr]), y[tr])
    joblib.dump(base, out / "baseline_gb.joblib")

    step("Training two-branch CNN")
    model, val_auc = train_cnn(G, L, y, tr, va, args.epochs)
    calib = fit_calibration(cnn_logits(model, G[va], L[va]), y[va])
    print(f"Calibration ({calib['method']}): a={calib['a']:.3f} b={calib['b']:.3f}")
    torch.save({"state_dict": model.state_dict(), "val_auc": val_auc, "calibration": calib}, out / "cnn.pt")

    s_cnn = cnn_scores(model, G[te], L[te], calib=calib)
    s_base = base.predict_proba(baseline_matrix(df.iloc[te]))[:, 1]
    yt = y[te]

    def metrics(s):
        pred = (s >= 0.5).astype(int)
        return {"auc": roc_auc_score(yt, s), "precision": precision_score(yt, pred, zero_division=0),
                "recall": recall_score(yt, pred, zero_division=0)}

    m, b = metrics(s_cnn), metrics(s_base)
    test = df.iloc[te].assign(cnn=s_cnn, baseline=s_base)
    fps = test[(test.planet == 0) & (test.cnn >= 0.5)].sort_values("cnn", ascending=False)
    fns = test[(test.planet == 1) & (test.cnn < 0.5)].sort_values("cnn")
    pr_m, pr_b = precision_recall_curve(yt, s_cnn), precision_recall_curve(yt, s_base)
    roc_m, roc_b = roc_curve(yt, s_cnn), roc_curve(yt, s_base)

    write_json(out / "split.json", {
        "train": sorted(df.kepid.iloc[tr].unique().tolist()), "val": sorted(df.kepid.iloc[va].unique().tolist()),
        "test": sorted(df.kepid.iloc[te].unique().tolist())})
    write_json(out / "test_predictions.json", test[["kepid", "tce_plnt_num", "av_training_set", "planet",
                                                     "cnn", "baseline"]].to_dict("records"))
    write_json(out / "manifest.json", {
        "module": "exoplanet", "version": version, "created_at": now_iso(),
        "model_name": "Two-view 1D CNN (global 2001 + local 201 bins)" + (" - quick run (pipeline check)" if quick else ""),
        "run_kind": "quick" if quick else "full",
        "n_tces": len(df), "n_stars": int(df.kepid.nunique()), "val_auc": val_auc,
        "data": "Kepler Q1-Q17 DR24 TCE labels (PC vs AFP/NTP) + MAST long-cadence light curves",
        "train_seconds": round(time.time() - t0, 1)})
    write_json(out / "metrics.json", {
        "module": "exoplanet", "version": version, "created_at": now_iso(),
        "model_name": "Two-view 1D CNN", "baseline_name": "Gradient boosting on period, duration, depth",
        "eval_split": f"{len(te)} TCEs on {df.kepid.iloc[te].nunique()} held-out stars (split by kepid)",
        "headline": {"metric": "ROC-AUC", "model": m["auc"], "baseline": b["auc"]},
        "rows": [{"group": "Held-out stars", "metrics": [
            {"name": "ROC-AUC", "model": m["auc"], "baseline": b["auc"]},
            {"name": "Planet precision @0.5", "model": m["precision"], "baseline": b["precision"]},
            {"name": "Planet recall @0.5", "model": m["recall"], "baseline": b["recall"]},
            {"name": "Test TCEs (planets)", "model": f"{len(te)} ({int(yt.sum())})", "baseline": f"{len(te)} ({int(yt.sum())})"},
        ]}],
        "curves": {"pr": {"model": _curve(pr_m[1], pr_m[0]), "baseline": _curve(pr_b[1], pr_b[0])},
                   "roc": {"model": _curve(roc_m[0], roc_m[1]), "baseline": _curve(roc_b[0], roc_b[1])}},
        "struggles": [f"False positive: KIC {r.kepid} TCE {r.tce_plnt_num} ({r.av_training_set}) scored {r.cnn:.2f}"
                      for r in fps.head(6).itertuples()]
                     + [f"Missed planet candidate: KIC {r.kepid} TCE {r.tce_plnt_num} scored {r.cnn:.2f}"
                        for r in fns.head(4).itertuples()],
    })
    print(f"Test ROC-AUC  CNN={m['auc']:.4f}  baseline={b['auc']:.4f}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
