"""Transit search for light curves without a known ephemeris (uploaded data).

Box Least Squares (astropy) on a detrended, binned light curve. Returns the best
period/epoch/duration/depth, the strongest alternative peaks and a periodogram.
"""
from __future__ import annotations

import numpy as np

from ..common.utils import clean_floats


def bin_lightcurve(t: np.ndarray, f: np.ndarray, width_days: float) -> tuple[np.ndarray, np.ndarray]:
    if len(t) < 2 or np.median(np.diff(t)) >= width_days * 0.9:
        return t, f
    idx = np.floor((t - t[0]) / width_days).astype(np.int64)
    uniq, inv, counts = np.unique(idx, return_inverse=True, return_counts=True)
    tb = np.bincount(inv, weights=t) / counts
    fb = np.bincount(inv, weights=f) / counts
    return tb, fb


def detrend(t: np.ndarray, f: np.ndarray, window_days: float = 1.0) -> np.ndarray:
    """Running median detrend (robust, keeps dips shorter than ~window/3)."""
    import pandas as pd

    # time relative to the first point: absolute JD/BJD (~2.46e6 d) would overflow datetime64
    s = pd.Series(f, index=pd.to_datetime(np.asarray(t, float) - float(np.nanmin(t)), unit="D", origin="unix"))
    trend = s.rolling(f"{int(window_days * 24 * 60)}min", center=True, min_periods=3).median().to_numpy()
    out = f / np.where(np.isfinite(trend) & (trend != 0), trend, np.nanmedian(f))
    return out


def bls_search(t: np.ndarray, f: np.ndarray, min_period: float = 0.5, max_period: float | None = None,
               max_grid: int = 25000) -> dict:
    from astropy.timeseries import BoxLeastSquares

    span = float(t.max() - t.min())
    if span < 1.0:
        raise ValueError(f"The light curve spans only {span:.2f} days; at least 1 day is needed to search for transits")
    cadence = float(np.median(np.diff(t)))
    tb, fb = bin_lightcurve(t, f, max(cadence, 10 / 1440))   # >= 10-minute bins for speed
    fd = detrend(tb, fb, window_days=1.5)
    ok = np.isfinite(fd)
    tb, fd = tb[ok], fd[ok]
    # clip upward outliers (flares, cosmic rays) only, never the dips
    med, mad = np.median(fd), np.median(np.abs(fd - np.median(fd))) * 1.4826
    keep = fd < med + 5 * max(mad, 1e-9)
    tb, fd = tb[keep], fd[keep]

    max_period = min(max_period or span / 2, span / 2, 120.0)
    min_period = max(min_period, 3 * cadence, 0.2)
    if max_period <= min_period:
        max_period = min(span / 1.5, min_period * 3)
    durations = np.array([d for d in (1, 1.5, 2, 3, 4, 6, 8, 10, 13) if d / 24 < 0.2 * min_period] or [1.0]) / 24
    # frequency grid with phase resolution ~ shortest duration / 3, capped for speed
    df_ = durations.min() / 3 / span ** 2
    n = int((1 / min_period - 1 / max_period) / df_)
    n = int(np.clip(n, 2000, max_grid))
    freqs = np.linspace(1 / max_period, 1 / min_period, n)
    periods = np.sort(1 / freqs)
    model = BoxLeastSquares(tb, fd)
    res = model.power(periods, durations, objective="snr")
    power = np.asarray(res.power)
    power = np.where(np.isfinite(power), power, 0)
    # detrend the periodogram's slow rise so long periods are not favoured
    import pandas as pd
    base = pd.Series(power).rolling(201, center=True, min_periods=1).median().to_numpy()
    sde_like = power - base
    order = np.argsort(-sde_like)
    picks: list[int] = []
    for i in order:
        p = periods[i]
        if all(abs(p / periods[j] - round(p / periods[j])) > 0.02 and abs(periods[j] / p - round(periods[j] / p)) > 0.02
               for j in picks):
            picks.append(int(i))
        if len(picks) == 4:
            break
    best = picks[0]
    noise = float(np.std(sde_like))

    def peak(i: int) -> dict:
        return {"period_days": float(periods[i]), "t0": float(res.transit_time[i]),
                "duration_hours": float(res.duration[i] * 24), "depth_ppm": float(res.depth[i] * 1e6),
                "snr": float(res.depth_snr[i]), "power": float(sde_like[i] / noise) if noise > 0 else 0.0}

    step = max(1, len(periods) // 2500)
    return {"best": peak(best), "alternatives": [peak(i) for i in picks[1:]],
            "periodogram": {"period": clean_floats(periods[::step], 5),
                            "power": clean_floats(sde_like[::step] / (noise or 1), 3)},
            "n_points_searched": int(len(tb)), "span_days": span, "cadence_minutes": cadence * 1440,
            "period_range": [float(min_period), float(max_period)]}
