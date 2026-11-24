"""Physical geometry for the 3D system view: star size/colour, orbit size, planet size,
equilibrium temperature and the habitable zone. Uses catalog values where available and
standard approximations otherwise (each value says where it came from)."""
from __future__ import annotations

import math

R_SUN_AU = 1 / 215.032
R_EARTH_PER_R_SUN = 109.08


def _ok(v) -> bool:
    try:
        return v is not None and math.isfinite(float(v)) and float(v) > 0
    except (TypeError, ValueError):
        return False


def star(teff=None, radius=None, logg=None, mass=None) -> dict:
    src = {}
    teff = float(teff) if _ok(teff) else (src.setdefault("teff", "assumed Sun-like") and 5772.0)
    radius = float(radius) if _ok(radius) else (src.setdefault("radius", "assumed Sun-like") and 1.0)
    if _ok(mass):
        mass = float(mass)
    elif _ok(logg):
        mass = min(max(10 ** (float(logg) - 4.438) * radius ** 2, 0.08), 20.0)
        src["mass"] = "from log g and radius"
    else:
        mass = 1.0
        src["mass"] = "assumed Sun-like"
    lum = radius ** 2 * (teff / 5772.0) ** 4
    return {"teff": teff, "radius_rsun": radius, "mass_msun": mass, "luminosity_lsun": lum,
            "hz_inner_au": 0.95 * math.sqrt(lum), "hz_outer_au": 1.67 * math.sqrt(lum),
            "assumed": src}


def planet(period_d: float, depth_ppm: float, duration_hr: float, st: dict, *, sma_au=None, dor=None,
           ror=None, impact=None) -> dict:
    src = {}
    if _ok(ror):
        ror = float(ror)
    else:
        ror = math.sqrt(max(depth_ppm, 1.0) / 1e6)
        src["radius"] = "from transit depth"
    ror = min(ror, 1.5)
    if _ok(sma_au):
        a_au = float(sma_au)
    else:
        a_au = (st["mass_msun"] * (period_d / 365.25) ** 2) ** (1 / 3)
        src["orbit"] = "from Kepler's third law"
    a_rstar = float(dor) if _ok(dor) else a_au / (st["radius_rsun"] * R_SUN_AU)
    if a_rstar < 1.5 and duration_hr > 0:  # inconsistent stellar params: use transit duration
        a_rstar = max(period_d * 24 / (math.pi * duration_hr), 1.5)
        src["orbit"] = "from transit duration"
    b = float(impact) if impact is not None and math.isfinite(float(impact)) and float(impact) >= 0 else 0.3
    inc = math.degrees(math.acos(min(b / a_rstar, 1.0)))
    teq = st["teff"] * math.sqrt(1 / (2 * a_rstar)) * 0.7 ** 0.25
    rp_earth = ror * st["radius_rsun"] * R_EARTH_PER_R_SUN
    hz = "inside" if a_au < st["hz_inner_au"] else "outside" if a_au > st["hz_outer_au"] else "within"
    kind = ("Earth-size" if rp_earth < 1.25 else "super-Earth" if rp_earth < 2 else "sub-Neptune" if rp_earth < 4
            else "Neptune-size" if rp_earth < 6 else "Jupiter-size" if rp_earth < 15 else "very large (likely a star)")
    return {"period_days": period_d, "a_au": a_au, "a_rstar": a_rstar, "ror": ror, "rp_earth": rp_earth,
            "impact": b, "inclination_deg": inc, "teq_k": teq, "insolation_earth": st["luminosity_lsun"] / a_au ** 2,
            "habitable_zone": hz, "size_class": kind, "derived": src}
