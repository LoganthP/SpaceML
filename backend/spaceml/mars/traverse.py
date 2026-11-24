"""Rover Traverse Explorer: Curiosity's real drive path + images taken along it.

Sources (both public NASA/JPL, fetched on demand and cached):
  * Waypoints - the MMGIS layer behind NASA's "Where is Curiosity?" map:
    https://mars.nasa.gov/mmgis-maps/MSL/Layers/json/MSL_waypoints.json
    GeoJSON points with RMC (site_drive), site, drive, sol, lon, lat, dist_total_m, drive_type.
  * Raw images - https://mars.nasa.gov/api/v1/raw_image_items/ (imageid, sol, site, drive,
    instrument, url, date_taken, extended.lmst / mast_az / mast_el / sample_type).

Locating an image. MSL camera product ids encode the rover motion counter: characters
19-21 = site, 22-25 = drive (e.g. NLB_486272784EDR_F0481570NCAM00415M_ -> site 48, drive 1570).
  exact      the waypoint list has that site/drive -> the rover's surveyed position
  site       same site, nearest earlier drive -> position within tens of metres
  sol        only the sol is known -> the position where the rover ended that sol's drive
The precision is always returned with the location and shown on the map.
"""
from __future__ import annotations

import json
import math
import re
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..common.fetch import SourceUnavailable, fetch, fetch_json, provenance
from ..config import MARS_DIR

WAYPOINTS_URL = "https://mars.nasa.gov/mmgis-maps/MSL/Layers/json/MSL_waypoints.json"
RAW_API = "https://mars.nasa.gov/api/v1/raw_image_items/"
SOURCE_WP = "NASA/JPL MMGIS (Curiosity waypoints)"
SOURCE_RAW = "NASA/JPL Mars raw image API"
TRAVERSE_DIR = MARS_DIR / "traverse"
WAYPOINTS_PATH = TRAVERSE_DIR / "MSL_waypoints.json"
RAW_DIR = TRAVERSE_DIR / "raw"
NAV_CAMERAS = ("NAV_LEFT_B", "NAV_LEFT_A")
ALLOWED_IMAGE_HOSTS = ("mars.jpl.nasa.gov", "mars.nasa.gov")

PRODUCT_RE = re.compile(r"(?P<inst>[A-Z]{2})[A-Z_](?P<spec>[A-Z_])(?P<sclk>\d{9})(?P<prod>[A-Z]{3})"
                        r"[A-Z_](?P<samp>[A-Z_])(?P<site>\d{3})(?P<drive>\d{4})")

# MSL spacecraft clock counts seconds from 2000-01-01T11:58:55.816 UTC (J2000).
SCLK_EPOCH = datetime(2000, 1, 1, 11, 58, 55, 816000, tzinfo=timezone.utc)
GALE_LON_E = 137.4417
LANDING_UTC = datetime(2012, 8, 6, 5, 17, 57, tzinfo=timezone.utc)   # sol 0


class TraverseError(ValueError):
    pass


# ------------------------------------------------------------------ time
def _msd(t: datetime) -> float:
    """Mars Sol Date (Allison & McEwen 2000 / Mars24), using TT-UTC = 69.184 s."""
    jd_ut = t.timestamp() / 86400.0 + 2440587.5
    jd_tt = jd_ut + 69.184 / 86400.0
    return (jd_tt - 2405522.0028779) / 1.0274912517


def _local_msd(t: datetime) -> float:
    return _msd(t) + GALE_LON_E / 360.0


_SOL0 = math.floor(_local_msd(LANDING_UTC))


def sol_of(t: datetime) -> int:
    """Curiosity mission sol (local mean solar day at Gale) for a UTC time."""
    return int(math.floor(_local_msd(t)) - _SOL0)


def sclk_to_utc(sclk: int) -> datetime:
    """Approximate UTC of an MSL spacecraft clock value (clock drift and leap seconds ignored: ± minutes)."""
    return SCLK_EPOCH + timedelta(seconds=int(sclk))


def parse_product_id(name: str) -> dict | None:
    base = Path(str(name)).name.upper()
    m = PRODUCT_RE.search(base)
    if not m:
        return None
    d = m.groupdict()
    t = sclk_to_utc(int(d["sclk"]))
    return {"product_id": base[m.start():m.start() + 36].split(".")[0], "instrument": d["inst"],
            "sclk": int(d["sclk"]), "site": int(d["site"]), "drive": int(d["drive"]),
            "utc_estimate": t.replace(microsecond=0).isoformat(), "sol_estimate": sol_of(t)}


# ------------------------------------------------------------------ waypoints
def _parse_waypoints(raw: dict) -> list[dict]:
    if not isinstance(raw, dict) or raw.get("type") != "FeatureCollection":
        raise TraverseError("Waypoint file is not a GeoJSON FeatureCollection - the NASA format may have changed")
    out = []
    for f in raw.get("features", []):
        p = f.get("properties") or {}
        coords = (f.get("geometry") or {}).get("coordinates") or []
        lon = p.get("lon", coords[0] if len(coords) > 1 else None)
        lat = p.get("lat", coords[1] if len(coords) > 1 else None)
        try:
            rec = {"site": int(p["site"]), "drive": int(p["drive"]), "sol": int(p["sol"]),
                   "lon": float(lon), "lat": float(lat),
                   "dist_total_m": float(p.get("dist_total_m") or 0.0),
                   "elev": float(coords[2]) if len(coords) > 2 and coords[2] is not None else None,
                   "drive_type": p.get("drive_type")}
        except (KeyError, TypeError, ValueError):
            continue
        if -90 <= rec["lat"] <= 90 and -360 <= rec["lon"] <= 360:
            out.append(rec)
    if len(out) < 2:
        raise TraverseError("Waypoint file contains no usable waypoints")
    out.sort(key=lambda r: (r["sol"], r["site"], r["drive"]))
    # cumulative distance (recompute when the file has none)
    if not any(r["dist_total_m"] for r in out):
        total = 0.0
        for a, b in zip(out, out[1:]):
            total += _dist_m(a, b)
            b["dist_total_m"] = total
    return out


def _dist_m(a: dict, b: dict) -> float:
    r = 3_389_500.0  # Mars mean radius, m
    p1, p2 = math.radians(a["lat"]), math.radians(b["lat"])
    dl = math.radians(b["lon"] - a["lon"])
    return r * math.acos(max(-1.0, min(1.0, math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(dl))))


def fetch_waypoints() -> dict:
    prov = fetch(WAYPOINTS_URL, WAYPOINTS_PATH, source=SOURCE_WP, timeout=60, retries=3, max_bytes=100_000_000)
    _cache.clear()
    wps = waypoints()
    return {**prov, "waypoints": len(wps), "last_sol": wps[-1]["sol"]}


_cache: dict = {}


def waypoints() -> list[dict]:
    if not WAYPOINTS_PATH.exists():
        raise FileNotFoundError("Curiosity waypoints are not downloaded yet")
    key = WAYPOINTS_PATH.stat().st_mtime
    if _cache.get("key") != key:
        _cache.update(key=key, data=_parse_waypoints(json.loads(WAYPOINTS_PATH.read_text())))
    return _cache["data"]


def status() -> dict:
    if not WAYPOINTS_PATH.exists():
        return {"downloaded": False, "source": SOURCE_WP, "url": WAYPOINTS_URL}
    wps = waypoints()
    return {"downloaded": True, "source": SOURCE_WP, "url": WAYPOINTS_URL, "provenance": provenance(WAYPOINTS_PATH),
            "waypoints": len(wps), "first_sol": wps[0]["sol"], "last_sol": wps[-1]["sol"],
            "total_km": round(wps[-1]["dist_total_m"] / 1000, 2)}


def locate(name_or_id: str, meta: dict | None = None) -> dict | None:
    """Position of an image from its product id (and API metadata when available)."""
    pid = parse_product_id(name_or_id)
    site = (meta or {}).get("site", pid and pid["site"])
    drive = (meta or {}).get("drive", pid and pid["drive"])
    sol = (meta or {}).get("sol")
    if site is None and sol is None:
        return None
    rec = {"site": site, "drive": drive, "sol": sol if sol is not None else (pid or {}).get("sol_estimate"),
           "sol_source": "NASA metadata" if sol is not None else "estimated from spacecraft clock",
           "product_id": (pid or {}).get("product_id") or name_or_id}
    try:
        wps = waypoints()
    except FileNotFoundError:
        return {**rec, "precision": "unlocated", "note": "Download the Curiosity waypoints to place this image on the map"}
    exact = next((w for w in wps if w["site"] == site and w["drive"] == drive), None)
    if exact:
        return {**rec, **_pos(exact), "precision": "exact", "note": "Rover position surveyed at this site/drive"}
    same_site = [w for w in wps if w["site"] == site and drive is not None and w["drive"] <= drive]
    if same_site:
        w = max(same_site, key=lambda x: x["drive"])
        return {**rec, **_pos(w), "precision": "site",
                "note": f"Nearest earlier surveyed position in site {site} (drive {w['drive']})"}
    if rec["sol"] is not None:
        on_sol = [w for w in wps if w["sol"] <= rec["sol"]]
        if on_sol:
            w = on_sol[-1]
            return {**rec, **_pos(w), "precision": "sol",
                    "note": f"Only the sol is known - shown at the rover's end-of-drive position for sol {w['sol']}"}
    return {**rec, "precision": "unlocated", "note": "No waypoint matches this site/drive or sol"}


def _pos(w: dict) -> dict:
    return {"lon": w["lon"], "lat": w["lat"], "dist_total_m": w["dist_total_m"], "waypoint_sol": w["sol"],
            "waypoint_site": w["site"], "waypoint_drive": w["drive"]}


# ------------------------------------------------------------------ raw images
def raw_images(sol: int, site: int | None = None, drive: int | None = None) -> dict:
    """Full-frame Navcam images taken on a sol (left eye only, to avoid stereo duplicates)."""
    if not 0 <= int(sol) <= 20000:
        raise TraverseError("Sol out of range")
    items, provs, errors = [], [], []

    def one(cam: str):
        q = urllib.parse.urlencode({"order": "sol desc", "per_page": 100, "page": 0,
                                    "condition_1": "msl:mission", "condition_2": f"{int(sol)}:sol:in", "search": cam})
        try:
            return fetch_json(f"{RAW_API}?{q}", RAW_DIR / f"sol_{int(sol):05d}_{cam}.json", source=SOURCE_RAW,
                              timeout=15, retries=1)
        except SourceUnavailable as e:
            return e

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(len(NAV_CAMERAS)) as ex:   # both cameras at once: one slow request, not two
        results = list(ex.map(one, NAV_CAMERAS))
    for res in results:
        if isinstance(res, Exception):
            errors.append(str(res))
            continue
        payload, prov = res
        provs.append(prov)
        for it in (payload or {}).get("items", []):
            ext = it.get("extended") or {}
            if (ext.get("sample_type") or "full") != "full":
                continue
            items.append({"imageid": it.get("imageid"), "sol": it.get("sol"), "site": it.get("site"),
                          "drive": it.get("drive"), "instrument": it.get("instrument"), "url": it.get("url"),
                          "date_taken": it.get("date_taken"), "lmst": ext.get("lmst"),
                          "mast_az": ext.get("mast_az"), "mast_el": ext.get("mast_el")})
    if errors and not provs:
        raise SourceUnavailable(SOURCE_RAW, RAW_API, errors[0].split("(")[-1].split(")")[0])
    seen, uniq = set(), []
    for it in items:
        if it["imageid"] and it["imageid"] not in seen:
            seen.add(it["imageid"])
            it["match"] = ("exact" if site is not None and it["site"] == site and it["drive"] == drive else
                           "site" if site is not None and it["site"] == site else "sol")
            uniq.append(it)
    order = {"exact": 0, "site": 1, "sol": 2}
    uniq.sort(key=lambda x: (order[x["match"]], x.get("date_taken") or ""))
    return {"sol": int(sol), "images": uniq, "provenance": provs}


def _find_meta(imageid: str) -> dict | None:
    for f in sorted(RAW_DIR.glob("sol_*.json"), reverse=True) if RAW_DIR.exists() else []:
        try:
            for it in json.loads(f.read_text()).get("items", []):
                if it.get("imageid") == imageid:
                    ext = it.get("extended") or {}
                    return {**{k: it.get(k) for k in ("imageid", "sol", "site", "drive", "instrument", "url", "date_taken")},
                            "lmst": ext.get("lmst"), "mast_az": ext.get("mast_az"), "mast_el": ext.get("mast_el")}
        except (OSError, ValueError):
            continue
    return None


def raw_image(imageid: str) -> tuple[Path, dict]:
    """Download (once) a raw image listed by the API. Only NASA hosts are allowed."""
    if not re.fullmatch(r"[A-Z0-9_]{20,60}", imageid):
        raise TraverseError("Invalid image id")
    meta = _find_meta(imageid)
    if not meta or not meta.get("url"):
        raise TraverseError("Unknown image - open its waypoint first so the image list is loaded")
    url = meta["url"].replace("http://", "https://", 1)
    host = urllib.parse.urlparse(url).hostname or ""
    if not any(host == h or host.endswith("." + h) for h in ALLOWED_IMAGE_HOSTS):
        raise TraverseError(f"Refusing to download from {host}")
    dest = RAW_DIR / "img" / f"{imageid}{Path(urllib.parse.urlparse(url).path).suffix or '.JPG'}"
    if not dest.exists():
        fetch(url, dest, source=SOURCE_RAW, timeout=60, retries=3, max_bytes=40_000_000)
    return dest, meta


# ------------------------------------------------------------------ analysis along the traverse
def sections(points: list[dict], section_m: float = 1000.0) -> list[dict]:
    """Aggregate terrain shares of analysed images per traverse section (by driven distance).

    points: [{dist_total_m, class_shares{...}, reviewed: bool, corrected: bool, sol}]
    """
    buckets: dict[int, dict] = {}
    for p in points:
        if p.get("dist_total_m") is None:
            continue
        k = int(p["dist_total_m"] // section_m)
        b = buckets.setdefault(k, {"index": k, "from_km": k * section_m / 1000, "to_km": (k + 1) * section_m / 1000,
                                   "images": 0, "reviewed": 0, "corrected": 0, "sols": [], "shares": {}})
        b["images"] += 1
        b["reviewed"] += int(bool(p.get("reviewed")))
        b["corrected"] += int(bool(p.get("corrected")))
        if p.get("sol") is not None:
            b["sols"].append(p["sol"])
        for c, v in (p.get("class_shares") or {}).items():
            b["shares"][c] = b["shares"].get(c, 0.0) + float(v)
    out = []
    for k in sorted(buckets):
        b = buckets[k]
        b["shares"] = {c: v / b["images"] for c, v in b["shares"].items()}
        b["sol_range"] = [min(b["sols"]), max(b["sols"])] if b["sols"] else None
        del b["sols"]
        out.append(b)
    return out


def main() -> None:
    import argparse

    from ..common.progress import fail, step
    ap = argparse.ArgumentParser(description="Curiosity traverse waypoints (NASA/JPL MMGIS)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download")
    sub.add_parser("status")
    a = ap.parse_args()
    if a.cmd == "download":
        step("Downloading Curiosity waypoints from NASA/JPL")
        try:
            r = fetch_waypoints()
        except SourceUnavailable as e:
            fail(f"source_unavailable: {e}")
        print(f"{r['waypoints']:,} waypoints up to sol {r['last_sol']} ({r['bytes'] / 1e6:.1f} MB, sha256 {r['sha256'][:16]}…)")
    else:
        print(status())


if __name__ == "__main__":
    main()
