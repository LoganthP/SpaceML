"""What Hubble and Webb are observing now, shown inside the app.

Source: NASA / STScI Space Telescope Live (https://spacetelescopelive.org). The site cannot be embedded in other pages,
so the backend reads the same public JSON its own frontend uses (`GET /api/get/{webb|hubble}` with the header
`endpoint: current`; undocumented, so the parser is defensive) and the app draws its own view.

The sky picture is a ground-based survey image of the target region (DSS2 colour, cut out by the CDS hips2fits service
at the observation's coordinates). Hubble and Webb images are not public in real time, so no live telescope image exists.

Everything fetched is cached under STORAGE_DIR/live, so the last known observation, its sky picture and the list of
observations this app has seen stay available offline (marked as cached, with the time they were fetched)."""
from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from spaceml.config import STORAGE_DIR

from .fetch import USER_AGENT, SourceUnavailable, _reason

SITE = "https://spacetelescopelive.org"
SKY = "https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
TELESCOPES = {"webb": "James Webb Space Telescope", "hubble": "Hubble Space Telescope"}
FOVS = (0.1, 0.5, 2.0)                       # degrees: target close-up, field, wide context
MAX_AGE = float(os.environ.get("SMC_LIVE_MAX_AGE", "120"))   # seconds before the current observation is refreshed
HISTORY = 30
INSTRUMENTS = {"NIRCAM": "NIRCam", "NIRSPEC": "NIRSpec", "MIRI": "MIRI", "NIRISS": "NIRISS", "FGS": "FGS",
               "WFC3": "WFC3", "ACS": "ACS", "COS": "COS", "STIS": "STIS"}


def root() -> Path:
    d = STORAGE_DIR / "live"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get(url: str, headers: dict | None = None, timeout: float = 12, max_bytes: int = 8_000_000) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json, image/*", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read(max_bytes + 1)
    except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError) as e:
        raise SourceUnavailable("Space Telescope Live" if SITE in url else "CDS sky survey", url, _reason(e)) from e
    if len(data) > max_bytes:
        raise SourceUnavailable("Space Telescope Live", url, "response too large")
    return data


def friendly(reason: str) -> str:
    """'network error: [Errno 111] Connection refused' -> 'no internet connection'."""
    r = reason or ""
    if r.startswith("network error") or "refused" in r or "Name or service" in r or "getaddrinfo" in r:
        return "no internet connection, or the site is down"
    if r == "timed out":
        return "the site did not answer in time"
    if r.startswith("HTTP "):
        return f"the site answered {r}"
    return r


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def _fix(text):
    """The upstream data has '?' where a minus sign was lost between digits ('SNR 0509?67.5')."""
    return re.sub(r"(\d)\?(\d)", "\\1\u2212\\2", text) if isinstance(text, str) else text


def _iso(v) -> str | None:
    if not v or not isinstance(v, str):
        return None
    try:
        t = datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize(tel: str, body: dict) -> dict:
    """Map the Webb and Hubble responses (different field names) to one shape."""
    d = (body or {}).get("data")
    if not isinstance(d, dict) or not d.get("id"):
        raise ValueError("no observation in the response")
    p = d.get("proposal") or {}
    webb = tel == "webb"
    if webb:
        t0 = (d.get("targets") or [{}])[0] or {}
        target = t0.get("standardName") or t0.get("name")
    else:
        target = d.get("targetName")
        if isinstance(target, str):
            target = re.sub(r"^NAME[ -]", "", re.sub(r"^ANY,", "", target)).replace("-", " ")
    start = _iso(d.get("scheduledStartTime") if webb else d.get("predictedStartTime"))
    end = _iso(d.get("scheduledEndTime") if webb else d.get("predictedEndTime"))
    ra = _num(d.get("boreSightRightAscensionInDegrees") if webb else d.get("targetRightAscensionInDegrees"))
    dec = _num(d.get("boreSightDeclinationInDegrees") if webb else d.get("targetDeclinationInDegrees"))
    if ra is not None and not 0 <= ra < 360:
        ra = None
    if dec is not None and not -90 <= dec <= 90:
        dec = None
    pi = p.get("primaryInvestigator") or {}
    nxt = ((body.get("_pagination") or {}).get("nextRecord") or {})
    return {
        "telescope": tel, "id": str(d["id"]),
        "target": _fix(target) or "Unknown target",
        "title": _fix(p.get("title")) or "Untitled program",
        "program": str(p["proposalID"]) if p.get("proposalID") is not None else None,
        "category": (p.get("scientificCategory") or {}).get("title"),
        "instruments": [INSTRUMENTS.get(i.get("code", ""), i.get("title")) for i in (p.get("instruments") or []) if isinstance(i, dict) and i.get("title")],
        "pi": pi.get("formalName") or None,
        "start": start, "end": end, "ra": ra, "dec": dec,
        "next_start": _iso(nxt.get("startTime")),
        "url": f"{SITE}/{tel}?obsId={urllib.parse.quote(str(d['id']))}",
    }


def phase(obs: dict, now: float | None = None) -> str:
    now = now or time.time()
    ts = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() if s else None   # noqa: E731
    a, b = ts(obs.get("start")), ts(obs.get("end"))
    if a and now < a:
        return "upcoming"
    if b and now > b:
        return "completed"
    return "in_progress" if a else "unknown"


def _read(p: Path, default):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return default


def _write(p: Path, data) -> None:
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(p)


def current(tel: str, refresh: bool = False) -> dict:
    """The current observation, fresh (`source: live`) or the last one fetched (`source: cached`)."""
    if tel not in TELESCOPES:
        raise KeyError(tel)
    d = root()
    cur, hist = d / f"{tel}-current.json", d / f"{tel}-history.json"
    cached = _read(cur, None)
    fresh = cached and not refresh and time.time() - cached.get("fetched_ts", 0) < MAX_AGE
    error = None
    if not fresh:
        try:
            body = json.loads(_get(f"{SITE}/api/get/{tel}", {"endpoint": "current"}))
            obs = normalize(tel, body)
            cached = {"observation": obs, "fetched_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                      "fetched_ts": time.time()}
            _write(cur, cached)
            h = [o for o in _read(hist, []) if o.get("id") != obs["id"]]
            _write(hist, [obs, *h][:HISTORY])
        except SourceUnavailable as e:
            error = f"Space Telescope Live is not reachable ({friendly(e.reason)})"
        except (ValueError, KeyError, TypeError) as e:
            error = f"Space Telescope Live sent an answer this app could not read ({e})"
    if not cached:
        raise SourceUnavailable("Space Telescope Live", f"{SITE}/{tel}", error.split("(", 1)[-1].rstrip(")") if error and "(" in error else "no data yet")
    obs = cached["observation"]
    return {"telescope": tel, "name": TELESCOPES[tel], "observation": {**obs, "phase": phase(obs)},
            "fetched_at": cached["fetched_at"], "source": "cached" if error else "live", "notice": error,
            "history": [{**o, "phase": phase(o)} for o in _read(hist, [])[1:12]]}


def _known(tel: str, obs_id: str) -> dict | None:
    d = root()
    for o in [(_read(d / f"{tel}-current.json", {}) or {}).get("observation") or {}, *_read(d / f"{tel}-history.json", [])]:
        if o.get("id") == obs_id:
            return o
    return None


def sky(tel: str, obs_id: str, fov: float) -> Path:
    """A survey image of the observation's sky region (only for observations this app has fetched)."""
    if tel not in TELESCOPES or fov not in FOVS or not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", obs_id):
        raise KeyError("unknown observation")
    o = _known(tel, obs_id)
    if not o or o.get("ra") is None or o.get("dec") is None:
        raise KeyError("no coordinates for this observation")
    out = root() / "sky" / f"{tel}-{obs_id}-{fov:g}.jpg"
    if out.exists() and out.stat().st_size > 0:
        return out
    q = urllib.parse.urlencode({"hips": "CDS/P/DSS2/color", "width": 720, "height": 480, "fov": fov, "projection": "TAN",
                                "coordsys": "icrs", "ra": f"{o['ra']:.6f}", "dec": f"{o['dec']:.6f}", "format": "jpg"})
    data = _get(f"{SKY}?{q}", timeout=20)
    if not data.startswith(b"\xff\xd8"):
        raise SourceUnavailable("CDS sky survey", SKY, "did not return an image")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(out)
    return out
