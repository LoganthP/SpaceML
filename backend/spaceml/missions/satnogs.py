"""SatNOGS: decoded engineering telemetry of one named satellite (default UWE-4, NORAD 43880, decoder 'uwe4').

Input: SatNOGS DB telemetry objects (GET https://db.satnogs.org/api/telemetry/?satellite=<norad>, token
required) saved as a JSON list - each has `frame` (hex), `timestamp` (reception time, UTC), `norad_cat_id`,
`observation_id`, `station_id`. Frames are decoded with the official satnogs-decoders Kaitai decoder; only
engineering fields (beacon_payload_*) are kept, protocol/header fields are dropped. Frames that fail to decode
or carry another packet type are counted, not silently dropped. Units are the decoder's raw units.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np

from . import catalog, store
from .fetch import FetchError, retry_after_seconds
from ..common.progress import report, step

DROP = ("crc", "command", "var_id", "typeandlength", "timestamp", "beacon_rate", "packet_id", "rf_message")


API_HOST = "db.satnogs.org"
API = f"https://{API_HOST}/api/telemetry/"
STATUS_FILE = "satnogs_status.json"


def _status_path() -> Path:
    from ..config import STORAGE_DIR
    return STORAGE_DIR / STATUS_FILE


def window_days() -> int:
    from .fetch import env_int
    return max(1, min(365, env_int("SATNOGS_WINDOW_DAYS", 14)))


def token_configured() -> bool:
    return bool(os.environ.get("SATNOGS_TOKEN", "").strip())


def status() -> dict:
    """Last outcome of a SatNOGS fetch (never contains the token): kind, message, next_attempt_at (epoch)."""
    from ..common.utils import read_json
    p = _status_path()
    st = read_json(p, {}) if p.exists() else {}
    if st.get("next_attempt_at") and st["next_attempt_at"] <= time.time():
        st["next_attempt_at"] = None
    return st


def _write_status(**kw) -> None:
    from ..common.utils import write_json
    p = _status_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    write_json(p, {**kw, "updated_at": time.time()})


def blocked_until() -> float | None:
    """Epoch until which a new fetch must not start (after HTTP 429)."""
    st = status()
    return st.get("next_attempt_at") if st.get("kind") == "rate_limited" else None


def record_failure(e: Exception) -> None:
    kind = getattr(e, "kind", None) or ("undecodable" if "decod" in str(e) else "error")
    ra = getattr(e, "retry_after", None)
    nxt = time.time() + ra if kind == "rate_limited" and ra else (time.time() + 15 * 60 if kind == "rate_limited" else None)
    _write_status(kind=kind, message=str(e)[:400], next_attempt_at=nxt)


def record_success() -> None:
    _write_status(kind="ok", message="installed", next_attempt_at=None)


def _next_link(headers, page) -> str | None:
    nxt = None
    for part in (headers.get("Link") or "").split(","):
        if 'rel="next"' in part:
            nxt = part.split(";")[0].strip(" <>")
    if isinstance(page, dict):
        nxt = page.get("next") or nxt
    return nxt or None


def _safe_next(url: str | None) -> str | None:
    """Only follow pagination links on the SatNOGS DB host over HTTPS - the token is never sent elsewhere."""
    if not url:
        return None
    u = urllib.parse.urlparse(url)
    if u.scheme != "https" or u.hostname != API_HOST:
        raise FetchError(f"pagination pointed to {u.scheme}://{u.hostname} - refusing to send the API token there",
                         kind="unsupported")
    return url


def _get(url: str, token: str, attempts: int = 4):
    """One page with bounded back-off on HTTP 429 (honours Retry-After, caps every wait at 120 s)."""
    for k in range(attempts):
        req = urllib.request.Request(url, headers={"Authorization": f"Token {token}", "Accept": "application/json",
                                                   "User-Agent": "space-ml-mission-control"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read() or b"[]"), r.headers
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise FetchError("SatNOGS DB rejected the API token (HTTP %d). Use the DB token (db.satnogs.org → avatar → "
                                 "Settings / API Token) - the SatNOGS Network token does not work here." % e.code,
                                 kind="auth") from e
            if e.code == 429:
                ra = retry_after_seconds(e.headers.get("Retry-After") if e.headers else None)
                if k == attempts - 1 or (ra and ra > 120):
                    wait = ra or 15 * 60
                    raise FetchError(f"SatNOGS DB is rate limiting this account (HTTP 429). Next attempt allowed after "
                                     f"{time.strftime('%H:%M UTC', time.gmtime(time.time() + wait))}; completed pages are kept "
                                     f"and the next attempt resumes.", kind="rate_limited", retry_after=wait) from e
                wait = min(120.0, ra or 10.0 * 2 ** k)
                step(f"SatNOGS DB rate limit (HTTP 429) - waiting {wait:.0f} s before retrying")
                time.sleep(wait)
                continue
            if e.code == 404:
                raise FetchError("SatNOGS DB telemetry endpoint not found (HTTP 404)", kind="not_found") from e
            if k == attempts - 1:
                raise FetchError(f"SatNOGS DB error HTTP {e.code}", kind="network") from e
            time.sleep(5 * 2 ** k)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if k == attempts - 1:
                raise FetchError(f"source_unavailable: SatNOGS DB not reachable ({e})", kind="network") from e
            time.sleep(5 * 2 ** k)
    raise FetchError("SatNOGS DB request failed", kind="network")


def fetch_window(token_file: str | None, start: str | None, end: str | None) -> Path:
    """Download UWE-4 telemetry for a UTC window into a checkpointed cache and return the merged JSON file.

    Completed pages are appended to pages.jsonl together with the cursor of the next page, so an interrupted
    or rate-limited fetch resumes where it stopped; frames are de-duplicated by (timestamp, frame)."""
    from ..config import DATA_DIR
    tok = None
    if token_file:
        tp = Path(token_file)
        tok = tp.read_text().strip() if tp.exists() else None
        tp.unlink(missing_ok=True)
    tok = tok or os.environ.get("SATNOGS_TOKEN", "").strip() or None
    if not tok:
        raise FetchError("SatNOGS DB requires an API token: add SATNOGS_TOKEN=<your DB token> to .env and restart, or paste "
                         "it in the Install panel (single use)", kind="auth")
    wait_until = blocked_until()
    if wait_until:
        raise FetchError(f"SatNOGS DB asked us to wait - next attempt allowed after "
                         f"{time.strftime('%H:%M UTC', time.gmtime(wait_until))}", kind="rate_limited",
                         retry_after=wait_until - time.time())
    cfg = catalog.get("satnogs")["satnogs"]
    if not end:
        end = time.strftime("%Y-%m-%d", time.gmtime(time.time() + 86400))
    if not start:
        start = time.strftime("%Y-%m-%d", time.gmtime(time.mktime(time.strptime(end, "%Y-%m-%d")) - window_days() * 86400))
    cache = DATA_DIR / "missions-source" / "cache" / "satnogs" / f"{cfg['norad']}_{start}_{end}"
    cache.mkdir(parents=True, exist_ok=True)
    pages, state_p = cache / "pages.jsonl", cache / "state.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else {}
    first = f"{API}?{urllib.parse.urlencode({'satellite': cfg['norad'], 'start': start + 'T00:00:00Z', 'end': end + 'T00:00:00Z', 'format': 'json'})}"
    url = state.get("next", first) if not state.get("done") else None
    n_pages = state.get("pages", 0)
    step(f"1/3 Download · UWE-4 frames {start} → {end} (UTC)" + (f", resuming after page {n_pages}" if n_pages else ""))
    while url:
        page, headers = _get(_safe_next(url), tok)
        rows = page if isinstance(page, list) else page.get("results", [])
        nxt = _safe_next(_next_link(headers, page))
        with open(pages, "a") as f:
            f.write(json.dumps(rows) + "\n")
        n_pages += 1
        state = {"next": nxt, "done": nxt is None, "pages": n_pages}
        state_p.write_text(json.dumps(state))                   # checkpoint after every completed page
        report(n_pages, n_pages + (1 if nxt else 0), f"downloaded page {n_pages} ({len(rows)} frames)")
        url = nxt
    seen, merged = set(), []
    if pages.exists():
        for line in pages.read_text().splitlines():
            for r in json.loads(line or "[]"):
                key = (r.get("timestamp"), r.get("frame"))
                if key not in seen:
                    seen.add(key)
                    merged.append(r)
    if not merged:
        raise FetchError(f"SatNOGS DB answered, but has no UWE-4 frames between {start} and {end}. Choose an earlier "
                         f"window (From / To dates) - recent coverage can be sparse.", kind="no_frames")
    out = cache / "telemetry.json"
    out.write_text(json.dumps(merged))
    step(f"2/3 Verify · {len(merged)} unique frames from {n_pages} page(s)")
    return out


def download(out: Path, token: str | None, start: str | None, end: str | None) -> Path:
    """Command-line helper kept for release preparation: same as fetch_window with an explicit token."""
    if token:
        os.environ["SATNOGS_TOKEN"] = token
    got = fetch_window(None, start, end)
    out.write_text(got.read_text())
    return out


HEX = __import__("re").compile(r"^[0-9A-Fa-f]{20,}$")


def read_rows(src: Path) -> list[dict]:
    """SatNOGS DB API JSON (list or paged {"results": [...]}) or the website's frames/CSV export
    (one frame per line: timestamp and hex frame separated by '|' or ',')."""
    text = src.read_text(errors="replace").strip()
    if text[:1] in "[{":
        raw = json.loads(text)
        return raw if isinstance(raw, list) else raw.get("results", [])
    cfg = catalog.get("satnogs")["satnogs"]
    rows = []
    for line in text.splitlines():
        parts = [x.strip().strip('"') for x in line.replace("|", ",").split(",")]
        frame = next((x for x in parts if HEX.match(x)), None)
        ts = next((x for x in parts if len(x) >= 10 and x[:4].isdigit() and "-" in x[:8]), None)
        if frame and ts:
            rows.append({"timestamp": ts.replace(" ", "T"), "frame": frame, "norad_cat_id": cfg["norad"]})
    return rows


def prepare(src: Path) -> dict:
    from satnogsdecoders import decoder as dec
    cfg = catalog.get("satnogs")["satnogs"]
    rows = read_rows(Path(src))
    if not rows:
        raise ValueError("no telemetry objects in the input")
    D = getattr(dec, cfg["decoder"].capitalize())
    recs, fails, other_sat, dup = [], 0, 0, 0
    seen = set()
    for r in rows:
        if int(r.get("norad_cat_id", cfg["norad"])) != cfg["norad"]:
            other_sat += 1
            continue
        key = (r.get("timestamp"), r.get("frame"))
        if key in seen:
            dup += 1
            continue
        seen.add(key)
        try:
            fields = dec.get_fields(D.from_bytes(bytes.fromhex(r["frame"])))
        except Exception:  # noqa: BLE001 - undecodable / other packet type
            fails += 1
            continue
        eng = {k: float(v) for k, v in fields.items()
               if k.startswith("beacon_payload_") and not any(x in k for x in DROP) and isinstance(v, (int, float))}
        if not eng:
            fails += 1
            continue
        tstr = str(r["timestamp"]).replace("Z", "+00:00")
        dt = datetime.fromisoformat(tstr)
        ts = (dt if dt.tzinfo else dt.replace(tzinfo=__import__("datetime").timezone.utc)).timestamp()
        recs.append((ts, eng, r.get("observation_id"), r.get("station_id")))
    if not recs:
        raise FetchError(f"{len(rows)} frames received, but the '{cfg['decoder']}' decoder could not parse any of them "
                         f"({fails} undecodable, {other_sat} other satellite) - nothing was installed", kind="undecodable")
    if len(recs) < 20:
        raise FetchError(f"only {len(recs)} decodable engineering frames ({fails} undecodable) - need at least 20 for replay; "
                         "choose a longer window", kind="undecodable")
    recs.sort(key=lambda x: x[0])
    t = np.asarray([r[0] for r in recs])
    names = sorted({k for r in recs for k in r[1]})
    usable = []
    for n in names:                                   # a channel must have enough readings that actually vary
        v = np.asarray([r[1].get(n, np.nan) for r in recs], dtype=np.float64)
        fin = v[np.isfinite(v)]
        if len(fin) >= 20 and np.nanstd(fin) > 0:
            usable.append(n)
    if not usable:
        raise FetchError(f"{len(recs)} frames decoded, but no field has 20+ varying readings - nothing usable to replay",
                         kind="undecodable")
    names = usable
    step(f"3/3 Import · {len(recs)} decoded frames, {len(names)} usable channels")
    w = store.Writer("satnogs")
    for n in names:
        v = np.asarray([r[1].get(n, np.nan) for r in recs], dtype=np.float64)
        w.add({"id": n, "label": n.replace("beacon_payload_", "").replace("_", " "), "field": n,
               "units": {"value": "decoder raw units"}, "decoder": cfg["decoder"]}, t, {"value": v})
    obs = {r[2] for r in recs if r[2]}
    stations = {r[3] for r in recs if r[3]}
    return w.publish({
        "source": {"name": f"SatNOGS DB telemetry - {cfg['satellite']} (NORAD {cfg['norad']})", "decoder": cfg["decoder"],
                   "satellite_id": cfg["norad"], "files": {Path(src).name: {"sha256": hashlib.sha256(Path(src).read_bytes()).hexdigest(),
                                                                            "bytes": Path(src).stat().st_size}},
                   "license": "SatNOGS DB data - verify terms before redistribution",
                   "attribution": "SatNOGS / Libre Space Foundation and the contributing ground-station observers."},
        "scope": f"{len(recs)} decoded frames, {len(names)} engineering fields", "time_is_epoch": True,
        "time_basis": "ground-station reception time of each frame (UTC); irregular sampling with pass gaps",
        "date_range": [datetime.utcfromtimestamp(t[0]).isoformat() + "Z", datetime.utcfromtimestamp(t[-1]).isoformat() + "Z"],
        "counts": {"frames_in": len(rows), "decoded": len(recs), "undecodable": fails, "other_satellite": other_sat,
                   "duplicates": dup, "observations": len(obs), "stations": len(stations), "fields": len(names)},
        "channels": [{"id": n, "name": n, "unit": "decoder raw units"} for n in names]})
