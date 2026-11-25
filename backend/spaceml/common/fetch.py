"""Reliable downloads from external archives (NASA SPDF, MAST, mars.nasa.gov ...).

* bounded timeouts and retries with backoff (no endless hangs)
* write to a temporary file, publish with an atomic rename (no half files in the cache)
* a provenance sidecar `<file>.source.json` (URL, retrieval time, size, sha256)
* one readable exception type, `SourceUnavailable`, so the UI can show
  "source unavailable" instead of a stack trace or - worse - an empty "clean" result
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

from spaceml.config import IS_VERCEL
if IS_VERCEL:
    from datetime import datetime, timezone

    def now_iso() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
else:
    from .utils import now_iso

USER_AGENT = "space-ml-mission-control/3.0 (+https://github.com/)"


class SourceUnavailable(RuntimeError):
    """The upstream archive could not be reached or returned an error."""

    def __init__(self, source: str, url: str, reason: str):
        self.source, self.url, self.reason = source, url, reason
        super().__init__(f"{source} is unavailable right now ({reason}). Cached data is kept; try again later. URL: {url}")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def _reason(e: BaseException) -> str:
    if isinstance(e, urllib.error.HTTPError):
        return f"HTTP {e.code}"
    if isinstance(e, urllib.error.URLError):
        r = e.reason
        if isinstance(r, socket.timeout) or "timed out" in str(r):
            return "timed out"
        return f"network error: {r}"
    if isinstance(e, (socket.timeout, TimeoutError)):
        return "timed out"
    return f"{type(e).__name__}: {e}"


def fetch(url: str, dest: Path, *, source: str, timeout: float = 60, retries: int = 3,
          max_bytes: int = 2_000_000_000, backoff: float = 2.0) -> dict:
    """Download `url` to `dest` atomically. Returns the provenance record.

    Retries transient failures (timeouts, 5xx, connection resets) `retries` times.
    HTTP 4xx (except 408/429) fail immediately - retrying will not help.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + f".{os.getpid()}.part")
    last: BaseException | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            h, size = hashlib.sha256(), 0
            with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    size += len(chunk)
                    if size > max_bytes:
                        raise SourceUnavailable(source, url, f"file larger than {max_bytes // 1_000_000} MB")
                    h.update(chunk)
                    f.write(chunk)
            if size == 0:
                raise SourceUnavailable(source, url, "empty response")
            tmp.replace(dest)
            prov = {"source": source, "url": url, "retrieved_at": now_iso(), "bytes": size,
                    "sha256": h.hexdigest(), "attempts": attempt}
            dest.with_name(dest.name + ".source.json").write_text(json.dumps(prov, indent=2))
            return prov
        except SourceUnavailable:
            tmp.unlink(missing_ok=True)
            raise
        except urllib.error.HTTPError as e:
            last = e
            tmp.unlink(missing_ok=True)
            if 400 <= e.code < 500 and e.code not in (408, 429):
                break
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError) as e:
            last = e
            tmp.unlink(missing_ok=True)
        if attempt < retries:
            time.sleep(backoff * attempt)
    raise SourceUnavailable(source, url, _reason(last) if last else "unknown error")


def fetch_json(url: str, cache: Path, *, source: str, max_age_s: float | None = None, timeout: float = 30,
               retries: int = 3) -> tuple[object, dict]:
    """JSON with an on-disk cache. Returns (data, provenance). Falls back to a stale cache
    (marked `stale: true`) when the source is unavailable."""
    fresh = cache.exists() and (max_age_s is None or time.time() - cache.stat().st_mtime < max_age_s)
    if not fresh:
        try:
            fetch(url, cache, source=source, timeout=timeout, retries=retries, max_bytes=200_000_000)
        except SourceUnavailable:
            if not cache.exists():
                raise
            prov = provenance(cache) or {}
            return json.loads(cache.read_text()), {**prov, "stale": True}
    return json.loads(cache.read_text()), provenance(cache) or {"source": source, "url": url}


def provenance(path: Path) -> dict | None:
    side = path.with_name(path.name + ".source.json")
    return json.loads(side.read_text()) if side.exists() else None
