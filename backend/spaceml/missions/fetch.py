"""Robust downloads for mission data: a stable cache, verified resume, checksums, and remote-ZIP member reads.

* ``download(url, dest, size=, md5=)`` keeps ``dest.part`` across retries and restarts. A resumed request must
  come back as HTTP 206 with a ``Content-Range`` starting exactly at the bytes already on disk; a server that
  ignores the range (200) restarts the file cleanly instead of appending a second copy. The finished file is
  checked against the expected size and MD5 before it is renamed into place; a mismatch deletes it.
* ``RemoteFile`` is a seekable, read-only file over HTTP range requests. ``zipfile.ZipFile(RemoteFile(url))``
  reads the central directory from the end of a remote archive and extracts single members without downloading
  the rest - but only when the host honours range requests (``supports_ranges`` checks that first).
* ``ensure_space(path, need)`` refuses to start when the disk cannot hold the download (plus extraction).
"""
from __future__ import annotations

import hashlib
import io
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from ..common.progress import report, step

UA = {"User-Agent": "space-ml-mission-control (+https://github.com/LoganthP)"}
CHUNK = 1 << 20


class FetchError(ValueError):
    """Readable download problem. ``kind``: auth | not_found | rate_limited | network | integrity | space | unsupported."""

    def __init__(self, msg: str, kind: str = "network", retry_after: float | None = None):
        super().__init__(msg)
        self.kind = kind
        self.retry_after = retry_after


def retry_after_seconds(value: str | None) -> float | None:
    """Retry-After is either delta-seconds or an HTTP date."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        from email.utils import parsedate_to_datetime
        return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
    except (TypeError, ValueError):
        return None


def ensure_space(path: Path, need_bytes: int, what: str = "this download") -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free < need_bytes:
        raise FetchError(f"not enough disk space for {what}: {need_bytes / 1e9:.1f} GB needed, "
                         f"{free / 1e9:.1f} GB free on the data volume", kind="space")


def md5_file(p: Path, label: str = "") -> str:
    h = hashlib.md5()  # noqa: S324 - integrity check against the publisher's MD5, not security
    total = p.stat().st_size
    done, t = 0, time.time()
    with open(p, "rb") as f:
        while chunk := f.read(8 * CHUNK):
            h.update(chunk)
            done += len(chunk)
            if time.time() - t > 2:
                report(done, total, f"verifying {label or p.name}: {done / 1e9:.2f} / {total / 1e9:.2f} GB")
                t = time.time()
    return h.hexdigest()


def _open(url: str, headers: dict, timeout: int):
    return urllib.request.urlopen(urllib.request.Request(url, headers={**UA, **headers}), timeout=timeout)


def download(url: str, dest: Path, size: int | None = None, md5: str | None = None, label: str = "",
             timeout: int = 120, attempts: int = 6) -> Path:
    """Resumable, verified download into ``dest`` (kept as ``dest.part`` until verified)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    label = label or dest.name
    if dest.exists():
        if (size is None or dest.stat().st_size == size) and (md5 is None or md5_file(dest, label) == md5):
            step(f"{label}: already downloaded and verified - reusing the cached copy")
            return dest
        dest.unlink()                                      # stale / corrupt cached copy
    part = dest.with_name(dest.name + ".part")
    if size and part.exists() and part.stat().st_size > size:
        part.unlink()
    last: Exception | None = None
    for attempt in range(attempts):
        have = part.stat().st_size if part.exists() else 0
        if size and have == size:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with _open(url, headers, timeout) as r:
                ctype = (r.headers.get("Content-Type") or "").lower()
                if "text/html" in ctype or "application/json" in ctype:
                    raise FetchError(f"{label}: the server returned {ctype.split(';')[0]} instead of the file "
                                     f"(wrong download link?)", kind="unsupported")
                if have and r.status == 206:
                    m = re.match(r"bytes (\d+)-(\d+)/(\d+|\*)", r.headers.get("Content-Range") or "")
                    if not m or int(m.group(1)) != have:
                        part.unlink(missing_ok=True)       # range not where we are: start over cleanly
                        raise FetchError(f"{label}: Content-Range {r.headers.get('Content-Range')!r} does not "
                                         f"continue at byte {have}; restarting", kind="network")
                    mode, done = "ab", have
                    total = int(m.group(3)) if m.group(3) != "*" else (size or 0)
                else:
                    if have:
                        step(f"{label}: the server ignored the resume request - restarting from 0")
                    mode, done = "wb", 0
                    total = int(r.headers.get("Content-Length") or 0) or (size or 0)
                if size and total and total != size:
                    raise FetchError(f"{label}: server reports {total:,} bytes, expected {size:,}", kind="integrity")
                with open(part, mode) as f:
                    t = time.time()
                    while chunk := r.read(CHUNK):
                        f.write(chunk)
                        done += len(chunk)
                        if time.time() - t > 2:
                            report(done, total or done, f"downloading {label}: {done / 1e6:,.0f}"
                                   + (f" / {total / 1e6:,.0f} MB" if total else " MB"))
                            t = time.time()
            if size is None or part.stat().st_size == size:
                break
        except urllib.error.HTTPError as e:
            if e.code == 416 and size and have == size:
                break
            if e.code in (401, 403):
                raise FetchError(f"{label}: HTTP {e.code} - access denied", kind="auth") from e
            if e.code == 404:
                raise FetchError(f"{label}: HTTP 404 - not found at {urllib.parse.urlparse(url).netloc}", kind="not_found") from e
            if e.code == 429:
                ra = retry_after_seconds(e.headers.get("Retry-After") if e.headers else None)
                raise FetchError(f"{label}: HTTP 429 - the server is rate limiting downloads", kind="rate_limited",
                                 retry_after=ra) from e
            last = e
        except FetchError as e:
            if e.kind != "network":
                raise
            last = e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            if isinstance(e, urllib.error.URLError) and "Name or service not known" in str(e):
                raise FetchError(f"no internet connection (could not resolve {urllib.parse.urlparse(url).netloc})",
                                 kind="network") from e
            last = e
        time.sleep(min(60, 3 * 2 ** attempt))
    else:
        raise FetchError(f"source_unavailable: could not download {label} after {attempts} attempts ({last}); the "
                         f"partial file is kept and the next attempt resumes it", kind="network")
    got = part.stat().st_size
    if size is not None and got != size:
        raise FetchError(f"{label}: {got:,} bytes downloaded, expected {size:,} - the partial file is kept", kind="network")
    if md5:
        step(f"Verifying {label} (MD5)")
        digest = md5_file(part, label)
        if digest != md5:
            part.unlink(missing_ok=True)
            raise FetchError(f"{label}: checksum mismatch (md5 {digest}, expected {md5}) - deleted, retry downloads it again",
                             kind="integrity")
    part.replace(dest)
    return dest


# ------------------------------------------------------------------ remote ZIP members via HTTP ranges
def supports_ranges(url: str, timeout: int = 60) -> tuple[bool, int | None]:
    """(True, total size) when a 1-byte range request comes back as 206 with a Content-Range."""
    try:
        with _open(url, {"Range": "bytes=0-0"}, timeout) as r:
            cr = r.headers.get("Content-Range") or ""
            m = re.match(r"bytes 0-0/(\d+)", cr)
            ok = r.status == 206 and bool(m)
            r.read(1)
            return ok, int(m.group(1)) if m else None
    except urllib.error.URLError as e:
        if not isinstance(e, urllib.error.HTTPError):
            raise FetchError(f"source_unavailable: {urllib.parse.urlparse(url).hostname} is not reachable from this machine "
                             f"({getattr(e, 'reason', e)}). Check the internet connection and press Install again - "
                             "anything already downloaded is kept.", kind="network") from e
        if e.code == 429:
            raise FetchError("the host is rate limiting requests (HTTP 429)", kind="rate_limited",
                             retry_after=retry_after_seconds(e.headers.get("Retry-After") if e.headers else None)) from e
        raise FetchError(f"range probe failed: HTTP {e.code}", kind="not_found" if e.code == 404 else "network") from e


class RemoteFile(io.RawIOBase):
    """Read-only, seekable view of a remote file (HTTP range requests, small read-ahead cache)."""

    def __init__(self, url: str, size: int, block: int = 1 << 20, timeout: int = 120):
        self.url, self.size, self.block, self.timeout = url, size, block, timeout
        self.pos = 0
        self._cache: tuple[int, bytes] = (-1, b"")
        self.bytes_fetched = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def _get(self, a: int, b: int) -> bytes:
        for attempt in range(5):
            try:
                with _open(self.url, {"Range": f"bytes={a}-{b - 1}"}, self.timeout) as r:
                    if r.status != 206:
                        raise FetchError("the host stopped honouring range requests", kind="unsupported")
                    data = r.read()
                self.bytes_fetched += len(data)
                return data
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    time.sleep(min(120, retry_after_seconds(e.headers.get("Retry-After") if e.headers else None) or 10 * 2 ** attempt))
                    continue
                raise FetchError(f"range read failed: HTTP {e.code}") from e
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt == 4:
                    raise FetchError(f"range read failed: {e}") from e
                time.sleep(2 * 2 ** attempt)
        raise FetchError("range read failed after retries", kind="rate_limited")

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if n == 0:
            return b""
        start, data = self._cache
        head = b""
        if start >= 0 and start <= self.pos < start + len(data):      # serve what the cache already holds
            head = data[self.pos - start:self.pos - start + n]
            self.pos += len(head)
            n -= len(head)
            if n == 0:
                return head
        if n >= self.block:
            out = self._get(self.pos, self.pos + n)
        else:
            end = min(self.size, self.pos + max(n, self.block))
            data = self._get(self.pos, end)
            self._cache = (self.pos, data)
            out = data[:n]
        self.pos += len(out)
        return head + out

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default
