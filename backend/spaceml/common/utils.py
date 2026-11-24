from __future__ import annotations

import json
import os
import random
import shutil
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_version() -> str:
    return "v" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def sanitize(obj: Any) -> Any:
    """Recursively make an object strict-JSON safe (NaN/inf -> None, numpy -> Python)."""
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(sanitize(obj), indent=2, default=_json_default, allow_nan=False))
    tmp.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def clean_floats(values, digits: int = 6) -> list:
    """Convert an array to a JSON-safe list (NaN/inf -> None), rounded."""
    arr = np.asarray(values, dtype=float)
    out = np.round(arr, digits).astype(object)
    out[~np.isfinite(arr)] = None
    return out.tolist()


def download(url: str, dest: Path, desc: str | None = None, timeout: int = 60) -> Path:
    """Stream a URL to disk with a simple progress line. Raises on failure."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "space-ml-mission-control/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, t0 = 0, time.time()
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if desc and sys.stdout.isatty():
                pct = f"{100 * done / total:5.1f}%" if total else f"{done / 1e6:.1f} MB"
                print(f"\r  {desc}: {pct} ({done / 1e6 / max(time.time() - t0, 1e-3):.1f} MB/s)",
                      end="", flush=True)
    if desc and sys.stdout.isatty():
        print()
    shutil.move(tmp, dest)
    return dest
