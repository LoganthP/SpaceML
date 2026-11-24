"""NASA SMAP / MSL telemetry: download, load and label handling.

Source: Hundman et al., "Detecting Spacecraft Anomalies Using LSTMs and
Nonparametric Dynamic Thresholding" (KDD 2018), NASA JPL telemanom release.

Files (after download):
    data/telemetry/train/<chan>.npy   (steps, inputs) column 0 = telemetry
    data/telemetry/test/<chan>.npy    column 1.. = one-hot command encodings
    data/telemetry/labeled_anomalies.csv
"""
from __future__ import annotations

import argparse
import ast
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import TELEMETRY_DIR
from ..common.progress import report
from ..common.utils import download

LABELS_URL = "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv"
DATA_ZIP_URL = "https://s3-us-west-2.amazonaws.com/telemanom/data.zip"
# Byte-identical copy of the same NASA release kept on GitHub (used if S3 is unreachable).
GITHUB_MIRROR = "https://raw.githubusercontent.com/imperial-qore/TranAD/main/data/SMAP_MSL"

LABELS_PATH = TELEMETRY_DIR / "labeled_anomalies.csv"
TRAIN_DIR = TELEMETRY_DIR / "train"
TEST_DIR = TELEMETRY_DIR / "test"


@dataclass
class Channel:
    chan_id: str
    spacecraft: str
    sequences: list[list[int]] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    num_values: int = 0

    def to_dict(self) -> dict:
        return {
            "chan_id": self.chan_id,
            "spacecraft": self.spacecraft,
            "anomaly_sequences": self.sequences,
            "anomaly_classes": self.classes,
            "num_values": self.num_values,
        }


def is_downloaded() -> bool:
    """The full NASA set (all ~82 channels) is present."""
    return LABELS_PATH.exists() and TRAIN_DIR.exists() and len(list(TRAIN_DIR.glob("*.npy"))) >= 80


def has_channel(chan_id: str) -> bool:
    return (TRAIN_DIR / f"{chan_id}.npy").exists() and (TEST_DIR / f"{chan_id}.npy").exists()


def is_usable() -> bool:
    """Labels plus at least one channel (e.g. the starter pack subset) are present."""
    return LABELS_PATH.exists() and TEST_DIR.exists() and any(TEST_DIR.glob("*.npy"))


def _parse_classes(s: str) -> list[str]:
    return [c.strip() for c in s.strip("[]").split(",") if c.strip()]


@lru_cache(maxsize=1)
def _load_channels_cached(mtime: float) -> dict[str, Channel]:
    df = pd.read_csv(LABELS_PATH).rename(columns={"class": "anom_class"})
    chans: dict[str, Channel] = {}
    # The official CSV lists P-2 twice (two overlapping expert labels). Merge rows by
    # channel so each channel file is processed once while every labeled event is kept.
    for row in df.itertuples(index=False):
        ch = chans.setdefault(row.chan_id, Channel(row.chan_id, row.spacecraft, num_values=int(row.num_values)))
        ch.sequences.extend([[int(a), int(b)] for a, b in ast.literal_eval(row.anomaly_sequences)])
        ch.classes.extend(_parse_classes(row.anom_class))
    return chans


def load_channels() -> dict[str, Channel]:
    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            "Telemetry labels not found. Run: python -m spaceml.telemetry.data download")
    return _load_channels_cached(LABELS_PATH.stat().st_mtime)


def load_arrays(chan_id: str) -> tuple[np.ndarray, np.ndarray]:
    tr, te = TRAIN_DIR / f"{chan_id}.npy", TEST_DIR / f"{chan_id}.npy"
    if not tr.exists() or not te.exists():
        raise FileNotFoundError(f"Telemetry arrays for {chan_id} not found in {TELEMETRY_DIR}")
    return np.load(tr).astype(np.float32), np.load(te).astype(np.float32)


# --------------------------------------------------------------------------- download
def _download_from_s3() -> None:
    zpath = TELEMETRY_DIR / "data.zip"
    download(DATA_ZIP_URL, zpath, desc="telemanom data.zip", timeout=120)
    with zipfile.ZipFile(zpath) as z:
        for member in z.namelist():
            parts = Path(member).parts
            # zip layout: data/train/X.npy, data/test/X.npy
            if len(parts) >= 3 and parts[-2] in ("train", "test") and member.endswith(".npy"):
                out = TELEMETRY_DIR / parts[-2] / parts[-1]
                out.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    zpath.unlink(missing_ok=True)


def _download_from_mirror(chan_ids: list[str]) -> None:
    jobs = [(split, c) for c in chan_ids for split in ("train", "test")]

    def fetch(job):
        split, c = job
        out = TELEMETRY_DIR / split / f"{c}.npy"
        if not out.exists():
            download(f"{GITHUB_MIRROR}/{split}/{c}.npy", out)
        return out

    with ThreadPoolExecutor(8) as ex:
        for i, _ in enumerate(ex.map(fetch, jobs), 1):
            report(i, len(jobs), f"{i}/{len(jobs)} files")


def download_all(source: str = "auto", from_dir: str | None = None) -> None:
    TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading labeled_anomalies.csv ...")
    download(LABELS_URL, LABELS_PATH)
    chan_ids = sorted(load_channels())

    if from_dir:
        src = Path(from_dir)
        for split in ("train", "test"):
            (TELEMETRY_DIR / split).mkdir(parents=True, exist_ok=True)
            for c in chan_ids:
                shutil.copy(src / split / f"{c}.npy", TELEMETRY_DIR / split / f"{c}.npy")
    elif source in ("auto", "s3"):
        try:
            print("Downloading NASA telemanom archive from S3 ...")
            _download_from_s3()
        except Exception as e:  # noqa: BLE001
            if source == "s3":
                raise
            print(f"  S3 download failed ({e}); falling back to GitHub mirror.")
            _download_from_mirror(chan_ids)
    else:
        _download_from_mirror(chan_ids)

    missing = [c for c in chan_ids if not (TRAIN_DIR / f"{c}.npy").exists()]
    if missing:
        raise SystemExit(f"Missing channels after download: {missing}")
    print(f"Telemetry ready: {len(chan_ids)} channels in {TELEMETRY_DIR}")


def main() -> None:
    p = argparse.ArgumentParser(description="SMAP/MSL telemetry dataset")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="download the NASA SMAP/MSL telemetry")
    d.add_argument("--source", choices=["auto", "s3", "github"], default="auto")
    d.add_argument("--from-dir", help="copy from an already extracted folder with train/ and test/")
    args = p.parse_args()
    if args.cmd == "download":
        download_all(args.source, args.from_dir)


if __name__ == "__main__":
    main()
