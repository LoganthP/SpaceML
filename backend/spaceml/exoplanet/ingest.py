"""Parse user-uploaded light-curve datasets into clean, analysis-ready targets.

Accepted inputs (any mix, also inside .zip / .tar / .tar.gz / .tgz / .gz archives):
  * FITS light curves from Kepler, K2, TESS (SPOC, QLP, TESS-SPOC, eleanor, K2SFF, EVEREST ...)
  * Tables: .csv .tsv .txt .dat .tbl (IPAC) .ecsv .xlsx .xls .json .parquet
    - columns are detected by name (time/bjd/btjd/bkjd/jd/mjd/hjd/date, flux/pdcsap_flux/
      sap_flux/rel_flux/..., or mag/magnitude which is converted to relative flux)
    - files without a header: first column = time, second = flux, third = error
    - a target/star/object/tic/kic column splits one table into several targets

Files with the same target id (FITS header OBJECT/KEPLERID/TICID/..., or the table's
target column) are stitched; each segment is normalised by its median first.
"""
from __future__ import annotations

import gzip
import io
import re
import shutil
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

MAX_FILES = 2000
MAX_TOTAL_BYTES = 2_000_000_000
MIN_POINTS = 50
TABLE_EXT = {".csv", ".tsv", ".txt", ".dat", ".tbl", ".ecsv", ".xlsx", ".xls", ".json", ".jsonl", ".ndjson", ".parquet", ".lc", ".ascii"}
FITS_EXT = {".fits", ".fit", ".fts"}
ARCHIVE_EXT = {".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz"}

TIME_NAMES = ["time", "bjd", "btjd", "bkjd", "bjd_tdb", "bjdtdb", "hjd", "jd", "mjd", "t", "times", "date",
              "datetime", "timestamp", "epoch", "jd_utc", "bjd_utc", "barycentric_time"]
FLUX_NAMES = ["pdcsap_flux", "kspsap_flux", "det_flux", "fcor", "flux", "sap_flux", "rel_flux", "relflux",
              "flux_norm", "normflux", "norm_flux", "normalized_flux", "corr_flux", "cal_flux", "flux_raw",
              "f", "intensity", "counts", "rate", "brightness", "lc"]
MAG_NAMES = ["mag", "magnitude", "tmag", "kepmag", "vmag", "rmag", "gmag", "imag", "diff_mag", "dmag", "m"]
ERR_NAMES = ["pdcsap_flux_err", "kspsap_flux_err", "flux_err", "fluxerr", "err", "error", "sigma", "e_flux",
             "flux_error", "dflux", "unc", "uncertainty", "sap_flux_err", "mag_err", "magerr", "e_mag"]
TARGET_NAMES = ["target", "target_id", "targetid", "object", "objectid", "star", "star_id", "name", "id",
                "tic", "tic_id", "ticid", "kic", "kic_id", "kepid", "epic", "epic_id", "source", "source_id"]
QUALITY_NAMES = ["quality", "sap_quality", "qual", "flag", "flags"]


class IngestError(ValueError):
    pass


@dataclass
class Segment:
    target: str
    time: np.ndarray
    flux: np.ndarray
    err: np.ndarray | None
    source: str
    columns: dict
    mission: str | None = None
    time_system: str = "days"
    meta: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", str(name).strip().lower().replace(" ", "_").replace("-", "_"))


def _pick(cols: dict[str, str], names: list[str]) -> str | None:
    for n in names:
        if n in cols:
            return cols[n]
    for n in names:  # prefix/suffix match e.g. "flux_corrected", "bjd_2457000"
        for k, orig in cols.items():
            if len(n) > 2 and (k.startswith(n + "_") or k.endswith("_" + n)) and "err" not in k:
                return orig
    return None


# ------------------------------------------------------------------ archives
from ..common.zipsupport import enable_deflate64

enable_deflate64()


def expand(path: Path, out_dir: Path) -> list[Path]:
    """Return a flat list of data files, safely extracting archives (no path traversal)."""
    files: list[Path] = []
    total = [0]

    def safe_target(name: str) -> Path | None:
        parts = [p for p in Path(name).parts if p not in ("..", "/", "\\") and not p.startswith(".")]
        if not parts or "__MACOSX" in parts:
            return None
        return out_dir.joinpath(*parts)

    def visit(p: Path, depth: int = 0):
        if len(files) > MAX_FILES:
            raise IngestError(f"Too many files in the upload (limit {MAX_FILES})")
        name = p.name.lower()
        if depth > 3:
            return
        if zipfile.is_zipfile(p):
            with zipfile.ZipFile(p) as z:
                for info in z.infolist():
                    if info.is_dir():
                        continue
                    dest = safe_target(info.filename)
                    if dest is None:
                        continue
                    total[0] += info.file_size
                    if total[0] > MAX_TOTAL_BYTES:
                        raise IngestError("Archive is too large once extracted (limit 2 GB)")
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(info) as src, open(dest, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    visit(dest, depth + 1)
            return
        if name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")) and tarfile.is_tarfile(p):
            with tarfile.open(p) as t:
                for m in t.getmembers():
                    if not m.isfile():
                        continue
                    dest = safe_target(m.name)
                    if dest is None:
                        continue
                    total[0] += m.size
                    if total[0] > MAX_TOTAL_BYTES:
                        raise IngestError("Archive is too large once extracted (limit 2 GB)")
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with t.extractfile(m) as src, open(dest, "wb") as dst:  # type: ignore[union-attr]
                        shutil.copyfileobj(src, dst)
                    visit(dest, depth + 1)
            return
        if name.endswith(".gz") and not name.endswith((".fits.gz", ".fit.gz", ".fts.gz")):
            dest = p.with_suffix("")
            with gzip.open(p, "rb") as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
            visit(dest, depth + 1)
            return
        files.append(p)

    visit(path)
    return files


def kind_of(p: Path) -> str | None:
    n = p.name.lower()
    if n.endswith((".fits.gz", ".fit.gz", ".fts.gz")) or p.suffix.lower() in FITS_EXT:
        return "fits"
    if p.suffix.lower() in TABLE_EXT:
        return "table"
    try:
        with open(p, "rb") as fh:
            head = fh.read(80)
        if head.startswith(b"SIMPLE  ="):
            return "fits"
        if head and all(32 <= b < 127 or b in (9, 10, 13) for b in head):
            return "table"
    except OSError:
        pass
    return None


# ------------------------------------------------------------------ FITS
def read_fits(p: Path) -> list[Segment]:
    from astropy.io import fits

    try:
        hdul = fits.open(p, memmap=False)
    except Exception as e:  # noqa: BLE001
        raise IngestError(f"{p.name}: not a readable FITS file ({e})") from e
    with hdul:
        h0 = hdul[0].header
        table = None
        for hdu in hdul[1:]:
            if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)) and hdu.data is not None:
                names = {_norm(n) for n in hdu.columns.names}
                if names & {"time", "bjd", "btjd", "t", "hjd", "jd", "mjd"}:
                    table = hdu
                    break
        if table is None:
            if hdul[0].data is not None and getattr(hdul[0].data, "ndim", 0) >= 2:
                raise IngestError(f"{p.name}: this is an image / target pixel file, not a light curve. "
                                  "Upload the light-curve (LC) product instead")
            raise IngestError(f"{p.name}: no light-curve table with a TIME column was found")
        h1 = table.header
        cols = {_norm(n): n for n in table.columns.names}
        tcol = _pick(cols, TIME_NAMES)
        fcol = _pick(cols, FLUX_NAMES)
        mcol = None if fcol else _pick(cols, MAG_NAMES)
        if not tcol or not (fcol or mcol):
            raise IngestError(f"{p.name}: could not find time and flux columns (has: {', '.join(table.columns.names[:12])})")
        d = table.data
        if np.asarray(d[fcol or mcol]).ndim > 1:
            raise IngestError(f"{p.name}: this is a target pixel file (an image per cadence), not a light curve. "
                              "Upload the light-curve (LC) product instead")
        t = np.asarray(d[tcol], dtype=np.float64)
        y = np.asarray(d[fcol or mcol], dtype=np.float64)
        ecol = _pick(cols, [f"{_norm(fcol)}_err"] + ERR_NAMES) if fcol else _pick(cols, ["mag_err", "magerr", "e_mag", "err"])
        e = np.asarray(d[ecol], dtype=np.float64) if ecol else None
        qcol = _pick(cols, QUALITY_NAMES)
        warnings = []
        if qcol:
            q = np.asarray(d[qcol])
            good = q == 0
            if good.sum() > 0.2 * len(q):
                t, y = t[good], y[good]
                e = e[good] if e is not None else None
            else:
                warnings.append(f"{p.name}: most cadences have quality flags set; kept all of them")
        # PDCSAP can be entirely NaN for some sectors: fall back to SAP
        if fcol and np.isfinite(y).sum() < MIN_POINTS and "SAP_FLUX" in table.columns.names and fcol != "SAP_FLUX":
            y = np.asarray(d["SAP_FLUX"], dtype=np.float64)
            fcol = "SAP_FLUX"
            warnings.append(f"{p.name}: PDCSAP flux empty, used SAP flux")

        def hv(*keys):
            for k in keys:
                for h in (h0, h1):
                    if k in h and h[k] not in (None, ""):
                        return h[k]
            return None

        mission = hv("TELESCOP", "MISSION")
        tid = hv("OBJECT")
        for key, prefix in (("TICID", "TIC"), ("KEPLERID", "KIC"), ("EPIC", "EPIC")):
            v = hv(key)
            if v not in (None, 0, "0"):
                tid = tid or f"{prefix} {v}"
        meta = {}
        for k, name in (("RA_OBJ", "ra"), ("DEC_OBJ", "dec"), ("TEFF", "teff"), ("RADIUS", "radius"),
                        ("LOGG", "logg"), ("TESSMAG", "tmag"), ("KEPMAG", "kepmag")):
            v = hv(k)
            try:
                if v is not None and np.isfinite(float(v)):
                    meta[name] = float(v)
            except (TypeError, ValueError):
                pass
        bjdref = (hv("BJDREFI") or 0) + (hv("BJDREFF") or 0)
        tsys = f"BJD - {bjdref:.0f}" if bjdref else str(h1.get("TIMESYS") or "days")
        mag = mcol is not None
        if mag:
            y, e = _mag_to_flux(y, e)
        return [Segment(str(tid or p.stem), t, y, e, p.name,
                        {"time": tcol, "flux": fcol or mcol, "error": ecol, "quality": qcol, "magnitudes": mag},
                        str(mission) if mission else None, tsys, meta, warnings)]


def _mag_to_flux(m: np.ndarray, e: np.ndarray | None):
    med = np.nanmedian(m)
    f = 10 ** (-0.4 * (m - med))
    ef = None if e is None else np.abs(f * 0.4 * np.log(10) * e)
    return f, ef


# ------------------------------------------------------------------ tables
def _read_json_table(p: Path) -> pd.DataFrame:
    """A JSON array/object, or JSON Lines (one object per line - the "Trailing data" case)."""
    import json
    if p.suffix.lower() == ".json":
        try:
            obj = pd.read_json(p)
            return obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)
        except ValueError as e:
            if "Trailing data" not in str(e) and "Extra data" not in str(e):
                raise IngestError(f"{p.name}: not valid JSON ({e})") from e
    rows = []
    with open(p, encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip().rstrip(",")
            if not line or line in ("[", "]"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise IngestError(f"{p.name}, line {i}: not valid JSON ({e.msg})") from e
    if not rows:
        raise IngestError(f"{p.name}: file is empty")
    return pd.json_normalize(rows, max_level=1)


def _read_table(p: Path) -> pd.DataFrame:
    ext = p.suffix.lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(p)
    if ext == ".parquet":
        return pd.read_parquet(p)
    if ext in (".json", ".jsonl", ".ndjson"):
        return _read_json_table(p)
    if ext in (".tbl", ".ecsv"):
        from astropy.table import Table
        return Table.read(p, format="ascii.ipac" if ext == ".tbl" else "ascii.ecsv").to_pandas()
    text = p.read_text(errors="replace")
    lines = [l for l in text.splitlines() if l.strip()]
    # strip comment lines but keep a commented header like "# time flux err"
    header_hint = None
    body = []
    for l in lines:
        s = l.lstrip()
        if s.startswith(("#", "%", "\\", "|", ";", "!")):
            cand = re.split(r"[\s,;|]+", s.lstrip("#%\\|;! ").strip())
            if len(cand) >= 2 and any(_norm(c) in TIME_NAMES for c in cand):
                header_hint = cand
            continue
        body.append(l)
    if not body:
        raise IngestError(f"{p.name}: file is empty")
    sample = "\n".join(body[:50])
    sep = "," if sample.count(",") >= len(body[:50]) else "\t" if sample.count("\t") >= len(body[:50]) else ";" \
        if sample.count(";") >= len(body[:50]) else r"\s+"
    first = re.split(sep if sep != r"\s+" else r"\s+", body[0].strip())
    has_header = any(re.search(r"[A-Za-z_]", c) and not re.fullmatch(r"[-+]?[\d.]+(e[-+]?\d+)?|nan|NaN|inf", c)
                     for c in first)
    df = pd.read_csv(io.StringIO("\n".join(body)), sep=sep, engine="python", header=0 if has_header else None)
    if not has_header:
        df.columns = header_hint[: df.shape[1]] + [f"col{i}" for i in range(len(header_hint or []), df.shape[1])] \
            if header_hint and len(header_hint) >= 2 else [f"col{i}" for i in range(df.shape[1])]
    return df


def _to_days(series: pd.Series, name: str) -> tuple[np.ndarray, str]:
    if not pd.api.types.is_numeric_dtype(series):
        dt = pd.to_datetime(series, errors="coerce", utc=True)
        if dt.notna().mean() > 0.9:
            jd = (dt - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy() / 86400 + 2440587.5
            return jd, "JD (from dates)"
        series = pd.to_numeric(series, errors="coerce")
    t = series.to_numpy(dtype=np.float64)
    finite = t[np.isfinite(t)]
    if len(finite) > 2:
        step = np.median(np.diff(np.sort(finite)))
        span = np.ptp(finite)
        if ("sec" in name.lower() or (step >= 10 and span > 5e4 and finite.min() < 2.3e6)):
            return t / 86400.0, "days (converted from seconds)"
        if "min" in name.lower() and step >= 0.5:
            return t / 1440.0, "days (converted from minutes)"
        if "hour" in name.lower() or name.lower() in ("hr", "hrs"):
            return t / 24.0, "days (converted from hours)"
    return t, name


def read_table(p: Path) -> list[Segment]:
    try:
        df = _read_table(p)
    except IngestError:
        raise
    except Exception as e:  # noqa: BLE001
        raise IngestError(f"{p.name}: could not read as a table ({e})") from e
    if df.empty or df.shape[1] < 2:
        raise IngestError(f"{p.name}: needs at least two columns (time and flux)")
    cols = {_norm(c): c for c in df.columns}
    tcol = _pick(cols, TIME_NAMES)
    fcol = _pick(cols, FLUX_NAMES)
    mcol = None if fcol else _pick(cols, MAG_NAMES)
    targcol = _pick(cols, TARGET_NAMES)
    warnings = []
    numeric = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.9 and c != targcol]
    if not tcol:
        tcol = numeric[0] if numeric else None
        if tcol is not None:
            warnings.append(f"{p.name}: no time column name recognised; used first numeric column '{tcol}'")
    if not (fcol or mcol):
        rest = [c for c in numeric if c != tcol]
        fcol = rest[0] if rest else None
        if fcol is not None:
            warnings.append(f"{p.name}: no flux column name recognised; used '{fcol}'")
    if tcol is None or (fcol or mcol) is None:
        raise IngestError(f"{p.name}: could not identify time and flux columns (columns: {', '.join(map(str, df.columns[:12]))})")
    ycol = fcol or mcol
    ecol = _pick(cols, [f"{_norm(ycol)}_err"] + (ERR_NAMES if fcol else ["mag_err", "magerr", "e_mag", "err", "error"]))
    if ecol is None and not targcol:
        rest = [c for c in numeric if c not in (tcol, ycol)]
        if len(rest) == 1 and str(rest[0]).startswith("col"):
            ecol = rest[0]
    t_all, tsys = _to_days(df[tcol], str(tcol))
    y_all = pd.to_numeric(df[ycol], errors="coerce").to_numpy(dtype=np.float64)
    e_all = pd.to_numeric(df[ecol], errors="coerce").to_numpy(dtype=np.float64) if ecol else None
    qcol = _pick(cols, QUALITY_NAMES)
    if qcol and pd.api.types.is_numeric_dtype(df[qcol]):
        good = df[qcol].fillna(0).to_numpy() == 0
        if good.mean() > 0.2:
            t_all, y_all = t_all[good], y_all[good]
            e_all = e_all[good] if e_all is not None else None
            df = df[good]
    mag = mcol is not None
    groups = [(str(k), (df[targcol] == k).to_numpy()) for k in df[targcol].dropna().unique()] if targcol else \
        [(p.stem, np.ones(len(t_all), bool))]
    out = []
    for name, sel in groups[:500]:
        y, e = y_all[sel], (e_all[sel] if e_all is not None else None)
        if mag:
            y, e = _mag_to_flux(y, e)
        out.append(Segment(name if targcol else p.stem, t_all[sel], y, e, p.name,
                           {"time": str(tcol), "flux": str(ycol), "error": str(ecol) if ecol else None,
                            "target": str(targcol) if targcol else None, "magnitudes": mag},
                           None, tsys, {}, list(warnings)))
    return out


# ------------------------------------------------------------------ cleaning + stitching
def clean(seg: Segment) -> Segment | None:
    t, y, e = seg.time, seg.flux, seg.err
    if t.ndim != 1 or y.ndim != 1 or len(t) != len(y):
        raise IngestError(f"{seg.source}: time and flux must be 1-D columns of equal length")
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    e = e[ok] if e is not None else None
    if len(t) < MIN_POINTS:
        seg.warnings.append(f"{seg.source}: only {len(t)} valid points - skipped")
        return None
    order = np.argsort(t, kind="stable")
    t, y = t[order], y[order]
    e = e[order] if e is not None else None
    keep = np.r_[True, np.diff(t) > 0]
    t, y = t[keep], y[keep]
    e = e[keep] if e is not None else None
    med = np.nanmedian(y)
    if not np.isfinite(med) or med == 0:
        seg.warnings.append(f"{seg.source}: flux median is zero - skipped")
        return None
    if med < 0:
        y = y - 2 * med  # differential flux around zero: shift positive before normalising
        med = np.nanmedian(y)
    y = y / med
    e = e / abs(med) if e is not None else None
    # upward outliers only (flares, cosmic rays); dips are kept
    mad = np.median(np.abs(y - 1)) * 1.4826
    if mad > 0:
        keep = y < 1 + 6 * mad
        t, y = t[keep], y[keep]
        e = e[keep] if e is not None else None
    seg.time, seg.flux, seg.err = t, y, e
    return seg


def _key(name: str) -> str:
    digits = re.findall(r"\d{5,}", name)
    if digits:
        return digits[0].lstrip("0")
    return re.sub(r"[^a-z0-9]", "", name.lower())


@dataclass
class Target:
    id: str
    name: str
    segments: list[Segment]

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        t = np.concatenate([s.time for s in self.segments])
        y = np.concatenate([s.flux for s in self.segments])
        has_err = all(s.err is not None for s in self.segments)
        e = np.concatenate([s.err for s in self.segments]) if has_err else None  # type: ignore[misc]
        o = np.argsort(t, kind="stable")
        return t[o], y[o], (e[o] if e is not None else None)


def group(segments: list[Segment], combine: bool = False) -> list[Target]:
    if combine and segments:
        name = segments[0].target
        return [Target("t0", name, segments)]
    by: dict[str, list[Segment]] = {}
    names: dict[str, str] = {}
    for s in segments:
        k = _key(s.target)
        by.setdefault(k, []).append(s)
        names.setdefault(k, s.target)
    return [Target(f"t{i}", names[k], segs) for i, (k, segs) in enumerate(by.items())]


def parse_files(files: list[Path], combine: bool = False) -> tuple[list[Target], list[str], list[str]]:
    segments, warnings, skipped = [], [], []
    for f in files:
        k = kind_of(f)
        if k is None:
            skipped.append(f"{f.name}: unsupported file type")
            continue
        try:
            segs = read_fits(f) if k == "fits" else read_table(f)
        except IngestError as e:
            skipped.append(str(e))
            continue
        except Exception as e:  # noqa: BLE001
            skipped.append(f"{f.name}: could not be read - the file may be truncated or corrupted ({e})")
            continue
        for s in segs:
            try:
                c = clean(s)
            except Exception as e:  # noqa: BLE001
                skipped.append(f"{f.name}: could not clean the data ({e})")
                continue
            warnings.extend(s.warnings)
            if c is not None:
                segments.append(c)
    if not segments:
        detail = "; ".join(skipped[:5]) or "no light-curve files found"
        raise IngestError(f"No usable light curve in the upload. {detail}")
    return group(segments, combine), warnings, skipped
