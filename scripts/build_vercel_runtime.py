"""Build a deterministic, secret-free training source archive for Vercel Sandbox.

No downloads or training happen during this build. Immutable starter bytes stay intact.
"""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def files():
    for directory in ("backend", "starter", "release/bundle", "release/mission-results", "samples"):
        for path in sorted((ROOT / directory).rglob("*"), key=lambda p: p.relative_to(ROOT).as_posix()):
            relative = path.relative_to(ROOT)
            if not path.is_file() or path.is_symlink():
                continue
            if any(part in ("__pycache__", ".pytest_cache", "tests", "static_data", "node_modules", ".venv", "secrets") for part in relative.parts):
                continue
            if path.name.startswith(".env") or path.name == "runtime_bundle.json" or path.suffix in (".pyc", ".log", ".tmp"):
                continue
            yield path


def main():
    target = ROOT / "frontend/public/runtime"
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target, suffix=".tmp", delete=False) as temporary:
        path = Path(temporary.name)
        with gzip.GzipFile(fileobj=temporary, mode="wb", mtime=0, filename="") as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
                for source in files():
                    content = source.read_bytes()
                    if source.suffix == ".py" and source.relative_to(ROOT).parts[0] == "backend":
                        content = content.replace(b"\r\n", b"\n")
                    info = tarfile.TarInfo(source.relative_to(ROOT).as_posix())
                    info.size, info.mode, info.mtime = len(content), 0o644, 0
                    archive.addfile(info, io.BytesIO(content))
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    destination = target / (checksum + ".tar.gz")
    path.replace(destination)
    manifest = {"sha256": checksum, "url": "/runtime/" + destination.name, "bytes": destination.stat().st_size}
    output = ROOT / "backend/app/runtime_bundle.json"
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".tmp", mode="w", encoding="utf-8", delete=False) as temporary:
        json.dump(manifest, temporary)
        temporary_path = Path(temporary.name)
    temporary_path.replace(output)
    print(f"Built Vercel training source: {manifest['bytes'] / 2**20:.1f} MiB; SHA-256 {checksum}")


if __name__ == "__main__":
    main()
