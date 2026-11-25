"""Check the worker's actual storage paths; filesystem capacity is not Drive quota."""
import os
from pathlib import Path
import shutil
import tempfile


def probe(mount: Path, paths: dict[str, Path]) -> dict:
    mount = mount.resolve()
    result = {"mount_path": str(mount), "paths_valid": False, "writable": False,
              "capacity_source": "unavailable", "free_bytes": None}
    if not all(path.resolve().is_relative_to(mount) for path in paths.values()):
        return {**result, "error": "A dataset, model or job path is outside the Drive mount"}
    result["paths_valid"] = True
    try:
        for path in paths.values():
            # Do not create missing mounts or silently fall back to /tmp.
            if not path.is_dir():
                raise FileNotFoundError("A required Drive directory is missing")
            with tempfile.TemporaryFile(dir=path) as file:
                file.write(b"space-ml-drive-probe")
                file.flush()
                os.fsync(file.fileno())
                file.seek(0)
                if file.read() != b"space-ml-drive-probe":
                    raise OSError("Drive read-back verification failed")
        result["writable"] = True
    except OSError as error:
        result["error"] = f"Drive write/read check failed ({type(error).__name__}, errno={error.errno})"
        return result
    try:
        disk = shutil.disk_usage(mount)
        result["filesystem_estimate"] = {"free_bytes": disk.free, "total_bytes": disk.total,
                                         "source": "filesystem_statvfs_not_drive_quota"}
    except OSError:
        # Missing capacity metadata does not invalidate a successful write/read.
        pass
    return result


def worker_status():
    from spaceml.config import DATA_DIR, MODELS_DIR, STORAGE_DIR
    return probe(Path("/smc"), {"datasets": DATA_DIR, "models": MODELS_DIR, "jobs": STORAGE_DIR})
