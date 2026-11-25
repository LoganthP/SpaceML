import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("storage_probe_test", ROOT / "backend/app/storage_health.py")
storage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(storage)


def test_probe_checks_actual_read_write_and_never_claims_quota(tmp_path):
    paths = {name: tmp_path / name for name in ("data", "models", "jobs")}
    for path in paths.values():
        path.mkdir()
        (path / "keep.txt").write_text("keep")
    result = storage.probe(tmp_path, paths)
    assert result["paths_valid"] and result["writable"]
    assert result["free_bytes"] is None and result["capacity_source"] == "unavailable"
    assert result["filesystem_estimate"]["source"] == "filesystem_statvfs_not_drive_quota"
    assert all([p.name for p in path.iterdir()] == ["keep.txt"] for path in paths.values())


def test_probe_rejects_paths_outside_mount_and_missing_directories(tmp_path):
    result = storage.probe(tmp_path / "mount", {"data": tmp_path / "outside"})
    assert not result["paths_valid"] and not result["writable"]
    assert not (tmp_path / "outside").exists()
    result = storage.probe(tmp_path, {"data": tmp_path / "missing"})
    assert result["paths_valid"] and not result["writable"]
    assert not (tmp_path / "missing").exists()


def test_read_only_mount_cannot_pass_write_probe(tmp_path, monkeypatch):
    def denied(**kwargs):
        raise PermissionError(13, "read-only mount")
    monkeypatch.setattr(storage.tempfile, "TemporaryFile", denied)
    result = storage.probe(tmp_path, {"data": tmp_path})
    assert result["paths_valid"] and not result["writable"]
    assert "PermissionError" in result["error"]
