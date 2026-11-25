import hashlib
import json

import pytest

from tests.test_release_and_jobs import _fresh


def test_published_install_preserves_old_data_and_is_idempotent(tmp_path, monkeypatch):
    _fresh(monkeypatch, tmp_path, "published")
    from spaceml.missions import store, published, models
    source = tmp_path / "publication"
    source.mkdir()
    contents = {"data/webb/manifest.json": {"fingerprint": "corrected"},
                "data/webb/items.json": [],
                "models/webb/validated/manifest.json": {"dataset_fingerprint": "corrected"}}
    records = []
    for name, body in contents.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(body).encode()
        path.write_bytes(data)
        records.append({"path": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    (source / "manifest.json").write_text(json.dumps({"models": {"webb": "validated"}, "files": records}))
    previous = store.mdir("webb")
    previous.mkdir(parents=True)
    (previous / "manifest.json").write_text(json.dumps({"fingerprint": "old"}))
    (previous / "user-data.txt").write_text("keep this")
    published.install(source)
    assert store.manifest("webb")["fingerprint"] == "corrected"
    backups = list((store.VERS / "webb").glob("*/user-data.txt"))
    assert len(backups) == 1 and backups[0].read_text() == "keep this"
    published.install(source)
    assert len(list((store.VERS / "webb").iterdir())) == 1
    model = models.ROOT / "webb/validated/manifest.json"
    assert model.is_file()
    (source / "data/webb/items.json").write_text("corrupted")
    with pytest.raises(ValueError, match="verification"):
        published.install(source)
    assert model.is_file() and backups[0].read_text() == "keep this"
