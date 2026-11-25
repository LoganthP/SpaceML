import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def test_runtime_source_preserves_all_original_dataset_bytes_and_excludes_secrets():
    subprocess.run([sys.executable, "scripts/build_vercel_runtime.py"], cwd=ROOT, check=True, capture_output=True)
    metadata = json.loads((ROOT / "backend/app/runtime_bundle.json").read_text())
    path = ROOT / "frontend/public" / metadata["url"].lstrip("/")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["sha256"]
    datasets = json.loads((ROOT / "DATASETS-MANIFEST.json").read_text())
    expected = {record["path"]: record["sha256"] for record in datasets["starter_pack"]["files"]}
    for dataset in datasets["release_bundle"]["datasets"]:
        expected.update({f"release/bundle/datasets/{dataset['id']}/{record['path']}": record["sha256"] for record in dataset["files"]})
    expected["starter/pack.json"] = hashlib.sha256((ROOT / "starter/pack.json").read_bytes()).hexdigest()
    verified = set()
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            parts = Path(member.name).parts
            assert not member.issym() and not member.islnk()
            assert not any(p.startswith(".env") or p in ("secrets", "__pycache__", "tests", "static_data", "node_modules") for p in parts)
            if member.name in expected:
                assert hashlib.sha256(archive.extractfile(member).read()).hexdigest() == expected[member.name]
                verified.add(member.name)
    assert verified == set(expected)


def test_private_runtime_archive_middleware():
    script = '''
import assert from "node:assert/strict";
import guard, {config} from "./proxy.js";
assert.equal(config.matcher, undefined); // matcher belongs only in vercel.json
process.env.SMC_INTERNAL_TOKEN = "s".repeat(48);
const url = "https://app.vercel.app/runtime/test.tar.gz";
assert.equal(guard(new Request(url)).status, 403);
assert.equal(guard(new Request(url, {headers:{"x-smc-internal-token":"wrong"}})).status, 403);
assert.equal(guard(new Request(url, {headers:{"x-smc-internal-token":"s".repeat(48)}})).headers.get("x-middleware-next"), "1");
process.env.SMC_INTERNAL_TOKEN = "short";
assert.equal(guard(new Request(url, {headers:{"x-smc-internal-token":"short"}})).status, 403);
'''
    result = subprocess.run(["node", "--input-type=module", "-"], input=script, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
