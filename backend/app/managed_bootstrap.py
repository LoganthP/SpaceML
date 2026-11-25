"""Standalone Python launcher inside Vercel Sandbox (stdlib until uv installs dependencies).

One Drive, one supervisor, one API/worker. No model/data deletion on restart.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tarfile
import time
import urllib.request


def main():
    import fcntl
    root = Path("/smc")
    root.mkdir(exist_ok=True)
    lock = open(root / ".runtime.lock", "a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        # Another cold Function can ask for startup concurrently. Never run a
        # second SQLite writer or replace a process with active work.
        try:
            state = json.loads((root / "runtime.json").read_text())
            if state.get("revision") != os.environ["SMC_BUNDLE_SHA256"]:
                request = urllib.request.Request("http://127.0.0.1:8000/api/jobs?limit=200", headers={"x-smc-internal-token": os.environ["SMC_INTERNAL_TOKEN"]})
                with urllib.request.urlopen(request, timeout=3) as response:
                    busy = any(j["status"] in ("queued", "running") for j in json.load(response))
                request = urllib.request.Request("http://127.0.0.1:8000/api/runs?limit=500", headers={"x-smc-internal-token": os.environ["SMC_INTERNAL_TOKEN"]})
                with urllib.request.urlopen(request, timeout=3) as response:
                    busy = busy or any(r["status"] in ("queued", "running") for r in json.load(response))
                if not busy:
                    os.kill(state["supervisor_pid"], signal.SIGTERM)
        except (OSError, ValueError, KeyError):
            pass
        return
    revision = os.environ["SMC_BUNDLE_SHA256"]
    if len(revision) != 64 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Invalid source checksum")
    source = root / "releases" / revision
    status_file = root / "runtime.json"
    def status(state, detail=None):
        temp = status_file.with_suffix(".tmp")
        temp.write_text(json.dumps({"state": state, "detail": detail, "revision": revision, "supervisor_pid": os.getpid()}))
        temp.replace(status_file)
    process = None
    def stop(*_):
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=25)
            except subprocess.TimeoutExpired:
                process.kill()
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop)
    try:
        status("installing", "Preparing the Vercel training environment")
        if not (source / ".complete").is_file():
            source.mkdir(parents=True, exist_ok=True)
            archive = root / (revision + ".tar.gz")
            headers = {"x-smc-internal-token": os.environ["SMC_INTERNAL_TOKEN"]}
            if os.environ.get("SMC_DEPLOYMENT_BYPASS"):
                headers["x-vercel-protection-bypass"] = os.environ["SMC_DEPLOYMENT_BYPASS"]
            with urllib.request.urlopen(urllib.request.Request(os.environ["SMC_SOURCE_URL"], headers=headers), timeout=120) as response, archive.open("wb") as output:
                digest = hashlib.sha256()
                while block := response.read(1 << 20):
                    digest.update(block)
                    output.write(block)
            if digest.hexdigest() != revision:
                raise ValueError("Training source checksum failed")
            with tarfile.open(archive, "r:gz") as package:
                for member in package.getmembers():
                    if not (source / member.name).resolve().is_relative_to(source.resolve()) or member.issym() or member.islnk():
                        raise ValueError("Unsafe training source archive")
                package.extractall(source, filter="data")
            (source / ".complete").touch()
        uv = shutil.which("uv")
        if not uv:
            raise RuntimeError("Use the Vercel managed Python image, which includes uv")
        env = {**os.environ, "UV_PYTHON_INSTALL_DIR": str(root / "python"), "UV_CACHE_DIR": str(root / "cache/uv")}
        venv = root / "venv"
        python = venv / "bin/python"
        if not python.is_file():
            subprocess.run([uv, "python", "install", "3.12"], env=env, check=True)
            subprocess.run([uv, "venv", "--python", "3.12", str(venv)], env=env, check=True)
        requirements = source / "backend/requirements-train.txt"
        dependency_hash = hashlib.sha256(b"".join((source / "backend" / name).read_bytes() for name in ("requirements-train.txt", "requirements-core.txt", "requirements-ml.txt"))).hexdigest()
        installed = root / ".dependencies"
        if not installed.is_file() or installed.read_text() != dependency_hash:
            subprocess.run([uv, "pip", "install", "--python", str(python), "torch==2.6.0", "torchvision==0.21.0", "--index-url", "https://download.pytorch.org/whl/cpu"], env=env, check=True)
            subprocess.run([uv, "pip", "install", "--python", str(python), "-r", str(requirements)], env=env, check=True)
            installed.write_text(dependency_hash)
        env.update(PYTHONPATH=str(source / "backend"), SMC_MODE="vercel", SMC_EXECUTION="worker", SMC_AUTH_REQUIRED="false",
                   SPACEML_DATA_DIR=str(root / "data"), SPACEML_MODELS_DIR=str(root / "models"),
                   SPACEML_STORAGE_DIR=str(root / "storage"), SPACEML_STARTER_DIR=str(source / "starter"),
                   SPACEML_BUNDLE_DIR=str(source / "release/bundle"), SMC_AUTO_INSTALL="off", SPACEML_AUTOSETUP="off",
                   SPACEML_BOOTSTRAP="0", HF_HOME=str(root / "cache/huggingface"), TORCH_HOME=str(root / "cache/torch"))
        process = subprocess.Popen([str(python), "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"], cwd=source / "backend", env=env)
        status("starting", "Installing bundled data and loading the API")
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"Training API stopped with status {return_code}")
        status("stopped")
    except Exception as error:
        status("error", str(error)[:300])
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
