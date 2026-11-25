"""Exercise real OPS-SAT UI training, reload, evaluation and process restart on isolated disk.

Requires optional training dependencies, Playwright Chromium and frontend/dist.
This checks the local persistent implementation, not a remote Docker host.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8001"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=10) as response:
        return json.load(response)


def wait_until(check, seconds=180):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            result = check()
            if result:
                return result
        except (OSError, ValueError):
            pass
        time.sleep(0.5)
    raise TimeoutError("Persistent workflow timed out")


def main():
    from playwright.sync_api import sync_playwright
    with tempfile.TemporaryDirectory(prefix="smc-workflow-") as temporary:
        base = Path(temporary)
        env = {**os.environ, "PYTHONPATH": str(ROOT / "backend"), "SMC_MODE": "local",
               "SMC_AUTO_INSTALL": "off", "SPACEML_AUTOSETUP": "off", "SPACEML_BOOTSTRAP": "0",
               "SPACEML_DATA_DIR": str(base / "data"), "SPACEML_MODELS_DIR": str(base / "models"),
               "SPACEML_STORAGE_DIR": str(base / "storage"), "SPACEML_STARTER_DIR": str(ROOT / "starter"),
               "SPACEML_TRAIN_CPUS": "2", "SMC_AUTH_REQUIRED": "true", "SMC_AUTH_USER": "verification",
               "SMC_AUTH_PASSWORD": "local-test-only", "SMC_PUBLIC_ORIGIN": BASE}
        subprocess.run([sys.executable, "-c", "from pathlib import Path; from spaceml import release; release.install(Path('release/bundle'))"],
                       cwd=ROOT, env=env, check=True, capture_output=True)
        log = open(base / "server.log", "w", encoding="utf-8")
        server = None
        def start():
            process = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8001"],
                                       cwd=ROOT, env=env, stdout=log, stderr=log,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            wait_until(lambda: get("/api/health").get("mode") == "local")
            return process
        try:
            server = start()
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True, args=["--enable-unsafe-swiftshader"])
                context = browser.new_context(http_credentials={"username": "verification", "password": "local-test-only"})
                page = context.new_page()
                page.goto(BASE + "/api/session", wait_until="networkidle")
                page.goto(BASE + "/datasets", wait_until="networkidle")
                card = page.get_by_role("link", name="ESA OPS-SAT", exact=True).locator("xpath=ancestor::div[contains(@class,'glass-inset')][1]")
                with page.expect_response(lambda response: response.url.endswith("/api/jobs") and response.request.method == "POST") as submitted:
                    card.get_by_role("button", name="Quick train", exact=False).first.click()
                job = submitted.value.json()
                assert submitted.value.status == 200, job
                page.reload(wait_until="networkidle")
                after_reload = get("/api/jobs/" + job["id"])
                assert after_reload["status"] in ("queued", "running", "succeeded"), after_reload
                finished = wait_until(lambda: (r if (r := get("/api/jobs/" + job["id"]))["status"] not in ("queued", "running") else None))
                assert finished["status"] == "succeeded", finished
                page.goto(BASE + "/telemetry?mission=opssat", wait_until="networkidle")
                with page.expect_response(lambda response: response.url.endswith("/api/runs") and response.request.method == "POST") as evaluated:
                    page.get_by_role("button", name="Evaluate ", exact=False).click()
                run = evaluated.value.json()
                assert evaluated.value.status == 200, run
                result = wait_until(lambda: (r if (r := get("/api/runs/" + run["id"]))["status"] not in ("queued", "running") else None))
                assert result["status"] == "succeeded", result
                model = get("/api/missions/opssat")["model"]
                weights = base / "models/missions/opssat" / model["version"] / "model.joblib"
                checksum = hashlib.sha256(weights.read_bytes()).hexdigest()
                server.terminate()
                server.wait(timeout=30)
                server = start()
                assert get("/api/jobs/" + job["id"])["status"] == "succeeded"
                assert get("/api/runs/" + run["id"])["status"] == "succeeded"
                assert get("/api/missions/opssat")["model"]["version"] == model["version"]
                assert hashlib.sha256(weights.read_bytes()).hexdigest() == checksum
                page.goto(BASE + "/telemetry?mission=opssat&run=" + run["id"], wait_until="networkidle")
                assert page.get_by_role("heading", name="Held-out evaluation").is_visible()
                report = {"environment": "isolated local process with persistent filesystem; Docker host not tested",
                          "job_id": job["id"], "status_after_reload": after_reload["status"], "run_id": run["id"],
                          "model_version": model["version"], "model_sha256": checksum,
                          "restart_preserved": ["model weights", "job history", "evaluation result"],
                          "passed": True}
                (ROOT / "persistent-workflow-validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(json.dumps(report, indent=2), flush=True)
                context.close()
                browser.close()
        except Exception:
            log.flush()
            print((base / "server.log").read_text(encoding="utf-8")[-6000:], file=sys.stderr)
            raise
        finally:
            if server and server.poll() is None:
                server.terminate()
                server.wait(timeout=30)
            log.close()


if __name__ == "__main__":
    main()
