"""UI regression check with simulated health states; this is NOT live training.

Run against `vercel dev --local --listen 3100`. Page data uses the actual local
gateway; only /api/health is replaced to exercise readiness/auth transitions.
"""
import json
import os
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BASE = os.environ.get("SMC_BROWSER_BASE_URL", "http://localhost:3100")


def main():
    state = {"value": "setup_required", "allowed": False}
    def health(route):
        ready = state["value"] == "ready"
        route.fulfill(json={"mode": "vercel", "execution": "managed", "state": state["value"],
            "capabilities": {"write": state["allowed"], "train": state["allowed"], "live": True},
            "checks": {"sandbox": {"available": ready}, "drive": {"name": "test-drive", "mounted": ready, "writable": ready}},
            "authentication": {"required": True, "authenticated": state["allowed"]},
            "hint": "Sign in to train" if ready and not state["allowed"] else "Connect Vercel Sandbox and Drive"})
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.route("**/api/health", health)
        page.goto(BASE + "/telemetry", wait_until="networkidle")
        run = page.get_by_role("button", name="Run anomaly detection", exact=True)
        expect(run).to_be_disabled()
        state["value"] = "ready"
        page.get_by_role("button", name="Check connection", exact=True).click()
        expect(page.get_by_text("Vercel compute ready", exact=True)).to_be_visible()
        expect(run).to_be_disabled()
        expect(page.get_by_role("link", name="Sign in", exact=True)).to_be_visible()
        state["allowed"] = True
        page.get_by_role("button", name="Check connection", exact=True).click()
        expect(run).to_be_enabled()
        page.goto(BASE + "/telemetry?mission=opssat", wait_until="networkidle")
        predict = page.get_by_role("button", name="Predict this segment", exact=True)
        expect(predict).to_be_enabled()
        state.update(value="error", allowed=False)
        page.get_by_role("button", name="Check connection", exact=True).click()
        expect(predict).to_be_disabled()
        expect(page.get_by_text("Prediction needs a trained model", exact=False)).to_have_count(0)
        # Recover without reloading or manually retrying: health errors must
        # continue polling instead of permanently freezing the capabilities.
        state.update(value="ready", allowed=True)
        expect(predict).to_be_enabled(timeout=22000)
        browser.close()
    report = {"simulated_health": True, "live_training_verified": False,
              "setup_blocks_inference": True, "anonymous_ready_requires_sign_in": True,
              "authenticated_ready_enables_inference": True, "startup_error_recovers_without_reload": True,
              "existing_model_not_misreported_as_missing": True}
    (ROOT / "readiness-ui-validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
