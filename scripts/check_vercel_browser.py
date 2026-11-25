"""Browser verification against `vercel dev --local --listen 3100`."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BASE = os.environ.get("SMC_BROWSER_BASE_URL", "http://localhost:3100").rstrip("/")
PAGES = ["/", "/telemetry", "/telemetry?mission=opssat", "/telemetry?mission=hubble", "/telemetry?mission=webb",
         "/exoplanet", "/mars", "/mars/traverse", "/replay", "/workbench", "/space-weather", "/review", "/runs", "/evaluation", "/datasets"]


def main():
    report = {"base_url": BASE, "pages": [], "console_errors": [], "page_errors": [], "http_errors": [], "click_errors": [], "overflow": []}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--enable-unsafe-swiftshader"])
        context = browser.new_context(viewport={"width": 1440, "height": 1000}, accept_downloads=True)
        page = context.new_page()
        page.on("console", lambda msg: report["console_errors"].append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda error: report["page_errors"].append(str(error)))
        page.on("response", lambda response: report["http_errors"].append({"url": response.url, "status": response.status}) if response.status >= 400 else None)
        for path in PAGES:
            page.goto(BASE + path, wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(400)
            page.reload(wait_until="networkidle", timeout=60000)
            page.wait_for_selector("#root button", timeout=20000)
            page.locator("details").evaluate_all("elements => elements.forEach(element => { element.open = true; })")
            page.wait_for_timeout(200)
            for width in (1440, 390):
                page.set_viewport_size({"width": width, "height": 1000})
                page.wait_for_timeout(150)
                dimensions = page.evaluate("({viewport: innerWidth, width: document.documentElement.scrollWidth})")
                if dimensions["width"] > dimensions["viewport"] + 1:
                    report["overflow"].append({"page": path, **dimensions})
            page.set_viewport_size({"width": 1440, "height": 1000})
            # Exercise every currently rendered visible button. Disabled write
            # buttons are verified as disabled, not clicked via forced events.
            buttons = page.locator("button").element_handles()
            # Do not let a top-level navigation click detach the page controls
            # before they have been exercised. Mission tabs run after actions.
            buttons.sort(key=lambda button: button.evaluate("element => element.closest('main') ? (element.closest('nav') ? 1 : 0) : 2"))
            count = 0
            for button in buttons:
                try:
                    if not button.is_visible() or not button.is_enabled():
                        continue
                    label = (button.inner_text() or button.get_attribute("aria-label") or button.get_attribute("title") or "icon")[:100]
                    try:
                        button.click(timeout=1200)
                    except Exception as exc:
                        # The mission map deliberately moves its HUD controls.
                        # Retry only a visible, enabled animated button; never
                        # bypass the disabled state of a writable action.
                        if "stable" not in str(exc) or not button.is_visible() or not button.is_enabled():
                            raise
                        button.click(force=True, timeout=1200)
                    count += 1
                    page.wait_for_timeout(70)
                    page.keyboard.press("Escape")
                except Exception as exc:
                    # Navigation/tabs may legitimately replace old nodes.
                    if "not attached" not in str(exc) and "resolved to" not in str(exc):
                        report["click_errors"].append({"page": path, "label": locals().get("label"), "error": str(exc)[:180]})
            report["pages"].append({"path": path, "visible_buttons_clicked": count})
            print(f"Checked {path}: {count} buttons", flush=True)
        context.close()
        browser.close()
    output = ROOT / "vercel-validation.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    failures = sum(len(report[k]) for k in ("console_errors", "page_errors", "http_errors", "overflow"))
    print(f"Browser checks: {len(report['pages'])} pages; {failures} errors/overflows; report {output.name}")
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
