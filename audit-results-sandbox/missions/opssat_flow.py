import asyncio, json, time, urllib.request
from playwright.async_api import async_playwright, expect
BASE = "http://localhost:8081"
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
get = lambda p: json.load(op.open(BASE + p))
async def main():
    out = {}
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
        pg = await b.new_page(viewport={"width": 1440, "height": 1000})
        errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
        t = time.time()
        await pg.goto(BASE + "/telemetry"); await pg.get_by_role("radio", name="ESA OPS-SAT").click()
        await expect(pg.get_by_role("listbox")).to_be_visible(timeout=15000)
        out["open_workspace_s"] = round(time.time() - t, 2)
        await pg.get_by_role("radiogroup", name="Split").get_by_role("radio", name="Held-out").click()
        await pg.get_by_role("radiogroup", name="Published label").get_by_role("radio", name="Anomaly").click()
        await pg.wait_for_timeout(800)
        t = time.time()
        await pg.get_by_role("listbox").get_by_role("option").first.click()
        await expect(pg.locator("figcaption").first).to_contain_text("samples", timeout=15000)
        out["first_replay_s"] = round(time.time() - t, 2)
        out["caption"] = (await pg.locator("figcaption").first.inner_text())[:200]
        await pg.screenshot(path="/tmp/shots10/opssat_replay.png", full_page=True)
        # Quick train from the card
        card = pg.locator("div.glass-inset", has_text="ESA OPS-SAT").first
        before = {j["id"] for j in get("/api/jobs?limit=50")}
        t = time.time()
        await card.get_by_role("button", name="Quick train").click()
        job = None
        for _ in range(200):
            js = [j for j in get("/api/jobs?limit=50") if j["id"] not in before]
            if js and js[0]["status"] not in ("queued", "running"):
                job = js[0]; break
            await pg.wait_for_timeout(1000)
        out["quick_train"] = {"job_id": job["id"], "status": job["status"], "seconds": round(time.time() - t, 1), "error": job.get("error")}
        await pg.wait_for_timeout(3000); await pg.reload(); await pg.wait_for_timeout(2500)
        await pg.get_by_role("listbox").get_by_role("option").first.click(); await pg.wait_for_timeout(1500)
        await pg.get_by_role("button", name="Predict this segment").click()
        await expect(pg.get_by_text("Prediction:")).to_be_visible(timeout=30000)
        out["predict"] = (await pg.get_by_text("Prediction:").inner_text())[:200]
        t = time.time()
        await pg.get_by_role("button", name="Evaluate").click()
        await expect(pg.get_by_text("were sent to Review Studio")).to_be_visible(timeout=60000)
        out["evaluate_s"] = round(time.time() - t, 2)
        out["eval_text"] = (await pg.locator("text=False alarms").first.inner_text())
        await pg.screenshot(path="/tmp/shots10/opssat_eval.png", full_page=True)
        await pg.get_by_role("link", name="Review them").click()
        await expect(pg.get_by_text("Waiting for review")).to_be_visible(timeout=15000)
        await pg.wait_for_timeout(2000)
        await pg.get_by_role("button", name="Inspect").first.click()
        await expect(pg.locator("figcaption").first).to_contain_text("published label", timeout=15000)
        out["review_inspect"] = (await pg.locator("figcaption").first.inner_text())[:220]
        await pg.screenshot(path="/tmp/shots10/review_inspect.png", full_page=False)
        n0 = get("/api/reviews/queue?module=missions&limit=1")["pending"]
        await pg.get_by_role("button", name="Confirm:").first.click(); await pg.wait_for_timeout(1500)
        n1 = get("/api/reviews/queue?module=missions&limit=1")["pending"]
        out["verdict"] = {"pending_before": n0, "pending_after": n1}
        out["page_errors"] = errs
        await b.close()
    print(json.dumps(out, indent=1))
asyncio.run(main())
