import asyncio, json, time, urllib.request
from playwright.async_api import async_playwright, expect
BASE = "http://localhost:8083"
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
get = lambda p: json.load(op.open(BASE + p))
MIS = [("hubble", "Hubble Space Telescope", "IEPP0101Q", True), ("webb", "James Webb Space Telescope", "jw01234003001", True)]
async def main():
    res = {}
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
        pg = await b.new_page(viewport={"width": 1440, "height": 1000})
        errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
        for mid, name, item, trainable in MIS:
            r = {}
            await pg.goto(BASE + "/telemetry"); await pg.get_by_role("radio", name=name).click()
            lb = pg.get_by_role("listbox"); await expect(lb).to_be_visible(timeout=15000)
            opt = lb.get_by_role("option", name=item) if item else lb.get_by_role("option").first
            t = time.time(); await opt.first.click()
            await expect(pg.locator("figcaption").first).to_contain_text("samples", timeout=15000)
            r["replay_s"] = round(time.time() - t, 2); r["caption"] = (await pg.locator("figcaption").first.inner_text())[:170]
            if trainable:
                card = pg.locator("div.glass-inset", has_text=name).first
                before = {j["id"] for j in get("/api/jobs?limit=50")}
                t = time.time(); await card.get_by_role("button", name="Quick train").click()
                job = None
                for _ in range(300):
                    js = [j for j in get("/api/jobs?limit=50") if j["id"] not in before]
                    if js and js[0]["status"] not in ("queued", "running"):
                        job = js[0]; break
                    await pg.wait_for_timeout(1000)
                r["quick_train"] = {"job": job["id"], "status": job["status"], "s": round(time.time() - t, 1), "error": job.get("error")}
                await pg.wait_for_timeout(2500); await pg.reload(); await pg.wait_for_timeout(2000)
                await (lb.get_by_role("option", name=item) if item else lb.get_by_role("option")).first.click(); await pg.wait_for_timeout(1200)
            t = time.time()
            await pg.get_by_role("button", name="Score this").click()
            await expect(pg.get_by_text("detector:")).to_be_visible(timeout=60000)
            r["score_s"] = round(time.time() - t, 2)
            r["score"] = (await pg.locator("text=detector:").locator("..").inner_text())[:200].replace("\n", " ")
            if trainable:
                await pg.get_by_role("button", name="Evaluate").click()
                await expect(pg.get_by_text("were sent to Review Studio")).to_be_visible(timeout=60000)
                r["evaluate"] = (await pg.get_by_text("were sent to Review Studio").inner_text())[:120]
            await pg.screenshot(path=f"/tmp/shots10/fixture_{mid}.png", full_page=True)
            res[mid] = r
        await pg.goto(BASE + "/review?module=missions"); await pg.wait_for_timeout(3000)
        res["review_queue_pending"] = get("/api/reviews/queue?module=missions&limit=1")["pending"]
        await pg.get_by_role("button", name="Inspect").first.click()
        await expect(pg.locator("figcaption").first).to_be_visible(timeout=15000)
        res["review_inspect"] = (await pg.locator("figcaption").first.inner_text())[:160]
        res["page_errors"] = errs
        await b.close()
    print(json.dumps(res, indent=1))
asyncio.run(main())
