import asyncio, json, urllib.request
from playwright.async_api import async_playwright
BASE = "http://localhost:8081"
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
get = lambda p: json.load(op.open(BASE + p))
async def main():
    log, nofocus = [], []
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
        pg = await b.new_page(viewport={"width": 1280, "height": 900})
        await pg.goto(BASE + "/telemetry"); await pg.wait_for_timeout(2500)
        async def active():
            return await pg.evaluate("""() => { const e = document.activeElement; const s = getComputedStyle(e);
              return {tag: e.tagName, role: e.getAttribute('role'), name: (e.getAttribute('aria-label') || e.innerText || e.placeholder || '').trim().slice(0,60),
                      outline: s.outlineStyle !== 'none' && parseFloat(s.outlineWidth) > 0 }; }""")
        async def tab_to(pred, limit=120, back=False):
            for _ in range(limit):
                await pg.keyboard.press("Shift+Tab" if back else "Tab")
                a = await active()
                if not a["outline"] and a["tag"] not in ("BODY",):
                    nofocus.append(a)
                if pred(a):
                    return a
            raise RuntimeError("could not reach target by Tab")
        a = await tab_to(lambda a: a["role"] == "radio" and "SMAP" in a["name"]); log.append(("mission radio focused", a["name"]))
        await pg.keyboard.press("ArrowRight"); await pg.wait_for_timeout(1500)
        a = await active(); log.append(("ArrowRight selects", a["name"], pg.url))
        a = await tab_to(lambda a: a["tag"] == "INPUT"); log.append(("search field", a["name"]))
        a = await tab_to(lambda a: a["role"] == "listbox"); log.append(("listbox focused", a["name"][:40]))
        await pg.keyboard.press("ArrowDown"); await pg.keyboard.press("ArrowDown"); await pg.wait_for_timeout(2500)
        cap = await pg.locator("figcaption").first.inner_text(); log.append(("chart after ArrowDown", cap[:90]))
        a = await tab_to(lambda a: a["tag"] == "BUTTON" and a["name"].startswith("Quick train"), back=True)
        before = {j["id"] for j in get("/api/jobs?limit=50")}
        await pg.keyboard.press("Enter"); await pg.wait_for_timeout(1500)
        new = [j for j in get("/api/jobs?limit=50") if j["id"] not in before]
        log.append(("Enter on Quick train ->", new[0]["task"] if new else None))
        live = await pg.locator("[role=status][aria-live=polite]").count()
        log.append(("live regions present", live))
        for _ in range(60):
            if new and get(f"/api/jobs/{new[0]['id']}")["status"] not in ("queued", "running"): break
            await pg.wait_for_timeout(1000)
        log.append(("job", get(f"/api/jobs/{new[0]['id']}")["status"]))
        await b.close()
    print(json.dumps({"steps": log, "focused_without_visible_outline": nofocus[:10]}, indent=1))
asyncio.run(main())
