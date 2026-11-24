import asyncio, json, sys
from playwright.async_api import async_playwright
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8081"
AXE = open("/tmp/axe/node_modules/axe-core/axe.min.js").read()
PAGES = [("/telemetry?mission=opssat", True), ("/review?module=missions", False), ("/datasets", False), ("/telemetry?mission=hubble", False), ("/telemetry", False)]
async def main():
    out = {"axe": {}, "overflow": {}}
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
        for theme in ("light", "dark"):
            ctx = await b.new_context(viewport={"width": 1280, "height": 900}, reduced_motion="reduce")
            await ctx.add_init_script(f"localStorage.setItem('smc-theme','{theme}')")
            pg = await ctx.new_page()
            for path, pick in PAGES:
                await pg.goto(BASE + path); await pg.wait_for_timeout(2500)
                if pick:
                    await pg.get_by_role("listbox").get_by_role("option").first.click(); await pg.wait_for_timeout(2500)
                await pg.add_script_tag(content=AXE)
                r = await pg.evaluate("""async () => { const r = await axe.run(document, {runOnly: {type: 'tag', values: ['wcag2a','wcag2aa','wcag21a','wcag21aa']}, resultTypes: ['violations']});
                    return r.violations.map(v => ({id: v.id, impact: v.impact, n: v.nodes.length, help: v.help, sample: v.nodes.slice(0,3).map(n => n.target.join(' ') + ' :: ' + (n.failureSummary||'').split('\\n').slice(0,2).join(' ')) })); }""")
                out["axe"][f"{theme} {path}"] = r
            await ctx.close()
        for w, h in ((375, 812), (768, 1024), (1280, 900)):
            ctx = await b.new_context(viewport={"width": w, "height": h})
            pg = await ctx.new_page()
            for path, pick in PAGES:
                await pg.goto(BASE + path); await pg.wait_for_timeout(2200)
                if pick:
                    await pg.get_by_role("listbox").get_by_role("option").first.click(); await pg.wait_for_timeout(2200)
                sw = await pg.evaluate("[document.documentElement.scrollWidth, window.innerWidth, [...document.querySelectorAll('body *')].filter(e => { const r = e.getBoundingClientRect(); return r.right > window.innerWidth + 1 && getComputedStyle(e).position !== 'fixed' && r.width > 0; }).slice(0,4).map(e => e.tagName + '.' + (e.className && e.className.baseVal === undefined ? String(e.className).split(' ').slice(0,3).join('.') : ''))]")
                out["overflow"][f"{w}px {path}"] = {"scrollWidth": sw[0], "innerWidth": sw[1], "page_scrolls_horizontally": sw[0] > sw[1], "wide": sw[2]}
                if path == "/telemetry?mission=opssat":
                    await pg.screenshot(path=f"/tmp/shots10/opssat_{w}.png", full_page=False)
            await ctx.close()
        await b.close()
    print(json.dumps(out, indent=1))
asyncio.run(main())
