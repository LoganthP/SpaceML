import json, sys, urllib.request
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
B = "http://localhost:8081"
g = lambda p: json.load(op.open(B + p))
def state():
    jobs = g("/api/jobs?limit=500"); jobs = jobs if isinstance(jobs, list) else jobs.get("jobs", [])
    return {"datasets": {m["id"]: m["dataset"]["installed"] for m in g("/api/missions")["missions"]},
            "models": {m["id"]: m["model"]["version"] for m in g("/api/missions")["missions"]},
            "opssat_versions": [v["version"] for v in g("/api/missions/opssat")["models"]],
            "jobs": sorted((j["id"], j["status"]) for j in jobs),
            "reviews": sorted((r["run_id"], r["item"], r["verdict"]) for r in g("/api/reviews")["reviews"]),
            "runs": sorted((r["id"], r["status"]) for r in g("/api/runs?module=missions&limit=500"))}
if sys.argv[1] == "save":
    json.dump(state(), open("/tmp/persist_before.json", "w"))
    s = json.load(open("/tmp/persist_before.json")); print({k: len(v) for k, v in s.items()})
else:
    a, b = json.load(open("/tmp/persist_before.json")), json.loads(json.dumps(state()))
    print({k: ("same" if a[k] == b[k] else f"DIFF {a[k]} -> {b[k]}") for k in a})
    new_jobs = [j for j in b["jobs"] if j not in a["jobs"] and j[0] not in {x[0] for x in a["jobs"]}]
    print("jobs created by restart:", new_jobs)
