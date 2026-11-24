import sys, json
from pathlib import Path

backend_dir = Path("backend").resolve()
sys.path.insert(0, str(backend_dir))

from app.main import app

openapi = app.openapi()
paths = openapi.get("paths", {})

routes = []
for path, methods in paths.items():
    for method, spec in methods.items():
        if method.lower() in ("get", "post", "put", "delete", "patch"):
            op_id = spec.get("operation_id", "")
            tags = spec.get("tags", [])
            routes.append({
                "method": method.upper(),
                "path": path,
                "operation_id": op_id,
                "tags": tags,
                "summary": spec.get("summary", "")
            })

routes.sort(key=lambda x: (x["path"], x["method"]))
print(f"Total canonical API endpoints from OpenAPI: {len(routes)}")
with open("scripts/backend_routes.json", "w") as f:
    json.dump(routes, f, indent=2)

for r in routes:
    print(f"{r['method']:<7} {r['path']:<45} [{','.join(r['tags'])}] {r['operation_id']}")
