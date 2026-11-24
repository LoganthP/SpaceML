#!/usr/bin/env bash
# Space ML Mission Control - standalone deployment audit (Linux/macOS). Same steps as scripts/audit.ps1.
#   scripts/audit.sh [--wait-full] [--skip-full] [--no-build]
set -euo pipefail
export SMC_PORT="${SMC_PORT:-8081}"
DC="docker compose -f docker-compose.yml -f docker-compose.offline.yml"
UI_FLAGS=(); BUILD=1
for a in "$@"; do case "$a" in --wait-full|--skip-full) UI_FLAGS+=("$a");; --no-build) BUILD=0;; esac; done

echo "== 1. Compose mounts"
$DC config --format json | python3 -c '
import json, sys
c = json.load(sys.stdin); vols = c["services"]["app"].get("volumes", [])
for v in vols: print(f"   {v[\"type\"]:8} {str(v.get(\"source\")):40} -> {v[\"target\"]}")
for t in ("/app/data", "/app/models", "/app/storage"):
    if not any(v["target"] == t and v["type"] in ("volume", "bind") for v in vols):
        sys.exit(f"{t} is not on a named volume or bind mount - recreating the container would lose it")'
CID=$($DC ps -a -q app || true)
if [ -n "$CID" ] && docker inspect "$CID" --format '{{range .Mounts}}{{.Type}} {{.Name}} {{.Destination}}{{"\n"}}{{end}}' \
     | grep -E '^volume [0-9a-f]{64} ' ; then
  echo "The current container keeps data in anonymous volumes. Copy it out first: docker cp $CID:/app/data ./data-backup"; exit 1
fi
[ "$BUILD" = 1 ] && { echo "== 2. Build"; $DC build app; }
echo "== 3. Start offline on http://localhost:$SMC_PORT"; $DC up -d app
wait_healthy() { for _ in $(seq 1 120); do curl -fs "http://localhost:$SMC_PORT/api/health" >/dev/null && return; sleep 3; done; echo "not healthy"; exit 1; }
wait_healthy
echo "== 4. Runtime check"
$DC exec app python -m pip show torch torchvision timm segmentation-models-pytorch
$DC exec app python -m pip check
$DC exec app python -c "import torch, torchvision, segmentation_models_pytorch; print(torch.__version__, torchvision.__version__); print(torchvision.ops.nms(torch.tensor([[0., 0., 1., 1.]]), torch.tensor([1.]), 0.5))"
echo "== 4b. AI4Mars as the web worker sees it"
$DC exec app python -m spaceml.mars.local check
echo "== 5. Audit"
$DC exec app python -m spaceml.audit env
$DC exec app python -m spaceml.audit data
$DC exec app python -m spaceml.audit ui "${UI_FLAGS[@]}"
echo "== 6. Restart persistence"
$DC exec app python -m spaceml.audit snapshot
$DC restart app; wait_healthy
$DC exec app python -m spaceml.audit verify-restart
echo "== 7. Report"
$DC exec app python -m spaceml.audit report
$DC cp app:/app/storage/audit ./audit-results
echo "Report: audit-results/report.md"
