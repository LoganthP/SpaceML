# Space ML Mission Control - standalone deployment audit (Windows PowerShell, from the project folder).
#   powershell -ExecutionPolicy Bypass -File scripts\audit.ps1            # quick runs + full-run startup checks
#   powershell -ExecutionPolicy Bypass -File scripts\audit.ps1 -WaitFull  # also let every full run finish (hours)
#   powershell -ExecutionPolicy Bypass -File scripts\audit.ps1 -Backup    # first save the data volumes to .\backup
# Datasets, models and job history live in named volumes and are kept; nothing is deleted.
param([switch]$WaitFull, [switch]$SkipFull, [switch]$Backup, [switch]$NoBuild, [string]$Port = "8081")
$ErrorActionPreference = "Stop"
$env:SMC_PORT = $Port
$dc = @("compose", "-f", "docker-compose.yml", "-f", "docker-compose.offline.yml")

Write-Host "== 1. Compose mounts (data must be on volumes before the container is recreated)"
$cfg = (docker @dc config --format json) | ConvertFrom-Json
$vols = $cfg.services.app.volumes
$vols | ForEach-Object { "   {0,-8} {1,-40} -> {2}" -f $_.type, $_.source, $_.target }
foreach ($t in "/app/data", "/app/models", "/app/storage") {
  if (-not ($vols | Where-Object { $_.target -eq $t -and $_.type -in @("volume", "bind") })) {
    throw "$t is not on a named volume or bind mount - stop: recreating the container would lose it."
  }
}
$cid = (docker @dc ps -a -q app)
if ($cid) {
  $mounts = (docker inspect $cid | ConvertFrom-Json)[0].Mounts
  $anon = $mounts | Where-Object { $_.Type -eq "volume" -and $_.Name -match '^[0-9a-f]{64}$' }
  if ($anon) {
    $anon | ForEach-Object { "   ANONYMOUS volume at $($_.Destination): $($_.Name)" }
    throw "The current container keeps data in anonymous volumes. Copy it out first: docker cp ${cid}:/app/data .\data-backup"
  }
  Write-Host "   existing container uses named volumes only - safe to recreate"
}
if ($Backup) {
  New-Item -ItemType Directory -Force backup | Out-Null
  foreach ($v in "smc-data", "smc-models", "smc-storage") {
    $name = ($cfg.volumes.$v.name); if (-not $name) { $name = "$($cfg.name)_$v" }
    docker run --rm -v "${name}:/v:ro" -v "${PWD}\backup:/b" alpine tar czf "/b/$v.tgz" -C /v .
    Write-Host "   saved backup\$v.tgz"
  }
}

if (-not $NoBuild) {
  Write-Host "== 2. Build (matched CPU torch/torchvision; the build fails if torchvision::nms is missing)"
  docker @dc build app
}
Write-Host "== 3. Start offline on http://localhost:$Port"
docker @dc up -d app
function Wait-Healthy {
  for ($i = 0; $i -lt 120; $i++) {
    try { if ((Invoke-WebRequest -UseBasicParsing "http://localhost:$Port/api/health" -TimeoutSec 5).StatusCode -eq 200) { return } } catch {}
    Start-Sleep 3
  }
  throw "app did not become healthy"
}
Wait-Healthy

Write-Host "== 4. Runtime check inside the job container"
docker @dc exec app python -m pip show torch torchvision timm segmentation-models-pytorch
docker @dc exec app python -m pip check
docker @dc exec app python -c "import torch, torchvision, segmentation_models_pytorch; print(torch.__version__, torchvision.__version__); print(torchvision.ops.nms(torch.tensor([[0., 0., 1., 1.]]), torch.tensor([1.]), 0.5))"

Write-Host "== 4b. AI4Mars as the web worker sees it (cwd, dataset root, manifest, usable/held-out per split)"
docker @dc exec app python -m spaceml.mars.local check

Write-Host "== 5. Audit: environment, datasets, UI workflows"
docker @dc exec app python -m spaceml.audit env
docker @dc exec app python -m spaceml.audit data
$ui = @("exec", "app", "python", "-m", "spaceml.audit", "ui")
if ($WaitFull) { $ui += "--wait-full" }
if ($SkipFull) { $ui += "--skip-full" }
docker @dc @ui

Write-Host "== 6. Restart persistence"
docker @dc exec app python -m spaceml.audit snapshot
docker @dc restart app
Wait-Healthy
docker @dc exec app python -m spaceml.audit verify-restart

Write-Host "== 7. Report"
docker @dc exec app python -m spaceml.audit report
docker @dc cp app:/app/storage/audit ./audit-results
Write-Host "Report: audit-results\report.md (raw evidence: audit-results\*.json)"
