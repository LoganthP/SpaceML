# Deploy Space ML Mission Control 3.1.10 on Vercel

This is one Vercel project with `backend` and `frontend` services. It requires no
external backend, database, Sandbox, Drive, queue, cron, or storage account.

## Import and deploy

1. Extract `space-ml-mission-control-3.1.10-vercel.zip`, commit the contents to your
   repository, and import that repository into Vercel.
2. Set the project Root Directory to the repository root. Use **Other** as the
   project framework preset; `vercel.json` selects FastAPI and Vite per service.
   Clear project-wide install/build/output overrides. Do not select the root
   Dockerfile as the Vercel application.
3. Set `SMC_MODE=vercel` for Production and Preview. Python is pinned to **3.12**
   by the root `.python-version` and `pyproject.toml`. No secrets are required.
4. Deploy through Git, or run from the repository root:

   ```sh
   vercel link
   vercel --prod
   ```

The public backend is `/api` and `/api/*`. All other public traffic goes to the
frontend. React Router deep links, including `/mars/traverse`, load after refresh.
SPA rewriting excludes actual assets, frozen data, and Vite development resources.
Hashed build assets have immutable one-year browser caching. Frozen data has CDN
caching and content-addressed filenames.

There are **no service bindings**. The frontend is static browser code and calls
same-origin `/api/*`; neither service makes server-side calls to the other. Never
put an internal service binding URL into `VITE_*` build variables.

## Settings

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `SMC_MODE` | `local`; the Function entrypoint defaults to `vercel` | The single application mode switch; set `vercel` in the dashboard explicitly |
| `STORAGE_DIR` | `/tmp/smc` on Vercel | Ephemeral live-feed cache; paths outside the OS temporary directory are rejected |
| `SMC_LIVE_MAX_AGE` | `120` seconds | Space Telescope Live refresh/cache age |

Local/Docker continues using `backend/requirements.txt`, mounted data/models/
storage, SQLite, existing job workers, and `SPACEML_STORAGE_DIR`. Existing
`docker-compose` and `scripts/dev.sh` commands are unchanged. Leave `SMC_MODE`
unset or `local` there. No local dataset, archive, or database is deleted by the
mode switch. Version reporting is updated to 3.1.10.

## What is read-only on Vercel, and why

| Operation | Vercel behavior | Reason |
| --- | --- | --- |
| Browse datasets, runs, models, evaluation, traverse, stored series, evidence and downloads | Works from frozen, verified outputs | Does not require durable writes or ML imports |
| Score a bundled telemetry sample; repeat a stored result | Returns the matching precomputed run, with `execution=precomputed` | Does not launch a worker or claim a new inference happened |
| Train/evaluate new models, install/download datasets, rescan | HTTP 409 `Available in local/Docker mode`; controls disabled with a tooltip | No persistent filesystem or background process |
| Upload datasets/models, delete data, activate versions, save reviews/masks/annotations | Same 409 and disabled controls | These require durable writes |
| Compute a result for an input absent from the frozen snapshot | 409 with a local/Docker hint | Torch and other scientific dependencies are optional and absent from the Function |
| Hubble/Webb live observation and survey image proxy | Works at runtime; `/tmp` cache plus `s-maxage`/stale CDN headers | A bounded HTTP proxy is suitable for a Function |

`/api/health` reports version, mode, read-only features, capabilities, and available
precomputed modules. The frontend reads capabilities once and preserves the
existing layout while disabling writable actions. The live proxy preserves the
last-known/offline notice. If a new cold instance cannot reach upstream and has
no cached observation, it returns an explicit unavailable notice; no observation
is invented. `/tmp` is created only when a cache is needed and may disappear.

The supplied assets contain telemetry data/models for 81 channels, OPS-SAT
segments, ten Webb visit series, flood scene archives, and Curiosity waypoints.
They do **not** contain Kepler folded light curves/CNN weights, AI4Mars training
pairs/Mars weights, or OMNI2 caches. Those pages retain their existing truthful
readiness/empty states; this release does not fabricate exoplanet candidates,
segmentation masks, or storm forecasts. The supplied Webb timestamp repair notice
is retained. Full local/Docker preparation remains available.

## Offline static-data build

The ZIP includes a frozen capsule computed from the actual starter archives and
existing trained models. Ordinary Vercel builds run:

```sh
python3 scripts/build_static_data.py
```

This standard-library-only step checks every source hash against
`DATASETS-MANIFEST.json`, checks frozen output hashes, and exports the capsule to
`frontend/public/data`. It does not import Torch or download datasets. Both
services run this check independently. Normal dependency installation contacts
the npm and Python package registries; data generation has no network allowlist.
The source ZIP stores frozen JSON once under `backend/static_data`; frontend JSON
copies are deliberately generated by this build step instead of duplicated in
the source upload. Static report/image/evidence assets remain in the ZIP.

To regenerate the capsule on a local preparation machine:

```sh
python -m pip install -r backend/requirements-train.txt
python scripts/build_static_data.py --refresh
```

This installs the existing starter pack and release bundle into a temporary build
directory, runs real inference, freezes API response shapes and evidence, and
removes only that temporary directory on completion. It never rewrites the source
archives or their manifests. Keep Torch/torchvision on matching CPU builds, as
documented for local installation. Commit the new capsule and static assets
together after regeneration. Adding Kepler/Mars/OMNI outputs requires supplying
their real prepared data/models through the release/starter pack first.

The Function runtime dependencies are the root `requirements.txt`/`pyproject.toml`
(FastAPI/Uvicorn and their small dependencies). Training dependencies remain
optional under `backend/requirements-train.txt`; local/Docker requirements retain
their original behavior. `app.main` selects the small API before importing any
local database, job, or scientific module.

## Verification and limits

```sh
vercel dev --local --listen 3100
python scripts/check_vercel_browser.py
python -m pip install -r tests_vercel/requirements.txt
python -m pytest tests_vercel -q
cd backend && python -m pytest -q
cd ../frontend && npx tsc --noEmit && npm run build
```

Local Vercel development uses the isolated root Python 3.12 environment; the
frontend port helper works on Windows and Linux. On Windows, stop/restart Vercel
after changing API Python files if the runtime's reloader waits on open requests.

The measured Linux Python 3.12 dependency wheels expand to 10.34 MiB. Backend
source plus the frozen capsule is approximately 37.58 MiB: **47.92 MiB combined**,
well below the standard **500 MB uncompressed Python Function limit**. This is a
measurement of the application payload, not a claim that a cloud production
build was uploaded. A Vercel build may add a small platform launcher. The Function
configuration excludes original models/archives and frontend/static assets from
its bundle; they remain in the release ZIP. Function memory is 1024 MB and maximum
duration is 30 seconds.

Function JSON responses are bounded at 4,000,000 bytes. Larger prebuilt responses
and all report/evidence binaries are served as static assets through redirects,
so a Function never streams an oversized body. Stored mission item lists are
filtered and paginated, and series requests honor column/range/point limits.
Source: [Vercel Function limits](https://vercel.com/docs/functions/limitations).

The local suite passes with legacy source tests skipped when their missions are
not declared in the v3.1.9 catalog (ESA/SatNOGS/FUNcube), plus platform-dependent
skips. This keeps removed missions out of the product instead of changing local
features to satisfy stale 3.1.6 integration tests. See `vercel-validation.json`
for browser results. The Docker engine was unavailable during validation; Docker
and Compose files and local development scripts were left unchanged by this work.

## Recreate and verify the ZIP

```sh
python scripts/make_vercel_zip.py
```

The packager explicitly preserves nested `starter/data`, `starter/models`, and
`starter/pack.json`. It opens the completed ZIP, validates ZIP CRCs, and verifies
all starter/release files against the copies of `DATASETS-MANIFEST.json` and
`starter/pack.json` **inside that ZIP**, including both `.tar.gz` archives. It
writes `space-ml-mission-control-3.1.10-vercel.verification.json` alongside it.

## Final project configuration

Both services are public through the top-level rewrites, with API rules first
and the frontend catch-all last. There are no internal services or bindings.

```json
{
  "$schema": "https://openapi.vercel.sh/vercel.json",
  "services": {
    "backend": {
      "root": ".",
      "runtime": "python",
      "framework": "fastapi",
      "entrypoint": "api.index:app",
      "buildCommand": "python3 scripts/build_static_data.py",
      "functions": {
        "api/index.py": {
          "maxDuration": 30,
          "memory": 1024,
          "excludeFiles": "{frontend/**,starter/**,models/**,data/**,storage/**,release/**,samples/**,notebooks/**,docs/**,audit*/**,deployment/**,scripts/**,tests*/**,backend/tests/**,**/*.zip,**/node_modules/**,**/.venv/**,**/.vercel/**}"
        }
      }
    },
    "frontend": {
      "root": "frontend",
      "framework": "vite",
      "installCommand": "npm ci --no-audit --no-fund",
      "buildCommand": "python3 ../scripts/build_static_data.py && npm run build",
      "devCommand": "node ../scripts/vercel_frontend_dev.cjs",
      "outputDirectory": "dist",
      "headers": [
        {
          "source": "/assets/(.*)",
          "headers": [
            {
              "key": "Cache-Control",
              "value": "public, max-age=31536000, immutable"
            }
          ]
        },
        {
          "source": "/data/(.*)",
          "headers": [
            {
              "key": "Cache-Control",
              "value": "public, max-age=300, stale-while-revalidate=86400"
            }
          ]
        }
      ],
      "rewrites": [
        {
          "source": "/((?!assets/|data/|src/|node_modules/|@|favicon[.]svg$).*)",
          "destination": "/index.html"
        }
      ]
    }
  },
  "rewrites": [
    {
      "source": "/api",
      "destination": {
        "service": "backend"
      }
    },
    {
      "source": "/api/(.*)",
      "destination": {
        "service": "backend"
      }
    },
    {
      "source": "/(.*)",
      "destination": {
        "service": "frontend"
      }
    }
  ]
}
```
