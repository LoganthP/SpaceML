# Space ML Mission Control — Vercel only

For the current disabled deployment and detailed PowerShell/dashboard steps, use
[VERCEL-SANDBOX-SETUP.md](VERCEL-SANDBOX-SETUP.md). It covers credentials,
Drive sizing, deployment protection, redeployment, operator sign-in and a real
OPS-SAT train/persist/restart acceptance check.

The active configuration runs the frontend, API gateway, scientific backend,
training worker and persistent storage on Vercel. No external backend URL,
Docker host or hosted database is required. Existing local/Docker commands remain
available for local development.

ESA Missions 1–3, FUNcube-1 and SatNOGS are removed from the catalog and mission
selector. Their original files and source adapters are retained.

## Connect and deploy

1. Extract `space-ml-mission-control-3.1.10-vercel.zip` and commit its contents.
   Preserve `.gitattributes` so dataset checksums survive Windows checkouts.
2. Import the repository into Vercel. Root Directory: repository root. Framework
   preset: **Other**. Clear project-wide install/build/output overrides; the
   services in `vercel.json` specify FastAPI/Python and Vite independently.
3. Add these backend environment variables for Production:

   | Variable | Value |
   | --- | --- |
   | `SMC_MODE` | `vercel` |
   | `SMC_DRIVE_NAME` | `spaceml-data` (the Vercel Drive created below; Production only) |
   | `SMC_INTERNAL_TOKEN` | A new random secret, at least 32 characters; generate with `python -c "import secrets; print(secrets.token_hex(32))"` |
   | `SMC_AUTH_USER` | Your operator login name |
   | `SMC_AUTH_PASSWORD` | A strong, unique operator password |
   | `SMC_AUTH_REQUIRED` | `true` (also the Vercel gateway default) |
   | `SMC_PUBLIC_ORIGIN` | Your public HTTPS origin, e.g. `https://spaceml.vercel.app` |

   Never put secrets in `VITE_*`, Git or the ZIP. Scope Preview settings separately.
   `SMC_INTERNAL_TOKEN` must be available to both the backend and the root routing
   middleware; use the project's shared environment variables.
4. Ensure your project/team permits **Vercel Sandbox and Vercel Drives**. The SDK
   uses the Function's Vercel OIDC identity to create project-owned resources.
   No third-party application backend or database integration is needed.
   The shared Google Drive folder is unrelated and is not used. Never put its
   URL or folder ID in `SMC_DRIVE_NAME`.
5. Deploy through Git, or from repository root:

   ```sh
   npm install -g vercel sandbox
   vercel login
   vercel link
   vercel env pull .env.local
   sandbox login
   vercel drives get-or-create spaceml-data --region iad1 --max-size 10GiB
   vercel --prod
   ```

   Install the optional Sandbox CLI if `sandbox` is unavailable, following the
   [current Drive CLI guide](https://vercel.com/docs/drives/cli-reference). Select the
   same team and `spaceml` project in both CLIs. The SDK can also create the named
   Drive on the first request, but the default 1 GB on Hobby/trial is too small
   for this stack: explicitly allocate adequate storage first. An existing
   Drive's region must match `SMC_REGION` (default `iad1`). Keep Preview's Drive
   separate by leaving `SMC_DRIVE_NAME` unset there. Production OIDC is automatic;
   `.env.local` is for local testing and must not be committed.

The public services remain `backend` at `/api` and `/api/*`, and `frontend` at
all other paths. The browser calls same-origin `/api/*`. There are no bindings:
the frontend is static, and Sandbox/Drive are SDK resources rather than additional
services. No external API rewrite is configured.

The first visit starts initialization. The page shows **Preparing Vercel training
compute** and polls readiness. Installing dependencies into persistent storage
can take several minutes the first time. Click **Sign in to train** when compute
is ready. HTTPS HTTP Basic credentials are managed by the browser, not stored
by frontend JavaScript. Missing settings produce `state=setup_required` with
the setting names in `/api/health`; training is not falsely advertised as working.
Published results remain browsable before setup is complete.
One status banner reports Sandbox availability and Drive writability. Install
and Train keep their action labels and are disabled with a setup hint until
both checks succeed. `/api/health/sandbox` and `/api/health/drive` expose those
checks separately; startup errors include the failing stage and error type.
The worker verifies dataset, model and job paths are under `/smc`, then writes,
syncs and reads a temporary file in each directory. A failed check blocks writes.
Capacity is shown as **Capacity unavailable** unless a trustworthy measurement
exists; the filesystem estimate in diagnostics is not a Vercel Drive quota.
An unauthenticated health request intentionally reports `write=false` and
`train=false` even after compute is ready; sign in to enable operator actions.

## Architecture and capabilities

| Component | Vercel resource | Behavior |
| --- | --- | --- |
| Vite frontend | Static frontend service | Existing pages, assets and SPA deep links |
| `api/index.py` | Python Function/backend service | Authentication, resource startup/readiness and bounded HTTP proxy |
| Existing FastAPI app and job worker | Vercel Sandbox | Real downloads, training, inference, analysis and reviews, outside Function request duration |
| Datasets, models, checkpoints, uploads, SQLite job/run/review records | Vercel Drive mounted at `/smc` | Retained across Function cold starts, Sandbox suspension and API restarts |
| Original starter assets and corrected Webb/OPS-SAT publication | Verified build source/runtime source archive | Original archives remain unchanged |

The managed image is supplied by Vercel (`vercel/sandbox/python:3.14`); you do not
build or host a Docker image. Its included `uv` installs Python 3.12 and the
existing CPU training dependencies into the Drive. Torch/torchvision stay out of
the Function bundle. The scientific pipelines are the same implementations used
locally. A cached result is never labeled as a new training execution.

Once ready and signed in, Download/Install and Train call the live backend.
Training still requires actual installed, validated data. Kepler, AI4Mars, OMNI2
and Hubble need their real source downloads/imports before training. Telemetry,
OPS-SAT, corrected Webb, traverse and scene archives have bundled assets.
Installed, ready to train and model available remain separate states. Source
outages or access requirements are surfaced through the existing job errors.

Runtime downloads from NASA/MAST/OMNIWeb/Hugging Face/PyPI are data/dependency
sources, not external application backends. Normal builds perform no such
downloads or training through the build scripts.

Browser uploads through the Function are limited to 4 MB. Use Download/Install
for large built-in datasets: the Sandbox fetches them directly into Vercel
storage. Responses above 4 MB redirect to a short-lived, read-only signed Sandbox
download so the Function stays under its 4.5 MB response limit. Arbitrary
multi-gigabyte browser uploads are not implemented by this gateway.

## Persistence, limits and security

Production has one stable named runtime and Drive. Previews use a separate
identity derived from the Git branch. One Sandbox owns the writable Drive; one
supervisor owns the API/worker. Updates use separate source folders and wait for
active jobs/runs before restarting. Datasets and models are not removed on resume.
The default production names are `space-ml-production` (Sandbox) and
`space-ml-production-data` (Drive); do not delete the Drive during restart checks.
Setting `SMC_DRIVE_NAME=spaceml-data` overrides that default. If you already have
data in the default Drive, keep its name instead: changing the variable selects
another Drive and does not migrate files.

Vercel Sandbox and Drive usage consumes account quotas and may be billed. Sandbox
and Drive capacity must accommodate dependencies as well as datasets. The
current Hobby/trial default of 1 GB is insufficient for this Python training stack;
check your plan and Drive size before installing large datasets. The app never
represents a placeholder or filesystem estimate as remaining Drive quota.
Session duration/capacity depend on the plan. The default session is 45 minutes. Set
`SMC_SANDBOX_SECONDS` within your plan's limit for longer sessions (up to 24 hours
on eligible plans). Suspension stops processes; on the next request the runtime
resumes and recovers interrupted jobs/runs using the same IDs. Existing pipeline
checkpoints/completed artifacts are reused where supported; a step without a
checkpoint restarts. This is not unlimited uninterrupted compute.

Official references: [Sandbox persistence](https://vercel.com/docs/sandbox/concepts/persistent-sandboxes),
[Vercel Drives](https://vercel.com/docs/drives),
[session limits](https://vercel.com/kb/guide/vercel-sandbox-duration-and-persistence).

Mutating public API routes require authentication and an approved browser origin.
The Sandbox port requires the internal token for every request. It is never sent
to the browser. Large downloads receive a path/query-bound GET-only signature
that expires after two minutes. To rotate the internal token, stop the Sandbox
first, change the secret and redeploy; preserve the Drive.

SQLite uses rollback-journal mode/full synchronous writes in the single mounted
worker. The Function never opens the database. Actual mounted filesystem and
restart behavior must be checked after connecting your Vercel account.

Optional settings: `SMC_REGION=iad1`, `SMC_SANDBOX_SECONDS=2700`,
`SMC_LIVE_MAX_AGE=120`, `SMC_SOURCE_ORIGIN` (explicit HTTPS source deployment
origin). Normally leave the source origin unset to use this exact deployment.
Protected deployment assets use `VERCEL_AUTOMATION_BYPASS_SECRET` automatically,
or the explicit `SMC_DEPLOYMENT_BYPASS` override. Keep these values backend-only.
For local SDK testing, use `vercel link && vercel env pull`. If OIDC is unavailable,
the SDK supports backend-only `VERCEL_TOKEN`, `VERCEL_PROJECT_ID`, `VERCEL_TEAM_ID`;
never commit or paste them into source.

## Verify after connecting Vercel

1. Wait for `/api/health`: `state=ready`, `execution=managed`,
   `storage=vercel-drive`. Sign in; `write` and `train` must become true.
   Confirm `/api/health/sandbox` and `/api/health/drive` both report `ok=true`,
   and the latter reports the intended Drive name, `mounted=true`,
   `paths_valid=true` and `writable=true`.
2. Start OPS-SAT Quick Train in the deployed UI. Reload while it runs and confirm
   the same job ID continues; open its held-out evaluation.
3. Stop the Sandbox session without deleting the Drive. Reopen the application;
   verify that model weights, evaluation, job history and reviews remain.
4. Interrupt a running job to verify same-ID recovery. Check corrected Webb
   timing/training, then download and train Kepler, AI4Mars and OMNI2 individually.
5. Confirm unauthenticated writes return 401 and wrong browser origins return
   403. Verify a large result redirects without exposing the internal secret.

You selected ZIP delivery and will connect Vercel yourself. Cloud provisioning,
dataset downloads/training, Drive filesystem behavior and Sandbox restart checks
are therefore **not verified on Vercel** in this workspace. Local tests must not
be interpreted as cloud deployment success.

## Local build and packaging

```sh
npm ci
python scripts/build_static_data.py
python scripts/build_vercel_runtime.py
cd frontend
npm ci
npm run build
cd ..
python scripts/make_vercel_zip.py
```

The deterministic runtime source archive contains application code, requirements,
starter/release data and samples. It excludes `.env` files, secrets, caches, tests
and frozen API responses. The root `proxy.js` routing middleware protects `/runtime/*` before
the static cache using the internal token; bootstrap requests include that token.
After deployment, verify an unauthenticated `/runtime/<hash>.tar.gz` request is
403 before starting the runtime. Do not remove this middleware or expose the
server source archive publicly. Build-generated runtime archives are recreated after
ZIP extraction. The final ZIP verifier checks `DATASETS-MANIFEST.json`,
`starter/pack.json`, frozen results and generated mission artifact hashes from
inside the ZIP. No original archive is deleted or replaced.

`docs/mission-validation.json` records corrected Webb timing/source checksums and
held-out metrics. Actual published weights/data are in `release/mission-results`.
`persistent-workflow-validation.json` reports the prior local process restart
check, not a cloud Sandbox/Drive check.
`readiness-ui-validation.json` records simulated setup/sign-in/startup recovery
transitions against the real frontend; it does not represent live cloud inference.

Local validation for this revision: 48 Vercel/fallback/gateway/security/archive
tests and 16 local mission/catalog/job/artifact/recovery tests passed. TypeScript
and the Vite production build passed. All 15 pages passed the browser audit with
zero console, HTTP, click or overflow errors. The authenticated local OPS-SAT
UI training/reload/evaluation/process-restart workflow also passed. Actual Vercel
provisioning and cloud training remain your post-connection acceptance checks.
Native `vercel dev --local` also started both services: health/catalog/deep links
returned 200 and the private runtime archive route returned 403 without its token.

The Linux Python 3.12 gateway dependency wheels plus application/frozen data
measure approximately 54 MiB, below the 500 MB Python Function limit. The large
runtime source archive is a protected static asset, not part of the Function.
See `vercel-runtime-size.json`; this is an application payload measurement,
excluding Vercel's launcher, not an uploaded production bundle measurement.
