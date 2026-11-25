# Vercel frontend with a persistent Docker API

The default `vercel.json` remains a **read-only demo** until a real backend is
connected. It uses a Python Function and frozen results, not a training worker.
Do not change `SMC_MODE` on that Function to simulate persistent storage.

The persistent option reuses the existing FastAPI application, job worker,
and `smc-data`, `smc-models`, `smc-storage` named Docker volumes. No hosted database
is required. Keep the same Compose project name when upgrading, so Docker selects
the existing volumes. Never run `docker compose down -v`.

## Backend

Use an always-running Docker host behind an HTTPS reverse proxy. Set backend-only
environment variables `SMC_AUTH_USER`, `SMC_AUTH_PASSWORD` (strong, unique), and
`SMC_PUBLIC_ORIGIN=https://spaceml.vercel.app`. Then run:

```sh
docker compose -f docker-compose.yml -f docker-compose.production.yml up -d --build app
```

The production override forces `SMC_MODE=local`, requires authentication settings,
and turns off automatic mission downloads. It leaves the existing mounts and
worker intact. Configure the host's TLS reverse proxy/firewall before exposing
the API. HTTP Basic authentication must only be used over HTTPS. Credentials
are stored only in backend environment variables; the browser manages its
authentication and Vite never receives the password. Requests from browser
origins other than `SMC_PUBLIC_ORIGIN` cannot mutate data. Missing or incorrect
credentials fail closed. API responses use `private, no-store` when authentication
is enabled. `/api/session` challenges for credentials and returns to the frontend;
the UI offers **Sign in to train**. Public read endpoints remain available.

`SATNOGS_TOKEN`, if used, belongs only on this backend. No token is shipped in
the frontend, Vercel config, frozen results, or ZIP. The catalog currently marks
SatNOGS, ESA Missions 1–3 and FUNcube-1 **Data missing**; source adapters are
retained, but their complete import/job integration has not been restored or
validated in this release. They are not advertised as installed or operational.
Hubble likewise remains missing until real jitter logs are imported.

## Frontend

Once the real backend HTTPS origin exists, generate a separate configuration:

```sh
python scripts/configure_persistent_backend.py --backend-url https://YOUR-BACKEND-HOST
vercel --local-config vercel.docker.json --prod
```

The generator rejects credentials, HTTP origins, paths and queries. The generated
config contains only the static `frontend` service. Both `/api` and `/api/*` are
forwarded to the backend with the `/api` prefix preserved; SPA routing stays on
the frontend. Use repository root and preset Other. No backend secret belongs
in Vercel or `VITE_*`; a build variable alone cannot alter Vercel rewrites.
For Git deployments, replace `vercel.json` with the reviewed generated config
on the deployment branch. Keep the original config if the public site should
remain the demo. The root `SMC_MODE=vercel` environment variable is irrelevant to
the static frontend; set `SMC_MODE=local` only on the Docker backend.

Vercel supports this external-origin routing through [external rewrites](https://vercel.com/docs/routing/rewrites#rewrites-to-external-origins).
Browser traffic stays same-origin. The external backend is not a Vercel service,
so no Vercel service binding is appropriate.

## Webb regeneration and publication

`scripts/publish_mission_results.py` downloads the exact original 10 MAST product
filenames recorded in the old snapshot, parses their FITS units, checks strictly
increasing finite times, validates each visit's duration and ~64 ms spacing,
trains OPS-SAT and Webb, evaluates held-out segments/visits, and publishes the
new mission API snapshots, results and evidence. It uses an isolated temporary
installation and never changes the original starter or release archives.
Downloaded FITS files are cached under ignored `data/verified-mast-webb`.
`docs/mission-validation.json` records original URLs, SHA-256 checksums, per-visit
timing and held-out evaluation metrics. Run with optional training dependencies:

```sh
python scripts/publish_mission_results.py
python scripts/build_static_data.py --augment
python scripts/build_static_data.py
```

Normal Vercel builds use only the last command and never download or train.
The release also ships the actual trained model weights and corrected
full-resolution Webb store in `release/mission-results`, with a separate SHA-256
manifest. The optional production Compose override mounts this read-only at
`/published`. To explicitly install these into existing persistent volumes:

```sh
docker compose -f docker-compose.yml -f docker-compose.production.yml exec app \
  python -m spaceml.missions.published --from /published
```

This verifies every file before installation, retains previous Webb data as a
version, and never overwrites existing models. Re-running it is safe. Original
starter/release manifests and archives are unchanged. It does not run at startup.

Alternatively, import the original files using the corrected importer
(`python -m spaceml.missions.prepare webb --from <FITS-folder>`, inside the
container), then run `python -m spaceml.missions.train webb --quick`. Preserve the
old data version and model; the existing store and model publisher version them.
The corrected parser recognizes MJD days rather than interpreting them as
milliseconds. Webb alerts remain review candidates, not confirmed faults.

## Deployment acceptance gate

Do not advertise the connected backend as verified until all these pass on the
actual host. This workspace cannot validate your Docker host or a URL not yet supplied.

1. Open public `/api/health`: expect `mode=local`, `storage=persistent`, and
   unauthenticated write/train false. Sign in; both capabilities become true.
2. Open Data & Training, confirm OPS-SAT dataset installed and ready to train.
   Start **Quick train** from the UI. Reload during training and confirm the same
   job ID continues; a new browser session must not restart the job.
3. Open its held-out evaluation and verify the published model version and result.
4. Restart the `app` container without deleting volumes. Confirm the model,
   evaluation, job history and reviews still exist. Interrupt a running training
   job separately to verify its documented interrupted/resume behavior.
5. Import corrected Webb FITS, inspect duration and sample spacing for all visits,
   train and inspect its held-out evaluation, then restart and recheck persistence.
6. Verify unauthenticated uploads/training return 401, wrong browser origins
   return 403, and legitimate authenticated requests succeed. Only then expand
   access and enable other source workflows after their own data validation.

The repository configuration is prepared; no external host has been provisioned,
no live rewrite has been changed, and the deployed persistence workflow remains
unverified until a backend URL and Docker runtime are available.

## Checks completed in this workspace

`scripts/check_local_persistence.py` exercised the built frontend against a real,
authenticated FastAPI process using isolated persistent directories. OPS-SAT
Quick Train was started from Data & Training; reloading showed the same job
still running. The held-out evaluation was opened from the mission page. After
restarting FastAPI, the same model file checksum, successful job ID and saved
evaluation remained. See `persistent-workflow-validation.json`. This is a local
process restart check, not a claim about the remote Docker volume configuration.

The local mission/catalog/job/artifact regressions passed (15 tests), along with
21 Vercel API/authentication tests and the TypeScript/Vite production build.
The production frontend browser audit passed on 15 pages and 188 button actions,
with zero console/page errors, failed HTTP requests, click errors or horizontal
overflow at desktop and mobile widths; see `vercel-validation.json`.
Webb's original ten source files were independently downloaded and validated;
the timing/evaluation report and actual model weights are included in the release.
