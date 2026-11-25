# Connect Space ML to Vercel Sandbox and Drive

## Why the current deployment looks static

On 25 November 2026, `https://spaceml.vercel.app/api/health` returned
`state=setup_required`, `Sandbox available=false`, `Drive mounted=false` and
`write=false`, with these missing settings:

- `SMC_INTERNAL_TOKEN` (at least 32 random characters)
- `SMC_AUTH_USER`
- `SMC_AUTH_PASSWORD`

The deployment URL ending in `480jhu6kk-loganthps-projects.vercel.app` redirected
unauthenticated requests to Vercel sign-in. A browser logged into Vercel can view
that deployment while a Sandbox downloading its code cannot. Loading a page or
seeing a published model does not prove that live compute is connected.

This repository already contains the Vercel API gateway and the full Python
worker. Setting the credentials, connecting storage and redeploying activates
that worker; uploading the static frontend alone cannot do so. The application
does not generate credentials silently or remove operator authentication.

Google Drive is not used. Do not paste your Google folder URL or ID into any of
the variables below. No external backend or hosted database is needed.

## 1. Link the correct Vercel project

Open PowerShell in the extracted repository, where `vercel.json` is located.
Use Node.js 22 or later and Python 3.12 for the local checks.

```powershell
npm install -g vercel@latest sandbox@latest
vercel login
vercel link --project spaceml --scope loganthps-projects
vercel env pull .env.local
sandbox login
```

Use the team that owns `spaceml`; replace `loganthps-projects` if the dashboard
shows a different team slug. Link the existing project, not a new one.
The `.env.local` file stays on your computer and must not be committed.
Pulling environment variables does not configure the deployed application.

In **Vercel -> spaceml -> Settings -> Build and Deployment**, use repository root
as Root Directory and **Other** as the project framework preset. Remove manual
project-wide install/build/output overrides. `vercel.json` defines the frontend
and backend services separately. Deploy the whole repository, including `api/`,
`backend/`, `scripts/`, `starter/`, `release/`, `proxy.js` and `vercel.json`.

## 2. Create storage large enough for Python and the quick datasets

Inspect existing Drives first:

```powershell
vercel drives list --project spaceml --scope loganthps-projects
```

For a NEW Drive, an example starting size for the dependencies and quick
workflows is 10 GiB, subject to your team's quota:

```powershell
vercel drives get-or-create spaceml-data --region iad1 --max-size 10GiB --project spaceml --scope loganthps-projects
```

The current command is **`vercel drives`**, not the older `sandbox drives`
example. Check the command's printed name, region and maximum size.

Vercel's default Hobby/Pro-trial Drive is 1 GB, which is too small for this
training environment once Python, CPU Torch, package caches and data are stored.
Drive capacity is shared by dependencies, datasets, models and job history.
10 GiB is a starting allocation for quick workflows, not a promise that all full
datasets fit. Inspect your plan's aggregate quota before allocating it.

An existing Drive's region and maximum size are fixed. If the command reports a
conflict, inspect the existing Drive; do not delete it or retry with random
settings. Reuse it if suitable. If it is too small, create a differently named,
larger Drive within your quota and migrate files separately, preserving the old
Drive until verification finishes. Changing `SMC_DRIVE_NAME` alone does not
migrate any files. An existing named Sandbox retains its mount: when deliberately
selecting a replacement Drive, stop the old session and use a fresh
`SMC_SANDBOX_NAME` (for example `space-ml-production-v2`) for the replacement
runtime. Preserve both old resources until data migration is verified. Substitute
your chosen Sandbox name in the later diagnostic/stop commands.
Creating or using resources can consume billed Vercel usage.

## 3. Add the required project environment variables

In **Vercel -> spaceml -> Settings -> Environment Variables**, add the following
to **Production**. They must be available to the backend and the root routing
middleware; use shared project variables, not frontend `VITE_*` variables.

| Name | Value |
| --- | --- |
| `SMC_MODE` | `vercel` |
| `SMC_INTERNAL_TOKEN` | A new random server-to-server secret, at least 32 characters |
| `SMC_AUTH_REQUIRED` | `true` |
| `SMC_AUTH_USER` | `operator`, or your chosen login name |
| `SMC_AUTH_PASSWORD` | A strong password you choose for the application operator |
| `SMC_DRIVE_NAME` | `spaceml-data`, exactly matching step 2 |
| `SMC_REGION` | `iad1`, matching the Drive's actual region |
| `SMC_PUBLIC_ORIGIN` | `https://spaceml.vercel.app` if that is your Production domain |
| `SMC_SANDBOX_SECONDS` | `2700`, within your plan's session limit |

Generate the internal secret locally:

```powershell
python -c "import secrets; print(secrets.token_hex(32))"
```

Paste that output into Vercel's secret field, not into chat, Git or frontend
code. The operator password is different from the internal secret and from your
Vercel account password. Do not include quotation marks or trailing spaces in
dashboard values. Leave `SMC_EXECUTION` unset: only the Sandbox worker sets it.
Normally leave `SMC_SOURCE_ORIGIN` unset so the bootstrap uses the same immutable
deployment as the API. Remove an old hardcoded source-origin override unless
you deliberately serve the matching runtime archive from that origin.

The Function uses its automatic Vercel OIDC identity for Sandbox and Drive.
Do not copy a short-lived local OIDC token into Production. `vercel env pull`
obtains a local token for development, not a permanent server credential.

## 4. Let Sandbox fetch the private runtime source

In **spaceml -> Settings -> Deployment Protection**, configure **Protection
Bypass for Automation** for this project. Keep the generated secret server-side.
Vercel supplies the selected secret as `VERCEL_AUTOMATION_BYPASS_SECRET`; this
revision automatically forwards it to the bootstrap as an HTTP header.

If that system variable is unavailable in your deployment, set
`SMC_DEPLOYMENT_BYPASS` to the project's automation bypass secret for the same
environment. Do not use your account password or application operator password
as the bypass secret. Do not add it to `VITE_*` or a public URL.

The private `/runtime/<hash>.tar.gz` archive also requires `SMC_INTERNAL_TOKEN`.
Keep `proxy.js` and the `/runtime/*` middleware matcher: a normal unauthenticated
request should be denied, not download the server source. Successful access to
the dashboard in your logged-in browser is not a bypass for Sandbox requests.

## 5. Redeploy after saving settings

```powershell
vercel --prod
```

Alternatively, redeploy the current commit as Production in the dashboard.
Environment changes do not modify an already-built deployment. Open the NEW
Production deployment/domain afterwards; an old immutable deployment URL still
uses its old settings. Start with `https://spaceml.vercel.app`, rather than the
old `480jhu6kk` link, after confirming that Production points to the new build.

You do not need to create a second Vercel project, manually expose the Sandbox
port or start a separate Docker service. On the first health request, the API
creates/resumes `space-ml-production`, attaches the named Drive read-write at
`/smc`, installs Python/dependencies once and starts FastAPI plus the job worker.
Dependency installation and starter validation can take several minutes.

## 6. Verify compute, storage and operator access separately

Open these URLs in a browser:

- `https://spaceml.vercel.app/api/health`
- `https://spaceml.vercel.app/api/health/sandbox`
- `https://spaceml.vercel.app/api/health/drive`

Or use the included diagnostic command:

```powershell
python scripts/check_vercel_setup.py https://spaceml.vercel.app --wait 900
```

Wait for the main health response to report:

```json
{
  "state": "ready",
  "execution": "managed",
  "storage": "vercel-drive",
  "checks": {
    "sandbox": { "available": true },
    "drive": {
      "name": "spaceml-data",
      "mounted": true,
      "paths_valid": true,
      "writable": true
    }
  }
}
```

An unauthenticated response can still have `write=false` and `train=false` when
compute is ready. Click **Sign in** / **Sign in to train**, or open
`https://spaceml.vercel.app/api/session`, and enter the `SMC_AUTH_USER` and
`SMC_AUTH_PASSWORD` from step 3. This is the application's login, separate from
Vercel's dashboard login. Recheck `/api/health` in that browser: both capabilities
must become true. A private command-line password prompt is also available:

```powershell
python scripts/check_vercel_setup.py https://spaceml.vercel.app --username operator --wait 900
```

The diagnostic returns a nonzero exit code for missing settings, unavailable
compute, a failed Drive check, missing operator access or an HTML/login response.
The UI now retries health checks, including after startup errors, and provides
**Check connection**. Capacity remains unknown unless genuinely measured; it is
not inferred from the Function's `/tmp` or from synthetic filesystem metadata.

## 7. Verify a real workflow, then enable the remaining datasets

1. Open `/telemetry?mission=opssat`, select **Quick train**, and note its job ID.
2. Reload while the job is running. Confirm the same job continues; wait for
   success, then select a segment and **Predict this segment**.
3. Run its held-out evaluation and open the saved result.
4. Stop the session without deleting the Sandbox or Drive:

   ```powershell
   sandbox stop space-ml-production --project spaceml --scope loganthps-projects
   ```

5. Reopen the application. Wait for readiness again, sign in if requested, and
   confirm the model, evaluation and job history remain. This proves persistence
   in your actual account; a passing health check alone does not.

Once that passes, use the existing pages as follows:

| Page | Required data/action |
| --- | --- |
| `/telemetry` | Bundled SMAP/MSL data/models; select a channel with a model, then Run anomaly detection |
| `/telemetry?mission=opssat` | Bundled OPS-SAT; Quick train, predict, evaluate |
| `/telemetry?mission=hubble` | Install the real Hubble files first, then use its analysis/scoring actions |
| `/telemetry?mission=webb` | Corrected bundled visits; train/evaluate or score a visit |
| `/replay` | Choose an installed mission/item and start the replay |
| `/exoplanet` | Download the real Kepler catalog/light curves in Data & Training, then train before model scoring |
| `/mars` | Download AI4Mars samples, run Quick train, then analyze a sample |
| `/mars/traverse` | Bundled traverse waypoints; open map and select waypoints; uncached remote images still require their source |
| `/space-weather` | Download OMNI2, then train the forecaster and analyze an event |

Missing data and missing models are different states. The app must not mark
Kepler, AI4Mars, OMNI2 or Hubble as installed merely because compute is ready.
Real downloads can fail on upstream availability, access or quota errors; inspect
the saved job error instead of repeatedly starting the same download. Quick
workflows should pass before attempting full training or multi-GB datasets.

## 8. Diagnose the exact stage if it is still disabled

| Symptom | Check and correction |
| --- | --- |
| `setup_required` lists credentials | Add those names in the deployment's environment, then redeploy; Production variables do not fix an old Preview build |
| API returns HTML/Vercel login | Use the public Production origin; configure the automation bypass for server requests to protected deployment assets |
| `create_drive` / `create_sandbox` error | Check project/team, Sandbox access, OIDC, region, active sessions and account quotas in Function logs |
| Source denied, redirected, or checksum error | Verify shared internal secret and automation bypass; remove stale source-origin override; deploy both services from the same revision |
| First install fails or disk fills | Inspect Drive capacity and runtime status; the default 1 GB Drive is insufficient; preserve existing data when selecting larger storage |
| `verify_mount` or `storage_error` | Inspect `/api/health/drive`; the existing named Sandbox must mount the expected Drive read-write at `/smc` |
| `ready` but `write=false` | Sign in through `/api/session`; do not disable authentication to hide the issue |
| Install/Train returns 403 after sign-in | Check `SMC_PUBLIC_ORIGIN` matches the actual browser origin and includes the HTTPS scheme |
| Data missing / model unavailable | Install/validate the dataset, wait for successful training, and inspect that job's real error if it fails |

Useful read-only diagnostics after linking:

```powershell
sandbox list --project spaceml --scope loganthps-projects
vercel drives list --project spaceml --scope loganthps-projects
sandbox exec space-ml-production --project spaceml --scope loganthps-projects -- cat /smc/runtime.json
```

The last file reports bootstrap state and the latest startup error. Function logs
also report the failing stage/type. Never print the Sandbox's environment or share
secrets while collecting diagnostics.

For Preview, repeat configuration for **Preview** and redeploy it. Use a separate
adequately sized Drive and Sandbox per branch; do not attach Production's Drive
to another writable Sandbox. The default branch-specific names prevent accidental
mixing, but any manually selected Preview names must also be distinct.

## References and verification boundary

- [Drive CLI and creation options](https://vercel.com/docs/drives/cli-reference)
- [Drive limits and billing](https://vercel.com/docs/drives/pricing)
- [Sandbox authentication/OIDC](https://vercel.com/docs/sandbox/concepts/authentication)
- [Sandbox CLI](https://vercel.com/docs/sandbox/sdk/cli)
- [Protection Bypass for Automation](https://vercel.com/docs/deployment-protection/methods-to-bypass-deployment-protection/protection-bypass-automation)

This workspace is not logged into or linked to your Vercel account. The ZIP
contains code and setup checks; it does not provision resources or change your
dashboard. Live training, real source downloads and Sandbox/Drive persistence
still require the authenticated workflow above. Original starter archives remain
in the ZIP and are verified against the original dataset manifest.
