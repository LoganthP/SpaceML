import { WriteButton, useReadOnly } from "../lib/capabilities";
import { CheckCircle2, Container, Database, ExternalLink, Laptop, Package, Play, RefreshCw, SquareTerminal, TriangleAlert } from "lucide-react";
import { useState } from "react";
import { activeJob, JobCard, useJobs, useOnJobFinished } from "../components/Jobs";
import { Catalog, useCatalog } from "../components/Catalog";
import { MarsModelUpload } from "../components/Setup";
import { CopyCommand, ErrorNotice, Glass, IconBadge, Page, Segmented, Spinner } from "../components/ui";
import { api, useApi } from "../lib/api";
import { MODULE_META, timeAgo } from "../lib/format";
import { usePrefs } from "../lib/theme";
import type { Dataset, ModelModule } from "../lib/types";

interface ModelVersion { module: ModelModule; version: string; model_name: string | null; created_at: string | null }
type Models = Record<ModelModule, { active: string | null; versions: ModelVersion[] }>;
type Env = "local" | "docker";

interface Action { task: string; label: string; cli: string; primary?: boolean; params?: { key: string; label: string; def: number; min: number; max: number }[]; note?: string }

const ACTIONS: Record<ModelModule, Action[]> = {
  telemetry: [
    { task: "telemetry.download", label: "Download telemetry", cli: "spaceml.telemetry.data download" },
    { task: "telemetry.train", label: "Train all channels + evaluate (fast)", primary: true, cli: "spaceml.telemetry.train --fast && … spaceml.telemetry.evaluate",
      note: "Fast mode: channels train in parallel on every 4th window - about 10x quicker with the same detections on the channels we compared. Resumable." },
    { task: "telemetry.retrain", label: "Retrain all channels from scratch (new version)", cli: "spaceml.telemetry.train --fast --version <new> && … spaceml.telemetry.evaluate --version <new>",
      note: "Keeps the current models until the new version covers more channels; then it becomes active automatically." },
    { task: "telemetry.evaluate", label: "Re-evaluate", cli: "spaceml.telemetry.evaluate" },
  ],
  exoplanet: [
    { task: "exoplanet.setup", label: "Full setup: catalog → light curves → train", primary: true, cli: "spaceml.quickstart --skip-telemetry --skip-mars --stars 300",
      params: [{ key: "n_stars", label: "stars", def: 300, min: 10, max: 5000 }, { key: "epochs", label: "epochs", def: 60, min: 3, max: 300 }],
      note: "6 parallel downloads, 8 of 17 quarters per star (half the files). 300 stars is the minimum for meaningful held-out metrics. Resumable." },
    { task: "exoplanet.catalog", label: "Re-download catalog (adds RA/Dec + stellar data)", cli: "spaceml.exoplanet.data tce" },
    { task: "exoplanet.lightcurves", label: "Download more light curves", cli: "spaceml.exoplanet.data lightcurves --n-stars 300 --max-quarters 8",
      params: [{ key: "n_stars", label: "stars", def: 300, min: 10, max: 5000 }] },
    { task: "exoplanet.train", label: "Train on cached stars", cli: "spaceml.exoplanet.train", params: [{ key: "epochs", label: "epochs", def: 60, min: 3, max: 300 }] },
  ],
  mars: [
    { task: "mars.samples", label: "Prepare curated samples", cli: "spaceml.mars.data samples --n 36" },
    { task: "mars.train_quick", label: "Train terrain model on this computer (CPU)", primary: true, cli: "spaceml.mars.train --quick && … spaceml.mars.evaluate --latest",
      params: [{ key: "max_train", label: "images", def: 400, min: 50, max: 5000 }, { key: "epochs", label: "epochs", def: 4, min: 1, max: 50 }],
      note: "DeepLabV3+ (ResNet-34) trains on installed AI4Mars pairs. Runtime depends on the image count and CPU." },
    { task: "mars.evaluate", label: "Re-evaluate", cli: "spaceml.mars.evaluate --latest" },
    { task: "mars.traverse", label: "Download Curiosity traverse waypoints", cli: "spaceml.mars.traverse download",
      note: "NASA/JPL mission-map waypoints (site, drive, sol, position) for the Rover Traverse Explorer - about 0.7 MB." },
  ],
  spaceweather: [
    { task: "spaceweather.setup", label: "One step: download → train → replay May 2024", primary: true,
      cli: "spaceml.spaceweather.data download --years 2010 2011 2012 2013 2014 2015 2016 2017 2024 && … spaceml.spaceweather.forecast --if-missing && … app.starter replay --event 2024-05",
      note: "Download, train and replay in one job (needs internet). Offline? Use the release bundle, or upload omni2_YYYY.dat files on the Mission Replay page." },
    { task: "spaceweather.download", label: "Download OMNIWeb years", cli: "spaceml.spaceweather.data download",
      note: "2003, 2010-2017 and 2024 (Halloween, St. Patrick's Day, Sept 2017 and Gannon storms): ~10 files, ~35 MB." },
    { task: "spaceweather.train", label: "Train Dst forecasters vs persistence", cli: "spaceml.spaceweather.forecast --train-years 2010-2016 --test-years 2017",
      note: "Chronological split; 5 horizons (+1 to +24 h); about a minute on a laptop." },
  ],
};

interface Starter { present: boolean; pack_version?: string; created_at?: string; reference_runs?: string[];
  contents?: { id: string; title: string; source: string; source_url?: string; data_date: string; license: string; files: number; bytes: number }[];
  installed?: { installed_at?: string; complete?: boolean; errors?: string[] }; not_in_pack?: { module: string; what: string; how: string }[] }

function StarterCard() {
  const { data } = useApi<Starter>("/api/starter");
  if (!data?.present) return null;
  return (
    <Glass className="hud p-5">
      <div className="flex flex-wrap items-center gap-2">
        <Package className="h-4 w-4 text-indigo-500" />
        <div className="kicker">Starter pack {data.pack_version}</div>
        {data.installed?.complete ? <span className="chip bg-emerald-100 text-emerald-700">installed · sha256 verified</span>
          : <span className="chip bg-amber-100 text-amber-800">{data.installed?.errors?.length ? `${data.installed.errors.length} file problem(s)` : "not installed"}</span>}
        <span className="text-[11px] text-ink-soft">{data.reference_runs?.length ?? 0} bundled reference runs (labelled “bundled reference” everywhere)</span>
      </div>
      <div className="mt-3 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {data.contents?.map((c) => (
          <div key={c.id} className="glass-inset p-3 text-xs">
            <div className="font-semibold">{c.title}</div>
            <div className="text-ink-soft">{c.source}</div>
            <div className="mt-1 text-[11px] text-ink-faint">{c.data_date} · {c.files} files · {(c.bytes / 1e6).toFixed(1)} MB · {c.license}</div>
          </div>))}
        {data.not_in_pack?.map((c) => (
          <div key={c.module} className="glass-inset border border-dashed border-amber-300/60 p-3 text-xs">
            <div className="font-semibold">{c.what} - not in the pack</div>
            <div className="text-ink-soft">{c.how}</div>
          </div>))}
      </div>
    </Glass>
  );
}

const cmd = (env: Env, cli: string) => cli.split(" && … ").map((c) =>
  env === "local" ? `python -m ${c}` : `docker compose exec app python -m ${c}`).join("\n");

function ActionRow({ a, module, env }: { a: Action; module: ModelModule; env: Env }) {
  const readOnly = useReadOnly();
  const catalog = useCatalog();
  const entry = catalog.data?.datasets.find((e) => e.module === module);
  const training = /train|retrain|setup|resume/.test(a.task);
  const blocked = training && (!entry || !entry.dataset.ready || entry.dataset.trainable === false || entry.status?.data === "needs_repair");
  const { jobs, start } = useJobs();
  const { showCommands } = usePrefs();
  const [cmdOpen, setCmdOpen] = useState(false);
  const busy = !!activeJob(jobs, module);
  const [vals, setVals] = useState<Record<string, number>>(Object.fromEntries((a.params ?? []).map((p) => [p.key, p.def])));
  const [err, setErr] = useState<string | null>(null);
  const cli = a.params ? a.cli.replace("--n-stars 300", `--n-stars ${vals.n_stars ?? 300}`).replace("--stars 300", `--stars ${vals.n_stars ?? 300}`) : a.cli;
  return (
    <div className="glass-inset space-y-2 p-3">
      <div className="flex flex-wrap items-center gap-2">
        <WriteButton operation={training ? "train" : "write"} title={blocked ? "Install and validate the required dataset first" : undefined} className={a.primary ? "btn-primary px-3 py-1.5 text-xs" : "btn-ghost px-3 py-1.5 text-xs"} disabled={busy || blocked}
          onClick={async () => { setErr(null); try { await start(a.task, vals); } catch (e) { setErr((e as Error).message); } }}>
          <Play className="h-3.5 w-3.5" /> {a.label}
        </WriteButton>
        {a.params?.map((p) => (
          <label key={p.key} className="flex items-center gap-1 text-[11px] text-ink-soft">
            {p.label}
            <input type="number" className="input w-20 px-2 py-1 text-xs" min={p.min} max={p.max} value={vals[p.key]}
              onChange={(e) => setVals({ ...vals, [p.key]: Number(e.target.value) })} />
          </label>))}
        {!readOnly && !showCommands && (
          <button onClick={() => setCmdOpen((o) => !o)} title="Show the terminal command"
            className={`ml-auto grid h-7 w-7 place-items-center rounded-lg transition ${cmdOpen ? "bg-slate-800 text-emerald-300" : "text-ink-faint hover:bg-white/70 hover:text-ink"}`}>
            <SquareTerminal className="h-3.5 w-3.5" />
          </button>)}
      </div>
      {a.note && !readOnly && <p className="text-[10px] leading-snug text-ink-faint">{a.note}</p>}
      {!readOnly && (showCommands || cmdOpen) && <div className="space-y-1">{cmd(env, cli).split("\n").map((c) => <CopyCommand key={c} cmd={c} always />)}</div>}
      {err && <p className="text-xs text-rose-700">{err}</p>}
    </div>
  );
}

export default function Datasets() {
  const readOnly = useReadOnly();
  const ds = useApi<Dataset[]>("/api/datasets");
  const models = useApi<Models>("/api/models");
  const { jobs } = useJobs();
  const { showCommands } = usePrefs();
  const [env, setEnv] = useState<Env>(() => (localStorage.getItem("smc-env") as Env) || "local");
  const [scanning, setScanning] = useState(false);
  useOnJobFinished(() => { ds.reload(); models.reload(); });
  const pickEnv = (e: Env) => { setEnv(e); try { localStorage.setItem("smc-env", e); } catch { /* private mode */ } };

  async function rescan() {
    setScanning(true);
    try { await api("/api/admin/rescan", { method: "POST" }); await Promise.all([models.reload(), ds.reload()]); } finally { setScanning(false); }
  }

  return (
    <Page title="Data & training"
      subtitle={readOnly ? "Read-only demo. Explore published data and results; downloads, training and uploads are available in Docker." : "Every dataset with its status, and a Train button for each model. Nothing downloads or trains until you click - with a release bundle everything works offline."}
      icon={<IconBadge gradient="from-emerald-400 to-teal-500"><Database className="h-6 w-6" /></IconBadge>}
      actions={
        <div className="flex flex-wrap items-center gap-2">
          {showCommands && <Segmented value={env} onChange={pickEnv} options={[
            { value: "local", label: <span className="flex items-center gap-1.5"><Laptop className="h-3.5 w-3.5" /> Local (backend/)</span> },
            { value: "docker", label: <span className="flex items-center gap-1.5"><Container className="h-3.5 w-3.5" /> Docker</span> }]} />}
          <WriteButton className="btn-ghost" onClick={rescan} disabled={scanning}>{scanning ? <Spinner /> : <RefreshCw className="h-4 w-4" />} Rescan</WriteButton>
        </div>}>
      {(ds.error || models.error) && <ErrorNotice message={(ds.error ?? models.error)!.message} />}
      <Catalog />
      <StarterCard />
      <details className="glass group p-4">
        <summary className="cursor-pointer select-none text-sm font-semibold">Advanced: individual data steps, model versions and evaluation reports</summary>
        <div className="mt-4">
      <div className="grid gap-4 lg:grid-cols-2">
        {ds.data?.map((d) => {
          const module = d.module as ModelModule;
          const meta = MODULE_META[module];
          const m = models.data?.[module];
          const moduleJobs = jobs.filter((j) => j.module === module).slice(0, 1);
          return (
            <Glass key={module} className="flex flex-col p-5">
              <div className="flex items-center justify-between">
                <span className={`chip ${meta.soft}`}>{meta.title}</span>
                {d.ready ? <CheckCircle2 className="h-5 w-5 text-emerald-500" /> : <TriangleAlert className="h-5 w-5 text-amber-500" />}
              </div>
              <h3 className="mt-3 text-lg font-bold">{d.name}</h3>
              <p className="mt-1 text-xs text-ink-soft">{d.description}</p>
              <a href={d.source_url} target="_blank" rel="noreferrer" className="mt-1 inline-flex items-center gap-1 text-xs font-semibold text-violet-600 hover:underline">
                Source <ExternalLink className="h-3 w-3" />
              </a>
              {Object.keys(d.stats).length > 0 && (
                <div className="mt-3 grid grid-cols-2 gap-2">
                  {Object.entries(d.stats).map(([k, v]) => (
                    <div key={k} className="glass-inset p-2.5"><div className="text-[10px] font-semibold uppercase tracking-wider text-ink-faint">{k}</div><div className="font-display text-lg font-bold">{Number(v).toLocaleString()}</div></div>
                  ))}
                </div>
              )}
              {moduleJobs.length > 0 && <div className="mt-3 space-y-2">{moduleJobs.map((j) => <JobCard key={j.id} job={j} />)}</div>}
              <div className="mt-4 space-y-2">
                <div className="label">Data & training</div>
                {ACTIONS[module].map((a) => <ActionRow key={a.task} a={a} module={module} env={env} />)}
                {module === "mars" && <div className="glass-inset p-3"><MarsModelUpload onDone={() => models.reload()} />
                  <p className="mt-1 text-[10px] text-ink-faint">The zip from the Colab/Kaggle notebook (models/mars/&lt;version&gt;/model.pt + manifest). It is checked before it becomes active.</p></div>}
              </div>
              <div className="mt-4 border-t border-white/70 pt-3">
                <div className="label mb-1.5">Installed model versions</div>
                {!m?.versions.length && <p className="text-xs text-ink-soft">None yet.</p>}
                {m?.versions.map((v) => (
                  <div key={v.version} className="flex items-center justify-between py-1 text-xs">
                    <span className="font-mono font-semibold">{v.version}{v.version === m.active && <span className="ml-2 chip bg-emerald-100 text-emerald-700">active</span>}</span>
                    <span className="text-ink-faint">{timeAgo(v.created_at)}</span>
                  </div>))}
              </div>
            </Glass>
          );
        })}
      </div>
        </div>
      </details>
    </Page>
  );
}
