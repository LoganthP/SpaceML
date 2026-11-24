import { AlertTriangle, Brain, X, CheckCircle2, CircleDashed, Database, Download, FolderOpen, HardDrive, PackageCheck, Play, RotateCcw, ShieldCheck, Square, Timer, Zap } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../lib/api";
import { ProgressBar, useJobs } from "./Jobs";
import { MarsModelUpload } from "./Setup";
import { Spinner } from "./ui";

/* ------------------------------------------------------------------ types */
interface CJob {
  id: string; task: string; title: string; status: string; progress: number | null; message: string | null; step: string | null;
  error: string | null; started_at: string | null; last_progress_at: string | null; waiting_for: string | null; created_at: string;
}
interface Action { task: string; label: string; params?: Record<string, unknown>; note?: string; downloads_first?: boolean; blocked?: string | null }
export interface CatalogEntry {
  id: string; module: string; title: string; error?: string;
  dataset: { ready: boolean; trainable?: boolean; blocked_reason?: string | null; files?: number; bytes?: number; summary: string; details?: Record<string, unknown> };
  model: { ready: boolean; version: string | null; name?: string; run_kind?: string; evaluated?: boolean; created_at?: string;
           headline?: { metric: string; model: number; baseline: number } | null } | null;
  train: Action[]; download: Action | null; resumable?: boolean; open_in_workbench?: string[];
  active_job: CJob | null; last_failure: CJob | null; mission?: string; notes?: string | null; setup_help?: string;
  status: { data: "installed" | "browse_only" | "missing" | "needs_repair"; repair?: string | null; data_label: string; model: "available" | "not_trained" | "not_applicable";
    next: { kind: "link" | "task" | "wait" | "none"; label: string; to?: string; task?: string; params?: Record<string, unknown> }; page: string;
    readiness?: Readiness; readiness_label?: string };
}
export type Readiness = "missing" | "installed" | "ready_to_train" | "model_available" | "analysis_only";
export const READINESS_LABEL: Record<Readiness, string> = { missing: "Missing", installed: "Installed", ready_to_train: "Ready to train",
  model_available: "Model available", analysis_only: "Analysis only" };
export const READINESS_TONE: Record<Readiness, string> = {
  missing: "bg-slate-200 text-slate-800 dark:bg-white/10 dark:text-slate-200",
  installed: "bg-amber-100 text-amber-900 dark:bg-amber-500/15 dark:text-amber-200",
  ready_to_train: "bg-sky-100 text-sky-900 dark:bg-sky-500/15 dark:text-sky-200",
  model_available: "bg-emerald-100 text-emerald-900 dark:bg-emerald-500/15 dark:text-emerald-200",
  analysis_only: "bg-violet-100 text-violet-900 dark:bg-violet-500/15 dark:text-violet-200" };
export interface CatalogSummaryData { datasets: number; installed: number; trainable: number; with_models: number; models_available: number;
  browse_only: number; needs_repair?: number; jobs_active: number; definitions: Record<string, string>;
  readiness?: Record<Readiness, number> }
export interface CatalogData {
  datasets: CatalogEntry[]; summary: CatalogSummaryData;
  release: { present: boolean; bundle_dir: string; bundle_version: string | null; total_bytes: number | null;
             installed: { bundle_version?: string; complete?: boolean; copied?: number; n_problems?: number; problems?: string[] };
             active_job: CJob | null; last_failure: CJob | null };
  heavy_jobs: { id: string; title: string; status: string }[]; free_bytes: number;
}

/** One shared catalog snapshot for the whole app (Overview, Data & Training, module and mission pages):
 *  same data, same refresh - polls every 2.5 s while a job runs, every 12 s otherwise, and immediately
 *  when any job finishes. */
const store = { data: null as CatalogData | null, error: null as string | null, subs: new Set<() => void>(), timer: 0, loading: false };
async function loadCatalog() {
  if (store.loading) return;
  store.loading = true;
  try { store.data = await api<CatalogData>("/api/catalog"); store.error = null; } catch (e) { store.error = (e as Error).message; }
  finally { store.loading = false; }
  store.subs.forEach((f) => f());
  schedule();
}
function schedule() {
  window.clearTimeout(store.timer);
  if (!store.subs.size) return;
  const busy = !!store.data && (store.data.datasets.some((d) => d.active_job) || !!store.data.release.active_job);
  store.timer = window.setTimeout(loadCatalog, busy ? 2500 : 12000);
}
if (typeof window !== "undefined") window.addEventListener("spaceml:job-finished", () => { void loadCatalog(); });
export const reloadCatalog = () => loadCatalog();

export function useCatalog() {
  const [, tick] = useState(0);
  useEffect(() => {
    const f = () => tick((x) => x + 1);
    store.subs.add(f);
    if (!store.data) void loadCatalog(); else schedule();
    return () => { store.subs.delete(f); if (!store.subs.size) window.clearTimeout(store.timer); };
  }, []);
  return { data: store.data, error: store.error, reload: reloadCatalog };
}

/* ------------------------------------------------------------------ helpers */
const fmtBytes = (b?: number | null) => (b == null ? "" : b >= 1e9 ? `${(b / 1e9).toFixed(2)} GB` : b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1e3))} kB`);
const ago = (iso?: string | null) => {
  if (!iso) return "";
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  return s < 60 ? `${Math.round(s)} s ago` : s < 3600 ? `${Math.round(s / 60)} min ago` : `${(s / 3600).toFixed(1)} h ago`;
};
const elapsed = (iso?: string | null) => {
  if (!iso) return "";
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  return s < 60 ? `${Math.round(s)} s` : s < 3600 ? `${Math.floor(s / 60)} min ${Math.round(s % 60)} s` : `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min`;
};

function useAct(reload: () => void) {
  const { start, refresh } = useJobs();
  const [pending, setPending] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const run = async (key: string, f: () => Promise<unknown>) => {
    setPending(key); setErr(null);
    try { await f(); refresh(); reload(); } catch (e) { setErr((e as Error).message); } finally { setPending(null); }
  };
  return {
    pending, err,
    startTask: (a: Action) => run(a.task + JSON.stringify(a.params ?? {}), () => start(a.task, a.params)),
    cancel: (id: string) => run("cancel", () => api(`/api/jobs/${id}/cancel`, { method: "POST" })),
    retry: (id: string) => run("retry", () => api(`/api/jobs/${id}/retry`, { method: "POST" })),
    dismiss: (id: string) => run("dismiss", () => api(`/api/jobs/${id}/dismiss`, { method: "POST" })),
  };
}

/* ------------------------------------------------------------------ job box */
function JobBox({ job, onCancel, busy }: { job: CJob; onCancel: () => void; busy: boolean }) {
  const queued = job.status === "queued";
  return (
    <div role="status" aria-live="polite" aria-label={`${job.title}: ${queued ? "queued" : `running ${Math.round((job.progress ?? 0) * 100)}%`}`}
      className={`space-y-1.5 rounded-xl border p-2.5 text-xs ${queued ? "border-amber-300/60 bg-amber-50/70 text-amber-950 dark:border-amber-400/20 dark:bg-amber-500/10 dark:text-amber-100" : "border-violet-300/60 bg-violet-50/70 text-violet-950 dark:border-violet-400/20 dark:bg-violet-500/10 dark:text-violet-100"}`}>
      <div className="flex items-center gap-2">
        {queued ? <Timer className="h-3.5 w-3.5" /> : <Spinner className="h-3.5 w-3.5" />}
        <b className="truncate">{job.title}</b>
        <button className="btn-ghost ml-auto px-2 py-0.5 text-[11px]" onClick={onCancel} disabled={busy} aria-label={`Cancel ${job.title}`}><Square className="h-3 w-3" aria-hidden="true" /> Cancel</button>
      </div>
      {queued ? (
        <p>Waiting for <b>{job.waiting_for ?? "another training job"}</b> to finish - one training job runs at a time so the dashboard stays responsive.</p>
      ) : (
        <>
          <p className="truncate">{job.message || job.step || "starting…"}</p>
          <ProgressBar value={job.progress} />
          <p className="flex flex-wrap gap-x-3 opacity-75">
            <span>{job.progress ? `${Math.round(job.progress * 100)}%` : "starting"}</span>
            <span>running {elapsed(job.started_at)}</span>
            <span>last update {ago(job.last_progress_at || job.started_at)}</span>
          </p>
        </>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ one dataset card */
function Pill({ tone, label, sub }: { tone: "ok" | "warn" | "off"; label: string; sub?: string }) {
  const cls = tone === "ok" ? "bg-emerald-50 text-emerald-800 dark:bg-emerald-500/10 dark:text-emerald-200"
    : tone === "warn" ? "bg-amber-50 text-amber-900 dark:bg-amber-500/10 dark:text-amber-100"
      : "bg-slate-100 text-slate-700 dark:bg-white/5 dark:text-slate-200";
  return (
    <div className={`flex min-w-0 items-center gap-1.5 rounded-xl px-2.5 py-1.5 text-xs ${cls}`}>
      {tone === "ok" ? <CheckCircle2 className="h-3.5 w-3.5 shrink-0" aria-hidden="true" /> : tone === "warn" ? <AlertTriangle className="h-3.5 w-3.5 shrink-0" aria-hidden="true" /> : <CircleDashed className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />}
      <span className="shrink-0 whitespace-nowrap font-semibold">{label}</span>{sub && <span className="min-w-0 truncate opacity-80" title={sub}>· {sub}</span>}
    </div>
  );
}

export function DatasetCard({ e, reload, compact = false }: { e: CatalogEntry; reload: () => void; compact?: boolean }) {
  const act = useAct(reload);
  const nav = useNavigate();
  const job = e.active_job;
  const m = e.model;
  const st = e.status;
  const fail = e.last_failure;
  const soft = fail?.status === "interrupted" || /^(.*: )?(Not available right now|source_unavailable)|rate limit/i.test(String(fail?.error ?? ""));
  const failText = (fail?.error ?? "").replace(/\s*Run:\s*.+$/, "");
  const trainBlocked = st?.data === "missing" ? "install the data first" : st?.data === "browse_only" ? (e.dataset.blocked_reason ?? "not enough data to train") : null;
  const openScene = async (name: string) => {
    const r = await api<{ id: string }>(`/api/workbench/samples/${encodeURIComponent(name)}`, { method: "POST" });
    nav(`/workbench?dataset=${r.id}&auto=1`);
  };
  const next = st?.next;
  const primaryTask = next?.kind === "task" ? next.task : null;
  const primary = !job && next && next.kind !== "wait" && (
    next.kind === "link" ? <Link className="btn-primary px-3 py-1.5 text-xs" to={next.to!}><Play className="h-3.5 w-3.5" aria-hidden="true" /> {next.label}</Link>
      : next.kind === "task" ? <button className="btn-primary px-3 py-1.5 text-xs" disabled={!!act.pending} onClick={() => act.startTask({ task: next.task!, label: next.label, params: next.params })}>
          {act.pending ? <Spinner className="h-3.5 w-3.5" /> : <Zap className="h-3.5 w-3.5" aria-hidden="true" />} {next.label}</button>
        : <span className="text-xs text-ink-soft">{next.label}</span>);
  const modelSub = m?.ready ? (m.headline ? `${m.headline.metric} ${Number(m.headline.model).toFixed(2)} vs ${Number(m.headline.baseline).toFixed(2)}` : m.evaluated ? m.version ?? "" : "not evaluated yet") : undefined;
  return (
    <div className="glass-inset space-y-2.5 p-3.5">
      <div className="flex items-start gap-2">
        <Database className="mt-0.5 h-4 w-4 shrink-0 text-violet-500" aria-hidden="true" />
        <div className="min-w-0 flex-1">
          <Link to={st?.page ?? "/"} className="font-semibold leading-tight hover:underline">{e.title}</Link>
          <div className="text-[11px] text-ink-soft">{e.dataset.summary}{e.dataset.bytes ? ` · ${fmtBytes(e.dataset.bytes)} on disk` : ""}</div>
        </div>
      </div>
      {st?.readiness && <span className={`chip ${READINESS_TONE[st.readiness]}`} title={st.data_label}>{READINESS_LABEL[st.readiness]}</span>}
      <div className="grid gap-1.5 sm:grid-cols-2">
        <Pill tone={st?.data === "installed" ? "ok" : st?.data === "browse_only" || st?.data === "needs_repair" ? "warn" : "off"} label={`Data: ${st?.data_label ?? (e.dataset.ready ? "Installed" : "Not installed")}`} />
        {m === null ? <Pill tone="ok" label="No model needed" sub="analysis only" /> :
          <Pill tone={m?.ready ? "ok" : "off"} label={m?.ready ? `Model: ${m.run_kind === "quick" ? "quick run" : "trained"}` : "Model: not trained yet"} sub={modelSub} />}
      </div>
      {e.error && <p className="text-[11px] text-rose-700 dark:text-rose-300">{e.error}</p>}
      {job && <JobBox job={job} onCancel={() => act.cancel(job.id)} busy={act.pending === "cancel"} />}
      {!job && fail && (
        <div role="alert" className={`space-y-1 rounded-xl border p-2.5 text-xs ${soft
          ? "border-amber-200/70 bg-amber-50/80 text-amber-900 dark:border-amber-400/20 dark:bg-amber-500/10 dark:text-amber-100"
          : "border-rose-200/70 bg-rose-50/80 text-rose-900 dark:border-rose-400/20 dark:bg-rose-500/10 dark:text-rose-100"}`}>
          <div className="flex items-center gap-1.5 font-semibold"><AlertTriangle className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
            <span className="min-w-0 truncate">{fail.title} - {soft ? "stopped by a restart" : fail.status.replace("_", " ")}</span>
            <button className="btn-ghost ml-auto px-2 py-0.5 text-[11px]" onClick={() => act.retry(fail.id)} disabled={!!act.pending}><RotateCcw className="h-3 w-3" aria-hidden="true" /> Retry</button>
            <button className="btn-ghost px-1.5 py-0.5 text-[11px]" title="Dismiss" aria-label="Dismiss this message" onClick={() => act.dismiss(fail.id)} disabled={!!act.pending}><X className="h-3 w-3" aria-hidden="true" /></button>
          </div>
          <p className="break-words opacity-90">{soft ? "The app restarted while this job ran. Finished parts were kept; press Retry to continue, or dismiss this note." : failText}</p>
        </div>
      )}
      {st?.repair && <p role="status" className="text-[11px] text-amber-900 dark:text-amber-200">{st.repair}</p>}
      {st?.data === "browse_only" && e.dataset.blocked_reason && <p className="text-[11px] text-amber-900 dark:text-amber-200">{e.dataset.blocked_reason}</p>}
      {!compact && !job && (
        <div className="flex flex-wrap items-center gap-1.5">
          {primary}
          {e.train.filter((a) => !(primaryTask === a.task && JSON.stringify(next?.params ?? {}) === JSON.stringify(a.params ?? {}))).map((a, i) => (
            <button key={i} className="btn-ghost px-3 py-1.5 text-xs" title={trainBlocked ? `Needs: ${trainBlocked}` : a.note}
              disabled={!!trainBlocked || !!act.pending} onClick={() => act.startTask(a)}>
              <Brain className="h-3.5 w-3.5" aria-hidden="true" /> {a.label}
            </button>))}
          {e.download && primaryTask !== e.download.task && <button className="btn-ghost px-3 py-1.5 text-xs" disabled={!!act.pending} onClick={() => act.startTask(e.download!)}><Download className="h-3.5 w-3.5" aria-hidden="true" /> {e.download.label}</button>}
          {e.open_in_workbench?.map((sc) => <button key={sc} className="btn-ghost px-3 py-1.5 text-xs" onClick={() => openScene(sc)}><FolderOpen className="h-3.5 w-3.5" aria-hidden="true" /> Open {sc.replace(".zip", "")} in Workbench</button>)}
          {st?.next.kind === "link" && st.next.to !== st.page && st.data !== "missing" && <Link className="btn-ghost px-3 py-1.5 text-xs" to={st.page}>Open page</Link>}
          {e.module === "mars" && e.id === "ai4mars" && <MarsModelUpload onDone={reload} />}
        </div>
      )}
      {!compact && !job && (e.train.length > 0 || e.notes || e.setup_help) && (
        <details className="text-[11px] text-ink-soft">
          <summary className="cursor-pointer font-semibold">Setup help</summary>
          <ul className="mt-1 space-y-0.5">
            {e.train.map((a, i) => <li key={i}><b>{a.label}:</b> {trainBlocked ? `not available yet - ${trainBlocked}` : a.note}</li>)}
            {e.notes && <li>{e.notes}</li>}
            {e.setup_help && <li className="break-all font-mono">{e.setup_help}</li>}
          </ul>
        </details>)}
      {act.err && <p role="alert" className="text-[11px] text-rose-700 dark:text-rose-300">{act.err}</p>}
    </div>
  );
}

/** The same three numbers everywhere (Overview, Data & Training): installed datasets, trainable datasets, available models. */
export function StatusCounts({ s }: { s: CatalogSummaryData }) {
  const item = (n: number, of: number, label: string, title: string) => (
    <span className="chip bg-white/80 text-ink dark:bg-white/10" title={title}><b className="mr-1">{n}/{of}</b>{label}</span>);
  void item;
  const r = s.readiness;
  const defs: Record<Readiness, string> = { missing: s.definitions.missing, installed: s.definitions.installed_only, ready_to_train: s.definitions.ready_to_train,
    model_available: s.definitions.model_available, analysis_only: s.definitions.analysis_only };
  return (
    <span className="flex flex-wrap items-center gap-1.5" aria-label={`${s.datasets} datasets by readiness`}>
      {r && (Object.keys(READINESS_LABEL) as Readiness[]).map((k) => (
        <span key={k} className={`chip ${READINESS_TONE[k]}`} title={defs[k]}><b className="mr-1">{r[k] ?? 0}</b>{READINESS_LABEL[k]}</span>))}
      {(s.needs_repair ?? 0) > 0 && <span className="chip bg-amber-100 text-amber-900">{s.needs_repair} need re-import</span>}
    </span>
  );
}

/* ------------------------------------------------------------------ release card */
export function ReleaseCard({ c, reload }: { c: CatalogData; reload: () => void }) {
  const act = useAct(reload);
  const r = c.release;
  const job = r.active_job;
  return (
    <div className="glass-inset space-y-2 p-3.5">
      <div className="flex flex-wrap items-center gap-2">
        <PackageCheck className="h-4 w-4 text-violet-500" />
        <b>Release bundle</b>
        {r.present ? <span className="chip bg-violet-100 text-violet-800">v{r.bundle_version} · {fmtBytes(r.total_bytes)}</span>
          : <span className="chip bg-slate-100 text-slate-600 dark:bg-white/10 dark:text-slate-300">not used</span>}
        {r.installed?.complete && r.installed.bundle_version === r.bundle_version &&
          <span className="chip bg-emerald-100 text-emerald-800"><ShieldCheck className="h-3 w-3" /> installed · checksums verified</span>}
        <span className="ml-auto flex items-center gap-1 text-[11px] text-ink-faint"><HardDrive className="h-3 w-3" /> {fmtBytes(c.free_bytes)} free</span>
        {r.present && !job && <button className="btn-ghost px-2.5 py-1 text-[11px]" disabled={!!act.pending} onClick={() => act.startTask({ task: "release.install", label: "Install" })}>
          <RotateCcw className="h-3 w-3" /> {r.installed?.complete ? "Re-verify" : "Install"}</button>}
      </div>
      {job && <JobBox job={job} onCancel={() => act.cancel(job.id)} busy={act.pending === "cancel"} />}
      {r.installed?.n_problems ? <p className="text-[11px] text-rose-600">{r.installed.n_problems} file(s) in the bundle are missing or corrupt, e.g. {r.installed.problems?.[0]}</p> : null}
      <p className="text-[11px] text-ink-soft">
        {r.present ? "Datasets from the bundle are copied into the app's data volume once (only missing or changed files) and work offline. The bundle itself is never modified."
          : <>No offline bundle is installed, which is fine with internet: a missing dataset downloads by itself when you press its <b>Download</b> or <b>Train</b> button. For machines without internet, prepare a bundle once (README → Release bundle).</>}
      </p>
      {act.err && <p className="text-[11px] text-rose-600">{act.err}</p>}
    </div>
  );
}

/* ------------------------------------------------------------------ full catalog */
export function Catalog() {
  const { data, error, reload } = useCatalog();
  if (error && !data) return <p className="text-sm text-rose-600">{error}</p>;
  if (!data) return <div className="flex items-center gap-2 text-sm text-ink-soft"><Spinner /> Reading datasets…</div>;
  return (
    <div className="glass hud space-y-3 p-5">
      <div className="flex flex-wrap items-center gap-2">
        <Database className="h-4 w-4 text-violet-500" />
        <div className="kicker">Datasets &amp; models</div>
        <StatusCounts s={data.summary} />
        {data.heavy_jobs.length > 0 && <span className="chip bg-violet-100 text-violet-700"><Spinner className="h-3 w-3" /> {data.heavy_jobs.length} training job(s)</span>}
      </div>
      <p className="text-xs text-ink-soft">Each card shows its <b>data</b> and its <b>model</b> separately, and the highlighted button is the next step.
        Nothing downloads or trains by itself. <b>Installed</b> = on this machine and explorable · <b>ready to train</b> = meets the trainer's requirements ·
        <b> model available</b> = can produce results now. Training runs one job at a time and keeps the last working model until a new one is finished.</p>
      <ReleaseCard c={data} reload={reload} />
      <div className="grid gap-3 lg:grid-cols-2 xl:grid-cols-3">
        {data.datasets.map((e) => <DatasetCard key={e.id} e={e} reload={reload} />)}
      </div>
    </div>
  );
}

/** Compact status for a module page: shown only when the data or model is missing, or a job is active. */
export function ModuleStatus({ ids, always = false }: { ids: string[]; always?: boolean }) {
  const { data, reload } = useCatalog();
  if (!data) return null;
  const es = data.datasets.filter((d) => ids.includes(d.id) && (always || !d.dataset.ready || (d.model && !d.model.ready) || d.active_job || d.last_failure));
  if (!es.length) return null;
  return <div className="grid gap-3 lg:grid-cols-2">{es.map((e) => <DatasetCard key={e.id} e={e} reload={reload} />)}</div>;
}

export function CatalogSummary() {
  const { data } = useCatalog();
  if (!data) return <div className="glass flex items-center gap-2 p-3 text-sm text-ink-soft" role="status"><Spinner /> Checking datasets and models…</div>;
  const jobs = data.datasets.filter((d) => d.active_job);
  const next = data.datasets.filter((d) => d.status?.next.kind === "task" && d.status.model === "not_trained" && d.status.data === "installed");
  return (
    <div className="glass flex w-full flex-wrap items-center gap-2 p-3 text-sm">
      <Database className="h-4 w-4 text-violet-500" aria-hidden="true" />
      <b>Datasets &amp; models:</b>
      <StatusCounts s={data.summary} />
      {jobs.map((d) => <span key={d.id} className="chip bg-violet-100 text-violet-800"><Spinner className="h-3 w-3" /> {d.active_job!.title} {d.active_job!.progress ? `${Math.round(d.active_job!.progress * 100)}%` : ""}</span>)}
      {next.length > 0 && <span className="text-xs text-ink-soft">Next: train {next.slice(0, 3).map((d) => d.title.split(" ")[0] === "ESA" ? d.title : d.title.split(/[·(]/)[0].trim()).join(", ")}</span>}
      <Link to="/datasets" className="ml-auto text-xs font-semibold text-violet-700 underline dark:text-violet-300">Open Data &amp; Training</Link>
    </div>
  );
}
