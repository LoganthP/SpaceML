import { WriteButton, FileInput, useCapabilities, useReadOnly, useCanRun, READ_ONLY_HINT } from "../../lib/capabilities";
import { CheckCircle2, CircleDashed, ClipboardCheck, Download, ExternalLink, Gauge, Inbox, Play, Satellite, Search, Table2, Upload } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import Plot from "../LazyPlot";
import { TelescopeLive } from "./TelescopeLive";
import { reloadCatalog, useCatalog } from "../Catalog";
import { useJobs, useOnJobFinished } from "../Jobs";
import { ErrorNotice, Glass, Kpi, Notice, Segmented, Spinner } from "../ui";
import { api, createRun, useApi, useRun } from "../../lib/api";

/* ---------------------------------------------------------------- types */
export interface MissionStatus {
  id: string; name: string; agency: string; family: "segments" | "pointing"; summary: string;
  item_label: string; time_basis: string; trainable: boolean; train_label: string | null; detect: string | null;
  evaluate: string | null;
  source: { title: string; doi: string | null; url: string; authors: string; license: string; redistribution: string; reference: string };
  repair?: { reason: string; label: string; task: string; params: Record<string, unknown> } | null;
  install?: { download: { label: string; size: string; fields: { name: string; label: string; placeholder?: string; help?: string; secret?: boolean }[] } | null;
    upload: { accept: string; hint: string } } | null;
  dataset: { installed: boolean; state: string; scope: string | null; counts: Record<string, number> | null;
    date_range: (string | null)[] | null; items: number | null; events: number | null; bytes: number | null;
    source_version: string | null; next_action: string | null };
  model: { ready: boolean; stale?: boolean; version: string | null; run_kind: string | null; headline: { metric: string; model: number; baseline: number } | null };
  versions?: { id: string; active: boolean; label: string; items: number | null; date_range: (string | null)[] | null; prepared_at: string | null }[];
  auto_install?: { label: string; size: string; automatic: boolean } | null;
  manifest?: { channels?: { id: string; name: string; unit: string | null }[]; detect_columns?: string[]; split_protocol?: string;
    time_is_epoch?: boolean; source?: { version?: string; doi?: string } } | null;
}
interface Item {
  id: string; label: string; n: number; t0: number | null; t1: number | null; columns: string[]; gaps: number;
  missing: Record<string, number>; split?: string; anomaly?: number; published_label?: string; channel?: string;
  units?: Record<string, string | null>; guiding?: Record<string, unknown>; warnings?: string[]; guiding_modes?: string[];
  cadence_s?: number | null; sampling_s?: number; events?: number;
}
interface Series {
  n_total: number; n_returned: number; downsampled: boolean; time_basis: string; time_is_epoch: boolean;
  t: number[]; cols: Record<string, (number | null)[]>; units: Record<string, string | null>;
  missing: Record<string, number>; gaps: { start: number; end: number; seconds: number }[];
  label?: number[]; labeled_samples?: number; events: { id: string; start: number; end: number; category: string }[];
}
interface Interval { start: number; end: number; n: number; score?: number | null }
interface ScoreResult {
  action: "score"; prob?: number; pred?: number; label?: number; split?: string;
  model_intervals?: Interval[]; baseline_intervals?: Interval[]; detector?: string; review_items: unknown[];
  context?: { guiding?: Record<string, unknown>; warnings?: string[]; guiding_modes?: string[] } | null;
}
interface ClassMetrics { precision: number; recall: number; f1: number; support: number }
interface EvalBlock {
  name: string; confusion_matrix?: { labels: string[]; matrix: number[][] }; per_class?: Record<string, ClassMetrics>;
  false_alarms?: number; missed_anomalies?: number; detected_anomalies?: number; accuracy?: number; f1_anomaly?: number; roc_auc?: number;
  events?: number; detected_events?: number; event_recall?: number | null; alarms?: number; alarm_precision?: number | null; threshold?: number;
}
interface EvalResult { action: "evaluate"; metrics: { headline: { metric: string; model: number; baseline: number }; eval_split: string;
  model: EvalBlock; baseline: EvalBlock; protocol?: string; labels?: string; alerts?: Record<string, number> }; review_items: unknown[] }

/* ---------------------------------------------------------------- helpers */
const fmtTime = (t: number, epoch: boolean) => (epoch ? new Date(t * 1000).toISOString().replace("T", " ").slice(0, 19) + " UTC" : `${t.toFixed(1)} s`);
const pct = (x: number | null | undefined) => (x == null ? "n/a" : `${(x * 100).toFixed(1)}%`);
const num3 = (x: number | null | undefined) => (x == null ? "n/a" : typeof x === "number" ? (Number.isInteger(x) ? String(x) : x.toFixed(3)) : String(x));

/* ---------------------------------------------------------------- workspace */
export default function MissionWorkspace({ mission, runId, onRun }: { mission: string; runId: string | null; onRun: (id: string | null) => void }) {
  const info = useApi<MissionStatus>(`/api/missions/${mission}`, [mission]);
  const [item, setItem] = useState<string | null>(null);
  const [announce, setAnnounce] = useState("");
  useOnJobFinished(() => info.reload());
  useEffect(() => { setItem(null); }, [mission]);

  const m = info.data;
  if (info.error) return <ErrorNotice message={info.error.message} />;
  if (!m) return <div className="flex items-center gap-2 p-6 text-sm text-ink-soft" role="status"><Spinner /> Loading {mission}…</div>;
  return (
    <div className="space-y-4">
      <p className="sr-only" aria-live="polite">{announce}</p>
      <Glass className="hud flex flex-wrap items-start gap-4 p-5">
        <div className="grid h-11 w-11 shrink-0 place-items-center rounded-2xl bg-gradient-to-br from-indigo-500 to-violet-500 text-white" aria-hidden="true"><Satellite className="h-5 w-5" /></div>
        <div className="min-w-0 flex-1 space-y-1">
          <h2 className="font-display text-xl font-bold">{m.name} <span className="text-sm font-medium text-ink-soft">· {m.agency}</span></h2>
          <p className="text-sm text-ink-soft">{m.summary}</p>
          <p className="text-xs text-ink-soft"><b>Time basis:</b> {m.dataset.installed ? (info.data?.manifest as { time_basis?: string } | undefined)?.time_basis ?? m.time_basis : m.time_basis}</p>
          <p className="text-xs text-ink-soft"><b>Source:</b> <a className="underline decoration-dotted" href={m.source.url} target="_blank" rel="noreferrer">{m.source.title}</a>
            {m.dataset.source_version ? ` · ${m.dataset.source_version}` : ""}{m.source.doi ? ` · DOI ${m.source.doi}` : ""} · {m.source.license}</p>
        </div>
        <div className="flex flex-wrap gap-2 text-xs">
          <span className={`chip ${m.dataset.installed ? "bg-emerald-100 text-emerald-800" : "bg-slate-200 text-slate-800"}`}>Data: {m.dataset.installed ? "installed" : "not installed"}</span>
          {m.trainable ? <span className={`chip ${m.model.ready ? "bg-emerald-100 text-emerald-800" : "bg-amber-100 text-amber-900"}`}>Model: {m.model.ready ? `${m.model.version}` : "not trained"}</span>
            : <span className="chip bg-slate-200 text-slate-800">{m.dataset.installed ? "Analysis only (no labels to train on)" : "Data missing"}</span>}
        </div>
      </Glass>

      {m.repair && <RepairBanner repair={m.repair} />}
      <StepStrip m={m} item={item} onChanged={info.reload} />

      {!m.dataset.installed ? <InstallPanel m={m} /> : (
        <>
          <div className="grid gap-3" style={{ gridTemplateColumns: `repeat(auto-fit, minmax(150px, 1fr))` }}>
            {Object.entries(m.dataset.counts ?? {}).slice(0, 6).map(([k, v]) => <Kpi key={k} label={k.replace(/_/g, " ")} value={typeof v === "number" ? v.toLocaleString() : String(v)} />)}
          </div>
          <div id="analyze" className="grid scroll-mt-28 grid-cols-1 gap-4 lg:grid-cols-[minmax(260px,320px)_minmax(0,1fr)]">
            <ItemPicker m={m} selected={item} onSelect={(id) => { setItem(id); setAnnounce(`Selected ${m.item_label} ${id}`); }} />
            <div className="min-w-0 space-y-4">
              {item ? <Viewer m={m} itemId={item} onAnnounce={setAnnounce} runId={runId} onRun={onRun} />
                : <Glass className="p-8 text-center text-sm text-ink-soft" role="status">Select a {m.item_label} on the left to replay its telemetry.</Glass>}
              {m.trainable && <Evaluation m={m} runId={runId} onRun={onRun} onAnnounce={setAnnounce} />}
            </div>
          </div>
          <details className="glass p-4 text-sm">
            <summary className="cursor-pointer font-semibold">More data options: specific files or your own upload</summary>
            <div className="mt-3"><InstallPanel m={m} embedded /></div>
          </details>
        </>)}
      {(mission === "hubble" || mission === "webb") && <TelescopeLive key={mission} mission={mission} />}
    </div>
  );
}

const fmtHeadline = (x: number) => (Number.isInteger(Number(x)) ? String(Number(x)) : Number(x).toFixed(3));

/* ---------------------------------------------------------------- Data → Train → Analyze → Review */
function StepStrip({ m, item, onChanged }: { m: MissionStatus; item: string | null; onChanged: () => void }) {
  const cat = useCatalog();
  const e = cat.data?.datasets.find((d) => d.id === `mission-${m.id}`);
  const { start } = useJobs();
  const [err, setErr] = useState<string | null>(null);
  const q = useApi<{ items: { mission?: string }[]; pending: number }>(m.dataset.installed ? "/api/reviews/queue?module=missions&limit=500" : null, [m.id, m.model.version]);
  const pending = (q.data?.items ?? []).filter((x) => x.mission === m.id).length;
  const job = e?.active_job;
  const step = (n: number, title: string, done: boolean, body: React.ReactNode) => (
    <li className={`glass-inset flex min-w-0 flex-col gap-1.5 p-3 ${done ? "" : ""}`}>
      <div className="flex items-center gap-1.5 text-xs font-semibold">
        {done ? <CheckCircle2 className="h-4 w-4 text-emerald-600" aria-hidden="true" /> : <CircleDashed className="h-4 w-4 text-ink-faint" aria-hidden="true" />}
        <span>{n}. {title}</span><span className="sr-only">{done ? "(done)" : "(to do)"}</span>
      </div>
      <div className="text-xs text-ink-soft">{body}</div>
    </li>);
  const train = async (mode: "quick" | "full" = "quick") => {
    setErr(null);
    try { await start(`missions.${m.id}.train`, { mode }); reloadCatalog(); } catch (x) { setErr((x as Error).message); }
  };
  const newest = async () => {
    setErr(null);
    try { await start(`missions.${m.id}.install`, {}); reloadCatalog(); } catch (x) { setErr((x as Error).message); }
  };
  const switchTo = async (vid: string) => {
    setErr(null);
    try { await api(`/api/missions/${m.id}/versions/${encodeURIComponent(vid)}/activate`, { method: "POST" }); reloadCatalog(); onChanged(); }
    catch (x) { setErr((x as Error).message); }
  };
  const readOnly = useCapabilities()?.capabilities.write === false;
  const vers = m.versions ?? [];
  const auto = m.auto_install;
  return (
    <nav aria-label={`${m.name} workflow`}>
      <ol className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
        {step(1, "Data", m.dataset.installed, m.dataset.installed
          ? <div className="space-y-1.5">
              <div>{(m.dataset.items ?? 0).toLocaleString()} {m.item_label}s{m.dataset.date_range?.[0] ? ` · ${String(m.dataset.date_range[0]).slice(0, 10)} → ${String(m.dataset.date_range[1] ?? "").slice(0, 10)}` : ""}{m.dataset.source_version ? ` · ${m.dataset.source_version}` : ""}</div>
              {vers.length > 1 && (
                <label className="flex items-center gap-1.5"><span className="shrink-0">Version</span>
                  <select aria-label="Data version" className="input min-w-0 flex-1 px-1.5 py-0.5 text-[11px]" value={vers.find((v) => v.active)?.id}
                    disabled={!!job || readOnly} title={readOnly ? READ_ONLY_HINT : undefined} onChange={(e) => switchTo(e.target.value)}>
                    {vers.map((v) => <option key={v.id} value={v.id}>{v.active ? "● " : ""}{v.label || v.id}</option>)}
                  </select>
                </label>)}
              {auto && !job && <WriteButton className="btn-ghost px-2 py-1 text-[11px]" onClick={newest} title={`${auto.label} (${auto.size}); the current data is kept as an earlier version`}>
                <Download className="h-3 w-3" aria-hidden="true" /> Get newest data</WriteButton>}
            </div>
          : job ? <span role="status"><Spinner className="mr-1 inline h-3 w-3" />{job.title} {job.progress ? `${Math.round(job.progress * 100)}%` : ""}</span>
            : readOnly ? "Available in Docker" : <a href="#install" className="btn-primary mt-1 px-3 py-1.5 text-xs"><Download className="h-3.5 w-3.5" aria-hidden="true" /> Install data</a>)}
        {step(2, "Train", !m.trainable || m.model.ready, !m.dataset.installed ? "Data missing" : !m.trainable ? "Not needed - no labels to train on; rolling detection flags intervals."
          : job ? <span role="status"><Spinner className="mr-1 inline h-3 w-3" />{job.title} {job.progress ? `${Math.round(job.progress * 100)}%` : ""}</span>
            : !m.dataset.installed ? "Available after the data is installed."
              : <div className="space-y-1.5">
                  {m.model.ready && <div>Model {m.model.version}{m.model.headline ? ` · ${m.model.headline.metric}: ${fmtHeadline(m.model.headline.model)} vs baseline ${fmtHeadline(m.model.headline.baseline)}` : ""}</div>}
                  {m.model.stale && <div className="text-amber-800 dark:text-amber-300">The last model was trained on another data version - train again on this one.</div>}
                  <div className="flex flex-wrap gap-1.5">
                    <WriteButton operation="train" disabled={!!m.repair} className={`${m.model.ready ? "btn-ghost" : "btn-primary"} px-2.5 py-1 text-[11px]`} onClick={() => train("quick")}><Play className="h-3 w-3" aria-hidden="true" /> Quick train</WriteButton>
                    <WriteButton operation="train" disabled={!!m.repair} className="btn-ghost px-2.5 py-1 text-[11px]" onClick={() => train("full")}>Full training</WriteButton>
                  </div>
                </div>)}
        {step(3, "Analyze", !!item, !m.dataset.installed ? "Replay and scoring open once data is installed."
          : <a href="#analyze" className="underline">{item ? `Viewing ${m.item_label} ${item}` : `Pick a ${m.item_label}`}</a>)}
        {step(4, "Review", false, m.dataset.installed
          ? <Link to="/review?module=missions" className="underline">{pending ? `${pending} item(s) waiting in Review Studio` : "Open Review Studio"}</Link>
          : "Flagged results appear here.")}
      </ol>
      {err && <p role="alert" className="mt-2 text-xs text-rose-700 dark:text-rose-300">{err}</p>}
    </nav>
  );
}

/* ---------------------------------------------------------------- Known-bad imported data: one-click fix */
function RepairBanner({ repair }: { repair: NonNullable<MissionStatus["repair"]> }) {
  const { start } = useJobs();
  const [state, setState] = useState<"idle" | "busy" | "started" | string>("idle");
  const go = async () => {
    setState("busy");
    try { await start(repair.task, repair.params); reloadCatalog(); setState("started"); }
    catch (e) { setState((e as Error).message); }
  };
  return (
    <Glass className="flex flex-wrap items-center gap-3 border border-amber-300/60 p-4" role="status">
      <p className="min-w-0 flex-1 text-sm text-amber-900 dark:text-amber-200">{repair.reason}</p>
      {state === "started" ? <span className="text-sm">Re-import started - this page updates when it finishes.</span>
        : <WriteButton className="btn-primary px-3 py-1.5 text-sm" disabled={state === "busy"} onClick={go}><Download className="h-4 w-4" aria-hidden="true" /> {repair.label}</WriteButton>}
      {!["idle", "busy", "started"].includes(state) && <p role="alert" className="w-full text-xs text-rose-700 dark:text-rose-300">{state}</p>}
    </Glass>
  );
}

/* ---------------------------------------------------------------- Install data: download or upload */
function InstallPanel({ m, embedded = false }: { m: MissionStatus; embedded?: boolean }) {
  const readOnly = useReadOnly();
  const cat = useCatalog();
  const spec = m.install;
  const { start } = useJobs();
  const [vals, setVals] = useState<Record<string, string>>({});
  const [files, setFiles] = useState<File[]>([]);
  const [busy, setBusy] = useState<null | "download" | "upload">(null);
  const [pct, setPct] = useState<number | null>(null);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const run = async (params: Record<string, unknown>) => {
    await start(`missions.${m.id}.install`, params);
    reloadCatalog();
    setMsg({ ok: true, text: "Install job started - progress is shown on the card above. The page updates by itself when it finishes." });
  };
  const download = async () => {
    setBusy("download"); setMsg(null);
    try { await run({ ...vals }); } catch (e) { setMsg({ ok: false, text: (e as Error).message }); } finally { setBusy(null); }
  };
  const upload = async () => {
    if (!files.length) return;
    setBusy("upload"); setMsg(null); setPct(0);
    try {
      const fd = new FormData();
      files.forEach((f) => fd.append("files", f));
      const res = await new Promise<{ folder: string }>((ok, bad) => {
        const x = new XMLHttpRequest();
        x.open("POST", `/api/missions/${m.id}/upload`);
        x.upload.onprogress = (ev) => ev.lengthComputable && setPct(ev.loaded / ev.total);
        x.onload = () => { try { const j = JSON.parse(x.responseText); x.status < 300 ? ok(j) : bad(new Error(j.detail ?? x.statusText)); } catch { bad(new Error(x.statusText)); } };
        x.onerror = () => bad(new Error("upload failed - is the app still running?"));
        x.send(fd);
      });
      await run({ source: res.folder });
      setFiles([]);
    } catch (e) { setMsg({ ok: false, text: (e as Error).message }); } finally { setBusy(null); setPct(null); }
  };
  if (readOnly) return <Notice title="Read-only demo">{m.dataset.installed ? "Published data can be explored above." : "Data missing: no validated dataset is published."} Installation and uploads are available in Docker.</Notice>;
  if (!spec) return <Notice title="Data missing">Import and validate real source data before enabling this mission.</Notice>;
  const dl = spec.download;
  const auto = m.auto_install;
  const fieldsOf = (names: string[]) => (dl?.fields ?? []).filter((f) => names.includes(f.name));
  const field = (f: NonNullable<typeof dl>["fields"][number]) => (
    <div key={f.name}>
      <label className="block text-xs font-semibold" htmlFor={`f-${m.id}-${f.name}`}>{f.label}</label>
      <input id={`f-${m.id}-${f.name}`} type={f.secret ? "password" : "text"} autoComplete="off" value={vals[f.name] ?? ""} placeholder={f.placeholder}
        onChange={(e) => setVals((v) => ({ ...v, [f.name]: e.target.value }))}
        className="mt-1 w-full rounded-xl border border-slate-300 bg-white/80 px-2 py-1.5 text-sm dark:border-white/15 dark:bg-white/5" />
      {f.help && <p className="mt-0.5 text-[11px] text-ink-soft">{f.help}</p>}
    </div>);
  const specific = fieldsOf(["files"]);
  const oneClick = async () => {
    setBusy("download"); setMsg(null);
    try { await run({}); } catch (e) { setMsg({ ok: false, text: (e as Error).message }); } finally { setBusy(null); }
  };
  const ce = cat.data?.datasets.find((d) => d.id === `mission-${m.id}`);
  const jobNow = ce?.active_job;
  const failed = !jobNow ? ce?.last_failure : null;
  return (
    <section id="install" className={`${embedded ? "" : "glass p-5"} scroll-mt-28 space-y-4`} aria-labelledby={`install-h-${m.id}`}>
      {!embedded && auto && (
        <div className="flex flex-wrap items-center gap-4">
          <div className="min-w-0 flex-1">
            <h3 id={`install-h-${m.id}`} className="font-display text-lg font-bold">Install {m.name} data</h3>
            <p className="text-sm text-ink-soft">{auto.label} · {auto.size}. The importer checks schema, units, timestamps and gaps and records the
              source IDs and checksums; after that everything works offline.{auto.automatic ? " This starts by itself when the app runs online." : ""}</p>
          </div>
          <WriteButton className="btn-primary px-4 py-2 text-sm" onClick={oneClick} disabled={!!busy || !!jobNow}>
            {busy === "download" ? <Spinner className="h-4 w-4" /> : <Download className="h-4 w-4" aria-hidden="true" />} Install data
          </WriteButton>
        </div>)}
      {!embedded && !auto && <h3 id={`install-h-${m.id}`} className="font-display text-lg font-bold">Install {m.name} data</h3>}
      {jobNow && (
        <div className="glass-inset space-y-1 p-3 text-sm" role="status">
          <div className="flex items-center gap-2 font-semibold"><Spinner className="h-3.5 w-3.5" /> {jobNow.title} · {jobNow.status}
            {jobNow.progress ? ` · ${Math.round(jobNow.progress * 100)}%` : ""}</div>
          {jobNow.message && <p className="text-xs text-ink-soft">{jobNow.message}</p>}
        </div>)}
      {failed && (() => {
        const text = String(failed.error ?? failed.message ?? "");
        const calm = /^(.*: )?(Not available right now|source_unavailable)|rate limit/i.test(text);
        return (
          <div role={calm ? "status" : "alert"} className={`space-y-1 rounded-xl border p-3 text-sm ${calm
            ? "border-amber-300/60 bg-amber-50/80 text-amber-950 dark:border-amber-400/20 dark:bg-amber-500/10 dark:text-amber-100"
            : "border-rose-300/60 bg-rose-50/80 text-rose-900 dark:border-rose-400/20 dark:bg-rose-500/10 dark:text-rose-100"}`}>
            <div className="font-semibold">{calm ? "Source not available right now" : "Last install attempt failed"}</div>
            <p className="text-xs">{text.replace(/^(.*?: )?(Not available right now: |source_unavailable: )/, "")}</p>
          </div>);
      })()}
      {msg && !jobNow && !failed && <p role={msg.ok ? "status" : "alert"} className={`text-sm ${msg.ok ? "text-emerald-800 dark:text-emerald-200" : "text-rose-700 dark:text-rose-300"}`}>{msg.text}</p>}
      <details className="group" open={embedded || !auto}>
        <summary className={`cursor-pointer text-sm font-semibold ${embedded ? "hidden" : ""}`}>More options: choose specific files, or upload files you already have</summary>
        <div className={`mt-3 grid gap-4 ${dl && specific.length ? "md:grid-cols-2" : ""}`}>
          {dl && specific.length > 0 && (
            <div className="glass-inset space-y-2 p-4">
              <div className="flex items-center gap-2 font-semibold"><Download className="h-4 w-4 text-indigo-600" aria-hidden="true" /> {dl.label}</div>
              <p className="text-xs text-ink-soft">Needs internet on this computer · {dl.size}. Downloads resume if interrupted.</p>
              {specific.map(field)}
              <WriteButton className="btn-primary px-3 py-1.5 text-xs" onClick={download} disabled={!!busy || !vals.files?.trim()}>
                {busy === "download" ? <Spinner className="h-3.5 w-3.5" /> : <Download className="h-3.5 w-3.5" aria-hidden="true" />} Download these
              </WriteButton>
            </div>)}
          <div className="glass-inset space-y-2 p-4">
            <div className="flex items-center gap-2 font-semibold"><Upload className="h-4 w-4 text-indigo-600" aria-hidden="true" /> Upload the source files</div>
            <p className="text-xs text-ink-soft">{spec.upload.hint}. Get them from <a className="underline" href={m.source.url} target="_blank" rel="noreferrer">{m.source.title}</a>.</p>
            <label className="block text-xs font-semibold" htmlFor={`u-${m.id}`}>Files ({spec.upload.accept})</label>
            <FileInput id={`u-${m.id}`} type="file" multiple accept={spec.upload.accept} onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
              className="block w-full text-xs file:mr-2 file:rounded-lg file:border-0 file:bg-indigo-600 file:px-3 file:py-1.5 file:text-white" />
            {files.length > 0 && <p className="text-[11px] text-ink-soft">{files.length} file(s), {(files.reduce((a, f) => a + f.size, 0) / 1e6).toFixed(1)} MB</p>}
            {pct != null && <div className="h-2 overflow-hidden rounded-full bg-slate-200" role="progressbar" aria-valuenow={Math.round(pct * 100)} aria-valuemin={0} aria-valuemax={100} aria-label="Upload progress">
              <div className="h-full bg-indigo-600" style={{ width: `${pct * 100}%` }} /></div>}
            <WriteButton className="btn-primary px-3 py-1.5 text-xs" onClick={upload} disabled={!!busy || !files.length}>
              {busy === "upload" ? <Spinner className="h-3.5 w-3.5" /> : <Upload className="h-3.5 w-3.5" aria-hidden="true" />} Upload and install
            </WriteButton>
          </div>
        </div>
        <details className="mt-3 text-xs text-ink-soft">
          <summary className="cursor-pointer font-semibold">Command line (optional)</summary>
          <p className="mt-1">Inside the app container: <code className="break-all rounded bg-slate-900 px-1.5 py-0.5 text-[11px] text-slate-100">python -m spaceml.missions.install {m.id} --download</code></p>
          <p className="mt-1">Redistribution: {m.source.redistribution}. {m.trainable ? `Once installed: ${m.train_label}.` : `Once installed: ${m.detect}, with flagged intervals sent to Review Studio.`}</p>
        </details>
      </details>
    </section>
  );
}

/* ---------------------------------------------------------------- item list (keyboard: arrows / Home / End) */
function ItemPicker({ m, selected, onSelect }: { m: MissionStatus; selected: string | null; onSelect: (id: string) => void }) {
  const [q, setQ] = useState("");
  const [split, setSplit] = useState<"all" | "train" | "test">("all");
  const [label, setLabel] = useState<"all" | "anomaly" | "nominal">("all");
  const [channel, setChannel] = useState("");
  const qs = new URLSearchParams({ limit: "400", ...(q ? { q } : {}), ...(split !== "all" ? { split } : {}), ...(label !== "all" ? { label } : {}), ...(channel ? { channel } : {}) });
  const list = useApi<{ total: number; items: Item[] }>(`/api/missions/${m.id}/items?${qs}`, [m.id, q, split, label, channel]);
  const ref = useRef<HTMLUListElement>(null);
  const items = list.data?.items ?? [];
  const seg = m.family === "segments";
  useEffect(() => {
    if (list.loading || !items.length) return;
    if (!selected || (!q && split === "all" && label === "all" && !channel && !items.some((x) => x.id === selected))) onSelect(items[0].id);
  }, [items, selected, list.loading]); // eslint-disable-line react-hooks/exhaustive-deps
  const channels = useMemo(() => [...new Set((m.manifest?.channels ?? []).map((c) => c.id))], [m]);
  const key = (e: React.KeyboardEvent) => {
    const i = items.findIndex((x) => x.id === selected);
    const j = e.key === "ArrowDown" ? i + 1 : e.key === "ArrowUp" ? Math.max(0, i - 1) : e.key === "Home" ? 0 : e.key === "End" ? items.length - 1 : -2;
    if (j === -2 || !items.length) return;
    e.preventDefault();
    const n = items[Math.min(items.length - 1, Math.max(0, j))];
    onSelect(n.id);
    window.requestAnimationFrame(() => ref.current?.querySelector<HTMLElement>(`[data-id="${CSS.escape(n.id)}"]`)?.scrollIntoView({ block: "nearest" }));
  };
  return (
    <Glass className="space-y-3 p-4">
      <label className="block text-xs font-semibold" htmlFor={`q-${m.id}`}>Find a {m.item_label}</label>
      <div className="relative">
        <Search className="pointer-events-none absolute left-2.5 top-2.5 h-4 w-4 text-ink-faint" aria-hidden="true" />
        <input id={`q-${m.id}`} value={q} onChange={(e) => setQ(e.target.value)} placeholder={`${m.item_label} id or name`}
          className="w-full rounded-xl border border-slate-300 bg-white/80 py-2 pl-8 pr-2 text-sm dark:border-white/15 dark:bg-white/5" />
      </div>
      {seg && (
        <div className="space-y-2">
          <Segmented label="Split" value={split} onChange={setSplit} options={[{ value: "all", label: "All" }, { value: "train", label: "Train" }, { value: "test", label: "Held-out" }]} />
          <Segmented label="Published label" value={label} onChange={setLabel} options={[{ value: "all", label: "Any" }, { value: "anomaly", label: "Anomaly" }, { value: "nominal", label: "Nominal" }]} />
        </div>)}
      {seg && channels.length > 1 && (
        <div>
          <label className="block text-xs font-semibold" htmlFor={`ch-${m.id}`}>Channel</label>
          <select id={`ch-${m.id}`} value={channel} onChange={(e) => setChannel(e.target.value)} className="mt-1 w-full rounded-xl border border-slate-300 bg-white/80 px-2 py-1.5 text-sm dark:border-white/15 dark:bg-white/5">
            <option value="">All channels</option>{channels.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>)}
      <div className="text-[11px] text-ink-soft" aria-live="polite">{list.loading ? "Loading…" : `${list.data?.total ?? 0} ${m.item_label}s`}{(list.data?.total ?? 0) > items.length ? ` (first ${items.length} shown - refine the search)` : ""}</div>
      {list.error && <ErrorNotice message={list.error.message} />}
      <ul ref={ref} role="listbox" aria-label={`${m.name} ${m.item_label}s`} tabIndex={0} onKeyDown={key}
        aria-activedescendant={selected ? `opt-${m.id}-${selected}` : undefined}
        className="scroll-thin max-h-[460px] space-y-1 overflow-y-auto pr-1 outline-none">
        {items.map((it) => {
          const on = it.id === selected;
          return (
            <li key={it.id} id={`opt-${m.id}-${it.id}`} data-id={it.id} role="option" aria-selected={on} onClick={() => onSelect(it.id)}
              className={`cursor-pointer rounded-xl border px-3 py-2 text-sm ${on ? "border-indigo-400 bg-white shadow dark:bg-white/10" : "border-transparent hover:bg-white/60 dark:hover:bg-white/5"}`}>
              <div className="truncate font-semibold">{it.label}</div>
              <div className="flex flex-wrap gap-x-2 text-[11px] text-ink-soft">
                <span>{it.n.toLocaleString()} samples</span>
                {it.split && <span>{it.split === "test" ? "held-out" : "train"}</span>}
                {it.anomaly !== undefined && <span className={it.anomaly ? "font-semibold text-rose-700 dark:text-rose-300" : ""}>{it.anomaly ? "▲ anomaly" : "nominal"}</span>}
                {it.gaps > 0 && <span>{it.gaps} gap{it.gaps > 1 ? "s" : ""}</span>}
                {!!it.events && <span>{it.events} annotated</span>}
                {it.warnings && it.warnings.length > 0 && <span className="font-semibold text-amber-800 dark:text-amber-300">⚠ {it.warnings[0]}</span>}
              </div>
            </li>);
        })}
      </ul>
    </Glass>
  );
}

/* ---------------------------------------------------------------- viewer + scoring */
function Viewer({ m, itemId, onAnnounce, runId, onRun }: { m: MissionStatus; itemId: string; onAnnounce: (s: string) => void; runId: string | null; onRun: (id: string | null) => void }) {
  const caps = useCapabilities();
  const canRun = useCanRun("missions", { mission: m.id, action: "score", item: itemId });
  const frozenUnavailable = caps?.mode === "vercel" && !canRun;
  const meta = useApi<{ items: Item[] }>(`/api/missions/${m.id}/items?q=${encodeURIComponent(itemId)}&limit=20`, [m.id, itemId]);
  const item = meta.data?.items.find((x) => x.id === itemId) ?? null;
  const allCols = item?.columns.filter((c) => !c.startsWith("flag:")) ?? [];
  const defaults = (m.manifest?.detect_columns ?? []).filter((c) => allCols.includes(c)).slice(0, 2);
  const [cols, setCols] = useState<string[]>([]);
  useEffect(() => { setCols(defaults.length ? defaults : allCols.slice(0, 1)); }, [itemId, item?.columns.join(",")]); // eslint-disable-line react-hooks/exhaustive-deps
  const s = useApi<Series>(cols.length ? `/api/missions/${m.id}/series?item=${encodeURIComponent(itemId)}&columns=${encodeURIComponent(cols.join(","))}&max_points=1600` : null, [m.id, itemId, cols.join(",")]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [scoreRun, setScoreRun] = useState<string | null>(null);
  const run = useRun<ScoreResult>(scoreRun);
  useEffect(() => { setScoreRun(null); setErr(null); }, [itemId]);
  useEffect(() => { if (s.data) onAnnounce(`Loaded ${s.data.n_total.toLocaleString()} samples of ${itemId}`); }, [s.data]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (run.run?.status === "succeeded") onAnnounce(`Scoring finished: ${(run.run.summary as { headline?: string })?.headline ?? ""}`);
    if (run.run?.status === "failed") onAnnounce(`Scoring failed: ${run.run.error}`);
  }, [run.run?.status]); // eslint-disable-line react-hooks/exhaustive-deps
  async function score() {
    setBusy(true); setErr(null);
    try { const r = await createRun("missions", { mission: m.id, action: "score", item: itemId }); setScoreRun(r.id); onRun(r.id); onAnnounce("Scoring started"); }
    catch (e) { setErr((e as Error).message); onAnnounce(`Could not start scoring: ${(e as Error).message}`); }
    finally { setBusy(false); }
  }
  const res = run.run?.status === "succeeded" ? (run.run.result as ScoreResult | undefined) ?? null : null;
  const canScore = !frozenUnavailable && (m.family !== "segments" || m.model.ready);
  void runId;
  return (
    <Glass className="space-y-3 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="font-display text-lg font-bold">{item?.label ?? itemId}</h3>
        {item?.split && <span className="chip bg-slate-200 text-slate-800">{item.split === "test" ? "held-out (test)" : "training"} split</span>}
        {item?.published_label && <span className={`chip ${item.anomaly ? "bg-rose-100 text-rose-800" : "bg-emerald-100 text-emerald-800"}`}>published label: {item.anomaly ? "anomaly" : "nominal"} ({item.published_label})</span>}
        {item?.guiding_modes && <span className="chip bg-sky-100 text-sky-900">guiding: {item.guiding_modes.join(", ")}</span>}
        <button className="btn-primary ml-auto px-3 py-1.5 text-xs" onClick={score} disabled={busy || !canScore || run.run?.status === "running" || run.run?.status === "queued"}
          title={frozenUnavailable ? READ_ONLY_HINT : canScore ? undefined : "Train a model first"}>
          {busy || run.run?.status === "running" || run.run?.status === "queued" ? <Spinner className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" aria-hidden="true" />}
          {frozenUnavailable ? READ_ONLY_HINT : m.family === "segments" ? "Predict this segment" : `Score this ${m.item_label}`}
        </button>
      </div>
      {!canScore && <p className="text-xs text-ink-soft">Prediction needs a trained model - press <b>Quick train</b> above (takes seconds).</p>}
      {allCols.length > 1 && (
        <fieldset className="flex flex-wrap gap-x-3 gap-y-1 text-xs">
          <legend className="mb-1 text-xs font-semibold">Channels to plot</legend>
          {allCols.map((c) => (
            <label key={c} className="inline-flex items-center gap-1">
              <input type="checkbox" checked={cols.includes(c)} onChange={(e) => setCols((x) => e.target.checked ? [...x, c] : x.filter((y) => y !== c))} />
              {c}{item?.units?.[c] ? ` [${item.units[c]}]` : ""}
            </label>))}
        </fieldset>)}
      {item?.guiding && Object.keys(item.guiding).length > 0 && (
        <p className="text-xs text-ink-soft"><b>Guiding flags (jif):</b> {Object.entries(item.guiding).map(([k, v]) => `${k}=${v ?? "missing"}`).join(" · ")}</p>)}
      {s.error && (/no item/i.test(s.error.message)
        ? <p className="text-sm text-ink-soft" role="status">Choose a {m.item_label} from the list.</p>
        : <ErrorNotice title={`Could not load this ${m.item_label}`} message={s.error.message} />)}
      {s.loading && !s.data && <div className="flex items-center gap-2 text-sm text-ink-soft" role="status"><Spinner /> Loading telemetry…</div>}
      {s.data && <SeriesChart s={s.data} title={item?.label ?? itemId} res={res} cadence={item?.cadence_s ?? item?.sampling_s ?? null} />}
      {err && <ErrorNotice message={err} />}
      {run.run && run.run.status !== "succeeded" && run.run.status !== "queued" && run.run.status !== "running" &&
        <ErrorNotice message={run.run.error ?? "Scoring failed"} title="Scoring failed" />}
      {res && <ScorePanel res={res} m={m} runId={run.run!.id} />}
    </Glass>
  );
}

function median(a: number[]) { if (!a.length) return NaN; const b = [...a].sort((x, y) => x - y); return b[Math.floor(b.length / 2)]; }
function diffs(t: number[]) { const out: number[] = []; for (let i = 1; i < t.length; i++) out.push(t[i] - t[i - 1]); return out; }
function quantile(sorted: number[], q: number) { return sorted.length ? sorted[Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))))] : NaN; }
function channelStats(y: (number | null)[]) {
  const v = y.filter((x): x is number => x != null && Number.isFinite(x)).sort((a, b) => a - b);
  if (!v.length) return { median: null, p05: null, p95: null, min: null, max: null, lo: null, hi: null, clipped: false };
  const lo = quantile(v, 0.005), hi = quantile(v, 0.995), min = v[0], max = v[v.length - 1];
  const span = hi - lo || 1;
  return { median: quantile(v, 0.5), p05: quantile(v, 0.05), p95: quantile(v, 0.95), min, max, lo, hi,
    clipped: max - hi > span * 0.5 || lo - min > span * 0.5 };
}
function fmtNum(x: number | null) {
  if (x == null || !Number.isFinite(x)) return "—";
  const a = Math.abs(x);
  return a !== 0 && (a >= 1e5 || a < 1e-3) ? x.toExponential(2) : Number(x.toPrecision(4)).toLocaleString();
}
function fmtDuration(sec: number) {
  if (!Number.isFinite(sec)) return "—";
  if (sec < 1) return `${(sec * 1000).toFixed(0)} ms`;
  if (sec < 120) return `${sec.toFixed(sec < 10 ? 2 : 0)} s`;
  if (sec < 7200) return `${(sec / 60).toFixed(1)} min`;
  if (sec < 172800) return `${(sec / 3600).toFixed(1)} h`;
  return `${(sec / 86400).toFixed(1)} days`;
}
function MiniStat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="glass-inset px-3 py-2">
      <div className="text-[10px] font-semibold uppercase tracking-wide text-ink-faint">{label}</div>
      <div className="font-display text-base font-bold tabular-nums">{value}</div>
      {sub && <div className="truncate text-[10px] text-ink-soft">{sub}</div>}
    </div>);
}

function SeriesChart({ s, title, res, cadence = null }: { s: Series; title: string; res: ScoreResult | null; cadence?: number | null }) {
  const [fullRange, setFullRange] = useState(false);
  const x = s.time_is_epoch ? s.t.map((t) => new Date(t * 1000).toISOString()) : s.t;
  const tx = (t: number) => (s.time_is_epoch ? new Date(t * 1000).toISOString() : t);
  const colors = ["#4f46e5", "#0891b2", "#be185d", "#15803d", "#b45309"];
  const traces = Object.entries(s.cols).map(([c, y], i) => ({ type: "scattergl" as const, mode: "lines" as const, x, y, name: `${c}${s.units[c] ? ` [${s.units[c]}]` : ""}`,
    line: { width: 1.2, color: colors[i % colors.length] }, connectgaps: false }));
  const shapes: Partial<Plotly.Shape>[] = [];
  s.gaps.slice(0, 100).forEach((g) => shapes.push({ type: "rect", xref: "x", yref: "paper", x0: tx(g.start), x1: tx(g.end), y0: 0, y1: 1, fillcolor: "rgba(100,116,139,0.18)", line: { width: 0 }, layer: "below" }));
  s.events.slice(0, 200).forEach((e) => shapes.push({ type: "rect", xref: "x", yref: "paper", x0: tx(e.start), x1: tx(e.end), y0: 0, y1: 1,
    fillcolor: e.category === "Anomaly" ? "rgba(225,29,72,0.16)" : "rgba(245,158,11,0.16)", line: { width: 0 }, layer: "below" }));
  if (s.label && s.labeled_samples && s.labeled_samples === s.n_total && !s.events.length)
    shapes.push({ type: "rect", xref: "paper", yref: "paper", x0: 0, x1: 1, y0: 0, y1: 1, fillcolor: "rgba(225,29,72,0.07)", line: { width: 0 }, layer: "below" });
  (res?.model_intervals ?? []).forEach((iv) => shapes.push({ type: "rect", xref: "x", yref: "paper", x0: tx(iv.start), x1: tx(iv.end), y0: 0.94, y1: 1, fillcolor: "rgba(225,29,72,0.9)", line: { width: 0 } }));
  (res?.baseline_intervals ?? []).forEach((iv) => shapes.push({ type: "rect", xref: "x", yref: "paper", x0: tx(iv.start), x1: tx(iv.end), y0: 0.88, y1: 0.93, fillcolor: "rgba(245,158,11,0.9)", line: { width: 0 } }));
  const missing = Object.entries(s.missing).filter(([, v]) => v > 0).map(([k, v]) => `${k}: ${v}`).join(", ");
  const summary = `${title}: ${s.n_total.toLocaleString()} samples from ${fmtTime(s.t[0] ?? 0, s.time_is_epoch)} to ${fmtTime(s.t[s.t.length - 1] ?? 0, s.time_is_epoch)}` +
    `${s.downsampled ? `, drawn as a min/max envelope of ${s.n_returned.toLocaleString()} points (analysis uses all samples)` : ""}; ` +
    `${s.gaps.length} data gap${s.gaps.length === 1 ? "" : "s"}; missing readings ${missing || "none"}` +
    `${s.events.length ? `; ${s.events.length} annotated interval(s)` : ""}${res ? `; ${res.model_intervals?.length ?? 0} model and ${res.baseline_intervals?.length ?? 0} baseline flagged interval(s)` : ""}.`;
  const rows = s.t.map((t, i) => ({ t, v: Object.fromEntries(Object.entries(s.cols).map(([c, y]) => [c, y[i]])) }));
  const names = Object.keys(s.cols);
  const stats = names.map((c) => channelStats(s.cols[c]));
  // one panel per channel, sharing the time axis: a channel with large values no longer flattens the others
  const n = Math.max(1, names.length);
  const gap = n > 1 ? 0.06 : 0;
  const h = (1 - gap * (n - 1)) / n;
  const layout: Partial<Plotly.Layout> & Record<string, unknown> = {
    xaxis: { title: { text: s.time_is_epoch ? "Time (UTC)" : "Seconds since start" }, type: s.time_is_epoch ? "date" : "linear", anchor: (n > 1 ? `y${n}` : "y") as Plotly.AxisName },
    shapes, showlegend: false, margin: { l: 64, r: 12, t: 10, b: 44 },
  };
  names.forEach((c, i) => {
    const key = i === 0 ? "yaxis" : `yaxis${i + 1}`;
    const top = 1 - i * (h + gap);
    const st = stats[i];
    const range = !fullRange && st.lo != null && st.hi != null && st.hi > st.lo ? [st.lo - (st.hi - st.lo) * 0.08, st.hi + (st.hi - st.lo) * 0.08] : undefined;
    layout[key] = { domain: [Math.max(0, top - h), top], title: { text: `${c}${s.units[c] ? ` [${s.units[c]}]` : ""}`, font: { size: 10 } }, range, autorange: range ? false : true, zeroline: false };
  });
  const panelTraces = traces.map((tr, i) => ({ ...tr, xaxis: "x", yaxis: i === 0 ? "y" : `y${i + 1}` }));
  const clipped = !fullRange && stats.some((st) => st.clipped);
  return (
    <figure className="space-y-2">
      <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
        <MiniStat label="Duration" value={fmtDuration(s.t[s.t.length - 1] - s.t[0])} sub={`${s.n_total.toLocaleString()} samples`} />
        <MiniStat label="Cadence" value={cadence ? fmtDuration(cadence) : s.t.length > 2 && !s.downsampled ? fmtDuration(median(diffs(s.t))) : "varies"} sub="median time between samples" />
        <MiniStat label="Data gaps" value={String(s.gaps.length)} sub={s.gaps.length ? `longest ${fmtDuration(Math.max(...s.gaps.map((g) => g.seconds)))}` : "continuous"} />
        <MiniStat label="Flagged" value={res ? `${res.model_intervals?.length ?? 0} / ${res.baseline_intervals?.length ?? 0}` : s.events.length ? String(s.events.length) : "—"}
          sub={res ? "model / baseline intervals" : s.events.length ? "annotated intervals" : "score this item to flag"} />
      </div>
      <div role="img" aria-label={summary}>
        <Plot height={Math.min(560, 150 + 120 * n)} data={panelTraces} layout={layout as Partial<Plotly.Layout>} />
      </div>
      <div className="flex flex-wrap items-center gap-2 text-[11px] text-ink-soft">
        {clipped && <span>Axes zoomed to the 0.5-99.5 % range so short spikes don't hide the normal behaviour.</span>}
        <button className="btn-ghost px-2 py-0.5 text-[11px]" onClick={() => setFullRange((v) => !v)}>{fullRange ? "Zoom to typical range" : "Show full range"}</button>
      </div>
      <div className="scroll-thin overflow-x-auto">
        <table className="w-full text-left text-[11px]">
          <caption className="sr-only">Per-channel statistics</caption>
          <thead className="text-ink-faint"><tr><th scope="col" className="py-1 pr-3">Channel</th><th scope="col" className="pr-3">Median</th><th scope="col" className="pr-3">Typical range (5-95 %)</th><th scope="col" className="pr-3">Min … max</th><th scope="col" className="pr-3">Missing</th></tr></thead>
          <tbody>{names.map((c, i) => (
            <tr key={c} className="border-t border-slate-200/60 dark:border-white/10">
              <td className="py-1 pr-3 font-semibold">{c}{s.units[c] ? ` [${s.units[c]}]` : ""}</td>
              <td className="pr-3 font-mono">{fmtNum(stats[i].median)}</td>
              <td className="pr-3 font-mono">{fmtNum(stats[i].p05)} … {fmtNum(stats[i].p95)}</td>
              <td className="pr-3 font-mono">{fmtNum(stats[i].min)} … {fmtNum(stats[i].max)}{stats[i].clipped ? " ⚑" : ""}</td>
              <td className="pr-3">{s.missing[c] ? `${s.missing[c].toLocaleString()}` : "none"}</td>
            </tr>))}</tbody>
        </table>
      </div>
      <figcaption className="text-xs text-ink-soft">
        {summary} <span>Legend: <span className="font-semibold text-rose-700 dark:text-rose-300">■ top bar</span> model flags,
          <span className="font-semibold text-amber-800 dark:text-amber-300"> ■ second bar</span> rolling-MAD baseline, shaded red = published anomaly / annotation, grey = data gap.</span>
      </figcaption>
      <details className="text-xs">
        <summary className="cursor-pointer font-semibold"><Table2 className="mr-1 inline h-3.5 w-3.5" aria-hidden="true" />Data table (plotted points)</summary>
        <div className="scroll-thin mt-2 max-h-64 overflow-auto">
          <table className="w-full text-left text-[11px]">
            <caption className="sr-only">{title} plotted values</caption>
            <thead><tr><th scope="col" className="pr-3">Time</th>{Object.keys(s.cols).map((c) => <th scope="col" key={c} className="pr-3">{c}{s.units[c] ? ` [${s.units[c]}]` : ""}</th>)}</tr></thead>
            <tbody>{rows.slice(0, 300).map((r, i) => <tr key={i}><td className="pr-3 font-mono">{fmtTime(r.t, s.time_is_epoch)}</td>{Object.keys(s.cols).map((c) => <td key={c} className="pr-3 font-mono">{r.v[c] == null ? "missing" : (r.v[c] as number).toPrecision(5)}</td>)}</tr>)}</tbody>
          </table>
          {rows.length > 300 && <p className="mt-1">First 300 of {rows.length} plotted points.</p>}
        </div>
      </details>
    </figure>
  );
}

function ScorePanel({ res, m, runId }: { res: ScoreResult; m: MissionStatus; runId: string }) {
  const review = <Link to="/review?module=missions" className="btn-ghost px-3 py-1.5 text-xs"><Inbox className="h-3.5 w-3.5" aria-hidden="true" /> Open in Review Studio</Link>;
  if (m.family === "segments") {
    const ok = res.pred === res.label;
    return (
      <div className="glass-inset flex flex-wrap items-center gap-3 p-3 text-sm" role="status">
        <ClipboardCheck className="h-4 w-4 text-indigo-600" aria-hidden="true" />
        <span>Prediction: <b>{res.pred ? "anomaly" : "nominal"}</b> (p = {res.prob?.toFixed(2)}) · published label <b>{res.label ? "anomaly" : "nominal"}</b> · {ok ? "✓ agrees" : "✕ disagrees"}{res.split === "train" ? " (training segment - not a held-out result)" : ""}</span>
        <span className="ml-auto">{review}</span>
      </div>);
  }
  const ivs = (res.model_intervals?.length ? res.model_intervals : res.baseline_intervals) ?? [];
  return (
    <div className="glass-inset space-y-2 p-3 text-sm" role="status">
      <div className="flex flex-wrap items-center gap-2">
        <Gauge className="h-4 w-4 text-indigo-600" aria-hidden="true" />
        <b>{res.model_intervals?.length ?? 0} model · {res.baseline_intervals?.length ?? 0} baseline interval(s)</b>
        <span className="text-xs text-ink-soft">detector: {res.detector}</span>
        <span className="ml-auto">{review}</span>
      </div>
      {res.context?.warnings && <p className="text-xs">Guiding context: {res.context.warnings.length ? res.context.warnings.join(" · ") : "no warnings in the jif summary"}</p>}
      {ivs.length === 0 ? <p className="text-xs text-ink-soft">Nothing unusual flagged in this {m.item_label}.</p> : (
        <ul className="space-y-0.5 text-xs">{ivs.slice(0, 8).map((iv, i) => <li key={i} className="font-mono">{fmtTime(iv.start, !!m.manifest?.time_is_epoch)} → {fmtTime(iv.end, !!m.manifest?.time_is_epoch)} · {iv.n} samples{iv.score != null ? ` · score ${iv.score}` : ""}</li>)}</ul>)}
      <p className="text-[11px] text-ink-soft">No verified fault labels exist for this source, so flags are review candidates, not detections. Run {runId}.</p>
    </div>
  );
}

/* ---------------------------------------------------------------- held-out evaluation */
function Evaluation({ m, runId, onRun, onAnnounce }: { m: MissionStatus; runId: string | null; onRun: (id: string | null) => void; onAnnounce: (s: string) => void }) {
  const readOnly = useReadOnly();
  const canEvaluate = useCanRun("missions", { mission: m.id, action: "evaluate" });
  const [rid, setRid] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const run = useRun<EvalResult>(rid);
  useEffect(() => {
    if (!runId) return;
    api<{ input_ref: { action?: string } }>(`/api/runs/${runId}`).then((r) => { if (r.input_ref?.action === "evaluate") setRid(runId); }).catch(() => undefined);
  }, [runId]);
  useEffect(() => { if (run.run?.status === "succeeded") onAnnounce("Held-out evaluation finished"); }, [run.run?.status]); // eslint-disable-line react-hooks/exhaustive-deps
  async function go() {
    setErr(null);
    try { const r = await createRun("missions", { mission: m.id, action: "evaluate" }); setRid(r.id); onRun(r.id); onAnnounce("Evaluation started"); }
    catch (e) { setErr((e as Error).message); onAnnounce(`Evaluation could not start: ${(e as Error).message}`); }
  }
  const res = run.run?.status === "succeeded" ? (run.run.result as EvalResult | undefined) : undefined;
  const met = res?.metrics;
  return (
    <Glass className="space-y-3 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="font-display text-lg font-bold">Held-out evaluation</h3>
        <span className="text-xs text-ink-soft">{m.evaluate}</span>
        <button className="btn-primary ml-auto px-3 py-1.5 text-xs" onClick={go} disabled={!canEvaluate || !m.model.ready || run.run?.status === "running" || run.run?.status === "queued"}>
          {run.run && ["queued", "running"].includes(run.run.status) ? <Spinner className="h-3.5 w-3.5" /> : <ClipboardCheck className="h-3.5 w-3.5" aria-hidden="true" />} {readOnly ? (canEvaluate ? "View sample result" : READ_ONLY_HINT) : `Evaluate ${m.model.version ?? ""}`}
        </button>
      </div>
      {!m.model.ready && <p className="text-xs text-ink-soft">{readOnly ? "No model result is published. Training is available in Docker." : <>No model yet - press <b>Quick train</b> above, then evaluate.</>}</p>}
      {err && <ErrorNotice message={err} />}
      {run.run?.status === "failed" && <ErrorNotice message={run.run.error ?? "evaluation failed"} />}
      {met && (
        <div className="space-y-3">
          <p className="text-xs text-ink-soft">{met.eval_split}{met.protocol ? ` · ${met.protocol}` : ""}{met.labels ? ` · ${met.labels}` : ""}</p>
          <div className="grid gap-3 md:grid-cols-2">
            {(["model", "baseline"] as const).map((k) => <MetricsCard key={k} title={k === "model" ? "Model" : "Baseline"} b={met[k]} />)}
          </div>
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <b>{res!.review_items.length}</b> held-out {m.family === "segments" ? "predictions" : "flagged intervals"} were sent to Review Studio (errors first).
            <Link to="/review?module=missions" className="btn-ghost px-3 py-1.5 text-xs"><ExternalLink className="h-3.5 w-3.5" aria-hidden="true" /> Review them</Link>
          </div>
        </div>)}
    </Glass>
  );
}

function MetricsCard({ title, b }: { title: string; b: EvalBlock }) {
  const cm = b.confusion_matrix;
  return (
    <div className="glass-inset space-y-2 p-3 text-xs">
      <div className="font-semibold">{title}: <span className="font-normal">{b.name}</span></div>
      {cm && (
        <table className="text-left">
          <caption className="sr-only">{title} confusion matrix: rows are published labels, columns are predictions</caption>
          <thead><tr><th scope="col" className="pr-3">published ↓ / predicted →</th>{cm.labels.map((l) => <th scope="col" key={l} className="pr-3">{l}</th>)}</tr></thead>
          <tbody>{cm.matrix.map((r, i) => <tr key={i}><th scope="row" className="pr-3 font-semibold">{cm.labels[i]}</th>{r.map((v, j) => <td key={j} className={`pr-3 font-mono ${i !== j && v ? "font-bold text-rose-700 dark:text-rose-300" : ""}`}>{v}</td>)}</tr>)}</tbody>
        </table>)}
      {b.per_class && (
        <table className="text-left">
          <caption className="sr-only">{title} per-class precision and recall</caption>
          <thead><tr><th scope="col" className="pr-3">class</th><th scope="col" className="pr-3">precision</th><th scope="col" className="pr-3">recall</th><th scope="col" className="pr-3">F1</th><th scope="col">n</th></tr></thead>
          <tbody>{Object.entries(b.per_class).map(([k, v]) => <tr key={k}><th scope="row" className="pr-3">{k}</th><td className="pr-3 font-mono">{num3(v.precision)}</td><td className="pr-3 font-mono">{num3(v.recall)}</td><td className="pr-3 font-mono">{num3(v.f1)}</td><td className="font-mono">{v.support}</td></tr>)}</tbody>
        </table>)}
      <ul className="space-y-0.5">
        {b.false_alarms != null && <li>False alarms: <b>{b.false_alarms}</b> · missed anomalies: <b>{b.missed_anomalies}</b>{b.roc_auc != null ? ` · ROC-AUC ${num3(b.roc_auc)}` : ""}</li>}
        {b.events != null && <li>Events {b.events} · detected {b.detected_events} (recall {pct(b.event_recall)}) · alarms {b.alarms}, of which false {b.false_alarms} (precision {pct(b.alarm_precision)})</li>}
      </ul>
    </div>
  );
}

export function useMissions() {
  return useApi<{ missions: MissionStatus[] }>("/api/missions");
}

export { Notice };
