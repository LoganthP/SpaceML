import { WriteButton, FileInput } from "../lib/capabilities";
import { CheckCircle2, CircleDashed, Clock, CloudOff, Play, RotateCcw, Rocket, Upload, XCircle } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import type { Module } from "../lib/types";
import { activeJob, JobButton, ProgressBar, useJobs } from "./Jobs";
import { Spinner } from "./ui";

/* ------------------------------------------------------------------ automatic setup */
export interface SetupStep {
  id: string; module: Module; title: string; detail: string; task: string; minutes: string; included: boolean;
  state: "pending" | "waiting" | "running" | "done" | "failed" | "source_unavailable" | "cancelled" | "interrupted";
  attempts: number; error: string | null; job_id: string | null; progress: number | null; message: string | null;
  eta_seconds?: number | null;
}
const fmtEta = (s?: number | null) => (s == null ? "" : s < 90 ? "about a minute left" : s < 3600 ? `about ${Math.round(s / 60)} min left` : `about ${(s / 3600).toFixed(1)} h left`);
export interface SetupStatus { profile: string; running: boolean; steps: SetupStep[]; done: number; total: number; complete: boolean }

export function useSetup() {
  const [data, setData] = useState<SetupStatus | null>(null);
  const load = useCallback(async () => { try { setData(await api<SetupStatus>("/api/setup")); } catch { /* API offline */ } }, []);
  useEffect(() => {
    load();
    const t = window.setInterval(load, data?.running ? 3000 : 15000);
    return () => window.clearInterval(t);
  }, [load, data?.running]);
  const run = useCallback(async (steps: string[] = []) => {
    await api("/api/setup/run", { method: "POST", body: JSON.stringify({ steps, force: true }) });
    await load();
  }, [load]);
  return { data, reload: load, run };
}

const STATE: Record<SetupStep["state"], { label: string; cls: string; icon: typeof Clock }> = {
  pending: { label: "to do", cls: "text-ink-faint", icon: CircleDashed },
  waiting: { label: "queued", cls: "text-amber-600", icon: Clock },
  running: { label: "running", cls: "text-violet-600", icon: Clock },
  done: { label: "ready", cls: "text-emerald-600", icon: CheckCircle2 },
  failed: { label: "failed", cls: "text-rose-600", icon: XCircle },
  source_unavailable: { label: "source unreachable", cls: "text-amber-600", icon: CloudOff },
  cancelled: { label: "cancelled", cls: "text-ink-soft", icon: XCircle },
  interrupted: { label: "interrupted", cls: "text-amber-600", icon: RotateCcw },
};

/** Card with every setup step: what is ready, what is running, why something failed + Retry. */
export function SetupPanel({ modules, compact = false }: { modules?: Module[]; compact?: boolean }) {
  const { data, run } = useSetup();
  const [busy, setBusy] = useState(false);
  if (!data) return null;
  const steps = data.steps.filter((s) => !modules || modules.includes(s.module));
  const missing = steps.filter((s) => s.state !== "done");
  if (compact && !missing.length) return null;
  const go = async (ids: string[] = []) => { setBusy(true); try { await run(ids.length ? ids : steps.map((s) => s.id)); } finally { setBusy(false); } };
  return (
    <div className="glass hud space-y-3 p-5">
      <div className="flex flex-wrap items-center gap-2">
        <Rocket className="h-4 w-4 text-violet-500" />
        <div className="kicker">Automatic setup</div>
        <span className="chip bg-white/70 text-ink-soft">{steps.filter((s) => s.state === "done").length}/{steps.length} ready</span>
        {data.running && <span className="chip bg-violet-100 text-violet-700"><Spinner className="h-3 w-3" /> working in the background</span>}
        <WriteButton className="btn-primary ml-auto px-3 py-1.5 text-xs" disabled={busy || data.running || !missing.length} onClick={() => go()}>
          {busy ? <Spinner /> : <Play className="h-3.5 w-3.5" />} {missing.length ? "Download & train everything missing" : "All set"}
        </WriteButton>
      </div>
      {!compact && <p className="text-xs text-ink-soft">Runs by itself when the app starts. Downloads and model training run side by side in the background
        (training uses every CPU core), with no terminal needed. Finished steps are skipped on the next start, and a step that stops making progress is
        stopped automatically and can be retried. You can keep using the dashboard meanwhile.</p>}
      <div className="space-y-2">
        {steps.map((s) => {
          const st = STATE[s.state] ?? STATE.pending;
          const Icon = st.icon;
          const live = s.state === "running" || s.state === "waiting";
          return (
            <div key={s.id} className="glass-inset space-y-1.5 p-3">
              <div className="flex items-center gap-2">
                {live ? <Spinner className="h-4 w-4 text-violet-600" /> : <Icon className={`h-4 w-4 ${st.cls}`} />}
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm font-semibold">{s.title}</div>
                  <div className="truncate text-[11px] text-ink-soft">{s.state === "waiting" ? "starts when the step it depends on is finished" : live && s.message ? s.message : s.detail}{s.state === "running" && s.eta_seconds != null ? ` · ${fmtEta(s.eta_seconds)}` : s.minutes && s.state !== "done" ? ` · ~${s.minutes} min` : ""}</div>
                </div>
                <span className={`text-[11px] font-semibold ${st.cls}`}>{st.label}</span>
                {!live && s.state !== "done" && (
                  <WriteButton className="btn-ghost px-2 py-1 text-[11px]" disabled={busy || data.running} onClick={() => go([s.id])}
                    title={s.state === "pending" ? "Start this step now" : "Try this step again"}>
                    {s.state === "pending" ? <Play className="h-3 w-3" /> : <RotateCcw className="h-3 w-3" />} {s.state === "pending" ? "Start" : "Retry"}
                  </WriteButton>)}
              </div>
              {s.state === "running" && <ProgressBar value={s.progress} />}
              {s.error && s.state !== "done" && !live && (
                <p className={`rounded-xl px-3 py-1.5 text-[11px] ${s.state === "failed" ? "bg-rose-50/80 text-rose-800" : "bg-amber-50/80 text-amber-900"}`}>
                  {s.state === "source_unavailable" && <b>Could not reach the data source - check the internet connection, then Retry. </b>}{s.error}
                </p>)}
            </div>);
        })}
      </div>
    </div>
  );
}

/** Small inline status for one module's page: "being prepared automatically" or a Start button. */
export function ModuleSetup({ module, stepIds, title }: { module: Module; stepIds: string[]; title: string }) {
  const { data, run } = useSetup();
  const { jobs } = useJobs();
  const job = activeJob(jobs, module);
  const steps = (data?.steps ?? []).filter((s) => stepIds.includes(s.id));
  const open = steps.filter((s) => s.state !== "done");
  const [busy, setBusy] = useState(false);
  if (!data || !open.length) return null;
  const live = open.find((s) => s.state === "running" || s.state === "waiting");
  const failed = open.find((s) => s.error && (s.state === "failed" || s.state === "source_unavailable"));
  return (
    <div className="rounded-2xl border border-violet-200/70 bg-violet-50/70 p-4 text-sm text-violet-950 dark:border-violet-400/20 dark:bg-violet-500/10 dark:text-violet-100">
      <div className="flex flex-wrap items-center gap-2 font-semibold">
        {live || job ? <Spinner className="h-4 w-4" /> : <Rocket className="h-4 w-4" />} {title}
        {!live && !job && (
          <WriteButton className="btn-primary ml-auto px-3 py-1.5 text-xs" disabled={busy || data.running}
            onClick={async () => { setBusy(true); try { await run(open.map((s) => s.id)); } finally { setBusy(false); } }}>
            {busy ? <Spinner /> : failed ? <RotateCcw className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />} {failed ? "Retry now" : "Prepare it now"}
          </WriteButton>)}
      </div>
      <div className="mt-2 space-y-1.5 text-[13px] opacity-90">
        {live || job ? (
          <>
            <p>{(job?.title ?? live?.title)} - {job?.message || job?.step || live?.message || "starting…"}</p>
            <ProgressBar value={job?.progress ?? live?.progress ?? null} />
            <p className="text-[11px] opacity-75">Runs in the background; this page updates by itself when it is ready.</p>
          </>
        ) : failed ? (
          <p>The last attempt did not finish: {failed.error}</p>
        ) : (
          <p>{open.map((s) => s.title).join(" · ")} {data.running ? "- queued in automatic setup." : "- not done yet."}</p>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ "Run: …" hints → buttons */
interface Hint { task: string; params?: Record<string, unknown>; label: string; module: Module }

const CLI: [RegExp, (m: RegExpMatchArray) => Hint][] = [
  [/spaceml\.telemetry\.data download/, () => ({ task: "telemetry.download", label: "Download telemetry", module: "telemetry" })],
  [/spaceml\.telemetry\.train(?:.*--channels\s+([A-Z]-\d+(?:\s+[A-Z]-\d+)*))?/, (m) => ({ task: "telemetry.train", params: m[1] ? { channels: m[1] } : {}, label: m[1] ? `Train ${m[1]}` : "Train telemetry models", module: "telemetry" })],
  [/spaceml\.exoplanet\.data tce/, () => ({ task: "exoplanet.catalog", label: "Download the Kepler catalog", module: "exoplanet" })],
  [/spaceml\.exoplanet\.data lightcurves/, () => ({ task: "exoplanet.lightcurves", label: "Download light curves", module: "exoplanet" })],
  [/spaceml\.exoplanet\.train/, () => ({ task: "exoplanet.train", label: "Train the exoplanet model", module: "exoplanet" })],
  [/spaceml\.spaceweather\.data download(?:\s+--years\s+([\d\s]+))?/, (m) => ({ task: "spaceweather.download", params: m[1] ? { years: m[1].trim().split(/\s+/).map(Number) } : {}, label: "Download OMNI2 data", module: "spaceweather" })],
  [/spaceml\.spaceweather\.forecast/, () => ({ task: "spaceweather.train", label: "Train the Dst forecaster", module: "spaceweather" })],
  [/spaceml\.mars\.data samples/, () => ({ task: "mars.samples", label: "Prepare Mars samples", module: "mars" })],
  [/spaceml\.mars\.train/, () => ({ task: "mars.train_quick", label: "Train the Mars model", module: "mars" })],
  [/spaceml\.mars\.traverse download/, () => ({ task: "mars.traverse", label: "Download Curiosity waypoints", module: "mars" })],
];
const SETUP: Record<string, Hint> = {
  telemetry: { task: "telemetry.train", params: {}, label: "Train telemetry models", module: "telemetry" },
  exoplanet: { task: "exoplanet.setup", params: {}, label: "Download data & train the exoplanet model", module: "exoplanet" },
  mars: { task: "mars.train_quick", params: {}, label: "Train the Mars terrain model", module: "mars" },
  spaceweather: { task: "spaceweather.setup", params: {}, label: "Download data & train the forecaster", module: "spaceweather" },
};

export function hintToTask(cmd: string): Hint | null {
  const c = cmd.trim();
  let m = c.match(/^task:([\w.]+)(?:\?(.*))?$/);
  if (m) {
    const params = Object.fromEntries(new URLSearchParams(m[2] ?? ""));
    return { task: m[1], params, label: "Fix it now", module: m[1].split(".")[0] as Module };
  }
  m = c.match(/^setup:(\w+)/);
  if (m && SETUP[m[1]]) return SETUP[m[1]];
  for (const [re, f] of CLI) {
    const x = c.match(re);
    if (x) return f(x);
  }
  return null;
}

export function HintAction({ cmd }: { cmd: string }) {
  const h = hintToTask(cmd);
  if (!h) return null;
  return <JobButton task={h.task} params={h.params} module={h.module} label={h.label} className="btn-primary px-3 py-1.5 text-xs" />;
}

/* ------------------------------------------------------------------ Mars model upload */
export function MarsModelUpload({ onDone }: { onDone?: () => void }) {
  const ref = useRef<HTMLInputElement>(null);
  const [state, setState] = useState<"idle" | "busy" | "ok" | "err">("idle");
  const [msg, setMsg] = useState("");
  async function up(f: File) {
    setState("busy"); setMsg("");
    const fd = new FormData();
    fd.append("file", f);
    try {
      const r = await api<{ version: string; has_metrics: boolean }>("/api/models/mars/upload", { method: "POST", body: fd });
      setState("ok"); setMsg(`Installed ${r.version}${r.has_metrics ? "" : " - evaluate it from Data & Training to see its metrics"}`);
      onDone?.();
    } catch (e) { setState("err"); setMsg((e as Error).message); }
  }
  return (
    <div className="space-y-1">
      <WriteButton className="btn-ghost px-3 py-1.5 text-xs" disabled={state === "busy"} onClick={() => ref.current?.click()}>
        {state === "busy" ? <Spinner /> : <Upload className="h-3.5 w-3.5" />} Upload a trained model (.zip)
      </WriteButton>
      <FileInput ref={ref} type="file" accept=".zip,application/zip" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) up(f); e.target.value = ""; }} />
      {msg && <p className={`text-[11px] ${state === "err" ? "text-rose-600" : "text-emerald-700"}`}>{msg}</p>}
    </div>
  );
}
