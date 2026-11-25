import { WriteButton } from "../lib/capabilities";
import { AnimatePresence, motion } from "framer-motion";
import { AlertTriangle, CheckCircle2, ChevronDown, CircleStop, Play, RotateCcw, TerminalSquare, XCircle } from "lucide-react";
import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { api } from "../lib/api";
import { duration, timeAgo } from "../lib/format";
import type { Job, Module } from "../lib/types";
import { Spinner } from "./ui";

interface JobsCtx { jobs: Job[]; refresh: () => void; start: (task: string, params?: Record<string, unknown>) => Promise<Job> }
const Ctx = createContext<JobsCtx>({ jobs: [], refresh: () => {}, start: async () => { throw new Error("no provider"); } });

export function JobsProvider({ children }: { children: ReactNode }) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const prevActive = useRef<Set<string>>(new Set());
  const refresh = useCallback(async () => {
    try {
      const j = await api<Job[]>("/api/jobs?limit=40");
      setJobs(j);
      const active = new Set(j.filter((x) => x.status === "running" || x.status === "queued").map((x) => x.id));
      const finished = j.filter((job) => prevActive.current.has(job.id) && !active.has(job.id)).map((job) => job.id);
      if (finished.length) window.dispatchEvent(new CustomEvent("spaceml:job-finished", { detail: finished }));
      prevActive.current = new Set([...active, ...[...prevActive.current].filter((id) => !j.some((job) => job.id === id))]);
    } catch { /* API offline: the dock shows it */ }
  }, []);
  const anyActive = jobs.some((j) => j.status === "running" || j.status === "queued");
  useEffect(() => {
    refresh();
    const t = window.setInterval(refresh, anyActive ? 1500 : 8000);
    return () => window.clearInterval(t);
  }, [refresh, anyActive]);
  const start = useCallback(async (task: string, params: Record<string, unknown> = {}) => {
    const j = await api<Job>("/api/jobs", { method: "POST", body: JSON.stringify({ task, params }) });
    prevActive.current.add(j.id);
    setJobs((current) => [j, ...current.filter((job) => job.id !== j.id)]);
    void refresh();
    return j;
  }, [refresh]);
  return <Ctx.Provider value={{ jobs, refresh, start }}>{children}</Ctx.Provider>;
}

export const useJobs = () => useContext(Ctx);

/** Re-run `fn` whenever a background job finishes (e.g. reload dataset stats). */
export function useOnJobFinished(fn: () => void) {
  useEffect(() => {
    const h = () => fn();
    window.addEventListener("spaceml:job-finished", h);
    return () => window.removeEventListener("spaceml:job-finished", h);
  }, [fn]);
}

export function activeJob(jobs: Job[], module: Module) {
  return jobs.find((j) => j.module === module && (j.status === "running" || j.status === "queued"));
}

export function ProgressBar({ value, color = "from-violet-500 to-fuchsia-500" }: { value: number | null; color?: string }) {
  const indeterminate = value == null || value <= 0;
  return (
    <div className="relative h-2 overflow-hidden rounded-full bg-white/70">
      {indeterminate ? (
        <motion.div className={`absolute inset-y-0 w-1/3 rounded-full bg-gradient-to-r ${color}`}
          animate={{ x: ["-100%", "300%"] }} transition={{ duration: 1.4, repeat: Infinity, ease: "easeInOut" }} />
      ) : (
        <motion.div className={`h-full rounded-full bg-gradient-to-r ${color}`} initial={false}
          animate={{ width: `${Math.max(2, value * 100)}%` }} transition={{ type: "spring", stiffness: 120, damping: 20 }} />
      )}
    </div>
  );
}

function LogConsole({ jobId, live }: { jobId: string; live: boolean }) {
  const [lines, setLines] = useState<string[]>([]);
  const box = useRef<HTMLPreElement>(null);
  useEffect(() => {
    let stop = false;
    const tick = async () => {
      try {
        const j = await api<Job>(`/api/jobs/${jobId}?lines=300`);
        if (!stop) setLines(j.log ?? []);
      } catch { /* ignore */ }
    };
    tick();
    const t = live ? window.setInterval(tick, 1500) : undefined;
    return () => { stop = true; window.clearInterval(t); };
  }, [jobId, live]);
  useEffect(() => { box.current?.scrollTo({ top: box.current.scrollHeight }); }, [lines]);
  return (
    <pre ref={box} className="scroll-thin max-h-64 overflow-auto rounded-2xl bg-slate-900/[0.86] p-3 font-mono text-[11px] leading-relaxed text-slate-100">
      {lines.length ? lines.map((l, i) => (
        <div key={i} className={l.startsWith("$ ") ? "text-cyan-300" : l.startsWith("::step::") ? "text-violet-300" : /error|failed|traceback/i.test(l) ? "text-rose-300" : ""}>
          {l.replace(/^::step::/, "▸ ").replace(/^::error::/, "✖ ")}
        </div>
      )) : <span className="text-slate-400">Waiting for output…</span>}
    </pre>
  );
}

const JOB_LABEL: Record<string, string> = {
  succeeded: "completed", failed: "failed", cancelled: "cancelled", interrupted: "interrupted (server stopped)",
  source_unavailable: "source unavailable",
};

export function JobCard({ job, compact = false }: { job: Job; compact?: boolean }) {
  const { refresh } = useJobs();
  const live = job.status === "running" || job.status === "queued";
  const [open, setOpen] = useState(false);
  void compact;
  const elapsed = job.started_at ? (job.finished_at ? new Date(job.finished_at).getTime() : Date.now()) - new Date(job.started_at).getTime() : null;
  return (
    <div className="glass-inset space-y-2 p-3">
      <div className="flex items-center gap-2">
        {live ? <Spinner className="h-4 w-4 text-violet-600" /> : job.status === "succeeded" ? <CheckCircle2 className="h-4 w-4 text-emerald-500" />
          : job.status === "failed" || job.status === "cancelled" ? <XCircle className="h-4 w-4 text-rose-500" /> : <AlertTriangle className="h-4 w-4 text-amber-500" />}
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm font-semibold">{job.title}</div>
          <div className="truncate text-[11px] text-ink-soft">{live ? (job.message || job.step || "starting…") : `${JOB_LABEL[job.status] ?? job.status} · ${timeAgo(job.finished_at ?? job.created_at)}${job.finished_at ? ` · ${duration(elapsed)}` : ""}`}</div>
          {live && elapsed != null && (job.progress ?? 0) > 0.02 && (
            <div className="font-mono text-[10px] text-violet-600 dark:text-violet-300">
              {Math.round((job.progress ?? 0) * 100)}% · {duration(elapsed)} elapsed · ~{duration(Math.max(0, elapsed * (1 / (job.progress ?? 1) - 1)))} left
            </div>)}
        </div>
        {live && (
          <WriteButton className="rounded-xl p-1.5 text-ink-faint transition hover:bg-rose-50 hover:text-rose-600" title="Cancel"
            onClick={async () => { await api(`/api/jobs/${job.id}/cancel`, { method: "POST" }); refresh(); }}>
            <CircleStop className="h-4 w-4" />
          </WriteButton>
        )}
        {!live && job.status !== "succeeded" && (
          <WriteButton className="rounded-xl p-1.5 text-ink-faint transition hover:bg-violet-50 hover:text-violet-600" title="Retry with the same settings"
            onClick={async () => { try { await api(`/api/jobs/${job.id}/retry`, { method: "POST" }); } catch { /* busy */ } refresh(); }}>
            <RotateCcw className="h-4 w-4" />
          </WriteButton>
        )}
        <button className="rounded-xl p-1.5 text-ink-faint transition hover:bg-white hover:text-ink" onClick={() => setOpen((o) => !o)} title={open ? "Hide log" : "Show log"}>
          <ChevronDown className={`h-4 w-4 transition ${open ? "rotate-180" : ""}`} />
        </button>
      </div>
      {live && <ProgressBar value={job.progress} />}
      {!live && job.status !== "succeeded" && job.error && (
        <p className={`rounded-xl px-3 py-2 text-xs ${job.status === "failed" ? "bg-rose-50/80 text-rose-800" : "bg-amber-50/80 text-amber-900"}`}>
          {job.status === "source_unavailable" && <b>Source unavailable - nothing was published. </b>}{job.error}
        </p>)}
      <AnimatePresence initial={false}>
        {open && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
            <div className="mb-1.5 flex items-center gap-1.5 text-[11px] text-ink-faint"><TerminalSquare className="h-3 w-3" /><code className="truncate">{job.command}</code></div>
            <LogConsole jobId={job.id} live={live} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export function JobButton({ task, params, label, module, className = "btn-primary", disabled }: {
  task: string; params?: Record<string, unknown>; label: ReactNode; module: Module; className?: string; disabled?: boolean;
}) {
  const { jobs, start } = useJobs();
  const busy = activeJob(jobs, module);
  const [err, setErr] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  return (
    <div className="space-y-1">
      <WriteButton operation={/train|retrain|setup|resume/.test(task) ? "train" : "write"} className={className} disabled={disabled || !!busy || pending}
        onClick={async () => { setErr(null); setPending(true); try { await start(task, params); } catch (e) { setErr((e as Error).message); } finally { setPending(false); } }}>
        {pending || busy?.task === task ? <Spinner /> : <Play className="h-4 w-4" />} {label}
      </WriteButton>
      {err && <p className="text-[11px] text-rose-600">{err}</p>}
    </div>
  );
}
