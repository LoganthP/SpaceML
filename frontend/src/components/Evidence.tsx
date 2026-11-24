import { CheckCircle2, ChevronDown, FileArchive, FileText, Fingerprint, Repeat2, ShieldCheck, XCircle } from "lucide-react";
import { useEffect, useState } from "react";
import { api, useRun } from "../lib/api";
import { num, timeAgo } from "../lib/format";
import type { Run } from "../lib/types";
import { OriginBadge, Spinner, StatusBadge } from "./ui";

interface Manifest {
  schema: string;
  origin: "computed" | "bundled";
  run: { id: string; module: string; created_at: string; repeat_of: string | null; input_label: string };
  data: { source?: string; window?: { start: string; end: string }; files?: Record<string, unknown>[]; quality?: string;
          location?: Record<string, unknown> | null; dataset?: Record<string, unknown> };
  model: { version: string; name?: string; created_at?: string; training?: Record<string, unknown>;
           heldout?: { metric: string; model: number; baseline: number } | null; eval_split?: string;
           files?: { name: string; sha256: string; bytes: number }[]; note?: string } | null;
  baseline: { name: string } | null;
  settings: Record<string, unknown>;
  reviews: { item: string; verdict: string; note: string | null }[];
  checks: { causality_check?: { passed: boolean; detail: string } };
  environment: Record<string, string>;
  repeat: { available: boolean; reason: string | null };
}

interface Repro { status: string; same_model?: boolean; compared?: number; differences?: { key: string; original: unknown; repeat: unknown }[] }

const short = (h?: unknown) => (typeof h === "string" ? `${h.slice(0, 10)}…` : "—");

function fileLabel(f: Record<string, unknown>) {
  if ("year" in f) return `OMNI2 ${f.year} · Dst ${f.dst_status}`;
  return String(f.name ?? f.file ?? "file");
}

/** Evidence for one run: where the data came from, which model, what was compared, and Repeat run. */
export function EvidencePanel({ runId, compact = false }: { runId: string; compact?: boolean }) {
  const [man, setMan] = useState<Manifest | null>(null);
  const [open, setOpen] = useState(!compact);
  const [repeatId, setRepeatId] = useState<string | null>(null);
  const [repeatErr, setRepeatErr] = useState<string | null>(null);
  const [repro, setRepro] = useState<Repro | null>(null);
  const { run: rep } = useRun(repeatId);

  useEffect(() => {
    setMan(null); setRepeatId(null); setRepro(null); setRepeatErr(null);
    api<Manifest>(`/api/runs/${runId}/manifest`).then(setMan).catch(() => setMan(null));
  }, [runId]);
  useEffect(() => {
    if (rep && rep.status !== "queued" && rep.status !== "running") {
      api<Repro>(`/api/runs/${rep.id}/reproducibility`).then(setRepro).catch(() => setRepro({ status: rep.status }));
    }
  }, [rep]);

  async function repeat() {
    setRepeatErr(null); setRepro(null);
    try {
      const r = await api<Run>(`/api/runs/${runId}/repeat`, { method: "POST" });
      setRepeatId(r.id);
    } catch (e) { setRepeatErr((e as Error).message); }
  }

  return (
    <div className="glass-inset space-y-3 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <Fingerprint className="h-4 w-4 text-violet-500" />
        <div className="font-display text-sm font-bold">Evidence</div>
        {man && <OriginBadge origin={man.origin} />}
        {man?.checks?.causality_check && (
          <span className={`chip ${man.checks.causality_check.passed ? "bg-emerald-100 text-emerald-700" : "bg-rose-100 text-rose-700"}`}
            title={man.checks.causality_check.detail}>
            <ShieldCheck className="h-3 w-3" /> {man.checks.causality_check.passed ? "no look-ahead" : "look-ahead detected"}
          </span>)}
        <div className="ml-auto flex flex-wrap gap-1.5">
          <a className="btn-ghost px-3 py-1.5 text-xs" href={`/api/runs/${runId}/evidence.zip`} download>
            <FileArchive className="h-3.5 w-3.5" /> Evidence (.zip)
          </a>
          <a className="btn-ghost px-3 py-1.5 text-xs" href={`/api/runs/${runId}/report?format=md`} download>
            <FileText className="h-3.5 w-3.5" /> Report
          </a>
          <button className="btn-ghost px-3 py-1.5 text-xs" onClick={repeat} disabled={!man?.repeat.available || (!!rep && (rep.status === "queued" || rep.status === "running"))}
            title={man?.repeat.reason ?? "Run the same input with the same model version and compare"}>
            <Repeat2 className="h-3.5 w-3.5" /> Repeat run
          </button>
          {compact && (
            <button className="rounded-xl p-1.5 text-ink-faint hover:bg-white" onClick={() => setOpen((o) => !o)} title="Details">
              <ChevronDown className={`h-4 w-4 transition ${open ? "rotate-180" : ""}`} />
            </button>)}
        </div>
      </div>

      {repeatErr && <p className="rounded-xl bg-rose-50/80 px-3 py-2 text-xs text-rose-800">{repeatErr}</p>}
      {rep && (
        <div className="flex flex-wrap items-center gap-2 rounded-xl bg-white/50 px-3 py-2 text-xs dark:bg-white/5">
          <span className="font-semibold">Repeat</span> <StatusBadge status={rep.status} />
          {repro?.status === "identical" && <span className="chip bg-emerald-100 text-emerald-700"><CheckCircle2 className="h-3 w-3" /> identical result · {repro.compared} values compared{repro.same_model ? " · same model" : ""}</span>}
          {repro?.status === "differs" && <span className="chip bg-amber-100 text-amber-800"><XCircle className="h-3 w-3" /> differs in {repro.differences?.length} value(s)</span>}
          {repro?.status === "differs" && repro.differences?.slice(0, 3).map((d) => (
            <code key={d.key} className="text-[10px]">{d.key}: {String(d.original)} → {String(d.repeat)}</code>))}
          {rep.status !== "succeeded" && rep.status !== "queued" && rep.status !== "running" && <span className="text-rose-700">{rep.error}</span>}
        </div>)}

      {!man && <div className="flex items-center gap-2 text-xs text-ink-soft"><Spinner className="h-3 w-3" /> Building manifest…</div>}
      {man && open && (
        <div className="grid gap-3 text-xs sm:grid-cols-2 xl:grid-cols-4">
          <div>
            <div className="label">Data</div>
            <div className="mt-1 font-semibold">{man.data.source ?? "—"}</div>
            {man.data.window && <div className="font-mono text-[11px] text-ink-soft">{man.data.window.start.slice(0, 16)} → {man.data.window.end.slice(0, 16)} UTC</div>}
            <ul className="mt-1 space-y-0.5 text-[11px] text-ink-soft">
              {(man.data.files ?? []).slice(0, 4).map((f, i) => (
                <li key={i} className="truncate">{fileLabel(f)} · <span className="font-mono">{short(f.cache_sha256 ?? f.sha256)}</span>
                  {"retrieved_at" in f && f.retrieved_at ? <> · {String(f.retrieved_at).slice(0, 10)}</> : null}</li>))}
              {man.data.dataset && <li>dataset sha <span className="font-mono">{short(man.data.dataset.sha256)}</span></li>}
            </ul>
          </div>
          <div>
            <div className="label">Model</div>
            {man.model ? <>
              <div className="mt-1 font-mono font-semibold">{man.model.version}</div>
              <div className="text-[11px] text-ink-soft">{man.model.eval_split ?? man.model.name}</div>
              {man.model.heldout && <div className="text-[11px]">held-out {man.model.heldout.metric}: <b>{num(man.model.heldout.model)}</b> (baseline {num(man.model.heldout.baseline)})</div>}
              <div className="text-[11px] text-ink-faint">{man.model.files?.length ?? 0} files hashed (sha256)</div>
            </> : <div className="mt-1 text-ink-soft">no trained model needed</div>}
          </div>
          <div>
            <div className="label">Compared with</div>
            <div className="mt-1 font-semibold">{man.baseline?.name ?? "—"}</div>
            <div className="label mt-2">Settings</div>
            <div className="font-mono text-[11px] text-ink-soft">{Object.entries(man.settings).slice(0, 4).map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`).join(" · ") || "defaults"}</div>
          </div>
          <div>
            <div className="label">Review & reproducibility</div>
            <div className="mt-1">{man.reviews.length} reviewer verdict(s){man.reviews.some((r) => r.note) ? " with notes" : ""}</div>
            <div className="text-[11px] text-ink-soft">created {timeAgo(man.run.created_at)}{man.run.repeat_of ? ` · repeat of ${man.run.repeat_of}` : ""}</div>
            <div className="font-mono text-[10px] text-ink-faint">{Object.entries(man.environment).map(([k, v]) => `${k} ${v}`).join(" · ")}</div>
          </div>
        </div>)}
    </div>
  );
}
