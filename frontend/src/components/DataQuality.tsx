import { AlertTriangle, CheckCircle2, CloudOff, Database, Gauge } from "lucide-react";
import type { ReactNode } from "react";

export interface QualityColumn {
  name: string; valid_fraction: number; missing: number; gaps: number; longest_gap: number; stale_samples: number;
  status: string; longest_gap_at?: [string, string];
}
export interface Quality {
  status: "ready" | "degraded" | "insufficient_data" | "source_unavailable" | "not_downloaded" | string;
  samples: number; columns: QualityColumn[]; findings: string[]; missing_years?: number[];
  provenance?: { year?: number; source?: string; retrieved_at?: string; dst_status?: string; cache_sha256?: string; note?: string }[];
}

const META: Record<string, { label: string; cls: string; icon: ReactNode; text: string }> = {
  ready: { label: "Ready", cls: "bg-emerald-100 text-emerald-700", icon: <CheckCircle2 className="h-3.5 w-3.5" />,
           text: "Enough valid data - results are meaningful." },
  degraded: { label: "Usable with gaps", cls: "bg-amber-100 text-amber-800", icon: <AlertTriangle className="h-3.5 w-3.5" />,
              text: "Some readings are missing or repeated; affected hours are marked in the results." },
  insufficient_data: { label: "Insufficient data", cls: "bg-rose-100 text-rose-700", icon: <AlertTriangle className="h-3.5 w-3.5" />,
                       text: "Too little valid data - an analysis would not mean anything." },
  source_unavailable: { label: "Source unavailable", cls: "bg-orange-100 text-orange-800", icon: <CloudOff className="h-3.5 w-3.5" />,
                        text: "The archive could not be reached. Nothing is shown as a result until the data arrives." },
  not_downloaded: { label: "Not downloaded yet", cls: "bg-slate-100 text-slate-700", icon: <Database className="h-3.5 w-3.5" />,
                    text: "The data for this window is not on this machine yet." },
};

export function QualityBadge({ status }: { status: string }) {
  const m = META[status] ?? META.degraded;
  return <span className={`chip ${m.cls}`}>{m.icon}{m.label}</span>;
}

/** Pre-flight data quality: missing readings, gaps, stale values, coverage, provenance. */
export function DataQualityPanel({ q, title = "Data quality", unit = "h", children }: { q: Quality; title?: string; unit?: string; children?: ReactNode }) {
  const m = META[q.status] ?? META.degraded;
  return (
    <div className="glass-inset space-y-3 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <Gauge className="h-4 w-4 text-sky-500" />
        <div className="font-display text-sm font-bold">{title}</div>
        <QualityBadge status={q.status} />
        {q.samples > 0 && <span className="text-[11px] text-ink-soft">{q.samples.toLocaleString()} {unit === "h" ? "hourly samples" : "samples"}</span>}
      </div>
      <p className="text-xs text-ink-soft">{m.text}</p>
      {q.columns.length > 0 && (
        <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 sm:grid-cols-3">
          {q.columns.map((c) => (
            <div key={c.name} title={c.longest_gap_at ? `longest gap ${c.longest_gap_at[0].slice(0, 16)} → ${c.longest_gap_at[1].slice(0, 16)}` : undefined}>
              <div className="flex items-baseline justify-between text-[11px]">
                <span className="font-semibold">{c.name}</span>
                <span className={c.status === "ready" ? "text-ink-soft" : c.status === "degraded" ? "text-amber-600" : "text-rose-600"}>
                  {(c.valid_fraction * 100).toFixed(c.valid_fraction > 0.999 ? 0 : 1)}%
                </span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-white/70 dark:bg-white/10">
                <div className={`h-full rounded-full ${c.status === "ready" ? "bg-emerald-400" : c.status === "degraded" ? "bg-amber-400" : "bg-rose-400"}`}
                  style={{ width: `${c.valid_fraction * 100}%` }} />
              </div>
              <div className="text-[10px] text-ink-faint">
                {c.missing ? `${c.missing} missing · longest gap ${c.longest_gap} ${unit}` : "no gaps"}{c.stale_samples ? ` · ${c.stale_samples} stale` : ""}
              </div>
            </div>))}
        </div>)}
      {q.findings.length > 0 && (
        <ul className="list-disc space-y-0.5 pl-4 text-[11px] text-ink-soft">{q.findings.slice(0, 5).map((f) => <li key={f}>{f}</li>)}</ul>)}
      {!!q.provenance?.length && (
        <div className="text-[10px] text-ink-faint">
          {q.provenance.map((p) => (
            <div key={p.year ?? p.source}>{p.year ? `OMNI2 ${p.year}` : p.source} · Dst {p.dst_status ?? "—"}
              {p.retrieved_at ? ` · retrieved ${p.retrieved_at.slice(0, 10)}` : ""}{p.cache_sha256 ? ` · sha256 ${p.cache_sha256.slice(0, 10)}…` : ""}{p.note ? ` · ${p.note}` : ""}</div>))}
        </div>)}
      {children}
    </div>
  );
}
