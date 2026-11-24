import { BarChart3, TrendingDown, TrendingUp } from "lucide-react";
import { Link } from "react-router-dom";
import Plot from "../components/LazyPlot";
import { ErrorNotice, Glass, IconBadge, Page, Spinner } from "../components/ui";
import { useApi } from "../lib/api";
import { MODULE_META, num, pct, timeAgo } from "../lib/format";
import type { MetricCell, ModuleEvaluation } from "../lib/types";

const isRatio = (name: string) => /precision|recall|f1|iou|accuracy|auc/i.test(name);

function cellValue(v: number | string, name: string) {
  if (typeof v === "string") return v;
  return isRatio(name) && !/auc/i.test(name) ? pct(v) : num(v, 3);
}

function MetricBar({ m, color }: { m: MetricCell; color: string }) {
  const numeric = typeof m.model === "number" && typeof m.baseline === "number" && isRatio(m.name);
  const better = typeof m.model === "number" && typeof m.baseline === "number"
    ? (m.lower_is_better ? m.model <= m.baseline : m.model >= m.baseline) : null;
  return (
    <div className="grid grid-cols-[150px_1fr] items-center gap-3 py-1.5 text-sm">
      <div className="text-xs font-semibold text-ink-soft">{m.name}</div>
      <div className="space-y-1">
        <div className="flex items-center gap-2">
          {numeric && <div className="h-2 flex-1 overflow-hidden rounded-full bg-white/70"><div className="h-full rounded-full" style={{ width: `${Math.max(0, Math.min(1, m.model as number)) * 100}%`, background: color }} /></div>}
          <span className="w-20 text-right font-semibold tabular-nums">{cellValue(m.model, m.name)}</span>
          {better != null && (better ? <TrendingUp className="h-3.5 w-3.5 text-emerald-500" /> : <TrendingDown className="h-3.5 w-3.5 text-rose-500" />)}
        </div>
        <div className="flex items-center gap-2 text-ink-faint">
          {numeric && <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-white/60"><div className="h-full rounded-full bg-slate-400/70" style={{ width: `${Math.max(0, Math.min(1, m.baseline as number)) * 100}%` }} /></div>}
          <span className="w-20 text-right text-xs tabular-nums">{cellValue(m.baseline, m.name)}</span>
          {better != null && <span className="w-3.5" />}
        </div>
      </div>
    </div>
  );
}

function ModuleCard({ e }: { e: ModuleEvaluation }) {
  const meta = MODULE_META[e.module];
  const a = e.active;
  return (
    <Glass className="p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="label" style={{ color: meta.color }}>{meta.title}</div>
          <h2 className="mt-1 text-lg font-bold">{a ? a.model_name : "Not evaluated yet"}</h2>
          {a && <p className="text-xs text-ink-soft">vs <b>{a.baseline_name}</b></p>}
        </div>
        {a && (
          <div className="text-right">
            <div className="label">{a.headline.metric}</div>
            <div className="font-display text-3xl font-bold" style={{ color: meta.color }}>{e.module === "exoplanet" ? num(a.headline.model, 3) : pct(a.headline.model)}</div>
            <div className="text-xs text-ink-soft">baseline {e.module === "exoplanet" ? num(a.headline.baseline, 3) : pct(a.headline.baseline)}</div>
          </div>
        )}
      </div>
      {!a && (
        <p className="mt-3 text-sm text-ink-soft">Train and evaluate the model to populate this card - see <Link className="font-semibold text-violet-600" to="/datasets">Datasets & Models</Link>.</p>
      )}
      {a && (
        <>
          <p className="mt-2 rounded-xl bg-white/50 px-3 py-2 text-xs text-ink-soft"><b>Held-out data:</b> {a.eval_split} · version <span className="font-mono">{a.version}</span> · {timeAgo(a.created_at)}</p>
          <div className="mt-2 flex items-center gap-4 text-[11px] text-ink-faint">
            <span className="flex items-center gap-1"><span className="h-2 w-4 rounded-full" style={{ background: meta.color }} /> model</span>
            <span className="flex items-center gap-1"><span className="h-2 w-4 rounded-full bg-slate-400/70" /> baseline</span>
          </div>
          <div className={`mt-2 grid gap-4 ${a.rows.length > 1 ? "lg:grid-cols-2" : ""}`}>
            {a.rows.map((r) => (
              <div key={r.group} className="glass-inset p-3">
                <div className="mb-1 text-sm font-bold">{r.group}</div>
                {r.metrics.map((m) => <MetricBar key={m.name} m={m} color={meta.color} />)}
              </div>
            ))}
          </div>
          {a.curves && (
            <div className="mt-4 grid gap-4 lg:grid-cols-2">
              {(["roc", "pr"] as const).map((k) => (
                <div key={k} className="glass-inset p-2">
                  <div className="label px-2 pt-1">{k === "roc" ? "ROC curve" : "Precision-recall (planet class)"}</div>
                  <Plot height={260} data={[
                    { x: a.curves![k].model.map((p) => p[0]), y: a.curves![k].model.map((p) => p[1]), type: "scatter", mode: "lines", name: "CNN", line: { color: meta.color, width: 2.5 } },
                    { x: a.curves![k].baseline.map((p) => p[0]), y: a.curves![k].baseline.map((p) => p[1]), type: "scatter", mode: "lines", name: "Baseline", line: { color: "#94a3b8", width: 2, dash: "dash" } },
                  ]} layout={{ xaxis: { title: { text: k === "roc" ? "False positive rate" : "Recall" }, range: [0, 1] }, yaxis: { title: { text: k === "roc" ? "True positive rate" : "Precision" }, range: [0, 1.02] } }} />
                </div>
              ))}
            </div>
          )}
          {a.struggles?.length > 0 && (
            <div className="mt-4">
              <div className="label mb-1.5">Where it struggled</div>
              <ul className="grid gap-1.5 sm:grid-cols-2">
                {a.struggles.slice(0, 10).map((s) => <li key={s} className="rounded-xl bg-rose-50/70 px-3 py-1.5 text-xs text-rose-900">{s}</li>)}
              </ul>
              {a.struggles.length > 10 && <p className="mt-1 text-[11px] text-ink-faint">+{a.struggles.length - 10} more</p>}
            </div>
          )}
          {e.history.length > 1 && (
            <div className="mt-4">
              <div className="label mb-1.5">Version history</div>
              <div className="flex flex-wrap gap-2">
                {e.history.map((h) => (
                  <span key={h.version} className={`chip ${h.version === e.active_version ? "bg-white shadow" : "bg-white/50"}`}>
                    <span className="font-mono">{h.version}</span>
                    {h.headline && <span className="text-ink-soft">{h.headline.metric} {e.module === "exoplanet" ? num(h.headline.model, 3) : pct(h.headline.model)}</span>}
                  </span>))}
              </div>
            </div>
          )}
        </>
      )}
    </Glass>
  );
}

export default function Evaluation() {
  const { data, error, loading } = useApi<ModuleEvaluation[]>("/api/evaluation");
  return (
    <Page title="Model evaluation"
      subtitle="Each model is measured on held-out data against a simple baseline, with the metric that suits the task: event-level precision/recall, ROC-AUC, and IoU per terrain class."
      icon={<IconBadge gradient="from-indigo-500 to-violet-600"><BarChart3 className="h-6 w-6" /></IconBadge>}>
      {error && <ErrorNotice message={error.message} />}
      {loading && <div className="flex items-center gap-2 text-sm text-ink-soft"><Spinner /> Loading…</div>}
      <div className="space-y-4">{data?.map((e) => <ModuleCard key={e.module} e={e} />)}</div>
    </Page>
  );
}
