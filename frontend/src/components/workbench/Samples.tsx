import { useMemo, useState } from "react";
import Plot from "../LazyPlot";
import { Glass, Kpi, Segmented, Spinner } from "../ui";
import { useApi } from "../../lib/api";
import { num, pct } from "../../lib/format";
import type { ClassificationResult, ClassProfile, SkillScores, WbDataset } from "../../lib/types";
import { Findings } from "./Insights";

/** Flare classes from quiet to strongest; any other labels get the remaining colours in order. */
const CLASS_COLORS: Record<string, string> = { Q: "#64748b", A: "#22c55e", B: "#0ea5e9", C: "#f59e0b", M: "#ef4444", X: "#a855f7" };
const SPARE = ["#0891b2", "#7c3aed", "#ea580c", "#16a34a", "#db2777", "#4f46e5"];
const colour = (c: string, i: number) => CLASS_COLORS[c] ?? SPARE[i % SPARE.length];
const big = (x: number | null | undefined) => (x == null ? "—" : Math.abs(x) >= 1e5 ? x.toExponential(2) : num(x, 3));

function ProfileChart({ p, height = 230, cadenceMin }: { p: ClassProfile; height?: number; cadenceMin?: number }) {
  const x = Array.from({ length: p.steps }, (_, i) => (cadenceMin ? (i - p.steps + 1) * cadenceMin / 60 : i));
  const traces = p.classes.flatMap((c, i) => [
    { x, y: c.q75, type: "scatter", mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false },
    { x, y: c.q25, type: "scatter", mode: "lines", line: { width: 0 }, fill: "tonexty", hoverinfo: "skip", showlegend: false,
      fillcolor: `${colour(c.class, i)}22` },
    { x, y: c.median, type: "scatter", mode: "lines", name: `${c.class} (${c.n})`, line: { width: 2, color: colour(c.class, i) } },
  ]);
  return (
    <Plot height={height} data={traces as Plotly.Data[]}
      layout={{ margin: { l: 56, r: 8, t: 8, b: 36 }, legend: { orientation: "h", y: 1.12 },
        xaxis: { title: { text: cadenceMin ? "hours before the end of the window" : "step" } },
        yaxis: { title: { text: p.param }, type: "linear" } } as Partial<Plotly.Layout>} />
  );
}

/* ---------------------------------------------------------------- dataset preview */
export function SamplesPreview({ ds }: { ds: WbDataset }) {
  const s = ds.samples!;
  const first = s.separation[0]?.param ?? s.params[0];
  const [param, setParam] = useState(first);
  const prof = useApi<ClassProfile>(`/api/workbench/datasets/${ds.id}/class-profile?param=${encodeURIComponent(param)}`, [param]);
  const total = s.n || 1;
  return (
    <div className="space-y-4">
      {s.context && <p className="text-xs leading-relaxed text-ink-soft">{s.context}</p>}
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <Kpi label="Labeled windows" value={s.n.toLocaleString()} sub={`${s.files.length} file${s.files.length > 1 ? "s" : ""}`} />
        <Kpi label="Parameters" value={s.params.length} sub={`${s.n_steps} steps each`} />
        <Kpi label="Classes" value={s.classes.length} sub={s.classes.join(" · ")} />
        <Kpi label="Missing values" value={pct(s.missing_fraction)} sub={s.skipped_records ? `${s.skipped_records} records skipped` : "all records read"} />
      </div>
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)]">
        <div className="space-y-3">
          <div className="label">Class balance</div>
          <div className="space-y-1.5">
            {s.classes.map((c, i) => (
              <div key={c} className="flex items-center gap-2 text-xs">
                <span className="w-8 font-mono font-bold">{c}</span>
                <div className="h-3 flex-1 overflow-hidden rounded-full bg-slate-200/70 dark:bg-white/10">
                  <div className="h-full rounded-full" style={{ width: `${Math.max(1.5, (s.class_counts[c] / total) * 100)}%`, background: colour(c, i) }} />
                </div>
                <span className="w-24 text-right tabular-nums">{s.class_counts[c].toLocaleString()} · {pct(s.class_counts[c] / total)}</span>
              </div>))}
          </div>
          {s.separation.length > 0 && (
            <>
              <div className="label pt-2">Parameters that best separate {s.flare ? "flaring from quiet" : "the rarest class"} windows</div>
              <div className="space-y-1">
                {s.separation.slice(0, 6).map((r) => (
                  <button key={r.param} onClick={() => setParam(r.param)}
                    className={`flex w-full items-center justify-between rounded-lg px-2 py-1 text-left text-xs transition ${r.param === param ? "bg-violet-100 text-violet-900 dark:bg-violet-500/20 dark:text-violet-100" : "hover:bg-white/60 dark:hover:bg-white/5"}`}>
                    <span className="font-mono font-semibold">{r.param}</span>
                    <span className="text-right text-ink-soft">effect size {num(r.d, 2)} · median {big(r.pos_median)} vs {big(r.neg_median)}</span>
                  </button>))}
              </div>
            </>)}
        </div>
        <div className="min-w-0 space-y-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="label">How {param} evolves by class (median, shaded 25-75 %)</div>
            <select aria-label="Parameter" className="input max-w-[12rem] px-2 py-1 text-xs" value={param} onChange={(e) => setParam(e.target.value)}>
              {s.params.map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
          </div>
          {prof.data ? <ProfileChart p={prof.data} cadenceMin={s.flare ? 12 : undefined} height={260} />
            : <div className="flex h-[260px] items-center justify-center gap-2 text-sm text-ink-soft"><Spinner /> Loading {param}…</div>}
          {prof.data?.info && <p className="text-[11px] text-ink-faint">{param}: {prof.data.info}</p>}
        </div>
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------- result viewer */
function Confusion({ s, title }: { s: SkillScores; title: string }) {
  const cells: [string, number, string][] = [
    ["Hit", s.tp, "bg-emerald-500/15 text-emerald-800 dark:text-emerald-200"], ["Missed", s.fn, "bg-rose-500/15 text-rose-800 dark:text-rose-200"],
    ["False alarm", s.fp, "bg-amber-500/15 text-amber-900 dark:text-amber-200"], ["Correct quiet", s.tn, "bg-slate-500/10"]];
  return (
    <div className="space-y-1.5">
      <div className="text-xs font-semibold">{title}</div>
      <div className="grid grid-cols-2 gap-1.5">
        {cells.map(([k, v, cls]) => (
          <div key={k} className={`rounded-xl p-2 text-center ${cls}`}>
            <div className="text-lg font-bold tabular-nums">{v}</div><div className="text-[10px] uppercase tracking-wide">{k}</div>
          </div>))}
      </div>
    </div>
  );
}

export function ClassificationViewer({ r }: { r: ClassificationResult }) {
  const [filter, setFilter] = useState<"issues" | "all">("issues");
  const rows = useMemo(() => r.samples.filter((x) => filter === "all" || x.outcome === "missed" || x.outcome === "false alarm"), [r.samples, filter]);
  const sets = [r.model, r.baseline, r.climatology];
  const metrics = [["tss", "TSS"], ["hss", "HSS"], ["recall", "Hit rate"], ["false_alarm_rate", "False-alarm rate"]] as const;
  const winner = r.model.tss > r.baseline.tss + 0.02 ? "model" : r.baseline.tss > r.model.tss + 0.02 ? "baseline" : "tie";
  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <Kpi label="Model TSS" value={num(r.model.tss, 2)} sub={r.model.auc != null ? `ROC AUC ${num(r.model.auc, 2)}` : "random forest"} accent="text-violet-600 dark:text-violet-300" />
        <Kpi label="Baseline TSS" value={num(r.baseline.tss, 2)} sub={r.baseline.name.replace("Single threshold on ", "threshold on ")} />
        <Kpi label="Flares caught" value={`${r.model.tp} / ${r.model.tp + r.model.fn}`} sub={`${r.model.fp} false alarm${r.model.fp === 1 ? "" : "s"}`} />
        <Kpi label="Windows tested" value={r.n_tested.toLocaleString()} sub={`${r.n_positive} positive · cut-off ${num(r.threshold, 2)}`} />
      </div>
      <Glass className="space-y-1 p-4 text-sm">
        <div className="font-semibold">{winner === "model" ? "The model beats the single-parameter threshold." : winner === "baseline" ? "The single-parameter threshold beats the model on this data." : "Model and threshold perform about the same."}</div>
        <p className="text-xs text-ink-soft">Target: {r.target}. Test: {r.split}. Alert cut-off {r.threshold_mode}.</p>
      </Glass>
      <div className="grid gap-4 xl:grid-cols-2">
        <Glass className="min-w-0 space-y-2 p-4">
          <div className="label">Skill scores on held-out windows</div>
          <Plot height={250} data={sets.map((s, i) => ({
            x: metrics.map(([, l]) => l), y: metrics.map(([k]) => s[k] as number), type: "bar", name: i === 0 ? "Model" : i === 1 ? "Threshold baseline" : "Always 'no flare'",
            marker: { color: ["#7c3aed", "#f59e0b", "#94a3b8"][i] } }))}
            layout={{ barmode: "group", yaxis: { range: [-0.1, 1], tickformat: ".1f" }, margin: { l: 40, r: 8, t: 8, b: 30 }, legend: { orientation: "h", y: 1.15 } } as Partial<Plotly.Layout>} />
          <p className="text-[11px] text-ink-faint">TSS = hit rate − false-alarm rate; HSS = improvement over chance. Both stay meaningful when flares are rare; accuracy does not.</p>
        </Glass>
        <Glass className="grid min-w-0 grid-cols-1 gap-4 p-4 sm:grid-cols-2">
          <Confusion s={r.model} title="Model" />
          <Confusion s={r.baseline} title="Threshold baseline" />
        </Glass>
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        <Glass className="min-w-0 space-y-2 p-4">
          <div className="label">What the model relies on (top features)</div>
          <Plot height={Math.max(220, r.importance.length * 18 + 40)} data={[{
            type: "bar", orientation: "h", x: r.importance.map((f) => f.importance).reverse(), y: r.importance.map((f) => f.feature.replace(":", " · ")).reverse(),
            marker: { color: "#7c3aed" } }]}
            layout={{ margin: { l: 150, r: 8, t: 4, b: 30 }, xaxis: { title: { text: "importance" } } } as Partial<Plotly.Layout>} />
        </Glass>
        <Glass className="min-w-0 space-y-2 p-4">
          <div className="label">Top parameters by class</div>
          {r.profiles.slice(0, 2).map((p) => <ProfileChart key={p.param} p={p} height={200} cadenceMin={r.flare ? 12 : undefined} />)}
        </Glass>
      </div>
      <Findings items={r.insights} />
      <Glass className="space-y-3 p-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="label">Windows · {filter === "issues" ? "misses and false alarms (also in Review Studio)" : "all tested windows"}</div>
          <Segmented label="Show" value={filter} onChange={setFilter} options={[{ value: "issues", label: "Issues" }, { value: "all", label: "All" }]} />
        </div>
        <div className="scroll-thin max-h-80 overflow-auto">
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-white/95 text-left text-ink-faint dark:bg-slate-900/95"><tr><th className="px-2 py-1.5">Window</th><th>Label</th><th>Probability</th><th>Outcome</th><th>File</th></tr></thead>
            <tbody>
              {rows.slice(0, 500).map((x) => (
                <tr key={x.id + x.file} className="border-t border-slate-200/60 dark:border-white/10">
                  <td className="px-2 py-1 font-mono">{x.id}</td>
                  <td><span className="chip" style={{ background: `${colour(x.label, 0)}22` }}>{x.label}</span></td>
                  <td className="tabular-nums">{num(x.prob, 2)}</td>
                  <td className={x.outcome === "missed" ? "text-rose-700 dark:text-rose-300" : x.outcome === "false alarm" ? "text-amber-800 dark:text-amber-300" : ""}>{x.outcome}</td>
                  <td className="text-ink-faint">{x.file}</td>
                </tr>))}
              {!rows.length && <tr><td colSpan={5} className="px-2 py-3 text-ink-soft">No misses or false alarms.</td></tr>}
            </tbody>
          </table>
        </div>
      </Glass>
    </div>
  );
}
