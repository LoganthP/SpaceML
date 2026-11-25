import { WriteButton, useCapabilities } from "../lib/capabilities";
import { EvidencePanel } from "../components/Evidence";
import { Activity, ArrowRight, FileSpreadsheet, FlaskConical, Image as ImageIcon, Orbit, Play, RefreshCw, SunMedium, Trash2 } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import Plot from "../components/LazyPlot";
import { DropZone } from "../components/RunTools";
import ParamForm, { initialValues, TYPE_CHIP } from "../components/workbench/ParamForm";
import { ChangeViewer, TimelineViewer } from "../components/workbench/Viewers";
import { Findings, InsightsViewer, ProfileCard, type Finding, type InsightsResult } from "../components/workbench/Insights";
import { ErrorNotice, Glass, IconBadge, Notice, Page, RunProgress, Segmented, Spinner, StatusBadge } from "../components/ui";
import { api, createRun, useApi, useRun } from "../lib/api";
import { num, pct, timeAgo, TERRAIN_COLORS } from "../lib/format";
import type { AnalysisSpec, ChangeResult, ClassificationResult, Run, TimelineResult, WbDataset, WbDatasetListItem, WorkbenchResult } from "../lib/types";
import { OverlayCanvas } from "./Mars";
import { ClassificationViewer, SamplesPreview } from "../components/workbench/Samples";

const KIND = {
  timeseries: { icon: Activity, label: "Time series", cls: "bg-tele-50 text-tele-700" },
  images: { icon: ImageIcon, label: "Images", cls: "bg-mars-50 text-mars-700" },
  lightcurve: { icon: Orbit, label: "Light curves", cls: "bg-exo-50 text-exo-700" },
  samples: { icon: SunMedium, label: "Labeled windows", cls: "bg-amber-50 text-amber-800" },
} as const;

/* ---------------------------------------------------------------- data preview */
function TablePreview({ ds }: { ds: WbDataset }) {
  const [tid, setTid] = useState(ds.tables![0].id);
  const t = ds.tables!.find((x) => x.id === tid)!;
  const cols = t.suggested.sensor_cols.slice(0, 4);
  const prev = useApi<{ step: number; series: Record<string, (number | null)[]> }>(
    `/api/workbench/datasets/${ds.id}/series?table=${tid}&columns=${encodeURIComponent(cols.join(","))}`, [tid]);
  return (
    <div className="space-y-3">
      {ds.tables!.length > 1 && (
        <select className="input max-w-sm" value={tid} onChange={(e) => setTid(e.target.value)}>
          {ds.tables!.map((x) => <option key={x.id} value={x.id}>{x.file}</option>)}
        </select>)}
      <div className="grid gap-3 xl:grid-cols-2">
        <div className="scroll-thin max-h-72 overflow-auto rounded-2xl bg-white/50">
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-white/90 text-left text-ink-faint"><tr><th className="px-3 py-2">Column</th><th>Type</th><th>Missing</th><th>Range</th></tr></thead>
            <tbody>
              {t.columns.map((c) => (
                <tr key={c.name} className="border-t border-white/70">
                  <td className="px-3 py-1.5 font-semibold">{c.name}
                    {c.name === t.suggested.time_col && <span className="ml-1 chip bg-violet-50 text-violet-700">time</span>}
                    {c.name === t.suggested.label_col && <span className="ml-1 chip bg-emerald-50 text-emerald-700">labels</span>}</td>
                  <td><span className={`chip ${TYPE_CHIP[c.type]}`}>{c.type}</span></td>
                  <td className={c.missing ? "text-amber-700" : "text-ink-faint"}>{c.missing ? pct(c.missing / t.rows) : "0"}</td>
                  <td className="text-ink-soft">{c.type === "text" ? (c.examples ?? []).join(", ").slice(0, 40) : `${typeof c.min === "number" ? num(c.min, 3) : c.min ?? "—"} … ${typeof c.max === "number" ? num(c.max, 3) : c.max ?? "—"}`}</td>
                </tr>))}
            </tbody>
          </table>
        </div>
        <div className="rounded-2xl bg-white/40 p-2">
          {prev.data ? (
            <Plot height={260} data={Object.entries(prev.data.series).map(([k, v], i) => ({
              y: v, x: v.map((_, j) => j * prev.data!.step), type: "scattergl", mode: "lines", name: k,
              line: { width: 1, color: ["#0891b2", "#7c3aed", "#ea580c", "#16a34a"][i % 4] } }))}
              layout={{ margin: { l: 44, r: 10, t: 24, b: 34 }, xaxis: { title: { text: "row" } } }} />
          ) : <div className="flex h-[260px] items-center justify-center gap-2 text-sm text-ink-soft"><Spinner /> Loading preview…</div>}
        </div>
      </div>
      <div className="scroll-thin overflow-x-auto rounded-2xl bg-white/40">
        <table className="w-full text-[11px]">
          <thead><tr className="text-left text-ink-faint">{t.columns.map((c) => <th key={c.name} className="whitespace-nowrap px-2 py-1.5">{c.name}</th>)}</tr></thead>
          <tbody>{t.head.map((row, i) => <tr key={i} className="border-t border-white/70">{row.map((v, j) => <td key={j} className="whitespace-nowrap px-2 py-1 font-mono">{v ?? "∅"}</td>)}</tr>)}</tbody>
        </table>
      </div>
      <p className="text-[11px] text-ink-faint">{t.rows.toLocaleString()} rows · first rows shown · types and the time/label columns were detected automatically and can be changed below.</p>
    </div>
  );
}

function ImagePreview({ ds }: { ds: WbDataset }) {
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-5">
      {ds.images!.map((im) => (
        <div key={im.id} className="glass-inset overflow-hidden">
          <img src={`/api/workbench/datasets/${ds.id}/images/${im.id}/thumb`} alt={im.file} className="aspect-square w-full object-cover" />
          <div className="space-y-0.5 p-2 text-[10px]">
            <div className="truncate text-xs font-semibold" title={im.file}>{im.file}</div>
            <div className="text-ink-soft">{im.width}×{im.height} · {im.bands} band{im.bands > 1 ? "s" : ""}</div>
            <div className="flex flex-wrap gap-1">
              {im.geo.georeferenced && <span className="chip bg-sky-100 text-sky-700" title={im.geo.crs}>GeoTIFF{im.geo.res ? ` · ${num(im.geo.res[0], 3)} m` : ""}</span>}
              {im.mask && <span className="chip bg-emerald-100 text-emerald-700">mask</span>}
            </div>
          </div>
        </div>))}
    </div>
  );
}

/* ---------------------------------------------------------------- results */
function TerrainView({ r, runId }: { r: Extract<WorkbenchResult, { viewer: "segmentation" }>; runId: string }) {
  const [opacity, setOpacity] = useState(0.55);
  const enabled = { soil: true, bedrock: true, sand: true, "big rock": true };
  return (
    <div className="grid gap-4 2xl:grid-cols-[1fr_300px]">
      <Glass className="space-y-3 p-4">
        <OverlayCanvas imageUrl={`/api/runs/${runId}/files/input.png`} maskUrl={`/api/runs/${runId}/files/mask.png`} opacity={opacity} enabled={enabled} />
        <input type="range" min={0} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(+e.target.value)} className="slider" />
      </Glass>
      <Glass className="space-y-2 p-4">
        <div className="label">Terrain share</div>
        {r.classes.map((c) => {
          const [cr, cg, cb] = TERRAIN_COLORS[c];
          return (
            <div key={c}>
              <div className="flex justify-between text-xs"><span className="font-semibold capitalize">{c}</span><span>{pct(r.class_shares[c])}</span></div>
              <div className="mt-1 h-2 rounded-full bg-white/70"><div className="h-full rounded-full" style={{ width: `${r.class_shares[c] * 100}%`, background: `rgb(${cr},${cg},${cb})` }} /></div>
            </div>);
        })}
        {r.evaluation && <p className="pt-2 text-xs">Mean IoU vs your mask: <b>{pct(r.evaluation.model.miou)}</b> (all-soil {pct(r.evaluation.baseline.miou)})</p>}
      </Glass>
    </div>
  );
}

function ResultView({ run }: { run: Run<WorkbenchResult> }) {
  const r = run.result!;
  return (
    <div className="space-y-4">
      <Glass className="flex flex-wrap items-center justify-between gap-3 p-4">
        <div>
          <div className="label">Result · {run.input_label}</div>
          <div className="font-display text-lg font-bold">{run.summary?.headline}</div>
          <div className="text-[11px] text-ink-soft">Run {run.id} · {timeAgo(run.created_at)} · {((run.duration_ms ?? 0) / 1000).toFixed(1)} s{run.model_version ? ` · model ${run.model_version}` : ""}</div>
        </div>
        <EvidencePanel runId={run.id} compact />
      </Glass>
      {r.viewer === "timeline" && <TimelineViewer r={r as TimelineResult} runId={run.id} />}
      {r.viewer === "change" && <ChangeViewer r={r as ChangeResult} runId={run.id} />}
      {r.viewer === "segmentation" && <TerrainView r={r as Extract<WorkbenchResult, { viewer: "segmentation" }>} runId={run.id} />}
      {(r.viewer as string) === "insights" && <InsightsViewer r={r as unknown as InsightsResult} runId={run.id} />}
      {(r.viewer as string) === "classification" && <ClassificationViewer r={r as unknown as ClassificationResult} />}
      {!["insights", "classification"].includes(r.viewer as string) && <Findings items={(r as unknown as { insights?: Finding[] }).insights} />}
    </div>
  );
}

/* ---------------------------------------------------------------- experiments for a dataset */
function Compare({ ids }: { ids: string[] }) {
  const [runs, setRuns] = useState<Run<WorkbenchResult>[] | null>(null);
  useEffect(() => {
    let alive = true;
    Promise.all(ids.map((id) => api<Run<WorkbenchResult>>(`/api/runs/${id}`))).then((r) => alive && setRuns(r)).catch(() => {});
    return () => { alive = false; };
  }, [ids.join(",")]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!runs) return <div className="flex items-center gap-2 text-sm text-ink-soft"><Spinner /> Loading runs…</div>;
  const params = runs.map((r) => (r.input_ref as { params: Record<string, unknown> }).params);
  const keys = Array.from(new Set(params.flatMap((p) => Object.keys(p))))
    .filter((k) => new Set(params.map((p) => JSON.stringify(p[k]))).size > 1);
  const label = (r: Run) => r.id.slice(0, 6);
  const ts = runs.every((r) => r.result?.viewer === "timeline");
  const ch = runs.every((r) => r.result?.viewer === "change");
  return (
    <div className="space-y-4 rounded-2xl border border-white/70 bg-white/40 p-4">
      <div className="kicker">Comparing {runs.length} runs</div>
      <div className="scroll-thin overflow-x-auto">
        <table className="w-full text-xs">
          <thead><tr className="text-left text-ink-faint"><th className="pb-2">Setting that differs</th>{runs.map((r) => <th key={r.id} className="font-mono">{label(r)}</th>)}</tr></thead>
          <tbody>
            {keys.length === 0 && <tr><td className="py-1.5 text-ink-soft" colSpan={runs.length + 1}>Same settings in every run.</td></tr>}
            {keys.map((k) => <tr key={k} className="border-t border-white/70"><td className="py-1.5 font-semibold">{k}</td>{params.map((p, i) => <td key={i} className="font-mono">{JSON.stringify(p[k])}</td>)}</tr>)}
          </tbody>
        </table>
      </div>
      {ts && (() => {
        const rs = runs.map((r) => r.result as TimelineResult);
        const hasM = rs.every((r) => r.metrics);
        return (
          <>
            {hasM && <Plot height={240} data={(["precision", "recall", "f1"] as const).flatMap((m, j) => [
              { x: runs.map(label), y: rs.map((r) => r.metrics!.robust[m]), type: "bar", name: `robust ${m}`, marker: { color: ["#db2777", "#f472b6", "#fbcfe8"][j] } },
              { x: runs.map(label), y: rs.map((r) => r.metrics!.zscore[m]), type: "bar", name: `z-score ${m}`, marker: { color: ["#d97706", "#fbbf24", "#fde68a"][j] } }])}
              layout={{ barmode: "group", yaxis: { range: [0, 1], tickformat: ".0%" }, margin: { l: 44, r: 10, t: 30, b: 30 } } as Partial<Plotly.Layout>} />}
            <div className="grid gap-3 md:grid-cols-2">
              {rs.map((r, i) => {
                const missed = r.labeled.filter((_, j) => r.metrics && !r.metrics.found_robust[j]);
                const fa = r.robust.intervals.filter((f) => r.labeled.length > 0 && !r.labeled.some((l) => f.start <= l.end && l.start <= f.end));
                return (
                  <div key={i} className="glass-inset p-3 text-xs">
                    <div className="font-mono font-bold">{label(runs[i])} · failures (robust)</div>
                    {!r.metrics ? <p className="text-ink-faint">No labels in this run.</p> : (
                      <ul className="mt-1 space-y-0.5">
                        {missed.map((l, j) => <li key={`m${j}`} className="text-rose-600 dark:text-rose-300">missed labeled event {String(l.x0).slice(0, 19)} → {String(l.x1).slice(0, 19)}</li>)}
                        {fa.slice(0, 6).map((f, j) => <li key={`f${j}`} className="text-amber-700 dark:text-amber-300">false alarm {String(f.x0).slice(0, 19)} ({f.length} steps)</li>)}
                        {!missed.length && !fa.length && <li className="text-emerald-600">no failures</li>}
                        {fa.length > 6 && <li className="text-ink-faint">+{fa.length - 6} more false alarms</li>}
                      </ul>)}
                  </div>);
              })}
            </div>
          </>);
      })()}
      {ch && (
        <table className="w-full text-xs">
          <thead><tr className="text-left text-ink-faint"><th className="pb-2">Run</th><th>Changed</th><th>Area</th><th>Threshold</th><th>Regions</th></tr></thead>
          <tbody>{runs.map((r) => { const c = r.result as ChangeResult; return (
            <tr key={r.id} className="border-t border-white/70"><td className="py-1.5 font-mono">{label(r)}</td><td>{pct(c.changed_fraction)}</td>
              <td>{c.changed_area_km2 != null ? `${num(c.changed_area_km2, 3)} km²` : "—"}</td><td>{num(c.threshold, 3)}</td><td>{c.regions.length}</td></tr>); })}</tbody>
        </table>)}
      {!ts && !ch && <p className="text-xs text-ink-soft">Pick runs of the same analysis to compare their metrics.</p>}
    </div>
  );
}

function Experiments({ datasetId, current, onOpen, refreshKey }: { datasetId: string; current: string | null; onOpen: (id: string) => void; refreshKey: unknown }) {
  const runs = useApi<Run[]>("/api/runs?module=workbench&limit=200", [refreshKey]);
  const [sel, setSel] = useState<string[]>([]);
  const mine = (runs.data ?? []).filter((r) => (r.input_ref as { dataset_id?: string }).dataset_id === datasetId);
  if (!mine.length) return null;
  const keys = ["z_threshold", "robust_threshold", "window", "train_fraction", "method", "threshold_mode", "min_region_pixels", "target", "folds", "trees"];
  return (
    <Glass className="p-5">
      <div className="flex items-center justify-between">
        <div className="kicker">Experiment workbench · runs on this dataset</div>
        <button className="text-xs font-semibold text-violet-600" onClick={runs.reload}><RefreshCw className="inline h-3 w-3" /> refresh</button>
      </div>
      <div className="scroll-thin mt-3 overflow-x-auto">
        <table className="w-full min-w-[700px] text-xs">
          <thead><tr className="text-left text-ink-faint"><th className="pb-2" /><th className="pb-2">Run</th><th>Analysis</th><th>Key settings</th><th>Result</th><th>Status</th><th /></tr></thead>
          <tbody>
            {mine.map((r) => {
              const inp = r.input_ref as { analysis: string; params: Record<string, unknown> };
              const m = (r.summary as { metrics?: { robust_f1?: number; zscore_f1?: number; model_tss?: number; baseline_tss?: number } } | null)?.metrics;
              return (
                <tr key={r.id} className={`border-t border-white/70 ${r.id === current ? "bg-white/70" : ""}`}>
                  <td className="pr-2">{r.status === "succeeded" && <input type="checkbox" className="accent-pink-500" checked={sel.includes(r.id)}
                    onChange={(e) => setSel((s) => e.target.checked ? [...s, r.id].slice(-4) : s.filter((x) => x !== r.id))} />}</td>
                  <td className="py-2 font-mono">{r.id.slice(0, 8)}<div className="font-sans text-[10px] text-ink-faint">{timeAgo(r.created_at)}</div></td>
                  <td>{inp.analysis}</td>
                  <td className="text-ink-soft">{keys.filter((k) => k in inp.params && inp.params[k] != null).map((k) => `${k}=${inp.params[k]}`).join(" · ")}</td>
                  <td>{r.summary?.headline ?? r.error}{m && m.robust_f1 != null && <div className="text-[10px] text-ink-faint">F1 robust {pct(m.robust_f1)} · z {pct(m.zscore_f1 ?? 0)}</div>}
                    {m && m.model_tss != null && <div className="text-[10px] text-ink-faint">TSS model {m.model_tss.toFixed(2)} · threshold {m.baseline_tss?.toFixed(2)}</div>}</td>
                  <td><StatusBadge status={r.status} /></td>
                  <td>{r.status === "succeeded" && r.id !== current && <button className="btn-ghost px-2 py-1 text-[11px]" onClick={() => onOpen(r.id)}>Open</button>}</td>
                </tr>);
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-[11px] text-ink-faint">Tick 2-4 runs to compare settings, metrics and failures side by side (same dataset, same chronological split).</p>
      {sel.length >= 2 && <div className="mt-3"><Compare ids={sel} /></div>}
    </Glass>
  );
}

/* ---------------------------------------------------------------- dataset detail */
function DatasetView({ id, runId, setRun, autoRun = false }: { id: string; runId: string | null; setRun: (r: string | null) => void; autoRun?: boolean }) {
  const caps = useCapabilities();
  const ds = useApi<WbDataset>(`/api/workbench/datasets/${id}`, [id]);
  const catalog = useApi<AnalysisSpec[]>("/api/workbench/analyses");
  const [analysisId, setAnalysisId] = useState<string | null>(null);
  const [values, setValues] = useState<Record<string, unknown>>({});
  const [err, setErr] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const { run, error: runErr } = useRun<WorkbenchResult>(runId);
  const compatible = useMemo(() => (catalog.data ?? []).filter((a) => ds.data?.compatible?.includes(a.id)), [catalog.data, ds.data]);
  const analysis = compatible.find((a) => a.id === analysisId);

  useEffect(() => {
    if (!ds.data || !compatible.length) return;
    const fromRun = run?.input_ref as { analysis?: string; params?: Record<string, unknown> } | undefined;
    const a = compatible.find((x) => x.id === (fromRun?.analysis ?? analysisId)) ?? compatible[0];
    setAnalysisId(a.id);
    setValues(fromRun?.params && fromRun.analysis === a.id ? fromRun.params : initialValues(a, ds.data));
  }, [ds.data, compatible.length, run?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  // right after an upload: start the recommended analysis so results and insights appear without extra clicks
  const autoDone = useRef(false);
  useEffect(() => {
    if (!caps?.capabilities.write || !autoRun || autoDone.current || runId || !ds.data || !analysis) return;
    const rec = ds.data.profile?.recommended;
    if (!rec || analysis.id !== rec) return;
    if (analysis.requires_model === "mars" && !ds.data.profile?.mars_model) return;
    autoDone.current = true;
    createRun("workbench", { dataset_id: ds.data.id, analysis: analysis.id, params: values })
      .then((r) => setRun(r.id)).catch((e) => setErr((e as Error).message));
  }, [autoRun, runId, ds.data, analysis, values, caps?.capabilities.write]); // eslint-disable-line react-hooks/exhaustive-deps

  if (ds.error) return <ErrorNotice message={ds.error.message} />;
  if (!ds.data) return <div className="flex items-center gap-2 text-sm text-ink-soft"><Spinner /> Loading dataset…</div>;
  const d = ds.data;
  const K = KIND[d.kind];

  async function start() {
    if (!analysis) return;
    setSubmitting(true);
    setErr(null);
    try {
      const r = await createRun("workbench", { dataset_id: d.id, analysis: analysis.id, params: values });
      setRun(r.id);
    } catch (e) { setErr((e as Error).message); } finally { setSubmitting(false); }
  }
  const busy = !!run && (run.status === "queued" || run.status === "running");

  return (
    <div className="space-y-4">
      <Glass className="p-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <span className={`chip ${K.cls}`}><K.icon className="h-3 w-3" /> {K.label}</span>
            <h2 className="mt-2 text-xl font-bold">{d.name}</h2>
            <p className="text-xs text-ink-soft">{d.summary} · added {timeAgo(d.created_at)} · content hash <span className="font-mono">{d.sha256}</span></p>
          </div>
        </div>
        <div className="mt-4">{d.samples ? <SamplesPreview ds={d} /> : d.tables ? <TablePreview ds={d} /> : d.images ? <ImagePreview ds={d} /> : null}</div>
      </Glass>

      <Glass className="space-y-4 p-5">
        {d.profile && <ProfileCard p={d.profile} analysisTitle={compatible.find((a) => a.id === d.profile!.recommended)?.title} />}
        <div className="label">Choose a compatible analysis</div>
        <div className="grid gap-2 md:grid-cols-3">
          {compatible.map((a) => (
            <button key={a.id} onClick={() => { setAnalysisId(a.id); setValues(initialValues(a, d)); }}
              className={`rounded-2xl border p-3 text-left transition ${a.id === analysisId ? "border-pink-300 bg-white shadow-md" : "border-white/70 bg-white/40 hover:bg-white/70"}`}>
              <div className="flex items-center gap-1.5 text-sm font-bold">{a.title}
                {a.id === d.profile?.recommended && <span className="chip bg-pink-100 px-1.5 text-[9px] text-pink-700">recommended</span>}</div>
              <div className="mt-1 text-[11px] leading-snug text-ink-soft">{a.description}</div>
              <div className="mt-2 text-[10px] text-ink-faint">Evaluation: {a.evaluation}</div>
            </button>))}
        </div>
        {analysis && (
          <>
            <ParamForm analysis={analysis} dataset={d} values={values} onChange={setValues} />
            <div className="flex flex-wrap items-center gap-3">
              <WriteButton className="btn-primary" onClick={start} disabled={submitting || busy}>{submitting ? <Spinner /> : <Play className="h-4 w-4" />} Run analysis</WriteButton>
              <button className="btn-ghost" onClick={() => setValues(initialValues(analysis, d))}>Reset settings</button>
              {analysis.requires_model === "mars" && !d.profile?.mars_model && <span className="text-xs text-amber-700">Needs the Mars terrain model - train it on Data &amp; Training (quick run: minutes).</span>}
              {analysis.requires_model === "mars (optional)" && <span className="text-xs text-ink-soft">{d.profile?.mars_model ? "Terrain model installed - soil/bedrock/sand/rock will be added." : "Works now; terrain classes are added once the Mars model is installed."}</span>}
            </div>
          </>
        )}
        {err && <ErrorNotice message={err} />}
      </Glass>

      {runErr && <ErrorNotice message={runErr} />}
      {busy && <RunProgress status={run!.status} label={run!.input_label} />}
      {run?.status === "failed" && <ErrorNotice title="Analysis failed" message={run.error ?? "Unknown error"} />}
      {run?.status === "succeeded" && run.result && <ResultView run={run} />}
      <Experiments datasetId={d.id} current={runId} onOpen={setRun} refreshKey={run?.status} />
    </div>
  );
}

/* ---------------------------------------------------------------- page */
export default function Workbench() {
  const [params, setParams] = useSearchParams();
  const datasetId = params.get("dataset");
  const runId = params.get("run");
  const list = useApi<WbDatasetListItem[]>("/api/workbench/datasets");
  const [kind, setKind] = useState<"auto" | "timeseries" | "images" | "lightcurve" | "samples">("auto");
  const [msg, setMsg] = useState<string | null>(null);
  const runOnly = useRun<WorkbenchResult>(!datasetId ? runId : null);

  // opening /workbench?run=<id> (from history) -> jump to its dataset
  useEffect(() => {
    const ds = (runOnly.run?.input_ref as { dataset_id?: string } | undefined)?.dataset_id;
    if (!datasetId && ds && runId) setParams({ dataset: ds, run: runId }, { replace: true });
  }, [runOnly.run, datasetId, runId, setParams]);

  const open = (item: WbDatasetListItem) => {
    if (item.kind === "lightcurve") return;
    setParams({ dataset: item.id });
  };

  return (
    <Page title="Data Workbench"
      subtitle="Bring your own observations: upload a table, a FITS light curve, images or a whole .zip. Preview and map the data, run a compatible analysis, review the flags and export a reproducible report."
      icon={<IconBadge gradient="from-pink-500 to-amber-400"><FlaskConical className="h-6 w-6" /></IconBadge>}>
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[330px_minmax(0,1fr)]">
        <div className="space-y-4">
          <Glass className="space-y-3 p-4">
            <Segmented value={kind} onChange={setKind} options={[
              { value: "auto", label: "Auto" }, { value: "timeseries", label: "Series" }, { value: "samples", label: "Windows" }, { value: "images", label: "Images" }, { value: "lightcurve", label: "Light curve" }]} />
            <DropZone url="/api/workbench/datasets" fields={{ kind }}
              hint={<>CSV · TSV · TXT · XLSX · JSON / JSON Lines · Parquet · NPY · FITS · PNG · JPEG · TIFF/GeoTIFF · .zip/.tar.gz · ≤ 1 GB</>}
              onError={setMsg}
              onDone={(r) => {
                const res = r as { id: string; kind: string };
                setMsg(null);
                list.reload();
                if (res.kind !== "lightcurve") setParams({ dataset: res.id, auto: "1" });   // run the recommended analysis
              }} />
            {msg && <ErrorNotice title="Upload problem" message={msg} />}
          </Glass>
          <Glass className="p-3">
            <div className="label px-1 pb-2">Your datasets</div>
            {list.loading && <div className="flex items-center gap-2 p-2 text-sm text-ink-soft"><Spinner /> Loading…</div>}
            {list.data?.length === 0 && <p className="p-2 text-sm text-ink-soft">Nothing uploaded yet.</p>}
            <div className="scroll-thin max-h-[560px] space-y-1 overflow-y-auto pr-1">
              {list.data?.map((d) => {
                const K = KIND[d.kind];
                const inner = (
                  <>
                    <div className={`grid h-9 w-9 shrink-0 place-items-center rounded-xl ${K.cls}`}><K.icon className="h-4 w-4" /></div>
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-sm font-semibold">{d.name}</div>
                      <div className="truncate text-[11px] text-ink-soft">{d.summary} · {timeAgo(d.created_at)}</div>
                    </div>
                  </>);
                return (
                  <div key={d.id} className={`group flex items-center gap-2 rounded-2xl px-2 py-2 transition ${d.id === datasetId ? "bg-white shadow" : "hover:bg-white/60"}`}>
                    {d.kind === "lightcurve"
                      ? <Link to="/exoplanet?source=upload" className="flex min-w-0 flex-1 items-center gap-2">{inner}<ArrowRight className="h-3.5 w-3.5 text-exo-500" /></Link>
                      : <button onClick={() => open(d)} className="flex min-w-0 flex-1 items-center gap-2 text-left">{inner}</button>}
                    <WriteButton title="Delete dataset" className="rounded-lg p-1 text-ink-faint opacity-0 transition hover:text-rose-600 group-hover:opacity-100"
                      onClick={async () => {
                        await api(d.kind === "lightcurve" ? `/api/exoplanet/datasets/${d.id}` : `/api/workbench/datasets/${d.id}`, { method: "DELETE" });
                        if (d.id === datasetId) setParams({});
                        list.reload();
                      }}><Trash2 className="h-3.5 w-3.5" /></WriteButton>
                  </div>);
              })}
            </div>
          </Glass>
        </div>
        <div className="min-w-0">
          {datasetId ? <DatasetView key={datasetId} id={datasetId} runId={runId} autoRun={params.get("auto") === "1"} setRun={(r) => setParams(r ? { dataset: datasetId, run: r } : { dataset: datasetId })} />
            : (
              <div className="grid gap-4 md:grid-cols-2">
                {[
                  { icon: FileSpreadsheet, t: "Telemetry & sensor tables", d: "Any CSV/XLSX/Parquet with a time or step column and numeric channels: spacecraft housekeeping, rover weather (MEDA), OMNIWeb solar-wind & geomagnetic indices. Add a 0/1 column or intervals to get precision/recall.", cls: "from-cyan-400 to-sky-500" },
                  { icon: ImageIcon, t: "Before / after imagery", d: "Two aligned images or GeoTIFFs of the same place (Landsat, Sentinel-2, NASA HLS, HiRISE, Worldview snapshots). Swipe between dates, map change and water extent, get the area in km².", cls: "from-orange-400 to-rose-500" },
                  { icon: SunMedium, t: "Labeled windows (e.g. solar flares)", d: "JSON / JSON Lines records with a label and a set of parameter series - like the SWAN-SF flare benchmark (IEEE BigData 2020). Class balance, which parameters separate the classes, and a forecast model scored with TSS / HSS against a single-threshold baseline.", cls: "from-amber-400 to-orange-500" },
                  { icon: Orbit, t: "Light curves", d: "Kepler / K2 / TESS FITS or tables with time + flux. Opens in the Exoplanet Finder: period search, candidate score, vetting checks and a 3D view of the system.", cls: "from-violet-500 to-fuchsia-500" },
                ].map((c) => (
                  <Glass key={c.t} hover className="p-5">
                    <IconBadge gradient={c.cls}><c.icon className="h-6 w-6" /></IconBadge>
                    <h3 className="mt-3 font-bold">{c.t}</h3>
                    <p className="mt-1 text-xs leading-relaxed text-ink-soft">{c.d}</p>
                  </Glass>))}
                <div className="md:col-span-2">
                  <Notice tone="info" title="Every run is reproducible">
                    The report records the dataset content hash, every setting, the model version and your review verdicts. Re-run the same settings later and compare results in the experiment table.
                  </Notice>
                </div>
              </div>)}
        </div>
      </div>
    </Page>
  );
}
