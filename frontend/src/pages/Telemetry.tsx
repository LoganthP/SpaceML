import { Activity, CheckCircle2, CircleDashed, Play, Search, Satellite } from "lucide-react";
import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import type { GlobeMarker, GlobeOrbit } from "../components/ContextGlobe";
import { EvidencePanel } from "../components/Evidence";
import { ReviewToggle, useReviews } from "../components/RunTools";
import { useSearchParams } from "react-router-dom";
import Plot from "../components/LazyPlot";

const ContextGlobe = lazy(() => import("../components/ContextGlobe"));
const HelioView = lazy(() => import("../components/HelioView"));
import { ErrorNotice, Glass, IconBadge, Kpi, Notice, Page, RunProgress, Segmented, Spinner } from "../components/ui";
import { createRun, useApi, useRun } from "../lib/api";
import { num } from "../lib/format";
import type { Channel, TelemetryResult } from "../lib/types";
import { useOnJobFinished } from "../components/Jobs";
import { ModuleStatus } from "../components/Catalog";

type Mission = "SMAP" | "MSL";

function ChannelList({ channels, selected, onSelect }: { channels: Channel[]; selected: string | null; onSelect: (c: string) => void }) {
  return (
    <div className="scroll-thin max-h-[560px] space-y-1.5 overflow-y-auto pr-1">
      {channels.map((c) => {
        const active = c.chan_id === selected;
        return (
          <button key={c.chan_id} onClick={() => onSelect(c.chan_id)}
            className={`flex w-full items-center gap-3 rounded-2xl border px-3 py-2.5 text-left transition ${active ? "border-tele-400/60 bg-white shadow-md" : "border-transparent hover:bg-white/60"}`}>
            <div className={`grid h-9 w-9 place-items-center rounded-xl text-xs font-bold ${active ? "bg-gradient-to-br from-cyan-400 to-sky-500 text-white" : "bg-white/80 text-tele-700"}`}>
              {c.chan_id.split("-")[0]}
            </div>
            <div className="min-w-0 flex-1">
              <div className="text-sm font-semibold">{c.chan_id}</div>
              <div className="truncate text-[11px] text-ink-soft">
                {c.anomaly_sequences.length} labeled · {[...new Set(c.anomaly_classes)].join(", ")}
              </div>
            </div>
            {c.has_model ? <CheckCircle2 className="h-4 w-4 text-emerald-500" aria-label="model trained" />
              : <CircleDashed className="h-4 w-4 text-ink-faint" aria-label="no model" />}
          </button>
        );
      })}
    </div>
  );
}

function Timeline({ r, showBaseline }: { r: TelemetryResult; showBaseline: boolean }) {
  const x = useMemo(() => r.actual.map((_, i) => i), [r]);
  const TOP: [number, number] = [0.36, 1];
  const shapes: Partial<Plotly.Shape>[] = [];
  r.labeled.forEach((l) => shapes.push({ type: "rect", xref: "x", yref: "paper", x0: l.start, x1: l.end, y0: 0, y1: 1,
    fillcolor: "rgba(16,185,129,0.14)", line: { width: 0 }, layer: "below" }));
  r.flagged.forEach((f) => {
    shapes.push({ type: "rect", xref: "x", yref: "paper", x0: f.start, x1: f.end + 1, y0: 0.965, y1: 1, fillcolor: "rgba(225,29,72,0.85)", line: { width: 0 } });
    shapes.push({ type: "rect", xref: "x", yref: "paper", x0: f.start, x1: f.end + 1, y0: 0, y1: 0.28, fillcolor: "rgba(225,29,72,0.12)", line: { width: 0 }, layer: "below" });
  });
  if (showBaseline) r.baseline.flagged.forEach((f) => shapes.push({ type: "rect", xref: "x", yref: "paper", x0: f.start, x1: f.end + 1,
    y0: 0.925, y1: 0.955, fillcolor: "rgba(245,158,11,0.85)", line: { width: 0 } }));
  shapes.push({ type: "line", xref: "paper", yref: "y2", x0: 0, x1: 1, y0: r.threshold, y1: r.threshold, line: { color: "#e11d48", width: 1.5, dash: "dash" } });
  shapes.push({ type: "line", xref: "x", yref: "paper", x0: r.window, x1: r.window, y0: 0, y1: 1, line: { color: "rgba(100,116,139,0.5)", width: 1, dash: "dot" } });

  const data: Plotly.Data[] = [
    { x, y: r.actual, type: "scattergl", mode: "lines", name: "Actual telemetry", line: { color: "#334155", width: 1.2 } },
    { x, y: r.predicted, type: "scattergl", mode: "lines", name: "LSTM prediction", line: { color: "#06b6d4", width: 1.4 } },
    { x, y: r.smoothed_error, type: "scattergl", mode: "lines", name: "Smoothed error", line: { color: "#8b5cf6", width: 1.3 }, yaxis: "y2" },
    { x: [null], y: [null], type: "scatter", mode: "markers", name: "Expert-labeled anomaly", marker: { color: "rgba(16,185,129,0.5)", symbol: "square", size: 12 } },
    { x: [null], y: [null], type: "scatter", mode: "markers", name: "Flagged for review", marker: { color: "#e11d48", symbol: "square", size: 12 } },
    ...(showBaseline ? [{ x: [null], y: [null], type: "scatter", mode: "markers", name: "z-score baseline", marker: { color: "#f59e0b", symbol: "square", size: 12 } } as Plotly.Data] : []),
  ];
  if (showBaseline) {
    data.push({ x: [0, r.n_steps], y: [r.baseline.upper, r.baseline.upper], type: "scatter", mode: "lines", name: "±3σ (train)", line: { color: "#f59e0b", width: 1, dash: "dot" }, hoverinfo: "skip" });
    data.push({ x: [0, r.n_steps], y: [r.baseline.lower, r.baseline.lower], type: "scatter", mode: "lines", showlegend: false, line: { color: "#f59e0b", width: 1, dash: "dot" }, hoverinfo: "skip" });
  }
  return (
    <Plot height={520} data={data} layout={{
      shapes, hovermode: "x unified",
      margin: { l: 56, r: 18, t: 40, b: 44 },
      xaxis: { title: { text: "Test time step" }, rangeslider: { visible: false }, anchor: "y2" },
      yaxis: { domain: TOP, title: { text: "Scaled value" } },
      yaxis2: { domain: [0, 0.28], title: { text: "Error" }, gridcolor: "rgba(15,23,42,0.06)", zeroline: false, rangemode: "tozero" },
    } as Partial<Plotly.Layout>} />
  );
}

const SMAP_ORBIT: GlobeOrbit[] = [{ label: "SMAP", inclinationDeg: 98.1, altitudeKm: 685, periodMin: 98.5, color: "#06b6d4" }];
const GALE: GlobeMarker[] = [
  { lat: -4.59, lon: 137.44, label: "Curiosity · Gale crater (MSL telemetry)", color: "#06b6d4", primary: true },
  { lat: 18.44, lon: 77.45, label: "Perseverance · Jezero", color: "#f97316" },
];
const JPL: GlobeMarker[] = [{ lat: 34.2, lon: -118.17, label: "JPL mission operations, Pasadena", color: "#7c3aed", primary: true }];

function MissionContext({ spacecraft }: { spacecraft: string }) {
  return spacecraft === "MSL" ? (
    <Suspense fallback={<div className="h-[340px]" />}>
      <div className="grid gap-4 lg:grid-cols-2">
        <ContextGlobe body="mars" markers={GALE} height={340} caption="Curiosity has driven in Gale crater since 2012; its housekeeping telemetry is downlinked via Mars orbiters." />
        <HelioView show={{ mars: true }} height={340} caption="Earth-Mars geometry today: commands and telemetry cross this gap with a one-way light time of several minutes." />
      </div>
    </Suspense>
  ) : (
    <Suspense fallback={<div className="h-[340px]" />}>
      <ContextGlobe body="earth" markers={JPL} orbits={SMAP_ORBIT} height={340}
        caption="SMAP (Soil Moisture Active Passive) flies a sun-synchronous polar orbit at 685 km, inclination 98.1°, one orbit every ~98.5 min." />
    </Suspense>
  );
}

function FlagReview({ runId, res }: { runId: string; res: TelemetryResult }) {
  const { reviews, set } = useReviews(runId);
  return (
    <Glass className="p-5">
      <div className="flex items-center justify-between">
        <div className="label">Intervals flagged for review</div>
        <span className="text-[10px] text-ink-faint">verdicts go to the Review Studio</span>
      </div>
      {res.flagged.length === 0 ? <p className="mt-3 text-sm text-ink-soft">Nothing exceeded the threshold.</p> : (
        <table className="mt-3 w-full text-sm">
          <thead><tr className="text-left text-xs text-ink-faint"><th className="pb-2">Interval</th><th>Peak error</th><th>Matches a label</th><th>Verdict</th></tr></thead>
          <tbody>
            {res.flagged.map((f, i) => {
              const hit = res.labeled.some((l) => f.start <= l.end && l.start <= f.end);
              const item = `flag:${f.start}-${f.end}`;
              return (
                <tr key={i} className="border-t border-white/70">
                  <td className="py-2 font-mono text-xs">{f.start}–{f.end}</td><td>{num(f.peak_error, 4)}</td>
                  <td>{hit ? <span className="chip bg-emerald-100 text-emerald-700">yes</span> : <span className="chip bg-amber-100 text-amber-800">false alarm</span>}</td>
                  <td><ReviewToggle item={item} review={reviews[item]} onSet={set} /></td>
                </tr>);
            })}
          </tbody>
        </table>)}
    </Glass>
  );
}

export default function Telemetry() {
  const [params, setParams] = useSearchParams();
  const runId = params.get("run");
  const [mission, setMission] = useState<Mission>("SMAP");
  const [query, setQuery] = useState("");
  const [onlyTrained, setOnlyTrained] = useState(true);
  const [selected, setSelected] = useState<string | null>(params.get("chan"));
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [showBaseline, setShowBaseline] = useState(true);
  const channels = useApi<Channel[]>("/api/telemetry/channels");
  useOnJobFinished(channels.reload);
  const { run, error: runError } = useRun<TelemetryResult>(runId);

  useEffect(() => {
    if (run?.input_ref?.chan_id) {
      const c = String(run.input_ref.chan_id);
      setSelected(c);
      const ch = channels.data?.find((x) => x.chan_id === c);
      if (ch) setMission(ch.spacecraft);
    }
  }, [run?.id, channels.data]); // eslint-disable-line react-hooks/exhaustive-deps

  const anyTrained = channels.data?.some((c) => c.has_model);
  const list = (channels.data ?? []).filter((c) => c.spacecraft === mission
    && (!onlyTrained || !anyTrained || c.has_model) && c.chan_id.toLowerCase().includes(query.toLowerCase()));
  const sel = channels.data?.find((c) => c.chan_id === selected);
  // never leave the analysis panel empty: preselect the first channel with a model (the user can change it)
  useEffect(() => {
    if (!selected && !params.get("run") && list.length) setSelected((list.find((c) => c.has_model) ?? list[0]).chan_id);
  }, [list.length, selected]); // eslint-disable-line react-hooks/exhaustive-deps

  async function start() {
    if (!selected) return;
    setSubmitting(true);
    setSubmitError(null);
    try {
      const r = await createRun("telemetry", { chan_id: selected });
      setParams({ run: r.id });
    } catch (e) {
      setSubmitError((e as Error).message);
    } finally {
      setSubmitting(false);
    }
  }

  const res = run?.status === "succeeded" ? run.result : undefined;
  const showingRunForSelected = run && String(run.input_ref.chan_id) === selected;

  return (
    <Page title="Spacecraft Anomaly Detection"
      subtitle="One LSTM per channel predicts the next telemetry reading from the last 250 steps of telemetry and commands. Smoothed prediction errors above an unsupervised threshold are flagged for review."
      icon={<IconBadge gradient="from-cyan-400 to-sky-500"><Activity className="h-6 w-6" /></IconBadge>}>
      <ModuleStatus ids={["telemetry"]} />
      {channels.error && <ErrorNotice message={channels.error.message} title="Telemetry data unavailable" />}
      <div className="grid gap-4 xl:grid-cols-[320px_1fr]">
        <Glass className="space-y-3 p-4">
          <Segmented value={mission} onChange={setMission} options={[{ value: "SMAP", label: "SMAP satellite" }, { value: "MSL", label: "Curiosity (MSL)" }]} />
          <div className="relative">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-ink-faint" />
            <input className="input pl-9" placeholder="Search channel (e.g. P-1)" value={query} onChange={(e) => setQuery(e.target.value)} />
          </div>
          {anyTrained && (
            <label className="flex cursor-pointer items-center gap-2 px-1 text-xs text-ink-soft">
              <input type="checkbox" checked={onlyTrained} onChange={(e) => setOnlyTrained(e.target.checked)} className="accent-cyan-600" />
              Only channels with a trained model
            </label>
          )}
          {channels.loading ? <div className="flex items-center gap-2 p-3 text-sm text-ink-soft"><Spinner /> Loading channels…</div>
            : <ChannelList channels={list} selected={selected} onSelect={(c) => { setSelected(c); setSubmitError(null); }} />}
          {!channels.loading && list.length === 0 && <p className="p-2 text-sm text-ink-soft">No channels match.</p>}
        </Glass>

        <div className="space-y-4">
          <Glass className="p-5">
            {!sel && channels.loading ? (
              <div className="flex flex-col items-center gap-3 py-10 text-center" role="status">
                <Spinner className="h-6 w-6" /><p className="font-semibold">Loading the 81 SMAP / MSL channels…</p>
              </div>
            ) : !sel ? (
              <div className="flex flex-col items-center gap-3 py-10 text-center">
                <Satellite className="h-10 w-10 text-tele-500" />
                <p className="font-semibold">Select a sensor channel</p>
                <p className="max-w-md text-sm text-ink-soft">Channels come from NASA's SMAP satellite and the Curiosity rover. Each has expert-labeled anomaly intervals in its test file.</p>
              </div>
            ) : (
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <div className="label">{sel.spacecraft} channel</div>
                  <div className="font-display text-2xl font-bold">{sel.chan_id}</div>
                  <div className="text-xs text-ink-soft">{sel.num_values.toLocaleString()} test steps · {sel.anomaly_sequences.length} labeled anomal{sel.anomaly_sequences.length === 1 ? "y" : "ies"} ({sel.anomaly_classes.join(", ")})</div>
                </div>
                <button className="btn-primary" onClick={start} disabled={submitting || !sel.has_model || run?.status === "running" || run?.status === "queued"}>
                  {submitting ? <Spinner /> : <Play className="h-4 w-4" />} Run anomaly detection
                </button>
              </div>
            )}
            {sel && !sel.has_model && (
              <div className="mt-4"><ErrorNotice title="No trained model for this channel"
                message={`This channel has no model yet. Run: task:telemetry.train?channels=${sel.chan_id}`} /></div>
            )}
            {submitError && <div className="mt-4"><ErrorNotice message={submitError} /></div>}
            {runError && <div className="mt-4"><ErrorNotice message={runError} /></div>}
            {run && (run.status === "queued" || run.status === "running") && <div className="mt-4"><RunProgress status={run.status} label={run.input_label} /></div>}
            {run?.status === "failed" && <div className="mt-4"><ErrorNotice title="Run failed" message={run.error ?? "Unknown error"} /></div>}
          </Glass>

          {res && showingRunForSelected && (
            <>
              <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
                <Kpi label="Flagged for review" value={res.flagged.length} accent="#e11d48" sub={`${res.pruned_sequences} weaker sequences pruned`} />
                <Kpi label="Labeled anomalies found" value={`${res.score.tp}/${res.labeled.length}`} accent="#0891b2" sub={`z-score baseline: ${res.baseline_score.tp}/${res.labeled.length}`} />
                <Kpi label="False alarms" value={res.score.fp} sub={`z-score baseline: ${res.baseline_score.fp}`} />
                <Kpi label="Threshold" value={num(res.threshold, 4)} sub={`mean + ${res.threshold_z}σ of errors (no labels used)`} />
              </div>
              <Glass className="p-4">
                <div className="mb-1 flex flex-wrap items-center justify-between gap-2 px-1">
                  <div className="label">Test telemetry · {res.chan_id} · model {res.model_version}</div>
                  <label className="flex cursor-pointer items-center gap-2 text-xs text-ink-soft">
                    <input type="checkbox" checked={showBaseline} onChange={(e) => setShowBaseline(e.target.checked)} className="accent-amber-500" />
                    Show z-score baseline
                  </label>
                </div>
                {res.quiet_channel && <Notice tone="info" title="Prediction errors stay tiny on this channel, so nothing is flagged (paper's minimum-error rule)." />}
                <Timeline r={res} showBaseline={showBaseline} />
                <p className="px-1 text-[11px] text-ink-faint">The first {res.window} steps have no prediction (the model needs a full history window). Drag to zoom, double-click to reset.</p>
              </Glass>
              <div className="grid gap-4 lg:grid-cols-2">
                <Glass className="p-5">
                  <div className="label">Expert-labeled anomalies</div>
                  <table className="mt-3 w-full text-sm">
                    <thead><tr className="text-left text-xs text-ink-faint"><th className="pb-2">Interval</th><th>Class</th><th>LSTM</th><th>z-score</th></tr></thead>
                    <tbody>
                      {res.labeled.map((l, i) => (
                        <tr key={i} className="border-t border-white/70">
                          <td className="py-2 font-mono text-xs">{l.start}–{l.end}</td><td className="capitalize">{l.class}</td>
                          <td>{l.found_by_model ? <span className="chip bg-emerald-100 text-emerald-700">found</span> : <span className="chip bg-rose-100 text-rose-700">missed</span>}</td>
                          <td>{l.found_by_baseline ? <span className="chip bg-emerald-100 text-emerald-700">found</span> : <span className="chip bg-slate-100 text-slate-600">missed</span>}</td>
                        </tr>))}
                    </tbody>
                  </table>
                </Glass>
                <FlagReview runId={run!.id} res={res} />
              </div>
              <Glass className="hud p-4">
                <div className="kicker mb-2">Mission context · where this telemetry comes from</div>
                <MissionContext spacecraft={res.spacecraft} />
              </Glass>
              <EvidencePanel runId={run!.id} compact />
            </>
          )}
        </div>
      </div>
    </Page>
  );
}
