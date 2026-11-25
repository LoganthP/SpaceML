import { WriteButton } from "../lib/capabilities";
import { CalendarRange, CloudLightning, Pause, Play, Plus, SkipBack, Trash2, Wind, Zap } from "lucide-react";
import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { JobButton, activeJob, useJobs, useOnJobFinished, JobCard } from "../components/Jobs";
import Plot from "../components/LazyPlot";
import { ErrorNotice, Glass, IconBadge, Kpi, Notice, Page, Segmented, Spinner } from "../components/ui";
import { api, useApi, usePoll } from "../lib/api";
import { num } from "../lib/format";
import type { ModuleEvaluation } from "../lib/types";
import { ModuleStatus } from "../components/Catalog";

const HelioView = lazy(() => import("../components/HelioView"));

interface Status { cached_years: number[]; default_years: number[]; variables: Record<string, { unit: string; label: string }>; model: null | { version: string; horizons: number[]; train_years: number[]; test_years: number[] } }
interface Ev { id: string; name: string; start: string; end: string; cached: boolean; min_dst?: number; max_kp?: number; max_v?: number }
interface Ann { id: string; start: string; end: string; label: string; note: string }
interface Series { time: string[]; step_hours: number; series: Record<string, (number | null)[]>; meta: Record<string, { unit: string; label: string }>; summary: { min_dst: number | null; max_kp: number | null; max_v: number | null; min_bz: number | null; coverage: number } }
interface Fc { horizon: number; issued: string[]; valid: string[]; forecast: (number | null)[]; persistence: (number | null)[]; truth: (number | null)[]; in_sample_years: number[]; version: string }

const PANELS = [
  { v: "V", color: "#f59e0b", title: "Speed" },
  { v: "Bz", color: "#06b6d4", title: "IMF Bz" },
  { v: "N", color: "#a855f7", title: "Density" },
  { v: "Pdyn", color: "#ec4899", title: "Pressure" },
  { v: "Dst", color: "#ef4444", title: "Dst" },
  { v: "Kp", color: "#22c55e", title: "Kp" },
];

const stormLevel = (dst: number | null) => dst == null ? { t: "—", c: "text-ink-faint" } : dst <= -250 ? { t: "extreme", c: "text-rose-600" }
  : dst <= -100 ? { t: "intense", c: "text-rose-500" } : dst <= -50 ? { t: "moderate", c: "text-orange-500" } : dst <= -30 ? { t: "weak", c: "text-amber-500" } : { t: "quiet", c: "text-emerald-500" };

function Readout({ label, value, unit, tone }: { label: string; value: string; unit: string; tone?: string }) {
  return (
    <div className="glass-inset px-3 py-2">
      <div className="font-mono text-[9px] uppercase tracking-[0.2em] text-ink-faint">{label}</div>
      <div className={`font-mono text-xl font-bold tabular-nums ${tone ?? ""}`}>{value}<span className="ml-1 text-[10px] font-semibold text-ink-faint">{unit}</span></div>
    </div>
  );
}

function LiveSolarWind() {
  const { data, error, loading, reload } = usePoll<{
    status: "live" | "stale" | "unavailable";
    observed_at?: string; speed?: number; density?: number; temperature?: number;
  }>("/api/live/solar-wind", 60_000);
  return (
    <Glass className="space-y-2 p-4" aria-label="NOAA solar wind">
      <div className="flex flex-wrap items-center gap-2">
        <Wind className="h-4 w-4 text-amber-600" />
        <h2 className="text-sm font-semibold">Latest solar wind</h2>
        <a href="https://www.swpc.noaa.gov/products/real-time-solar-wind" target="_blank" rel="noreferrer" className="text-xs underline">NOAA SWPC</a>
        <button className="btn-ghost ml-auto px-2 py-1 text-xs" onClick={reload} disabled={loading}>Refresh solar wind</button>
      </div>
      {data?.observed_at ? <>
        <p className="text-xs text-ink-soft">{data.status === "live" && !error ? "Latest measurement" : "Last available measurement"} · {new Date(data.observed_at).toLocaleString()}</p>
        <div className="grid grid-cols-3 gap-2">
          <Readout label="Speed" value={num(data.speed, 0)} unit="km/s" />
          <Readout label="Density" value={num(data.density, 1)} unit="cm⁻³" />
          <Readout label="Temperature" value={num(data.temperature, 0)} unit="K" />
        </div>
      </> : <p className="text-xs text-ink-soft">{loading ? "Checking NOAA…" : "NOAA solar-wind measurements are temporarily unavailable."}</p>}
    </Glass>
  );
}

function Replay({ s, events, notes }: { s: Series; events: Ev[]; notes: Ann[] }) {
  const n = s.time.length;
  const [i, setI] = useState(0);
  const [playing, setPlaying] = useState(false);
  useEffect(() => { setI(0); setPlaying(false); }, [s]);
  useEffect(() => {
    if (!playing) return;
    const t = setInterval(() => setI((x) => { if (x >= n - 1) { setPlaying(false); return x; } return Math.min(n - 1, x + Math.max(1, Math.round(n / 240))); }), 120);
    return () => clearInterval(t);
  }, [playing, n]);
  const at = (v: string) => s.series[v]?.[i] ?? null;
  const t0 = s.time[0], t1 = s.time[n - 1];
  const shapes: Partial<Plotly.Shape>[] = [
    ...events.filter((e) => e.end >= t0 && e.start <= t1).map((e) => ({ type: "rect", xref: "x", yref: "paper", x0: e.start, x1: e.end, y0: 0, y1: 1, fillcolor: "rgba(239,68,68,0.07)", line: { width: 0 }, layer: "below" }) as Partial<Plotly.Shape>),
    ...notes.filter((e) => e.end >= t0 && e.start <= t1).map((e) => ({ type: "rect", xref: "x", yref: "paper", x0: e.start, x1: e.end, y0: 0, y1: 1, fillcolor: "rgba(168,85,247,0.12)", line: { color: "rgba(168,85,247,0.5)", width: 1, dash: "dot" }, layer: "below" }) as Partial<Plotly.Shape>),
    { type: "line", xref: "x", yref: "paper", x0: s.time[i], x1: s.time[i], y0: 0, y1: 1, line: { color: "#7c3aed", width: 2 } },
  ];
  const k = PANELS.length, gap = 0.02, hgt = (1 - gap * (k - 1)) / k;
  const layout: Record<string, unknown> = { shapes, hovermode: "x unified", showlegend: false, margin: { l: 64, r: 16, t: 10, b: 36 },
    annotations: [...events, ...notes.map((a) => ({ ...a, name: a.label }))].filter((e) => e.end >= t0 && e.start <= t1).map((e) => ({
      x: e.start, y: 1, xref: "x", yref: "paper", text: (e as { name: string }).name, showarrow: false, xanchor: "left", yanchor: "bottom", font: { size: 10, color: "#a855f7" } })) };
  const data: Plotly.Data[] = PANELS.map((p, j) => {
    const ax = j === 0 ? "" : String(j + 1);
    layout[`yaxis${ax}`] = { domain: [1 - (j + 1) * hgt - j * gap, 1 - j * hgt - j * gap], title: { text: `${p.title}<br><sub>${s.meta[p.v]?.unit ?? ""}</sub>`, font: { size: 10 } }, zeroline: p.v === "Bz" || p.v === "Dst" };
    return { x: s.time, y: s.series[p.v], xaxis: "x", yaxis: `y${ax}`, type: p.v === "Kp" ? "bar" : "scattergl", mode: "lines", name: p.v,
      line: { color: p.color, width: 1.4 }, marker: { color: p.color } } as Plotly.Data;
  });
  layout.xaxis = { type: "date", anchor: `y${k}` };
  const dst = at("Dst"), lvl = stormLevel(dst), bz = at("Bz");
  return (
    <Glass className="hud space-y-3 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <button className="btn-primary px-3 py-1.5 text-xs" onClick={() => { if (i >= n - 1) setI(0); setPlaying((p) => !p); }}>{playing ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}{playing ? "Pause" : "Replay storm"}</button>
        <button className="btn-ghost px-3 py-1.5 text-xs" onClick={() => { setI(0); setPlaying(false); }}><SkipBack className="h-3.5 w-3.5" /></button>
        <input type="range" min={0} max={n - 1} value={i} onChange={(e) => { setPlaying(false); setI(+e.target.value); }} className="slider min-w-[200px] flex-1" />
        <span className="font-mono text-xs tabular-nums text-ink-soft">{s.time[i]?.replace("T", " ").slice(0, 16)} UTC</span>
      </div>
      <div className="grid grid-cols-3 gap-2 md:grid-cols-6">
        <Readout label="Wind speed" value={num(at("V"), 0)} unit="km/s" tone={(at("V") ?? 0) > 600 ? "text-orange-500" : ""} />
        <Readout label="IMF Bz" value={num(bz, 1)} unit="nT" tone={(bz ?? 0) < -10 ? "text-rose-500" : (bz ?? 0) < 0 ? "text-amber-500" : "text-emerald-500"} />
        <Readout label="Density" value={num(at("N"), 1)} unit="cm⁻³" />
        <Readout label="Pressure" value={num(at("Pdyn"), 1)} unit="nPa" />
        <Readout label="Dst" value={num(dst, 0)} unit="nT" tone={lvl.c} />
        <Readout label="Kp" value={num(at("Kp"), 1)} unit={lvl.t} tone={lvl.c} />
      </div>
      <Plot height={620} data={data} layout={layout as Partial<Plotly.Layout>} />
      <p className="text-[10px] text-ink-faint">Hourly OMNI2 data{s.step_hours > 1 ? ` (shown every ${s.step_hours} h; extremes kept)` : ""}. Red shading: catalogued storms; dotted purple: your annotations. Southward Bz (negative) with fast wind drives Dst down.</p>
    </Glass>
  );
}

function Forecast({ start, end, model, evalHorizons }: { start: string; end: string; model: Status["model"]; evalHorizons: Record<string, { rmse: number; rmse_persistence: number; skill: number; forecast_hits: number; storm_onsets: number }> | null }) {
  const [h, setH] = useState(6);
  const fc = useApi<Fc>(model ? `/api/spaceweather/forecast?start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}&horizon=${h}` : null, [h, start, end]);
  if (!model) return null;
  const hs = model.horizons;
  const ph = evalHorizons ? Object.entries(evalHorizons).sort((a, b) => +a[0] - +b[0]) : [];
  return (
    <Glass className="hud space-y-4 p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="kicker">Forecast experiment · model {model.version}</div>
          <h3 className="mt-1 text-lg font-bold">Dst {h} h ahead: gradient boosting vs persistence</h3>
          <p className="text-xs text-ink-soft">Trained on {model.train_years[0]}–{model.train_years[model.train_years.length - 1]}, tested on {model.test_years.join(", ")}. Each forecast uses only data available at its issue time.</p>
        </div>
        <Segmented value={String(h)} onChange={(v) => setH(+v)} options={hs.map((x) => ({ value: String(x), label: `+${x} h` }))} />
      </div>
      {fc.error && <ErrorNotice message={fc.error.message} />}
      {fc.data && fc.data.in_sample_years.length > 0 && (
        <Notice title={`This window is inside the training years (${fc.data.in_sample_years.join(", ")})`}>Forecasts here are in-sample. Pick an event in the test years ({model.test_years.join(", ")}) for an honest comparison.</Notice>)}
      {fc.loading && <div className="flex items-center gap-2 text-sm text-ink-soft"><Spinner /> Forecasting…</div>}
      {fc.data && (
        <Plot height={300} data={[
          { x: fc.data.valid, y: fc.data.truth, type: "scatter", mode: "lines", name: "Observed Dst", line: { color: "#ef4444", width: 2 } },
          { x: fc.data.valid, y: fc.data.forecast, type: "scatter", mode: "lines", name: `Model (+${h} h)`, line: { color: "#7c3aed", width: 2 } },
          { x: fc.data.valid, y: fc.data.persistence, type: "scatter", mode: "lines", name: "Persistence", line: { color: "#94a3b8", width: 1.5, dash: "dash" } },
        ]} layout={{ xaxis: { type: "date", title: { text: "Valid time (UTC)" } }, yaxis: { title: { text: "Dst (nT)" } },
          shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: -50, y1: -50, line: { color: "rgba(239,68,68,0.4)", dash: "dot", width: 1 } }] } as Partial<Plotly.Layout>} />)}
      {ph.length > 0 && (
        <div className="grid gap-4 lg:grid-cols-[1fr_1.1fr]">
          <Plot height={220} data={[
            { x: ph.map(([k]) => `+${k} h`), y: ph.map(([, m]) => m.rmse), type: "bar", name: "Model", marker: { color: "#7c3aed" } },
            { x: ph.map(([k]) => `+${k} h`), y: ph.map(([, m]) => m.rmse_persistence), type: "bar", name: "Persistence", marker: { color: "#94a3b8" } },
          ]} layout={{ barmode: "group", yaxis: { title: { text: "Test RMSE (nT)" } }, margin: { l: 56, r: 10, t: 24, b: 30 } } as Partial<Plotly.Layout>} />
          <table className="w-full self-center text-xs">
            <thead><tr className="text-left text-ink-faint"><th className="pb-2">Horizon</th><th>RMSE</th><th>Persistence</th><th>Skill</th><th>Storm onsets</th></tr></thead>
            <tbody>{ph.map(([k, m]) => (
              <tr key={k} className="border-t border-white/70"><td className="py-1.5 font-semibold">+{k} h</td><td>{m.rmse.toFixed(1)}</td><td>{m.rmse_persistence.toFixed(1)}</td>
                <td className={m.skill > 0 ? "font-semibold text-emerald-600" : "text-rose-600"}>{(m.skill * 100).toFixed(0)}%</td><td>{m.forecast_hits}/{m.storm_onsets}</td></tr>))}</tbody>
          </table>
        </div>)}
    </Glass>
  );
}

export default function SpaceWeather() {
  const st = useApi<Status>("/api/spaceweather/status");
  const ev = useApi<{ catalog: Ev[]; annotations: Ann[] }>("/api/spaceweather/events");
  const evals = useApi<ModuleEvaluation[]>("/api/evaluation");
  const { jobs } = useJobs();
  const [win, setWin] = useState<{ start: string; end: string; label: string } | null>(null);
  const [ann, setAnn] = useState({ label: "", note: "" });
  useOnJobFinished(() => { st.reload(); ev.reload(); evals.reload(); });

  const cachedEvents = useMemo(() => (ev.data?.catalog ?? []).filter((e) => e.cached), [ev.data]);
  useEffect(() => {
    if (!win && cachedEvents.length) {
      const e = cachedEvents.find((x) => x.id === "2017-09") ?? cachedEvents[cachedEvents.length - 1];
      setWin({ start: e.start, end: e.end, label: e.name });
    }
  }, [cachedEvents, win]);
  const series = useApi<Series>(win ? `/api/spaceweather/series?start=${encodeURIComponent(win.start)}&end=${encodeURIComponent(win.end)}&vars=V,Bz,N,Pdyn,Dst,Kp` : null, [win?.start, win?.end]);
  const swEval = evals.data?.find((e) => e.module === "spaceweather");
  const perH = (swEval?.active as unknown as { per_horizon?: Record<string, never> } | null)?.per_horizon ?? null;
  const job = activeJob(jobs, "spaceweather");
  const noData = st.data && st.data.cached_years.length === 0;

  async function addAnnotation() {
    if (!win || !ann.label.trim()) return;
    await api("/api/spaceweather/annotations", { method: "POST", body: JSON.stringify({ start: win.start, end: win.end, label: ann.label, note: ann.note }) });
    setAnn({ label: "", note: "" });
    ev.reload();
  }

  return (
    <Page title="Space Weather Explorer"
      subtitle="Replay historical geomagnetic storms from NASA OMNIWeb: solar wind measured at L1 next to Earth's response (Dst, Kp). Annotate events and compare a trained Dst forecaster with a persistence baseline, with a chronological split and a visible forecast horizon."
      icon={<IconBadge gradient="from-amber-400 to-rose-500"><CloudLightning className="h-6 w-6" /></IconBadge>}>
      <LiveSolarWind />
      <ModuleStatus ids={["omni2"]} />
      {st.error && <ErrorNotice message={st.error.message} />}
      {job && <JobCard job={job} />}
      {noData && !job && (
        <Glass className="hud flex flex-wrap items-center justify-between gap-4 p-6">
          <div>
            <div className="kicker">No data yet</div>
            <p className="mt-1 max-w-xl text-sm text-ink-soft">Download hourly OMNI2 files for 2003, 2010–2017 and 2024: the Halloween 2003, St. Patrick's Day 2015, September 2017 and May 2024 storms, plus a training span. ~35 MB.</p>
          </div>
          <JobButton task="spaceweather.download" module={"spaceweather"} label={<><Wind className="h-4 w-4" /> Download OMNIWeb data</>} />
        </Glass>)}
      {st.data && !noData && (
        <div className="grid gap-4 xl:grid-cols-[340px_1fr]">
          <div className="space-y-4">
            <Glass className="p-4">
              <div className="kicker mb-2">Storm catalog</div>
              <div className="scroll-thin max-h-[360px] space-y-1.5 overflow-y-auto pr-1">
                {ev.data?.catalog.map((e) => (
                  <button key={e.id} disabled={!e.cached} onClick={() => setWin({ start: e.start, end: e.end, label: e.name })}
                    className={`w-full rounded-2xl px-3 py-2 text-left transition disabled:opacity-40 ${win?.start === e.start ? "bg-white shadow" : "hover:bg-white/60"}`}>
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-sm font-semibold">{e.name}</span>
                      {e.min_dst != null && <span className={`font-mono text-[11px] font-bold ${stormLevel(e.min_dst).c}`}>{Math.round(e.min_dst)} nT</span>}
                    </div>
                    <div className="text-[11px] text-ink-soft">{e.start.slice(0, 10)}{e.cached ? (e.max_kp != null ? ` · Kp ${e.max_kp.toFixed(1)} · ${Math.round(e.max_v ?? 0)} km/s` : "") : " · year not downloaded"}</div>
                  </button>))}
              </div>
            </Glass>
            <Glass className="space-y-2 p-4">
              <div className="kicker flex items-center gap-1.5"><CalendarRange className="h-3 w-3" /> Custom window</div>
              <div className="grid grid-cols-2 gap-2">
                <input type="datetime-local" className="input px-2 py-1.5 text-xs" value={win?.start.slice(0, 16) ?? ""} onChange={(e) => win && setWin({ ...win, start: `${e.target.value}:00Z`, label: "Custom window" })} />
                <input type="datetime-local" className="input px-2 py-1.5 text-xs" value={win?.end.slice(0, 16) ?? ""} onChange={(e) => win && setWin({ ...win, end: `${e.target.value}:00Z`, label: "Custom window" })} />
              </div>
              <p className="text-[10px] text-ink-faint">Cached years: {st.data.cached_years.join(", ")}</p>
            </Glass>
            <Glass className="space-y-2 p-4">
              <div className="kicker">Annotate this window</div>
              <input className="input py-2 text-sm" placeholder="Label, e.g. CME shock arrival" value={ann.label} onChange={(e) => setAnn({ ...ann, label: e.target.value })} />
              <textarea className="input py-2 text-xs" rows={2} placeholder="Note (optional)" value={ann.note} onChange={(e) => setAnn({ ...ann, note: e.target.value })} />
              <WriteButton className="btn-primary w-full py-2 text-xs" disabled={!ann.label.trim() || !win} onClick={addAnnotation}><Plus className="h-3.5 w-3.5" /> Save annotation</WriteButton>
              {ev.data?.annotations.map((a) => (
                <div key={a.id} className="flex items-center gap-2 rounded-xl bg-white/50 px-2 py-1.5 text-xs">
                  <button className="min-w-0 flex-1 truncate text-left font-semibold" onClick={() => setWin({ start: a.start, end: a.end, label: a.label })}>{a.label}</button>
                  <span className="text-[10px] text-ink-faint">{a.start.slice(0, 10)}</span>
                  <WriteButton className="text-ink-faint hover:text-rose-600" onClick={async () => { await api(`/api/spaceweather/annotations/${a.id}`, { method: "DELETE" }); ev.reload(); }}><Trash2 className="h-3 w-3" /></WriteButton>
                </div>))}
            </Glass>
          </div>
          <div className="min-w-0 space-y-4">
            <div className="grid gap-4 2xl:grid-cols-[1fr_380px]">
              <Glass className="hud p-4">
                <div className="kicker mb-2">Mission context · where the data comes from</div>
                <Suspense fallback={<div className="flex h-[300px] items-center justify-center"><Spinner /></div>}>
                  <HelioView date={win?.start} show={{ l1: true, wind: true }} height={300}
                    caption="Solar wind streams past L1 (ACE / Wind / DSCOVR, ~1.5 million km sunward) about an hour before it reaches Earth's magnetosphere." />
                </Suspense>
              </Glass>
              {series.data && (
                <Glass className="hud space-y-2 p-4">
                  <div className="kicker">{win?.label}</div>
                  <div className="grid grid-cols-2 gap-2">
                    <Kpi label="Min Dst" value={`${num(series.data.summary.min_dst, 0)} nT`} sub={stormLevel(series.data.summary.min_dst).t + " storm"} accent="#ef4444" />
                    <Kpi label="Max Kp" value={num(series.data.summary.max_kp, 1)} accent="#16a34a" />
                    <Kpi label="Peak wind" value={`${num(series.data.summary.max_v, 0)} km/s`} accent="#d97706" />
                    <Kpi label="Min Bz" value={`${num(series.data.summary.min_bz, 1)} nT`} accent="#0891b2" />
                  </div>
                  <p className="text-[10px] text-ink-faint">{(series.data.summary.coverage * 100).toFixed(0)}% of hours have solar-wind data.</p>
                  {!st.data.model && <div className="pt-1"><JobButton task="spaceweather.train" module={"spaceweather"} className="btn-primary w-full" label={<><Zap className="h-4 w-4" /> Train forecaster</>} /></div>}
                </Glass>)}
            </div>
            {series.error && <ErrorNotice message={series.error.message} />}
            {series.loading && <div className="flex items-center gap-2 text-sm text-ink-soft"><Spinner /> Loading OMNIWeb series…</div>}
            {series.data && <Replay s={series.data} events={ev.data?.catalog ?? []} notes={ev.data?.annotations ?? []} />}
            {win && <Forecast start={win.start} end={win.end} model={st.data.model} evalHorizons={perH} />}
          </div>
        </div>)}
    </Page>
  );
}
