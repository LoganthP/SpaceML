import { FileInput, WriteButton } from "../lib/capabilities";
import { ChevronLeft, ChevronRight, Clapperboard, Eye, EyeOff, Pause, Play, Radar, Rewind, Sparkles, Telescope, Trophy } from "lucide-react";
import { lazy, Suspense, useCallback, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { DataQualityPanel, type Quality } from "../components/DataQuality";
import { EvidencePanel } from "../components/Evidence";
import { JobButton, JobCard, activeJob, useJobs, useOnJobFinished } from "../components/Jobs";
import Plot from "../components/LazyPlot";
import { ErrorNotice, Glass, IconBadge, Notice, OriginBadge, Page, Segmented, Spinner } from "../components/ui";
import { api, useApi } from "../lib/api";
import { num, timeAgo } from "../lib/format";
import type { Run } from "../lib/types";
import { ModuleStatus } from "../components/Catalog";

const HelioView = lazy(() => import("../components/HelioView"));

interface Ev { id: string; name: string; start: string; end: string; cached: boolean; min_dst?: number }
interface Status { cached_years: number[]; model: null | { version: string; horizons: number[]; train_years: number[]; test_years: number[] } }
interface Lead { issued: string | null; lead_hours: number | null }
interface HStat {
  horizon: number; n: number; rmse: number | null; rmse_persistence: number | null; skill: number | null; mae: number | null;
  better_hours: number; worse_hours: number; tie_hours: number; storm_rmse: number | null; storm_rmse_persistence: number | null;
  forecast_min: number | null; observed_min: number | null; peak_error?: number; first_warning: Lead | null;
  persistence_first_warning: Lead | null; degraded_issuances: number;
}
interface TL { time: string; kind: string; label: string; detail: string }
interface ReplayResult {
  window: { start: string; end: string }; horizons: number[]; lookback_hours: number; tie_nt: number; storm_threshold: number;
  issued: string[]; inputs_complete: boolean[]; persistence: (number | null)[];
  forecast: Record<string, (number | null)[]>; truth: Record<string, (number | null)[]>; verdict: Record<string, (number | null)[]>;
  per_horizon: Record<string, HStat>; observed: Record<string, (number | null)[]> & { time: string[] };
  timeline: TL[]; onset: string | null; quality: Quality; causality_check: { passed: boolean; detail: string };
  model: { version: string; train_years: number[]; test_years: number[] }; in_sample_years: number[];
  event: { id: string; name: string } | null;
}
interface ReplayRow { id: string; status: Run["status"]; error: string | null; created_at: string; origin: "computed" | "bundled"; model_version: string | null; input_ref: { event_id?: string } }

const H = (ms: number) => new Date(ms).toISOString();
const addH = (iso: string, h: number) => H(new Date(iso).getTime() + h * 3600_000);
const fmtT = (iso: string) => `${iso.slice(5, 10)} ${iso.slice(11, 16)} UTC`;
const level = (d: number | null | undefined) => d == null ? "—" : d <= -250 ? "superstorm" : d <= -100 ? "intense storm" : d <= -50 ? "moderate storm" : "quiet / weak";

/** Contiguous issuance spans where persistence beat the model. */
function spans(issued: string[], v: (number | null)[], want: number) {
  const out: { from: string; to: string; hours: number }[] = [];
  let s = -1;
  v.forEach((x, i) => {
    if (x === want && s < 0) s = i;
    if ((x !== want || i === v.length - 1) && s >= 0) {
      const e = x === want ? i : i - 1;
      out.push({ from: issued[s], to: issued[e], hours: e - s + 1 });
      s = -1;
    }
  });
  return out.sort((a, b) => b.hours - a.hours);
}

function Question({ n, title, icon, children }: { n: number; title: string; icon: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="glass-inset relative overflow-hidden p-4">
      <div className="pointer-events-none absolute -right-3 -top-5 font-display text-7xl font-black text-violet-500/10">{n}</div>
      <div className="flex items-center gap-2 text-sm font-bold">{icon}{title}</div>
      <div className="mt-2 space-y-1 text-[13px] leading-relaxed">{children}</div>
    </div>
  );
}

function ReplayView({ run }: { run: Run<ReplayResult> }) {
  const r = run.result!;
  const [h, setH] = useState(String(r.horizons.includes(6) ? 6 : r.horizons[Math.floor(r.horizons.length / 2)]));
  const n = r.issued.length;
  const onsetIdx = r.onset ? Math.max(0, r.issued.findIndex((t) => t >= addH(r.onset!, -12))) : 0;
  const [i, setI] = useState(onsetIdx > 0 ? onsetIdx : Math.floor(n / 3));
  const [playing, setPlaying] = useState(false);
  const [reveal, setReveal] = useState(false);
  useEffect(() => {
    if (!playing) return;
    const t = setInterval(() => setI((x) => { if (x >= n - 1) { setPlaying(false); return x; } return x + 1; }), 260);
    return () => clearInterval(t);
  }, [playing, n]);
  const st = r.per_horizon[h];
  const hn = Number(h);
  const t = r.issued[i];
  const obs = r.observed;
  const lastIdx = obs.time.findIndex((x) => x > t);
  const cut = lastIdx < 0 ? obs.time.length : lastIdx;

  const fig = useMemo(() => {
    const issuedUpTo = r.issued.slice(0, i + 1);
    const validTimes = issuedUpTo.map((x) => addH(x, hn));
    const fc = r.forecast[h].slice(0, i + 1);
    const pers = r.persistence.slice(0, i + 1);
    const dstPast = obs.Dst.map((v, k) => (k < cut ? v : null));
    const dstFuture = obs.Dst.map((v, k) => (k >= cut - 1 ? v : null));
    const verdict = r.verdict[h];
    const vColors = verdict.map((v, k) => k > i ? "rgba(148,163,184,0.18)" : v === 1 ? "#10b981" : v === -1 ? "#f43f5e" : v === 0 ? "#cbd5e1" : "rgba(148,163,184,0.3)");
    const xr: [string, string] = [obs.time[0], obs.time[obs.time.length - 1]];
    const data: Plotly.Data[] = [
      { x: obs.time, y: dstPast, type: "scatter", mode: "lines", name: "Observed Dst (known at issuance)", line: { color: "#ef4444", width: 2.5 }, xaxis: "x", yaxis: "y" },
      ...(reveal ? [{ x: obs.time, y: dstFuture, type: "scatter", mode: "lines", name: "Observed later (not known yet)", line: { color: "#ef4444", width: 1.5, dash: "dot" }, opacity: 0.45, xaxis: "x", yaxis: "y" } as Plotly.Data] : []),
      { x: validTimes, y: fc, type: "scatter", mode: "lines", name: `Model forecast +${h} h (at valid time)`, line: { color: "#7c3aed", width: 2.5 }, xaxis: "x", yaxis: "y" },
      { x: validTimes, y: pers, type: "scatter", mode: "lines", name: "Persistence baseline", line: { color: "#64748b", width: 1.6, dash: "dash" }, xaxis: "x", yaxis: "y" },
      { x: [addH(t, hn)], y: [r.forecast[h][i]], type: "scatter", mode: "markers", name: "Latest forecast", marker: { color: "#7c3aed", size: 11, line: { color: "white", width: 2 } }, xaxis: "x", yaxis: "y", showlegend: false },
      { x: r.issued, y: r.issued.map(() => 1), type: "bar", marker: { color: vColors }, name: "Model vs persistence", xaxis: "x", yaxis: "y2", showlegend: false,
        hovertemplate: "%{x}<extra>issued here</extra>" },
      { x: obs.time, y: obs.V.map((v, k) => (k < cut ? v : null)), type: "scatter", mode: "lines", line: { color: "#f59e0b", width: 1.5 }, name: "Speed", xaxis: "x", yaxis: "y3", showlegend: false },
      { x: obs.time, y: obs.Bz.map((v, k) => (k < cut ? v : null)), type: "scatter", mode: "lines", line: { color: "#06b6d4", width: 1.5 }, name: "IMF Bz", xaxis: "x", yaxis: "y4", showlegend: false, fill: "tozeroy", fillcolor: "rgba(6,182,212,0.12)" },
      { x: obs.time, y: obs.Pdyn.map((v, k) => (k < cut ? v : null)), type: "scatter", mode: "lines", line: { color: "#ec4899", width: 1.5 }, name: "Pressure", xaxis: "x", yaxis: "y5", showlegend: false },
    ];
    const shapes: Partial<Plotly.Shape>[] = [
      { type: "rect", xref: "x", yref: "paper", x0: addH(t, -r.lookback_hours), x1: t, y0: 0, y1: 1, fillcolor: "rgba(124,58,237,0.08)", line: { width: 0 }, layer: "below" },
      { type: "rect", xref: "x", yref: "paper", x0: t, x1: xr[1], y0: 0, y1: 1, fillcolor: "rgba(148,163,184,0.10)", line: { width: 0 }, layer: "below" },
      { type: "line", xref: "x", yref: "paper", x0: t, x1: t, y0: 0, y1: 1, line: { color: "#7c3aed", width: 2 } },
      { type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: r.storm_threshold, y1: r.storm_threshold, line: { color: "rgba(249,115,22,0.6)", width: 1, dash: "dot" } },
      ...r.timeline.map((e) => ({ type: "line", xref: "x", yref: "y", x0: e.time, x1: e.time, y0: Math.min(...(obs.Dst.filter((v) => v != null) as number[])), y1: 30,
        line: { color: "rgba(168,85,247,0.35)", width: 1, dash: "dot" } }) as Partial<Plotly.Shape>),
    ];
    const layout: Partial<Plotly.Layout> = {
      shapes, hovermode: "x unified", margin: { l: 58, r: 14, t: 34, b: 30 },
      legend: { orientation: "h", y: 1.07, x: 0, font: { size: 11 } },
      xaxis: { range: xr, type: "date", anchor: "y5" },
      yaxis: { domain: [0.5, 1], title: { text: "Dst (nT)", font: { size: 11 } } },
      yaxis2: { domain: [0.44, 0.48], showticklabels: false, showgrid: false, range: [0, 1] },
      yaxis3: { domain: [0.30, 0.42], title: { text: "km/s", font: { size: 10 } } },
      yaxis4: { domain: [0.15, 0.28], title: { text: "Bz nT", font: { size: 10 } } },
      yaxis5: { domain: [0, 0.13], title: { text: "nPa", font: { size: 10 } } },
      annotations: [
        { x: addH(t, -r.lookback_hours / 2), y: 1.0, xref: "x", yref: "paper", text: "inputs the model used", showarrow: false, font: { size: 10, color: "#7c3aed" }, yanchor: "bottom" },
        { x: t, y: 0.465, xref: "x", yref: "paper", text: "◀ issued so far", showarrow: false, xanchor: "right", font: { size: 9, color: "#64748b" } },
      ],
    };
    return { data, layout };
  }, [r, h, hn, i, t, cut, obs, reveal]);

  const verd = r.verdict[h][i];
  const truth = r.truth[h][i];
  const fcNow = r.forecast[h][i];
  const persNow = r.persistence[i];
  const worse = spans(r.issued, r.verdict[h], -1).slice(0, 3);
  const better = spans(r.issued, r.verdict[h], 1).slice(0, 2);
  const minT = r.timeline.find((e) => e.kind === "min");
  const shock = r.timeline.find((e) => e.kind === "shock");
  const fw = st.first_warning, pw = st.persistence_first_warning;

  return (
    <div className="space-y-4">
      <div className="grid gap-3 lg:grid-cols-3">
        <Question n={1} title="What happened?" icon={<Radar className="h-4 w-4 text-rose-500" />}>
          <p>Dst fell to <b className="text-rose-600">{num(st.observed_min ?? null, 0)} nT</b> ({level(st.observed_min)}){minT ? <> at <b>{fmtT(minT.time)}</b></> : null}.</p>
          {shock && <p className="text-ink-soft">{shock.label} at {fmtT(shock.time)} - {shock.detail.split(" (")[0]}.</p>}
          <p className="text-[11px] text-ink-faint">Timeline below is detected from the OMNI data, not hand-entered.</p>
        </Question>
        <Question n={2} title="What could the model know at the time?" icon={<Telescope className="h-4 w-4 text-violet-500" />}>
          {fw?.issued ? <p>First +{h} h storm warning issued <b>{fmtT(fw.issued)}</b> - <b className={(fw.lead_hours ?? 0) > 0 ? "text-emerald-600" : "text-amber-600"}>
            {(fw.lead_hours ?? 0) >= 0 ? `${fw.lead_hours} h before` : `${-(fw.lead_hours ?? 0)} h after`}</b> Dst crossed −50 nT.
            Persistence can only say so {pw?.lead_hours == null || pw.lead_hours === 0 ? "at the moment it is observed" : pw.lead_hours < 0 ? `${-pw.lead_hours} h after` : `${pw.lead_hours} h before`}.</p>
            : <p>The model never forecast storm level at +{h} h in this window.</p>}
          <p className="text-ink-soft">Forecast minimum {num(st.forecast_min ?? null, 0)} nT vs observed {num(st.observed_min ?? null, 0)} nT
            {st.peak_error != null && <> (peak {st.peak_error > 0 ? "under" : "over"}-forecast by {Math.abs(st.peak_error).toFixed(0)} nT)</>}.</p>
        </Question>
        <Question n={3} title="Where did the baseline do better?" icon={<Trophy className="h-4 w-4 text-amber-500" />}>
          <p>Persistence was closer in <b className="text-rose-600">{st.worse_hours} h</b>, the model in <b className="text-emerald-600">{st.better_hours} h</b> (ties within ±{r.tie_nt} nT: {st.tie_hours} h).</p>
          {worse.length ? <p className="text-ink-soft">Longest baseline wins: {worse.map((w) => `${fmtT(w.from)} (${w.hours} h)`).join(", ")}.</p> : null}
          {better.length ? <p className="text-ink-soft">Longest model wins: {better.map((w) => `${fmtT(w.from)} (${w.hours} h)`).join(", ")}.</p> : null}
        </Question>
      </div>

      <Glass className="p-4">
        <div className="mb-2 flex flex-wrap items-center gap-2">
          <Segmented value={h} onChange={setH} options={r.horizons.map((x) => ({ value: String(x), label: `+${x} h` }))} />
          <div className="ml-auto flex items-center gap-1.5">
            <button className="btn-ghost px-2.5 py-1.5" title="Start" onClick={() => setI(0)}><Rewind className="h-4 w-4" /></button>
            <button className="btn-ghost px-2.5 py-1.5" title="Previous hour" onClick={() => setI((x) => Math.max(0, x - 1))}><ChevronLeft className="h-4 w-4" /></button>
            <button className="btn-primary px-3 py-1.5" onClick={() => { if (i >= n - 1) setI(0); setPlaying((p) => !p); }}>
              {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />} {playing ? "Pause" : "Play"}
            </button>
            <button className="btn-ghost px-2.5 py-1.5" title="Next hour" onClick={() => setI((x) => Math.min(n - 1, x + 1))}><ChevronRight className="h-4 w-4" /></button>
            <button className={`btn-ghost px-2.5 py-1.5 text-xs ${reveal ? "text-violet-600" : ""}`} onClick={() => setReveal((v) => !v)}
              title="Show what was observed after the issuance time (the model could not see it)">
              {reveal ? <Eye className="h-4 w-4" /> : <EyeOff className="h-4 w-4" />} future
            </button>
          </div>
        </div>
        <input type="range" min={0} max={n - 1} value={i} onChange={(e) => { setPlaying(false); setI(Number(e.target.value)); }}
          className="w-full accent-violet-600" aria-label="Issuance time" />
        <div className="relative h-3">
          {r.timeline.map((e) => {
            const k = r.issued.findIndex((x) => x >= e.time);
            if (k < 0) return null;
            return <button key={e.kind + e.time} onClick={() => { setPlaying(false); setI(k); }} title={`${e.label} · ${fmtT(e.time)}`}
              className="absolute top-0 h-3 w-1.5 -translate-x-1/2 rounded-full bg-violet-400/70 hover:bg-violet-600" style={{ left: `${(k / Math.max(1, n - 1)) * 100}%` }} />;
          })}
        </div>
        <Plot data={fig.data} layout={fig.layout} height={620} />
        <div className="mt-1 flex flex-wrap gap-3 text-[11px] text-ink-soft">
          <span><span className="mr-1 inline-block h-2 w-3 rounded-sm bg-emerald-500" />model closer</span>
          <span><span className="mr-1 inline-block h-2 w-3 rounded-sm bg-rose-500" />persistence closer</span>
          <span><span className="mr-1 inline-block h-2 w-3 rounded-sm bg-slate-300" />tie (±{r.tie_nt} nT)</span>
          <span><span className="mr-1 inline-block h-2 w-3 rounded-sm bg-violet-300" />shaded: the {r.lookback_hours} h of data the model used</span>
          <span>grey: after the issuance time - unknown to the model</span>
        </div>
      </Glass>

      <div className="grid gap-4 xl:grid-cols-[1.2fr_1fr]">
        <Glass className="p-4">
          <div className="kicker">Forecast issued</div>
          <div className="font-mono text-2xl font-bold">{fmtT(t)}</div>
          <div className="mt-1 text-xs text-ink-soft">
            Inputs: solar wind + IMF at L1 (time-shifted to Earth) and Dst/Kp up to this hour ·{" "}
            {r.inputs_complete[i] ? <span className="text-emerald-600">complete</span> : <span className="text-amber-600">incomplete - some solar-wind values missing this hour</span>}
          </div>
          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-sm">
              <thead><tr className="text-left text-[11px] text-ink-faint"><th className="pb-1">Valid at</th><th>Model</th><th>Persistence</th><th>Observed later</th><th>Closer</th></tr></thead>
              <tbody>
                {r.horizons.map((x) => {
                  const v = r.verdict[String(x)][i];
                  return (
                    <tr key={x} className={`border-t border-white/60 ${String(x) === h ? "bg-violet-500/5" : ""}`}>
                      <td className="py-1.5 font-mono text-xs">+{x} h · {fmtT(addH(t, x)).slice(6)}</td>
                      <td className="font-semibold text-violet-600 tabular-nums">{num(r.forecast[String(x)][i], 0)} nT</td>
                      <td className="tabular-nums text-ink-soft">{num(r.persistence[i], 0)} nT</td>
                      <td className="tabular-nums">{num(r.truth[String(x)][i], 0)} nT</td>
                      <td>{v === 1 ? <span className="chip bg-emerald-100 text-emerald-700">model</span> : v === -1 ? <span className="chip bg-rose-100 text-rose-700">persistence</span> : v === 0 ? <span className="chip bg-slate-100 text-slate-600">tie</span> : "—"}</td>
                    </tr>);
                })}
              </tbody>
            </table>
          </div>
          {verd != null && truth != null && fcNow != null && persNow != null && (
            <p className="mt-2 text-xs text-ink-soft">At +{h} h the model was off by <b>{Math.abs(fcNow - truth).toFixed(0)} nT</b>, persistence by <b>{Math.abs(persNow - truth).toFixed(0)} nT</b>.</p>)}
        </Glass>
        <Glass className="p-4">
          <div className="kicker">Timeline · detected from the data</div>
          <ol className="mt-2 space-y-1.5">
            {r.timeline.map((e) => {
              const k = r.issued.findIndex((x) => x >= e.time);
              return (
                <li key={e.kind + e.time}>
                  <button className={`w-full rounded-xl px-2 py-1.5 text-left transition hover:bg-white/60 dark:hover:bg-white/5 ${k === i ? "bg-violet-500/10" : ""}`}
                    onClick={() => { if (k >= 0) { setPlaying(false); setI(k); } }}>
                    <div className="flex items-baseline gap-2"><span className="font-mono text-[11px] text-violet-600">{fmtT(e.time)}</span><span className="text-sm font-semibold">{e.label}</span></div>
                    <div className="text-[11px] text-ink-soft">{e.detail}</div>
                  </button>
                </li>);
            })}
          </ol>
        </Glass>
      </div>

      <Glass className="p-4">
        <div className="kicker">Scoreboard · every horizon, this storm only</div>
        <div className="mt-2 overflow-x-auto">
          <table className="w-full min-w-[720px] text-sm">
            <thead><tr className="text-left text-[11px] text-ink-faint"><th className="pb-1">Horizon</th><th>RMSE model</th><th>RMSE persistence</th><th>Skill</th><th>Storm-hour RMSE</th><th>Hours model / baseline closer</th><th>First warning (lead)</th></tr></thead>
            <tbody>
              {r.horizons.map((x) => {
                const s = r.per_horizon[String(x)];
                return (
                  <tr key={x} className="border-t border-white/60">
                    <td className="py-1.5 font-semibold">+{x} h</td>
                    <td className="tabular-nums">{num(s.rmse, 1)}</td><td className="tabular-nums text-ink-soft">{num(s.rmse_persistence, 1)}</td>
                    <td className={`font-semibold tabular-nums ${(s.skill ?? 0) > 0 ? "text-emerald-600" : "text-rose-600"}`}>{s.skill == null ? "—" : `${s.skill > 0 ? "+" : ""}${(s.skill * 100).toFixed(0)}%`}</td>
                    <td className="tabular-nums">{num(s.storm_rmse, 1)} <span className="text-ink-faint">vs {num(s.storm_rmse_persistence, 1)}</span></td>
                    <td className="tabular-nums"><span className="text-emerald-600">{s.better_hours}</span> / <span className="text-rose-600">{s.worse_hours}</span></td>
                    <td className="tabular-nums">{s.first_warning?.lead_hours == null ? "none" : `${s.first_warning.lead_hours} h`}</td>
                  </tr>);
              })}
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-[11px] text-ink-faint">Skill = 1 − RMSE<sub>model</sub>/RMSE<sub>persistence</sub>. Model {r.model.version} trained on {r.model.train_years[0]}–{r.model.train_years[r.model.train_years.length - 1]}, tested on {r.model.test_years.join(", ")}.</p>
      </Glass>
    </div>
  );
}

export default function MissionReplay() {
  const [params, setParams] = useSearchParams();
  const eventId = params.get("event") ?? "2024-05";
  const events = useApi<{ catalog: Ev[] }>("/api/spaceweather/events");
  const status = useApi<Status>("/api/spaceweather/status");
  const replays = useApi<ReplayRow[]>(`/api/spaceweather/replays?event_id=${eventId}`, [eventId]);
  const ev = events.data?.catalog.find((e) => e.id === eventId);
  const quality = useApi<Quality>(ev ? `/api/spaceweather/quality?start=${ev.start}&end=${ev.end}` : null, [ev?.id]);
  const runRow = params.get("run") ? { id: params.get("run")! } : replays.data?.find((r) => r.status === "succeeded");
  const run = useApi<Run<ReplayResult>>(runRow ? `/api/runs/${runRow.id}` : null, [runRow?.id]);
  const { jobs } = useJobs();
  const busy = activeJob(jobs, "spaceweather");
  const lastJob = jobs.find((j) => j.module === "spaceweather");
  const [starting, setStarting] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const reloadAll = useCallback(() => { events.reload(); status.reload(); replays.reload(); quality.reload(); }, [events, status, replays, quality]);
  useOnJobFinished(reloadAll);
  const pending = replays.data?.find((r) => r.status === "queued" || r.status === "running");
  useEffect(() => {
    if (!pending) return;
    const t = setInterval(replays.reload, 1500);
    return () => clearInterval(t);
  }, [pending, replays.reload]);

  const haveData = ev ? status.data?.cached_years.includes(Number(ev.start.slice(0, 4))) : false;
  const haveModel = !!status.data?.model;

  async function runReplay() {
    setErr(null); setStarting(true);
    try {
      await api("/api/runs", { method: "POST", body: JSON.stringify({ module: "spaceweather", input: { event_id: eventId } }) });
      replays.reload();
      setParams((p) => { p.delete("run"); return p; });
    } catch (e) { setErr((e as Error).message); } finally { setStarting(false); }
  }

  const res = run.data?.result;
  return (
    <Page title="Mission Replay" subtitle="Step through a real geomagnetic storm hour by hour: what happened, what the forecaster could have said at the time, and where the simple baseline did better."
      icon={<IconBadge gradient="from-violet-600 to-rose-500"><Clapperboard className="h-6 w-6" /></IconBadge>}
      actions={run.data && <OriginBadge origin={run.data.origin} />}>
      <ModuleStatus ids={["omni2"]} />
      <div className="mb-4 flex flex-wrap gap-2">
        {(events.data?.catalog ?? []).map((e) => (
          <button key={e.id} onClick={() => setParams({ event: e.id })}
            className={`rounded-2xl border px-3 py-2 text-left transition ${e.id === eventId ? "border-violet-400 bg-violet-500/10" : "border-white/70 bg-white/40 hover:bg-white/70 dark:border-white/10 dark:bg-white/5"}`}>
            <div className="text-xs font-semibold">{e.name}</div>
            <div className="font-mono text-[10px] text-ink-soft">{e.start.slice(0, 10)} · {e.cached ? (e.min_dst != null ? `min Dst ${e.min_dst.toFixed(0)} nT` : "cached") : "not downloaded"}</div>
          </button>))}
      </div>

      {events.error && <ErrorNotice message={events.error.message} />}
      {res && run.data && (
        <>
          {res.in_sample_years.length > 0 && (
            <div className="mb-3"><Notice tone="warn" title={`In-sample: the model was trained on ${res.in_sample_years.join(", ")}`}>
              This replay is not a fair test - pick an event outside the training years (the May 2024 storm is out-of-sample for the default model).</Notice></div>)}
          <div className="mb-3 flex flex-wrap items-center gap-2 text-xs text-ink-soft">
            <Sparkles className="h-3.5 w-3.5 text-violet-500" />
            Replay computed {timeAgo(run.data.finished_at)} with model <code>{res.model.version}</code>
            {replays.data && replays.data.length > 1 && <> · {replays.data.length} replays of this event</>}
            <WriteButton className="btn-ghost ml-auto px-3 py-1.5 text-xs" onClick={runReplay} disabled={starting || !!pending || !haveModel}>
              {starting || pending ? <Spinner className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />} Recompute here
            </WriteButton>
          </div>
          <ReplayView run={run.data} />
          <div className="mt-4 grid gap-4 xl:grid-cols-2">
            <EvidencePanel runId={run.data.id} />
            <DataQualityPanel q={res.quality} title="Data quality in this window" />
          </div>
          <Glass className="mt-4 p-4">
            <div className="kicker mb-2">Where the data comes from · Sun → L1 → Earth on {res.window.start.slice(0, 10)}</div>
            <Suspense fallback={<div className="flex h-[300px] items-center justify-center"><Spinner /></div>}>
              <HelioView date={res.window.start} show={{ l1: true, wind: true }} height={320}
                caption="OMNI combines ACE/Wind/DSCOVR measurements at the L1 point, time-shifted to Earth's bow shock." />
            </Suspense>
          </Glass>
        </>
      )}

      {!res && (run.loading || replays.loading) && <div className="flex items-center gap-2 p-6 text-sm text-ink-soft"><Spinner /> Loading replay…</div>}
      {!res && !run.loading && !replays.loading && (
        <div className="grid gap-4 lg:grid-cols-[1.3fr_1fr]">
          <Glass className="space-y-3 p-5">
            <div className="font-display text-lg font-bold">{ev?.name ?? eventId}</div>
            {pending ? <div className="flex items-center gap-2 text-sm"><Spinner /> Replaying {ev?.name}…</div>
              : haveData && haveModel ? (
                <>
                  <p className="text-sm text-ink-soft">Data and forecaster are installed. The replay issues a forecast every hour of the event using only data up to that hour, then scores it against what was observed.</p>
                  <WriteButton className="btn-primary" onClick={runReplay} disabled={starting}>{starting ? <Spinner /> : <Play className="h-4 w-4" />} Run replay</WriteButton>
                </>
              ) : (
                <>
                  <p className="text-sm text-ink-soft">
                    {!haveData && <>OMNI2 {ev?.start.slice(0, 4)} is not on this machine yet. </>}
                    {!haveModel && <>No Dst forecaster is trained yet. </>}
                    One step downloads the NASA OMNIWeb years (≈3 MB each), trains the forecaster on 2010–2016 (tested on 2017) and replays May 2024 - a few minutes on a CPU.
                  </p>
                  <JobButton task="spaceweather.setup" module="spaceweather" label="Download, train and replay" />
                  {lastJob?.status === "source_unavailable" && (
                    <Notice tone="warn" title="NASA SPDF could not be reached from this machine">
                      <p>No result is shown until real data arrives. On a machine with internet, download <code>omni2_2024.dat</code> (and 2010–2017) from spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/ and upload them here, then press the button above again:</p>
                      <OmniImport />
                    </Notice>)}
                </>)}
            {err && <ErrorNotice message={err} />}
            {replays.data?.filter((r) => r.status !== "succeeded" && r.status !== "queued" && r.status !== "running").slice(0, 1).map((r) => (
              <ErrorNotice key={r.id} title={`Last replay: ${r.status.replace("_", " ")}`} message={r.error ?? ""} />))}
            {busy && <JobCard job={busy} />}
            {!busy && lastJob && lastJob.status !== "succeeded" && <JobCard job={lastJob} />}
          </Glass>
          {quality.data && <DataQualityPanel q={quality.data} title="Data for this event" />}
        </div>)}
    </Page>
  );
}


function OmniImport() {
  const [msg, setMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  return (
    <div className="space-y-1">
      <label className="btn-ghost cursor-pointer px-3 py-1.5 text-xs">
        {busy ? <Spinner /> : <Play className="h-3.5 w-3.5" />} Upload omni2_YYYY.dat files
        <FileInput type="file" multiple accept=".dat,.txt" className="hidden" onChange={async (e) => {
          const files = Array.from(e.target.files ?? []);
          if (!files.length) return;
          setBusy(true); setMsg(null);
          const fd = new FormData();
          files.forEach((f) => fd.append("files", f));
          try {
            const r = await api<{ imported: { file: string; records: number }[]; cached_years: number[] }>("/api/spaceweather/import", { method: "POST", body: fd });
            setMsg(`Imported ${r.imported.length} file(s); years on disk: ${r.cached_years.join(", ")}`);
          } catch (er) { setMsg((er as Error).message); } finally { setBusy(false); e.target.value = ""; }
        }} />
      </label>
      {msg && <p className="text-[11px]">{msg}</p>}
    </div>
  );
}
