import { motion } from "framer-motion";
import { useReadOnly } from "../lib/capabilities";
import { Activity, ArrowUpRight, BarChart3, CheckCircle2, CircleDashed, Clapperboard, CloudLightning, Command, FlaskConical, Highlighter, History, Mountain, Orbit, Package, Radio, Satellite, Sparkles } from "lucide-react";
import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { JobCard } from "../components/Jobs";
import type { Hotspot } from "../components/MissionMap";
import { DualGauge, Donut, Heatmap, Spark, StackBar } from "../components/MiniCharts";
import { ErrorNotice, OriginBadge, Spinner, StatusBadge } from "../components/ui";
import { useApi } from "../lib/api";
import { MODULE_META, num, pct, timeAgo, TERRAIN_COLORS } from "../lib/format";
import type { ModuleOverview, Overview as OverviewT } from "../lib/types";
import { CatalogSummary, useCatalog } from "../components/Catalog";

const MissionMap = lazy(() => import("../components/MissionMap"));
const ICONS = { telemetry: Activity, exoplanet: Orbit, mars: Mountain, workbench: FlaskConical, spaceweather: CloudLightning, missions: Satellite };

const pop = {
  hidden: { opacity: 0, y: 24, scale: 0.97 },
  show: { opacity: 1, y: 0, scale: 1, transition: { type: "spring", stiffness: 220, damping: 22 } },
};

function Tile({ className = "", children, to, kicker, icon: Icon, accent }: {
  className?: string; children: React.ReactNode; to?: string; kicker?: string; icon?: typeof Activity; accent?: string;
}) {
  const body = (
    <motion.div variants={pop} whileHover={{ y: -4 }} transition={{ type: "spring", stiffness: 300, damping: 20 }}
      className={`glass hud relative flex h-full flex-col overflow-hidden p-5 ${className}`}>
      {kicker && <div className="kicker flex items-center gap-1.5">{Icon && <Icon className="h-3.5 w-3.5" style={accent ? { color: accent } : undefined} />}{kicker}</div>}
      {children}
      {to && <ArrowUpRight className="absolute right-4 top-4 h-4 w-4 text-ink-faint" />}
      {to && <Link to={to} aria-label={kicker ?? "Open workspace"} className="absolute inset-0" />}
    </motion.div>
  );
  return to ? <div className="block h-full">{body}</div> : body;
}

/** A tile whose data is not installed yet: one centred hint plus the action, so the tile never looks empty. */
function Empty({ icon: Icon, color, text, action }: { icon: typeof Activity; color: string; text: string; action: string }) {
  const readOnly = useReadOnly();
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-2 py-4 text-center">
      <span className="grid h-10 w-10 place-items-center rounded-2xl" style={{ background: `${color}1f`, color }}><Icon className="h-5 w-5" aria-hidden="true" /></span>
      <p className="max-w-[16rem] text-xs text-ink-soft">{readOnly ? "No sample result is published for this module." : text}</p>
      <span className="text-[11px] font-semibold" style={{ color }}>{readOnly ? "Vercel compute required" : `${action} →`}</span>
    </div>);
}

const fmtHead = (m: ModuleOverview) => !m.headline ? null
  : m.module === "exoplanet" ? num(m.headline.model, 3) : pct(m.headline.model, 0);

function Clock() {
  const [t, setT] = useState(new Date());
  useEffect(() => { const i = setInterval(() => setT(new Date()), 1000); return () => clearInterval(i); }, []);
  return <span className="tabular-nums">{t.toISOString().slice(11, 19)}</span>;
}

function TelemetryStrip() {
  const { data } = useApi<{ chan_id: string; spacecraft: string; step: number; n_steps: number; values: (number | null)[]; labeled: { start: number; end: number }[] }>("/api/telemetry/preview");
  if (!data) return <div className="flex h-24 items-center text-xs text-ink-soft">Telemetry not downloaded yet.</div>;
  return (
    <>
      <Spark values={data.values} color="#06b6d4" height={92} highlights={data.labeled.map((l) => [l.start / data.step, l.end / data.step])} />
      <div className="mt-1 flex justify-between font-mono text-[10px] text-ink-faint">
        <span>{data.spacecraft} · {data.chan_id} · {data.n_steps.toLocaleString()} steps</span>
        <span className="flex items-center gap-1"><span className="h-2 w-3 rounded-sm bg-emerald-400/50" /> labeled anomaly</span>
      </div>
    </>
  );
}

interface ReplayRow { id: string; status: string; origin: "computed" | "bundled"; summary: { headline?: string; skill?: number | null; observed_min?: number | null; better_hours?: number; worse_hours?: number; horizon?: number } | null; error: string | null }
interface Starter { present: boolean; pack_version?: string; created_at?: string; reference_runs?: string[]; contents?: { id: string; title: string; data_date: string }[];
                    not_in_pack?: { module: string; what: string; how: string }[] }

function FlagshipReplay() {
  const replays = useApi<ReplayRow[]>("/api/spaceweather/replays?event_id=2024-05&limit=5");
  const starter = useApi<Starter>("/api/starter");
  const ok = replays.data?.find((r) => r.status === "succeeded");
  const run = useApi<{ result: { observed: { Dst: (number | null)[] }; forecast: Record<string, (number | null)[]>; issued: string[]; observed_time?: string[] } }>(ok ? `/api/runs/${ok.id}` : null, [ok?.id]);
  const s = ok?.summary;
  return (
    <Link to="/replay?event=2024-05" className="block h-full">
      <motion.div variants={pop} whileHover={{ y: -4 }} className="glass hud relative h-full overflow-hidden p-5">
        <div className="pointer-events-none absolute -right-10 -top-10 h-40 w-40 rounded-full bg-gradient-to-br from-violet-500/30 to-rose-500/20 blur-2xl" />
        <div className="kicker flex items-center gap-1.5"><Clapperboard className="h-3.5 w-3.5 text-violet-500" /> Flagship · Mission Replay</div>
        <div className="mt-1 font-display text-xl font-bold">May 2024 superstorm</div>
        {ok && s ? (
          <>
            <div className="mt-1 flex items-center gap-2 text-xs text-ink-soft"><OriginBadge origin={ok.origin} /> Dst min <b className="text-rose-600">{s.observed_min?.toFixed(0)} nT</b></div>
            {run.data?.result && <div className="mt-2"><Spark values={run.data.result.observed.Dst} color="#ef4444" height={64} /></div>}
            <div className="mt-1 text-[12px]">+{s.horizon} h forecast: model closer <b className="text-emerald-600">{s.better_hours} h</b>, persistence closer <b className="text-rose-600">{s.worse_hours} h</b>
              {s.skill != null && <> · skill <b>{(s.skill * 100).toFixed(0)}%</b></>}</div>
          </>
        ) : (
          <p className="mt-2 text-[12px] text-ink-soft">{replays.data?.some((r) => r.status === "running" || r.status === "queued") ? "Replay running…" :
            "Replay the storm hour by hour - forecasts as issued at the time vs the persistence baseline. Needs the OMNI2 data and the forecaster (release bundle, or Data & Training)."}</p>
        )}
        {starter.data?.present && (
          <div className="mt-3 flex items-center gap-1.5 border-t border-white/60 pt-2 text-[10px] text-ink-faint dark:border-white/10">
            <Package className="h-3 w-3" /> starter pack {starter.data.pack_version} · {starter.data.reference_runs?.length ?? 0} bundled reference runs
            {!!starter.data.not_in_pack?.length && <> · {starter.data.not_in_pack.map((x) => x.module).join(", ")} available from Data &amp; Training</>}
          </div>)}
        <ArrowUpRight className="absolute right-4 top-4 h-4 w-4 text-ink-faint" />
      </motion.div>
    </Link>
  );
}

export default function Overview() {
  const { data, error, loading } = useApi<OverviewT>("/api/overview");
  const catalog = useCatalog();
  const missionSpots = useMemo<Hotspot[]>(() => {
    const ds = catalog.data?.datasets ?? [];
    const st = (id: string) => ds.find((d) => d.id === `mission-${id}`);
    const label = (id: string) => { const e = st(id); return !e ? "" : e.status?.model === "available" ? "model ready" : e.status?.data === "installed" ? "data installed" : "install data"; };
    const leo = ["opssat", "hubble"];
    const inst = leo.filter((id) => st(id)?.status?.data === "installed").length;
    const sat = (id: string, name: string, color: string) => ({ id, name, color, installed: st(id)?.status?.data === "installed", path: `/telemetry?mission=${id}` });
    return [
      { id: "webb", title: "Webb · Sun-Earth L2", sub: label("webb"), path: "/telemetry?mission=webb", color: "#eab308", ready: st("webb")?.status?.data === "installed" },
      { id: "missions", title: "Earth-orbit missions", sub: `OPS-SAT · Hubble — ${inst}/${leo.length} installed`, path: "/telemetry?mission=opssat", color: "#22c55e", ready: inst > 0,
        sats: [sat("opssat", "OPS-SAT", "#22c55e"), sat("hubble", "Hubble", "#38bdf8")] },
    ];
  }, [catalog.data]);
  const spots = useMemo<Hotspot[]>(() => {
    if (!data) return [];
    const by = Object.fromEntries(data.modules.map((m) => [m.module, m])) as Record<string, ModuleOverview | undefined>;
    const sub = (m?: ModuleOverview) => !m ? "" : m.headline ? `${m.headline.metric.split(" (")[0]} ${fmtHead(m)}` : m.model_version ? "model ready" : m.dataset_ready ? "needs model" : "needs data";
    const ready = (m?: ModuleOverview) => !!m?.model_version;
    return [
      { id: "telemetry", title: "SMAP · MSL telemetry", sub: sub(by.telemetry), path: "/telemetry", color: "#06b6d4", ready: ready(by.telemetry) },
      { id: "spaceweather", title: "L1 solar wind", sub: sub(by.spaceweather), path: "/space-weather", color: "#f59e0b", ready: ready(by.spaceweather) || !!data.space_weather },
      { id: "exoplanet", title: "Kepler field", sub: sub(by.exoplanet), path: "/exoplanet", color: "#8b5cf6", ready: ready(by.exoplanet) },
      { id: "mars", title: "Curiosity · Gale crater", sub: sub(by.mars), path: "/mars", color: "#f97316", ready: ready(by.mars) },
      { id: "workbench", title: "Ground segment · your data", sub: `${data.workbench?.datasets ?? 0} datasets`, path: "/workbench", color: "#ec4899", ready: true },
      ...missionSpots,
    ];
  }, [data, missionSpots]);

  if (error) return <ErrorNotice message={error.message} />;
  if (loading && !data) return <div className="flex items-center gap-2 p-8 text-sm text-ink-soft"><Spinner /> Acquiring signal…</div>;
  if (!data) return null;
  const totalRuns = Object.values(data.run_status).reduce((s, n) => s + (n ?? 0), 0);
  const okRate = totalRuns ? (data.run_status.succeeded ?? 0) / totalRuns : null;
  const online = data.modules.filter((m) => m.model_version).length;
  const summary = catalog.data?.summary;
  const running = data.jobs.filter((j) => j.status === "running" || j.status === "queued");
  const wb = data.workbench;
  const reviews = Object.values(wb?.reviews ?? {}).reduce((a, b) => a + (b ?? 0), 0);
  const sw = data.space_weather;

  return (
    <motion.div initial="hidden" animate="show" variants={{ show: { transition: { staggerChildren: 0.05 } } }}
      className="grid auto-rows-min grid-cols-1 gap-4 md:grid-cols-6 xl:grid-cols-12">

      <div className="md:col-span-6 xl:col-span-12"><CatalogSummary /></div>

      {/* hero: inner Solar System mission map */}
      <motion.div variants={pop} className="glass relative overflow-hidden md:col-span-6 xl:col-span-9 xl:row-span-2">
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_50%_55%,rgba(255,255,255,0.7),transparent_65%)] dark:bg-[radial-gradient(ellipse_at_50%_55%,rgba(76,29,149,0.35),transparent_65%)]" />
        <div className="pointer-events-none relative z-10 flex flex-col gap-3 p-6 pb-0 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <div className="kicker flex items-center gap-2"><span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400" /> Live mission map · inner Solar System</div>
            <h1 className="mt-2 text-3xl font-bold leading-tight sm:text-[2.6rem]">
              Every model, <span className="bg-gradient-to-r from-cyan-500 via-violet-500 to-orange-500 bg-clip-text text-transparent">where its data was born.</span>
            </h1>
            <p className="mt-2 max-w-lg text-sm text-ink-soft">Earth, Mars and the spacecraft are placed for today's date (distances schematic). Drag to rotate, scroll to zoom, pick a spacecraft below to fly to it, and click a label to open its data and model.</p>
          </div>
          <div className="pointer-events-auto flex shrink-0 flex-wrap gap-2 sm:flex-col">
            <Link to="/workbench" className="btn-primary"><FlaskConical className="h-4 w-4" /> Bring your data</Link>
            <Link to="/evaluation" className="btn-ghost"><BarChart3 className="h-4 w-4" /> Scoreboard</Link>
            <span className="hidden items-center gap-1 font-mono text-[10px] text-ink-faint sm:flex"><Command className="h-3 w-3" />K · jump anywhere</span>
          </div>
        </div>
        <Suspense fallback={<div className="flex h-[480px] items-center justify-center"><Spinner /></div>}>
          <MissionMap spots={spots} height={480} />
        </Suspense>
      </motion.div>

      {/* flagship: Mission Replay */}
      <div className="md:col-span-3 xl:col-span-3 xl:row-span-1"><FlagshipReplay /></div>

      {/* mission status console */}
      <motion.div variants={pop} className="glass hud overflow-hidden p-5 md:col-span-3 xl:col-span-3">
        <div className="kicker flex items-center gap-1.5"><Radio className="h-3.5 w-3.5" /> Mission status</div>
        <div className="mt-3 rounded-2xl bg-slate-950/[0.04] p-3 font-mono text-[12px] leading-6 dark:bg-black/30">
          <div className="flex justify-between"><span className="text-ink-faint">UTC</span><span className="font-semibold"><Clock /></span></div>
          <div className="flex justify-between"><span className="text-ink-faint">MODELS AVAILABLE</span><span className="font-semibold text-emerald-600 dark:text-emerald-300">{summary ? `${summary.models_available}/${summary.with_models}` : `${online}/${data.modules.length}`}</span></div>
          <div className="flex justify-between"><span className="text-ink-faint">RUNS LOGGED</span><span className="font-semibold">{totalRuns}</span></div>
          <div className="flex justify-between"><span className="text-ink-faint">SUCCESS RATE</span><span className="font-semibold">{okRate == null ? "—" : pct(okRate, 0)}</span></div>
          <div className="flex justify-between"><span className="text-ink-faint">JOBS ACTIVE</span><span className={`font-semibold ${running.length ? "text-violet-600 dark:text-violet-300" : ""}`}>{running.length}</span></div>
          <div className="flex justify-between"><span className="text-ink-faint">REVIEW VERDICTS</span><span className="font-semibold">{reviews}</span></div>
          <div className="flex justify-between"><span className="text-ink-faint">USER DATASETS</span><span className="font-semibold">{wb?.datasets ?? 0}</span></div>
        </div>
        <div className="mt-3 flex flex-wrap gap-1.5">
          {data.modules.map((m) => {
            const Icon = ICONS[m.module];
            return (
              <Link key={m.module} to={MODULE_META[m.module].path} title={MODULE_META[m.module].title}
                className={`grid h-9 w-9 place-items-center rounded-xl transition hover:scale-110 ${m.model_version ? "bg-gradient-to-br text-white shadow " + MODULE_META[m.module].gradient : "bg-white/60 text-ink-faint"}`}>
                <Icon className="h-4 w-4" />
              </Link>);
          })}
        </div>
      </motion.div>

      {/* scoreboard */}
      <motion.div variants={pop} className="md:col-span-3 xl:col-span-3">
        <Tile to="/evaluation" kicker="Held-out scoreboard">
          <div className="mt-1 font-mono text-[9px] text-ink-faint">outer ring model · inner ring baseline</div>
          <div className="mt-2 grid flex-1 grid-cols-2 content-center gap-x-2 gap-y-3">
            {data.modules.map((m) => {
              const meta = MODULE_META[m.module];
              const Icon = ICONS[m.module];
              const h = m.headline;
              return (
                <div key={m.module} className="flex min-w-0 flex-col items-center gap-1 text-center">
                  <DualGauge model={h ? Math.max(0, h.model) : null} baseline={h ? Math.max(0, h.baseline) : null} color={meta.color} size={56}>
                    <Icon className="h-4 w-4" style={{ color: meta.color }} />
                  </DualGauge>
                  <div className="w-full min-w-0 leading-tight">
                    <div className="flex items-center justify-center gap-1 text-[13px] font-bold">{meta.short}{m.model_version ? <CheckCircle2 className="h-3 w-3 text-emerald-500" /> : <CircleDashed className="h-3 w-3 text-amber-500" />}</div>
                    <div className="truncate text-[10px] text-ink-soft">{h ? <><b className="font-mono text-[12px]" style={{ color: meta.color }}>{fmtHead(m)}</b> {h.metric}</> : m.dataset_ready ? "model not trained" : "no data yet"}</div>
                  </div>
                </div>);
            })}
          </div>
        </Tile>
      </motion.div>

      {/* data tiles */}
      <motion.div variants={pop} className="md:col-span-6 xl:col-span-4">
        <Tile to="/telemetry" kicker="Spacecraft telemetry · real test data" icon={Activity} accent="#06b6d4"><div className="mt-3"><TelemetryStrip /></div></Tile>
      </motion.div>
      <motion.div variants={pop} className="md:col-span-3 xl:col-span-3">
        <Tile to="/exoplanet" kicker="A folded transit" icon={Orbit} accent="#8b5cf6">
          {data.kepler_snippet ? (<>
            <div className="mt-3"><Spark values={data.kepler_snippet.local} color="#8b5cf6" height={80} /></div>
            <div className="mt-1 font-mono text-[10px] text-ink-faint">KIC {data.kepler_snippet.kepid} · P {data.kepler_snippet.period_days.toFixed(2)} d · {Math.round(data.kepler_snippet.depth_ppm)} ppm</div>
          </>) : <Empty icon={Orbit} color="#8b5cf6" text="Download Kepler light curves to see a real folded transit." action="Get Kepler data" />}
        </Tile>
      </motion.div>
      <motion.div variants={pop} className="md:col-span-3 xl:col-span-2">
        <Tile to="/mars" kicker="Gale crater terrain" icon={Mountain} accent="#f97316">
          {data.mars_terrain ? (<>
            <div className="mt-4"><StackBar height={14} parts={Object.entries(data.mars_terrain.shares).map(([k, v]) => ({ label: k, value: v, color: `rgb(${TERRAIN_COLORS[k].join(",")})` }))} /></div>
            <div className="mt-3 space-y-1 text-[11px]">
              {Object.entries(data.mars_terrain.shares).map(([k, v]) => (
                <div key={k} className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full" style={{ background: `rgb(${TERRAIN_COLORS[k].join(",")})` }} /><span className="capitalize text-ink-soft">{k}</span><b className="ml-auto font-mono">{pct(v, 0)}</b></div>))}
            </div>
          </>) : <Empty icon={Mountain} color="#f97316" text="Prepare the curated AI4Mars samples." action="Get Mars data" />}
        </Tile>
      </motion.div>
      <motion.div variants={pop} className="md:col-span-6 xl:col-span-3">
        <Tile to="/space-weather" kicker="Geomagnetic storm · Dst" icon={CloudLightning} accent="#f59e0b">
          {sw ? (<>
            <div className="mt-3"><Spark values={sw.dst} color="#ef4444" height={80} /></div>
            <div className="mt-1 flex justify-between font-mono text-[10px] text-ink-faint"><span className="truncate">{sw.event}</span><span className="text-rose-500">{num(sw.min_dst, 0)} nT</span></div>
          </>) : <Empty icon={CloudLightning} color="#f59e0b" text="Download OMNIWeb data to replay storms like Halloween 2003 or May 2024." action="Get space-weather data" />}
        </Tile>
      </motion.div>

      {/* activity + workbench + log */}
      <motion.div variants={pop} className="md:col-span-3 xl:col-span-4">
        <Tile to="/runs" kicker="Run activity · 12 weeks" icon={History}>
          <div className="mt-3 flex items-center gap-4">
            <Donut size={92} parts={[
              { label: "succeeded", value: data.run_status.succeeded ?? 0, color: "#10b981" },
              { label: "failed", value: data.run_status.failed ?? 0, color: "#f43f5e" },
              { label: "running", value: (data.run_status.running ?? 0) + (data.run_status.queued ?? 0), color: "#38bdf8" }]}
              center={<div><div className="font-display text-xl font-bold">{totalRuns}</div><div className="text-[9px] uppercase tracking-wider text-ink-faint">runs</div></div>} />
            <div className="min-w-0 flex-1"><Heatmap activity={data.activity} /></div>
          </div>
        </Tile>
      </motion.div>
      <motion.div variants={pop} className="md:col-span-3 xl:col-span-5">
        <Tile to="/workbench" kicker="Data workbench" icon={FlaskConical} accent="#ec4899">
          <div className="mt-3 grid grid-cols-3 gap-2 text-center">
            {[["series", wb?.by_kind.timeseries ?? 0], ["images", wb?.by_kind.images ?? 0], ["light curves", wb?.by_kind.lightcurve ?? 0]].map(([k, v]) => (
              <div key={k} className="glass-inset py-2"><div className="font-display text-xl font-bold">{v}</div><div className="text-[9px] text-ink-faint">{k}</div></div>))}
          </div>
          <Link to="/review" className="relative z-10 mt-3 flex items-center gap-2 text-[11px] text-ink-soft hover:text-ink">
            <Highlighter className="h-3.5 w-3.5 text-violet-500" /> {wb?.reviews.confirmed ?? 0} confirmed · {wb?.reviews.dismissed ?? 0} dismissed · {wb?.reviews.corrected ?? 0} masks corrected
          </Link>
        </Tile>
      </motion.div>
      <motion.div variants={pop} className="glass hud p-5 md:col-span-6 xl:col-span-12">
        <div className="flex items-center justify-between">
          <div className="kicker flex items-center gap-1.5"><Sparkles className="h-3.5 w-3.5" /> Mission log</div>
          <Link to="/runs" className="text-xs font-semibold text-violet-600 hover:underline dark:text-violet-300">All runs</Link>
        </div>
        {data.recent_runs.length === 0 ? <p className="mt-3 text-sm text-ink-soft">No runs yet. Click a label on the map to start.</p> : (
          <div className="mt-2 space-y-0.5 border-l border-dashed border-violet-300/50 pl-3">
            {data.recent_runs.slice(0, 6).map((r) => (
              <Link key={r.id} to={r.module === "spaceweather" ? `/replay?event=${(r.input_ref as { event_id?: string }).event_id ?? "2024-05"}&run=${r.id}` : `${MODULE_META[r.module].path}?run=${r.id}`} className="relative flex items-center gap-2 rounded-xl px-2 py-1.5 transition hover:bg-white/60">
                <span className="absolute -left-[17px] h-2 w-2 rounded-full" style={{ background: MODULE_META[r.module].color }} />
                <span className="w-14 shrink-0 font-mono text-[10px] text-ink-faint">{r.created_at.slice(11, 16)}Z</span>
                <span className="min-w-0 flex-1 truncate text-[13px] font-semibold sm:w-56 sm:flex-none sm:shrink-0">{r.input_label}</span>
                <span className="hidden flex-1 truncate text-[11px] text-ink-soft sm:block">{r.summary?.headline ?? r.error ?? ""}</span>
                <span className="hidden shrink-0 sm:inline"><OriginBadge origin={r.origin} /></span>
                <span className="shrink-0"><StatusBadge status={r.status} /></span>
              </Link>))}
          </div>)}
        <div className="mt-1 text-right font-mono text-[9px] text-ink-faint">last update {timeAgo(data.recent_runs[0]?.created_at)}</div>
      </motion.div>

      {running.length > 0 && (
        <motion.div variants={pop} className="glass hud p-5 md:col-span-6 xl:col-span-12">
          <div className="kicker">Background jobs</div>
          <div className="mt-3 grid gap-3 md:grid-cols-2 xl:grid-cols-3">{running.slice(0, 3).map((j) => <JobCard key={j.id} job={j} compact />)}</div>
        </motion.div>
      )}
    </motion.div>
  );
}
