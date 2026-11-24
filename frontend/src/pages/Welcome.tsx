import { motion, useInView, useReducedMotion } from "framer-motion";
import {
  ArrowRight,
  ArrowUpRight,
  Database,
  ExternalLink,
  Eye,
  FlaskConical,
  Highlighter,
  Layers,
  Radio,
  Sparkles,
  Telescope,
} from "lucide-react";
import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import type { Hotspot } from "../components/MissionMap";
import { Spinner } from "../components/ui";
import { useApi } from "../lib/api";
import { NAV } from "../lib/nav";
import type { CatalogData } from "../components/Catalog";
import type { Overview as OverviewT } from "../lib/types";

const MissionMap = lazy(() => import("../components/MissionMap"));

interface LiveWebb {
  telescope: string;
  name: string;
  observation: {
    id: string;
    target: string;
    title: string;
    program?: string;
    category?: string;
    instruments: string[];
    pi?: string;
    start?: string;
    end?: string;
    phase?: string;
  };
  fetched_at: string;
  source: "live" | "cached";
  notice?: string | null;
}

interface HealthResponse {
  status: string;
  version?: string;
  data_dir?: string;
  models_dir?: string;
}

/** Animated number counter that respects prefers-reduced-motion. */
function CountUp({ end, duration = 1.0 }: { end: number; duration?: number }) {
  const shouldReduceMotion = useReducedMotion();
  const [val, setVal] = useState(shouldReduceMotion ? end : 0);
  const ref = useRef<HTMLSpanElement>(null);
  const isInView = useInView(ref, { once: true, margin: "-20px" });

  useEffect(() => {
    if (shouldReduceMotion) {
      setVal(end);
      return;
    }
    if (!isInView || isNaN(end) || end <= 0) {
      setVal(end);
      return;
    }
    let startTime: number | null = null;
    let animId: number;
    const step = (timestamp: number) => {
      if (!startTime) startTime = timestamp;
      const progress = Math.min((timestamp - startTime) / (duration * 1000), 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setVal(Math.round(eased * end));
      if (progress < 1) {
        animId = requestAnimationFrame(step);
      }
    };
    animId = requestAnimationFrame(step);
    return () => cancelAnimationFrame(animId);
  }, [isInView, end, duration, shouldReduceMotion]);

  return <span ref={ref} className="tabular-nums">{val.toLocaleString()}</span>;
}

export default function Welcome() {
  const shouldReduceMotion = useReducedMotion();

  // 1. Live status data
  const catalog = useApi<CatalogData>("/api/catalog");
  const overview = useApi<OverviewT>("/api/overview");
  const liveWebb = useApi<LiveWebb>("/api/live/webb");
  const health = useApi<HealthResponse>("/api/health");

  // Mission Map Hotspots
  const spots = useMemo<Hotspot[]>(() => [
    { id: "telemetry", title: "SMAP · MSL telemetry", sub: "Real spacecraft test data", path: "/telemetry", color: "#06b6d4", ready: true },
    { id: "spaceweather", title: "L1 solar wind", sub: "Dst geomagnetic storm forecast", path: "/space-weather", color: "#f59e0b", ready: true },
    { id: "exoplanet", title: "Kepler field", sub: "Transit signal discovery", path: "/exoplanet", color: "#8b5cf6", ready: true },
    { id: "mars", title: "Curiosity · Gale crater", sub: "Terrain segmentation", path: "/mars", color: "#f97316", ready: true },
    { id: "workbench", title: "Ground segment · your data", sub: "Analyze custom datasets", path: "/workbench", color: "#ec4899", ready: true },
    { id: "webb", title: "Webb · Sun-Earth L2", sub: "Space telescope live feed", path: "/telemetry?mission=webb", color: "#eab308", ready: true },
    {
      id: "missions",
      title: "Earth-orbit missions",
      sub: "OPS-SAT · Hubble",
      path: "/telemetry?mission=opssat",
      color: "#22c55e",
      ready: true,
      sats: [
        { id: "opssat", name: "OPS-SAT", color: "#22c55e", installed: true, path: "/telemetry?mission=opssat" },
        { id: "hubble", name: "Hubble", color: "#38bdf8", installed: true, path: "/telemetry?mission=hubble" },
      ],
    },
  ], []);

  // Compute stats for strip
  const catalogSummary = catalog.data?.summary;
  const overviewData = overview.data;

  const totalRuns = useMemo(() => {
    if (!overviewData?.run_status) return null;
    return Object.values(overviewData.run_status).reduce((sum, n) => sum + (n ?? 0), 0);
  }, [overviewData]);

  const modelsAvailable = useMemo(() => {
    if (catalogSummary?.models_available !== undefined) return catalogSummary.models_available;
    if (overviewData?.modules) return overviewData.modules.filter((m) => !!m.model_version).length;
    return null;
  }, [catalogSummary, overviewData]);

  const datasetsInstalled = catalogSummary?.installed ?? null;
  const readyToTrain = catalogSummary?.readiness?.ready_to_train ?? null;

  const isStatusLoading = (catalog.loading && !catalog.data) || (overview.loading && !overview.data);
  const isStatusOffline = Boolean(catalog.error && overview.error);

  // Module cards mapped from nav.ts
  const moduleRoutes = useMemo(() => [
    "/telemetry",
    "/exoplanet",
    "/mars",
    "/mars/traverse",
    "/replay",
    "/workbench",
    "/review",
    "/evaluation",
  ], []);

  const moduleCards = useMemo(() => {
    return moduleRoutes.map((route) => {
      const item = NAV.find((n) => n.to === route);
      if (!item) {
        return {
          to: route,
          label: route.replace("/", ""),
          hint: "",
          icon: Layers,
          grad: "from-slate-500 to-indigo-500",
        };
      }
      return item;
    });
  }, [moduleRoutes]);

  // Motion variants
  const containerVariants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: shouldReduceMotion ? { duration: 0.1 } : { staggerChildren: 0.08, delayChildren: 0.1 },
    },
  };

  const itemFadeUp = {
    hidden: shouldReduceMotion ? { opacity: 1, y: 0 } : { opacity: 0, y: 16 },
    show: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.35, ease: "easeOut" },
    },
  };

  const sectionReveal = {
    hidden: shouldReduceMotion ? { opacity: 1, y: 0 } : { opacity: 0, y: 24 },
    show: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.45, ease: "easeOut" },
    },
  };

  return (
    <div className="space-y-12 pb-16">
      {/* 1. HERO SECTION */}
      <motion.section
        variants={containerVariants}
        initial="hidden"
        animate="show"
        aria-label="Welcome Hero"
        className="relative pt-2"
      >
        <div className="flex flex-col gap-6 lg:gap-8">
          {/* Hero text & CTAs */}
          <motion.div variants={itemFadeUp} className="max-w-3xl space-y-4">
            <div className="kicker inline-flex items-center gap-2 rounded-full border border-violet-200/60 bg-violet-50/70 px-3 py-1 text-xs dark:border-violet-500/20 dark:bg-violet-950/40">
              <Sparkles className="h-3.5 w-3.5 text-violet-500" aria-hidden="true" />
              <span>Space ML Mission Control</span>
            </div>

            <h1 className="text-3xl font-extrabold tracking-tight sm:text-5xl md:text-6xl">
              Every model,{" "}
              <span className="welcome-gradient-text bg-gradient-to-r from-cyan-500 via-violet-500 to-orange-500 bg-clip-text text-transparent">
                where its data was born.
              </span>
            </h1>

            <p className="max-w-2xl text-base text-ink-soft sm:text-lg sm:leading-relaxed">
              Real NASA telemetry, exoplanet hunting, Mars terrain segmentation, and geomagnetic storm forecasts in one unified mission control.
            </p>

            <div className="flex flex-wrap items-center gap-3 pt-2">
              <Link to="/" className="btn-primary">
                Open Mission Control <ArrowRight className="h-4 w-4" aria-hidden="true" />
              </Link>
              <Link to="/workbench" className="btn-ghost">
                <FlaskConical className="h-4 w-4 text-pink-500" aria-hidden="true" /> Bring your data
              </Link>
            </div>
          </motion.div>

          {/* Hero Visual: MissionMap on >= 640px, static gradient orb on < 640px */}
          <motion.div variants={itemFadeUp} className="w-full">
            {/* Desktop / Tablet visual container */}
            <div className="hidden sm:block">
              <div className="glass hud relative overflow-hidden rounded-3xl p-1">
                <div className="pointer-events-none absolute inset-0 z-10 bg-[radial-gradient(ellipse_at_50%_40%,rgba(255,255,255,0.4),transparent_65%)] dark:bg-[radial-gradient(ellipse_at_50%_40%,rgba(76,29,149,0.25),transparent_65%)]" />
                <Suspense
                  fallback={
                    <div className="flex h-[420px] flex-col items-center justify-center gap-3 text-ink-soft">
                      <Spinner className="h-8 w-8 text-violet-500" />
                      <span className="font-mono text-xs">Loading mission orbit map…</span>
                    </div>
                  }
                >
                  <MissionMap spots={spots} height={420} />
                </Suspense>
              </div>
            </div>

            {/* Mobile visual (< 640px): Lightweight static gradient orb */}
            <div className="glass relative flex h-64 w-full flex-col items-center justify-center overflow-hidden p-6 text-center sm:hidden">
              <div className="absolute -inset-4 bg-gradient-to-tr from-cyan-500/20 via-violet-500/25 to-orange-500/20 blur-2xl" />
              <div className="relative flex h-28 w-28 items-center justify-center rounded-full bg-gradient-to-br from-cyan-400 via-violet-600 to-orange-400 shadow-[0_0_35px_rgba(168,85,247,0.5)]">
                <div className="flex h-24 w-24 items-center justify-center rounded-full bg-slate-950/80 backdrop-blur-md">
                  <span className="font-display text-sm font-bold tracking-wider text-white">SMC</span>
                </div>
              </div>
              <div className="relative mt-4">
                <span className="kicker text-[11px]">Inner Solar System</span>
                <p className="mt-1 text-xs text-ink-soft">Interactive 3D mission map available on larger screens.</p>
              </div>
            </div>
          </motion.div>
        </div>
      </motion.section>

      {/* 2. LIVE STATUS STRIP */}
      <motion.section
        variants={sectionReveal}
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, margin: "-40px" }}
        aria-label="System Live Status"
        className="glass hud overflow-hidden p-5 sm:p-6"
      >
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/60 pb-3 dark:border-white/10">
          <div className="kicker flex items-center gap-2">
            <Radio className="h-3.5 w-3.5 text-emerald-500" aria-hidden="true" />
            <span>Mission telemetry & readiness status</span>
          </div>
          <span className="font-mono text-[11px] text-ink-faint">GET /api/catalog · GET /api/overview</span>
        </div>

        {isStatusOffline ? (
          <div className="py-6 text-center text-sm text-ink-soft">
            Offline - start the app to see live status
          </div>
        ) : (
          <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4 lg:grid-cols-4">
            {/* Stat 1: Models Available */}
            <div className="glass-inset p-3.5 sm:p-4">
              <div className="label">Models Available</div>
              <div className="mt-1 font-display text-2xl font-bold text-emerald-600 dark:text-emerald-400 sm:text-3xl">
                {isStatusLoading ? (
                  <div className="h-8 w-16 animate-pulse rounded-lg bg-slate-200 dark:bg-white/10" />
                ) : (
                  <CountUp end={modelsAvailable ?? 0} />
                )}
              </div>
              <div className="mt-1 text-xs text-ink-soft">Ready for immediate inference</div>
            </div>

            {/* Stat 2: Runs Logged */}
            <div className="glass-inset p-3.5 sm:p-4">
              <div className="label">Runs Logged</div>
              <div className="mt-1 font-display text-2xl font-bold text-violet-600 dark:text-violet-400 sm:text-3xl">
                {isStatusLoading ? (
                  <div className="h-8 w-16 animate-pulse rounded-lg bg-slate-200 dark:bg-white/10" />
                ) : (
                  <CountUp end={totalRuns ?? 0} />
                )}
              </div>
              <div className="mt-1 text-xs text-ink-soft">Reproducible run records</div>
            </div>

            {/* Stat 3: Datasets Installed */}
            <div className="glass-inset p-3.5 sm:p-4">
              <div className="label">Datasets Installed</div>
              <div className="mt-1 font-display text-2xl font-bold text-cyan-600 dark:text-cyan-400 sm:text-3xl">
                {isStatusLoading ? (
                  <div className="h-8 w-16 animate-pulse rounded-lg bg-slate-200 dark:bg-white/10" />
                ) : (
                  <CountUp end={datasetsInstalled ?? 0} />
                )}
              </div>
              <div className="mt-1 text-xs text-ink-soft">Verified space telemetry sets</div>
            </div>

            {/* Stat 4: Ready to Train */}
            <div className="glass-inset p-3.5 sm:p-4">
              <div className="label">Ready to Train</div>
              <div className="mt-1 font-display text-2xl font-bold text-amber-600 dark:text-amber-400 sm:text-3xl">
                {isStatusLoading ? (
                  <div className="h-8 w-16 animate-pulse rounded-lg bg-slate-200 dark:bg-white/10" />
                ) : (
                  <CountUp end={readyToTrain ?? 0} />
                )}
              </div>
              <div className="mt-1 text-xs text-ink-soft">Pipelines with prepped data</div>
            </div>
          </div>
        )}
      </motion.section>

      {/* 3. MODULES GRID */}
      <motion.section
        variants={sectionReveal}
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, margin: "-40px" }}
        aria-label="Mission Control Modules"
        className="space-y-4"
      >
        <div className="flex flex-wrap items-end justify-between gap-2">
          <div>
            <div className="kicker">Core System Architecture</div>
            <h2 className="mt-1 text-2xl font-bold text-ink sm:text-3xl">Specialized Mission Domains</h2>
          </div>
          <p className="max-w-md text-xs text-ink-soft">
            Explore dedicated ML pipelines spanning orbital telemetry, astrophysics, planetary science, and real-time operations.
          </p>
        </div>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
          {moduleCards.map((card) => {
            const Icon = card.icon;
            return (
              <motion.div
                key={card.to}
                whileHover={shouldReduceMotion ? undefined : { y: -4 }}
                transition={{ type: "spring", stiffness: 300, damping: 20 }}
                className="h-full"
              >
                <Link
                  to={card.to}
                  className="glass hud group relative flex h-full flex-col justify-between overflow-hidden p-5 transition focus-visible:ring-2 focus-visible:ring-violet-500"
                >
                  <div>
                    <div className="flex items-start justify-between gap-2">
                      <span
                        className={`grid h-12 w-12 shrink-0 place-items-center rounded-2xl bg-gradient-to-br ${card.grad} text-white shadow-md transition-all duration-300 group-hover:scale-105 group-hover:shadow-[0_0_24px_rgba(168,85,247,0.55)] group-focus-visible:shadow-[0_0_24px_rgba(168,85,247,0.55)]`}
                        aria-hidden="true"
                      >
                        <Icon className="h-6 w-6" />
                      </span>
                      <span className="grid h-8 w-8 place-items-center rounded-xl text-ink-faint transition group-hover:text-ink">
                        <ArrowUpRight className="h-4 w-4" aria-hidden="true" />
                      </span>
                    </div>

                    <h3 className="mt-4 text-base font-bold text-ink transition group-hover:text-violet-600 dark:group-hover:text-violet-300">
                      {card.label}
                    </h3>
                    <p className="mt-1 text-xs leading-relaxed text-ink-soft">
                      {card.hint}
                    </p>
                  </div>

                  <div className="mt-4 flex items-center gap-1 text-[11px] font-semibold text-violet-600 dark:text-violet-400">
                    <span>Open {card.label}</span>
                    <span aria-hidden="true" className="transition group-hover:translate-x-0.5">→</span>
                  </div>
                </Link>
              </motion.div>
            );
          })}
        </div>
      </motion.section>

      {/* 4. WEBB TEASER */}
      <motion.section
        variants={sectionReveal}
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, margin: "-40px" }}
        aria-label="Webb Live Observation Teaser"
        className="glass hud overflow-hidden p-6 sm:p-7"
      >
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/60 pb-3 dark:border-white/10">
          <div className="kicker flex items-center gap-2">
            <Telescope className="h-4 w-4 text-amber-500" aria-hidden="true" />
            <span>Deep Space Observing · James Webb Space Telescope</span>
          </div>
          <div className="flex items-center gap-2">
            {liveWebb.data ? (
              liveWebb.data.source === "live" ? (
                <span className="chip bg-emerald-100 text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300">
                  <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-500" aria-hidden="true" />
                  Live stream
                </span>
              ) : (
                <span className="chip bg-amber-100 text-amber-900 dark:bg-amber-950/60 dark:text-amber-200">
                  Offline · last known
                </span>
              )
            ) : null}
          </div>
        </div>

        {liveWebb.error ? (
          <div className="py-4 text-xs text-ink-soft">
            Live observing data is not available right now. Connect to Space Telescope Live to monitor current deep space targets.
          </div>
        ) : liveWebb.loading && !liveWebb.data ? (
          <div className="flex items-center gap-2 py-6 text-xs text-ink-soft">
            <Spinner className="h-4 w-4 text-amber-500" />
            <span>Acquiring Webb target telemetry…</span>
          </div>
        ) : liveWebb.data ? (
          <div className="mt-4 flex flex-col justify-between gap-5 sm:flex-row sm:items-center">
            <div className="space-y-1.5">
              <div className="label">Target in sight</div>
              <h2 className="text-xl font-extrabold text-ink sm:text-2xl">
                {liveWebb.data.observation.target}
              </h2>
              <p className="max-w-xl text-xs text-ink-soft">
                {liveWebb.data.observation.title}
                {liveWebb.data.observation.category ? ` · ${liveWebb.data.observation.category}` : ""}
              </p>

              {/* Instrument chips */}
              <div className="flex flex-wrap items-center gap-1.5 pt-1">
                {liveWebb.data.observation.instruments.map((inst) => (
                  <span
                    key={inst}
                    className="chip border border-white/70 bg-white/60 font-mono text-[11px] text-ink dark:border-white/10 dark:bg-white/10"
                  >
                    {inst}
                  </span>
                ))}
              </div>
            </div>

            <div className="shrink-0">
              <Link to="/telemetry?mission=webb" className="btn-primary">
                <Eye className="h-4 w-4" aria-hidden="true" />
                <span>Webb Mission Feed</span>
              </Link>
            </div>
          </div>
        ) : null}
      </motion.section>

      {/* 5. HOW IT WORKS (3 STEPS) */}
      <motion.section
        variants={sectionReveal}
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, margin: "-40px" }}
        aria-label="How it works workflow"
        className="space-y-4"
      >
        <div>
          <div className="kicker">Operational Workflow</div>
          <h2 className="mt-1 text-2xl font-bold text-ink sm:text-3xl">How It Works</h2>
        </div>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          {/* Step 1 */}
          <Link
            to="/datasets"
            className="glass hud group relative flex flex-col justify-between p-6 transition focus-visible:ring-2 focus-visible:ring-violet-500"
          >
            <div>
              <div className="flex items-center justify-between">
                <span className="font-mono text-sm font-bold text-violet-600 dark:text-violet-400">01</span>
                <span className="grid h-10 w-10 place-items-center rounded-xl bg-violet-100 text-violet-700 dark:bg-violet-900/40 dark:text-violet-300">
                  <Database className="h-5 w-5" aria-hidden="true" />
                </span>
              </div>
              <h3 className="mt-4 text-lg font-bold text-ink">Install Data</h3>
              <p className="mt-2 text-xs leading-relaxed text-ink-soft">
                Acquire curated flight telemetry, Kepler flux curves, Curiosity Martian imagery, and space weather solar wind indices with single-click downloads.
              </p>
            </div>
            <div className="mt-6 flex items-center gap-1.5 text-xs font-semibold text-violet-600 dark:text-violet-400">
              <span>Data &amp; Training</span>
              <span aria-hidden="true" className="transition group-hover:translate-x-1">→</span>
            </div>
          </Link>

          {/* Step 2 */}
          <Link
            to="/telemetry"
            className="glass hud group relative flex flex-col justify-between p-6 transition focus-visible:ring-2 focus-visible:ring-violet-500"
          >
            <div>
              <div className="flex items-center justify-between">
                <span className="font-mono text-sm font-bold text-cyan-600 dark:text-cyan-400">02</span>
                <span className="grid h-10 w-10 place-items-center rounded-xl bg-cyan-100 text-cyan-700 dark:bg-cyan-900/40 dark:text-cyan-300">
                  <FlaskConical className="h-5 w-5" aria-hidden="true" />
                </span>
              </div>
              <h3 className="mt-4 text-lg font-bold text-ink">Train &amp; Analyse</h3>
              <p className="mt-2 text-xs leading-relaxed text-ink-soft">
                Run telemetry anomaly detection, train 1D-CNN transit classifiers, segment rover terrain, or simulate solar storms with real-time inference.
              </p>
            </div>
            <div className="mt-6 flex items-center gap-1.5 text-xs font-semibold text-cyan-600 dark:text-cyan-400">
              <span>Telemetry Hub</span>
              <span aria-hidden="true" className="transition group-hover:translate-x-1">→</span>
            </div>
          </Link>

          {/* Step 3 */}
          <Link
            to="/review"
            className="glass hud group relative flex flex-col justify-between p-6 transition focus-visible:ring-2 focus-visible:ring-violet-500"
          >
            <div>
              <div className="flex items-center justify-between">
                <span className="font-mono text-sm font-bold text-rose-600 dark:text-rose-400">03</span>
                <span className="grid h-10 w-10 place-items-center rounded-xl bg-rose-100 text-rose-700 dark:bg-rose-900/40 dark:text-rose-300">
                  <Highlighter className="h-5 w-5" aria-hidden="true" />
                </span>
              </div>
              <h3 className="mt-4 text-lg font-bold text-ink">Review &amp; Validate</h3>
              <p className="mt-2 text-xs leading-relaxed text-ink-soft">
                Inspect flagged anomalies in Review Studio, confirm or dismiss predictions, correct masks, and export cryptographic run evidence.
              </p>
            </div>
            <div className="mt-6 flex items-center gap-1.5 text-xs font-semibold text-rose-600 dark:text-rose-400">
              <span>Review Studio</span>
              <span aria-hidden="true" className="transition group-hover:translate-x-1">→</span>
            </div>
          </Link>
        </div>
      </motion.section>

      {/* 6. FOOTER */}
      <footer className="glass hud mt-12 p-6 sm:p-8" aria-label="Page footer">
        <div className="flex flex-col gap-6 md:flex-row md:items-start md:justify-between">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <div className="relative h-6 w-6">
                <div className="absolute inset-0 rounded-full bg-gradient-to-br from-cyan-400 via-violet-500 to-orange-400 shadow-sm" />
              </div>
              <span className="font-display text-sm font-bold tracking-tight">Space ML Mission Control</span>
              {health.data?.version && (
                <span className="chip border border-white/60 bg-white/70 font-mono text-[10px] text-ink-soft dark:border-white/10 dark:bg-white/10">
                  v{health.data.version}
                </span>
              )}
            </div>
            <p className="max-w-md text-xs leading-relaxed text-ink-soft">
              Real spaceflight telemetry, machine learning pipelines, and science workbench for deep space exploration missions.
            </p>
            <div className="text-[11px] text-ink-faint">
              Data credits: NASA/JPL (SMAP, MSL Curiosity, Kepler), ESA (OPS-SAT), and STScI (Hubble, JWST).
            </div>
          </div>

          <div className="flex flex-wrap gap-6 text-xs sm:gap-10">
            <div>
              <div className="label">Documentation</div>
              <ul className="mt-2 space-y-1.5 font-medium text-ink-soft">
                <li>
                  <a
                    href="https://github.com"
                    target="_blank"
                    rel="noreferrer"
                    className="flex items-center gap-1 hover:text-ink"
                  >
                    <span>Quickstart Guide</span>
                    <ExternalLink className="h-3 w-3" aria-hidden="true" />
                  </a>
                </li>
                <li>
                  <Link to="/datasets" className="hover:text-ink">Datasets Catalog</Link>
                </li>
                <li>
                  <Link to="/evaluation" className="hover:text-ink">Model Benchmarks</Link>
                </li>
              </ul>
            </div>

            <div>
              <div className="label">Navigation</div>
              <ul className="mt-2 space-y-1.5 font-medium text-ink-soft">
                <li>
                  <Link to="/" className="hover:text-ink">Mission Map Overview</Link>
                </li>
                <li>
                  <Link to="/workbench" className="hover:text-ink">Data Workbench</Link>
                </li>
                <li>
                  <Link to="/runs" className="hover:text-ink">Run History</Link>
                </li>
              </ul>
            </div>
          </div>
        </div>

        <div className="mt-6 border-t border-white/60 pt-4 text-center font-mono text-[10px] text-ink-faint dark:border-white/10 sm:text-left">
          Space ML Mission Control · Open Source Space Exploration
        </div>
      </footer>
    </div>
  );
}
