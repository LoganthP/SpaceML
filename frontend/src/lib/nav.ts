import { Activity, BarChart3, Clapperboard, CloudLightning, Database, FlaskConical, Highlighter, History, LayoutGrid, Mountain, Orbit, Route, type LucideIcon } from "lucide-react";

export interface NavItem { to: string; label: string; icon: LucideIcon; end?: boolean; grad: string; hint: string }

const I = {
  overview: { to: "/", label: "Overview", icon: LayoutGrid, end: true, grad: "from-slate-500 to-indigo-500", hint: "Status of every mission at a glance" },
  replay: { to: "/replay", label: "Mission Replay", icon: Clapperboard, grad: "from-violet-600 to-rose-500", hint: "Step through a real storm hour by hour" },
  telemetry: { to: "/telemetry", label: "Telemetry & Missions", icon: Activity, grad: "from-cyan-400 to-sky-500", hint: "SMAP/MSL, OPS-SAT, Hubble, Webb" },
  exoplanet: { to: "/exoplanet", label: "Exoplanet Finder", icon: Orbit, grad: "from-violet-500 to-fuchsia-500", hint: "Kepler transit signals" },
  mars: { to: "/mars", label: "Mars Terrain", icon: Mountain, end: true, grad: "from-orange-400 to-rose-500", hint: "Segment Curiosity images" },
  traverse: { to: "/mars/traverse", label: "Rover Traverse", icon: Route, grad: "from-orange-500 to-amber-400", hint: "Curiosity's drive on a map" },
  weather: { to: "/space-weather", label: "Space Weather", icon: CloudLightning, grad: "from-amber-400 to-rose-500", hint: "Dst forecasts from solar wind" },
  workbench: { to: "/workbench", label: "Data Workbench", icon: FlaskConical, grad: "from-pink-500 to-amber-400", hint: "Analyse your own files" },
  review: { to: "/review", label: "Review Studio", icon: Highlighter, grad: "from-violet-500 to-indigo-600", hint: "Confirm or dismiss flagged results" },
  runs: { to: "/runs", label: "Run History", icon: History, grad: "from-slate-400 to-slate-600", hint: "Every saved result" },
  evaluation: { to: "/evaluation", label: "Evaluation", icon: BarChart3, grad: "from-indigo-500 to-violet-600", hint: "Model metrics vs baselines" },
  datasets: { to: "/datasets", label: "Data & Training", icon: Database, grad: "from-emerald-400 to-teal-500", hint: "Install data, train models" },
} satisfies Record<string, NavItem>;

/** Grouped navigation: a mission's path is Data & Training → (mission page) → Analyze → Review Studio. */
export const NAV_GROUPS: { label: string; items: NavItem[] }[] = [
  { label: "Overview", items: [I.overview] },
  { label: "Missions", items: [I.replay, I.telemetry, I.exoplanet, I.mars, I.traverse, I.weather] },
  { label: "Data & Training", items: [I.datasets] },
  { label: "Analyze", items: [I.workbench, I.evaluation, I.runs] },
  { label: "Review Studio", items: [I.review] },
];

export const NAV: NavItem[] = NAV_GROUPS.flatMap((g) => g.items);
/** Labeled shortcuts on the phone bottom bar (everything else is in the Menu sheet). */
export const MOBILE_NAV: NavItem[] = [I.overview, I.telemetry, I.datasets, I.review];
