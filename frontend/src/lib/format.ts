export const pct = (v: number | null | undefined, digits = 1) =>
  v == null || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(digits)}%`;

export const num = (v: number | string | null | undefined, digits = 3) => {
  if (v == null) return "—";
  if (typeof v === "string") return v;
  if (Number.isNaN(v)) return "—";
  return Number.isInteger(v) ? v.toLocaleString() : v.toFixed(digits);
};

export const timeAgo = (iso: string | null | undefined) => {
  if (!iso) return "—";
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return new Date(iso).toLocaleDateString();
};

export const duration = (ms: number | null | undefined) =>
  ms == null ? "—" : ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`;

export const MODULE_META = {
  missions: {
    title: "Multi-Mission Telemetry",
    short: "Missions",
    path: "/telemetry",
    color: "#4f46e5",
    gradient: "from-indigo-500 to-violet-500",
    soft: "bg-indigo-100 text-indigo-800",
    output: "Flag for review",
  },
  telemetry: {
    title: "Spacecraft Anomaly Detection",
    short: "Telemetry",
    path: "/telemetry",
    color: "#0891b2",
    gradient: "from-cyan-400 to-sky-500",
    soft: "bg-tele-50 text-tele-700",
    output: "Flag for review",
  },
  exoplanet: {
    title: "Exoplanet Finder",
    short: "Exoplanets",
    path: "/exoplanet",
    color: "#7c3aed",
    gradient: "from-violet-500 to-fuchsia-500",
    soft: "bg-exo-50 text-exo-700",
    output: "Candidate score",
  },
  mars: {
    title: "Mars Terrain Segmentation",
    short: "Mars terrain",
    path: "/mars",
    color: "#ea580c",
    gradient: "from-orange-400 to-rose-500",
    soft: "bg-mars-50 text-mars-700",
    output: "Terrain classification",
  },
  spaceweather: {
    title: "Space Weather Explorer",
    short: "Space weather",
    path: "/space-weather",
    color: "#d97706",
    gradient: "from-amber-400 to-rose-500",
    soft: "bg-amber-50 text-amber-700",
    output: "Dst forecast",
  },
  workbench: {
    title: "Data Workbench",
    short: "Workbench",
    path: "/workbench",
    color: "#db2777",
    gradient: "from-pink-500 to-amber-400",
    soft: "bg-pink-50 text-pink-700",
    output: "Analysis result",
  },
} as const;

export const TERRAIN_COLORS: Record<string, [number, number, number]> = {
  soil: [250, 204, 21],
  bedrock: [59, 130, 246],
  sand: [244, 63, 94],
  "big rock": [16, 185, 129],
};
