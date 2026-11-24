import { motion } from "framer-motion";

/** Line with shaded intervals (e.g. labeled anomalies). Values may contain nulls. */
export function Spark({ values, color, highlights = [], height = 90, fill = true }: {
  values: (number | null)[]; color: string; highlights?: [number, number][]; height?: number; fill?: boolean;
}) {
  const v = values.map((x) => (x == null ? NaN : x));
  const finite = v.filter(Number.isFinite);
  if (!finite.length) return <div style={{ height }} />;
  const lo = Math.min(...finite), hi = Math.max(...finite), W = 600, H = height, n = v.length;
  const X = (i: number) => (i / Math.max(n - 1, 1)) * W;
  const Y = (y: number) => H - 6 - ((y - lo) / (hi - lo || 1)) * (H - 12);
  let d = "", pen = false;
  v.forEach((y, i) => {
    if (!Number.isFinite(y)) { pen = false; return; }
    d += `${pen ? "L" : "M"}${X(i).toFixed(1)},${Y(y).toFixed(1)}`;
    pen = true;
  });
  const id = `g${color.replace("#", "")}`;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="w-full" style={{ height }}>
      <defs><linearGradient id={id} x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor={color} stopOpacity="0.28" /><stop offset="1" stopColor={color} stopOpacity="0" /></linearGradient></defs>
      {highlights.map(([a, b], i) => <rect key={i} x={X(a)} width={Math.max(X(b) - X(a), 3)} y={0} height={H} fill="rgba(16,185,129,0.18)" />)}
      {fill && <path d={`${d}L${W},${H}L0,${H}Z`} fill={`url(#${id})`} />}
      <motion.path d={d} fill="none" stroke={color} strokeWidth={1.8} vectorEffect="non-scaling-stroke"
        initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1.4, ease: "easeOut" }} />
    </svg>
  );
}

/** GitHub-style calendar of daily run counts (last `days` days). */
export function Heatmap({ activity, days = 84 }: { activity: { date: string; total: number }[]; days?: number }) {
  const by = new Map(activity.map((a) => [a.date, a.total]));
  const today = new Date();
  const cells = Array.from({ length: days }, (_, i) => {
    const d = new Date(today);
    d.setUTCDate(d.getUTCDate() - (days - 1 - i));
    const key = d.toISOString().slice(0, 10);
    return { key, n: by.get(key) ?? 0 };
  });
  const max = Math.max(1, ...cells.map((c) => c.n));
  const cols = Math.ceil(days / 7);
  const shade = (n: number) => n === 0 ? "rgba(148,163,184,0.18)" : `rgba(124,58,237,${0.25 + 0.75 * (n / max)})`;
  return (
    <div className="grid gap-[3px]" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`, gridAutoFlow: "column", gridTemplateRows: "repeat(7, minmax(0, 1fr))" }}>
      {cells.map((c, i) => (
        <motion.div key={c.key} title={`${c.key}: ${c.n} run${c.n === 1 ? "" : "s"}`} className="aspect-square rounded-[4px]"
          style={{ background: shade(c.n) }} initial={{ opacity: 0, scale: 0.4 }} animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: i * 0.004, type: "spring", stiffness: 300, damping: 20 }} />))}
    </div>
  );
}

/** Donut from named parts. */
export function Donut({ parts, size = 120, center }: { parts: { label: string; value: number; color: string }[]; size?: number; center?: React.ReactNode }) {
  const total = parts.reduce((s, p) => s + p.value, 0);
  const r = 42, c = 2 * Math.PI * r;
  let acc = 0;
  return (
    <div className="relative shrink-0" style={{ width: size, height: size }}>
      <svg viewBox="0 0 100 100" className="h-full w-full -rotate-90">
        <circle cx="50" cy="50" r={r} fill="none" stroke="rgba(148,163,184,0.2)" strokeWidth="11" />
        {total > 0 && parts.map((p) => {
          const len = (p.value / total) * c;
          const el = <motion.circle key={p.label} cx="50" cy="50" r={r} fill="none" stroke={p.color} strokeWidth="11" strokeLinecap="butt"
            strokeDasharray={`${len} ${c - len}`} initial={{ strokeDashoffset: c }} animate={{ strokeDashoffset: -acc }}
            transition={{ duration: 1, ease: "easeOut" }} />;
          acc += len;
          return el;
        })}
      </svg>
      <div className="absolute inset-0 grid place-items-center text-center">{center}</div>
    </div>
  );
}

/** Two concentric arcs: model (outer, coloured) vs baseline (inner, grey), both in [0, 1]. */
export function DualGauge({ model, baseline, color, size = 104, children }: { model: number | null; baseline: number | null; color: string; size?: number; children?: React.ReactNode }) {
  const arc = (r: number, v: number | null, stroke: string, w: number) => {
    const c = 2 * Math.PI * r * 0.75;
    return (
      <>
        <circle cx="50" cy="50" r={r} fill="none" stroke="rgba(148,163,184,0.18)" strokeWidth={w} strokeDasharray={`${c} ${2 * Math.PI * r}`} strokeLinecap="round" />
        {v != null && <motion.circle cx="50" cy="50" r={r} fill="none" stroke={stroke} strokeWidth={w} strokeLinecap="round"
          strokeDasharray={`${c} ${2 * Math.PI * r}`} initial={{ strokeDashoffset: c }} animate={{ strokeDashoffset: c * (1 - Math.min(Math.max(v, 0), 1)) }}
          transition={{ type: "spring", stiffness: 60, damping: 16 }} />}
      </>
    );
  };
  return (
    <div className="relative shrink-0" style={{ width: size, height: size }}>
      <svg viewBox="0 0 100 100" className="h-full w-full rotate-[135deg]">
        {arc(42, model, color, 9)}
        {arc(30, baseline, "#94a3b8", 6)}
      </svg>
      <div className="absolute inset-0 grid place-items-center text-center">{children}</div>
    </div>
  );
}

export function StackBar({ parts, height = 14 }: { parts: { label: string; value: number; color: string }[]; height?: number }) {
  const total = parts.reduce((s, p) => s + p.value, 0) || 1;
  return (
    <div className="flex w-full overflow-hidden rounded-full bg-white/60" style={{ height }}>
      {parts.map((p, i) => (
        <motion.div key={p.label} title={`${p.label} ${(100 * p.value / total).toFixed(1)}%`} style={{ background: p.color }}
          initial={{ width: 0 }} animate={{ width: `${(p.value / total) * 100}%` }} transition={{ delay: 0.1 * i, duration: 0.8, ease: "easeOut" }} />))}
    </div>
  );
}
