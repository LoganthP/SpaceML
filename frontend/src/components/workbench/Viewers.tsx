import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import type { GlobeMarker } from "../ContextGlobe";

const ContextGlobe = lazy(() => import("../ContextGlobe"));
import Plot from "../LazyPlot";
import { ReviewToggle, useReviews } from "../RunTools";
import { Glass, Kpi, Notice, Segmented } from "../ui";
import { num, pct } from "../../lib/format";
import type { ChangeResult, TimelineResult } from "../../lib/types";

const CH_COLORS = ["#0891b2", "#7c3aed", "#ea580c", "#16a34a", "#db2777", "#2563eb", "#ca8a04", "#0f766e", "#9333ea", "#dc2626", "#4f46e5", "#65a30d"];

/* ================================================================ time series */
export function TimelineViewer({ r, runId }: { r: TimelineResult; runId: string }) {
  const [which, setWhich] = useState<"robust" | "zscore">("robust");
  const [showBands, setShowBands] = useState(true);
  const { reviews, set } = useReviews(runId);
  const det = r[which];
  const other = which === "robust" ? r.zscore : r.robust;
  const xIsDate = r.x_kind === "datetime";

  const shapes: Partial<Plotly.Shape>[] = [
    ...r.labeled.map((l) => ({ type: "rect", xref: "x", yref: "paper", x0: l.x0, x1: l.x1, y0: 0, y1: 1, fillcolor: "rgba(16,185,129,0.15)", line: { width: 0 }, layer: "below" }) as Partial<Plotly.Shape>),
    ...det.intervals.map((f) => ({ type: "rect", xref: "x", yref: "paper", x0: f.x0, x1: f.x1, y0: 0.955, y1: 1, fillcolor: "rgba(219,39,119,0.9)", line: { width: 0 } }) as Partial<Plotly.Shape>),
    ...det.intervals.map((f) => ({ type: "rect", xref: "x", yref: "paper", x0: f.x0, x1: f.x1, y0: 0, y1: 0.95, fillcolor: "rgba(219,39,119,0.08)", line: { width: 0 }, layer: "below" }) as Partial<Plotly.Shape>),
    ...other.intervals.map((f) => ({ type: "rect", xref: "x", yref: "paper", x0: f.x0, x1: f.x1, y0: 0.915, y1: 0.945, fillcolor: "rgba(245,158,11,0.85)", line: { width: 0 } }) as Partial<Plotly.Shape>),
    { type: "line", xref: "x", yref: "paper", x0: r.train_end, x1: r.train_end, y0: 0, y1: 1, line: { color: "rgba(100,116,139,0.6)", dash: "dot", width: 1 } },
  ];
  const data: Plotly.Data[] = r.channels.flatMap((c, i) => {
    const col = CH_COLORS[i % CH_COLORS.length];
    const tr: Plotly.Data[] = [{ x: r.x, y: c.values, type: "scattergl", mode: "lines", name: c.unit ? `${c.name} [${c.unit}]` : c.name, line: { color: col, width: 1.2 } }];
    if (showBands && r.channels.length <= 3) {
      tr.push({ x: [r.x[0], r.x[r.x.length - 1]], y: [c.upper, c.upper], type: "scatter", mode: "lines", showlegend: false, hoverinfo: "skip", line: { color: col, dash: "dot", width: 0.8 } });
      tr.push({ x: [r.x[0], r.x[r.x.length - 1]], y: [c.lower, c.lower], type: "scatter", mode: "lines", showlegend: false, hoverinfo: "skip", line: { color: col, dash: "dot", width: 0.8 } });
    }
    return tr;
  });
  data.push(
    { x: [null], y: [null], type: "scatter", mode: "markers", name: "labeled", marker: { color: "rgba(16,185,129,0.5)", symbol: "square", size: 11 } },
    { x: [null], y: [null], type: "scatter", mode: "markers", name: `${which === "robust" ? "robust" : "z-score"} flags`, marker: { color: "#db2777", symbol: "square", size: 11 } },
    { x: [null], y: [null], type: "scatter", mode: "markers", name: `${which === "robust" ? "z-score" : "robust"} flags`, marker: { color: "#f59e0b", symbol: "square", size: 11 } },
  );
  const m = r.metrics;
  const reviewed = det.intervals.filter((_, i) => reviews[`${which}:${i}`]).length;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Kpi label="Robust detector flags" value={r.robust.intervals.length} accent="#db2777" sub={m ? `F1 ${pct(m.robust.f1)} · ${m.robust.fp} false alarms` : "no labels given"} />
        <Kpi label="z-score baseline flags" value={r.zscore.intervals.length} accent="#d97706" sub={m ? `F1 ${pct(m.zscore.f1)} · ${m.zscore.fp} false alarms` : "no labels given"} />
        <Kpi label="Labeled events" value={m ? m.n_labeled : "—"} sub={m ? `found: robust ${m.robust.tp}, z-score ${m.zscore.tp}` : "add a 0/1 column or intervals"} />
        <Kpi label="Rows analysed" value={r.n_rows.toLocaleString()} sub={`${r.channels.length} channel(s) · ${r.file}`} />
      </div>
      <Glass className="p-4">
        <div className="mb-1 flex flex-wrap items-center justify-between gap-2 px-1">
          <Segmented value={which} onChange={setWhich} options={[{ value: "robust", label: "Rolling robust" }, { value: "zscore", label: "z-score baseline" }]} />
          <label className="flex cursor-pointer items-center gap-2 text-xs text-ink-soft">
            <input type="checkbox" checked={showBands} onChange={(e) => setShowBands(e.target.checked)} className="accent-pink-500" /> z-score bands
          </label>
        </div>
        <Plot height={440} data={data} layout={{
          shapes, hovermode: "x unified", margin: { l: 56, r: 18, t: 36, b: 48 },
          xaxis: { title: { text: r.x_label }, type: xIsDate ? "date" : "linear" },
          yaxis: { title: { text: r.channels.length === 1 && r.channels[0].unit ? `${r.channels[0].name} [${r.channels[0].unit}]` : "value" } },
        } as Partial<Plotly.Layout>} />
        <p className="px-1 text-[11px] text-ink-faint">{det.name}. Dotted vertical line = end of the reference window used for z-score statistics. Drag to zoom, double-click to reset.</p>
      </Glass>
      {m && (
        <Glass className="p-5">
          <div className="label">Experiment · same data, same split, two detectors</div>
          <table className="mt-3 w-full text-sm">
            <thead><tr className="text-left text-xs text-ink-faint"><th className="pb-2">Event-level</th><th>Rolling robust</th><th>z-score baseline</th></tr></thead>
            <tbody>
              {(["precision", "recall", "f1"] as const).map((k) => (
                <tr key={k} className="border-t border-white/70"><td className="py-2 capitalize">{k}</td>
                  <td className={`font-semibold ${m.robust[k] >= m.zscore[k] ? "text-emerald-600" : ""}`}>{pct(m.robust[k])}</td>
                  <td className={`font-semibold ${m.zscore[k] > m.robust[k] ? "text-emerald-600" : ""}`}>{pct(m.zscore[k])}</td></tr>))}
              <tr className="border-t border-white/70"><td className="py-2">False alarms</td><td>{m.robust.fp}</td><td>{m.zscore.fp}</td></tr>
            </tbody>
          </table>
        </Glass>
      )}
      <Glass className="p-5">
        <div className="flex items-center justify-between">
          <div className="label">Review flagged intervals · {which === "robust" ? "robust" : "z-score"}</div>
          <span className="text-[11px] text-ink-soft">{reviewed}/{det.intervals.length} reviewed · verdicts go into the run report</span>
        </div>
        {det.intervals.length === 0 ? <p className="mt-3 text-sm text-ink-soft">Nothing exceeded the threshold. Lower it in the settings to explore.</p> : (
          <div className="scroll-thin mt-3 max-h-80 overflow-y-auto">
            <table className="w-full text-sm">
              <thead><tr className="text-left text-xs text-ink-faint"><th className="pb-2">From</th><th>To</th><th>Steps</th><th>Matches a label</th><th>Verdict</th></tr></thead>
              <tbody>
                {det.intervals.map((f, i) => {
                  const hit = r.labeled.some((l) => f.start <= l.end && l.start <= f.end);
                  return (
                    <tr key={i} className="border-t border-white/70">
                      <td className="py-1.5 font-mono text-xs">{String(f.x0).replace("T", " ").replace("Z", "")}</td>
                      <td className="font-mono text-xs">{String(f.x1).replace("T", " ").replace("Z", "")}</td>
                      <td>{f.length}</td>
                      <td>{r.labeled.length === 0 ? <span className="text-ink-faint">—</span> : hit ? <span className="chip bg-emerald-100 text-emerald-700">yes</span> : <span className="chip bg-amber-100 text-amber-800">no</span>}</td>
                      <td><ReviewToggle item={`${which}:${i}`} review={reviews[`${which}:${i}`]} onSet={set} /></td>
                    </tr>);
                })}
              </tbody>
            </table>
          </div>)}
      </Glass>
    </div>
  );
}

/* ================================================================ change detection */
const CHANGE_COLORS: Record<string, [number, number, number]> = { "1": [236, 72, 153], "2": [249, 115, 22], "3": [56, 189, 248] };

function useMaskOverlay(url: string, enabled: Record<string, boolean>, alpha: number) {
  const [data, setData] = useState<string | null>(null);
  const [ids, setIds] = useState<{ d: Uint8ClampedArray; w: number; h: number } | null>(null);
  useEffect(() => {
    const img = new Image();
    img.onload = () => {
      const c = document.createElement("canvas");
      c.width = img.naturalWidth; c.height = img.naturalHeight;
      const ctx = c.getContext("2d", { willReadFrequently: true })!;
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(img, 0, 0);
      setIds({ d: ctx.getImageData(0, 0, c.width, c.height).data, w: c.width, h: c.height });
    };
    img.src = url;
  }, [url]);
  useEffect(() => {
    if (!ids) return;
    const c = document.createElement("canvas");
    c.width = ids.w; c.height = ids.h;
    const ctx = c.getContext("2d")!;
    const out = ctx.createImageData(ids.w, ids.h);
    const a = Math.round(alpha * 255);
    for (let i = 0; i < ids.d.length; i += 4) {
      const k = String(ids.d[i]);
      const col = enabled[k] ? CHANGE_COLORS[k] : undefined;
      if (col) { out.data[i] = col[0]; out.data[i + 1] = col[1]; out.data[i + 2] = col[2]; out.data[i + 3] = a; }
    }
    ctx.putImageData(out, 0, 0);
    setData(c.toDataURL());
  }, [ids, enabled, alpha]);
  return data;
}

function Footprint({ r }: { r: ChangeResult }) {
  const fp = r.footprint_lonlat!;
  const markers = useMemo<GlobeMarker[]>(() => [{ lat: (fp[1] + fp[3]) / 2, lon: (fp[0] + fp[2]) / 2, primary: true, color: "#ec4899",
    label: `${r.before.file} · ${((fp[1] + fp[3]) / 2).toFixed(2)}°, ${((fp[0] + fp[2]) / 2).toFixed(2)}°` }], [fp, r.before.file]);
  return (
    <Glass className="hud p-4">
      <div className="kicker mb-2">Scene location on Earth</div>
      <Suspense fallback={<div className="h-[340px]" />}>
        <ContextGlobe body="earth" markers={markers} footprint={fp} height={340}
          caption={`Footprint ${fp[1].toFixed(3)}°…${fp[3].toFixed(3)}° lat, ${fp[0].toFixed(3)}°…${fp[2].toFixed(3)}° lon · ${r.crs}`} />
      </Suspense>
    </Glass>
  );
}

export function ChangeViewer({ r, runId }: { r: ChangeResult; runId: string }) {
  const [split, setSplit] = useState(50);
  const [alpha, setAlpha] = useState(0.6);
  const [enabled, setEnabled] = useState<Record<string, boolean>>(() => Object.fromEntries(Object.keys(r.legend).map((k) => [k, true])));
  const [focus, setFocus] = useState<number | null>(null);
  const { reviews, set } = useReviews(runId);
  const base = `/api/runs/${runId}/files`;
  const overlay = useMaskOverlay(`${base}/change.png`, enabled, alpha);
  const box = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const move = (clientX: number) => {
    const b = box.current?.getBoundingClientRect();
    if (b) setSplit(Math.min(100, Math.max(0, ((clientX - b.left) / b.width) * 100)));
  };
  const area = r.changed_area_km2;
  const hist = useMemo(() => r.histogram && {
    x: r.histogram.edges.slice(0, -1).map((e, i) => (e + r.histogram!.edges[i + 1]) / 2), y: r.histogram.counts }, [r]);

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Kpi label={r.method === "water" ? "Water change" : "Changed surface"} value={pct(r.changed_fraction)} accent="#db2777" sub="of valid overlapping pixels" />
        <Kpi label="Changed area" value={area == null ? `${(r.counts.changed ?? (r.counts.gained ?? 0) + (r.counts.lost ?? 0)).toLocaleString()} px` : `${num(area, 3)} km²`} sub={r.area_basis} />
        {r.method === "water"
          ? <Kpi label="Gained / lost" value={`${(r.counts.gained ?? 0).toLocaleString()} / ${(r.counts.lost ?? 0).toLocaleString()} px`} sub={`NDWI > ${r.threshold}`} />
          : <Kpi label="Threshold" value={num(r.threshold, 3)} sub={String(r.settings.threshold_mode) === "otsu" ? "automatic (Otsu)" : "manual"} />}
        <Kpi label="Valid overlap" value={pct(r.valid_fraction)} sub={`${r.width}×${r.height} px · ${r.bands} band(s)${r.georeferenced ? " · georeferenced" : ""}`} />
      </div>
      {r.notes.map((n) => <Notice key={n} tone="info" title={n} />)}
      <div className="grid gap-4 2xl:grid-cols-[1fr_320px]">
        <Glass className="space-y-3 p-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="label">Swipe: {r.before.file} ◂▸ {r.after.file}</div>
            <div className="flex flex-wrap gap-1.5">
              {Object.entries(r.legend).map(([k, name]) => {
                const [cr, cg, cb] = CHANGE_COLORS[k];
                return (
                  <button key={k} onClick={() => setEnabled((s) => ({ ...s, [k]: !s[k] }))}
                    className={`chip border ${enabled[k] ? "border-white bg-white shadow-sm" : "border-transparent bg-white/40 opacity-50"}`}>
                    <span className="h-3 w-3 rounded-full" style={{ background: `rgb(${cr},${cg},${cb})` }} />{name}
                  </button>);
              })}
            </div>
          </div>
          <div ref={box} className="relative select-none overflow-hidden rounded-2xl bg-slate-200/60" style={{ aspectRatio: `${r.width} / ${r.height}` }}
            onPointerDown={(e) => { dragging.current = true; move(e.clientX); (e.target as HTMLElement).setPointerCapture?.(e.pointerId); }}
            onPointerMove={(e) => dragging.current && move(e.clientX)} onPointerUp={() => { dragging.current = false; }}>
            <img src={`${base}/after.png`} alt="after" className="absolute inset-0 h-full w-full object-fill" draggable={false} />
            <img src={`${base}/before.png`} alt="before" className="absolute inset-0 h-full w-full object-fill" draggable={false}
              style={{ clipPath: `inset(0 ${100 - split}% 0 0)` }} />
            {overlay && <img src={overlay} alt="change" className="pointer-events-none absolute inset-0 h-full w-full object-fill [image-rendering:pixelated]" />}
            {focus != null && r.regions[focus] && (() => {
              const [x0, y0, x1, y1] = r.regions[focus].bbox;
              return <div className="pointer-events-none absolute rounded border-2 border-white shadow-[0_0_0_2px_#db2777]"
                style={{ left: `${(x0 / r.width) * 100}%`, top: `${(y0 / r.height) * 100}%`, width: `${((x1 - x0) / r.width) * 100}%`, height: `${((y1 - y0) / r.height) * 100}%` }} />;
            })()}
            <div className="pointer-events-none absolute inset-y-0 w-0.5 bg-white shadow-[0_0_8px_rgba(0,0,0,0.4)]" style={{ left: `${split}%` }}>
              <div className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white px-2 py-1 text-[10px] font-bold text-ink shadow">◂ ▸</div>
            </div>
            <span className="absolute left-2 top-2 rounded-full bg-white/85 px-2 py-0.5 text-[10px] font-bold">BEFORE</span>
            <span className="absolute right-2 top-2 rounded-full bg-white/85 px-2 py-0.5 text-[10px] font-bold">AFTER</span>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-xs font-semibold text-ink-soft">Overlay</span>
            <input type="range" min={0} max={1} step={0.05} value={alpha} onChange={(e) => setAlpha(+e.target.value)} className="slider" />
            <span className="w-10 text-right text-xs tabular-nums">{Math.round(alpha * 100)}%</span>
          </div>
        </Glass>
        <div className="space-y-4">
          <Glass className="p-4">
            <div className="label">Largest changed regions · review</div>
            {r.regions.length === 0 && <p className="mt-2 text-sm text-ink-soft">No regions above the size limit.</p>}
            <div className="mt-2 space-y-1.5">
              {r.regions.map((g, i) => (
                <div key={i} onMouseEnter={() => setFocus(i)} onMouseLeave={() => setFocus(null)}
                  className={`flex items-center gap-2 rounded-xl px-2 py-1.5 text-xs transition ${focus === i ? "bg-white" : "bg-white/40"}`}>
                  <span className="font-semibold">#{i + 1}</span>
                  <span className="flex-1 text-ink-soft">{g.area_km2 != null ? `${num(g.area_km2, 3)} km²` : `${g.pixels.toLocaleString()} px`}</span>
                  <ReviewToggle item={`region:${i}`} review={reviews[`region:${i}`]} onSet={set} />
                </div>))}
            </div>
          </Glass>
          {hist && (
            <Glass className="p-4">
              <div className="label">Change magnitude histogram</div>
              <Plot height={180} data={[{ x: hist.x, y: hist.y, type: "bar", marker: { color: "#f9a8d4" } }]}
                layout={{ margin: { l: 40, r: 10, t: 8, b: 30 }, yaxis: { type: "log" }, bargap: 0.05,
                  shapes: [{ type: "line", x0: r.histogram!.threshold, x1: r.histogram!.threshold, yref: "paper", y0: 0, y1: 1, line: { color: "#db2777", width: 2 } }] } as Partial<Plotly.Layout>} />
              <p className="text-[10px] text-ink-faint">Pixels right of the pink line are counted as changed.</p>
            </Glass>
          )}
        </div>
      </div>
      {r.footprint_lonlat && <Footprint r={r} />}
    </div>
  );
}
