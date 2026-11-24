import { AlertTriangle, Info, Layers, ScanSearch, Sparkles, Tags } from "lucide-react";
import { useState } from "react";
import { pct, TERRAIN_COLORS } from "../../lib/format";
import LazyPlot from "../LazyPlot";
import { Glass, Kpi, Segmented } from "../ui";

export interface Finding { kind: "info" | "warn" | "classification" | "terrain"; title: string; text: string }
export interface DatasetProfile {
  category: string; label: string; confidence: number; reasons: string[]; facts: Record<string, unknown>;
  recommended: string | null; why?: string; mars_model?: boolean;
}

const KIND = {
  info: { icon: Info, cls: "border-sky-200/70 bg-sky-50/70 text-sky-950 dark:border-sky-400/20 dark:bg-sky-500/10 dark:text-sky-100" },
  warn: { icon: AlertTriangle, cls: "border-amber-200/80 bg-amber-50/80 text-amber-950 dark:border-amber-400/20 dark:bg-amber-500/10 dark:text-amber-100" },
  classification: { icon: Tags, cls: "border-violet-200/80 bg-violet-50/80 text-violet-950 dark:border-violet-400/20 dark:bg-violet-500/10 dark:text-violet-100" },
  terrain: { icon: Layers, cls: "border-emerald-200/80 bg-emerald-50/80 text-emerald-950 dark:border-emerald-400/20 dark:bg-emerald-500/10 dark:text-emerald-100" },
};

/** Plain-language findings of a result. */
export function Findings({ items, title = "What we found" }: { items?: Finding[]; title?: string }) {
  if (!items?.length) return null;
  return (
    <Glass className="hud space-y-3 p-5">
      <div className="flex items-center gap-2"><Sparkles className="h-4 w-4 text-pink-500" /><div className="kicker">{title}</div></div>
      <div className="grid gap-2 md:grid-cols-2">
        {items.map((f, i) => {
          const K = KIND[f.kind] ?? KIND.info;
          return (
            <div key={i} className={`rounded-2xl border p-3 text-[13px] leading-snug ${K.cls}`}>
              <div className="mb-0.5 flex items-center gap-1.5 font-semibold"><K.icon className="h-3.5 w-3.5" />{f.title}</div>
              <div className="opacity-90">{f.text}</div>
            </div>);
        })}
      </div>
    </Glass>
  );
}

/** "What is this dataset?" - shown right after upload, before choosing an analysis. */
export function ProfileCard({ p, analysisTitle }: { p: DatasetProfile; analysisTitle?: string }) {
  return (
    <div className="rounded-2xl border border-pink-200/70 bg-gradient-to-br from-pink-50/80 to-amber-50/60 p-4 dark:border-pink-400/20 dark:from-pink-500/10 dark:to-amber-500/5">
      <div className="flex flex-wrap items-center gap-2">
        <ScanSearch className="h-4 w-4 text-pink-500" />
        <span className="text-[11px] font-semibold uppercase tracking-wider text-ink-faint">Detected</span>
        <span className="font-display text-base font-bold">{p.label}</span>
        <span className="chip bg-white/70 text-ink-soft">{Math.round(p.confidence * 100)}% sure</span>
      </div>
      <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-soft">
        {p.reasons.map((r) => <li key={r}>• {r}</li>)}
      </ul>
      {analysisTitle && <p className="mt-2 text-xs"><b>Recommended: {analysisTitle}.</b> {p.why}</p>}
    </div>
  );
}

/* ------------------------------------------------------------------ image collection viewer */
interface ImgRow {
  id: string; file: string; width: number; height: number; scene: string; scene_label: string; why: string[]; group?: number;
  features: Record<string, number>; texture: { fine: number; textured: number; blocky: number };
  regions?: Record<string, number>; rocks?: number; rock_density?: number; color?: boolean;
  terrain?: Record<string, number>; overlay?: string; thumb: string; region_map?: string; texture_map?: string; error?: string;
  nasa?: { mission: string; camera: string; sol?: number } | null;
}
export interface InsightsResult {
  viewer: "insights"; n_images: number; n_analysed: number; images: ImgRow[]; insights: Finding[];
  groups: { id: number; label: string; members: string[] }[]; duplicates: { a_file: string; b_file: string; distance: number }[];
  scene_counts: Record<string, number>; scene_labels: Record<string, string>; model_used: boolean;
  summary: { regions?: Record<string, number>; rocks?: number } & Record<string, unknown>;
}

const SCENE_CLS: Record<string, string> = {
  landscape: "bg-sky-100 text-sky-800", ground: "bg-amber-100 text-amber-800", mixed: "bg-violet-100 text-violet-800",
  hardware: "bg-indigo-100 text-indigo-800", blank: "bg-rose-100 text-rose-700",
};
const REGION_COLOR: Record<string, string> = { terrain: "#d97706", hardware: "#6366f1", shadow: "#312e81", sky: "#38bdf8" };
const rgb = (c: string) => `rgb(${(TERRAIN_COLORS[c] ?? [148, 163, 184]).join(",")})`;
type View = "photo" | "regions" | "texture" | "terrain";

function Bar({ parts }: { parts: [string, number, string][] }) {
  return (
    <div className="flex h-2 overflow-hidden rounded-full bg-white/60">
      {parts.map(([k, v, c]) => <div key={k} title={`${k} ${pct(v, 0)}`} style={{ width: `${Math.max(0, v) * 100}%`, background: c }} />)}
    </div>
  );
}
const short = (f: string) => (f.length > 26 ? `${f.slice(0, 12)}…${f.slice(-10)}` : f);

function srcFor(x: ImgRow, view: View, runId: string) {
  const p = view === "regions" ? x.region_map : view === "texture" ? x.texture_map : view === "terrain" ? x.overlay : x.thumb;
  return `/api/runs/${runId}/files/${p ?? x.thumb}`;
}

export function InsightsViewer({ r, runId }: { r: InsightsResult; runId: string }) {
  const [group, setGroup] = useState<string>("all");
  const [view, setView] = useState<View>(r.images.some((x) => x.region_map) ? "regions" : "photo");
  const [open, setOpen] = useState<ImgRow | null>(null);
  const ok = r.images.filter((x) => !x.error);
  const imgs = ok.filter((x) => group === "all" || String(x.group) === group);
  const reg = r.summary?.regions;
  const views: { value: View; label: string }[] = [
    { value: "photo", label: "Photo" },
    ...(ok.some((x) => x.region_map) ? [{ value: "regions" as View, label: "Regions" }, { value: "texture" as View, label: "Roughness" }] : []),
    ...(r.model_used ? [{ value: "terrain" as View, label: "Terrain model" }] : []),
  ];
  const seg = ok.filter((x) => x.terrain);
  const avgTerrain = seg.length ? (["soil", "bedrock", "sand", "big rock"] as const).map((c) => [c, seg.reduce((a, x) => a + (x.terrain![c] ?? 0), 0) / seg.length] as const) : null;
  const names = ok.map((x, i) => `${i + 1}. ${short(x.file)}`);   // unique labels: shortened names can collide
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Kpi label="Images classified" value={r.n_analysed} sub={r.n_images > r.n_analysed ? `of ${r.n_images} (first ${r.n_analysed})` : "all images"} />
        <Kpi label="Open terrain" value={reg ? pct(reg.terrain, 0) : "-"} sub={reg ? `hardware ${pct(reg.hardware, 0)} · shadow ${pct(reg.shadow, 0)} · sky ${pct(reg.sky, 0)}` : ""} />
        <Kpi label="Rocks / pebbles" value={r.summary?.rocks ?? "-"} sub={`${r.groups.length} visual group(s) · ${r.duplicates.length} near-duplicate pair(s)`} />
        <Kpi label="Terrain model" value={r.model_used ? "applied" : "not installed"} accent={r.model_used ? "#059669" : "#d97706"}
          sub={avgTerrain ? avgTerrain.map(([c, v]) => `${c} ${pct(v, 0)}`).join(" · ") : "region map + texture only"} />
      </div>

      <Findings items={r.insights} />

      {ok.length > 0 && ok[0].regions && (
        <div className="grid gap-4 xl:grid-cols-2">
          <Glass className="p-4">
            <div className="kicker mb-1">What each image shows</div>
            <LazyPlot height={Math.max(260, 40 + ok.length * 26)} data={(["terrain", "hardware", "shadow", "sky"] as const).map((k) => ({
              type: "bar", orientation: "h", name: k, y: names, x: ok.map((x) => (x.regions?.[k] ?? 0) * 100),
              marker: { color: REGION_COLOR[k] }, hovertemplate: `%{y}<br>${k} %{x:.0f}%<extra></extra>`,
            }))} layout={{ barmode: "stack", margin: { l: 170, r: 10, t: 30, b: 36 }, legend: { orientation: "h", y: 1.12, x: 0 },
              xaxis: { title: { text: "% of frame" }, range: [0, 100] }, yaxis: { automargin: true, autorange: "reversed", type: "category" } }} />
          </Glass>
          <Glass className="p-4">
            <div className="kicker mb-1">Ground roughness vs rock density (terrain pixels only)</div>
            <LazyPlot height={300} data={[{
              type: "scatter", mode: "markers", x: ok.map((x) => x.features.roughness), y: ok.map((x) => x.rock_density ?? 0),
              text: ok.map((x) => `${short(x.file)}<br>${x.scene_label}<br>terrain ${pct(x.regions?.terrain ?? 0, 0)}`),
              marker: { size: ok.map((x) => 8 + 30 * (x.regions?.terrain ?? 0)), color: ok.map((x) => x.texture.fine), colorscale: "YlOrBr", reversescale: true,
                showscale: true, colorbar: { title: { text: "fine" }, thickness: 10 }, line: { width: 1, color: "#fff" } },
              hovertemplate: "%{text}<br>roughness %{x:.3f}<br>%{y:.1f} rocks / 10k px<extra></extra>",
            }]} layout={{ margin: { l: 56, r: 10, t: 10, b: 44 }, xaxis: { title: { text: "median local roughness" } }, yaxis: { title: { text: "rocks per 10k terrain px" } } }} />
            <p className="text-[10px] text-ink-faint">Bubble size = share of open terrain; colour = share of fine-grained ground.</p>
          </Glass>
        </div>
      )}

      <Glass className="space-y-3 p-4">
        <div className="flex flex-wrap items-center gap-2">
          <div className="kicker">Every image, classified</div>
          <Segmented value={view} onChange={(v) => setView(v as View)} options={views} />
          {r.groups.length > 1 && (
            <Segmented value={group} onChange={setGroup} options={[{ value: "all", label: "All" },
              ...r.groups.map((g) => ({ value: String(g.id), label: `${g.label.split(" · ")[0].split(" ")[0]} (${g.members.length})` }))]} />)}
        </div>
        <Legend view={view} />
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-4">
          {imgs.map((x, i) => (
            <button key={x.id} type="button" onClick={() => setOpen(x)} className="glass-inset space-y-1.5 p-2 text-left transition hover:ring-2 hover:ring-pink-300">
              <img src={srcFor(x, view, runId)} alt={x.file} loading={i < 12 ? "eager" : "lazy"} className="aspect-[4/3] w-full rounded-lg bg-slate-800 object-cover" />
              <div className="flex items-center gap-1.5">
                <span className={`chip px-1.5 text-[10px] ${SCENE_CLS[x.scene] ?? ""}`}>{x.scene_label}</span>
                {x.nasa && <span className="truncate text-[10px] text-ink-faint">{x.nasa.camera}{x.nasa.sol != null ? ` · sol ${x.nasa.sol}` : ""}</span>}
              </div>
              <div className="truncate font-mono text-[10px] text-ink-soft" title={x.file}>{x.file}</div>
              {x.regions && <Bar parts={(["terrain", "hardware", "shadow", "sky"] as const).map((k) => [k, x.regions![k] ?? 0, REGION_COLOR[k]])} />}
              {x.terrain ? (
                <Bar parts={Object.entries(x.terrain).map(([k, v]) => [k, v, rgb(k)])} />
              ) : (
                <Bar parts={[["fine", x.texture.fine, "#fbbf24"], ["textured", x.texture.textured, "#a78bfa"], ["blocky", x.texture.blocky, "#475569"]]} />
              )}
              <div className="text-[10px] text-ink-soft">{x.regions ? `terrain ${pct(x.regions.terrain, 0)} · ` : ""}{x.rocks ?? 0} rocks · fine ground {pct(x.texture.fine, 0)}</div>
            </button>))}
        </div>
        <p className="text-[11px] text-ink-faint">Regions, texture and rocks are measured from the pixels (edge density, darkness, smoothness) - heuristics, not a trained classifier.
          {r.model_used ? " Terrain classes come from the installed DeepLabV3+ model, on terrain pixels only." : " Install the Mars model to add soil / bedrock / sand / big rock."}</p>
      </Glass>
      {open && <Detail x={open} runId={runId} model={r.model_used} onClose={() => setOpen(null)} />}
    </div>
  );
}

function Legend({ view }: { view: View }) {
  const items: [string, string][] = view === "regions" ? [["terrain (unchanged)", "transparent"], ["rover hardware", REGION_COLOR.hardware], ["shadow", REGION_COLOR.shadow], ["sky", REGION_COLOR.sky], ["rock / pebble", "#fb923c"]]
    : view === "texture" ? [["smooth", "rgb(30,41,99)"], ["fine texture", "rgb(20,150,160)"], ["textured", "rgb(250,204,21)"], ["rough / rocky", "rgb(220,38,38)"]]
    : view === "terrain" ? Object.keys(TERRAIN_COLORS).map((c) => [c, rgb(c)] as [string, string]) : [];
  if (!items.length) return null;
  return (
    <div className="flex flex-wrap gap-3 text-[11px] text-ink-soft">
      {items.map(([k, c]) => <span key={k} className="flex items-center gap-1"><span className="h-2.5 w-2.5 rounded-sm border border-white/60" style={{ background: c }} />{k}</span>)}
    </div>
  );
}

function Detail({ x, runId, model, onClose }: { x: ImgRow; runId: string; model: boolean; onClose: () => void }) {
  const panes: [View, string][] = [["photo", "Photo"], ["regions", "Regions + rocks"], ["texture", "Roughness"], ...(model && x.overlay ? [["terrain", "Terrain model"] as [View, string]] : [])];
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" onClick={onClose}>
      <div className="glass scroll-thin max-h-[92vh] w-full max-w-6xl space-y-3 overflow-y-auto p-5" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center gap-2">
          <span className={`chip ${SCENE_CLS[x.scene] ?? ""}`}>{x.scene_label}</span>
          <span className="truncate font-mono text-xs">{x.file}</span>
          <button className="btn-ghost ml-auto px-3 py-1 text-xs" onClick={onClose}>Close</button>
        </div>
        <div className="grid gap-2 md:grid-cols-2">
          {panes.map(([v, label]) => (
            <figure key={v} className="space-y-1"><img src={srcFor(x, v, runId)} alt={label} className="w-full rounded-lg bg-slate-800" />
              <figcaption className="text-[11px] text-ink-soft">{label}</figcaption></figure>))}
        </div>
        <div className="grid gap-2 text-xs sm:grid-cols-2 lg:grid-cols-4">
          {x.regions && Object.entries(x.regions).map(([k, v]) => <div key={k} className="glass-inset p-2"><div className="text-ink-faint">{k}</div><div className="font-display text-lg font-bold">{pct(v, 0)}</div></div>)}
          <div className="glass-inset p-2"><div className="text-ink-faint">rocks / pebbles</div><div className="font-display text-lg font-bold">{x.rocks ?? 0}</div></div>
          <div className="glass-inset p-2"><div className="text-ink-faint">fine / blocky ground</div><div className="font-display text-lg font-bold">{pct(x.texture.fine, 0)} / {pct(x.texture.blocky, 0)}</div></div>
          <div className="glass-inset p-2"><div className="text-ink-faint">sharpness</div><div className="font-display text-lg font-bold">{x.features.sharpness.toFixed(1)}</div></div>
          {x.terrain && <div className="glass-inset p-2 sm:col-span-2"><div className="text-ink-faint">terrain model</div><div>{Object.entries(x.terrain).map(([k, v]) => `${k} ${pct(v, 0)}`).join(" · ")}</div></div>}
        </div>
        <p className="text-[11px] text-ink-soft">Why: {x.why.join("; ")}</p>
      </div>
    </div>
  );
}
