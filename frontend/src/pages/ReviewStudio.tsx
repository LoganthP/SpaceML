import { WriteButton, useCapabilities } from "../lib/capabilities";
import { CheckCheck, ExternalLink, FileJson, FileSpreadsheet, FolderArchive, Highlighter, Inbox, RefreshCw, Tag } from "lucide-react";
import { lazy, Suspense, useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { ErrorNotice, Glass, IconBadge, Page, Segmented, Spinner } from "../components/ui";
import { api, useApi } from "../lib/api";
import { MODULE_META, timeAgo } from "../lib/format";
import type { Module, Run } from "../lib/types";

const MaskEditor = lazy(() => import("../components/MaskEditor"));

interface ReviewRow {
  run_id: string; item: string; verdict: string; note: string | null; updated_at: string; module: Module; input_label: string;
  kind?: string; analysis?: string | null; chan_id?: string; start?: number; end?: number; x0?: string | number; x1?: string | number;
  kepid?: number; tce_plnt_num?: number; period_days?: number; score?: number | null; area_km2?: number | null; pixels?: number;
  mission?: string; target?: string; label?: number | string; pred?: number; detector?: string;
}

const VERDICT = {
  confirmed: "bg-emerald-100 text-emerald-700", dismissed: "bg-rose-100 text-rose-700",
  unsure: "bg-amber-100 text-amber-800", corrected: "bg-violet-100 text-violet-700",
} as Record<string, string>;

function describe(r: ReviewRow) {
  switch (r.kind) {
    case "telemetry_interval": return `${r.chan_id} steps ${r.start}–${r.end}`;
    case "series_interval": return `${String(r.x0).replace("T", " ").slice(0, 19)} → ${String(r.x1).replace("T", " ").slice(0, 19)}`;
    case "change_region": return r.area_km2 != null ? `${r.area_km2.toFixed(3)} km² region` : `${r.pixels} px region`;
    case "transit_candidate": return r.kepid ? `KIC ${r.kepid} TCE ${r.tce_plnt_num} · P ${r.period_days?.toFixed(3)} d` : `uploaded target · P ${r.period_days?.toFixed(3)} d`;
    case "terrain_mask": return "terrain mask corrected";
    case "mission_segment": return `${r.mission} segment ${r.target} · label ${r.label ? "anomaly" : "nominal"} · pred ${r.pred ? "anomaly" : "nominal"}`;
    case "mission_interval": return `${r.mission} ${r.target} interval (${r.detector})`;
    case "labeled_sample": return `window ${(r as ReviewRow & { sample_id?: string }).sample_id ?? r.item} · label ${String(r.label ?? "")}`;
    default: return r.item;
  }
}

function openPath(r: ReviewRow) {
  return `${MODULE_META[r.module].path}?run=${r.run_id}`;
}

export default function ReviewStudio() {
  const [sp] = useSearchParams();
  const [module, setModule] = useState<"all" | Module>(() => (sp.get("module") as Module | null) ?? "all");
  const [verdict, setVerdict] = useState<"all" | "confirmed" | "dismissed" | "unsure" | "corrected">("all");
  const qs = new URLSearchParams({ ...(module !== "all" ? { module } : {}), ...(verdict !== "all" ? { verdict } : {}) });
  const data = useApi<{ reviews: ReviewRow[]; counts: Record<string, number> }>(`/api/reviews?${qs}`, [module, verdict]);
  const all = useApi<{ reviews: ReviewRow[]; counts: Record<string, number> }>("/api/reviews");
  const marsRuns = useApi<Run[]>("/api/runs?module=mars&limit=40");
  const wbRuns = useApi<Run[]>("/api/runs?module=workbench&limit=60");
  const [maskRun, setMaskRun] = useState<string | null>(null);
  const terrainRuns = [...(marsRuns.data ?? []), ...(wbRuns.data ?? []).filter((r) => (r.input_ref as { analysis?: string }).analysis === "image.terrain")]
    .filter((r) => r.status === "succeeded");
  const c = all.data?.counts ?? {};
  const total = Object.values(c).reduce((a, b) => a + b, 0);
  const exp = (f: string) => `/api/reviews/export?format=${f}${module !== "all" ? `&module=${module}` : ""}`;
  const openMask = (id: string) => { setMaskRun(id); window.setTimeout(() => document.getElementById("mask-editor")?.scrollIntoView({ behavior: "smooth" }), 150); };

  return (
    <Page title="Label & Review Studio"
      subtitle="Judge what the models flagged - telemetry intervals, mission alerts, change regions, transit candidates, flare forecasts and terrain masks - with the evidence next to each item. Your verdicts export as labels for the next training run."
      icon={<IconBadge gradient="from-violet-500 to-indigo-600"><Highlighter className="h-6 w-6" /></IconBadge>}>
      <Glass className="flex flex-wrap items-center gap-x-5 gap-y-2 px-5 py-3">
        <div className="flex items-center gap-2"><Tag className="h-4 w-4 text-violet-500" aria-hidden="true" /><span className="font-display text-lg font-bold">{total}</span><span className="text-xs text-ink-soft">verdicts</span></div>
        <div className="flex flex-wrap gap-1.5 text-xs">
          {["confirmed", "dismissed", "unsure", "corrected"].map((v) => <span key={v} className={`chip ${VERDICT[v]}`}>{v} {c[v] ?? 0}</span>)}
        </div>
        <div className="ml-auto flex flex-wrap items-center gap-2">
          <span className="text-xs text-ink-soft">Export labels</span>
          <a className="btn-ghost px-2.5 py-1 text-xs" href={exp("csv")} download><FileSpreadsheet className="h-3.5 w-3.5" aria-hidden="true" /> CSV</a>
          <a className="btn-ghost px-2.5 py-1 text-xs" href={exp("json")} download><FileJson className="h-3.5 w-3.5" aria-hidden="true" /> JSON</a>
          <a className="btn-ghost px-2.5 py-1 text-xs" href={exp("zip")} download><FolderArchive className="h-3.5 w-3.5" aria-hidden="true" /> ZIP + masks</a>
        </div>
      </Glass>

      <ReviewQueue module={module} setModule={setModule} onChange={() => { all.reload(); data.reload(); }} onMask={openMask} />

      <Glass className="space-y-3 p-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="kicker">Your verdicts</div>
          <Segmented label="Verdict" value={verdict} onChange={setVerdict} options={[
            { value: "all", label: "Any" }, { value: "confirmed", label: "Confirmed" }, { value: "dismissed", label: "Dismissed" },
            { value: "unsure", label: "Unsure" }, { value: "corrected", label: "Corrected" }]} />
        </div>
        {data.error && <ErrorNotice message={data.error.message} />}
        {data.loading && !data.data && <div className="flex items-center gap-2 p-3 text-sm text-ink-soft"><Spinner /> Loading…</div>}
        {data.data?.reviews.length === 0 && (
          <p className="flex items-center gap-2 text-sm text-ink-soft"><CheckCheck className="h-4 w-4 text-violet-500" aria-hidden="true" />
            {verdict === "all" ? "No verdicts yet - judge items in the queue above; they are listed here and in the exports." : `No ${verdict} verdicts${module !== "all" ? ` for ${MODULE_META[module].short}` : ""}.`}</p>)}
        {!!data.data?.reviews.length && (
          <div className="scroll-thin max-h-[360px] overflow-auto">
            <table className="w-full min-w-[760px] text-sm">
              <thead className="sticky top-0 bg-white/95 dark:bg-slate-900/95"><tr className="text-left text-xs text-ink-faint"><th className="px-2 py-2">Module</th><th>Item</th><th>Run input</th><th>Verdict</th><th>Note</th><th>When</th><th /></tr></thead>
              <tbody>
                {data.data.reviews.map((r) => (
                  <tr key={`${r.run_id}-${r.item}`} className="border-t border-slate-200/60 dark:border-white/10">
                    <td className="px-2 py-1.5"><span className={`chip ${MODULE_META[r.module].soft}`}>{MODULE_META[r.module].short}</span></td>
                    <td className="font-mono text-xs">{describe(r)}</td>
                    <td className="max-w-[220px] truncate text-xs text-ink-soft">{r.input_label}</td>
                    <td><span className={`chip ${VERDICT[r.verdict] ?? ""}`}>{r.verdict}</span></td>
                    <td className="max-w-[200px] truncate text-xs text-ink-soft">{r.note}</td>
                    <td className="text-xs text-ink-faint">{timeAgo(r.updated_at)}</td>
                    <td className="text-right"><Link to={openPath(r)} className="btn-ghost px-2 py-1 text-[11px]" aria-label={`Open the run of ${describe(r)}`}><ExternalLink className="h-3 w-3" aria-hidden="true" /> Open</Link></td>
                  </tr>))}
              </tbody>
            </table>
          </div>)}
      </Glass>

      {(terrainRuns.length > 0 || maskRun) && (
        <Glass className="space-y-3 p-5">
          <div>
            <div className="kicker">Correct a terrain mask</div>
            <p className="text-xs text-ink-soft">Pick a Mars terrain run, paint the right class over mistakes (sand matters most for driving) and save. Saved masks are part of the ZIP export.</p>
          </div>
          <div className="scroll-thin flex gap-2 overflow-x-auto pb-1">
            {terrainRuns.map((r) => (
              <button key={r.id} onClick={() => setMaskRun(r.id)} aria-pressed={maskRun === r.id}
                className={`w-28 shrink-0 overflow-hidden rounded-xl text-left ring-2 transition ${maskRun === r.id ? "ring-violet-500" : "ring-transparent hover:ring-violet-200"}`}>
                <img src={`/api/runs/${r.id}/files/input.png`} onError={(e) => { (e.target as HTMLImageElement).src = `/api/runs/${r.id}/files/source.png`; }} alt="" className="aspect-square w-full object-cover" />
                <div className="truncate bg-white/80 px-1.5 py-1 text-[10px] font-semibold dark:bg-slate-900/80">{r.input_label}</div>
              </button>))}
          </div>
          <div id="mask-editor" />
          {maskRun && <Suspense fallback={<Spinner />}><MaskEditor key={maskRun} runId={maskRun} onSaved={() => { all.reload(); data.reload(); }} /></Suspense>}
        </Glass>)}
    </Page>
  );
}


interface QueueItem { run_id: string; item: string; kind: string; title: string; detail: string; hint?: string | null; mask?: boolean; thumb?: string;
  module: Module; input_label: string; origin?: string; created_at: string; repeats?: number;
  mission?: string; target?: string; label?: number; pred?: number; prob?: number; outcome?: string; start?: number; end?: number }

interface Evidence {
  type: "series" | "images" | "mission"; facts?: Record<string, unknown>;
  x?: (number | string)[]; x_label?: string; shade?: [number | string, number | string][]; markers?: boolean;
  panels?: { name: string; x?: (number | string)[]; hline?: number | null; lines: { name: string; y: (number | null)[]; bold?: boolean; dash?: boolean }[] }[];
  images?: { src: string; label: string }[]; mask?: boolean;
  mission?: string; target?: string; start?: number; end?: number;
}

const MODULES: Module[] = ["missions", "telemetry", "workbench", "exoplanet", "mars"];

function ReviewQueue({ module, setModule, onChange, onMask }: { module: "all" | Module; setModule: (m: "all" | Module) => void; onChange: () => void; onMask: (runId: string) => void }) {
  const readOnly = useCapabilities()?.capabilities.write === false;
  const q = useApi<{ items: QueueItem[]; pending: number; runs_scanned: number; by_module: Record<string, number> }>(`/api/reviews/queue?limit=150${module !== "all" ? `&module=${module}` : ""}`, [module]);
  const allQ = useApi<{ by_module: Record<string, number>; pending: number }>("/api/reviews/queue?limit=1");
  const [done, setDone] = useState<Record<string, string>>({});
  const [sel, setSel] = useState<string | null>(null);
  const [note, setNote] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const items = q.data?.items ?? [];
  const left = items.filter((it) => !done[`${it.run_id}/${it.item}`]);
  const key = (it: QueueItem) => `${it.run_id}/${it.item}`;
  const current = left.find((it) => key(it) === sel) ?? left[0] ?? null;
  useEffect(() => { setNote(""); }, [current && key(current)]); // eslint-disable-line react-hooks/exhaustive-deps
  async function verdict(it: QueueItem, v: "confirmed" | "dismissed" | "unsure") {
    if (readOnly) return;
    setErr(null);
    try {
      await api(`/api/runs/${it.run_id}/reviews/${encodeURIComponent(it.item)}`, { method: "PUT", body: JSON.stringify({ verdict: v, note }) });
      const i = left.findIndex((x) => key(x) === key(it));
      const next = left[i + 1] ?? left[i - 1] ?? null;
      setDone((d) => ({ ...d, [key(it)]: v }));
      setSel(next ? key(next) : null);
      onChange(); allQ.reload();
    } catch (e) { setErr((e as Error).message); }
  }
  const onKey = (e: React.KeyboardEvent) => {
    if (!current || (e.target as HTMLElement).tagName === "INPUT") return;
    const i = left.findIndex((x) => key(x) === key(current));
    if (e.key === "ArrowDown" || e.key === "j") { e.preventDefault(); const n = left[Math.min(left.length - 1, i + 1)]; n && setSel(key(n)); }
    else if (e.key === "ArrowUp" || e.key === "k") { e.preventDefault(); const n = left[Math.max(0, i - 1)]; n && setSel(key(n)); }
    else if (!current.mask && e.key === "1") verdict(current, "confirmed");
    else if (!current.mask && e.key === "2") verdict(current, "dismissed");
    else if (!current.mask && e.key === "3") verdict(current, "unsure");
  };
  const counts = allQ.data?.by_module ?? {};
  const pendingAll = allQ.data?.pending ?? 0;
  const open = Math.max(0, (q.data?.pending ?? 0) - Object.keys(done).length);
  return (
    <Glass className="space-y-3 p-4" onKeyDown={onKey}>
      <div className="flex flex-wrap items-center gap-2">
        <Inbox className="h-4 w-4 text-violet-500" aria-hidden="true" />
        <div className="kicker">Waiting for review</div>
        <span className="chip bg-violet-100 text-violet-700">{open} open</span>
        <button className="btn-ghost ml-auto px-2.5 py-1 text-[11px]" onClick={() => { setDone({}); q.reload(); allQ.reload(); }}><RefreshCw className="h-3 w-3" aria-hidden="true" /> Refresh</button>
      </div>
      <div className="flex flex-wrap gap-1.5" role="group" aria-label="Filter by module">
        {(["all", ...MODULES] as ("all" | Module)[]).map((mo) => {
          const n = mo === "all" ? pendingAll : counts[mo] ?? 0;
          return (
            <button key={mo} onClick={() => { setModule(mo); setSel(null); }} aria-pressed={module === mo}
              className={`rounded-full border px-3 py-1 text-xs font-semibold transition ${module === mo ? "border-violet-400 bg-violet-600 text-white" : "border-slate-300/70 hover:bg-white/60 dark:border-white/15 dark:hover:bg-white/5"}`}>
              {mo === "all" ? "All" : MODULE_META[mo].short} <span className={module === mo ? "" : "opacity-75"}>{n}</span>
            </button>);
        })}
      </div>
      {err && <ErrorNotice message={err} />}
      {q.loading && !q.data && <div className="flex items-center gap-2 text-sm text-ink-soft"><Spinner /> Collecting flags, regions, candidates and masks…</div>}
      {q.data && left.length === 0 && (
        <div className="flex flex-col items-center gap-2 py-8 text-center text-sm text-ink-soft">
          <CheckCheck className="h-8 w-8 text-violet-500" aria-hidden="true" />
          {pendingAll ? "Nothing waiting in this module." : "Nothing waiting. Score a mission item, run anomaly detection, a Workbench analysis, an exoplanet search or a Mars terrain run - whatever it flags appears here with its evidence."}
          <div className="flex flex-wrap justify-center gap-2 pt-1">
            <Link to="/telemetry?mission=opssat" className="btn-ghost px-3 py-1 text-xs">Missions</Link>
            <Link to="/workbench" className="btn-ghost px-3 py-1 text-xs">Workbench</Link>
            <Link to="/telemetry" className="btn-ghost px-3 py-1 text-xs">Anomaly detection</Link>
          </div>
        </div>)}
      {left.length > 0 && current && (
        <div className="grid gap-4 lg:grid-cols-[minmax(280px,380px)_minmax(0,1fr)]">
          <ul role="listbox" aria-label="Items waiting for review" tabIndex={0} aria-activedescendant={`rq-${key(current).replace(/[^A-Za-z0-9_-]/g, "_")}`}
            className="scroll-thin max-h-[560px] space-y-1 overflow-y-auto pr-1 outline-none">
            {left.map((it) => {
              const on = key(it) === key(current);
              return (
                <li key={key(it)} id={`rq-${key(it).replace(/[^A-Za-z0-9_-]/g, "_")}`} role="option" aria-selected={on} onClick={() => setSel(key(it))}
                  className={`cursor-pointer rounded-xl border px-3 py-2 ${on ? "border-violet-400 bg-white shadow dark:bg-white/10" : "border-transparent hover:bg-white/60 dark:hover:bg-white/5"}`}>
                  <div className="flex items-center gap-2">
                    <span className={`chip shrink-0 ${MODULE_META[it.module].soft}`}>{MODULE_META[it.module].short}</span>
                    <span className="truncate text-sm font-semibold">{it.title}</span>
                  </div>
                  <div className="mt-0.5 truncate text-[11px] text-ink-soft">{it.detail}{it.repeats ? ` · flagged in ${it.repeats + 1} runs` : ""} · {timeAgo(it.created_at)}</div>
                </li>);
            })}
          </ul>
          <ReviewDetail key={key(current)} it={current} note={note} setNote={setNote} onVerdict={(v) => verdict(current, v)} onMask={onMask} />
        </div>)}
      <p className="text-[11px] text-ink-faint">Keyboard: ↑ ↓ (or j / k) move · 1 confirm · 2 false alarm · 3 unsure. Repeated flags from re-runs of the same item are shown once.</p>
    </Glass>
  );
}

function ReviewDetail({ it, note, setNote, onVerdict, onMask }: { it: QueueItem; note: string; setNote: (s: string) => void;
  onVerdict: (v: "confirmed" | "dismissed" | "unsure") => void; onMask: (runId: string) => void }) {
  const ev = useApi<Evidence>(`/api/reviews/evidence?run_id=${encodeURIComponent(it.run_id)}&item=${encodeURIComponent(it.item)}`, [it.run_id, it.item]);
  return (
    <section className="glass-inset min-w-0 space-y-3 p-4" aria-label={`Evidence for ${it.title}`}>
      <div className="flex flex-wrap items-start gap-2">
        <div className="min-w-0 flex-1">
          <h3 className="font-display text-base font-bold">{it.title}</h3>
          <p className="text-xs text-ink-soft">{it.detail} · run of “{it.input_label}” · {timeAgo(it.created_at)}</p>
        </div>
        <Link to={`${MODULE_META[it.module].path}?run=${it.run_id}`} className="btn-ghost px-2.5 py-1 text-[11px]"><ExternalLink className="h-3 w-3" aria-hidden="true" /> Open run</Link>
      </div>
      {ev.loading && !ev.data && <div className="flex h-48 items-center justify-center gap-2 text-sm text-ink-soft" role="status"><Spinner /> Loading evidence…</div>}
      {ev.error && <p className="text-sm text-ink-soft">No preview for this item ({ev.error.message}). Open the run to inspect it.</p>}
      {ev.data && <EvidenceView ev={ev.data} it={it} />}
      {ev.data?.facts && (
        <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs sm:grid-cols-3">
          {Object.entries(ev.data.facts).filter(([, v]) => v != null && v !== "").map(([k, v]) => (
            <div key={k} className="min-w-0"><dt className="text-ink-faint">{k}</dt><dd className="truncate font-semibold">{typeof v === "boolean" ? (v ? "yes" : "no") : typeof v === "number" ? Number(v.toPrecision(4)).toLocaleString() : String(v)}</dd></div>))}
        </dl>)}
      {it.hint && <p className="text-xs text-ink-soft">Suggestion from the expert labels / model score: <b>{it.hint}</b></p>}
      {it.mask ? (
        <div className="flex flex-wrap gap-2">
          <button className="btn-primary px-3 py-1.5 text-sm" onClick={() => onMask(it.run_id)}><Highlighter className="h-4 w-4" aria-hidden="true" /> Correct the mask</button>
          <WriteButton className="btn-ghost px-3 py-1.5 text-sm" onClick={() => onVerdict("confirmed")}>✓ Mask looks right</WriteButton>
        </div>
      ) : (
        <div className="space-y-2">
          <label className="block text-xs font-semibold" htmlFor="review-note">Note (optional)</label>
          <input id="review-note" value={note} onChange={(e) => setNote(e.target.value)} placeholder="why - e.g. thruster firing, known safe mode, sensor glitch"
            className="w-full rounded-xl border border-slate-300 bg-white/80 px-2 py-1.5 text-sm dark:border-white/15 dark:bg-white/5" />
          <div className="grid grid-cols-3 gap-2">
            <WriteButton className="rounded-xl bg-emerald-700 px-3 py-2 text-sm font-bold text-white hover:bg-emerald-800" onClick={() => onVerdict("confirmed")}>✓ Real (1)</WriteButton>
            <WriteButton className="rounded-xl bg-rose-700 px-3 py-2 text-sm font-bold text-white hover:bg-rose-800" onClick={() => onVerdict("dismissed")}>✕ False alarm (2)</WriteButton>
            <WriteButton className="rounded-xl bg-amber-400 px-3 py-2 text-sm font-bold hover:bg-amber-300" style={{ color: "#1c1917" }} onClick={() => onVerdict("unsure")}>? Unsure (3)</WriteButton>
          </div>
        </div>)}
    </section>
  );
}

const EvidencePlot = lazy(() => import("../components/LazyPlot"));

function EvidenceView({ ev, it }: { ev: Evidence; it: QueueItem }) {
  if (ev.type === "mission" && ev.mission && ev.target)
    return <MissionPreview mission={ev.mission} target={ev.target} it={{ ...it, start: ev.start ?? it.start, end: ev.end ?? it.end }} />;
  if (ev.type === "images")
    return (
      <div className={`grid gap-2 ${(ev.images?.length ?? 0) > 1 ? "grid-cols-2 md:grid-cols-3" : "grid-cols-1"}`}>
        {ev.images?.map((im) => (
          <figure key={im.src} className="overflow-hidden rounded-xl bg-slate-900/80">
            <img src={im.src} alt={im.label} loading="lazy" className="max-h-72 w-full object-contain" />
            <figcaption className="px-2 py-1 text-[11px] text-slate-200">{im.label}</figcaption>
          </figure>))}
      </div>);
  const panels = ev.panels ?? [];
  const n = Math.max(1, panels.length);
  const gap = n > 1 ? 0.08 : 0, h = (1 - gap * (n - 1)) / n;
  const palette = ["#4f46e5", "#0891b2", "#be185d", "#15803d", "#b45309", "#64748b"];
  const traces = panels.flatMap((p, i) => p.lines.map((l, j) => ({
    type: "scatter", mode: ev.markers ? "markers" : "lines", x: p.x ?? ev.x, y: l.y, name: l.name, showlegend: i === 0 || n === 1,
    xaxis: i === 0 ? "x" : `x${i + 1}`, yaxis: i === 0 ? "y" : `y${i + 1}`,
    line: { width: l.bold ? 2.2 : 1.2, dash: l.dash ? "dot" : "solid", color: palette[j % palette.length] }, marker: { size: 3, color: palette[j % palette.length] } })));
  const layout: Record<string, unknown> = { margin: { l: 56, r: 8, t: 8, b: 40 }, legend: { orientation: "h", y: -0.18 }, shapes: [] as unknown[] };
  panels.forEach((p, i) => {
    const top = 1 - i * (h + gap);
    layout[i === 0 ? "yaxis" : `yaxis${i + 1}`] = { domain: [Math.max(0, top - h), top], title: { text: p.name.length > 34 ? p.name.slice(0, 34) + "…" : p.name, font: { size: 10 } } };
    layout[i === 0 ? "xaxis" : `xaxis${i + 1}`] = { anchor: i === 0 ? "y" : `y${i + 1}`, title: i === n - 1 ? { text: ev.x_label } : undefined,
      matches: i > 0 && !p.x ? "x" : undefined, showticklabels: i === n - 1 || !!p.x };
    (ev.shade ?? []).forEach(([a, b]) => (layout.shapes as unknown[]).push({ type: "rect", xref: i === 0 ? "x" : `x${i + 1}`, yref: `${i === 0 ? "y" : `y${i + 1}`} domain`,
      x0: a, x1: b, y0: 0, y1: 1, fillcolor: "rgba(225,29,72,0.15)", line: { width: 0 }, layer: "below" }));
    if (p.hline != null) (layout.shapes as unknown[]).push({ type: "line", xref: `${i === 0 ? "x" : `x${i + 1}`} domain`, yref: i === 0 ? "y" : `y${i + 1}`,
      x0: 0, x1: 1, y0: p.hline, y1: p.hline, line: { color: "#e11d48", dash: "dash", width: 1 } });
  });
  return (
    <figure>
      <Suspense fallback={<Spinner />}><EvidencePlot height={Math.min(520, 160 + 130 * n)} data={traces as Plotly.Data[]} layout={layout as Partial<Plotly.Layout>} /></Suspense>
      {ev.shade?.length ? <figcaption className="text-[11px] text-ink-soft">Shaded red: the flagged interval{panels.some((p) => p.hline != null) ? "; dashed line: alert threshold" : ""}.</figcaption> : null}
    </figure>);
}


const MissionPreviewChart = lazy(() => import("../components/LazyPlot"));

/** Inline evidence for a multi-mission review item: the signal, the published label and the prediction. */
function MissionPreview({ mission, target, it }: { mission: string; target: string; it: QueueItem }) {
  const s = useApi<{ t: number[]; cols: Record<string, (number | null)[]>; units: Record<string, string | null>; time_is_epoch: boolean; n_total: number }>(
    `/api/missions/${mission}/series?item=${encodeURIComponent(target)}&max_points=800`, [mission, target]);
  if (s.error) return <ErrorNotice message={s.error.message} />;
  if (!s.data) return <div className="flex items-center gap-2 p-2 text-xs" role="status"><Spinner /> Loading signal…</div>;
  const d = s.data;
  const x = d.time_is_epoch ? d.t.map((t) => new Date(t * 1000).toISOString()) : d.t;
  const [col, y] = Object.entries(d.cols)[0] ?? ["value", []];
  const tx = (t: number) => (d.time_is_epoch ? new Date(t * 1000).toISOString() : t);
  const shapes = it.start != null && it.end != null ? [{ type: "rect" as const, xref: "x" as const, yref: "paper" as const, x0: tx(it.start), x1: tx(it.end), y0: 0, y1: 1, fillcolor: "rgba(225,29,72,0.18)", line: { width: 0 } }] : [];
  const text = it.kind === "mission_segment"
    ? `Segment ${target}: published label ${it.label ? "anomaly" : "nominal"}, model prediction ${it.pred ? "anomaly" : "nominal"} (p=${it.prob?.toFixed(2)}), outcome ${it.outcome ?? (it.label === it.pred ? "agree" : "disagree")}; ${d.n_total} samples.`
    : `${target}: flagged interval shaded red; ${d.n_total} samples.`;
  return (
    <figure className="mt-2 space-y-1">
      <div role="img" aria-label={text}>
        <Suspense fallback={<Spinner />}><MissionPreviewChart height={200} data={[{ type: "scattergl", mode: "lines", x, y, name: col, line: { width: 1.2, color: "#4f46e5" } }]}
          layout={{ xaxis: { title: { text: d.time_is_epoch ? "Time (UTC)" : "Seconds" }, type: d.time_is_epoch ? "date" : "linear" },
            yaxis: { title: { text: d.units[col] ? `${col} [${d.units[col]}]` : `${col} (source units)` } }, shapes, margin: { t: 8, b: 40, l: 56, r: 8 } }} /></Suspense>
      </div>
      <figcaption className="text-xs text-ink-soft">{text}</figcaption>
    </figure>
  );
}
