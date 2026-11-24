import { Crosshair, Download, Filter, ImageOff, MapPinned, Mountain, Route, ScanSearch, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { JobButton, JobCard, useJobs, useOnJobFinished } from "../components/Jobs";
import { MarsModelUpload } from "../components/Setup";
import { ModuleStatus } from "../components/Catalog";
import Plot from "../components/LazyPlot";
import { ErrorNotice, Glass, IconBadge, Notice, OriginBadge, Page, Segmented, Spinner, StatusBadge } from "../components/ui";
import { api, createRun, useApi, useRun } from "../lib/api";
import { TERRAIN_COLORS, num, pct } from "../lib/format";

interface Overview {
  status: { downloaded: boolean; source: string; url: string; waypoints?: number; first_sol?: number; last_sol?: number; total_km?: number;
            provenance?: { retrieved_at: string; sha256: string } | null };
  waypoints: null | { sol: number[]; site: number[]; drive: number[]; lon: number[]; lat: number[]; dist_total_m: number[]; step: number };
  analysed: Analysed[];
}
interface Analysed {
  run_id: string; label: string; origin?: string; class_shares: Record<string, number> | null; sol: number | null; site: number | null;
  drive: number | null; lat: number | null; lon: number | null; dist_total_m: number | null; precision: string; reviewed: boolean;
  corrected: boolean; top_class: string | null;
}
interface RawImg { imageid: string; sol: number; site: number; drive: number; instrument: string; date_taken: string; lmst: string | null;
                   mast_el: string | null; match: "exact" | "site" | "sol" }
interface WaypointInfo { sol: number; site: number | null; drive: number | null; raw_status: string; raw: { images: RawImg[]; error?: string };
                         analysed: Analysed[]; samples: { id: string; name: string; site: number; drive: number }[] }
interface Sections { sections: { index: number; from_km: number; to_km: number; images: number; reviewed: number; corrected: number;
                                 sol_range: [number, number] | null; shares: Record<string, number> }[]; points: number }
interface MarsResult { class_shares: Record<string, number>; location?: { precision: string; note: string } }

const rgb = (c: string) => `rgb(${(TERRAIN_COLORS[c] ?? [148, 163, 184]).join(",")})`;
const PRECISION: Record<string, { label: string; cls: string; symbol: string }> = {
  exact: { label: "exact site/drive", cls: "bg-emerald-100 text-emerald-700", symbol: "circle" },
  site: { label: "within site", cls: "bg-sky-100 text-sky-700", symbol: "diamond" },
  sol: { label: "sol-level only", cls: "bg-amber-100 text-amber-800", symbol: "square" },
  unlocated: { label: "not located", cls: "bg-slate-100 text-slate-600", symbol: "x" },
};

function Shares({ s }: { s: Record<string, number> }) {
  return (
    <div className="flex h-2 overflow-hidden rounded-full">
      {Object.entries(s).map(([c, v]) => <div key={c} style={{ width: `${v * 100}%`, background: rgb(c) }} title={`${c} ${pct(v, 0)}`} />)}
    </div>
  );
}

function AnalyzeImage({ img, onDone, autoStart = 0 }: { img: RawImg; onDone: () => void; autoStart?: number }) {
  const [runId, setRunId] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const { run } = useRun<MarsResult>(runId);
  useEffect(() => { if (run?.status === "succeeded") onDone(); }, [run?.status, onDone]);
  useEffect(() => {
    if (!autoStart || runId) return;
    createRun("mars", { raw_image_id: img.imageid }).then((r) => setRunId(r.id)).catch((e) => setErr((e as Error).message));
  }, [autoStart]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <div className="space-y-1">
      {!runId && <button className="btn-ghost w-full justify-center px-2 py-1 text-[11px]" onClick={async () => {
        setErr(null);
        try { setRunId((await createRun("mars", { raw_image_id: img.imageid })).id); } catch (e) { setErr((e as Error).message); }
      }}><ScanSearch className="h-3.5 w-3.5" /> Segment terrain</button>}
      {run && run.status !== "succeeded" && <StatusBadge status={run.status} />}
      {run?.status === "succeeded" && run.result && (
        <div className="space-y-1">
          <Shares s={run.result.class_shares} />
          <Link to={`/mars?run=${run.id}`} className="text-[11px] font-semibold text-violet-600 hover:underline">Open mask · correct it</Link>
        </div>)}
      {(err || run?.error) && <p className="text-[10px] text-rose-600">{err ?? run?.error}</p>}
    </div>
  );
}

export default function Traverse() {
  const ov = useApi<Overview>("/api/mars/traverse");
  const models = useApi<Record<string, { active: string | null }>>("/api/models");
  const [sel, setSel] = useState<{ sol: number; site: number; drive: number } | null>(null);
  const [wp, setWp] = useState<WaypointInfo | null>(null);
  const [wpLoading, setWpLoading] = useState(false);
  const [solRange, setSolRange] = useState<[number, number]>([0, 20000]);
  const [terrain, setTerrain] = useState<string>("all");
  const [reviewedOnly, setReviewedOnly] = useState(false);
  const [sectionM, setSectionM] = useState("1000");
  const [dlErr, setDlErr] = useState<string | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [goto, setGoto] = useState("");
  const { jobs } = useJobs();
  const tjob = jobs.find((j) => j.task === "mars.traverse");
  const qs = `section_m=${sectionM}&sol_min=${solRange[0]}&sol_max=${solRange[1]}${terrain !== "all" ? `&terrain=${encodeURIComponent(terrain)}` : ""}${reviewedOnly ? "&reviewed_only=true" : ""}`;
  const sections = useApi<Sections>(ov.data?.status.downloaded ? `/api/mars/traverse/sections?${qs}` : null, [qs, ov.data?.status.downloaded]);
  const reload = useCallback(() => { ov.reload(); sections.reload(); }, [ov, sections]);
  useOnJobFinished(useCallback(() => { reload(); models.reload(); }, [reload, models]));
  const hasMarsModel = !!models.data?.mars?.active;
  const [batch, setBatch] = useState(0);
  const [zoom, setZoom] = useState<string | null>(null);
  useEffect(() => {
    if (!zoom) return;
    const k = (e: KeyboardEvent) => { if (e.key === "Escape") setZoom(null); };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [zoom]);

  useEffect(() => {
    const w = ov.data?.waypoints;
    if (w && solRange[1] === 20000) setSolRange([w.sol[0], w.sol[w.sol.length - 1]]);
  }, [ov.data?.waypoints]); // eslint-disable-line react-hooks/exhaustive-deps

  const openWaypoint = useCallback(async (p: { sol: number; site: number; drive: number }) => {
    setSel(p); setWp(null); setWpLoading(true);
    try { setWp(await api<WaypointInfo>(`/api/mars/traverse/waypoint?sol=${p.sol}&site=${p.site}&drive=${p.drive}`)); }
    catch (e) { setWp({ sol: p.sol, site: p.site, drive: p.drive, raw_status: "error", raw: { images: [], error: (e as Error).message }, analysed: [], samples: [] }); }
    finally { setWpLoading(false); }
  }, []);

  const w = ov.data?.waypoints;
  // open the most recent waypoint right away so the page is never empty
  const autoOpened = useRef(false);
  useEffect(() => {
    if (!w || autoOpened.current || sel || !w.sol.length) return;
    autoOpened.current = true;
    const k = w.sol.length - 1;
    openWaypoint({ sol: w.sol[k], site: w.site[k], drive: w.drive[k] });
  }, [w, sel, openWaypoint]);
  const points = (ov.data?.analysed ?? []).filter((a) => a.lat != null && (a.sol == null || (a.sol >= solRange[0] && a.sol <= solRange[1]))
    && (terrain === "all" || a.top_class === terrain) && (!reviewedOnly || a.reviewed));

  const fig = useMemo(() => {
    if (!w) return null;
    const keep = w.sol.map((s) => s >= solRange[0] && s <= solRange[1]);
    const f = <T,>(a: T[]) => a.filter((_, k) => keep[k]);
    const data: Plotly.Data[] = [
      { x: w.lon, y: w.lat, type: "scattergl", mode: "lines", line: { color: "rgba(148,163,184,0.35)", width: 1 }, hoverinfo: "skip", showlegend: false },
      { x: f(w.lon), y: f(w.lat), type: "scattergl", mode: "markers", name: "Waypoints",
        marker: { size: 5, color: f(w.sol), colorscale: "Plasma", showscale: true, colorbar: { title: { text: "sol" }, thickness: 10, len: 0.6 } },
        customdata: f(w.sol).map((s, k) => [s, f(w.site)[k], f(w.drive)[k], f(w.dist_total_m)[k] / 1000]) as unknown as Plotly.Datum[],
        hovertemplate: "sol %{customdata[0]} · site %{customdata[1]} / drive %{customdata[2]}<br>%{customdata[3]:.2f} km driven<extra></extra>" },
      ...Object.keys(PRECISION).filter((k) => k !== "unlocated").map((prec) => {
        const ps = points.filter((p) => p.precision === prec);
        return { x: ps.map((p) => p.lon), y: ps.map((p) => p.lat), type: "scatter", mode: "markers", name: `Analysed (${PRECISION[prec].label})`,
          marker: { size: 13, symbol: PRECISION[prec].symbol, color: ps.map((p) => rgb(p.top_class ?? "")), line: { color: ps.map((p) => p.corrected ? "#7c3aed" : "white"), width: 2 } },
          text: ps.map((p) => `${p.label}<br>${p.top_class} · ${p.corrected ? "mask corrected" : p.reviewed ? "reviewed" : "not reviewed"}`),
          hovertemplate: "%{text}<extra></extra>" } as Plotly.Data;
      }),
      ...(sel ? [{ x: [w.lon[w.sol.findIndex((s, k) => s === sel.sol && w.site[k] === sel.site)]], y: [w.lat[w.sol.findIndex((s, k) => s === sel.sol && w.site[k] === sel.site)]],
        type: "scatter", mode: "markers", name: "Selected", marker: { size: 20, symbol: "star", color: "#7c3aed", line: { color: "white", width: 2 } }, hoverinfo: "skip" } as Plotly.Data] : []),
    ];
    return { data, layout: { hovermode: "closest", margin: { l: 60, r: 10, t: 10, b: 44 }, xaxis: { title: { text: "longitude °E" } },
      yaxis: { title: { text: "latitude °N" }, scaleanchor: "x", scaleratio: 1 }, legend: { orientation: "h", y: -0.12 } } as Partial<Plotly.Layout> };
  }, [w, solRange, points, sel]);

  async function download() {
    setDownloading(true); setDlErr(null);
    try { await api("/api/mars/traverse/download", { method: "POST" }); ov.reload(); }
    catch (e) { setDlErr((e as Error).message); } finally { setDownloading(false); }
  }

  const st = ov.data?.status;
  return (
    <Page title="Rover Traverse Explorer" subtitle="Curiosity's real drive through Gale crater. Pick a waypoint to see the Navcam images taken there, segment their terrain and review the masks in place."
      icon={<IconBadge gradient="from-orange-500 to-amber-400"><Route className="h-6 w-6" /></IconBadge>}
      actions={<Link to="/mars" className="btn-ghost"><Mountain className="h-4 w-4" /> Mars Terrain</Link>}>
      <ModuleStatus ids={["traverse", "ai4mars"]} />
      {ov.error && <ErrorNotice message={ov.error.message} title="The traverse could not be loaded" />}
      {!ov.data && !ov.error && (
        <Glass className="flex h-72 flex-col items-center justify-center gap-2 text-sm text-ink-soft" role="status">
          <Spinner className="h-5 w-5" /> Loading Curiosity's waypoints and the traverse map…
        </Glass>)}
      {st && !st.downloaded && (
        <Glass className="space-y-3 p-5">
          <div className="font-display text-lg font-bold">Download Curiosity's waypoints</div>
          <p className="text-sm text-ink-soft">The surveyed rover positions (site, drive, sol, latitude/longitude, distance driven) from NASA/JPL's mission map layer - about 0.7 MB. Images are fetched per waypoint from NASA's raw-image service when you open it.</p>
          <button className="btn-primary" onClick={download} disabled={downloading}>{downloading ? <Spinner /> : <Download className="h-4 w-4" />} Download waypoints</button>
          {dlErr && <Notice tone="warn" title="NASA/JPL could not be reached">{dlErr}</Notice>}
          {tjob && <JobCard job={tjob} />}
        </Glass>)}

      {st?.downloaded && w && fig && (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-2 text-xs text-ink-soft">
            <MapPinned className="h-4 w-4 text-orange-500" />
            <b>{st.waypoints?.toLocaleString()}</b> waypoints · sols {st.first_sol}–{st.last_sol} · <b>{st.total_km} km</b> driven ·
            source {st.source}{st.provenance?.retrieved_at ? `, retrieved ${st.provenance.retrieved_at.slice(0, 10)}` : ""}
            <button className="btn-ghost ml-auto px-2.5 py-1 text-[11px]" onClick={download} disabled={downloading}>{downloading ? <Spinner className="h-3 w-3" /> : <Download className="h-3 w-3" />} Update</button>
          </div>
          <div className="grid gap-4 xl:grid-cols-[1.5fr_1fr]">
            <Glass className="p-3">
              <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
                <Filter className="h-3.5 w-3.5 text-ink-faint" />
                sols <input aria-label="First sol" className="input w-20 px-2 py-1 text-xs" type="number" value={solRange[0]} onChange={(e) => setSolRange([Number(e.target.value), solRange[1]])} />
                – <input aria-label="Last sol" className="input w-20 px-2 py-1 text-xs" type="number" value={solRange[1]} onChange={(e) => setSolRange([solRange[0], Number(e.target.value)])} />
                <select aria-label="Terrain filter" className="input px-2 py-1 text-xs" value={terrain} onChange={(e) => setTerrain(e.target.value)}>
                  <option value="all">any terrain</option>{Object.keys(TERRAIN_COLORS).map((c) => <option key={c} value={c}>mostly {c}</option>)}
                </select>
                <label className="flex items-center gap-1"><input type="checkbox" checked={reviewedOnly} onChange={(e) => setReviewedOnly(e.target.checked)} /> reviewed only</label>
                <form className="ml-auto flex items-center gap-1" onSubmit={(e) => {
                  e.preventDefault();
                  const s = Number(goto);
                  const k = w.sol.findIndex((x) => x >= s);
                  if (k >= 0) openWaypoint({ sol: w.sol[k], site: w.site[k], drive: w.drive[k] });
                }}>
                  <Crosshair className="h-3.5 w-3.5 text-ink-faint" /><input aria-label="Go to sol" className="input w-24 px-2 py-1 text-xs" placeholder="go to sol" value={goto} onChange={(e) => setGoto(e.target.value)} />
                </form>
              </div>
              <Plot data={fig.data} layout={fig.layout} height={560} onClick={(e) => {
                const p = e.points?.[0];
                const cd = p?.customdata as unknown as number[] | undefined;
                if (cd && cd.length >= 3) openWaypoint({ sol: cd[0], site: cd[1], drive: cd[2] });
              }} />
              <p className="mt-1 text-[11px] text-ink-faint">Click a waypoint. Marker shape = how precisely an analysed image is located: ● exact site/drive · ◆ within the site · ■ sol only. Purple ring = mask corrected by a reviewer.</p>
            </Glass>
            <Glass className="flex min-h-0 flex-col gap-3 p-4 xl:h-0 xl:min-h-full">
              {!sel && <div className="py-16 text-center text-sm text-ink-soft"><MapPinned className="mx-auto mb-2 h-8 w-8 text-orange-400" />Select a waypoint on the map</div>}
              {sel && (
                <>
                  <div>
                    <div className="kicker">Waypoint</div>
                    <div className="font-display text-xl font-bold">Sol {sel.sol} · site {sel.site} / drive {sel.drive}</div>
                  </div>
                  {!hasMarsModel && (
                    <Notice tone="info" title="No Mars terrain model installed yet">
                      <p>Images can be browsed now; segmenting them needs the terrain model. Train it here or on Data &amp; Training, or upload a GPU-trained model:</p>
                      <div className="flex flex-wrap items-start gap-2">
                        <JobButton task="mars.train_quick" module="mars" label="Quick run (pipeline check)" className="btn-primary px-3 py-1.5 text-xs" />
                        <MarsModelUpload onDone={() => models.reload()} />
                      </div>
                    </Notice>)}
                  {hasMarsModel && wp && wp.raw.images.length > 0 && (
                    <button className="btn-ghost px-3 py-1.5 text-xs" onClick={() => setBatch((b) => b + 1)}>
                      <ScanSearch className="h-3.5 w-3.5" /> Segment the first {Math.min(8, wp.raw.images.length)} images
                    </button>)}
                  {wpLoading && <div className="flex items-center gap-2 text-sm text-ink-soft"><Spinner /> Asking NASA's raw-image service for Navcam frames…</div>}
                  {wp?.raw_status === "source_unavailable" && <Notice tone="warn" title="Raw-image service unavailable">{wp.raw.error}</Notice>}
                  {wp?.raw_status === "error" && <ErrorNotice message={wp.raw.error ?? ""} />}
                  {wp && wp.raw.images.length === 0 && wp.raw_status === "no_images" && <p className="text-sm text-ink-soft"><ImageOff className="mr-1 inline h-4 w-4" />No full-frame Navcam images listed for sol {wp.sol}.</p>}
                  {wp && wp.raw.images.length > 0 && (
                    <div className="scroll-thin grid max-h-[70vh] min-h-[240px] flex-1 auto-rows-max grid-cols-2 gap-2 overflow-y-auto pr-1 sm:grid-cols-3 xl:max-h-none xl:grid-cols-2 2xl:grid-cols-3">
                      {wp.raw.images.slice(0, 24).map((img, k) => (
                        <div key={img.imageid} className="glass-inset space-y-1 p-1.5">
                          <button type="button" className="block w-full overflow-hidden rounded-lg" onClick={() => setZoom(img.imageid)} title="Open full size">
                            <img src={`/api/mars/traverse/raw/${img.imageid}`} loading="lazy" alt={`Navcam ${img.imageid}`} className="aspect-[4/3] w-full bg-slate-800 object-cover transition hover:scale-[1.03]"
                              onError={(e) => { (e.target as HTMLImageElement).style.opacity = "0.15"; (e.target as HTMLImageElement).title = "NASA image server did not respond - try again later"; }} />
                          </button>
                          <div className="flex items-center gap-1"><span className={`chip px-1.5 text-[9px] ${PRECISION[img.match].cls}`}>{PRECISION[img.match].label}</span></div>
                          <div className="truncate font-mono text-[9px] text-ink-faint" title={img.imageid}>{img.lmst?.replace("Sol-", "") ?? img.date_taken?.slice(0, 16)}</div>
                          {hasMarsModel && <AnalyzeImage img={img} onDone={reload} autoStart={batch > 0 && k < 8 ? batch : 0} />}
                        </div>))}
                    </div>)}
                  {!!wp?.analysed.length && (
                    <div>
                      <div className="label mb-1">Already analysed here</div>
                      {wp.analysed.map((a) => (
                        <Link key={a.run_id} to={`/mars?run=${a.run_id}`} className="block rounded-xl px-2 py-1.5 hover:bg-white/60 dark:hover:bg-white/5">
                          <div className="flex items-center gap-2 text-xs font-semibold">{a.label} <OriginBadge origin={a.origin} />{a.corrected && <span className="chip bg-violet-100 text-violet-700">corrected</span>}</div>
                          {a.class_shares && <Shares s={a.class_shares} />}
                        </Link>))}
                    </div>)}
                </>)}
            </Glass>
          </div>

          <Glass className="p-4">
            <div className="mb-2 flex flex-wrap items-center gap-2">
              <div className="kicker">Terrain along the traverse · by distance driven</div>
              <div className="ml-auto"><Segmented value={sectionM} onChange={setSectionM} options={[{ value: "500", label: "0.5 km" }, { value: "1000", label: "1 km" }, { value: "2000", label: "2 km" }, { value: "5000", label: "5 km" }]} /></div>
            </div>
            {sections.data && sections.data.sections.length === 0 && (
              <p className="py-6 text-center text-sm text-ink-soft">No located terrain predictions match the filters yet - segment a few images at different waypoints to compare sections.</p>)}
            {sections.data && sections.data.sections.length > 0 && (
              <>
                <Plot height={300} data={Object.keys(TERRAIN_COLORS).map((c) => ({
                  x: sections.data!.sections.map((s) => `${s.from_km}–${s.to_km} km`), y: sections.data!.sections.map((s) => s.shares[c] ?? 0),
                  type: "bar", name: c, marker: { color: rgb(c) }, hovertemplate: `${c} %{y:.0%}<extra></extra>`,
                }))} layout={{ barmode: "stack", yaxis: { tickformat: ".0%", range: [0, 1] }, margin: { l: 50, r: 10, t: 30, b: 50 } }} />
                <div className="mt-1 flex flex-wrap gap-3 text-[11px] text-ink-soft">
                  {sections.data.sections.map((s) => (
                    <span key={s.index}>{s.from_km}–{s.to_km} km: {s.images} image(s){s.sol_range ? `, sols ${s.sol_range[0]}–${s.sol_range[1]}` : ""}, {s.reviewed} reviewed, {s.corrected} corrected</span>))}
                </div>
              </>)}
            <p className="mt-2 text-[11px] text-ink-faint">Shares are averaged per section from the model's masks ({num(points.length)} located image(s) after filters). Reviewed masks are human-checked; corrected ones replace the prediction in the Review Studio export.</p>
          </Glass>
        </div>)}
      {zoom && (
        <div role="dialog" aria-modal="true" aria-label={`Navcam image ${zoom}`} onClick={() => setZoom(null)}
          className="fixed inset-0 z-[80] flex items-center justify-center bg-slate-950/85 p-4 backdrop-blur-sm">
          <figure className="relative max-h-full max-w-5xl" onClick={(e) => e.stopPropagation()}>
            <img src={`/api/mars/traverse/raw/${zoom}?thumb=false`} alt={`Navcam ${zoom}`} className="max-h-[85vh] w-auto rounded-2xl object-contain shadow-2xl" />
            <figcaption className="mt-2 text-center font-mono text-xs text-slate-300">{zoom} · NASA/JPL-Caltech · Esc to close</figcaption>
            <button autoFocus type="button" aria-label="Close image" onClick={() => setZoom(null)}
              className="absolute right-2 top-2 grid h-9 w-9 place-items-center rounded-full bg-slate-900/80 text-white hover:bg-slate-800"><X className="h-4 w-4" /></button>
          </figure>
        </div>)}
    </Page>
  );
}
