import { WriteButton, useCapabilities } from "../lib/capabilities";
import { EvidencePanel } from "../components/Evidence";
import { Database, FlaskConical, Globe2, Orbit, Search, SlidersHorizontal, Sparkles, Star, Trash2, UploadCloud } from "lucide-react";
import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { activeJob, ProgressBar, useJobs, useOnJobFinished } from "../components/Jobs";
import { ModuleStatus } from "../components/Catalog";
import Plot from "../components/LazyPlot";
import { DropZone, ReviewToggle, useReviews } from "../components/RunTools";
import type { SkyStar } from "../components/SkyScene";
import { ErrorNotice, Glass, IconBadge, Kpi, Notice, Page, RunProgress, Segmented, Spinner } from "../components/ui";
import { api, createRun, useApi, useRun } from "../lib/api";
import { num } from "../lib/format";
import type { ExoplanetResult, Featured, PredRow, SkyCatalog, Tce, UploadDataset, UploadTarget } from "../lib/types";

const SystemScene = lazy(() => import("../components/SystemScene"));
const SkyScene = lazy(() => import("../components/SkyScene"));
const HelioView = lazy(() => import("../components/HelioView"));

const LABEL_STYLE: Record<string, string> = {
  PC: "bg-emerald-100 text-emerald-700", AFP: "bg-rose-100 text-rose-700", NTP: "bg-slate-200 text-slate-700", UNK: "bg-slate-100 text-slate-500",
};
const LABEL_NAME: Record<string, string> = {
  PC: "Planet candidate", AFP: "Astrophysical false positive", NTP: "Non-transiting phenomenon", UNK: "Unlabeled",
};

function ScoreRing({ value }: { value: number | null }) {
  const r = 52, c = 2 * Math.PI * r;
  return (
    <div className="relative h-36 w-36 shrink-0">
      <svg viewBox="0 0 128 128" className="h-full w-full -rotate-90">
        <defs><linearGradient id="ring" x1="0" x2="1"><stop offset="0" stopColor="#8b5cf6" /><stop offset="1" stopColor="#ec4899" /></linearGradient></defs>
        <circle cx="64" cy="64" r={r} fill="none" stroke="rgba(255,255,255,0.9)" strokeWidth="12" />
        {value != null && <circle cx="64" cy="64" r={r} fill="none" stroke="url(#ring)" strokeWidth="12" strokeLinecap="round"
          strokeDasharray={c} strokeDashoffset={c * (1 - value)} style={{ transition: "stroke-dashoffset 900ms cubic-bezier(.2,.8,.2,1)" }} />}
      </svg>
      <div className="absolute inset-0 grid place-items-center text-center">
        <div>
          <div className="font-display text-3xl font-bold text-exo-700">{value == null ? "—" : value.toFixed(2)}</div>
          <div className="text-[10px] font-semibold uppercase tracking-wider text-ink-faint">{value == null ? "no model yet" : "candidate score"}</div>
        </div>
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------- left: Kepler catalog */
function FeaturedPanel({ onPick }: { onPick: (k: number, p: number) => void }) {
  const { data } = useApi<Featured>("/api/exoplanet/featured");
  if (!data || (!data.kepler90.length && !data.test_planets.length && !data.false_positives_caught.length)) return null;
  const group = (title: string, rows: PredRow[]) => rows.length > 0 && (
    <div>
      <div className="label mb-1.5">{title}</div>
      <div className="grid gap-1.5">{rows.map((r) => (
        <button key={`${r.kepid}-${r.tce_plnt_num}`} onClick={() => onPick(r.kepid, r.tce_plnt_num)}
          className="glass-inset flex items-center justify-between gap-2 px-3 py-2 text-left text-xs transition hover:bg-white/80">
          <span className="font-semibold">KIC {r.kepid} · {r.tce_plnt_num}</span><span className="text-ink-soft">score {r.cnn.toFixed(2)}</span>
        </button>))}</div>
    </div>
  );
  return (
    <Glass className="space-y-4 p-4">
      {data.kepler90.length > 0 && (
        <div>
          <div className="label mb-1.5 flex items-center gap-1"><Star className="h-3 w-3" /> Kepler-90 system</div>
          <div className="flex flex-wrap gap-1.5">
            {data.kepler90.map((t) => (
              <button key={t.tce_plnt_num} onClick={() => onPick(t.kepid, t.tce_plnt_num)} className="chip bg-white/80 text-exo-700 hover:bg-white">
                TCE {t.tce_plnt_num} · {t.period_days.toFixed(1)} d
              </button>))}
          </div>
        </div>
      )}
      {group("Held-out planets", data.test_planets)}
      {group("False positives caught", data.false_positives_caught)}
      {group("False positives missed", data.false_positives_missed)}
    </Glass>
  );
}

function CatalogPanel({ busy, onPick }: { busy: boolean; onPick: (k: number, p: number, cached?: boolean) => void }) {
  const [q, setQ] = useState("");
  const [label, setLabel] = useState<"all" | "PC" | "AFP" | "NTP">("all");
  const [cachedOnly, setCachedOnly] = useState(true);
  const qs = new URLSearchParams({ q, limit: "60", cached_only: String(cachedOnly), ...(label !== "all" ? { label } : {}) });
  const tces = useApi<Tce[]>(`/api/exoplanet/tces?${qs}`, []);
  useOnJobFinished(tces.reload);
  useEffect(() => { if (tces.data && tces.data.length === 0 && cachedOnly && !q && label === "all") setCachedOnly(false); }, [tces.data]); // eslint-disable-line
  return (
    <>
      <Glass className="space-y-3 p-4">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-ink-faint" />
          <input className="input pl-9" placeholder="Search Kepler ID (e.g. 11442793)" value={q} onChange={(e) => setQ(e.target.value)} />
        </div>
        <Segmented value={label} onChange={setLabel} options={[{ value: "all", label: "All" }, { value: "PC", label: "PC" }, { value: "AFP", label: "AFP" }, { value: "NTP", label: "NTP" }]} />
        <label className="flex cursor-pointer items-center gap-2 px-1 text-xs text-ink-soft">
          <input type="checkbox" checked={cachedOnly} onChange={(e) => setCachedOnly(e.target.checked)} className="accent-violet-600" />
          Only signals with a cached light curve (instant)
        </label>
        {tces.error && <ErrorNotice message={tces.error.message} title="Kepler catalog unavailable" />}
        {tces.loading && <div className="flex items-center gap-2 p-2 text-sm text-ink-soft"><Spinner /> Searching…</div>}
        <div className="scroll-thin max-h-[360px] space-y-1.5 overflow-y-auto pr-1">
          {tces.data?.map((t) => (
            <button key={`${t.kepid}-${t.tce_plnt_num}`} onClick={() => onPick(t.kepid, t.tce_plnt_num, t.cached)} disabled={busy}
              className="flex w-full items-center gap-3 rounded-2xl px-3 py-2 text-left transition hover:bg-white/70 disabled:opacity-60">
              <div className="min-w-0 flex-1">
                <div className="text-sm font-semibold">KIC {t.kepid} <span className="text-ink-faint">· TCE {t.tce_plnt_num}</span></div>
                <div className="text-[11px] text-ink-soft">P = {t.period_days.toFixed(2)} d · {t.duration_hours.toFixed(1)} h · {Math.round(t.depth_ppm)} ppm{t.cached ? "" : " · downloads from MAST"}</div>
              </div>
              <span className={`chip ${LABEL_STYLE[t.label] ?? LABEL_STYLE.UNK}`}>{t.label}</span>
            </button>))}
          {tces.data?.length === 0 && <p className="p-2 text-sm text-ink-soft">No signals match.</p>}
        </div>
      </Glass>
      <FeaturedPanel onPick={(k, p) => onPick(k, p)} />
    </>
  );
}

/* ---------------------------------------------------------------- left: uploads */
function TargetRow({ t, busy, onRun }: { t: UploadTarget; busy: boolean; onRun: (t: UploadTarget, eph?: { period: number; t0: number; duration_hours: number }) => void }) {
  const [open, setOpen] = useState(false);
  const [eph, setEph] = useState({ period: "", t0: "", duration_hours: "" });
  const ephOk = eph.period && eph.t0 && eph.duration_hours;
  return (
    <div className="glass-inset p-3">
      <div className="flex items-center gap-2">
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm font-semibold">{t.name}</div>
          <div className="text-[11px] text-ink-soft">
            {t.mission ? `${t.mission} · ` : ""}{t.n_points.toLocaleString()} pts · {t.span_days.toFixed(1)} d · {t.cadence_minutes.toFixed(1)} min cadence
          </div>
        </div>
        <button className="rounded-xl p-1.5 text-ink-faint hover:bg-white" title="Set the ephemeris yourself" onClick={() => setOpen((o) => !o)}>
          <SlidersHorizontal className="h-3.5 w-3.5" />
        </button>
        <button className="btn-primary px-3 py-1.5 text-xs" disabled={busy || (!t.searchable && !ephOk)}
          onClick={() => onRun(t, ephOk ? { period: +eph.period, t0: +eph.t0, duration_hours: +eph.duration_hours } : undefined)}>
          <Sparkles className="h-3.5 w-3.5" /> Analyse
        </button>
      </div>
      {t.note && <p className="mt-1.5 text-[11px] text-amber-700">{t.note}</p>}
      {open && (
        <div className="mt-2 grid grid-cols-3 gap-1.5">
          {(["period", "t0", "duration_hours"] as const).map((k) => (
            <input key={k} className="input px-2 py-1.5 text-xs" placeholder={k === "period" ? "Period (d)" : k === "t0" ? `T0 (${t.time_system})` : "Duration (h)"}
              value={eph[k]} onChange={(e) => setEph({ ...eph, [k]: e.target.value })} inputMode="decimal" />
          ))}
          <p className="col-span-3 text-[10px] text-ink-faint">Leave empty to search for the period automatically (box least squares).</p>
        </div>
      )}
    </div>
  );
}

function UploadPanel({ busy, onRun }: { busy: boolean; onRun: (ds: string, t: UploadTarget, eph?: { period: number; t0: number; duration_hours: number }) => void }) {
  const list = useApi<UploadDataset[]>("/api/exoplanet/datasets");
  const [msg, setMsg] = useState<{ tone: "error" | "info"; text: string; link?: string } | null>(null);
  return (
    <>
      <Glass className="space-y-3 p-4">
        <DropZone url="/api/workbench/datasets" fields={{ kind: "auto" }}
          title="Upload light curves"
          hint={<>FITS (Kepler, K2, TESS…), CSV/TSV/TXT/XLSX/JSON/Parquet with time + flux or mag, or a .zip/.tar.gz of many files · up to 1 GB</>}
          onError={(m) => setMsg({ tone: "error", text: m })}
          onDone={(r) => {
            const res = r as { kind: string; id: string; name: string };
            if (res.kind === "lightcurve") { setMsg(null); list.reload(); }
            else setMsg({ tone: "info", text: `“${res.name}” looks like ${res.kind === "images" ? "images" : "a general time series"}, not a light curve. It was added to the Workbench instead.`, link: `/workbench?dataset=${res.id}` });
          }} />
        {msg && (msg.tone === "error" ? <ErrorNotice title="Upload problem" message={msg.text} /> :
          <Notice tone="info" title={msg.text}>{msg.link && <Link className="font-semibold text-violet-700 underline" to={msg.link}>Open in Workbench</Link>}</Notice>)}
      </Glass>
      {list.loading && <div className="flex items-center gap-2 px-2 text-sm text-ink-soft"><Spinner /> Loading uploads…</div>}
      {list.data?.length === 0 && <p className="px-2 text-sm text-ink-soft">No uploads yet. Try a TESS or Kepler FITS file from MAST, or any CSV with time and flux columns.</p>}
      {list.data?.map((d) => (
        <Glass key={d.id} className="space-y-2 p-4">
          <div className="flex items-center gap-2">
            <Database className="h-4 w-4 text-violet-600" />
            <div className="min-w-0 flex-1">
              <div className="truncate text-sm font-bold">{d.filename}</div>
              <div className="text-[11px] text-ink-soft">{d.targets.length} target(s) · {d.n_files} file(s){d.skipped.length ? ` · ${d.skipped.length} skipped` : ""}</div>
            </div>
            <WriteButton className="rounded-xl p-1.5 text-ink-faint hover:bg-rose-50 hover:text-rose-600" title="Delete dataset"
              onClick={async () => { await api(`/api/exoplanet/datasets/${d.id}`, { method: "DELETE" }); list.reload(); }}>
              <Trash2 className="h-3.5 w-3.5" />
            </WriteButton>
          </div>
          {d.warnings.slice(0, 3).map((w) => <p key={w} className="text-[11px] text-amber-700">{w}</p>)}
          <div className="scroll-thin max-h-[300px] space-y-1.5 overflow-y-auto pr-1">
            {d.targets.map((t) => <TargetRow key={t.id} t={t} busy={busy} onRun={(tt, eph) => onRun(d.id, tt, eph)} />)}
          </div>
        </Glass>
      ))}
    </>
  );
}

/* ---------------------------------------------------------------- results */
function SpaceView({ r }: { r: ExoplanetResult }) {
  const [tab, setTab] = useState<"system" | "sky" | "observatory">("system");
  const sky = useApi<SkyCatalog>(tab === "sky" ? "/api/exoplanet/sky" : null);
  const selected = useMemo(() => r.sky ? { kepid: r.kepid, ra: r.sky.ra, dec: r.sky.dec, name: r.target_name } : null, [r]);
  const stars = useMemo(() => (sky.data?.stars ?? []) as SkyStar[], [sky.data]);
  const sel = r.system.planets.find((p) => p.selected) ?? r.system.planets[0];
  const st = r.system.star;
  const assumed = Object.entries({ ...st.assumed, ...(sel?.derived ?? {}) });
  return (
    <Glass className="p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div className="label flex items-center gap-1.5"><Globe2 className="h-3.5 w-3.5" /> Space view · {r.target_name}</div>
        <Segmented value={tab} onChange={setTab} options={[{ value: "system", label: "Planetary system 3D" }, { value: "sky", label: "Position on the sky" }, { value: "observatory", label: "Observatory" }]} />
      </div>
      <div className="grid gap-4 lg:grid-cols-[1fr_280px]">
        <Suspense fallback={<div className="flex h-[440px] items-center justify-center gap-2 text-sm text-ink-soft"><Spinner /> Loading 3D view…</div>}>
          {tab === "system" ? <SystemScene system={r.system} /> : tab === "observatory" ? <Observatory r={r} /> : (
            sky.loading ? <div className="flex h-[440px] items-center justify-center gap-2 text-sm text-ink-soft"><Spinner /> Loading sky catalog…</div>
              : stars.length || selected ? <SkyScene stars={stars} selected={selected} />
                : <div className="flex h-[440px] flex-col items-center justify-center gap-2 text-center text-sm text-ink-soft">
                  <p>No sky positions available{sky.data?.reason ? `: ${sky.data.reason}` : ""}.</p>
                  <p className="text-xs">Re-download the catalog on <Link to="/datasets" className="font-semibold text-violet-600">Data &amp; Training</Link> (it now includes RA/Dec), or upload a FITS file with RA_OBJ/DEC_OBJ in its header.</p>
                </div>
          )}
        </Suspense>
        <div className="space-y-3 text-sm">
          <div className="glass-inset p-3">
            <div className="label">Host star</div>
            <div className="mt-1 grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
              <span className="text-ink-soft">Temperature</span><span className="font-semibold">{Math.round(st.teff)} K</span>
              <span className="text-ink-soft">Radius</span><span className="font-semibold">{st.radius_rsun.toFixed(2)} R☉</span>
              <span className="text-ink-soft">Mass</span><span className="font-semibold">{st.mass_msun.toFixed(2)} M☉</span>
              <span className="text-ink-soft">Luminosity</span><span className="font-semibold">{st.luminosity_lsun.toFixed(2)} L☉</span>
              <span className="text-ink-soft">Habitable zone</span><span className="font-semibold">{st.hz_inner_au.toFixed(2)}–{st.hz_outer_au.toFixed(2)} AU</span>
              {r.sky && <><span className="text-ink-soft">RA / Dec</span><span className="font-semibold">{r.sky.ra.toFixed(3)}° / {r.sky.dec.toFixed(3)}°</span></>}
            </div>
          </div>
          {sel && (
            <div className="glass-inset p-3">
              <div className="label">This signal as a planet</div>
              <div className="mt-1 grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
                <span className="text-ink-soft">Radius</span><span className="font-semibold">{sel.rp_earth.toFixed(2)} R⊕ · {sel.size_class}</span>
                <span className="text-ink-soft">Orbit</span><span className="font-semibold">{sel.a_au.toFixed(3)} AU ({sel.a_rstar.toFixed(1)} R★)</span>
                <span className="text-ink-soft">Inclination</span><span className="font-semibold">{sel.inclination_deg.toFixed(1)}°</span>
                <span className="text-ink-soft">Equilibrium T</span><span className="font-semibold">{Math.round(sel.teq_k)} K</span>
                <span className="text-ink-soft">Insolation</span><span className="font-semibold">{num(sel.insolation_earth, 2)} × Earth</span>
                <span className="text-ink-soft">Habitable zone</span>
                <span className={`font-semibold ${sel.habitable_zone === "within" ? "text-emerald-600" : sel.habitable_zone === "inside" ? "text-orange-600" : "text-sky-600"}`}>
                  {sel.habitable_zone === "within" ? "inside the habitable zone" : sel.habitable_zone === "inside" ? "too hot (closer than the zone)" : "too cold (beyond the zone)"}</span>
              </div>
            </div>
          )}
          {assumed.length > 0 && (
            <p className="text-[11px] leading-snug text-ink-faint">
              Derived values: {assumed.map(([k, v]) => `${k} ${v}`).join("; ")}. Positions use Kepler's third law; treat them as estimates.
            </p>
          )}
          {r.system.planets.length > 1 && <p className="text-[11px] text-ink-soft">{r.system.planets.length} signals on this star are shown together; the analysed one is highlighted.</p>}
        </div>
      </div>
    </Glass>
  );
}

function Observatory({ r }: { r: ExoplanetResult }) {
  const mission = (r.source === "kepler" ? "Kepler" : r.mission ?? "").toUpperCase();
  const tess = mission.includes("TESS");
  const kepler = mission.includes("KEPLER") || mission.includes("K2");
  const date = tess ? "2019-01-15" : kepler ? "2011-06-01" : null;
  const target = r.sky ? { ra: r.sky.ra, dec: r.sky.dec, label: r.target_name } : null;
  return (
    <HelioView date={date} show={{ kepler: kepler || (!tess && !!target), tess }} target={target} height={440}
      caption={tess ? "TESS orbits Earth every 13.7 days and observes 24°x96° sectors; each sector is watched for ~27 days."
        : kepler ? "Kepler trailed Earth around the Sun and stared at one fixed field in Cygnus-Lyra for four years (2009-2013); the cone points at this star."
          : "Unknown observatory: the view shows the line of sight from the inner Solar System toward this star."} />
  );
}

function Diagnostics({ r }: { r: ExoplanetResult }) {
  const d = r.diagnostics;
  return (
    <div className="grid gap-4 2xl:grid-cols-2">
      {d && (
        <Glass className="p-4">
          <div className="label">Vetting checks</div>
          <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-3">
            <Kpi label="Odd depth" value={d.odd_depth_ppm == null ? "—" : `${Math.round(d.odd_depth_ppm)} ppm`} />
            <Kpi label="Even depth" value={d.even_depth_ppm == null ? "—" : `${Math.round(d.even_depth_ppm)} ppm`}
              sub={d.odd_even_sigma == null ? undefined : `odd/even differ by ${d.odd_even_sigma.toFixed(1)}σ`} accent={d.odd_even_sigma != null && d.odd_even_sigma > 3 ? "#e11d48" : undefined} />
            <Kpi label="Secondary eclipse" value={d.secondary_depth_ppm == null ? "—" : `${Math.round(d.secondary_depth_ppm)} ppm`}
              sub={d.secondary_sigma == null ? undefined : `${d.secondary_sigma.toFixed(1)}σ at phase 0.5`} accent={d.secondary_sigma != null && d.secondary_sigma > 3 ? "#e11d48" : undefined} />
          </div>
          <p className="mt-2 text-[11px] text-ink-soft">Different odd/even depths or a secondary eclipse (above 3σ, shown in red) point to an eclipsing binary rather than a planet.</p>
        </Glass>
      )}
      {r.bls && (
        <Glass className="p-4">
          <div className="label">Period search (box least squares)</div>
          <Plot height={220} data={[
            { x: r.bls.periodogram.period, y: r.bls.periodogram.power, type: "scattergl", mode: "lines", line: { color: "#7c3aed", width: 1.2 }, name: "BLS power" },
            { x: [r.bls.best.period_days], y: [r.bls.best.power], type: "scatter", mode: "markers", marker: { color: "#db2777", size: 10 }, name: "best" },
          ]} layout={{ xaxis: { title: { text: "Period (days)" }, type: "log" }, yaxis: { title: { text: "Power" } }, showlegend: false }} />
          <div className="flex flex-wrap gap-1.5 text-[11px]">
            {r.bls.alternatives.slice(0, 4).map((a) => <span key={a.period_days} className="chip bg-white/70 text-ink-soft">{a.period_days.toFixed(4)} d · {a.snr > 0 ? `SNR ${a.snr.toFixed(1)}` : `power ${a.power.toFixed(1)}`}</span>)}
          </div>
        </Glass>
      )}
      {r.raw_preview && (
        <Glass className="p-4 2xl:col-span-2">
          <div className="label">Uploaded light curve (normalised, stitched)</div>
          <Plot height={200} data={[{ x: r.raw_preview.x, y: r.raw_preview.y, type: "scattergl", mode: "markers", marker: { size: 2, color: "rgba(100,116,139,0.55)" } }]}
            layout={{ xaxis: { title: { text: `Time (${r.data_summary?.time_system ?? "days"})` } }, yaxis: { title: { text: "Relative flux" } } }} />
        </Glass>
      )}
    </div>
  );
}

function ViewPlots({ r }: { r: ExoplanetResult }) {
  return (
    <div className="grid gap-4 2xl:grid-cols-2">
      <Glass className="p-4">
        <div className="label px-1">Global view · full orbit (2,001 bins)</div>
        <Plot height={280} data={[
          { x: r.scatter_global.x, y: r.scatter_global.y, type: "scattergl", mode: "markers", name: "Flattened flux", marker: { size: 2.5, color: "rgba(100,116,139,0.35)" }, yaxis: "y2" },
          { x: r.global_view.x, y: r.global_view.y, type: "scattergl", mode: "lines", name: "Binned view (model input)", line: { color: "#7c3aed", width: 1.6 } },
        ]} layout={{ xaxis: { title: { text: "Orbital phase" } }, yaxis: { title: { text: "Normalised" } },
          yaxis2: { overlaying: "y", side: "right", showgrid: false, showticklabels: false } } as Partial<Plotly.Layout>} />
      </Glass>
      <Glass className="p-4">
        <div className="label px-1">Local view · transit close-up (201 bins)</div>
        <Plot height={280} data={[
          { x: r.scatter_local.x, y: r.scatter_local.y, type: "scattergl", mode: "markers", name: "Flattened flux", marker: { size: 3, color: "rgba(100,116,139,0.4)" }, yaxis: "y2" },
          { x: r.local_view.x, y: r.local_view.y, type: "scatter", mode: "lines", name: "Binned view (model input)", line: { color: "#db2777", width: 2 } },
        ]} layout={{ xaxis: { title: { text: "Hours from mid-transit" } }, yaxis: { title: { text: "Normalised" } },
          yaxis2: { overlaying: "y", side: "right", showgrid: false, showticklabels: false } } as Partial<Plotly.Layout>} />
      </Glass>
    </div>
  );
}

function NoModelHint() {
  const { jobs, start } = useJobs();
  const job = activeJob(jobs, "exoplanet");
  return (
    <div className="glass-inset space-y-2 p-3 text-xs">
      <p className="font-semibold">No exoplanet CNN trained yet - views, diagnostics and the 3D system still work.</p>
      {job ? (<><div className="text-ink-soft">{job.title}: {job.step ?? job.status}</div><ProgressBar value={job.progress} /></>) : (
        <div className="flex flex-wrap gap-1.5">
          <WriteButton className="btn-primary px-3 py-1.5 text-xs" onClick={() => start("exoplanet.setup", { n_stars: 300 })}><FlaskConical className="h-3.5 w-3.5" /> Download 300 stars + train</WriteButton>
          <WriteButton className="btn-ghost px-3 py-1.5 text-xs" onClick={() => start("exoplanet.train", {})}>Train on cached stars</WriteButton>
        </div>
      )}
    </div>
  );
}

function ResultHeader({ r, runId }: { r: ExoplanetResult; runId: string }) {
  const { reviews, set } = useReviews(runId);
  const title = r.source === "kepler" ? `KIC ${r.kepid} · TCE ${r.tce_plnt_num}` : r.target_name;
  return (
    <Glass className="space-y-4 p-5">
      <div className="flex flex-wrap items-center gap-6">
        <ScoreRing value={r.candidate_score} />
        <div className="min-w-[240px] flex-1 space-y-2">
          <div className="label">{r.source === "kepler" ? "Kepler signal" : `Uploaded · ${r.dataset_name ?? ""}`}</div>
          <div className="font-display text-2xl font-bold">{title}</div>
          {r.name && <div className="text-sm text-exo-700">{r.name}</div>}
          <div className="flex flex-wrap gap-2">
            {r.catalog_label && <span className={`chip ${LABEL_STYLE[r.catalog_label] ?? LABEL_STYLE.UNK}`}>Catalog: {LABEL_NAME[r.catalog_label] ?? r.catalog_label}</span>}
            {r.model_available && (
              <span className={`chip ${r.split === "test" || r.split === "unseen" ? "bg-sky-100 text-sky-700" : "bg-amber-100 text-amber-800"}`}>
                {r.split === "test" ? "Held-out test star" : r.split === "unseen" ? "Not used in training" : `Star in ${r.split} split`}
              </span>)}
            <span className="chip bg-white/80 text-ink-soft">Ephemeris: {r.ephemeris_source}</span>
          </div>
          <p className="text-xs text-ink-soft">A candidate score ranks how planet-like the transit shape looks. It is not a confirmation.</p>
          <div className="flex flex-wrap items-center gap-3">
            <span className="flex items-center gap-2 text-xs text-ink-soft">Your verdict <ReviewToggle item="candidate" review={reviews.candidate} onSet={set} /></span>
            <EvidencePanel runId={runId} compact />
          </div>
        </div>
      </div>
      {!r.model_available && <NoModelHint />}
      <div className="grid grid-cols-2 gap-2 md:grid-cols-5">
        <Kpi label="Baseline score" value={r.baseline_score == null ? "—" : r.baseline_score.toFixed(2)} sub="period/duration/depth" />
        <Kpi label="Period" value={`${num(r.tce.period_days, 3)} d`} />
        <Kpi label="Duration" value={`${num(r.tce.duration_hours, 2)} h`} />
        <Kpi label="Depth" value={`${Math.round(r.tce.depth_ppm)} ppm`} sub={`measured ${Math.round(r.measured_depth_ppm)}`} />
        <Kpi label="Transits folded" value={`~${r.n_transits}`} sub={`${r.n_points.toLocaleString()} points`} />
      </div>
    </Glass>
  );
}

/* ---------------------------------------------------------------- page */
export default function Exoplanet() {
  const caps = useCapabilities();
  const [params, setParams] = useSearchParams();
  const runId = params.get("run");
  const [source, setSource] = useState<"kepler" | "upload">(params.get("source") === "upload" ? "upload" : "kepler");
  const [pendingDownload, setPendingDownload] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const { run, error: runError } = useRun<ExoplanetResult>(runId);

  async function submit(input: Record<string, unknown>) {
    setSubmitError(null);
    if (!caps?.capabilities.write) { setSubmitError(caps?.hint ?? "Sign in or connect Vercel compute to analyze this signal"); return; }
    try {
      const r = await createRun("exoplanet", input);
      setParams({ run: r.id, ...(source === "upload" ? { source: "upload" } : {}) });
    } catch (e) {
      setSubmitError((e as Error).message);
    }
  }
  const kicParam = params.get("kic");
  useEffect(() => {  // deep link from the command palette: /exoplanet?kic=11442793
    if (caps?.capabilities.write && kicParam && /^\d{5,9}$/.test(kicParam) && !runId) submit({ kepid: Number(kicParam), tce_plnt_num: Number(params.get("tce") ?? 1) });
  }, [kicParam, caps?.capabilities.write]); // eslint-disable-line react-hooks/exhaustive-deps
  const res = run?.status === "succeeded" ? run.result : undefined;
  const busy = !!run && (run.status === "queued" || run.status === "running");

  return (
    <Page title="Exoplanet Finder"
      subtitle="Search Kepler signals or upload your own light curves. Each signal is flattened, folded and binned into a full-orbit view and a transit close-up, scored by a two-branch 1D CNN against a period/duration/depth baseline, vetted, and placed in 3D."
      icon={<IconBadge gradient="from-violet-500 to-fuchsia-500"><Orbit className="h-6 w-6" /></IconBadge>}>
      <ModuleStatus ids={["kepler"]} />
      <div className="grid gap-4 xl:grid-cols-[360px_1fr]">
        <div className="space-y-4">
          <Segmented value={source} onChange={setSource} options={[
            { value: "kepler", label: <span className="flex items-center gap-1.5"><Star className="h-3.5 w-3.5" /> Kepler catalog</span> },
            { value: "upload", label: <span className="flex items-center gap-1.5"><UploadCloud className="h-3.5 w-3.5" /> My observations</span> }]} />
          {source === "kepler"
            ? <CatalogPanel busy={busy} onPick={(kepid, plnt, cached) => { setPendingDownload(cached === false); submit({ kepid, tce_plnt_num: plnt }); }} />
            : <UploadPanel busy={busy} onRun={(ds, t, eph) => { setPendingDownload(false); submit({ dataset_id: ds, target_id: t.id, ...(eph ?? {}) }); }} />}
        </div>

        <div className="min-w-0 space-y-4">
          {submitError && <ErrorNotice message={submitError} />}
          {runError && <ErrorNotice message={runError} />}
          {busy && (
            <div className="space-y-2">
              <RunProgress status={run!.status} label={run!.input_label} />
              {pendingDownload && <Notice tone="info" title="First look at this star: downloading every Kepler quarter from MAST, then flattening and folding. This can take a minute; the result is cached." />}
            </div>
          )}
          {run?.status === "failed" && <ErrorNotice title="Run failed" message={run.error ?? "Unknown error"} />}
          {!run && !submitError && (
            <Glass className="flex flex-col items-center gap-3 py-16 text-center">
              <Orbit className="h-10 w-10 text-exo-500" />
              <p className="font-semibold">Choose a Kepler signal or upload observations</p>
              <p className="max-w-md text-sm text-ink-soft">Results include the candidate score, folded views, eclipsing-binary checks, a 3D model of the planetary system and the star's position on the sky.</p>
            </Glass>
          )}
          {res && run && (
            <>
              <ResultHeader r={res} runId={run.id} />
              {res.model_available && (res.split === "train" || res.split === "val") && (
                <Notice title="This star was used during training">Its score is not an independent estimate. Use held-out test stars to judge the model.</Notice>
              )}
              <SpaceView r={res} />
              <ViewPlots r={res} />
              <Diagnostics r={res} />
              <p className="px-1 text-[11px] text-ink-faint">Model {res.model_version ?? "none"} · eclipsing binaries often show a V-shaped dip, odd/even depth differences or a secondary eclipse at phase 0.5.</p>
            </>
          )}
        </div>
      </div>
    </Page>
  );
}
