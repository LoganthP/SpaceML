import { Clock3, Crosshair, ExternalLink, History, ImageOff, Radio, RefreshCw, Telescope } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../../lib/api";
import { Glass, Segmented, Spinner } from "../ui";

interface Obs {
  id: string; target: string; title: string; program: string | null; category: string | null; instruments: string[];
  pi: string | null; start: string | null; end: string | null; ra: number | null; dec: number | null;
  next_start: string | null; url: string; phase: "in_progress" | "upcoming" | "completed" | "unknown";
}
interface Live { name: string; observation: Obs; fetched_at: string; source: "live" | "cached"; notice: string | null; history: Obs[] }

const PHASE: Record<Obs["phase"], { label: string; cls: string }> = {
  in_progress: { label: "Observing now", cls: "bg-emerald-100 text-emerald-800 dark:bg-emerald-500/15 dark:text-emerald-200" },
  upcoming: { label: "Up next", cls: "bg-sky-100 text-sky-800 dark:bg-sky-500/15 dark:text-sky-200" },
  completed: { label: "Last observation", cls: "bg-slate-100 text-slate-700 dark:bg-white/10 dark:text-slate-200" },
  unknown: { label: "Scheduled", cls: "bg-slate-100 text-slate-700 dark:bg-white/10 dark:text-slate-200" },
};
type Fov = "0.1" | "0.5" | "2";
const FOV: { value: Fov; label: string }[] = [{ value: "0.1", label: "Close-up 0.1°" }, { value: "0.5", label: "Field 0.5°" }, { value: "2", label: "Wide 2°" }];

const utc = (s: string | null) => s ? `${s.slice(0, 16).replace("T", " ")} UTC` : "—";
const local = (s: string | null) => s ? new Date(s).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : "";
function dur(ms: number) {
  const m = Math.max(0, Math.round(ms / 60000));
  return m >= 60 ? `${Math.floor(m / 60)} h ${m % 60} min` : `${m} min`;
}
const ago = (s: string) => { const m = Math.round((Date.now() - new Date(s).getTime()) / 60000); return m < 1 ? "just now" : m < 60 ? `${m} min ago` : `${Math.round(m / 60)} h ago`; };
function radec(ra: number | null, dec: number | null) {
  if (ra == null || dec == null) return "—";
  const h = ra / 15, hh = Math.floor(h), mm = Math.floor((h - hh) * 60), ss = ((h - hh) * 60 - mm) * 60;
  const sg = dec < 0 ? "−" : "+", d = Math.abs(dec), dd = Math.floor(d), dm = Math.floor((d - dd) * 60), ds = Math.round(((d - dd) * 60 - dm) * 60);
  return `${hh}h ${mm}m ${ss.toFixed(0)}s · ${sg}${dd}° ${dm}′ ${ds}″`;
}

/** What Hubble or Webb is observing, drawn inside the app from Space Telescope Live's public data (the site itself
 *  cannot be embedded). Refreshes every minute while the page is open; offline it shows the last known observation. */
export function TelescopeLive({ mission }: { mission: "hubble" | "webb" }) {
  const [data, setData] = useState<Live | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [fov, setFov] = useState<Fov>("0.5");
  const [imgOk, setImgOk] = useState(true);
  const [, tick] = useState(0);
  const telescope = mission === "hubble" ? "Hubble" : "Webb";

  const load = async (refresh = false) => {
    setBusy(true);
    try { setData(await api<Live>(`/api/live/${mission}${refresh ? "?refresh=true" : ""}`)); setErr(null); }
    catch (e) { setErr((e as Error).message); }
    finally { setBusy(false); }
  };
  useEffect(() => {
    void load();
    const i = window.setInterval(() => { if (document.visibilityState === "visible") void load(); }, 60_000);
    const t = window.setInterval(() => tick((n) => n + 1), 30_000);          // progress bar and "x min ago"
    return () => { window.clearInterval(i); window.clearInterval(t); };
  }, [mission]); // eslint-disable-line react-hooks/exhaustive-deps

  const o = data?.observation;
  useEffect(() => setImgOk(true), [o?.id, fov]);
  const a = o?.start ? new Date(o.start).getTime() : null, b = o?.end ? new Date(o.end).getTime() : null;
  const now = Date.now();
  const frac = a && b && b > a ? Math.min(1, Math.max(0, (now - a) / (b - a))) : null;

  return (
    <Glass className="p-4">
      <div className="flex flex-wrap items-center gap-2">
        <Radio className="h-5 w-5 shrink-0 text-indigo-600 dark:text-indigo-300" aria-hidden="true" />
        <h3 className="font-semibold">What is {telescope} observing right now?</h3>
        {data && <span className={`chip text-[11px] ${data.source === "live" ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-500/15 dark:text-emerald-200" : "bg-amber-100 text-amber-900 dark:bg-amber-500/15 dark:text-amber-100"}`}>
          {data.source === "live" ? <><span className="mr-1 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-500" />Live · updated {ago(data.fetched_at)}</> : `Offline · last known, fetched ${ago(data.fetched_at)}`}
        </span>}
        <button className="btn-ghost ml-auto px-2.5 py-1 text-xs" onClick={() => load(true)} disabled={busy} aria-label="Refresh live observation">
          {busy ? <Spinner className="h-3.5 w-3.5" /> : <RefreshCw className="h-3.5 w-3.5" aria-hidden="true" />} Refresh
        </button>
      </div>

      {!data && !err && <div className="flex items-center gap-2 py-8 text-sm text-ink-soft" role="status"><Spinner /> Asking Space Telescope Live…</div>}
      {!data && err && (
        <p role="status" className="mt-3 rounded-xl bg-slate-100 p-3 text-sm text-ink-soft dark:bg-white/5">{err}</p>)}
      {data && !o && data.notice && <p role="status" className="mt-3 rounded-xl bg-slate-100 p-3 text-sm text-ink-soft dark:bg-white/5">{data.notice}</p>}

      {o && (
        <div className="mt-3 grid gap-4 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
          <figure className="min-w-0">
            <div className="relative aspect-[3/2] overflow-hidden rounded-2xl bg-slate-950">
              {o.ra != null && imgOk ? (
                <img key={`${o.id}-${fov}`} src={`/api/live/${mission}/sky/${encodeURIComponent(o.id)}?fov=${fov}`} onError={() => setImgOk(false)}
                  alt={`Sky survey image around ${o.target}`} className="h-full w-full object-cover" />
              ) : (
                <div className="flex h-full flex-col items-center justify-center gap-2 text-center text-xs text-slate-400">
                  <ImageOff className="h-6 w-6" aria-hidden="true" />{o.ra == null ? "No sky coordinates for this observation." : "Sky image not available right now."}
                </div>)}
              {o.ra != null && imgOk && <Crosshair className="pointer-events-none absolute left-1/2 top-1/2 h-10 w-10 -translate-x-1/2 -translate-y-1/2 text-cyan-300/90 drop-shadow" aria-hidden="true" />}
              <span className="absolute left-2 top-2 rounded-lg bg-slate-950/70 px-2 py-0.5 font-mono text-[10px] text-slate-200">{radec(o.ra, o.dec)}</span>
            </div>
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <Segmented label="Field of view" value={fov} onChange={setFov} options={FOV} />
              <figcaption className="text-[10px] leading-tight text-ink-faint">Ground-based DSS2 survey image of the target region (CDS hips2fits).
                {" "}{telescope}'s own images are not released in real time.</figcaption>
            </div>
          </figure>

          <div className="min-w-0 space-y-3">
            <div>
              <span className={`chip text-[11px] ${PHASE[o.phase].cls}`}>{PHASE[o.phase].label}</span>
              <div className="mt-1 flex items-start gap-2">
                <Telescope className="mt-1 h-5 w-5 shrink-0 text-violet-500" aria-hidden="true" />
                <div className="min-w-0">
                  <div className="font-display text-2xl font-bold leading-tight">{o.target}</div>
                  <div className="text-sm text-ink-soft">{o.title}</div>
                </div>
              </div>
            </div>
            {frac != null && (
              <div>
                <div className="h-2 overflow-hidden rounded-full bg-slate-200 dark:bg-white/10" role="progressbar" aria-label="Observation progress"
                  aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(frac * 100)}>
                  <div className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-violet-500" style={{ width: `${frac * 100}%` }} />
                </div>
                <div className="mt-1 flex justify-between text-[11px] text-ink-soft">
                  <span>{o.phase === "in_progress" && b ? `${dur(b - now)} left` : o.phase === "upcoming" && a ? `starts in ${dur(a - now)}` : `${Math.round(frac * 100)}%`}</span>
                  <span>{a && b ? `${dur(b - a)} total` : ""}</span>
                </div>
              </div>)}
            <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1.5 text-xs">
              <dt className="text-ink-faint">Start</dt><dd title={local(o.start)}>{utc(o.start)}</dd>
              <dt className="text-ink-faint">End</dt><dd title={local(o.end)}>{utc(o.end)}</dd>
              {o.instruments.length > 0 && <><dt className="text-ink-faint">Instruments</dt>
                <dd className="flex flex-wrap gap-1">{o.instruments.map((i) => <span key={i} className="chip bg-violet-100 px-1.5 text-[10px] text-violet-800 dark:bg-violet-500/15 dark:text-violet-200">{i}</span>)}</dd></>}
              {o.category && <><dt className="text-ink-faint">Science</dt><dd>{o.category}</dd></>}
              {o.program && <><dt className="text-ink-faint">Program</dt><dd>{o.program}{o.pi ? ` · PI ${o.pi}` : ""}</dd></>}
              {o.next_start && <><dt className="text-ink-faint">Next</dt><dd className="flex items-center gap-1"><Clock3 className="h-3 w-3" aria-hidden="true" />{utc(o.next_start)}</dd></>}
            </dl>
            {data?.notice && <p role="status" className="rounded-xl bg-amber-50 px-3 py-2 text-[11px] text-amber-900 dark:bg-amber-500/10 dark:text-amber-100">{data.notice}. Showing the last observation fetched; it refreshes by itself when the connection is back.</p>}
            <a href={o.url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-[11px] font-semibold text-violet-700 hover:underline dark:text-violet-300">
              Full schedule on Space Telescope Live <ExternalLink className="h-3 w-3" aria-hidden="true" /><span className="sr-only">(opens in a new tab)</span>
            </a>
          </div>
        </div>)}

      {!!data?.history.length && (
        <details className="mt-3 border-t border-slate-200/70 pt-2 dark:border-white/10">
          <summary className="flex cursor-pointer items-center gap-1.5 text-xs font-semibold"><History className="h-3.5 w-3.5" aria-hidden="true" /> Earlier observations seen by this app ({data.history.length})</summary>
          <ul className="mt-2 grid gap-1.5 sm:grid-cols-2 xl:grid-cols-3">
            {data.history.map((h) => (
              <li key={h.id} className="glass-inset min-w-0 px-2.5 py-1.5 text-xs">
                <div className="truncate font-semibold" title={h.target}>{h.target}</div>
                <div className="truncate text-[10px] text-ink-faint" title={h.title}>{utc(h.start)} · {h.instruments.join(", ") || h.title}</div>
              </li>))}
          </ul>
        </details>)}
      <p className="mt-2 text-[10px] text-ink-faint">Data: NASA / STScI Space Telescope Live. Separate from the archived pointing data above, which works offline.</p>
    </Glass>
  );
}
