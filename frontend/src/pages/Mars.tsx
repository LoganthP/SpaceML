import { FileInput, useCapabilities } from "../lib/capabilities";
import { EvidencePanel } from "../components/Evidence";
import { ImagePlus, Mountain, Paintbrush, Upload } from "lucide-react";
import { lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";
import type { GlobeMarker } from "../components/ContextGlobe";

const ContextGlobe = lazy(() => import("../components/ContextGlobe"));
const MaskEditor = lazy(() => import("../components/MaskEditor"));
import { useSearchParams } from "react-router-dom";
import { JobButton, useOnJobFinished } from "../components/Jobs";
import { MarsModelUpload } from "../components/Setup";
import { ModuleStatus } from "../components/Catalog";
import { ErrorNotice, Glass, IconBadge, Notice, Page, RunProgress, Segmented, Spinner } from "../components/ui";
import { api, createRun, useApi, useRun } from "../lib/api";
import { pct, TERRAIN_COLORS } from "../lib/format";
import type { MarsResult, MarsSample } from "../lib/types";

const CLASSES = ["soil", "bedrock", "sand", "big rock"];

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((res, rej) => {
    const img = new Image();
    img.onload = () => res(img);
    img.onerror = () => rej(new Error(`Could not load ${src}`));
    img.src = src;
  });
}

/** Draws the rover image and a colour overlay built from a class-id PNG. */
export function OverlayCanvas({ imageUrl, maskUrl, opacity, enabled }: { imageUrl: string; maskUrl: string; opacity: number; enabled: Record<string, boolean> }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [assets, setAssets] = useState<{ img: HTMLImageElement; ids: Uint8ClampedArray } | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    setAssets(null);
    Promise.all([loadImage(imageUrl), loadImage(maskUrl)]).then(([img, mask]) => {
      const c = document.createElement("canvas");
      c.width = img.naturalWidth;
      c.height = img.naturalHeight;
      const ctx = c.getContext("2d", { willReadFrequently: true })!;
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(mask, 0, 0, c.width, c.height);
      const data = ctx.getImageData(0, 0, c.width, c.height).data;
      if (alive) setAssets({ img, ids: data });
    }).catch((e) => alive && setErr(e.message));
    return () => { alive = false; };
  }, [imageUrl, maskUrl]);

  useEffect(() => {
    if (!assets || !canvas.current) return;
    const { img, ids } = assets;
    const cv = canvas.current;
    cv.width = img.naturalWidth;
    cv.height = img.naturalHeight;
    const ctx = cv.getContext("2d")!;
    ctx.drawImage(img, 0, 0);
    const overlay = ctx.createImageData(cv.width, cv.height);
    const o = overlay.data;
    const lut = CLASSES.map((c) => (enabled[c] ? TERRAIN_COLORS[c] : null));
    const a = Math.round(opacity * 255);
    for (let i = 0; i < ids.length; i += 4) {
      const col = lut[ids[i]];
      if (col) { o[i] = col[0]; o[i + 1] = col[1]; o[i + 2] = col[2]; o[i + 3] = a; }
    }
    const tmp = document.createElement("canvas");
    tmp.width = cv.width; tmp.height = cv.height;
    tmp.getContext("2d")!.putImageData(overlay, 0, 0);
    ctx.drawImage(tmp, 0, 0);
  }, [assets, opacity, enabled]);

  if (err) return <ErrorNotice message={err} />;
  return (
    <div className="relative overflow-hidden rounded-2xl bg-slate-200/60">
      {!assets && <div className="flex h-72 items-center justify-center gap-2 text-sm text-ink-soft"><Spinner /> Rendering overlay…</div>}
      <canvas ref={canvas} className={`h-auto w-full ${assets ? "" : "hidden"}`} />
    </div>
  );
}

const MARS_SITES: GlobeMarker[] = [
  { lat: -4.59, lon: 137.44, label: "Curiosity · Gale crater (AI4Mars images)", color: "#f97316", primary: true },
  { lat: 18.44, lon: 77.45, label: "Perseverance · Jezero crater", color: "#7c3aed" },
  { lat: 4.5, lon: 135.62, label: "InSight · Elysium Planitia", color: "#06b6d4" },
];

function MarsExtras({ runId }: { runId: string }) {
  const [edit, setEdit] = useState(false);
  return (
    <>
      <Glass className="hud space-y-3 p-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="kicker">Label & review · fix the model's mistakes</div>
          <div className="flex flex-wrap gap-2">
            <button className={edit ? "btn-primary px-3 py-1.5 text-xs" : "btn-ghost px-3 py-1.5 text-xs"} onClick={() => setEdit((e) => !e)}>
              <Paintbrush className="h-3.5 w-3.5" /> {edit ? "Close editor" : "Correct this mask"}
            </button>
          </div>
        </div>
        <EvidencePanel runId={runId} compact />
        {edit && <Suspense fallback={<Spinner />}><MaskEditor runId={runId} /></Suspense>}
      </Glass>
      <Glass className="hud p-4">
        <div className="kicker mb-2">Where these images were taken</div>
        <Suspense fallback={<div className="h-[360px]" />}>
          <ContextGlobe body="mars" markers={MARS_SITES} height={360}
            caption="AI4Mars uses Curiosity NAVCAM images from Gale crater (4.6°S, 137.4°E). Globe surface is illustrative; site positions are real." />
        </Suspense>
      </Glass>
    </>
  );
}

export default function Mars() {
  const readOnly = useCapabilities()?.capabilities.write === false;
  const [params, setParams] = useSearchParams();
  const runId = params.get("run");
  const samples = useApi<MarsSample[]>("/api/mars/samples");
  const models = useApi<Record<string, { active: string | null }>>("/api/models");
  useOnJobFinished(useCallback(() => { samples.reload(); models.reload(); }, [samples.reload, models.reload])); // eslint-disable-line react-hooks/exhaustive-deps
  const { run, error: runError } = useRun<MarsResult>(runId);
  const [opacity, setOpacity] = useState(0.55);
  const [enabled, setEnabled] = useState<Record<string, boolean>>({ soil: true, bedrock: true, sand: true, "big rock": true });
  const [layer, setLayer] = useState<"pred" | "label">("pred");
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const noModel = models.data && !models.data.mars?.active;
  async function runSample(sample_id: string) {
    setSubmitError(null);
    try { setParams({ run: (await createRun("mars", { sample_id })).id }); } catch (e) { setSubmitError((e as Error).message); }
  }
  async function upload(file: File) {
    if (readOnly) return;
    setSubmitError(null);
    setUploading(true);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const up = await api<{ upload_id: string }>("/api/mars/uploads", { method: "POST", body: fd });
      setParams({ run: (await createRun("mars", { upload_id: up.upload_id })).id });
    } catch (e) { setSubmitError((e as Error).message); } finally { setUploading(false); }
  }

  const res = run?.status === "succeeded" ? run.result : undefined;
  const busy = run && (run.status === "queued" || run.status === "running");
  const fileUrl = (name: string) => `/api/runs/${run!.id}/files/${name}`;

  return (
    <Page title="Mars Terrain Segmentation"
      subtitle="DeepLabV3+ (ResNet encoder) labels every pixel of a Curiosity NAVCAM image as soil, bedrock, sand or big rock. The rover body and terrain beyond 30 m are excluded."
      icon={<IconBadge gradient="from-orange-400 to-rose-500"><Mountain className="h-6 w-6" /></IconBadge>}>
      <ModuleStatus ids={["ai4mars"]} />
      {noModel && (
        <Notice tone="info" title="No Mars terrain model installed yet">
          <p>Train it from the local AI4Mars pairs - a quick run (pipeline check) takes minutes on a CPU, full training takes hours. For the best accuracy, train on a free
            Colab/Kaggle GPU with <b>notebooks/mars_training.ipynb</b> and upload the <code>mars_model.zip</code> it produces here.</p>
          <div className="flex flex-wrap items-start gap-2">
            <JobButton task="mars.train_quick" module="mars" label="Quick run (pipeline check)" className="btn-primary px-3 py-1.5 text-xs" />
            <MarsModelUpload onDone={() => models.reload()} />
          </div>
        </Notice>
      )}
      <div className="grid gap-4 xl:grid-cols-[360px_1fr]">
        <div className="space-y-4">
          <Glass className="p-4">
            <div
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }} onDragLeave={() => setDragOver(false)}
              onDrop={(e) => { e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files[0]; if (f) upload(f); }}
              onClick={() => !noModel && fileRef.current?.click()}
              className={`flex ${noModel ? "pointer-events-none opacity-50" : "cursor-pointer"} flex-col items-center gap-2 rounded-2xl border-2 border-dashed p-6 text-center transition ${dragOver ? "border-mars-400 bg-mars-50/80" : "border-mars-400/40 bg-white/40 hover:bg-white/70"}`}>
              {uploading ? <Spinner className="h-6 w-6 text-mars-600" /> : <Upload className="h-6 w-6 text-mars-600" />}
              <div className="text-sm font-semibold">Upload a rover image</div>
              <div className="text-[11px] text-ink-soft">PNG/JPEG, grayscale NAVCAM works best · max 15 MB</div>
              <FileInput ref={fileRef} type="file" accept="image/png,image/jpeg" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) upload(f); e.target.value = ""; }} />
            </div>
          </Glass>
          <Glass className="p-4">
            <div className="label mb-2 flex items-center gap-1"><ImagePlus className="h-3.5 w-3.5" /> Curated expert-labeled samples</div>
            {samples.loading && <div className="flex items-center gap-2 text-sm text-ink-soft"><Spinner /> Loading…</div>}
            {samples.data?.length === 0 && (
              <div className="space-y-2 text-sm text-ink-soft"><p>No samples yet - they come with the AI4Mars dataset (release bundle or Data &amp; Training).</p>
                <JobButton task="mars.samples" module="mars" label="Download samples now" className="btn-ghost px-3 py-1.5 text-xs" /></div>
            )}
            <div className="scroll-thin grid max-h-[520px] grid-cols-3 gap-2 overflow-y-auto pr-1">
              {samples.data?.map((s) => (
                <button key={s.id} onClick={() => runSample(s.id)} disabled={!!busy || !!noModel} title={s.name}
                  className="group relative overflow-hidden rounded-xl ring-2 ring-transparent transition hover:ring-mars-400 disabled:cursor-not-allowed">
                  <img src={`/api/mars/samples/${s.id}/thumb`} alt={s.name} loading="lazy" className="aspect-square w-full object-cover transition group-hover:scale-105" />
                  {s.label_shares.sand > 0.02 && <span className="absolute left-1 top-1 rounded-md bg-rose-500/90 px-1.5 text-[9px] font-bold text-white">SAND</span>}
                </button>))}
            </div>
          </Glass>
        </div>

        <div className="space-y-4">
          {submitError && <ErrorNotice message={submitError} />}
          {runError && <ErrorNotice message={runError} />}
          {busy && <RunProgress status={run!.status} label={run!.input_label} />}
          {run?.status === "failed" && <ErrorNotice title="Run failed" message={run.error ?? "Unknown error"} />}
          {!run && (
            <Glass className="flex flex-col items-center gap-3 py-16 text-center">
              <Mountain className="h-10 w-10 text-mars-500" />
              <p className="font-semibold">Pick a Curiosity image or upload your own</p>
              <p className="max-w-md text-sm text-ink-soft">The output is a terrain classification per pixel. Sand matters most for driving: soft sand is what trapped the Spirit rover.</p>
            </Glass>
          )}
          {res && run && (
            <div className="grid gap-4 2xl:grid-cols-[1fr_300px]">
              <Glass className="space-y-3 p-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="label">{run.input_label} · {res.width}×{res.height} · model {res.model_version}</div>
                  {res.files.label && <Segmented value={layer} onChange={setLayer} options={[{ value: "pred", label: "Model prediction" }, { value: "label", label: "Expert label" }]} />}
                </div>
                <OverlayCanvas imageUrl={fileUrl("input.png")} maskUrl={fileUrl(layer === "label" && res.files.label ? "label.png" : "mask.png")} opacity={opacity} enabled={enabled} />
                <div className="flex flex-wrap items-center gap-4">
                  <div className="flex min-w-[220px] flex-1 items-center gap-3">
                    <span className="text-xs font-semibold text-ink-soft">Opacity</span>
                    <input type="range" min={0} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(+e.target.value)} className="slider" />
                    <span className="w-10 text-right text-xs tabular-nums">{Math.round(opacity * 100)}%</span>
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {CLASSES.map((c) => {
                      const [r, g, b] = TERRAIN_COLORS[c];
                      return (
                        <button key={c} onClick={() => setEnabled((s) => ({ ...s, [c]: !s[c] }))}
                          className={`chip border transition ${enabled[c] ? "border-white bg-white shadow-sm" : "border-transparent bg-white/40 opacity-50"}`}>
                          <span className="h-3 w-3 rounded-full" style={{ background: `rgb(${r},${g},${b})` }} /> <span className="capitalize">{c}</span>
                        </button>);
                    })}
                  </div>
                </div>
              </Glass>
              <div className="space-y-4">
                <Glass className="p-4">
                  <div className="label">Predicted terrain share</div>
                  <div className="mt-3 space-y-2.5">
                    {CLASSES.map((c) => {
                      const [r, g, b] = TERRAIN_COLORS[c];
                      return (
                        <div key={c}>
                          <div className="flex justify-between text-xs"><span className="font-semibold capitalize">{c}</span><span className="tabular-nums">{pct(res.class_shares[c])}</span></div>
                          <div className="mt-1 h-2 overflow-hidden rounded-full bg-white/70">
                            <div className="h-full rounded-full transition-all duration-700" style={{ width: `${res.class_shares[c] * 100}%`, background: `rgb(${r},${g},${b})` }} />
                          </div>
                        </div>);
                    })}
                  </div>
                  {res.masked_fraction > 0 && <p className="mt-3 text-[11px] text-ink-soft">{pct(res.masked_fraction)} of pixels excluded (rover body / beyond 30 m).</p>}
                </Glass>
                {res.evaluation && (
                  <Glass className="p-4">
                    <div className="label">Against the expert label</div>
                    <p className="mt-1 text-[11px] text-ink-soft">{pct(res.evaluation.labeled_fraction)} of this image has an expert label.</p>
                    <table className="mt-2 w-full text-xs">
                      <thead><tr className="text-left text-ink-faint"><th className="py-1">IoU</th><th>Model</th><th>All-soil</th></tr></thead>
                      <tbody>
                        <tr className="border-t border-white/70 font-semibold"><td className="py-1.5">Mean</td><td>{pct(res.evaluation.model.miou)}</td><td>{pct(res.evaluation.baseline.miou)}</td></tr>
                        {CLASSES.map((c) => (
                          <tr key={c} className="border-t border-white/70"><td className="py-1.5 capitalize">{c}</td>
                            <td>{res.evaluation!.model.iou[c] == null ? "n/a" : pct(res.evaluation!.model.iou[c]!)}</td>
                            <td>{res.evaluation!.baseline.iou[c] == null ? "n/a" : pct(res.evaluation!.baseline.iou[c]!)}</td></tr>))}
                      </tbody>
                    </table>
                    <p className="mt-2 text-[11px] text-ink-faint">n/a = class not present in this image's label.</p>
                  </Glass>
                )}
              </div>
            </div>
          )}
          {res?.location && res.location.precision !== "unlocated" && (
            <div className="glass-inset flex flex-wrap items-center gap-2 px-4 py-2.5 text-xs">
              <span className="font-semibold">Sol {res.location.sol} · site {res.location.site} / drive {res.location.drive}</span>
              <span className="chip bg-sky-100 text-sky-700">location: {res.location.precision}</span>
              <span className="text-ink-soft">{res.location.note}</span>
              <a href="/mars/traverse" className="ml-auto font-semibold text-violet-600 hover:underline">Show on the traverse →</a>
            </div>)}
          {res && run && <MarsExtras runId={run.id} />}
          {!res && !busy && run?.status !== "failed" && runId && <div className="flex items-center gap-2 text-sm text-ink-soft"><Spinner /> Loading run…</div>}
        </div>
      </div>
    </Page>
  );
}
