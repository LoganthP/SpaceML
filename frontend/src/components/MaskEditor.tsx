import { Brush, Eraser, Redo2, Save, Undo2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { TERRAIN_COLORS } from "../lib/format";
import { Spinner } from "./ui";

const CLASSES = ["soil", "bedrock", "sand", "big rock"] as const;

function load(src: string) {
  return new Promise<HTMLImageElement>((res, rej) => {
    const i = new Image();
    i.onload = () => res(i);
    i.onerror = () => rej(new Error(`Could not load ${src}`));
    i.src = `${src}${src.includes("?") ? "&" : "?"}t=${Date.now()}`;
  });
}

/**
 * Paint corrections onto a terrain mask. Starts from the corrected mask if one was saved,
 * otherwise from the model prediction. Saves class ids (0-3, 255 = unlabeled) as a PNG.
 */
export default function MaskEditor({ runId, onSaved }: { runId: string; onSaved?: (changed: number) => void }) {
  const view = useRef<HTMLCanvasElement>(null);
  const ids = useRef<Uint8Array | null>(null);
  const img = useRef<HTMLImageElement | null>(null);
  const size = useRef({ w: 0, h: 0 });
  const history = useRef<Uint8Array[]>([]);
  const future = useRef<Uint8Array[]>([]);
  const [cls, setCls] = useState<number>(2);
  const [brush, setBrush] = useState(14);
  const [ready, setReady] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const painting = useRef(false);

  const render = () => {
    const cv = view.current, d = ids.current, im = img.current;
    if (!cv || !d || !im) return;
    const { w, h } = size.current;
    const ctx = cv.getContext("2d")!;
    ctx.drawImage(im, 0, 0, w, h);
    const ov = ctx.createImageData(w, h);
    for (let i = 0; i < d.length; i++) {
      const c = TERRAIN_COLORS[CLASSES[d[i]] ?? ""];
      if (c) { ov.data[i * 4] = c[0]; ov.data[i * 4 + 1] = c[1]; ov.data[i * 4 + 2] = c[2]; ov.data[i * 4 + 3] = 120; }
    }
    const tmp = document.createElement("canvas");
    tmp.width = w; tmp.height = h;
    tmp.getContext("2d")!.putImageData(ov, 0, 0);
    ctx.drawImage(tmp, 0, 0);
  };

  useEffect(() => {
    let alive = true;
    (async () => {
      const base = `/api/runs/${runId}/files`;
      const [im, mask] = await Promise.all([
        load(`${base}/input.png`).catch(() => load(`${base}/source.png`)),
        load(`${base}/corrected_mask.png`).catch(() => load(`${base}/mask.png`)),
      ]);
      if (!alive) return;
      const w = mask.naturalWidth, h = mask.naturalHeight;
      const c = document.createElement("canvas");
      c.width = w; c.height = h;
      const ctx = c.getContext("2d", { willReadFrequently: true })!;
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(mask, 0, 0);
      const px = ctx.getImageData(0, 0, w, h).data;
      const a = new Uint8Array(w * h);
      for (let i = 0; i < a.length; i++) a[i] = px[i * 4];
      ids.current = a;
      img.current = im;
      size.current = { w, h };
      view.current!.width = w; view.current!.height = h;
      render();
      setReady(true);
    })().catch((e) => setMsg((e as Error).message));
    return () => { alive = false; };
  }, [runId]); // eslint-disable-line react-hooks/exhaustive-deps

  const paint = (e: React.PointerEvent) => {
    const cv = view.current!, d = ids.current;
    if (!d) return;
    const r = cv.getBoundingClientRect();
    const { w, h } = size.current;
    const x = ((e.clientX - r.left) / r.width) * w, y = ((e.clientY - r.top) / r.height) * h;
    const rad = brush * (w / r.width);
    for (let yy = Math.max(0, Math.floor(y - rad)); yy < Math.min(h, y + rad); yy++)
      for (let xx = Math.max(0, Math.floor(x - rad)); xx < Math.min(w, x + rad); xx++)
        if ((xx - x) ** 2 + (yy - y) ** 2 <= rad * rad) d[yy * w + xx] = cls;
    render();
  };

  async function save() {
    const d = ids.current;
    if (!d) return;
    setSaving(true); setMsg(null);
    const { w, h } = size.current;
    const c = document.createElement("canvas");
    c.width = w; c.height = h;
    const ctx = c.getContext("2d")!;
    const out = ctx.createImageData(w, h);
    for (let i = 0; i < d.length; i++) { out.data[i * 4] = out.data[i * 4 + 1] = out.data[i * 4 + 2] = d[i]; out.data[i * 4 + 3] = 255; }
    ctx.putImageData(out, 0, 0);
    const blob: Blob = await new Promise((res) => c.toBlob((b) => res(b!), "image/png"));
    const fd = new FormData();
    fd.append("file", blob, "corrected_mask.png");
    try {
      const r = await fetch(`/api/runs/${runId}/mask`, { method: "PUT", body: fd });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail ?? r.statusText);
      setMsg(`Saved - ${(j.changed_fraction * 100).toFixed(1)}% of pixels relabeled vs the model.`);
      onSaved?.(j.changed_fraction);
    } catch (e) { setMsg((e as Error).message); } finally { setSaving(false); }
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        {CLASSES.map((c, i) => {
          const [r, g, b] = TERRAIN_COLORS[c];
          return (
            <button key={c} onClick={() => setCls(i)} className={`chip border transition ${cls === i ? "border-white bg-white shadow" : "border-transparent bg-white/40"}`}>
              <span className="h-3 w-3 rounded-full" style={{ background: `rgb(${r},${g},${b})` }} /><span className="capitalize">{c}</span>
            </button>);
        })}
        <button onClick={() => setCls(255)} className={`chip border ${cls === 255 ? "border-white bg-white shadow" : "border-transparent bg-white/40"}`}><Eraser className="h-3 w-3" /> unlabeled</button>
        <label className="ml-2 flex items-center gap-2 text-xs text-ink-soft"><Brush className="h-3.5 w-3.5" />
          <input type="range" min={3} max={60} value={brush} onChange={(e) => setBrush(+e.target.value)} className="slider w-28" /></label>
        <button className="btn-ghost px-2 py-1.5" title="Undo" onClick={() => { const p = history.current.pop(); if (p && ids.current) { future.current.push(ids.current.slice()); ids.current = p; render(); } }}><Undo2 className="h-3.5 w-3.5" /></button>
        <button className="btn-ghost px-2 py-1.5" title="Redo" onClick={() => { const n = future.current.pop(); if (n && ids.current) { history.current.push(ids.current.slice()); ids.current = n; render(); } }}><Redo2 className="h-3.5 w-3.5" /></button>
        <button className="btn-primary ml-auto px-3 py-1.5 text-xs" onClick={save} disabled={!ready || saving}>{saving ? <Spinner /> : <Save className="h-3.5 w-3.5" />} Save corrected mask</button>
      </div>
      {!ready && !msg && <div className="flex items-center gap-2 text-sm text-ink-soft"><Spinner /> Loading image and mask…</div>}
      <canvas ref={view} className={`w-full cursor-crosshair touch-none rounded-2xl ${ready ? "" : "hidden"}`}
        onPointerDown={(e) => { if (!ids.current) return; history.current.push(ids.current.slice()); if (history.current.length > 30) history.current.shift(); future.current = []; painting.current = true; (e.target as HTMLElement).setPointerCapture(e.pointerId); paint(e); }}
        onPointerMove={(e) => painting.current && paint(e)} onPointerUp={() => { painting.current = false; }} />
      {msg && <p className="text-xs text-ink-soft">{msg}</p>}
      <p className="text-[10px] text-ink-faint">Paint with the selected class. Saved masks use AI4Mars ids (0 soil, 1 bedrock, 2 sand, 3 big rock, 255 unlabeled) and are exported from the Review Studio for retraining.</p>
    </div>
  );
}
