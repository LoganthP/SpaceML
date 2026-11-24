import { Check, CircleHelp, Download, FileJson, FileText, UploadCloud, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { api, uploadWithProgress } from "../lib/api";
import { Spinner } from "./ui";

/* ------------------------------------------------------------------ report download */
export function ReportButtons({ runId }: { runId: string }) {
  return (
    <div className="flex flex-wrap gap-1.5">
      <a className="btn-ghost px-3 py-1.5 text-xs" href={`/api/runs/${runId}/report?format=md`} download>
        <FileText className="h-3.5 w-3.5" /> Report (.md)
      </a>
      <a className="btn-ghost px-3 py-1.5 text-xs" href={`/api/runs/${runId}/report?format=json`} download>
        <FileJson className="h-3.5 w-3.5" /> Full result (.json)
      </a>
    </div>
  );
}

/* ------------------------------------------------------------------ human review */
export type Verdict = "confirmed" | "dismissed" | "unsure";
export interface Review { item: string; verdict: Verdict; note: string | null }

export function useReviews(runId: string | null) {
  const [reviews, setReviews] = useState<Record<string, Review>>({});
  const load = useCallback(async () => {
    if (!runId) return;
    const r = await api<Review[]>(`/api/runs/${runId}/reviews`);
    setReviews(Object.fromEntries(r.map((x) => [x.item, x])));
  }, [runId]);
  useEffect(() => { setReviews({}); load().catch(() => {}); }, [load]);
  const set = useCallback(async (item: string, verdict: Verdict | null, note = "") => {
    if (!runId) return;
    if (verdict === null) await api(`/api/runs/${runId}/reviews/${item}`, { method: "DELETE" });
    else await api(`/api/runs/${runId}/reviews/${item}`, { method: "PUT", body: JSON.stringify({ verdict, note }) });
    await load();
  }, [runId, load]);
  return { reviews, set };
}

export function ReviewToggle({ item, review, onSet }: { item: string; review?: Review; onSet: (item: string, v: Verdict | null) => void }) {
  const opts: { v: Verdict; icon: ReactNode; cls: string; title: string }[] = [
    { v: "confirmed", icon: <Check className="h-3 w-3" />, cls: "bg-emerald-500 text-white", title: "Confirm: real event" },
    { v: "dismissed", icon: <X className="h-3 w-3" />, cls: "bg-rose-500 text-white", title: "Dismiss: false alarm" },
    { v: "unsure", icon: <CircleHelp className="h-3 w-3" />, cls: "bg-amber-400 text-white", title: "Unsure" },
  ];
  return (
    <div className="inline-flex rounded-full border border-white/80 bg-white/60 p-0.5">
      {opts.map((o) => (
        <button key={o.v} title={o.title} onClick={() => onSet(item, review?.verdict === o.v ? null : o.v)}
          className={`grid h-6 w-6 place-items-center rounded-full transition ${review?.verdict === o.v ? o.cls : "text-ink-faint hover:bg-white"}`}>
          {o.icon}
        </button>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ upload drop zone */
export function DropZone({ url, accept, hint, onDone, onError, fields, title = "Drop a file or click to upload", compact = false }: {
  url: string; accept?: string; hint: ReactNode; title?: string; compact?: boolean;
  onDone: (res: unknown) => void; onError: (msg: string) => void; fields?: Record<string, string>;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [progress, setProgress] = useState<number | null>(null);
  const [over, setOver] = useState(false);
  const [name, setName] = useState("");

  async function send(f: File) {
    onError("");
    setName(f.name);
    setProgress(0);
    const fd = new FormData();
    fd.append("file", f);
    Object.entries(fields ?? {}).forEach(([k, v]) => fd.append(k, v));
    try {
      onDone(await uploadWithProgress(url, fd, setProgress));
    } catch (e) {
      onError((e as Error).message);
    } finally {
      setProgress(null);
    }
  }
  const busy = progress !== null;
  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files[0]; if (f && !busy) send(f); }}
      onClick={() => !busy && input.current?.click()}
      className={`relative flex cursor-pointer flex-col items-center gap-1.5 overflow-hidden rounded-2xl border-2 border-dashed text-center transition
        ${compact ? "p-4" : "p-6"} ${over ? "border-violet-400 bg-violet-50/80" : "border-violet-300/50 bg-white/40 hover:bg-white/70"}`}>
      {busy ? <Spinner className="h-6 w-6 text-violet-600" /> : <UploadCloud className="h-6 w-6 text-violet-600" />}
      <div className="text-sm font-semibold">{busy ? (progress! < 1 ? `Uploading ${name}… ${Math.round(progress! * 100)}%` : `Parsing ${name}…`) : title}</div>
      <div className="text-[11px] text-ink-soft">{hint}</div>
      {busy && <div className="absolute inset-x-0 bottom-0 h-1 bg-violet-100"><div className="h-full bg-gradient-to-r from-violet-500 to-fuchsia-500 transition-all" style={{ width: `${(progress ?? 0) * 100}%` }} /></div>}
      <input ref={input} type="file" accept={accept} className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) send(f); e.target.value = ""; }} />
    </div>
  );
}

export function DownloadLink({ href, children }: { href: string; children: ReactNode }) {
  return <a href={href} download className="inline-flex items-center gap-1 text-xs font-semibold text-violet-600 hover:underline"><Download className="h-3 w-3" />{children}</a>;
}
