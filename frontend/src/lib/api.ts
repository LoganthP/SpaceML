import { useCallback, useEffect, useRef, useState } from "react";
import type { Module, Run } from "./types";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

function detailMessage(detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((d) => (d as { msg?: string }).msg ?? JSON.stringify(d)).join("; ");
  return JSON.stringify(detail);
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, {
      ...init,
      headers: init?.body instanceof FormData ? init?.headers : { "Content-Type": "application/json", ...init?.headers },
    });
  } catch {
    throw new ApiError(0, "Cannot reach the API. Is the backend running on port 8000?");
  }
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) msg = detailMessage(body.detail);
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, msg);
  }
  return res.json() as Promise<T>;
}

export function createRun(module: Module, input: Record<string, unknown>) {
  return api<Run>("/api/runs", { method: "POST", body: JSON.stringify({ module, input }) });
}

/** Generic loader with refresh. */
export function useApi<T>(path: string | null, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(!!path);
  const load = useCallback(async () => {
    if (!path) return;
    setLoading(true);
    try {
      setData(await api<T>(path));
      setError(null);
    } catch (e) {
      setError(e as ApiError);
    } finally {
      setLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, ...deps]);
  useEffect(() => {
    load();
  }, [load]);
  return { data, error, loading, reload: load, setData };
}

/** Polls a run until it finishes, then returns it with its result. */
export function useRun<R>(runId: string | null) {
  const [run, setRun] = useState<Run<R> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<number>();
  useEffect(() => {
    setRun(null);
    setError(null);
    if (!runId) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const r = await api<Run<R>>(`/api/runs/${runId}`);
        if (cancelled) return;
        setRun(r);
        if (r.status === "queued" || r.status === "running") timer.current = window.setTimeout(tick, 900);
      } catch (e) {
        if (!cancelled) setError((e as Error).message);
      }
    };
    tick();
    return () => {
      cancelled = true;
      window.clearTimeout(timer.current);
    };
  }, [runId]);
  return { run, error };
}

/** Upload with real progress events (fetch cannot report upload progress). */
export function uploadWithProgress<T>(url: string, form: FormData, onProgress: (f: number) => void): Promise<T> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      let body: unknown = null;
      try { body = JSON.parse(xhr.responseText); } catch { /* not JSON */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body as T);
      else {
        const d = (body as { detail?: unknown } | null)?.detail;
        reject(new ApiError(xhr.status, typeof d === "string" ? d : d ? JSON.stringify(d) : `${xhr.status} ${xhr.statusText}`));
      }
    };
    xhr.onerror = () => reject(new ApiError(0, "Upload failed - is the backend running?"));
    xhr.send(form);
  });
}

/** Polls a list endpoint while `active` is true. */
export function usePoll<T>(path: string, ms: number, active = true) {
  const r = useApi<T>(path);
  useEffect(() => {
    if (!active) return;
    const t = window.setInterval(r.reload, ms);
    return () => window.clearInterval(t);
  }, [active, ms, r.reload]);
  return r;
}
