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
    throw new ApiError(0, "Cannot reach Space ML. Check your connection and retry.");
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
  const [result, setResult] = useState<{ path: string; data: T | null; error: ApiError | null; loading: boolean } | null>(null);
  const request = useRef(0);
  const load = useCallback(async () => {
    if (!path) return;
    const id = ++request.current;
    const update = (data: T | undefined, error: ApiError | null, loading: boolean) => {
      if (request.current !== id) return;
      setResult((prior) => ({ path, data: data === undefined ? (prior?.path === path ? prior.data : null) : data, error, loading }));
    };
    update(undefined, null, true);
    try {
      update(await api<T>(path), null, false);
    } catch (e) {
      update(undefined, e as ApiError, false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, ...deps]);
  useEffect(() => {
    void load();
    return () => { ++request.current; };
  }, [load]);
  const current = result?.path === path ? result : null;
  return { data: current?.data ?? null, error: current?.error ?? null, loading: current?.loading ?? !!path, reload: load };
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
    let retries = 0;
    const tick = async () => {
      try {
        const r = await api<Run<R>>(`/api/runs/${runId}`);
        if (cancelled) return;
        retries = 0;
        setError(null);
        setRun(r);
        if (r.status === "queued" || r.status === "running") timer.current = window.setTimeout(tick, 900);
      } catch (e) {
        if (cancelled) return;
        const transient = e instanceof ApiError && (e.status === 0 || e.status === 429 || e.status >= 500);
        if (transient && retries < 10) {
          retries += 1;
          timer.current = window.setTimeout(tick, Math.min(5000, 1000 * retries));
          return;
        }
        setError((e as Error).message);
      }
    };
    tick();
    return () => {
      cancelled = true;
      window.clearTimeout(timer.current);
    };
  }, [runId]);
  return { run: run?.id === runId ? run : null, error };
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
