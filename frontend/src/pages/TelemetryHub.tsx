import { Satellite } from "lucide-react";
import { lazy, Suspense, useEffect, useRef } from "react";
import { useSearchParams } from "react-router-dom";
import { Spinner } from "../components/ui";
import { api } from "../lib/api";
import { useOnJobFinished } from "../components/Jobs";
import { READINESS_LABEL, useCatalog } from "../components/Catalog";
import { useMissions } from "../components/missions/MissionWorkspace";
import SmapMsl from "./Telemetry";

const MissionWorkspace = lazy(() => import("../components/missions/MissionWorkspace"));

/** Telemetry: pick a mission. NASA SMAP/MSL keeps its existing workflow; every other source opens the
 *  multi-mission workspace (replay, quick train where supported, scoring, review). */
export default function TelemetryHub() {
  const [sp, setSp] = useSearchParams();
  const ms = useMissions();
  useOnJobFinished(ms.reload);
  const runId = sp.get("run");
  const asked = sp.get("mission") ?? "smap-msl";
  const known = asked === "smap-msl" || !ms.data || ms.data.missions.some((m) => m.id === asked);
  const mission = known ? asked : "smap-msl";                 // old links to removed missions open SMAP/MSL
  const waiting = mission !== "smap-msl" && !ms.data;
  useEffect(() => {
    if (ms.data && !known) setSp((p) => { p.delete("mission"); return p; }, { replace: true });
  }, [ms.data, known]); // eslint-disable-line react-hooks/exhaustive-deps
  const resolved = useRef<string | null>(null);
  useEffect(() => {                 // a ?run= link from Runs / Review Studio: open the run's mission
    if (!runId || sp.get("mission") || resolved.current === runId) return;
    resolved.current = runId;
    api<{ module: string; input_ref: { mission?: string } }>(`/api/runs/${runId}`).then((r) => {
      if (r.module === "missions" && r.input_ref.mission) setSp((p) => { p.set("mission", r.input_ref.mission!); return p; }, { replace: true });
    }).catch(() => undefined);
  }, [runId]); // eslint-disable-line react-hooks/exhaustive-deps
  const choose = (id: string) => setSp((p) => { const n = new URLSearchParams(); if (id !== "smap-msl") n.set("mission", id); void p; return n; });
  const cat = useCatalog();
  const rd = (id: string) => cat.data?.datasets.find((d) => d.id === id)?.status?.readiness;
  const options = [{ id: "smap-msl", name: "NASA SMAP / MSL", state: READINESS_LABEL[rd("telemetry") ?? "model_available"].toLowerCase() },
    ...(ms.data?.missions ?? []).map((m) => { const r = rd(`mission-${m.id}`); return { id: m.id, name: m.name, state: r ? READINESS_LABEL[r].toLowerCase() : m.dataset.installed ? "installed" : "missing" }; })];
  const move = (e: React.KeyboardEvent<HTMLDivElement>) => {
    const i = options.findIndex((o) => o.id === mission);
    const d = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 0;
    if (!d) return;
    e.preventDefault();
    const n = options[(i + d + options.length) % options.length];
    choose(n.id);
    window.requestAnimationFrame(() => document.getElementById(`mission-opt-${n.id}`)?.focus());
  };
  return (
    <div className="space-y-4">
      <nav aria-label="Telemetry source" className="glass flex flex-wrap items-center gap-2 p-3">
        <Satellite className="h-4 w-4 text-indigo-600" aria-hidden="true" />
        <span id="mission-label" className="text-sm font-semibold">Mission</span>
        <div role="radiogroup" aria-labelledby="mission-label" onKeyDown={move} className="flex flex-wrap gap-1.5">
          {options.map((o) => {
            const on = o.id === mission;
            return (
              <button key={o.id} id={`mission-opt-${o.id}`} type="button" role="radio" aria-checked={on} tabIndex={on ? 0 : -1} onClick={() => choose(o.id)}
                className={`rounded-xl border px-2.5 py-1 text-xs font-semibold transition ${on ? "border-indigo-500 bg-indigo-600 text-white" : "border-slate-300 bg-white/70 text-slate-800 hover:border-indigo-400 dark:border-white/15 dark:bg-white/5 dark:text-slate-100"}`}>
                {o.name} <span className={`ml-1 font-normal ${on ? "text-indigo-100" : "text-slate-600 dark:text-slate-300"}`}>· {o.state}</span>
              </button>);
          })}
        </div>
      </nav>
      {waiting ? <div className="flex items-center gap-2 p-6 text-sm" role="status"><Spinner /> Loading missions…</div>
        : mission === "smap-msl" ? <SmapMsl /> : (
        <Suspense fallback={<div className="flex items-center gap-2 p-6 text-sm" role="status"><Spinner /> Loading workspace…</div>}>
          <MissionWorkspace key={mission} mission={mission} runId={runId}
            onRun={(id) => setSp((p) => { if (id) p.set("run", id); else p.delete("run"); return p; }, { replace: true })} />
        </Suspense>)}
    </div>
  );
}
