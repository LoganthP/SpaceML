import { ExternalLink, History, RefreshCw, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ErrorNotice, Glass, IconBadge, OriginBadge, Page, Segmented, Spinner, StatusBadge } from "../components/ui";
import { api, useApi } from "../lib/api";
import { duration, MODULE_META, timeAgo } from "../lib/format";
import type { Module, Run } from "../lib/types";

export default function Runs() {
  const [module, setModule] = useState<"all" | Module>("all");
  const runs = useApi<Run[]>(`/api/runs?limit=200${module !== "all" ? `&module=${module}` : ""}`, [module]);
  const active = runs.data?.some((r) => r.status === "queued" || r.status === "running");

  useEffect(() => {
    if (!active) return;
    const t = setInterval(runs.reload, 2000);
    return () => clearInterval(t);
  }, [active, runs.reload]);

  async function remove(id: string) {
    await api(`/api/runs/${id}`, { method: "DELETE" });
    runs.reload();
  }

  return (
    <Page title="Run history" subtitle="Every prediction is saved with its input, model version, status, timing and result."
      icon={<IconBadge gradient="from-slate-500 to-slate-700"><History className="h-6 w-6" /></IconBadge>}
      actions={<button className="btn-ghost" onClick={runs.reload}><RefreshCw className="h-4 w-4" /> Refresh</button>}>
      <Glass className="max-w-full overflow-hidden p-4">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <Segmented value={module} onChange={setModule} options={[
            { value: "all", label: "All" }, { value: "telemetry", label: "Telemetry" },
            { value: "exoplanet", label: "Exoplanets" }, { value: "mars", label: "Mars" }, { value: "spaceweather", label: "Space weather" },
            { value: "workbench", label: "Workbench" }]} />
          <span className="text-xs text-ink-soft">{runs.data?.length ?? 0} runs</span>
        </div>
        {runs.error && <ErrorNotice message={runs.error.message} />}
        {runs.loading && !runs.data && <div className="flex items-center gap-2 p-4 text-sm text-ink-soft"><Spinner /> Loading…</div>}
        {runs.data?.length === 0 && <p className="p-6 text-center text-sm text-ink-soft">No runs yet.</p>}
        {!!runs.data?.length && (
          <div className="scroll-thin max-w-full overflow-x-auto">
            <table className="w-full min-w-[860px] text-sm">
              <thead>
                <tr className="text-left text-xs text-ink-faint">
                  <th className="px-2 pb-2">Module</th><th>Input</th><th>Result</th><th>Model</th><th>Status</th><th>Origin</th><th>Duration</th><th>Created</th><th aria-label="Actions"><span className="sr-only">Actions</span></th>
                </tr>
              </thead>
              <tbody>
                {runs.data.map((r) => (
                  <tr key={r.id} className="border-t border-white/70 transition hover:bg-white/50">
                    <td className="px-2 py-2.5">
                      <span className={`chip ${MODULE_META[r.module].soft}`}>{MODULE_META[r.module].short}</span>
                    </td>
                    <td className="font-semibold">{r.input_label}</td>
                    <td className="max-w-[280px] truncate text-ink-soft" title={r.error ?? ""}>{r.summary?.headline ?? (r.error ? <span className="text-rose-600">{r.error}</span> : "—")}</td>
                    <td className="font-mono text-xs">{r.model_version ?? "—"}</td>
                    <td><StatusBadge status={r.status} /></td>
                    <td><OriginBadge origin={r.origin} />{r.repeat_of && <span className="ml-1 text-[10px] text-ink-faint" title={`repeat of ${r.repeat_of}`}>↻</span>}</td>
                    <td className="tabular-nums">{duration(r.duration_ms)}</td>
                    <td className="text-xs text-ink-soft">{timeAgo(r.created_at)}</td>
                    <td className="whitespace-nowrap text-right">
                      {r.status === "succeeded" && (
                        <Link to={r.module === "spaceweather" ? `/replay?event=${(r.input_ref as { event_id?: string }).event_id ?? ""}&run=${r.id}` : `${MODULE_META[r.module].path}?run=${r.id}`} className="btn-ghost px-2.5 py-1.5 text-xs"><ExternalLink className="h-3.5 w-3.5" /> Open</Link>
                      )}
                      {r.status !== "queued" && r.status !== "running" && (
                        <button onClick={() => remove(r.id)} className="ml-1 rounded-xl p-2 text-ink-faint transition hover:bg-rose-50 hover:text-rose-600" title="Delete run">
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Glass>
    </Page>
  );
}
