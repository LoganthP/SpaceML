import { HelpCircle } from "lucide-react";
import type { AnalysisSpec, ParamSpec, WbDataset } from "../../lib/types";

/** Initial values: declared defaults, refined with what the importer detected in the data. */
export function initialValues(a: AnalysisSpec, ds: WbDataset): Record<string, unknown> {
  const v: Record<string, unknown> = {};
  for (const p of a.params) v[p.name] = p.default ?? (p.type === "columns" ? [] : p.type === "text" ? "" : p.type === "units" ? {} : null);
  const t = ds.tables?.[0];
  if (t) {
    v.table = t.id;
    if ("time_col" in v) v.time_col = t.suggested.time_col;
    if ("label_col" in v) v.label_col = t.suggested.label_col;
    if ("sensor_cols" in v) v.sensor_cols = t.suggested.sensor_cols.slice(0, 4);
  }
  const imgs = ds.images ?? [];
  if ("image" in v) v.image = imgs[0]?.id ?? null;
  if ("before" in v) v.before = imgs[0]?.id ?? null;
  if ("after" in v) v.after = imgs[1]?.id ?? null;
  if ("method" in v && imgs[0] && imgs[0].bands >= 4) v.method = "water";
  return v;
}

function shown(p: ParamSpec, values: Record<string, unknown>) {
  return Object.entries(p.show_if ?? {}).every(([k, want]) => values[k] === want);
}

function Field({ p, children }: { p: ParamSpec; children: React.ReactNode }) {
  return (
    <label className="block space-y-1">
      <span className="flex items-center gap-1 text-xs font-semibold text-ink-soft">
        {p.label}{p.required && <span className="text-rose-500">*</span>}
        {p.help && <span title={p.help}><HelpCircle className="h-3 w-3 text-ink-faint" /></span>}
      </span>
      {children}
      {p.help && <span className="block text-[10px] leading-snug text-ink-faint">{p.help}</span>}
    </label>
  );
}

const TYPE_CHIP: Record<string, string> = {
  numeric: "bg-sky-100 text-sky-700", binary: "bg-emerald-100 text-emerald-700",
  datetime: "bg-violet-100 text-violet-700", text: "bg-slate-100 text-slate-600",
};

export default function ParamForm({ analysis, dataset, values, onChange }: {
  analysis: AnalysisSpec; dataset: WbDataset; values: Record<string, unknown>; onChange: (v: Record<string, unknown>) => void;
}) {
  const set = (k: string, v: unknown) => onChange({ ...values, [k]: v });
  const table = dataset.tables?.find((t) => t.id === values.table) ?? dataset.tables?.[0];
  const cols = (filter?: string[]) => (table?.columns ?? []).filter((c) => !filter || filter.includes(c.type));
  const wide = (p: ParamSpec) => ["columns", "image", "text", "units"].includes(p.type);

  return (
    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
      {analysis.params.filter((p) => shown(p, values)).map((p) => {
        const v = values[p.name];
        let input: React.ReactNode = null;
        if (p.type === "table") {
          input = (
            <select className="input" value={String(v ?? "")} onChange={(e) => {
              const t = dataset.tables!.find((x) => x.id === e.target.value)!;
              onChange({ ...values, table: t.id, time_col: t.suggested.time_col, label_col: t.suggested.label_col, sensor_cols: t.suggested.sensor_cols.slice(0, 4) });
            }}>
              {dataset.tables?.map((t) => <option key={t.id} value={t.id}>{t.file} ({t.rows.toLocaleString()} rows)</option>)}
            </select>);
        } else if (p.type === "column") {
          input = (
            <select className="input" value={String(v ?? "")} onChange={(e) => set(p.name, e.target.value || null)}>
              <option value="">{p.optional ? "— none —" : "Choose a column"}</option>
              {cols(p.filter).map((c) => <option key={c.name} value={c.name}>{c.name} ({c.type})</option>)}
            </select>);
        } else if (p.type === "columns") {
          const sel = (v as string[]) ?? [];
          const options = cols(p.filter).filter((c) => c.name !== values.time_col && c.name !== values.label_col);
          input = (
            <div className="flex max-h-40 flex-wrap gap-1.5 overflow-y-auto rounded-2xl border border-white/80 bg-white/50 p-2 scroll-thin">
              {options.length === 0 && <span className="text-xs text-ink-soft">No numeric columns</span>}
              {options.map((c) => {
                const on = sel.includes(c.name);
                return (
                  <button type="button" key={c.name} onClick={() => set(p.name, on ? sel.filter((x) => x !== c.name) : [...sel, c.name].slice(0, p.max ?? 12))}
                    className={`chip border transition ${on ? "border-pink-300 bg-pink-500 text-white" : "border-white bg-white/80 text-ink-soft hover:text-ink"}`}>
                    {c.name}
                  </button>);
              })}
            </div>);
        } else if (p.type === "image") {
          input = (
            <div className="flex gap-2 overflow-x-auto pb-1 scroll-thin">
              {dataset.images?.map((im) => (
                <button type="button" key={im.id} onClick={() => set(p.name, im.id)} title={im.file}
                  className={`relative w-24 shrink-0 overflow-hidden rounded-xl ring-2 transition ${v === im.id ? "ring-pink-500" : "ring-transparent hover:ring-pink-200"}`}>
                  <img src={`/api/workbench/datasets/${dataset.id}/images/${im.id}/thumb`} alt={im.file} className="aspect-square w-full object-cover" />
                  <span className="absolute inset-x-0 bottom-0 truncate bg-white/85 px-1 text-[9px] font-semibold">{im.file}</span>
                </button>))}
            </div>);
        } else if (p.type === "select") {
          input = (
            <select className="input" value={String(v)} onChange={(e) => set(p.name, e.target.value)}>
              {p.options?.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
            </select>);
        } else if (p.type === "units") {
          const sel = (values.sensor_cols as string[]) ?? [];
          const u = (v as Record<string, string>) ?? {};
          input = sel.length === 0 ? <p className="text-xs text-ink-faint">Select sensor columns first.</p> : (
            <div className="flex flex-wrap gap-2">
              {sel.map((c) => (
                <label key={c} className="flex items-center gap-1.5 rounded-xl border border-white/80 bg-white/50 py-1 pl-2.5 pr-1 text-xs">
                  <span className="font-semibold">{c}</span>
                  <input className="w-20 rounded-lg bg-white/80 px-2 py-1 text-xs outline-none" placeholder="unit"
                    value={u[c] ?? ""} onChange={(e) => set(p.name, { ...u, [c]: e.target.value })} />
                </label>))}
            </div>);
        } else if (p.type === "text") {
          input = <textarea className="input min-h-[44px]" rows={2} value={String(v ?? "")} onChange={(e) => set(p.name, e.target.value)} placeholder="optional" />;
        } else {
          input = (
            <input className="input" type="number" inputMode="decimal" step={p.step ?? (p.type === "integer" ? 1 : "any")} min={p.min} max={p.max}
              value={v == null ? "" : String(v)} placeholder={p.optional ? "optional" : ""}
              onChange={(e) => set(p.name, e.target.value === "" ? null : Number(e.target.value))} />);
        }
        return <div key={p.name} className={wide(p) ? "md:col-span-2 xl:col-span-3" : ""}><Field p={p}>{input}</Field></div>;
      })}
      {table && (
        <div className="md:col-span-2 xl:col-span-3 flex flex-wrap gap-1.5 text-[10px]">
          {Object.entries(TYPE_CHIP).map(([k, c]) => <span key={k} className={`chip ${c}`}>{k}</span>)}
          <span className="text-ink-faint">column types detected on import</span>
        </div>
      )}
    </div>
  );
}

export { TYPE_CHIP };
