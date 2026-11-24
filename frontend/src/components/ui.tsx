import { motion, type HTMLMotionProps } from "framer-motion";
import { AlertTriangle, Check, CheckCircle2, Copy, Loader2, Terminal, XCircle } from "lucide-react";
import { useState, type ReactNode } from "react";
import { usePrefs } from "../lib/theme";
import { HintAction, hintToTask } from "./Setup";
import type { RunStatus } from "../lib/types";

export const stagger = {
  hidden: {},
  show: { transition: { staggerChildren: 0.03 } },
};
export const rise = {
  hidden: { opacity: 0, y: 8 },
  show: { opacity: 1, y: 0, transition: { duration: 0.18, ease: "easeOut" } },
};

export function Glass({ className = "", hover = false, children, ...rest }: HTMLMotionProps<"div"> & { hover?: boolean }) {
  return (
    <motion.div
      variants={rise}
      whileHover={hover ? { y: -4, boxShadow: "0 22px 50px rgba(31,41,115,0.16)" } : undefined}
      transition={{ type: "spring", stiffness: 300, damping: 22 }}
      className={`glass ${className}`}
      {...rest}
    >
      {children}
    </motion.div>
  );
}

export function Page({ title, subtitle, icon, actions, children }: {
  title: string; subtitle?: ReactNode; icon?: ReactNode; actions?: ReactNode; children: ReactNode;
}) {
  return (
    <motion.div initial="hidden" animate="show" variants={stagger} className="space-y-5">
      <motion.header variants={rise} className="flex flex-wrap items-end justify-between gap-4">
        <div className="flex items-center gap-4">
          {icon}
          <div>
            <h1 className="text-2xl font-bold text-ink sm:text-3xl">{title}</h1>
            {subtitle && <p className="mt-1 max-w-3xl text-sm text-ink-soft">{subtitle}</p>}
          </div>
        </div>
        {actions}
      </motion.header>
      {children}
    </motion.div>
  );
}

export function IconBadge({ gradient, children }: { gradient: string; children: ReactNode }) {
  return (
    <div className={`grid h-12 w-12 place-items-center rounded-2xl bg-gradient-to-br ${gradient} text-white shadow-lg`}>
      {children}
    </div>
  );
}

export function Spinner({ className = "h-4 w-4" }: { className?: string }) {
  return <Loader2 className={`animate-spin ${className}`} />;
}

const STATUS_STYLE: Record<RunStatus, [string, string]> = {
  queued: ["bg-slate-100 text-slate-600", "queued"],
  running: ["bg-sky-100 text-sky-700", "running"],
  succeeded: ["bg-emerald-100 text-emerald-700", "completed"],
  failed: ["bg-rose-100 text-rose-700", "failed"],
  interrupted: ["bg-amber-100 text-amber-800", "interrupted"],
  source_unavailable: ["bg-orange-100 text-orange-800", "source unavailable"],
  insufficient_data: ["bg-amber-100 text-amber-800", "insufficient data"],
};

export function StatusBadge({ status }: { status: RunStatus }) {
  const [cls, label] = STATUS_STYLE[status] ?? ["bg-slate-100 text-slate-600", status];
  return (
    <span className={`chip ${cls}`}>
      {status === "running" || status === "queued" ? <Spinner className="h-3 w-3" /> :
        status === "succeeded" ? <CheckCircle2 className="h-3 w-3" /> : status === "failed" ? <XCircle className="h-3 w-3" /> :
        <AlertTriangle className="h-3 w-3" />}
      {label}
    </span>
  );
}

/** "Bundled reference" (shipped in the starter pack) vs "Computed here". */
export function OriginBadge({ origin }: { origin?: string | null }) {
  return origin === "bundled"
    ? <span className="chip bg-indigo-100 text-indigo-700" title="Reference run shipped in the starter pack">bundled reference</span>
    : <span className="chip bg-teal-100 text-teal-700" title="Computed on this machine">computed here</span>;
}

function CommandLine({ cmd }: { cmd: string }) {
  const [done, setDone] = useState(false);
  return (
    <button
      onClick={() => {
        navigator.clipboard?.writeText(cmd);
        setDone(true);
        setTimeout(() => setDone(false), 1400);
      }}
      className="group flex w-full items-center gap-2 rounded-xl border border-white/80 bg-slate-900/[0.04] px-3 py-2 text-left font-mono text-xs text-ink transition hover:bg-white/80 dark:bg-black/30"
      title="Copy command"
    >
      <Terminal className="h-3.5 w-3.5 shrink-0 text-emerald-500" />
      <span className="flex-1 break-all">{cmd}</span>
      {done ? <Check className="h-3.5 w-3.5 text-emerald-600" /> : <Copy className="h-3.5 w-3.5 text-ink-faint group-hover:text-ink" />}
    </button>
  );
}

/**
 * Terminal command with copy button. Hidden behind a small "terminal" toggle unless the user
 * switched "show terminal commands" on in the dock (or `always` is set), so pages stay clean.
 */
export function CopyCommand({ cmd, always = false }: { cmd: string; always?: boolean }) {
  const { showCommands } = usePrefs();
  const [open, setOpen] = useState(false);
  if (always || showCommands || open) return <CommandLine cmd={cmd} />;
  return (
    <button onClick={() => setOpen(true)}
      className="inline-flex items-center gap-1.5 rounded-full border border-white/80 bg-white/50 px-2.5 py-1 font-mono text-[10px] font-semibold text-ink-faint transition hover:text-ink">
      <Terminal className="h-3 w-3" /> terminal command
    </button>
  );
}

export function Notice({ tone = "warn", title, children }: { tone?: "warn" | "error" | "info"; title: string; children?: ReactNode }) {
  const tones = {
    warn: "border-amber-200/80 bg-amber-50/80 text-amber-900",
    error: "border-rose-200/80 bg-rose-50/80 text-rose-900",
    info: "border-sky-200/80 bg-sky-50/80 text-sky-900",
  };
  return (
    <div role={tone === "error" ? "alert" : "status"} className={`rounded-2xl border p-4 text-sm ${tones[tone]}`}>
      <div className="flex items-center gap-2 font-semibold">
        <AlertTriangle className="h-4 w-4" aria-hidden="true" />
        {title}
      </div>
      {children && <div className="mt-2 space-y-2 text-[13px] leading-relaxed opacity-90">{children}</div>}
    </div>
  );
}

/** Shows an API error; if it contains a shell command hint ("Run: ..."), renders it copyable. */
export function ErrorNotice({ message, title = "Something needs attention" }: { message: string; title?: string }) {
  const m = message.match(/Run:\s*(.+)$/);
  const action = m ? hintToTask(m[1]) : null;
  return (
    <Notice tone={action ? "warn" : "error"} title={title}>
      <p>{m ? message.slice(0, m.index).trim() : message}</p>
      {action ? <HintAction cmd={m![1]} /> : m && !/^(task|setup):/.test(m[1]) && <CopyCommand cmd={m[1]} />}
    </Notice>
  );
}

export function Kpi({ label, value, sub, accent }: { label: string; value: ReactNode; sub?: ReactNode; accent?: string }) {
  return (
    <div className="glass-inset p-4">
      <div className="label">{label}</div>
      <div className="mt-1 font-display text-2xl font-bold" style={accent ? { color: accent } : undefined}>{value}</div>
      {sub && <div className="mt-0.5 text-xs text-ink-soft">{sub}</div>}
    </div>
  );
}

export function Segmented<T extends string>({ value, options, onChange, label }: {
  value: T; options: { value: T; label: ReactNode }[]; onChange: (v: T) => void; label?: string;
}) {
  // A radio group: Tab focuses the selected option, arrow keys move the selection (WAI-ARIA radio pattern).
  const move = (e: React.KeyboardEvent<HTMLDivElement>) => {
    const i = options.findIndex((o) => o.value === value);
    const d = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 0;
    if (!d) return;
    e.preventDefault();
    const n = options[(i + d + options.length) % options.length];
    onChange(n.value);
    window.requestAnimationFrame(() => (e.currentTarget.querySelector(`[data-v="${CSS.escape(n.value)}"]`) as HTMLElement | null)?.focus());
  };
  return (
    <div role="radiogroup" aria-label={label} onKeyDown={move} className="inline-flex flex-wrap rounded-2xl border border-white/80 bg-white/50 p-1">
      {options.map((o) => (
        <button
          key={o.value}
          type="button" role="radio" aria-checked={value === o.value} tabIndex={value === o.value ? 0 : -1} data-v={o.value}
          onClick={() => onChange(o.value)}
          className={`relative rounded-xl px-3.5 py-1.5 text-sm font-semibold transition ${value === o.value ? "text-ink" : "text-ink-soft hover:text-ink"}`}
        >
          {value === o.value && (
            <motion.span layoutId={`seg-${options.map((x) => x.value).join("")}`} className="absolute inset-0 rounded-xl bg-white shadow"
              transition={{ type: "spring", stiffness: 400, damping: 30 }} />
          )}
          <span className="relative">{o.label}</span>
        </button>
      ))}
    </div>
  );
}

export function RunProgress({ status, label }: { status: RunStatus; label: string }) {
  return (
    <div className="flex items-center gap-3 rounded-2xl border border-white/80 bg-white/60 px-4 py-3">
      <Spinner className="h-5 w-5 text-violet-500" />
      <div className="text-sm">
        <div className="font-semibold">{status === "queued" ? "Queued" : "Running inference"} · {label}</div>
        <div className="text-xs text-ink-soft">The background worker is loading the model and processing the data…</div>
      </div>
    </div>
  );
}
