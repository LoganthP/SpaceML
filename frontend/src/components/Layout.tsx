import { motion } from "framer-motion";
import { ChevronDown, Command, Menu as MenuIcon, Moon, Search, Sun, SquareTerminal } from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { NavLink, useLocation, useNavigate } from "react-router-dom";
import { useCapabilities, useReadOnly } from "../lib/capabilities";
import { MOBILE_NAV, NAV_GROUPS } from "../lib/nav";
import { usePrefs } from "../lib/theme";
import CommandPalette from "./CommandPalette";
import SpaceBackdrop from "./SpaceBackdrop";
import { useJobs } from "./Jobs";



function NavMenu({ group }: { group: (typeof NAV_GROUPS)[number] }) {
  const loc = useLocation();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const active = group.items.some((n) => (n.end ? loc.pathname === n.to : loc.pathname.startsWith(n.to) && n.to !== "/"));
  useEffect(() => { setOpen(false); }, [loc.pathname, loc.search]);
  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => { if (!ref.current?.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [open]);
  const cls = (on: boolean) => `flex h-10 items-center gap-1.5 whitespace-nowrap rounded-xl px-3 text-sm font-semibold transition ${on ? "bg-slate-900 text-white dark:bg-indigo-600" : "text-ink-soft hover:bg-white/80 hover:text-ink dark:hover:bg-white/10"}`;
  if (group.items.length === 1) {
    const n = group.items[0];
    return <NavLink to={n.to} end={n.end} className={({ isActive }) => cls(isActive)}><n.icon className="h-4 w-4" aria-hidden="true" />{group.label}</NavLink>;
  }
  return (
    <div ref={ref} className="relative" onKeyDown={(e) => { if (e.key === "Escape") { setOpen(false); (ref.current?.querySelector("button") as HTMLElement | null)?.focus(); } }}>
      <button type="button" className={cls(active)} aria-expanded={open} aria-haspopup="true" onClick={() => setOpen((o) => !o)}>
        {group.label}<ChevronDown className={`h-4 w-4 transition ${open ? "rotate-180" : ""}`} aria-hidden="true" />
      </button>
      {open && (
        <div className="absolute left-0 top-full z-50 mt-2 w-80 rounded-2xl border border-white/80 bg-white/95 p-2 shadow-glass-lg backdrop-blur-2xl dark:border-white/10 dark:bg-[#11152f]/95">
          <ul className="space-y-0.5">
            {group.items.map((n) => (
              <li key={n.to}>
                <NavLink to={n.to} end={n.end} className={({ isActive }) => `flex items-start gap-3 rounded-xl px-3 py-2 ${isActive ? "bg-indigo-50 dark:bg-white/10" : "hover:bg-slate-100 dark:hover:bg-white/5"}`}>
                  <span className={`grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-gradient-to-br ${n.grad} text-white`} aria-hidden="true"><n.icon className="h-4 w-4" /></span>
                  <span><span className="block text-sm font-semibold text-ink">{n.label}</span><span className="block text-xs text-ink-soft">{n.hint}</span></span>
                </NavLink>
              </li>))}
          </ul>
        </div>)}
    </div>
  );
}

function MissionClock() {
  const [now, setNow] = useState(new Date());
  useEffect(() => { const t = setInterval(() => setNow(new Date()), 1000); return () => clearInterval(t); }, []);
  return (
    <div className="hidden text-right leading-tight xl:block">
      <div className="font-mono text-[13px] font-semibold tabular-nums text-ink">{now.toISOString().slice(11, 19)} UTC</div>
      <div className="text-[10px] font-medium uppercase tracking-wider text-ink-faint">Mission time · {now.toISOString().slice(0, 10)}</div>
    </div>
  );
}

function JobsRing() {
  const { jobs } = useJobs();
  const nav = useNavigate();
  const active = jobs.filter((j) => j.status === "running" || j.status === "queued");
  const p = active.length ? active.reduce((a, j) => a + (j.progress ?? 0), 0) / active.length : 0;
  const r = 15, c = 2 * Math.PI * r;
  return (
    <button onClick={() => nav("/datasets")} title={active.length ? active.map((j) => j.title).join("\n") : "No background jobs running"}
      className="relative grid h-11 w-11 place-items-center rounded-2xl bg-white/70 transition hover:bg-white">
      <svg viewBox="0 0 40 40" className="absolute inset-0 h-full w-full -rotate-90">
        <circle cx="20" cy="20" r={r} fill="none" stroke="rgba(148,163,184,0.25)" strokeWidth="3" />
        {active.length > 0 && <circle cx="20" cy="20" r={r} fill="none" stroke="url(#jg)" strokeWidth="3" strokeLinecap="round"
          strokeDasharray={c} strokeDashoffset={c * (1 - Math.max(p, 0.04))} className="transition-all duration-700" />}
        <defs><linearGradient id="jg"><stop offset="0" stopColor="#22d3ee" /><stop offset="1" stopColor="#a855f7" /></linearGradient></defs>
      </svg>
      <span className={`relative text-xs font-bold ${active.length ? "text-violet-700" : "text-ink-faint"}`}>{active.length || "·"}</span>
      {active.length > 0 && <span className="absolute -right-0.5 -top-0.5 h-2.5 w-2.5 animate-ping rounded-full bg-violet-400" />}
    </button>
  );
}

function ThemeToggle() {
  const { theme, toggleTheme } = usePrefs();
  const dark = theme === "dark";
  return (
    <button onClick={toggleTheme} title={dark ? "Switch to light theme" : "Switch to dark theme"} aria-label="Toggle theme"
      className="relative h-11 w-[74px] rounded-2xl bg-white/70 p-1 transition hover:bg-white">
      <span className="absolute inset-1 flex items-center justify-between px-2 text-ink-faint"><Sun className="h-4 w-4" /><Moon className="h-4 w-4" /></span>
      <motion.span layout transition={{ type: "spring", stiffness: 500, damping: 32 }}
        className={`relative grid h-9 w-9 place-items-center rounded-xl shadow-md ${dark ? "ml-auto bg-gradient-to-br from-indigo-500 to-violet-700 text-white" : "bg-gradient-to-br from-amber-300 to-orange-400 text-white"}`}>
        {dark ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
      </motion.span>
    </button>
  );
}

function CommandsToggle() {
  const { showCommands, setShowCommands } = usePrefs();
  return (
    <button onClick={() => setShowCommands(!showCommands)} aria-pressed={showCommands}
      title={showCommands ? "Hide terminal commands" : "Show terminal commands next to every action"}
      className={`hidden h-11 w-11 place-items-center rounded-2xl transition lg:grid ${showCommands ? "bg-gradient-to-br from-slate-700 to-slate-900 text-emerald-300 shadow-md" : "bg-white/70 text-ink-faint hover:bg-white hover:text-ink"}`}>
      <SquareTerminal className="h-[18px] w-[18px]" />
    </button>
  );
}

function ApiPulse() {
  const caps = useCapabilities();
  const data = caps && !caps.error ? caps : null;
  const error = caps?.error;
  return (
    <span title={data ? "API connected" : error ? "API offline - start the backend on port 8000" : "Connecting…"}
      className={`h-2.5 w-2.5 rounded-full ${data ? "bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.9)]" : error ? "bg-rose-500" : "bg-amber-400"}`} />
  );
}

function Dock({ onSearch }: { onSearch: () => void }) {
  const readOnly = useReadOnly();
  const caps = useCapabilities();
  return (
    <header className="pointer-events-none sticky top-0 z-40 px-3 pt-3">
      <motion.div initial={{ y: -40, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ type: "spring", stiffness: 200, damping: 24 }}
        style={{ backgroundColor: "rgb(var(--glass-rgb) / 0.93)" }}
        className="pointer-events-auto mx-auto flex max-w-[1400px] items-center gap-3 rounded-[28px] border border-white/80 px-3 py-2 shadow-glass-lg backdrop-blur-2xl dark:border-white/10">
        <NavLink to="/" className="flex items-center gap-2.5 pr-1">
          <div className="relative h-11 w-11">
            <div className="absolute inset-0 animate-[spin_8s_linear_infinite] rounded-full bg-[conic-gradient(from_0deg,#22d3ee,#a855f7,#f97316,#22d3ee)] p-[2px]">
              <div className="h-full w-full rounded-full bg-white/90 dark:bg-[#0d1130]" />
            </div>
            <div className="absolute inset-[9px] rounded-full bg-gradient-to-br from-cyan-400 via-violet-500 to-orange-400 shadow-[0_0_14px_rgba(168,85,247,0.6)]" />
          </div>
          <div className="hidden leading-tight sm:block">
            <div className="font-display text-[15px] font-bold">Space ML</div>
            <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-ink-faint">{readOnly ? "Read-only demo" : "Mission Control"}</div>
          </div>
        </NavLink>
        {caps?.authentication?.required && !caps.authentication.authenticated && <a href="/api/session" className="btn-ghost text-xs">Sign in to train</a>}
        <nav aria-label="Main" className="mx-auto hidden items-center gap-1 rounded-[20px] bg-white/40 p-1 dark:bg-white/5 lg:flex">
          {NAV_GROUPS.map((g) => <NavMenu key={g.label} group={g} />)}
        </nav>
        <div className="ml-auto flex items-center gap-2 lg:ml-0">
          <button onClick={onSearch} className="flex h-11 items-center gap-2 rounded-2xl bg-white/70 px-3 text-sm text-ink-soft transition hover:bg-white hover:text-ink">
            <Search className="h-4 w-4" />
            <span className="hidden lg:inline">Search</span>
            <kbd className="hidden items-center gap-0.5 rounded-md border border-slate-200 bg-white px-1.5 text-[10px] font-semibold lg:flex"><Command className="h-2.5 w-2.5" />K</kbd>
          </button>
          <JobsRing />
          <CommandsToggle />
          <ThemeToggle />
          <MissionClock />
          <ApiPulse />
        </div>
      </motion.div>
    </header>
  );
}

function MobileDock() {
  const [open, setOpen] = useState(false);
  const loc = useLocation();
  useEffect(() => { setOpen(false); }, [loc.pathname]);
  return (
    <>
      {open && (
        <div className="fixed inset-0 z-50 bg-slate-950/40 backdrop-blur-sm lg:hidden" onClick={() => setOpen(false)}>
          <nav aria-label="All pages" onClick={(e) => e.stopPropagation()}
            className="scroll-thin absolute inset-x-3 bottom-24 max-h-[70vh] overflow-y-auto rounded-[24px] border border-white/80 bg-white p-3 shadow-glass-lg dark:border-white/10 dark:bg-[#11152f]">
            {NAV_GROUPS.map((g) => (
              <div key={g.label} className="mb-2">
                <div className="kicker px-2 pb-1">{g.label}</div>
                {g.items.map((n) => (
                  <NavLink key={n.to} to={n.to} end={n.end} className={({ isActive }) => `flex items-center gap-3 rounded-xl px-2 py-2 text-sm font-semibold ${isActive ? "bg-indigo-50 dark:bg-white/10" : ""}`}>
                    <n.icon className="h-4 w-4 text-indigo-600 dark:text-indigo-300" aria-hidden="true" />{n.label}
                  </NavLink>))}
              </div>))}
          </nav>
        </div>)}
      <nav aria-label="Main" className="fixed inset-x-3 bottom-3 z-40 grid grid-cols-5 gap-1 rounded-[24px] border border-white/80 bg-white/90 p-1.5 shadow-glass-lg backdrop-blur-2xl dark:border-white/10 dark:bg-[#11152f]/90 lg:hidden">
        {MOBILE_NAV.map(({ to, icon: Icon, end, label }) => (
          <NavLink key={to} to={to} end={end}
            className={({ isActive }) => `flex flex-col items-center gap-0.5 rounded-2xl py-1.5 text-[10px] font-semibold leading-tight ${isActive ? "bg-slate-900 text-white dark:bg-indigo-600" : "text-ink-soft"}`}>
            <Icon className="h-5 w-5" aria-hidden="true" /><span className="text-center">{label.replace(" & Missions", "").replace(" & Training", "").replace(" Studio", "")}</span>
          </NavLink>))}
        <button type="button" onClick={() => setOpen((o) => !o)} aria-expanded={open}
          className={`flex flex-col items-center gap-0.5 rounded-2xl py-1.5 text-[10px] font-semibold ${open ? "bg-slate-900 text-white" : "text-ink-soft"}`}>
          <MenuIcon className="h-5 w-5" aria-hidden="true" />Menu
        </button>
      </nav>
    </>
  );
}

export default function Layout({ children }: { children: ReactNode }) {
  const readOnly = useReadOnly();
  const [palette, setPalette] = useState(false);
  const loc = useLocation();
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); setPalette((p) => !p); }
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, []);
  return (
    <div className="min-h-screen">
      <SpaceBackdrop />
      <Dock onSearch={() => setPalette(true)} />
      <main key={loc.pathname} className="px-4 pb-28 pt-6 sm:px-6 lg:pb-12">
        <div className="h-5 text-xs font-semibold text-ink-soft sm:hidden">{readOnly ? "Read-only demo" : ""}</div>
        <div className="mx-auto max-w-[1400px]">{children}</div>
      </main>
      <MobileDock />
      <CommandPalette open={palette} onClose={() => setPalette(false)} />
    </div>
  );
}
