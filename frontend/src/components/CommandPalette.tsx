import { AnimatePresence, motion } from "framer-motion";
import { Activity, CornerDownLeft, Moon, Orbit, Play, Search, SquareTerminal, Sun, Upload } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../lib/api";
import { usePrefs } from "../lib/theme";
import type { Channel } from "../lib/types";
import { useJobs } from "./Jobs";
import { NAV } from "../lib/nav";

interface Item { id: string; group: string; label: string; hint?: string; icon: React.ComponentType<{ className?: string }>; run: () => void }

export default function CommandPalette({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { theme, toggleTheme, showCommands, setShowCommands } = usePrefs();
  const nav = useNavigate();
  const { start } = useJobs();
  const [q, setQ] = useState("");
  const [sel, setSel] = useState(0);
  const [channels, setChannels] = useState<Channel[]>([]);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!open) return;
    setQ(""); setSel(0);
    setTimeout(() => input.current?.focus(), 30);
    if (!channels.length) api<Channel[]>("/api/telemetry/channels").then(setChannels).catch(() => {});
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps

  const items = useMemo<Item[]>(() => {
    const go = (to: string) => () => { nav(to); onClose(); };
    const job = (task: string, params = {}) => () => { start(task, params).catch(() => {}); nav("/datasets"); onClose(); };
    const base: Item[] = [
      ...NAV.map((n) => ({ id: n.to, group: "Go to", label: n.label, icon: n.icon, run: go(n.to) })),
      { id: "upload", group: "Actions", label: "Upload light-curve dataset", hint: "FITS · CSV · ZIP", icon: Upload, run: go("/exoplanet?source=upload") },
      { id: "k90", group: "Actions", label: "Analyze Kepler-90", hint: "KIC 11442793", icon: Orbit, run: go("/exoplanet?kic=11442793") },
      { id: "theme", group: "Settings", label: theme === "dark" ? "Switch to light theme" : "Switch to dark theme", icon: theme === "dark" ? Sun : Moon, run: () => { toggleTheme(); onClose(); } },
      { id: "cmds", group: "Settings", label: showCommands ? "Hide terminal commands" : "Show terminal commands everywhere", icon: SquareTerminal, run: () => { setShowCommands(!showCommands); onClose(); } },
      { id: "wb", group: "Actions", label: "Upload any dataset to the Workbench", hint: "CSV · FITS · GeoTIFF · ZIP", icon: Upload, run: go("/workbench") },
      { id: "sw", group: "Actions", label: "Replay the May 2024 Gannon storm", icon: Play, run: go("/space-weather") },
      { id: "j6", group: "Background jobs", label: "Download OMNIWeb space-weather data", icon: Play, run: job("spaceweather.download") },
      { id: "j7", group: "Background jobs", label: "Train Dst forecasters", icon: Play, run: job("spaceweather.train") },
      { id: "j8", group: "Background jobs", label: "Train all telemetry channels (fast)", icon: Play, run: job("telemetry.train") },
      { id: "j1", group: "Background jobs", label: "Download Kepler light curves (300 stars)", icon: Play, run: job("exoplanet.lightcurves", { n_stars: 300 }) },
      { id: "j2", group: "Background jobs", label: "Train exoplanet model", icon: Play, run: job("exoplanet.train") },
      { id: "j3", group: "Background jobs", label: "Re-download Kepler catalog (adds sky positions)", icon: Play, run: job("exoplanet.catalog") },
      { id: "j4", group: "Background jobs", label: "Train quick CPU Mars model", icon: Play, run: job("mars.train_quick") },
      { id: "j5", group: "Background jobs", label: "Evaluate telemetry models", icon: Play, run: job("telemetry.evaluate") },
    ];
    const t = q.trim().toLowerCase();
    const digits = t.replace(/\D/g, "");
    const extra: Item[] = [];
    if (digits.length >= 5) extra.push({ id: "kic", group: "Kepler", label: `Analyze KIC ${digits}`, icon: Orbit, run: go(`/exoplanet?kic=${digits}`) });
    channels.filter((c) => t && c.chan_id.toLowerCase().includes(t)).slice(0, 6).forEach((c) =>
      extra.push({ id: c.chan_id, group: "Telemetry channels", label: `${c.chan_id} · ${c.spacecraft}`, hint: c.has_model ? "model ready" : "no model", icon: Activity, run: go(`/telemetry?chan=${c.chan_id}`) }));
    return [...extra, ...base.filter((i) => !t || `${i.label} ${i.hint ?? ""} ${i.group}`.toLowerCase().includes(t))];
  }, [q, channels, nav, onClose, start, theme, toggleTheme, showCommands, setShowCommands]);

  useEffect(() => setSel(0), [q]);
  const groups = [...new Set(items.map((i) => i.group))];

  return (
    <AnimatePresence>
      {open && (
        <motion.div className="fixed inset-0 z-50 flex items-start justify-center bg-slate-900/20 px-4 pt-[12vh] backdrop-blur-sm"
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={onClose}>
          <motion.div onClick={(e) => e.stopPropagation()} initial={{ y: -20, scale: 0.97 }} animate={{ y: 0, scale: 1 }} exit={{ y: -10, scale: 0.98 }}
            transition={{ type: "spring", stiffness: 380, damping: 30 }} className="glass-strong w-full max-w-xl overflow-hidden p-0">
            <div className="flex items-center gap-3 border-b border-white/70 px-4 py-3">
              <Search className="h-5 w-5 text-ink-faint" />
              <input ref={input} value={q} onChange={(e) => setQ(e.target.value)} placeholder="Jump to a page, channel, Kepler ID or action…"
                className="flex-1 bg-transparent text-sm outline-none placeholder:text-ink-faint"
                onKeyDown={(e) => {
                  if (e.key === "ArrowDown") { e.preventDefault(); setSel((s) => Math.min(s + 1, items.length - 1)); }
                  if (e.key === "ArrowUp") { e.preventDefault(); setSel((s) => Math.max(s - 1, 0)); }
                  if (e.key === "Enter") items[sel]?.run();
                  if (e.key === "Escape") onClose();
                }} />
              <kbd className="rounded-md border border-slate-200 bg-white px-1.5 text-[10px] font-semibold text-ink-faint">ESC</kbd>
            </div>
            <div className="scroll-thin max-h-[52vh] overflow-y-auto p-2">
              {items.length === 0 && <p className="p-4 text-center text-sm text-ink-soft">Nothing matches.</p>}
              {groups.map((g) => (
                <div key={g} className="mb-1">
                  <div className="label px-3 py-1.5">{g}</div>
                  {items.filter((i) => i.group === g).map((i) => {
                    const idx = items.indexOf(i);
                    return (
                      <button key={i.id} onMouseEnter={() => setSel(idx)} onClick={i.run}
                        className={`flex w-full items-center gap-3 rounded-xl px-3 py-2 text-left text-sm transition ${idx === sel ? "bg-white shadow-sm" : ""}`}>
                        <i.icon className="h-4 w-4 text-violet-600" />
                        <span className="flex-1 font-medium">{i.label}</span>
                        {i.hint && <span className="text-xs text-ink-faint">{i.hint}</span>}
                        {idx === sel && <CornerDownLeft className="h-3.5 w-3.5 text-ink-faint" />}
                      </button>
                    );
                  })}
                </div>
              ))}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
