import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";

export type Theme = "light" | "dark";

interface Prefs {
  theme: Theme;
  toggleTheme: () => void;
  /** Show terminal commands everywhere (off: commands are tucked behind a small toggle). */
  showCommands: boolean;
  setShowCommands: (v: boolean) => void;
}

const Ctx = createContext<Prefs>({ theme: "light", toggleTheme: () => {}, showCommands: false, setShowCommands: () => {} });

function read<T extends string>(key: string, fallback: T): T {
  try { return (localStorage.getItem(key) as T) || fallback; } catch { return fallback; }
}
function write(key: string, v: string) {
  try { localStorage.setItem(key, v); } catch { /* private mode */ }
}

/** Applied before React renders (main.tsx) so the first paint already has the right theme. */
export function applyStoredTheme() {
  document.documentElement.classList.toggle("dark", read<Theme>("smc-theme", "light") === "dark");
}

export function PrefsProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>(() => read<Theme>("smc-theme", "light"));
  const [showCommands, setShow] = useState(() => read<string>("smc-commands", "off") === "on");
  useEffect(() => {
    const root = document.documentElement;
    root.classList.add("theme-switching");
    root.classList.toggle("dark", theme === "dark");
    root.style.colorScheme = theme;
    write("smc-theme", theme);
    const t = window.setTimeout(() => root.classList.remove("theme-switching"), 400);
    window.dispatchEvent(new CustomEvent("smc-theme", { detail: theme }));
    return () => window.clearTimeout(t);
  }, [theme]);
  const toggleTheme = useCallback(() => setTheme((t) => (t === "light" ? "dark" : "light")), []);
  const setShowCommands = useCallback((v: boolean) => { setShow(v); write("smc-commands", v ? "on" : "off"); }, []);
  return <Ctx.Provider value={{ theme, toggleTheme, showCommands, setShowCommands }}>{children}</Ctx.Provider>;
}

export const usePrefs = () => useContext(Ctx);
export const useTheme = () => useContext(Ctx).theme;

/** Theme-dependent colours for canvas / WebGL / Plotly, which cannot read Tailwind classes. */
export function palette(theme: Theme) {
  const dark = theme === "dark";
  return {
    dark,
    text: dark ? "#cbd5f5" : "#334155",
    faint: dark ? "#7c87b5" : "#94a3b8",
    grid: dark ? "rgba(165,180,252,0.10)" : "rgba(15,23,42,0.06)",
    axis: dark ? "rgba(165,180,252,0.25)" : "rgba(15,23,42,0.15)",
    plotBg: dark ? "rgba(10,14,40,0.35)" : "rgba(255,255,255,0.35)",
    hoverBg: dark ? "#141a3d" : "#ffffff",
    hoverBorder: dark ? "#3b4380" : "#e2e8f0",
    sceneGrid: dark ? 0x3b4380 : 0xc7d2fe,
    sceneGrid2: dark ? 0x252b5c : 0xe0e7ff,
    points: dark ? 0xc7d2fe : 0xa5b4fc,
  };
}
