import { createContext, forwardRef, useContext, useEffect, useState, type ButtonHTMLAttributes, type InputHTMLAttributes, type ReactNode } from "react";

interface Capabilities { checks?: { sandbox: { available: boolean }; drive: { mounted?: boolean; writable: boolean | null; name?: string } }; startup_error?: { stage: string; type: string } | null; detail?: string; mode: "local" | "vercel"; execution?: "local" | "static" | "managed"; state?: string; hint?: string; setup_required?: string[]; error?: boolean; authentication?: { required: boolean; authenticated: boolean }; stored_inputs?: { module: string; input: Record<string, unknown> }[]; precomputed_modules?: string[]; capabilities: { write: boolean; train: boolean; live: boolean } }
const Context = createContext<Capabilities | null>(null);
let health: Promise<Capabilities> | undefined;
export const refreshCapabilities = () => window.dispatchEvent(new Event("spaceml:refresh-capabilities"));
export const READ_ONLY_HINT = "Vercel compute required";
export const useReadOnly = () => { const caps = useCapabilities(); return caps?.mode === "vercel" && caps.execution !== "managed"; };
export const useOnVercel = () => useCapabilities()?.mode === "vercel";

export function CapabilitiesProvider({ children }: { children: ReactNode }) {
  const [value, setValue] = useState<Capabilities | null>(null);
  useEffect(() => {
    const fetchHealth = () => fetch("/api/health").then(async (response) => {
      if (!response.ok) throw new Error("Health unavailable");
      return response.json() as Promise<Capabilities>;
    });
    let active = true;
    let timer = 0;
    let generation = 0;
    const check = () => {
      const current = ++generation;
      health ??= fetchHealth();
      health.then((v) => {
        if (!active || current !== generation) return;
        setValue(v);
        if (v.execution === "managed") {
          const delay = ["starting", "updating"].includes(v.state ?? "") ? 5000 : 15000;
          timer = window.setTimeout(() => { health = undefined; check(); }, delay);
        }
      }).catch(() => {
        if (active && current === generation) {
          setValue((previous) => ({ ...previous, mode: previous?.mode ?? "local", error: true,
            hint: "API connection unavailable; retrying", capabilities: { write: false, train: false, live: false } }));
          timer = window.setTimeout(() => { health = undefined; check(); }, 15000);
        }
      });
    };
    const refresh = () => { window.clearTimeout(timer); health = undefined; check(); };
    window.addEventListener("spaceml:refresh-capabilities", refresh);
    check();
    return () => { active = false; window.clearTimeout(timer); window.removeEventListener("spaceml:refresh-capabilities", refresh); };
  }, []);
  return <Context.Provider value={value}>{children}</Context.Provider>;
}

export const useCapabilities = () => useContext(Context);
export function useCanRun(module: string, input: Record<string, unknown>) {
  const caps = useCapabilities();
  return !!caps && !caps.error && (caps.execution === "managed" ? caps.capabilities.write : caps.mode !== "vercel" ? caps.capabilities.write : !!caps.stored_inputs?.some((r) =>
    r.module === module && Object.entries(input).every(([key, value]) => r.input[key] === value)));
}

/** Preserve each button's placement, classes, and local behavior. */
export function WriteButton({ operation = "write", ...props }: ButtonHTMLAttributes<HTMLButtonElement> & { operation?: "write" | "train" }) {
  const caps = useCapabilities();
  const readOnly = caps?.capabilities.write === false || caps?.capabilities[operation] === false;
  return <button {...props} disabled={props.disabled || readOnly || caps === null}
    title={readOnly ? (caps?.hint ?? (caps?.mode === "vercel" ? READ_ONLY_HINT : "Sign in to continue")) : props.title} data-smc-write="true">{props.children}</button>;
}

export const FileInput = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(function FileInput(props, ref) {
  const caps = useCapabilities();
  return <input {...props} ref={ref} disabled={props.disabled || caps?.capabilities.write === false || caps === null}
    title={caps?.capabilities.write === false ? caps.hint ?? READ_ONLY_HINT : props.title} />;
});
