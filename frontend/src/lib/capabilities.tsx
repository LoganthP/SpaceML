import { createContext, forwardRef, useContext, useEffect, useState, type ButtonHTMLAttributes, type InputHTMLAttributes, type ReactNode } from "react";

interface Capabilities { mode: "local" | "vercel"; error?: boolean; authentication?: { required: boolean; authenticated: boolean }; stored_inputs?: { module: string; input: Record<string, unknown> }[]; precomputed_modules?: string[]; capabilities: { write: boolean; train: boolean; live: boolean } }
const Context = createContext<Capabilities | null>(null);
let health: Promise<Capabilities> | undefined;
export const READ_ONLY_HINT = "Available in Docker";
export const useReadOnly = () => useCapabilities()?.mode === "vercel";

export function CapabilitiesProvider({ children }: { children: ReactNode }) {
  const [value, setValue] = useState<Capabilities | null>(null);
  useEffect(() => {
    health ??= fetch("/api/health").then(async (response) => {
      if (!response.ok) throw new Error("Health unavailable");
      return response.json() as Promise<Capabilities>;
    });
    let active = true;
    health.then((v) => { if (active) setValue(v); }).catch(() => {
      if (active) setValue({ mode: "local", error: true, capabilities: { write: false, train: false, live: false } });
    });
    return () => { active = false; };
  }, []);
  return <Context.Provider value={value}>{children}</Context.Provider>;
}

export const useCapabilities = () => useContext(Context);
export function useCanRun(module: string, input: Record<string, unknown>) {
  const caps = useCapabilities();
  return !!caps && !caps.error && (caps.mode !== "vercel" || !!caps.stored_inputs?.some((r) =>
    r.module === module && Object.entries(input).every(([key, value]) => r.input[key] === value)));
}

/** Preserve each button's placement, classes, and local behavior. */
export function WriteButton({ operation = "write", ...props }: ButtonHTMLAttributes<HTMLButtonElement> & { operation?: "write" | "train" }) {
  const caps = useCapabilities();
  const readOnly = caps?.capabilities.write === false || caps?.capabilities[operation] === false;
  return <button {...props} disabled={props.disabled || readOnly || caps === null}
    title={readOnly ? (caps?.mode === "vercel" ? READ_ONLY_HINT : "Sign in to continue") : props.title} data-smc-write="true">{readOnly && caps?.mode === "vercel" ? READ_ONLY_HINT : props.children}</button>;
}

export const FileInput = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(function FileInput(props, ref) {
  const caps = useCapabilities();
  return <input {...props} ref={ref} disabled={props.disabled || caps?.capabilities.write === false || caps === null}
    title={caps?.capabilities.write === false ? READ_ONLY_HINT : props.title} />;
});
