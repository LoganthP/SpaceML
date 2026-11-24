import { AlertTriangle, RefreshCw } from "lucide-react";
import { Component, type ReactNode } from "react";
import { useLocation } from "react-router-dom";

interface State { error: Error | null; info: string }

/** A crash inside one page shows a recoverable card instead of blanking the whole app. */
class Boundary extends Component<{ children: ReactNode; name?: string }, State> {
  state: State = { error: null, info: "" };
  static getDerivedStateFromError(error: Error): Partial<State> { return { error }; }
  componentDidCatch(error: Error, info: { componentStack?: string | null }) {
    // after a redeploy, old lazy chunks no longer exist: reload once to fetch the new ones
    if (/dynamically imported module|Loading chunk|Importing a module script failed/i.test(error.message)
        && !sessionStorage.getItem("smc-chunk-reload")) {
      sessionStorage.setItem("smc-chunk-reload", "1");
      window.location.reload();
      return;
    }
    this.setState({ info: info.componentStack ?? "" });
    console.error("[Space ML] view crashed:", error, info.componentStack);
  }
  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="glass mx-auto my-6 max-w-3xl space-y-3 p-6">
        <div className="flex items-center gap-2 text-rose-600"><AlertTriangle className="h-5 w-5" />
          <h2 className="font-display text-lg font-bold">This {this.props.name ?? "view"} hit an error</h2></div>
        <p className="text-sm text-ink-soft">The rest of the app still works - use the navigation bar, or try again. Your data and running jobs are not affected.</p>
        <pre className="scroll-thin max-h-40 overflow-auto rounded-xl bg-slate-900/90 p-3 text-[11px] text-rose-200">{this.state.error.message}</pre>
        <div className="flex gap-2">
          <button className="btn-primary px-3 py-1.5 text-xs" onClick={() => this.setState({ error: null, info: "" })}><RefreshCw className="h-3.5 w-3.5" /> Try again</button>
          <button className="btn-ghost px-3 py-1.5 text-xs" onClick={() => window.location.reload()}>Reload page</button>
        </div>
        {this.state.info && <details className="text-[10px] text-ink-faint"><summary>Technical details</summary><pre className="whitespace-pre-wrap">{this.state.info.trim().split("\n").slice(0, 12).join("\n")}</pre></details>}
      </div>
    );
  }
}

export function PageBoundary({ children }: { children: ReactNode }) {
  const loc = useLocation();
  return <Boundary key={loc.pathname} name="page">{children}</Boundary>;
}

export function SectionBoundary({ children, name }: { children: ReactNode; name: string }) {
  return <Boundary name={name}>{children}</Boundary>;
}

export default Boundary;
