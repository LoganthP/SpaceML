import { lazy, Suspense, type ComponentProps } from "react";
import { Spinner } from "./ui";

const Inner = lazy(() => import("./Plot"));

export default function LazyPlot(props: ComponentProps<typeof Inner>) {
  return (
    <Suspense fallback={<div className="flex items-center justify-center gap-2 text-sm text-ink-soft" style={{ height: props.height ?? 360 }}><Spinner /> Loading chart…</div>}>
      <Inner {...props} />
    </Suspense>
  );
}
