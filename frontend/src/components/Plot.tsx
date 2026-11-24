import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";
import type { PlotParams } from "react-plotly.js";
import type { Data, Layout } from "plotly.js";
import { palette, useTheme } from "../lib/theme";

const PlotlyComponent = createPlotlyComponent(Plotly) as React.ComponentType<PlotParams>;

function baseLayout(theme: "light" | "dark"): Partial<Layout> {
  const c = palette(theme);
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: c.plotBg,
    font: { family: "Inter Variable, system-ui, sans-serif", size: 12, color: c.text },
    margin: { l: 56, r: 18, t: 16, b: 44 },
    hoverlabel: { bgcolor: c.hoverBg, bordercolor: c.hoverBorder, font: { color: c.text } },
    legend: { orientation: "h", y: 1.12, x: 0, bgcolor: "rgba(0,0,0,0)" },
    xaxis: { gridcolor: c.grid, zeroline: false, linecolor: c.axis },
    yaxis: { gridcolor: c.grid, zeroline: false, linecolor: c.axis },
  };
}

export default function Plot({ data, layout, height = 360, className = "", onClick }: {
  data: Data[]; layout?: Partial<Layout>; height?: number; className?: string;
  onClick?: (e: Readonly<import("plotly.js").PlotMouseEvent>) => void;
}) {
  const theme = useTheme();
  const base = baseLayout(theme);
  const c = palette(theme);
  const extraAxes = Object.fromEntries(Object.entries(layout ?? {}).filter(([k]) => /^[xy]axis\d+$/.test(k))
    .map(([k, v]) => [k, { gridcolor: c.grid, linecolor: c.axis, ...(v as object) }]));
  const merged = {
    ...base,
    ...layout,
    ...extraAxes,
    xaxis: { ...base.xaxis, ...layout?.xaxis },
    yaxis: { ...base.yaxis, ...layout?.yaxis },
    height,
    autosize: true,
  } as Partial<Layout>;
  return (
    <PlotlyComponent
      className={`w-full ${className}`}
      data={data}
      layout={merged}
      useResizeHandler
      onClick={onClick}
      style={{ width: "100%", height }}
      config={{ displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d"] }}
    />
  );
}
