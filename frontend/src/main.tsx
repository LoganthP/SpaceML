import "@fontsource-variable/inter";
import "@fontsource-variable/space-grotesk";
import "@fontsource-variable/jetbrains-mono";
import React, { lazy, Suspense } from "react";
import ReactDOM from "react-dom/client";
import { MotionConfig } from "framer-motion";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import Boundary, { PageBoundary } from "./components/ErrorBoundary";
import { JobsProvider } from "./components/Jobs";
import { CapabilitiesProvider } from "./lib/capabilities";
import Layout from "./components/Layout";
import { Spinner } from "./components/ui";
import "./index.css";
import "./theme-dark.css";
import { applyStoredTheme, PrefsProvider } from "./lib/theme";
const Datasets = lazy(() => import("./pages/Datasets"));
const Evaluation = lazy(() => import("./pages/Evaluation"));
const Exoplanet = lazy(() => import("./pages/Exoplanet"));
const Mars = lazy(() => import("./pages/Mars"));
const Overview = lazy(() => import("./pages/Overview"));
const Runs = lazy(() => import("./pages/Runs"));
const Telemetry = lazy(() => import("./pages/TelemetryHub"));
const Workbench = lazy(() => import("./pages/Workbench"));
const SpaceWeather = lazy(() => import("./pages/SpaceWeather"));
const ReviewStudio = lazy(() => import("./pages/ReviewStudio"));
const MissionReplay = lazy(() => import("./pages/MissionReplay"));
const Traverse = lazy(() => import("./pages/Traverse"));

applyStoredTheme();
window.addEventListener("load", () => window.setTimeout(() => sessionStorage.removeItem("smc-chunk-reload"), 10000));

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Boundary name="app">
    <MotionConfig reducedMotion="user">
    <PrefsProvider>
    <BrowserRouter>
      <CapabilitiesProvider>
      <JobsProvider>
      <Layout>
        <PageBoundary>
        <Suspense fallback={<div className="flex items-center gap-2 p-6 text-sm text-ink-soft" role="status"><Spinner /> Loading workspace…</div>}>
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/telemetry" element={<Telemetry />} />
          <Route path="/exoplanet" element={<Exoplanet />} />
          <Route path="/mars" element={<Mars />} />
          <Route path="/mars/traverse" element={<Traverse />} />
          <Route path="/replay" element={<MissionReplay />} />
          <Route path="/workbench" element={<Workbench />} />
          <Route path="/space-weather" element={<SpaceWeather />} />
          <Route path="/review" element={<ReviewStudio />} />
          <Route path="/runs" element={<Runs />} />
          <Route path="/evaluation" element={<Evaluation />} />
          <Route path="/datasets" element={<Datasets />} />
          <Route path="*" element={<Overview />} />
        </Routes>
        </Suspense>
        </PageBoundary>
      </Layout>
      </JobsProvider>
      </CapabilitiesProvider>
    </BrowserRouter>
    </PrefsProvider>
    </MotionConfig>
    </Boundary>
  </React.StrictMode>,
);
