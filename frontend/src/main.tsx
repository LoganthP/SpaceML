import "@fontsource-variable/inter";
import "@fontsource-variable/space-grotesk";
import "@fontsource-variable/jetbrains-mono";
import React from "react";
import ReactDOM from "react-dom/client";
import { MotionConfig } from "framer-motion";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import Boundary, { PageBoundary } from "./components/ErrorBoundary";
import { JobsProvider } from "./components/Jobs";
import Layout from "./components/Layout";
import "./index.css";
import "./theme-dark.css";
import { applyStoredTheme, PrefsProvider } from "./lib/theme";
import Datasets from "./pages/Datasets";
import Evaluation from "./pages/Evaluation";
import Exoplanet from "./pages/Exoplanet";
import Mars from "./pages/Mars";
import Overview from "./pages/Overview";
import Runs from "./pages/Runs";
import Telemetry from "./pages/TelemetryHub";
import Workbench from "./pages/Workbench";
import SpaceWeather from "./pages/SpaceWeather";
import ReviewStudio from "./pages/ReviewStudio";
import MissionReplay from "./pages/MissionReplay";
import Traverse from "./pages/Traverse";
import { Spinner } from "./components/ui";

const Welcome = React.lazy(() => import("./pages/Welcome"));

applyStoredTheme();
window.addEventListener("load", () => window.setTimeout(() => sessionStorage.removeItem("smc-chunk-reload"), 10000));

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Boundary name="app">
    <MotionConfig reducedMotion="user">
    <PrefsProvider>
    <BrowserRouter>
      <JobsProvider>
      <Layout>
        <PageBoundary>
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/welcome" element={<React.Suspense fallback={<div className="flex min-h-[50vh] items-center justify-center"><Spinner className="h-8 w-8 text-violet-500" /></div>}><Welcome /></React.Suspense>} />
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
        </PageBoundary>
      </Layout>
      </JobsProvider>
    </BrowserRouter>
    </PrefsProvider>
    </MotionConfig>
    </Boundary>
  </React.StrictMode>,
);
