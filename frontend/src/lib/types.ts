export type Module = "telemetry" | "exoplanet" | "mars" | "workbench" | "spaceweather" | "missions";
export type ModelModule = "telemetry" | "exoplanet" | "mars" | "spaceweather";
export type RunStatus = "queued" | "running" | "succeeded" | "failed" | "interrupted" | "source_unavailable" | "insufficient_data";

export interface Run<R = unknown> {
  id: string;
  module: Module;
  input_ref: Record<string, unknown>;
  input_label: string;
  model_version: string | null;
  status: RunStatus;
  error: string | null;
  summary: Record<string, unknown> & { headline?: string } | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  duration_ms: number | null;
  origin?: "computed" | "bundled";
  repeat_of?: string | null;
  result?: R;
}

export interface Headline { metric: string; model: number; baseline: number }

export interface ModuleOverview {
  module: Module;
  output_label: string;
  dataset_ready: boolean;
  dataset_stats: Record<string, number>;
  setup: string[];
  model_version: string | null;
  model_name: string | null;
  headline: Headline | null;
  runs: { total: number; succeeded: number; failed: number };
  last_run: Run | null;
}

export interface Overview {
  modules: ModuleOverview[];
  recent_runs: Run[];
  activity: ({ date: string; total: number } & Partial<Record<Module, number>>)[];
  run_status: Partial<Record<RunStatus, number>>;
  kepler_snippet: null | { kepid: number; tce_plnt_num: number; label: string; period_days: number; depth_ppm: number; local: (number | null)[] };
  mars_terrain: null | { images: number; shares: Record<string, number> };
  jobs: Job[];
  space_weather?: { event: string; start: string; dst: (number | null)[]; min_dst: number | null; max_kp: number | null; years: number; annotations: number } | null;
  workbench?: { datasets: number; by_kind: Partial<Record<"timeseries" | "images" | "lightcurve", number>>; runs: number;
    reviews: Partial<Record<"confirmed" | "dismissed" | "unsure" | "corrected", number>>; recent: { id: string; name: string; kind: string }[] };
}

export interface Dataset {
  module: Module;
  name: string;
  description: string;
  source_url: string;
  setup: string[];
  ready: boolean;
  stats: Record<string, number>;
  uploaded_datasets?: number;
  catalog?: { present: boolean; extended: boolean; has_sky: boolean };
}

export interface MetricCell { name: string; model: number | string; baseline: number | string; lower_is_better?: boolean }
export interface MetricRow { group: string; metrics: MetricCell[]; confusion?: number[][] }
export interface EvaluationPayload {
  module: Module;
  version: string;
  created_at: string;
  model_name: string;
  baseline_name: string;
  eval_split: string;
  headline: Headline;
  rows: MetricRow[];
  struggles: string[];
  curves?: { pr: { model: number[][]; baseline: number[][] }; roc: { model: number[][]; baseline: number[][] } };
  per_channel?: { chan_id: string; spacecraft: string; model: { tp: number; fp: number; fn: number }; baseline: { tp: number; fp: number; fn: number } }[];
}
export interface ModuleEvaluation {
  module: Module;
  output_label: string;
  active_version: string | null;
  active: EvaluationPayload | null;
  history: { version: string; created_at: string; headline: Headline | null }[];
}

// ---------------- telemetry
export interface Channel {
  chan_id: string;
  spacecraft: "SMAP" | "MSL";
  anomaly_sequences: number[][];
  anomaly_classes: string[];
  num_values: number;
  has_model: boolean;
  downloaded: boolean;
}
export interface TelemetryResult {
  chan_id: string;
  spacecraft: string;
  n_steps: number;
  window: number;
  actual: (number | null)[];
  predicted: (number | null)[];
  smoothed_error: (number | null)[];
  threshold: number;
  threshold_z: number;
  quiet_channel: boolean;
  pruned_sequences: number;
  flagged: { start: number; end: number; peak_error: number }[];
  baseline: { name: string; flagged: { start: number; end: number }[]; upper: number; lower: number };
  labeled: { start: number; end: number; class: string; found_by_model: boolean; found_by_baseline: boolean }[];
  score: { tp: number; fp: number; fn: number };
  baseline_score: { tp: number; fp: number; fn: number };
  n_commands: number;
  model_version: string;
}

// ---------------- exoplanet
export interface Tce {
  kepid: number;
  tce_plnt_num: number;
  period_days: number;
  duration_hours: number;
  depth_ppm: number;
  label: string;
  cached: boolean;
  name: string | null;
}
export interface XY { x: (number | null)[]; y: (number | null)[] }
export interface StarInfo { teff: number; radius_rsun: number; mass_msun: number; luminosity_lsun: number; hz_inner_au: number; hz_outer_au: number; assumed: Record<string, string> }
export interface PlanetInfo {
  id: string; label: string | null; selected: boolean; period_days: number; a_au: number; a_rstar: number; ror: number;
  rp_earth: number; impact: number; inclination_deg: number; teq_k: number; insolation_earth: number;
  habitable_zone: "inside" | "within" | "outside"; size_class: string; derived: Record<string, string>;
}
export interface SystemInfo { star: StarInfo; planets: PlanetInfo[] }
export interface BlsPeak { period_days: number; t0: number; duration_hours: number; depth_ppm: number; snr: number; power: number }
export interface ExoplanetResult {
  source: "kepler" | "upload";
  kepid?: number;
  tce_plnt_num?: number;
  dataset_id?: string;
  dataset_name?: string;
  target_id?: string;
  target_name: string;
  mission?: string | null;
  name: string | null;
  candidate_score: number | null;
  baseline_score: number | null;
  model_available: boolean;
  catalog_label: string | null;
  split: "train" | "val" | "test" | "unseen" | "no model";
  ephemeris_source: string;
  tce: { period_days: number; epoch: number; epoch_system?: string | null; duration_hours: number; depth_ppm: number; model_snr: number | null; planet_radius_earth: number | null };
  measured_depth_ppm: number;
  n_points: number;
  n_transits: number;
  diagnostics: null | { odd_depth_ppm: number | null; even_depth_ppm: number | null; odd_even_sigma: number | null; secondary_depth_ppm: number | null; secondary_sigma: number | null; noise_ppm: number | null };
  global_view: XY;
  local_view: XY;
  scatter_global: XY;
  scatter_local: XY;
  system: SystemInfo;
  sky: { ra: number; dec: number } | null;
  bls?: { best: BlsPeak; alternatives: BlsPeak[]; periodogram: { period: number[]; power: number[] }; span_days: number; cadence_minutes: number; period_range: number[] } | null;
  raw_preview?: XY;
  data_summary?: { n_points: number; span_days: number; cadence_minutes: number; segments: number; time_system: string };
  model_version: string | null;
}
export interface UploadTarget {
  id: string; name: string; n_points: number; span_days: number; cadence_minutes: number; segments: number;
  mission: string | null; time_system: string; columns: Record<string, string | boolean | null>; sources: string[];
  star: Record<string, number>; searchable: boolean; note: string | null; preview?: XY; noise_ppm: number;
}
export interface UploadDataset {
  id: string; filename: string; created_at: string; size_bytes: number; n_files: number; combine: boolean;
  targets: UploadTarget[]; warnings: string[]; skipped: string[]; parse_seconds: number;
}
export interface SkyCatalog { available: boolean; fields?: string[]; stars: [number, number, number, number, number][]; reason?: string }

// ---------------- jobs
export type JobStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled" | "interrupted" | "source_unavailable";
export interface Job {
  id: string; task: string; module: Module; title: string; params: Record<string, unknown>; status: JobStatus;
  progress: number | null; step: string | null; message: string | null; error: string | null; command: string;
  created_at: string; started_at: string | null; finished_at: string | null; log?: string[];
}
export interface PredRow { kepid: number; tce_plnt_num: number; av_training_set: string; planet: number; cnn: number; baseline: number }
export interface Featured {
  kepler90: { kepid: number; tce_plnt_num: number; label: string; period_days: number }[];
  test_planets: PredRow[];
  false_positives_caught: PredRow[];
  false_positives_missed: PredRow[];
}

// ---------------- mars
export interface MarsSample {
  id: string;
  name: string;
  split: string;
  width: number;
  height: number;
  labeled_fraction: number;
  label_shares: Record<string, number>;
}
export interface MarsResult {
  width: number;
  height: number;
  classes: string[];
  class_shares: Record<string, number>;
  masked_fraction: number;
  files: { input: string; mask: string; label?: string };
  location?: { sol: number | null; site: number | null; drive: number | null; precision: string; note?: string; lat?: number; lon?: number; sol_source?: string } | null;
  evaluation: null | {
    labeled_fraction: number;
    model: { miou: number; pixel_acc: number; iou: Record<string, number | null> };
    baseline: { miou: number; pixel_acc: number; iou: Record<string, number | null> };
  };
  model_version: string;
}

// ---------------- workbench
export interface ParamSpec {
  name: string; label: string; type: "table" | "column" | "columns" | "number" | "integer" | "select" | "text" | "image" | "boolean" | "units";
  default: unknown; required?: boolean; optional?: boolean; min?: number; max?: number; step?: number; help?: string;
  filter?: string[]; options?: { value: string; label: string }[]; show_if?: Record<string, unknown>;
}
export interface AnalysisSpec {
  id: string; title: string; description: string; input: "timeseries" | "images" | "lightcurve"; output: string;
  requires_model: string | null; evaluation: string; params: ParamSpec[]; min_images?: number;
}
export interface ColumnProfile { name: string; type: "numeric" | "binary" | "datetime" | "text"; missing: number; min?: number | string | null; max?: number | string | null; mean?: number | null; std?: number | null; examples?: string[] }
export interface WbTable { id: string; file: string; rows: number; columns: ColumnProfile[]; head: (string | number | null)[][]; suggested: { time_col: string | null; label_col: string | null; sensor_cols: string[] } }
export interface WbImage { id: string; file: string; width: number; height: number; bands: number; mask?: string; geo: { georeferenced: boolean; crs?: string; res?: [number, number]; projected?: boolean } }
export interface WbDataset {
  id: string; kind: "timeseries" | "images" | "lightcurve" | "samples"; name: string; created_at: string; sha256: string | null; summary: string;
  samples?: WbSamples;
  tables?: WbTable[]; images?: WbImage[]; compatible?: string[]; source_files?: string[]; n_source_files?: number;
  profile?: { category: string; label: string; confidence: number; reasons: string[]; facts: Record<string, unknown>;
              recommended: string | null; why?: string; mars_model?: boolean };
}
export interface WbSamples {
  n: number; params: string[]; n_steps: number; classes: string[]; class_counts: Record<string, number>; files: string[];
  missing_fraction: number; skipped_records: number; flare: boolean; context: string | null;
  separation: { param: string; d: number; pos_median: number; neg_median: number }[];
}
export interface ClassProfile { param: string; info: string | null; steps: number; classes: { class: string; n: number; median: (number | null)[]; q25: (number | null)[]; q75: (number | null)[] }[] }
export interface SkillScores { name: string; tp: number; fp: number; fn: number; tn: number; tss: number; hss: number; recall: number; false_alarm_rate: number; precision: number; f1: number; auc?: number | null }
export interface ClassificationResult {
  viewer: "classification"; dataset_id: string; target: string; split: string; n: number; n_tested: number; n_positive: number;
  threshold: number; threshold_mode: string; class_counts: Record<string, number>; flare: boolean;
  model: SkillScores; baseline: SkillScores; climatology: SkillScores;
  importance: { feature: string; importance: number }[]; profiles: ClassProfile[];
  samples: { id: string; label: string; positive: boolean; prob: number; predicted: boolean; file: string; outcome: "hit" | "missed" | "false alarm" | "correct quiet" }[];
  insights: { kind: "info" | "warn"; title: string; text: string }[];
}
export interface WbDatasetListItem { id: string; kind: WbDataset["kind"]; name: string; created_at: string; sha256: string | null; summary: string }
export interface WbInterval { start: number; end: number; x0: string | number; x1: string | number; length?: number; channels?: string[] }
export interface PRF { precision: number; recall: number; f1: number; tp: number; fp: number; fn: number }
export interface TimelineResult {
  viewer: "timeline"; analysis: string; dataset_id: string; file: string; n_rows: number; x_kind: "step" | "numeric" | "datetime"; x_label: string;
  x: (string | number)[]; train_end: string | number; settings: Record<string, unknown>;
  channels: { name: string; unit?: string; values: (number | null)[]; upper: number; lower: number; train_mean: number; train_std: number; missing_fraction: number; flags_zscore: number; flags_robust: number }[];
  zscore: { name: string; intervals: WbInterval[] }; robust: { name: string; intervals: WbInterval[] };
  labeled: WbInterval[]; metrics: null | { zscore: PRF; robust: PRF; n_labeled: number; found_zscore: boolean[]; found_robust: boolean[] };
}
export interface ChangeResult {
  viewer: "change"; analysis: string; method: "difference" | "water"; width: number; height: number; bands: number; threshold: number;
  legend: Record<string, string>; counts: Record<string, number>; valid_fraction: number; changed_fraction: number;
  changed_area_km2: number | null; pixel_area_m2: number | null; area_basis: string; notes: string[]; georeferenced: boolean; crs: string | null;
  footprint_lonlat?: [number, number, number, number] | null;
  regions: { pixels: number; area_km2: number | null; bbox: [number, number, number, number] }[];
  histogram: null | { counts: number[]; edges: number[]; threshold: number };
  before: { id: string; file: string }; after: { id: string; file: string }; settings: Record<string, unknown>;
}
export type WorkbenchResult = TimelineResult | ChangeResult | (MarsResult & { viewer: "segmentation"; analysis: string; image: { id: string; file: string } });
