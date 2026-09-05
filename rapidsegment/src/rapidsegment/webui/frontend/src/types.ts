// Shared response shapes from the FastAPI backend.

export interface DistRow {
  val: string | null;
  cnt: number;
}

export interface TInfo {
  col: string;
  n_distinct: number;
  is_binary: boolean;
  binary_label?: string | null;
  event_rate?: number | null;
  null_count: number;
  null_pct: number;
  dist?: DistRow[];
  n_total: number;
}

export interface M1Status {
  loaded: boolean;
  data_modified: boolean;
  dataset_name: string;
  db_file: string;
  db_file_mod: string;
  active_db: string;
  db_mb: number;
  n_rows: number;
  n_cols: number;
  target_col?: string | null;
  type_overrides: Record<string, string>;
  tinfo?: TInfo | null;
  max_upload_mb: number;
}

export interface SummarizeRow {
  column_name: string;
  column_type: string;
  null_percentage: number;
  approx_unique: number;
  min?: unknown;
  max?: unknown;
  avg?: unknown;
}

export interface Quality {
  n_rows: number;
  n_cols: number;
  n_numeric: number;
  n_categorical: number;
  db_mb: number;
  types: { column_name: string; data_type: string }[];
  profiling: Record<string, unknown>[];
  summarize: SummarizeRow[];
}

export interface M2Options {
  bin_options: string[];
  bin_map: Record<string, string>;
  bin_rmap: Record<string, string>;
  metric_options: string[];
  metric_map: Record<string, string>;
  metric_rmap: Record<string, string>;
  sort_priority_options: { value: string; label: string }[];
  sort_priority_map: Record<string, string>;
  sort_priority_rmap: Record<string, string>;
  n_jobs_options: string[];
  n_jobs_map: Record<string, number>;
  n_jobs_rmap: Record<string, string>;
  expand_log_options: string[];
  grid_size_range: [number, number];
  grid_lift_range: [number, number];
}

export interface Issue {
  field: string;
  message: string;
}

export interface BuildResult {
  cfg: Record<string, any>;
  issues: Issue[];
  estimate_seconds: number;
  columns: string[];
}

export interface LeaderboardRow {
  exp_id: string;
  name: string;
  created_at: string;
  data_rows: number;
  data_cols: number;
  status: string;
  execution_time_sec: number;
  target_col: string;
  primary_key: string;
  segments_count: number;
  avg_lift: number;
  max_lift: number;
  coverage_pct: number;
  cumulative_event_capture: number;
  baseline_rate: number;
  error_msg?: string | null;
  dataset_name: string;
  config: Record<string, any>;
}

export interface ExperimentRef {
  exp_id: string;
  name: string;
  created_at: string;
  status: string;
  target_col?: string;
  data_rows?: number;
}

export interface LogRecord {
  ts: string;
  level: string;
  msg: string;
  src?: string;
}

export interface Segment {
  segment_id?: number;
  rule_string?: string;
  sql_filter?: string;
  count?: number;
  rate?: number;
  lift?: number;
  support?: number;
  [k: string]: unknown;
}

export interface CoverageRow {
  segment?: number | string;
  capture_rate?: number;
  [k: string]: unknown;
}

export interface RunSnapshot {
  exp_id: string;
  status: string;
  finalized: boolean;
  step: number;
  step_names: string[];
  elapsed: number;
  n_rows: number;
  n_cols: number;
  target_col: string;
  experiment_name: string;
  segments_found: number;
  coverage_pct: number | null;
  avg_lift: number | null;
  best_lift: number | null;
  best_rule: string | null;
  current_feature: string | null;
  stop_reason?: string | null;
  error_msg?: string | null;
  cancel_requested?: boolean;
  save_error?: string | null;
  top_candidates: { rule_string: string; count: number; lift: number }[];
  segments_preview?: { id: number; rule_string: string; sql_filter: string; count: number; rate: number; lift: number }[];
  logs: LogRecord[];
  log_count: number;
}

export interface ExperimentFull {
  exp_id: string;
  name: string;
  created_at: string;
  status: string;
  execution_time_sec: number;
  target_col?: string;
  data_rows?: number;
  data_cols?: number;
  dataset_name?: string;
  config?: Record<string, any>;
  result?: {
    segments?: Segment[];
    coverage?: CoverageRow[];
    segments_count?: number;
    avg_lift?: number;
    max_lift?: number;
    coverage_pct?: number;
    baseline_rate_pct?: number;
    cumulative_event_capture?: number;
    error_msg?: string | null;
    stop_reason?: string | null;
    [k: string]: unknown;
  };
  logs?: LogRecord[];
  [k: string]: unknown;
}

export interface Figure {
  data: any[];
  layout: Record<string, any>;
}
