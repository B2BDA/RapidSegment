# RapidSegment — Code Graph

Versioned map of the `RapidSegment` codebase. Updated for repo HEAD `e45bce0` (main, `rapidsegment` v1.2.2.post1).

---

## 1. Repository Layout

```
/workspace
├── README.md                    # Main marketing/docs README (authoritative)
├── README.md.old                # Legacy README (keep in sync? likely stale)
├── Banner.png
├── RapidSegment_Business_Deck.pdf
├── Examples/
│   ├── Example.ipynb            # Bank marketing demo (bank-full.csv)
│   ├── Example2.ipynb           # Churn demo (train.csv / test.csv)
│   ├── compare_segmenters.ipynb # Interactive RapidSegment vs DT comparison w/ charts
│   ├── decision_tree_segmentation.py  # Bare iterative decision-tree segmenter (sklearn)
│   ├── compare_segmenters.py          # RapidSegment vs decision-tree comparison harness
│   ├── tune_rapidsegment_capture.py   # Capture-optimization sweep for RapidSegment
│   ├── bank-full.csv            # UCI bank marketing dataset (target col = "Target", yes/no)
│   ├── train.csv                # Churn training set
│   └── test.csv                 # Churn test set
└── rapidsegment/                # Installable Python package (src layout)
    ├── pyproject.toml           # uv_build backend, v1.2.2.post1
    ├── uv.lock
    ├── LICENSE
    ├── .python-version
    ├── README.md
    └── src/rapidsegment/
        ├── __init__.py          # Public API exports
        ├── py.typed
        ├── builder.py           # StrategicSegmentBuilder (1968 lines)
        ├── scorer.py            # StrategicSegmentScore (246 lines)
        └── utils/
            ├── __init__.py      # exports UniversalDataLoader
            ├── data_loader.py   # UniversalDataLoader (264 lines)
            ├── on_gcp_feature_selection.py  # BigQueryFeatureSelector (315 lines)
            └── undersampler.sql # BigQuery downsampling snippet
```

---

## 2. Public API Surface

Defined in `rapidsegment/src/rapidsegment/__init__.py`:

```python
__all__ = ["UniversalDataLoader", "StrategicSegmentBuilder", "StrategicSegmentScore"]
```

`__version__` is read from package metadata (`importlib.metadata.version("rapidsegment")`), falling back to `0.0.0+dev` when running from source without install. Author: Bishwarup Biswas.

---

## 3. Modules, Classes, and Methods

### 3.1 `utils/data_loader.py` — `UniversalDataLoader`

Data ingestion layer. Normalizes inputs to PyArrow Tables with all numeric columns cast to `float64`.

| Method | Line | Purpose |
|---|---|---|
| `__init__(project_id, dataset_id, table_id, file_path)` | 38 | Source config; only one source type is active |
| `load(fallback_data=None)` | 50 | Dispatch: (1) in-memory `fallback_data` → cast numerics; (2) BigQuery ids set → `_load_from_bigquery`; (3) `file_path` → `_load_from_file`; else `ValueError` |
| `_cast_table_numerics_to_float(table)` (static) | 88 | Casts integer/float/decimal columns to `float64` (`pc.cast(..., safe=False)`), keeps others as-is |
| `_load_from_file()` | 134 | Ext by extension: `.parquet` (pa_pq), `.csv` (pa_csv), `.arrow`/`.feather` (ipc), `.xlsx`/`.xls` (openpyxl), else `ValueError` |
| `_load_excel_to_arrow()` | 163 | openpyxl read-only; positional column tracking, missing headers named `_col_{i}` |
| `_load_from_bigquery()` | 204 | With `google.cloud.bigquery`: `get_table` schema, `SAFE_CAST` numeric types → `FLOAT64`, `query().to_arrow()`. Without lib: returns DuckDB scan macro string `bigquery_scan('project','dataset','table')` |

### 3.2 `utils/on_gcp_feature_selection.py` — `BigQueryFeatureSelector`

Screens features natively in BigQuery using naive IV + stddev thresholds; results land in a local DuckDB relation. **Warning: can incur significant BigQuery cost.**

| Method | Line | Purpose |
|---|---|---|
| `__init__(project_id, dataset_id, table_id, target_column, iv_threshold=0.02, stddev_threshold=1e-5, min_bin_n_event=1, min_bin_n_nonevent=1, bins=10, batch_size=15, binary_columns=None, bq_client=None)` | 50 | Stores config; creates BQ client |
| `_detect_binary_numerical_columns(numerical_columns)` | 80 | One query, `COUNT(DISTINCT col)` per numeric col; keeps distinct ≤ 2 as binary flags |
| `_get_table_schema()` | 118 | Classifies columns; binary overrides or auto-detect; binary moved to categorical bucket |
| `_build_batch_query(numerical_chunk, categorical_chunk)` | 161 | SQL: numeric → NTILE(`bins`) bins; categorical → `CAST(col AS STRING)` bins; IV template with min-bin protection (`goods_in_bin`/`bads_in_bin` thresholds), `+0.0001` smoothing; numeric stddev computed, categorical stddev hardcoded `9999.0` |
| `screen_features()` | 254 | Batches by `batch_size`, runs queries, inserts Arrow into DuckDB, filters `stddev > threshold AND naive_iv >= threshold`, sorts by IV desc; returns DuckDB relation `(feature_name, feature_stddev, naive_iv)` |

### 3.3 `utils/undersampler.sql`

BigQuery downsampling snippet: keep 100% of `target_y = 1`, deterministic 10% of `target_y = 0` via `ABS(MOD(FARM_FINGERPRINT(row_id), 100)) < 10`.

### 3.4 `builder.py` — `StrategicSegmentBuilder` (core engine)

#### Module-level helpers

| Symbol | Line | Purpose |
|---|---|---|
| `_BRACKET_REGEX = re.compile(r"\[(.*?)\]", re.DOTALL)` | 39 | Fast bracket parsing for rule strings |
| `setup_disk_backed_db(base_dir="experiments")` | 42 | Creates `segmentation_{YYYYMMDD}_{uuid8}.duckdb` + `tmp_{...}` temp dir, returns `(db_path, temp_dir)` |

#### Constructor params (`__init__`, line 80)

| Param | Default | Meaning |
|---|---|---|
| `target` | (required) | Binary target column |
| `n_jobs` | `-1` | Parallel workers; `-1` → `max(1, cpu_count - 1)` |
| `min_sample_size` | 1000 | Absolute min rows (hard constraint, snapshotted per run) |
| `min_lift` | 1.5 | Absolute min lift (hard constraint) |
| `min_events` | 100 | Min positive events |
| `top_n_vars` | 15 | Features passed into Apriori engine |
| `max_segments` | 10 | Max segments extracted |
| `max_feature_reuse` | 1 | Max times a feature can appear across segments |
| `param_grid` | `{}` | `{min_sample_size:[...], min_lift:[...]}` grid to sweep |
| `enable_diversity` | False | Blocks rules mixing features from the same `feature_groups` |
| `enable_1way/2way/3way` | True | Toggles rule arity |
| `feature_groups` | `{}` | Business category → column list |
| `ignore_features` | `[]` | Columns dropped before IV |
| `sort_priority` | `"rate_lift_count"` | Champion ranking key (14 variants) |
| `binning_method` | `"optimal_cart"` | `"naive"` \| `"optimal_cart"` \| `"optimal_quantile"` \| `"optimal"` (alias of cart); validated in `__init__` |
| `naive_bins` | 5 | Quantile bins for naive path |
| `max_expansion_hops` | 0 | Adjacent-bin merge distance (0 = off) |
| `selection_metric` | `"iv"` | Feature ranking metric: `"iv"` or `"response_rate"` |
| `expand_log_mode` | `"none"` | `"none"` \| `"summary"` \| `"champion"` \| `"full"` |
| `db_path` / `db_temp_dir` | None | Optional explicit DuckDB file + temp dir; auto-created if missing |

#### Key attributes (state)

- `self.segments: List[Dict]` — extracted segments
- `self.diagnostics_: List[Dict]` — per-iteration audit records
- `self.stop_reason: Optional[str]`
- `self.feature_usage_counts: Dict[str, int]`
- `self._feature_to_group: Dict[str, str]` — reverse group map
- `self._columns_types`, `self._categorical_cols` — schema memo
- `self.min_sample_size/min_lift/min_events` are MUTATED inside `extract_segments` during grid search and restored after (see below)

#### Methods

| Method | Line | Purpose |
|---|---|---|
| `_resolve_optb_dtype(duckdb_type)` (static) | 186 | VARCHAR/CHAR/STRING/TEXT/UUID → `"categorical"`, else `"numerical"` |
| `_validate_feature_groups(columns)` | 196 | Raises `ValueError` if a declared group feature is missing; rebuilds `_feature_to_group` |
| `get_group(var)` | 222 | Group of feature, or feature name itself |
| `is_diverse(combo)` | 228 | True if all groups distinct (no-op when `enable_diversity=False`) |
| `_get_sort_key(rule)` | 237 | Returns tuple per `sort_priority` + rule-string tie-breaker (14 orderings: permutations of lift/count/rate/events triples) |
| `compute_iv_ranking_and_bin(con, eligible_cols, columns_types)` | 284 | Parallel IV + bins via joblib threads. **Naive path**: QUANTILE_CONT edges, forced `-inf`/`inf`, half-open bins, `Missing` bin, IV in SQL. **Optimal path**: masked-array handling, lexsort for deterministic CART fit, `OptimalBinning(name, dtype, prebinning_method)` fit, IV from bin table, `metric="indices"` for categoricals + `_sanitize_bin_label`. Returns `(ranking, precomputed_bins)` sorted stable by `-metric` then variable name |
| `_bin_sort_key(label)` (static) | 551 | Numeric-aware sort key for bin labels (`[lo, hi)` → float) |
| `_expand_adjacent_bins(con, combo, base_rate, base_results, seen_rules)` | 568 | Merges adjacent bins per variable (Python groupby, no pandas). Uses cumulative sums for sliding windows; merged label `[{sorted labels}]`; skips `Missing`; dedupes via `seen_rules` |
| `_candidate_windows(idx, n, max_hops)` (static) | 715 | Yields `(lo, hi)` windows around `idx`; excludes windows spanning the whole domain (degenerate rule guard) |
| `_sanitize_bin_label(label)` (static) | 740 | Rebuilds categorical labels from array-like (OptBinning + pandas≥3 fix) |
| `_agg_combinations(con, combo_list, base_rate)` | 754 | Builds one SQL GROUP BY query per combo, chunks by 100 (`UNION ALL`), falls back to individual execution on batch failure. Returns rule dicts `{rule, count, rate, lift, events, combo_vars}`; rate/lift expressed as percentages (`events/count*100`). For `binning_method == "naive"` triggers `_expand_adjacent_bins` per combo and emits expansion logs per `expand_log_mode` |
| `parse_rule_to_sql(rule_str)` | 897 | Rule grammar → SQL WHERE predicate (detailed in §5) |
| `extract_segments(data)` | 1112 | Main pipeline (detailed in §4) |
| `evaluate_final_coverage(original_data)` | 1649 | Hierarchical CASE evaluation over original data; returns per-segment KPIs + cumulative capture (SQL window functions) |
| `explain_feature_journey(feature_name)` | 1717 | Prints per-iteration audit trail of a feature |
| `explain_no_segments()` | 1756 | Human-readable report why extraction stopped (constraints, funnel, near-miss gaps) |
| `generate_feature_health_report(original_data, features)` | 1822 | DuckDB-native bin health report: NTILE bins for numerics, categorical/missing bins; returns pandas DataFrame |

#### Segment output schema

Each segment dict in `self.segments`:
`segment_id` (1-based iteration), `rule_string`, `sql_filter`, `count`, `rate` (%), `lift`, `meta_applied_sample_size`, `meta_applied_min_lift` (grid params that admitted the winner).

### 3.5 `scorer.py` — `StrategicSegmentScore`

| Method | Line | Purpose |
|---|---|---|
| `__init__(target_col, primary_key, segment_cols)` | 45 | Stores config, initializes `model_artifact` |
| `calculate_and_export_weights(data, export_path)` | 56 | Full scorecard pipeline (detailed below) |

#### Scorecard pipeline (inside `calculate_and_export_weights`)

1. **DB setup**: file-backed DuckDB `score_experiment_{timestamp}.db` (removed first if exists); `CREATE OR REPLACE TABLE df AS SELECT * FROM data`.
2. **Master aggregation**: single query — `COUNT(*) total_pop`, `SUM(target) total_ev`, per-segment `COUNT(flag=1)` and `SUM(flag=1 ? target : 0)`. Validates pop/events > 0.
3. **Weights**: per segment `response_rate = ev/cnt`, `capture_rate = ev/total_ev`, `lift = rr/baseline`, `raw_weight = rr * 100`, exported weight = `int(round(raw_weight))`. Zero-volume/zero-event segments get weight 0.
4. **Decile-resolution warning**: counts distinct non-zero weights; `< 10` logs a warning about repeated thresholds.
5. **Scoring**: `scored_population` table = `SUM(flag * weight)` linear expression.
6. **Deciles**: `QUANTILE_DISC(total_score, [0.1..1.0])` over `total_score > 0` (active population; zero-score baseline excluded), reversed to descending, `decile_min_thresholds = {"1": q1, ...}`.
7. **Artifact**: JSON with `model_metadata` (population counts, active pct, baseline event rate), `segment_weights`, `decile_min_thresholds`; written to `export_path`.

---

## 4. `extract_segments` Algorithm Walkthrough

State reset → snapshot absolute thresholds (`abs_min_sample_size/events/lift`) → DB setup (auto-create if none) with `SET threads`, `SET memory_limit`, `PRAGMA temp_directory` → register `current_df` (adds `__rs_row_id` via `ROW_NUMBER()`, casts target to DOUBLE) → compute `columns_types`, `_categorical_cols`, `eligible_cols` (excludes target, row id, ignored) → build `experiments` grid from `param_grid` (cartesian product of sizes × lifts) → lock `original_base_rate`.

**Per iteration** `i = 1..max_segments`:
1. Read `current_base_rate`, `current_volume` from `current_df`. Stop if base rate 0 or volume < min floor sample size (record `stop_reason`).
2. `compute_iv_ranking_and_bin` → `iv_ranking`, `precomputed_bins`.
3. Build `iteration_snapshot` diagnostics for every eligible feature (excluded by reuse / zero score / outside top-N / eligible).
4. `allowed_vars` = ranking filtered by `feature_usage_counts < max_feature_reuse`; `top_vars` = first `top_n_vars`; stop if empty.
5. Build `binned_data` dict from precomputed bins (only vars with >1 unique bin) → register `binned_df` table.
6. Apply grid-min thresholds to `self.min_sample_size/min_lift` (mutation; restored after loop).
7. **Candidate generation (Apriori)**:
   - 1-way: aggregate all single vars → `valid_1way_vars` (surviving features).
   - 2-way: pairs of surviving vars (diverse only) → `valid_2way_sets`.
   - 3-way: triplets where every pair is in `valid_2way_sets` AND diverse.
   - Arity toggles gate which levels enter `all_candidate_rules`.
8. **Grid shortlist**: for each grid config, keep best rule by `_get_sort_key` meeting that config's thresholds; tag with `grid_min_sample_size/lift`.
9. **Raw validation**: candidates sorted; for each, `parse_rule_to_sql` → validate against RAW `current_df` (`COUNT`/`SUM`); first passing `abs_min_sample_size/events/lift` becomes `selected_candidate`; else record `closest_miss` (near-miss gaps).
10. Store segment with actual counts/rates/lift; bump `feature_usage_counts` for winner vars.
11. **Residual update (NULL-safe)**: `current_df ← WHERE NOT (sql) OR (sql) IS NULL`. This mirrors the hierarchical CASE in `evaluate_final_coverage`.
12. Champion/expansion logging per `expand_log_mode`.

Cleanup in `finally`: close con; if DB auto-created, remove db file + temp dir. Restore original thresholds. If loop finished with full `max_segments`, set stop_reason. Returns `self.segments`.

---

## 5. Rule String Grammar → SQL (`parse_rule_to_sql`)

Rules are `&`-joined `col=value` parts. Value forms handled in order:

1. **Multi-range numeric** `[[10, 20), [20, 30)]` (requires non-categorical + `),` `[` separator): merged into a single bounding range (`>= lower` / `< upper`, with `[`/`]` → `>=`/`<=`).
2. **Merged categorical lists** `[[male], [female]]`: `IN ('male', 'female')`; guard: `[[VIP]]` (single nested pair) → equality on `VIP`.
3. **Explicit categoricals** (contains quotes/"Array"/"Categorical", >2 tokens, or non-numeric tokens): `IN (...)`; items parsed via `ast.literal_eval` with fallback splitting.
4. **Special/Missing**: `col IS NULL`.
5. **Standard intervals** `[lo, hi)` → range predicates (numeric); categorical → `IN`.
6. **Single value** `[123]` or `[Value]` → equality (numeric bare, categorical quoted).

Helpers: `_quote_sql_ident` (double-quote escaping `"` → `""`), `_quote_sql_string` (single-quote escaping), `_strip_wrapping_quotes` (only matched outer quote pair — guards `Say "hi"`).

---

## 6. Key Design Properties / Gotchas

- **Lift definition**: `rate% / (base_rate * 100)`, i.e. `(evt/cnt) / base_rate`. In `_agg_combinations` rate is `events/count*100`, lift divides by `base_rate*100`.
- **Original base rate locked** once at start; all candidate lifts computed against `original_base_rate`, not the shrinking residual base.
- **Threshold mutation**: `self.min_sample_size/min_lift` temporarily set to grid-minimum during candidate generation, then restored from snapshots.
- **Determinism**: stable sort by `(metric desc, variable)`; lexsort before OptBinning fit; `_get_sort_key` rule-string tie-breaker (guards against Python hash randomization + unordered GROUP BY).
- **Apriori trade-off**: strong 3-way rules can be missed if an underlying pair fails `min_sample_size` — documented in README.
- **Expansion only on naive binning**; window spanning full domain is excluded (degenerate-rule guard).
- **NULL handling**: NULLs never match rules, remain in residual, land in `ELSE 0` bucket in final coverage.
- **Target-leak feature**: OptBinning drops it during segment creation; BigQueryFeatureSelector marks IV 0.
- **Zero-score population** excluded from decile calibration (only active scored population).
- **Storage**: extraction defaults to disk-backed DuckDB under `experiments/` with auto-cleanup; scorer uses file-backed DB `score_experiment_{ts}.db`.

---

## 7. Data Flow Diagram

```
UniversalDataLoader ──(PyArrow Table, numerics→float64)──> input data
                                                              │
StrategicSegmentBuilder.extract_segments(data)
   current_df = data + __rs_row_id, target→DOUBLE
        │
   loop i=1..max_segments on shrinking current_df:
        ├─ compute_iv_ranking_and_bin ──> iv_ranking + precomputed_bins
        ├─ build binned_df (top_n_vars, reuse-filtered)
        ├─ _agg_combinations: 1-way → 2-way → 3-way (Apriori + diversity)
        ├─ [naive only] _expand_adjacent_bins (max_expansion_hops)
        ├─ grid shortlist (_get_sort_key) 
        ├─ parse_rule_to_sql + raw validation on current_df
        └─ residual update: WHERE NOT(sql) OR (sql) IS NULL
        │
   => self.segments [ {segment_id, rule_string, sql_filter, count, rate, lift, meta_*} ]
        │
evaluate_final_coverage(original_data)  → hierarchical CASE KPIs + cumulative capture
explain_feature_journey / explain_no_segments / generate_feature_health_report

Segments → binary flags (seg_N) → StrategicSegmentScore.calculate_and_export_weights
        → JSON { model_metadata, segment_weights, decile_min_thresholds }

BigQueryFeatureSelector.screen_features() → DuckDB relation (feature, stddev, naive_iv)
        → feeds UniversalDataLoader / feature filtering
```

---

## 8. Dependencies & Constraints

- Python ≥ 3.9; build backend `uv_build`; version `1.2.2.post1`.
- Runtime: `duckdb>=1.5.4`, `joblib>=1.5.3`, `numpy>=2.5.1`, `optbinning>=0.21.0`, `pandas>=2.2.0`, `psutil>=7.2.2`, `pyarrow>=25.0.0`.
- Optional extras: `excel` (openpyxl), `gcp` (google-cloud-bigquery), `prettytable`.
- Packaging: src layout; package name `rapidsegment`; module `rapidsegment` inside `rapidsegment/src/rapidsegment`.

---

## 9. Known Code Notes / Comments to Honor

- `builder.py:432` — sorting only the *fit* input, transform on original order, to keep bin labels aligned to `__rs_row_id` (fixes spurious 2/3-way candidates).
- `builder.py:477` — pandas 3 categorical label stringification workaround via `metric="indices"`.
- `builder.py:998` — single-category-with-brackets guard (`[[VIP]]`) vs multi-merge.
- `builder.py:920` — `_strip_wrapping_quotes` never strips interior quotes (`Say "hi"`).
- `builder.py:715` — degenerate whole-domain window exclusion.
- `scorer.py:156` — decile resolution warning when distinct non-zero weights < 10.

---

## 10. Examples (notebooks + scripts)

- **Example.ipynb** (bank-full.csv, target `Target` 0/1 recoded from `yes/no`): `UniversalDataLoader` → param_grid `{min_sample_size:[20000,15000,10000,5000,500], min_lift:[3.0,2.0,1.5]}` → builder `(top_n_vars=10, max_segments=10, max_feature_reuse=5, 1/2/3way)` → `extract_segments` → `evaluate_final_coverage` → PrettyTable → manual DuckDB segment flags → `StrategicSegmentScore` → decile band scoring.
- **Example2.ipynb** (train.csv, target `Exited`): same flow; demonstrates `explain_no_segments()`.
- **Notebooks reference hardcoded path `/workspaces/RapidSegment/Examples/...`** (may need updating for local runs).
- **bank-full.csv format caveat**: header row is comma-separated (17 cols, target named `Target`), data rows are comma-separated too (standard CSV). `UniversalDataLoader` parses it correctly.

### `decision_tree_segmentation.py` (bare iterative decision tree)

Greedy, self-contained baseline segmenter using `sklearn.tree.DecisionTreeClassifier`:
1. Label-encodes string columns (pandas 3 `str` dtype detection via `is_string_dtype`), recodes `Target` yes/no → 0/1.
2. Fits a tree (depth 6, `min_samples_leaf` 1000, `class_weight="balanced"`) on full data; enumerates all root-to-leaf paths via `_extract_leaf_paths`.
3. `_simplify_conditions` collapses repeated same-feature splits into tightest ranges / categorical intersections.
4. Picks the path with **maximum event capture** among valid candidates (`count ≥ 1000`, `events ≥ 100`, `lift ≥ 1.3` vs original base).
5. Removes matched rows, refits on residual, repeats until no valid path (`MAX_SEGMENTS=12` cap).
6. `print_report` prints hierarchical table with cumulative events / event capture % / pop capture %.

Key functions: `_simplify_conditions`, `_decode_conditions` (encoded-int masks), `_format_rule` (decode ints→categories), `_extract_leaf_paths` (iterative DFS), `_best_path`, `run_decision_tree_segmentation(data)`, `print_report(segments)`.

### `compare_segmenters.py` (comparison harness)

Runs both segmenters on `bank-full.csv`:
- **RapidSegment (capture-tuned)**: `StrategicSegmentBuilder` `(sort_priority="lift_events_rate", max_feature_reuse=5, min_lift=1.0, top_n_vars=15, max_segments=12)` → `extract_segments` + `evaluate_final_coverage`.
- **Decision tree (original bare config)**: `run_decision_tree_segmentation` (min_lift=1.3).
- Prints per-method hierarchical tables, a summary (segments, events captured, event capture %, pop capture %, weighted avg lift, best lift, avg rule conditions, runtime), capture efficiency (capture%/pop%), and the capture winner.

Result on bank data (tuned RS vs original DT): RapidSegment **86.99% event capture @ 32.06% pop / wlift 2.71** (11 segs, ~59s) vs DecisionTree **83.85% @ 28.04% pop / wlift 2.99** (9 segs, ~2s). RapidSegment wins event capture.

### `compare_segmenters.ipynb` (interactive comparison notebook)

Jupyter version of the harness with richer presentation. Sections: setup/tuning knobs → data load → RapidSegment extraction + per-segment PrettyTable → decision-tree extraction + per-segment PrettyTable → side-by-side summary → 2x2 matplotlib figure (cumulative **capture curve**, per-segment **lift**, per-segment **events captured**, per-segment **response rate**) → top-rule highlights → conclusions with honest caveats. Regenerate with `/tmp/opencode/build_notebook.py`. Execution verified via nbclient (all cells pass, chart embedded).

### `tune_rapidsegment_capture.py` (tuning sweep)

Sweeps builder configs to maximize event capture. Key findings on bank data:
- `sort_priority="events_lift_rate"` is too greedy: picks a giant low-lift segment first (24k rows, lift 1.39) → 92.7% capture but at 61% pop / wlift 1.52. Avoid for this data.
- `sort_priority="lift_events_rate"` + `max_feature_reuse=5` + `min_lift=1.0` is the winner: 86.99% capture @ 32.1% pop, wlift 2.71, 11 segments (beats original DT's 83.85%).
- `min_lift=1.05` variant: 84.67% @ 29.8% pop, wlift 2.84 (closest-pop win).
- `binning_method="naive"` underperforms (68.5% capture). `selection_metric` iv vs response_rate give identical results here.
- Note: when the DT is also tuned down to `min_lift=1.0`, it reaches 89.5% capture @ 36.6% pop / wlift 2.45 (DT is capture-max by construction, but at lower lift and higher pop).
