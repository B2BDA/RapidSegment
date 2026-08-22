# RapidSegment Streamlit UI — Project Memory

Handoff document. Read this first when resuming work so we can jump straight back in.

## What this project is
A Streamlit no-code UI on top of the **RapidSegment** PyPI library (`rapidsegment==1.2.2.post1`, class `StrategicSegmentBuilder`). Spec lives in `UI_MVP.md` (converted from Solara → Streamlit). 6 modules; all 1–6 done.

## File map
- `C:\Users\Bishwarup\Downloads\UI_MVP.md` — spec + Project Status tracker (module-by-module ✅/⬜)
- `C:\Users\Bishwarup\OneDrive\Documents\RapidSegment\` — **canonical git repo** (branch `Dev`). App = `rapidsegment_ui/app.py` + `pages/1_Data_Loader.py` … `pages/4_Results_Dashboard.py`. Run with `streamlit run app.py` from that dir.
- `C:\Users\Bishwarup\Downloads\Module_1_data_loader.py` / `Module_2_workbench.py` / `Module_3_execution.py` / `Module_4_results.py` — standalone copies of each module
- ⚠ **SYNC RULE**: the standalone file and the page copy of each module must stay byte-identical (keep them in sync via `Copy-Item` after any edit — done with `python -m py_compile` + hash/size check)

## Storage conventions
Suite dir = `<cwd>/.rapidsegment_suite/`:
- `module1_data.duckdb` — table `udl_data` (current dataset)
- `suite_data.db` — `experiments` table (exp_id PK, name, created_at, data_rows, data_cols, status, execution_time_sec, target_col, primary_key, builder_params JSON, segments_count, avg_lift, max_lift, coverage_pct, baseline_rate, error_msg)
- `templates.json`, `oauth_cache.json`, `artifacts/<exp_id>/` (workbench.duckdb, tmp/, logs, sql, config)

## Verified library API (from wheel source, `builder.py` 1968 lines)
Constructor — 23 params, only `target` required:
- `n_jobs=-1` (→ cpu_count-1), `min_sample_size=1000`, `min_lift=1.5`, `min_events=100`, `top_n_vars=15`, `max_segments=10`, `max_feature_reuse=1`, `param_grid=None` (`{"min_sample_size":[...],"min_lift":[...]}` — ONLY these two keys are read; others silently ignored)
- `enable_diversity=False`, `enable_1way/2way/3way=True`, `feature_groups=None` (`{group:[cols]}`; validated ONLY when diversity enabled; unknown cols → ValueError; duplicate col in 2 groups → last wins)
- `ignore_features=None` (target/ignored cols must not be in groups), `sort_priority="rate_lift_count"` (unvalidated; unknown → `(lift, rate, count)`), `binning_method="optimal_cart"` (`naive|optimal_cart|optimal_quantile|optimal` alias; ONLY param that raises ValueError on bad input), `naive_bins=5`, `max_expansion_hops=0` (naive-binning only), `selection_metric="iv"` (`iv|response_rate`; anything else behaves as iv), `expand_log_mode="none"` (`none|summary|champion|full`; invalid → `summary` silently), `db_path`/`db_temp_dir=None` (auto-creates `./experiments/` DuckDB, deleted after run if auto)
- **sort_priority — all 14 valid values** (sort tuple, descending): `rate_lift_count`, `lift_rate_count`, `lift_count_rate`, `count_lift_rate`, `count_rate_lift`, `rate_count_lift`, `events_lift_rate`, `events_rate_lift`, `lift_events_rate`, `rate_events_lift`, `events_count_rate`, `events_rate_count`, `count_events_rate`, `rate_events_count`
- Run: `builder.extract_segments(df)` → `builder.segments` (dicts: segment_id, rule_string, sql_filter, count, rate, lift, meta_applied_sample_size, meta_applied_min_lift); plus `stop_reason` (7 possible strings), `diagnostics_` (per-iteration {iteration, residual_volume, base_rate, features_state, winning_segment} + candidate_funnel + near_miss), `feature_usage_counts`
- Public helpers: `parse_rule_to_sql(rule_str)`, `evaluate_final_coverage(data)` ⚠ DON'T CALL, `explain_feature_journey(feature)`, `explain_no_segments()` (formatted stop-reason report), `generate_feature_health_report(data, features)` → DataFrame, `get_group(var)`, `is_diverse(combo)`

## CRITICAL gotchas
1. **`evaluate_final_coverage()` hangs** (DuckDB file-lock on shared db_path). Never call it. Use `compute_coverage_local(segments, df, target)` — in-memory DuckDB reimplementation (CASE per segment, window fns for base rate/cumulative captures, `ORDER BY CASE WHEN segment = 0 THEN 999999 ELSE segment END`). Present in both M2 and M3.
2. Library logger name: `StrategicEngine` (that's what M3 hooks to capture logs into the console).
3. UI validation is deliberately stricter than the library (min_events ≤ min_sample_size, naive_bins ≥ 3, target not in groups/ignore, ≥1 rule type) — keep these guards.
4. Scorecard decile warning: `StrategicSegmentScore` needs ≥10 distinct weights (S:156-164) → guide users to `max_segments ≥ 10` for Module 4.
5. `MAX_JOBS = max(1, os.cpu_count() or 4)` used for the n_jobs widget; `-1 (all but one core)` label ↔ value -1 mapping via N_JOBS_MAP/RMAP.
6. **`explain_feature_journey(feature)` prints to stdout and RETURNS None** — to show it in Streamlit, wrap the call in `contextlib.redirect_stdout(io.StringIO())` and read the buffer. (`explain_no_segments()` correctly returns a string.)
7. **Plotly `go.Sunburst` with `branchvalues="total"` requires every parent `value` == sum of its children's `values`.** Setting groups/root to 0 while children have counts makes the chart render nothing. Accumulate group + root counts from the segments. Rule complexity = # of feature predicates AND-ed in a rule (1-way/2-way/3-way).
8. **Stale/labeled `binning_method` in stored configs breaks M4 diagnostics.** `StrategicSegmentBuilder.__init__` is the ONLY constructor arg that raises (ValueError on non-canonical `binning_method`); it lowercases but does NOT strip spaces, so `"Optimal (CART)"` (the M2 widget label) ≠ `"optimal_cart"`. Older runs / rows duplicated from them may hold labels. Fix: `_normalize_cfg()` in M4 maps labels→values and validates enums before reconstructing the builder for Feature Journey / no-segments. `build_params()` already normalizes on save; the danger is pre-existing `suite_data.db` rows.
9. **Leaderboard date filter inversion.** `read_all_experiments()` returns rows DESC, so `dates[0]` is newest and `dates[-1]` is oldest — never use `dates[0]/dates[-1]` for the default range. Use `min(dates)`/`max(dates)`, else the inverted range filters out ALL rows ("No experiments match the current filters").

## Session-state contract (module handoffs)
- M1 writes: `loaded`, `tinfo` (per-col: n_distinct, is_binary, binary_label, event_rate, null_count, null_pct, dist_df, n_total), `target_col`, `workbench_ready`, `type_overrides` → switches via `st.switch_page("pages/2_Workbench.py")`
- M2 writes: `experiment` {exp_id, name, created_at, status, execution_time_sec, target_col, primary_key, data_rows, data_cols, config, result{segments, coverage, stop_reason, segments_count, avg_lift, max_lift, coverage_pct, baseline_rate_pct}}, `last_config`; upserts `suite_data.db`
- M2 → M3 handoff: "Run Experiment" sets `st.session_state["pending_run"]` (jsonable config) + `switch_page("pages/3_Execution_Console.py")`; M3 pops it and executes. M3 also supports re-running/re-viewing last config and previously saved experiments.
- M2 `apply_config(cfg)` maps builder-format dict → widget session keys (incl. label↔value transforms for binning_method, selection_metric, sort_priority, n_jobs); presets QUICK_DISCOVERY/CONSERVATIVE carry all params now.

## Workbench widget keys (all 23 params exposed)
wb_experiment_name, wb_description, wb_data_table, wb_target_col, wb_primary_key, wb_top_n_vars, wb_max_segments, wb_max_feature_reuse, wb_groups (+ wb_new_group/wb_group_*), wb_ignore_features, wb_enable_diversity, wb_sort_priority, wb_n_jobs, wb_binning_method, wb_naive_bins, wb_max_expansion_hops, wb_enable_1way/2way/3way, wb_selection_metric, wb_expand_log_mode, wb_min_sample_size, wb_min_lift (0.5–20.0), wb_min_events, wb_enable_grid, wb_grid_sizes, wb_grid_lifts, wb_template_name, wb_preset

## Progress
- ✅ M1 Data Loader — local/BigQuery/samples, DuckDB, 4 profiling tabs, quality report, download CSV/JSON, Proceed→Workbench
- ✅ M2 Workbench — all builder params, presets, feature groups, grid search (3×3), templates.json, validation, latest-results preview (segments + coverage tables)
- ✅ M3 Execution Console — 6-step timeline (parses library log lines), live KPIs, log terminal (level filter, copy), SQL inspector, cancel-with-partial-save (status="cancelled"), export hub (Logs.txt/SQL.sql/Config.json), suite_data.db upsert
- ✅ M4 Results Dashboard — `Module_4_results.py` + `rapidsegment_ui/pages/4_Results_Dashboard.py`. Summary cards, segments table (with weight col from scorecard), 5 Plotly charts (lift scatter, stacked distribution, **rule-complexity sunburst** — inner ring = 1/2/3-way complexity groups, outer ring = per-segment sized by population), decile line, feature-importance bar), `StrategicSegmentScore` scorecard (creates seg_N flag cols, needs max_segments≥10), scorecard JSON preview, diagnostics: **dedicated "Feature Journey" expander** (select feature → `explain_feature_journey` captured via `redirect_stdout`), Feature Health Report via `generate_feature_health_report` (no extraction needed), no-segments explanation via `explain_no_segments`. Export hub (CSV/JSON/SQL/HTML/ZIP). Reads live `st.session_state["experiment"]` or falls back to latest `suite_data.db` row + artifact `result.json`. Diagnostic builder cached in `st.session_state["m4_diag_builder"]` so journey persists across reruns.
  - **Bugs fixed during M4:** (a) Feature Journey was nested inside the no-segments button branch and used `st.code(b.explain_feature_journey(...))` which printed nothing (returns None) → moved to its own expander + `redirect_stdout`. (b) Sunburst used `branchvalues="total"` with group/root values = 0 → rendered nothing → now accumulate group/root counts. (c) `_build_diag_builder` raised `ValueError` on stale labeled `binning_method` in stored configs → added `_normalize_cfg()` (labels→values + enum validation) before reconstructing the builder (gotcha 8).
- ✅ M5 Leaderboard — `Module_5_leaderboard.py` + `rapidsegment_ui/pages/5_Leaderboard.py`. Reads all experiments from `suite_data.db` (EXP_COLS: exp_id,name,created_at,data_rows,data_cols,status,execution_time_sec,target_col,primary_key,builder_params[JSON],segments_count,avg_lift,max_lift,coverage_pct,baseline_rate,error_msg). Ranked sortable grid, sidebar filters (search/status/date-range/min-avg-lift), summary cards, per-exp sparkline (avg/max lift + coverage bar), row actions: Clone→`wb_pending`+switch 2, View Results→`load_full_experiment()` (DB row + artifacts/<exp_id>/result.json) into `st.session_state["experiment"]` + switch 4, Export Config (JSON download), Duplicate (new uuid), Delete. Two-run KPI face-off + `param_diff` (only differing keys). Wired into `app.py` (🏆 link). **Fix:** default date range now uses `min(dates)`/`max(dates)` (gotcha 9) — previously used `dates[0]`/`dates[-1]` which, with DESC ordering, produced an inverted range hiding all rows ("No experiments match the current filters").
- ✅ M6 Arena — `Module_6_arena.py` + `rapidsegment_ui/pages/6_Arena.py`. 1v1 comparison: KPI face-off (winner per metric, higher-better except exec time), full parameter diff (all keys, "Different?" flag), segment overlap (shared/unique/Jaccard, overlaid lift scatter, shared-segment lift table), SQL diff per segment. Uses `read_all_experiments` + `load_full_experiment` (artifacts/<exp_id>/result.json for segments/sql). Wired into `app.py` (⚔️ link).

## Experiment SAVE / LOAD feature — VERIFIED PRESENT
- **Saving (automatic on every run):** Module 3 `_finalise_run()` calls `upsert_experiment(exp)` → `suite_data.db` (experiments table) AND `_write_artifacts(run, exp)` → `artifacts/<exp_id>/result.json` + `scorecard.json`. Experiment name comes from the Workbench's `wb_experiment_name` field. No explicit "Save" button — implicit at run end (cancelled/failed also record status).
- **Loading (multiple paths):**
  - Module 2 "Clone from Leaderboard" → `read_leaderboard()` → `wb_pending` dict → re-applies config to Workbench.
  - Module 4 `load_experiment()` → live session else **latest** DB row (+ `load_segments_from_artifacts` recovers segments from result.json).
  - Module 5 "View Results" → `load_full_experiment(exp_id)` (DB row + result.json) into `st.session_state["experiment"]` → switch to Module 4. Also "Clone to Workbench".
  - Module 6 reads the same store for comparison.
- **Gap (optional, not requested):** No explicit user-triggered "Save Experiment" / "Load Experiment" button on a dedicated screen — save is tied to run completion; loading is via Leaderboard/Arena/Results rather than a direct "open by name" on the main page.

## Useful commands
- Run app: `streamlit run app.py` (in `rapidsegment_ui/`)
- Compile check: `python -m py_compile <files...>`
- Sync copies: `Copy-Item -Force Module_2_workbench.py rapidsegment_ui\pages\2_Workbench.py` (same for M1/M3)
- Library source (read-only reference): `C:\Users\Bishwarup\AppData\Local\Temp\opencode\rs_wheel\unzipped\rapidsegment\builder.py`
- Wheel URL: `https://files.pythonhosted.org/packages/df/0e/ccd2e5982c7d8fd09c7d7dfa6ad1eb5d641de5ea14726f11cff006d6c8cd/rapidsegment-1.2.2.post1-py3-none-any.whl`