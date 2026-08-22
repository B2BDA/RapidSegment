# RapidSegment Jupyter Native No-Code Platform: Improved Design

## Executive Summary

This document presents an **enhanced UI/UX plan** for the RapidSegment Jupyter-native platform that leverages the power of the RapidSegment library while maintaining a polished, zero-configuration Jupyter experience.

**Key Improvements:**
- Data pipeline clarity with explicit column-level profiling
- Context-aware workbench with smart parameter presets
- Rich results visualization with actionable insights
- Diagnostic drilldown capabilities
- Seamless experiment comparison and reproducibility

---

## Architecture Overview

### Tech Stack (Maintained)
- **Environment**: Jupyter Lab / Notebook
- **UI Framework**: Solara (Python-based, Material Design)
- **Backend Logic**: RapidSegment library (with Solara wrapper modules)
- **Data Engine**: DuckDB (profiling, segment validation)
- **Storage**: Local embedded DB + filesystem (`.rapidsegment_suite/`)

### System Flow

```
1. DATA SOURCE
   ↓ [Local CSV/Parquet/Excel OR BigQuery]
   ↓
2. DATA PROFILING & PREVIEW
   ↓ [Column inference, target validation, data quality checks]
   ↓
3. WORKBENCH: PARAMETER CONFIGURATION
   ↓ [Smart presets, feature groups, constraints]
   ↓
4. EXECUTION & MONITORING
   ↓ [Real-time progress, logs, segment discovery]
   ↓
5. RESULTS & VISUALIZATION
   ↓ [Tables, charts, SQL export, diagnostic reports]
   ↓
6. LEADERBOARD & ARENA
   ↓ [Experiment tracking, 1v1 comparison, templates]
```

---

## Project Status (Implementation Tracker)

> The Streamlit multipage app lives in `rapidsegment_ui/` (run `streamlit run app.py`); standalone module files (`Module_*.py`) mirror the pages.

| Module | Status | Notes |
|---|---|---|
| 1 — Data Source & Profiling | ✅ **Done** | Local file / BigQuery / sample datasets, DuckDB persistence, profiling tabs, target validation, quality report, Proceed→Workbench |
| 2 — Workbench | ✅ **Done** | All 23 `StrategicSegmentBuilder` constructor params exposed (incl. 14 `sort_priority` values, `n_jobs`, `expand_log_mode`), presets, feature groups, grid search, templates, validation, latest-results preview |
| 3 — Execution & Artifact Console | ✅ **Done** | 6-step timeline, live KPIs, log terminal + SQL inspector, cancel-with-partial-save, export hub (Logs.txt / SQL.sql / Config.json), `suite_data.db` persistence |
| 4 — Results Dashboard & Visualization | ✅ **Done** | Summary cards, segments table (with scorecard weight column), 5 Plotly charts (lift-vs-volume scatter, stacked distribution, **rule-complexity sunburst** — inner ring groups segments by 1/2/3-way complexity, outer ring per segment sized by population, `StrategicSegmentScore` scorecard + JSON preview, diagnostics with **dedicated Feature Journey expander** (audit trail per feature), feature health report, no-segments explanation, export hub (CSV/JSON/SQL/HTML/ZIP) |
| 5 — Leaderboard | ✅ **Done** | Ranked grid (name/date/size/segments/lift/coverage/status, sortable), sidebar filters (search, status, date range, min avg-lift), summary cards (count, avg time, best lift, top binning method), per-experiment sparkline + row actions (Clone to Workbench, View Results, Export Config, Duplicate, Delete), two-run KPI face-off + parameter-diff |
| 6 — Arena (1v1 comparison) | ✅ **Done** | KPI face-off with winners, full parameter diff (differing fields flagged), segment overlap (shared/unique/Jaccard + overlaid lift distribution + shared-segment lift table), SQL diff per segment; wired into `app.py` |

**Known caveats / deviations:**
- ⚠ `builder.evaluate_final_coverage()` hangs in this environment (DuckDB file-lock on shared `db_path`) — never call it; coverage is computed locally (`compute_coverage_local`).
- Scorecard (`StrategicSegmentScore`) needs ≥10 distinct segments for smooth decile resolution (library warns otherwise) — guide users to `max_segments ≥ 10`.
- Solara→Streamlit conversion: `st.switch_page` / `st.page_link` for navigation; there is no Jupyter-native mode.
- `db_path`/`db_temp_dir` are internal-only (per-experiment artifact dirs), not user-facing.

### Recently fixed (post-implementation)
- **M4 diagnostics crash (Feature Journey / no-segment explanation):** `_build_diag_builder` reconstructs `StrategicSegmentBuilder` from the stored experiment config. The constructor's only hard validation is on `binning_method`, which lowercases but does **not** strip spaces, so a stored *label* like `"Optimal (CART)"` (the M2 widget label) ≠ `"optimal_cart"` and raised `ValueError`. A new `_normalize_cfg()` maps labels→canonical values and validates enums before reconstruction, so pre-existing/`duplicate` rows with stale labels now work.
- **M5 Leaderboard "No experiments match the current filters":** experiments are read DESC, so the default date range was `(newest, oldest)` — an inverted range that filtered out everything. Now uses `min(dates)`/`max(dates)`.

---

## Module 1: Enhanced Data Source & Profiling

### Purpose
Load data from multiple sources and deliver instant, actionable insights about data quality and readiness.

### Design Details

#### 1.1 Source Selection
- **Radio buttons**: Local File | BigQuery | Sample Datasets
- **Smart defaults**: Auto-detect common paths (e.g., `./data/`, `~/Downloads/`)
- **Sample datasets**: Quick-start with RapidSegment example datasets (bank-full.csv, train.csv)

#### 1.2 File Upload & Parsing
- **Drag-and-drop zone** with file size limits
- **Format detection**: Auto-infer CSV vs. Parquet vs. Excel vs. Arrow
- **Encoding support**: UTF-8, Latin-1, auto-detect
- **Progress indicator**: File load, profiling progress

#### 1.3 BigQuery Integration
- **OAuth flow** (one-time, cached in `.rapidsegment_suite/`)
- **Dataset/table autocomplete** (lazy-loaded dropdown)
- **Streaming preview** (first 1000 rows without full table scan)
- **Size estimation** before load

#### 1.4 Data Preview & Column Profiling

**Preview Table** (fixed height, scrollable):
- First 100 rows, sortable columns, filterable

**Column Metadata Panel** (right sidebar or collapsible):
For each column:
- **Name**, **Type** (inferred or manual override)
- **Cardinality** (# unique values)
- **Null %** (missing data ratio)
- **Distribution** (for numeric: min/max/mean; for categorical: top 5 values)
- **Type warnings** (e.g., "Column X looks numeric but has text")

#### 1.5 Target Column Selection & Validation
- **Dropdown** with column list, pre-filtered to binary columns (or multi-class warning)
- **Validation checks**:
  - ✓ Binary (0/1, True/False, Yes/No)
  - ⚠ Multi-class (offer binarization helper)
  - ✓ Event rate (display: "3.5% event rate" with color coding)
  - ⚠ Class imbalance warning (if >99% or <1% events)
- **Class distribution chart**: Mini bar chart (events vs. non-events)

#### 1.6 Data Quality Report
Inline metrics:
- Total rows, columns, numeric/categorical split
- Missing value summary (% per column, worst offenders flagged)
- Memory footprint (in MB)
- Recommendation: "Data ready for segmentation ✓" or warnings

#### 1.7 Actions
- **"Proceed to Workbench" button** (enabled only after target selected)
- **"Upload Different File" link** (reset flow)
- **"Download Profiling Report"** (CSV or JSON)

#### 1.8 Module Tech Stack
- **UI**: Solara `InputText`, `Select`, `FileUpload`, custom Solara components
- **Profiling**: DuckDB queries for descriptive stats
- **Preview**: `ipydatagrid` or Solara's native table viewer
- **Charts**: Lightweight Plotly or Recharts (via AnyWidget)

---

## Module 2: The Workbench (Enhanced)

### Purpose
Configure all RapidSegment parameters with smart defaults, interactive validation, and parameter presets.

### Design Details

#### 2.1 Layout: Two-Column Design
- **Left Column**: Grouped parameter inputs (collapsible sections)
- **Right Column**: Real-time parameter summary + validation checklist

#### 2.2 Parameter Sections (Left Column)

##### 2.2.1 Basic Settings
- **Experiment Name** (text input, auto-generated: `exp_YYYY-MM-DD_HH-MM`)
- **Description** (optional, multi-line text)
- **Data table name** (pre-selected from profiling, editable)
- **Target column** (pre-selected, editable)
- **Primary key column** (for scorecard; optional but recommended)

##### 2.2.2 Segment Discovery Strategy
- **top_n_vars** (slider, 5–50, default 15)
- **max_segments** (slider, 1–20, default 10)
- **max_feature_reuse** (slider, 1–5, default 1)
- **Feature grouping** (collapsible section):
  - Add business categories (e.g., "Delinquency", "Utilization")
  - Multi-select columns per group
  - Enable diversity toggle (prevents mixing groups in one rule)
- **Ignore features** (multi-select of column list)

##### 2.2.3 Binning & Rule Complexity
- **Binning method** (radio buttons):
  - "Optimal (CART)" – target-aware cuts, slower but more predictive
  - "Optimal (Quantile)" – stable quantile cuts
  - "Naive" – fast equal-frequency quantiles (for large datasets)
- **Naive bins** (slider, 3–20, only visible if "Naive" selected)
- **Max expansion hops** (slider, 0–5, default 0)
- **Rule generation** (toggles): Enable 1-way, 2-way, 3-way rules
- **Selection metric** (dropdown: "IV" or "Response Rate")

##### 2.2.4 Hard Constraints
- **min_sample_size** (number input, default 1000)
- **min_lift** (decimal input, default 1.5)
- **min_events** (number input, default 100)

##### 2.2.5 Advanced: Grid Search (Optional)
- **Enable grid search** (toggle, default False)
- If enabled:
  - Multi-select: min_sample_size values (e.g., 500, 1000, 2000)
  - Multi-select: min_lift values (e.g., 1.5, 2.0, 3.0)
  - Display: "Evaluating X combinations" (updates in real-time)

#### 2.3 Right Column: Parameter Summary & Validation

**Real-Time Summary** (card-based layout):
- 📊 Segment Discovery stats
- 🔢 Rule Complexity settings
- ⚙️ Constraints overview

**Validation Checklist**:
- ✓ Target column selected
- ✓ Data loaded (X rows, Y columns)
- ⚠ Class imbalance detected
- ✓ Parameters valid
- ⚠ Grid search time estimate

**Preset Templates**:
- "Quick Discovery" (aggressive)
- "Conservative" (strict)
- "Last experiment" (clone previous)

#### 2.4 Action Bar (Sticky Footer)
- **"Save as Template" button**
- **"Clone from Leaderboard" dropdown**
- **"Run Experiment" button** (prominent, black)
- **Estimated time**

#### 2.5 Module Tech Stack
- **UI**: Solara reactive components
- **State**: JSON-based parameter dictionaries
- **Validation**: Client-side checks
- **Templates**: JSON stored in `.rapidsegment_suite/templates.json`

---

## Module 3: Real-Time Execution & Artifact Console

### Purpose
Monitor extraction progress and immediately access logs, SQL filters, and generated artifacts.

### Design Details

#### 3.1 Progress Tracking
- **Status Timeline**: 6 extraction steps with visual indicators
- **Elapsed time** (e.g., "3m 42s")
- **Segment counter**: "Found 3 segments so far..."
- **Current feature** being processed

#### 3.2 Live Metrics Panel
- **Real-time KPIs** (update every 2–5 seconds):
  - Segments found (count + top candidates)
  - Total coverage %
  - Average lift
  - Best segment so far

#### 3.3 Split-Pane Console
- **Left Pane: Log Terminal**
  - Dark background, monospace font
  - Auto-scroll, filterable by level (Info/Warning/Error)
  - Copy-to-clipboard button
  
- **Right Pane: SQL Inspector**
  - Live SQL filter generation
  - Syntax highlighting
  - Copy SQL button per segment

#### 3.4 Cancel & Interrupt
- **Cancel button** (red, prominent during extraction)
- Graceful shutdown with partial results saved

#### 3.5 Export Hub (After Completion)
- **Download buttons**: Logs.txt, SQL.sql, Config.json

#### 3.6 Module Tech Stack
- **UI**: Solara split-pane, text areas
- **Logging**: Python `logging` piped to Solara state
- **Real-time**: Async threads + state reactivity
- **Syntax**: Pygments for SQL highlighting

---

## Module 4: Results Dashboard & Visualization — ✅ Done

### Purpose
Display extracted segments with rich context, metrics, and actionable export options.

### Design Details

#### 4.1 Summary Cards (Top Section)
- Total Segments extracted
- Coverage % of population
- Average Lift (color-coded)
- Execution Time
- Baseline Event Rate

#### 4.2 Segments Table (Main Section)
**Columns**: Segment | Rule | Count | Event Rate | Lift | Capture % | Weight | Actions
- Sortable by any column
- Expandable rows (full SQL WHERE clause)
- Copy-to-clipboard for SQL

#### 4.3 Visualizations Tab (5 Chart Types)

1. **Lift vs. Volume Scatter**
   - X: Count (volume)
   - Y: Lift
   - Bubble size: Capture rate
   - Color: Segment ID

2. **Segment Distribution (Stacked Bar)**
   - Events (color) + Non-Events (lighter)
   - Hover for percentage breakdown

3. **Rule Complexity Breakdown (Sunburst)**
   - Inner: 1-way/2-way/3-way split
   - Outer: Individual segments
   - Size/color by lift

4. **Decile Thresholds (Line Chart)**
   - X: Decile (1–10)
   - Y: Min score threshold

5. **Feature Importance (Horizontal Bar)**
   - X: # times feature used
   - Y: Feature name
   - Color: By feature group

#### 4.4 Scorecard JSON Preview
- Expandable JSON viewer
- model_metadata, segment_weights, decile_min_thresholds
- Copy-to-clipboard

#### 4.5 Diagnostic Drilldown
- **Feature Journey**: Audit trail per feature
- **Feature Health Report**: Bin-level stats (downloadable CSV)
- **No Segments Explanation**: Why extraction stopped + recommendations

#### 4.6 Export Hub
- CSV (segments)
- JSON (scorecard)
- SQL (deployment-ready)
- HTML (printable report)
- ZIP (all artifacts)

#### 4.7 Actions
- "Run Another Experiment"
- "Save Experiment"
- "Compare with Another" (send to Arena)

#### 4.8 Module Tech Stack
- **Tables**: ipydatagrid or Solara table viewer
- **Charts**: Recharts/Plotly + D3.js
- **JSON viewer**: Custom Solara component
- **Export**: Python pathlib, pandas, JSON, HTML generation

---

## Module 5: Enhanced Leaderboard (Experiment Tracking)

### Purpose
Central hub to track all experiments, rank by metrics, and enable quick cloning or comparison.

### Design Details

#### 5.1 Ranked Data Grid
**Columns**: Rank | Name | Date | Data Size | Segments | Avg Lift | Max Lift | Status | Actions
- Sortable by any column
- Filterable by date range, status, lift threshold

#### 5.2 Inline Sparklines
- Segment distribution (mini bar chart)
- Lift trend (mini line chart)
- Rule complexity split

#### 5.3 Row-Level Actions
- Clone to Workbench
- View Results
- Compare with Another
- Delete
- Duplicate
- Export

#### 5.4 Summary Stats (Top)
- Total experiments
- Avg. extraction time
- Most used parameters
- Best-performing experiment

#### 5.5 Module Tech Stack
- **Database**: DuckDB or SQLite queries
- **UI**: ipydatagrid or Solara table viewer
- **Charts**: Lightweight SVG or Plotly micro-charts

---

## Module 6: Arena (1v1 Comparison)

### Purpose
Side-by-side analytical view to understand why one experiment outperformed another.

### Design Details

#### 6.1 Experiment Selection
- Dropdown A: Select first experiment
- Dropdown B: Select second experiment
- "Compare" button

#### 6.2 KPI Face-Off
```
Metric Name       |  Exp A Value  |  Δ  |  Exp B Value
────────────────────────────────────────────────────
Avg Lift          |     2.1       | ← → |     1.8
Max Lift          |     3.2       | ← → |     2.9
# Segments        |      5        | ← → |      8
Coverage %        |    42.1%      | ← → |    51.3%
Execution Time    |    3m 42s     | ← → |    5m 21s
```
- Delta bar chart for visual representation

#### 6.3 Parameter Diff
Show **only parameters that differ**

#### 6.4 Segment Comparison Table
Side-by-side: Exp A segments vs. Exp B segments

#### 6.5 SQL Diff Viewer (GitHub-Style)
- Left pane: Exp A SQL
- Right pane: Exp B SQL
- Highlighted diff lines

#### 6.6 Performance Implications
- Recommendations based on diff
- Example: "Naive binning: 3 more segments, 1.5 min faster"

#### 6.7 Module Tech Stack
- **Diff engine**: Python difflib or JS-based viewer
- **UI**: Multi-column Solara layout

---

## Storage Architecture

### Directory Structure
```
my_project/
├── notebook.ipynb
├── data.csv
└── .rapidsegment_suite/
    ├── suite_data.db
    ├── profiling.db
    ├── templates.json
    ├── oauth_cache.json
    └── artifacts/
        ├── exp_20250214_abc123/
        │   ├── metadata.json
        │   ├── segments.csv
        │   ├── scorecard.json
        │   ├── sql_script.sql
        │   ├── logs.txt
        │   └── report.html
        └── exp_20250213_def456/
```

### Database Schema (suite_data.db)

**Table: experiments**
```sql
CREATE TABLE experiments (
    exp_id TEXT PRIMARY KEY,
    name TEXT,
    created_at TIMESTAMP,
    data_rows INT,
    data_cols INT,
    status TEXT,
    execution_time_sec FLOAT,
    target_col TEXT,
    primary_key TEXT,
    builder_params JSON,
    segments_count INT,
    avg_lift FLOAT,
    max_lift FLOAT,
    coverage_pct FLOAT,
    baseline_rate FLOAT,
    error_msg TEXT
);
```

**Table: segments**
```sql
CREATE TABLE segments (
    segment_id TEXT,
    exp_id TEXT REFERENCES experiments(exp_id),
    segment_seq INT,
    rule_string TEXT,
    sql_filter TEXT,
    count INT,
    event_count INT,
    event_rate FLOAT,
    lift FLOAT,
    capture_rate FLOAT,
    weight INT,
    PRIMARY KEY (segment_id, exp_id)
);
```

**Table: templates**
```sql
CREATE TABLE templates (
    template_id TEXT PRIMARY KEY,
    name TEXT,
    description TEXT,
    builder_params JSON,
    created_at TIMESTAMP
);
```

---

## User Flows

### Flow 1: Quick Experiment (New User)
1. Open notebook, run Solara app
2. Select: "Sample Datasets"
3. Load "bank-full.csv"
4. Auto-detect columns, select target
5. Click "Use Defaults & Run"
6. Monitor progress
7. View results + export

**Time**: ~5–7 minutes

### Flow 2: Advanced Tuning (Power User)
1. Load custom CSV
2. Profile data
3. Configure: feature groups, grid search (3×3), binning method
4. Save as template "FastBinning"
5. Run experiment
6. Compare with previous run in Arena
7. Export SQL + HTML report

**Time**: ~15–25 minutes

### Flow 3: Diagnostic Deep-Dive
1. After experiment, view diagnostic details
2. Inspect Feature Journey (e.g., max_dpd_12m)
3. View Feature Health Report
4. Understand feature selection
5. Adjust constraints and re-run

**Time**: ~10 minutes

---

## Implementation Phases

### Phase 1: MVP (Weeks 1–3)
**Deliverables:**
- Data Source & Profiling module (local + basic BQ)
- Workbench with core parameters
- Real-time progress (basic status bar)
- Results table (basic export: CSV segments)
- Leaderboard (simple list)

**Out of scope:**
- Grid search, templates, diagnostics, Arena
- Advanced visualizations, HTML reports

### Phase 2: Enhanced UI (Weeks 4–6)
**Deliverables:**
- Full results dashboard with visualizations
- Diagnostic drilldown
- Export options (JSON, SQL, HTML)
- Template management
- Improved leaderboard (filtering, sorting, inline charts)

**Out of scope:**
- Arena comparison, experiment branching

### Phase 3: Arena & Advanced (Weeks 7–8)
**Deliverables:**
- Arena (1v1 comparison)
- Experiment cloning + branching
- Grid search execution + visualization
- Performance optimizations (async, caching, streaming)

---

## Key Improvements Over Original MVP

### 1. Data Profiling Clarity
- Column-level metadata (cardinality, null %, distribution)
- Target validation with class imbalance warnings
- Data quality score before execution

### 2. Smart Workbench
- Preset templates (Quick Discovery, Conservative, Last Experiment)
- Context-aware validation (grid search time estimate)
- Feature groups + diversity toggle
- Clone-from-leaderboard for reproducibility

### 3. Rich Results
- Multiple visualization types (lift scatter, distribution, complexity)
- Scorecard JSON preview
- Diagnostic drilldown (feature journeys, health reports)
- Multi-format export (CSV, JSON, SQL, HTML)

### 4. Leaderboard as Central Hub
- Inline sparklines for quick insight
- One-click cloning or comparison
- Ranked by multiple metrics

### 5. Arena for Comparison
- KPI face-off with delta visualization
- Parameter diff (only changed values)
- SQL diff viewer
- Actionable recommendations

### 6. Storage & Reproducibility
- Structured DB for fast querying
- JSON-based templates for sharing configs
- Artifact organization by experiment ID
- Logs + metadata for debugging

---

## Accessibility & Performance

### Accessibility (A11y)
- ARIA labels on all interactive elements
- Keyboard navigation (Tab, Enter, Escape, Arrow keys)
- High contrast mode for charts
- Color-blind friendly palettes

### Performance
- **Frontend**: Virtualize large tables (100+ segments), lazy-load charts
- **Backend**: Async execution thread, real-time state updates
- **Data transfer**: Stream logs, paginate large result sets
- **Caching**: Cache profiling results, experiment metadata

---

## Success Metrics

By end of Phase 3:
- **New users**: first experiment in <5 min
- **Power users**: configure + compare in <20 min
- **Performance**: extraction in <5 min for 50K rows
- **Reliability**: 100% artifact preservation, zero data loss
- **Satisfaction**: Clear results, one-click exports, full transparency

---

## Next Steps

1. ✅ **Review & approve** this enhanced design
2. ✅ **Implement Module 1** (Data Loader) — `Module_1_data_loader.py`
3. ✅ **Implement Module 2** (Workbench) — `Module_2_workbench.py`
4. ✅ **Implement Module 3** (Execution & Artifact Console) — `Module_3_execution.py`
5. ✅ **Implement Module 4** (Results Dashboard) — `Module_4_results.py`
6. ⬜ **Implement Module 5** (Leaderboard) — reads `suite_data.db` experiments; clone-to-workbench via `apply_config`; sparklines
7. ⬜ **Implement Module 6** (Arena) — 1v1 KPI face-off, param diff, SQL diff
8. ⬜ **Beta test** with internal users, then iterate