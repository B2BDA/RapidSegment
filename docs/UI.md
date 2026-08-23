# RapidSegment — Web UI Guide

RapidSegment ships with a no-code, multi-page **Streamlit** application that wraps the `rapidsegment` engine. It is organized as a **Home** hub plus six modules (**M1–M6**), each mapped to one stage of the segmentation workflow. This guide covers installation, launch, and what every page offers.

> Runtime state (experiments, artifacts, and the DuckDB suite database) is stored under `.rapidsegment_suite/` next to the app.

## Installation

The UI is an optional extra. Installing it pulls in `streamlit` and `plotly`:

```bash
pip install "rapidsegment[ui]"
```

If you also want Excel upload or the BigQuery connector, add those extras:

```bash
pip install "rapidsegment[ui,excel,gcp]"
```

### Install directly from `main`

From a local checkout (editable — changes reflect immediately, best for development):

```bash
cd rapidsegment
pip install -e ".[ui]"
```

Or straight from the remote `main` branch, no clone required:

```bash
pip install "git+https://github.com/B2BDA/RapidSegment.git@main#subdirectory=rapidsegment&extras=ui"
```

## Launch

Once installed, start the app with a single command:

```bash
rapidsegment-ui
```

This opens the Streamlit server at **http://localhost:8501** and auto-discovers the six module pages. Equivalent ways to start it:

```bash
python -c "from rapidsegment.ui import run_ui; run_ui()"
# or point Streamlit directly at the installed entry script:
streamlit run <path-to-site-packages>/rapidsegment/ui/app.py
```

### Stopping the app

Use the **Exit UI** button in the sidebar — it is available on every page and terminates the Streamlit **server process** (not just the current page), so the app fully closes.

## Theme

The UI uses a custom **emerald "hacker-terminal"** look: black background, emerald primary `#34D399`, light-emerald text `#6EE7B7`, JetBrains Mono font, and a subtle grid / CRT-scanline background with glassmorphism on the sidebar and header. It is applied on every page via `apply_cyberpunk_theme()` (`rapidsegment/ui/_theme.py`). `.streamlit/config.toml` exists solely to raise Streamlit's upload/message limit (`[server] maxUploadSize/maxMessageSize = 2000`); the theme palette is injected by `_theme.py`, not declared in config.toml.

One notable detail: the **Leaderboard 🏆 Best performer** card is rendered as a bright-emerald `st.success` box with black text for high contrast (targeted via `[data-testid="stAlertContentSuccess"]` in `_theme.py`).

## Pages & Modules

### Home
Navigation hub that links to every module.

### M1 · Data Loader & Profiling
- Load data from a local file, BigQuery, or built-in samples.
- **BigQuery**: enter a table directly as `project_id.dataset_id.table_id` (or `dataset_id.table_id` to use your default GCP project) and click **Load table**. Authentication uses your **environment** credentials — `gcloud auth application-default login` or `GOOGLE_APPLICATION_CREDENTIALS` — so no secrets are stored in the app. The `google-cloud-bigquery` client is optional (install via the `gcp` extra); if it is missing, the panel tells you how to install it. An optional **Browse BigQuery** expander lists datasets/tables and previews the first 1,000 rows using the same environment credentials.
- Automatic column profiling, a data-quality report, and CSV / JSON download.
- **Column Metadata**: set type overrides (Categorical / Numeric / etc.). Applying them writes a *transformed* copy, `module1_data_modified.duckdb` (real DuckDB column types plus target encoded as `0/1`), which every downstream module reads automatically via `active_db()`.
- **Dataset name**: a label stored with each experiment so the Leaderboard can group runs by dataset.

### M2 · Workbench
- Configure the `StrategicSegmentBuilder` (binning method, Apriori pruning thresholds, grid-search settings, max segments, target / primary-key columns).
- Save and load configurations and preview before running.

### M3 · Execution Console
- Runs segment extraction with a 6-step live timeline, live KPIs, a filterable log terminal, and a SQL inspector.
- **Cancel with partial save**: stopping a run still persists what was extracted (status = `cancelled`).
- Export hub (logs / SQL / config JSON) and automatic persistence of the experiment into `suite_data.db` (including `dataset_name`).

### M4 · Results Dashboard
- Summary cards and a sortable segments table (with `weight`).
- Five Plotly charts: lift scatter, stacked distribution, **rule-complexity sunburst** (1 / 2 / 3-way), decile calibration line, and feature-importance bar.
- `StrategicSegmentScore` scorecard (decile scorecard; needs `max_segments >= 10`).
- **Feature Journey** (how a feature is used across segments) and **Feature Health Report** (`generate_feature_health_local` — per-feature bin statistics that respect type overrides; numeric bins are cast so min / max ordering is correct).
- No-segments explanation and an export hub (CSV / JSON / SQL / HTML / ZIP).

### M5 · Leaderboard
- Best experiment **per dataset**, ranked by a chosen KPI (avg lift, max lift, coverage %, segment count, or **Cumulative Event Capture %**) with a **best-performer** highlight (the emerald 🏆 card).
- Light filters (status + name search), summary cards, row actions (clone to Workbench, view results, export JSON, duplicate, delete), and a two-run **Compare** view.

### M6 · Arena
- 1v1 experiment comparison: KPI face-off with winners (avg lift, max lift, coverage %, segments, **Cumulative Event Capture %**, data rows, exec time), a full parameter diff (differing fields highlighted), segment-overlap analysis with overlaid lift distributions, and a SQL diff of matching segments.

## Tutorial Video

<!-- Add your tutorial video here (YouTube embed or recorded demo link). -->

_Coming soon._
