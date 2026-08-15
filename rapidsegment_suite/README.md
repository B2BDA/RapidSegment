# RapidSegment Suite - No-Code Solara Platform

**Stack:** Solara UI + DuckDB (DB + profiling) + Pandas (light viewing only) + RapidSegment library

## Run

```bash
pip install -r requirements.txt
solara run app.py --host 0.0.0.0 --port 8765
# Open http://localhost:8765
```

Or in Jupyter Lab:

```python
import solara
from app import Page
solara.display(Page())
```

## What is No-Code?

- **Upload CSV/Parquet** via drag-drop (no code)
- **DuckDB-native profiling** (no pandas for profiling):
  - Number of columns, numeric vs string vs other
  - % null rate per column via `SUM(CASE WHEN col IS NULL) / COUNT(*)`
  - Event rate of user-defined target column via `AVG(target)` / `SUM(target)`
  - All computed via `DESCRIBE table` and `SELECT COUNT(DISTINCT ...) FROM table` in DuckDB
- **Workbench**: sliders/dropdowns to tune `StrategicSegmentBuilder` params (target, top_n_vars, max_segments, min_sample_size, min_lift, binning_method, sort_priority, etc.)
- **Run**: calls `StrategicSegmentBuilder.extract_segments()` + `StrategicSegmentScore.calculate_and_export_weights()` via DuckDB
- **Leaderboard**: DuckDB OLAP query, black bar visualizations
- **Console**: logs + SQL + segments + scorecard
- **Arena**: 1v1 KPI face-off, param diff, SQL diff

## Data Profiling - DuckDB Only (no pandas)

Implemented in `rapidsegment_suite/data_profiler_duckdb.py`:

```python
profiler = DuckDBProfiler(db_path=":memory:")
profiler.load_table(file_path="data.csv", table_name="main_data") # uses read_csv, read_parquet native

report = profiler.profile(table_name="main_data", target_col="default_flag")
# report = {
#   total_rows, total_columns, num_numeric, num_string,
#   columns: [{column, type, is_numeric, is_string, null_pct, distinct_count}, ...],
#   event_info: {target, total, events, event_rate, event_rate_pct, distinct_vals, min, max}
# }
```

All null % and event rate computed via DuckDB SQL, not pandas.

## Architecture

```
.rapidsegment_suite/
├── suite_data.db          <- DuckDB experiments (params_json, segments_json, metrics_json)
├── profiling.db           <- DuckDB profiling DB
└── artifacts/
    └── exp_*/
        ├── logs.txt
        ├── query.sql
        ├── segments.json
        ├── scorecard.json
```

## Modules

1. **Data & Profiling**: FileDrop + DuckDB DESCRIBE + null % bars (solid black)
2. **Workbench**: Solara reactive state -> builder params -> Run
3. **Leaderboard**: DuckDB -> pandas light view -> black bars
4. **Console**: Split-pane logs + SQL
5. **Arena**: difflib SQL diff + param diff

## For Dev

If `rapidsegment` not installed, runner falls back to mock so UI still demoable.
