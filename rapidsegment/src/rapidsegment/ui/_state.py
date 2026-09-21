"""Shared state, paths, and helpers for all RapidSegment UI pages.

Every page imports paths (SUITE_DIR, DB_FILE, …) and utility functions
(active_db, db_query, rerun, …) from here instead of duplicating them.
"""
import os

import duckdb
import pandas as pd
import streamlit as st

# ── Paths (must match modules 1–5) ──────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE) if os.path.basename(_HERE) == "pages" else _HERE
SUITE_DIR = os.path.join(_PROJECT_ROOT, ".rapidsegment_suite")
os.makedirs(SUITE_DIR, exist_ok=True)
DB_FILE = os.path.join(SUITE_DIR, "module1_data.duckdb")
DB_FILE_MOD = os.path.join(SUITE_DIR, "module1_data_modified.duckdb")
SUITE_DB = os.path.join(SUITE_DIR, "suite_data.db")
ARTIFACTS_DIR = os.path.join(SUITE_DIR, "artifacts")
PROFILING_JSON = os.path.join(SUITE_DIR, "module1_profiling.json")
TEMPLATES_FILE = os.path.join(SUITE_DIR, "templates.json")

# ── Experiment columns (authoritative — use everywhere) ──────────────────────
EXP_COLS = [
    "exp_id", "name", "created_at", "data_rows", "data_cols", "status",
    "execution_time_sec", "target_col", "primary_key", "builder_params",
    "segments_count", "avg_lift", "max_lift", "coverage_pct",
    "cumulative_event_capture", "baseline_rate", "error_msg", "dataset_name",
]

# ── Data-type helpers ────────────────────────────────────────────────────────
NUMERIC = {
    "INTEGER", "BIGINT", "DOUBLE", "FLOAT", "DECIMAL", "REAL",
    "SMALLINT", "TINYINT", "HUGEINT", "INT", "UBIGINT", "UINTEGER",
}


def is_num(t):
    return any(k in str(t).upper() for k in NUMERIC)


# ── Dataset accessor ─────────────────────────────────────────────────────────
def active_db():
    """Return the materialized *modified* dataset if it exists, else the raw load.

    Module 1 writes a transformed copy (``module1_data_modified.duckdb``) when the
    user applies metadata (type overrides + target 1/0). Every downstream read
    goes through this so DuckDB sees the actual changed types, not just the UI.
    """
    return DB_FILE_MOD if os.path.exists(DB_FILE_MOD) else DB_FILE


# ── DuckDB helpers (short-lived connections) ─────────────────────────────────
def db_write(arrow_table):
    con = duckdb.connect(DB_FILE)
    try:
        con.execute("DROP TABLE IF EXISTS udl_data")
        con.execute("CREATE TABLE udl_data AS SELECT * FROM arrow_table")
    finally:
        con.close()


def db_query(sql, read_only=True):
    path = active_db()
    if not os.path.exists(path):
        return pd.DataFrame()
    con = duckdb.connect(path, read_only=read_only)
    try:
        result = con.execute(sql).df()
    finally:
        con.close()
    return result


def db_scalar(sql):
    path = active_db()
    if not os.path.exists(path):
        return 0
    con = duckdb.connect(path, read_only=True)
    try:
        result = con.execute(sql).fetchone()[0]
    finally:
        con.close()
    return result


def db_exec(sql):
    con = duckdb.connect(active_db())
    try:
        con.execute(sql)
    finally:
        con.close()


# ── Streamlit helpers ────────────────────────────────────────────────────────
def rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


def card():
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()


def toggle(label, key, help=None):
    try:
        return st.toggle(label, key=key, help=help)
    except AttributeError:
        return st.checkbox(label, key=key, help=help)


# ── Serialisation helpers ────────────────────────────────────────────────────
def _jsonable(obj):
    """Recursively coerce *obj* to JSON-safe types (numpy scalars → Python)."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def fmt_duration(secs):
    secs = max(0, int(secs))
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m {secs % 60:02d}s"
    return f"{secs // 3600}h {secs % 3600 // 60:02d}m"


# ── SQL builders (shared by pages 3 & 4) ───────────────────────────────────
def _build_coverage_sql(segments, target):
    from rapidsegment.builder import _quote_sql_ident
    if not segments:
        return "-- No segments — coverage query skipped."
    indent = "            "
    case_sql = ("\n" + indent).join(
        f"WHEN {seg['sql_filter']} THEN {seg['segment_id']}" for seg in segments
    )
    return f"""
WITH PER_SEG_KPIS AS (
    SELECT CASE {case_sql} ELSE 0 END AS segment,
           COUNT(*) AS total_count,
           SUM(CAST({_quote_sql_ident(target)} AS DOUBLE)) AS target_events,
           (SUM(CAST({_quote_sql_ident(target)} AS DOUBLE)) * 100.0 / COUNT(*)) AS response_rate
    FROM input_data_view
    GROUP BY 1
),
BASE_KPIS AS (
    SELECT *, SUM(total_count) OVER() AS total_population,
             SUM(target_events) OVER() AS total_target_events,
             (SUM(target_events) OVER() * 1.0 / SUM(total_count) OVER()) * 100 AS base_response_rate
    FROM PER_SEG_KPIS
),
CUMULATIVE_KPIS AS (
    SELECT *, SUM(total_count) OVER (ORDER BY CASE WHEN segment = 0 THEN 999999 ELSE segment END) AS cum_count,
             SUM(target_events) OVER (ORDER BY CASE WHEN segment = 0 THEN 999999 ELSE segment END) AS cum_events
    FROM BASE_KPIS
)
SELECT segment, total_count, target_events, response_rate, base_response_rate,
       (total_count * 100.0 / total_population) AS capture_rate,
       (response_rate / NULLIF(base_response_rate, 0)) AS lift,
       (cum_count * 100.0 / NULLIF(total_population, 0)) AS cumulative_sample_capture,
       (cum_events * 100.0 / NULLIF(total_target_events, 0)) AS cumulative_event_capture
FROM CUMULATIVE_KPIS
ORDER BY CASE WHEN segment = 0 THEN 999999 ELSE segment END
"""


def _build_sql_script(segments, coverage, cfg=None, exp=None):
    cfg = cfg or {}
    exp = exp or {}
    table = cfg.get("data_table") or "udl_data"
    target = cfg.get("target_col") or ""
    lines = [
        "-- =====================================================================",
        "-- RapidSegment — deployable segment SQL",
        f"-- Experiment : {exp.get('name', '')} ({exp.get('exp_id', '')})",
        f"-- Status     : {exp.get('status', '')}",
        f"-- Target     : {target}",
        f"-- Table      : {table}",
        "-- =====================================================================",
        "",
        "-- 1. Per-segment WHERE filters (copy into your own query)",
    ]
    for s in segments:
        lines.append(f"-- Segment {s['segment_id']} · {s['rule_string']}")
        lines.append(f"SELECT * FROM {table} WHERE ({s['sql_filter']});")
        lines.append("")
    lines.append("-- 2. Full segment assignment (CASE WHEN, in extraction order)")
    lines.append("SELECT *,")
    if segments:
        case_lines = ",\n".join(
            f"         WHEN ({s['sql_filter']}) THEN {s['segment_id']}" for s in segments
        )
        lines.append(f"       CASE\n{case_lines}\n         ELSE 0 END AS segment")
    else:
        lines.append("       0 AS segment  -- no segments found")
    lines.append(f"FROM {table};")
    lines.append("")
    lines.append("-- 3. Final coverage (CTE — run against the original table)")
    lines.append(_build_coverage_sql(segments, target))
    return "\n".join(lines)


def render_coverage_table(segments, coverage, weights=None):
    """Render the coverage table with meta_applied + weight columns merged from segments."""
    if not coverage:
        st.caption("No coverage rows — experiment produced no segments.")
        return
    cov_df = pd.DataFrame(coverage)
    seg_by_id = {s["segment_id"]: s for s in segments}
    w_map = weights or {}
    for col in ("meta_applied_sample_size", "meta_applied_min_lift"):
        cov_df[col] = cov_df["segment"].map(
            lambda sid, _col=col: str(seg_by_id.get(int(sid), {}).get(_col, ""))
            if pd.notna(sid) and int(sid) != 0 else ""
        ).astype(str)
    cov_df["weight"] = cov_df["segment"].map(
        lambda sid: w_map.get(int(sid), 0) if pd.notna(sid) and int(sid) != 0 else 0
    )
    st.dataframe(cov_df, height=300, width='stretch', hide_index=True)
