"""Shared state, paths, and helpers for all RapidSegment UI pages.

Every page imports paths (SUITE_DIR, DB_FILE, …) and utility functions
(active_db, db_query, rerun, …) from here instead of duplicating them.
"""
import os

import duckdb
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
    con = duckdb.connect(active_db(), read_only=read_only)
    try:
        result = con.execute(sql).df()
    finally:
        con.close()
    return result


def db_scalar(sql):
    con = duckdb.connect(active_db(), read_only=True)
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
