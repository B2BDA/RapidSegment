"""Shared storage + DuckDB helpers for the RapidSegment web backend.

Mirrors the conventions of the Streamlit implementation (``rapidsegment/ui``):
everything lives under a ``.rapidsegment_suite`` runtime directory, DuckDB table
name is always ``udl_data``, and ``active_db()`` routes reads to the materialized
*modified* dataset when present, else the raw load.
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime

import duckdb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUITE_DIR = os.path.join(BASE_DIR, ".rapidsegment_suite")
os.makedirs(SUITE_DIR, exist_ok=True)

DB_FILE = os.path.join(SUITE_DIR, "module1_data.duckdb")
DB_FILE_MOD = os.path.join(SUITE_DIR, "module1_data_modified.duckdb")
PROFILING_JSON = os.path.join(SUITE_DIR, "module1_profiling.json")
SUITE_DB = os.path.join(SUITE_DIR, "suite_data.db")
ARTIFACTS_DIR = os.path.join(SUITE_DIR, "artifacts")
TEMPLATES_FILE = os.path.join(SUITE_DIR, "templates.json")
STATE_FILE = os.path.join(SUITE_DIR, "m1_state.json")
UPLOADS_DIR = os.path.join(SUITE_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

EXP_COLS = [
    "exp_id", "name", "created_at", "data_rows", "data_cols", "status",
    "execution_time_sec", "target_col", "primary_key", "builder_params",
    "segments_count", "avg_lift", "max_lift", "coverage_pct",
    "baseline_rate", "cumulative_event_capture", "error_msg", "dataset_name",
]

# ── Dataset routing ───────────────────────────────────────────────────────────
def active_db():
    """Return the materialized *modified* dataset if it exists, else the raw load."""
    return DB_FILE_MOD if os.path.exists(DB_FILE_MOD) else DB_FILE


def is_loaded():
    """True when the raw DuckDB file exists and still contains the udl_data table
    (robust to a reset that drops the table but keeps the file)."""
    if not os.path.exists(DB_FILE):
        return False
    try:
        con = duckdb.connect(DB_FILE, read_only=True)
        try:
            return con.execute(
                "SELECT 1 FROM information_schema.tables WHERE table_name='udl_data'"
            ).fetchone() is not None
        finally:
            con.close()
    except Exception:
        return False


# ── DuckDB helpers (short-lived connections, like the Streamlit code) ─────────
def db_query(db_path, sql, read_only=True):
    con = duckdb.connect(db_path, read_only=read_only)
    try:
        return con.execute(sql).df()
    finally:
        con.close()


def db_scalar(db_path, sql):
    con = duckdb.connect(db_path, read_only=True)
    try:
        row = con.execute(sql).fetchone()
        return row[0] if row else None
    finally:
        con.close()


def table_cols(db_path):
    con = duckdb.connect(db_path, read_only=True)
    try:
        return [r[0] for r in con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='udl_data' ORDER BY ordinal_position"
        ).fetchall()]
    finally:
        con.close()


# ── M1 session state (survives across stateless FastAPI requests) ─────────────
DEFAULT_STATE = {
    "dataset_name": "",
    "loaded": False,
    "tinfo": None,
    "bq_datasets": None,
    "bq_tables": None,
    "bq_preview": None,
    "type_overrides": {},
    "target_col": None,
    "data_modified": False,
}


def load_state():
    if not os.path.exists(STATE_FILE):
        return dict(DEFAULT_STATE)
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        merged = dict(DEFAULT_STATE)
        merged.update(data)
        return merged
    except Exception:
        return dict(DEFAULT_STATE)


def save_state(state):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)
    except Exception:
        pass


# ── Experiment table (contract of Module 3) ───────────────────────────────────
def upsert_experiment(exp):
    con = duckdb.connect(SUITE_DB)
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                exp_id TEXT PRIMARY KEY,
                name TEXT,
                created_at TIMESTAMP,
                data_rows INT,
                data_cols INT,
                status TEXT,
                execution_time_sec DOUBLE,
                target_col TEXT,
                primary_key TEXT,
                builder_params JSON,
                segments_count INT,
                avg_lift DOUBLE,
                max_lift DOUBLE,
                coverage_pct DOUBLE,
                baseline_rate DOUBLE,
                cumulative_event_capture DOUBLE,
                error_msg TEXT,
                dataset_name TEXT
            )
            """
        )
        for col in ("dataset_name TEXT", "cumulative_event_capture DOUBLE"):
            try:
                con.execute(f"ALTER TABLE experiments ADD COLUMN {col}")
            except Exception:
                pass
        con.execute(
            """
            INSERT OR REPLACE INTO experiments VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, CAST(? AS JSON), ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                exp["exp_id"], exp["name"], exp["created_at"],
                exp["data_rows"], exp["data_cols"], exp["status"],
                exp["execution_time_sec"], exp["target_col"], exp["primary_key"],
                json.dumps(exp["config"]),
                exp["result"]["segments_count"], exp["result"]["avg_lift"],
                exp["result"]["max_lift"], exp["result"]["coverage_pct"],
                exp["result"]["baseline_rate_pct"],
                exp["result"]["cumulative_event_capture"],
                exp["result"].get("error_msg"),
                exp.get("dataset_name", ""),
            ],
        )
    finally:
        con.close()


def read_all_experiments():
    if not os.path.exists(SUITE_DB):
        return []
    try:
        con = duckdb.connect(SUITE_DB, read_only=False)
        try:
            for col in ("dataset_name TEXT", "cumulative_event_capture DOUBLE"):
                try:
                    con.execute(f"ALTER TABLE experiments ADD COLUMN {col}")
                except Exception:
                    pass
            has = con.execute(
                "SELECT 1 FROM information_schema.tables WHERE table_name='experiments'"
            ).fetchone()
            if not has:
                return []
            rows = con.execute(
                f"SELECT {', '.join(EXP_COLS)} FROM experiments ORDER BY created_at DESC"
            ).fetchall()
        finally:
            con.close()
    except Exception as exc:
        raise RuntimeError(f"Could not read experiment database: {exc}") from exc

    out = []
    for r in rows:
        d = dict(zip(EXP_COLS, r))
        for k in ("builder_params", "error_msg"):
            if isinstance(d.get(k), str) and d[k]:
                try:
                    d[k] = json.loads(d[k])
                except Exception:
                    pass
        out.append(d)
    return out


def delete_experiment(exp_id):
    con = duckdb.connect(SUITE_DB)
    try:
        con.execute("DELETE FROM experiments WHERE exp_id = ?", [exp_id])
    finally:
        con.close()
    art = os.path.join(ARTIFACTS_DIR, exp_id)
    if os.path.isdir(art):
        shutil.rmtree(art, ignore_errors=True)


def duplicate_experiment(exp_id):
    full = load_full_experiment(exp_id)
    if not full:
        return None
    new_id = str(uuid.uuid4())
    row = dict(full)
    row["exp_id"] = new_id
    row["name"] = f"{full.get('name', 'exp')} (copy)"
    row["created_at"] = datetime.now().isoformat()
    con = duckdb.connect(SUITE_DB)
    try:
        placeholders = ",".join(["?"] * len(EXP_COLS))
        con.execute(
            f"INSERT OR REPLACE INTO experiments ({','.join(EXP_COLS)}) "
            f"VALUES ({placeholders})",
            [row.get(c) for c in EXP_COLS],
        )
    finally:
        con.close()
    os.makedirs(os.path.join(ARTIFACTS_DIR, new_id), exist_ok=True)
    src = os.path.join(ARTIFACTS_DIR, exp_id, "result.json")
    dst = os.path.join(ARTIFACTS_DIR, new_id, "result.json")
    if os.path.exists(src):
        shutil.copy(src, dst)
    return row


def read_leaderboard_raw():
    """Return (exp_id, name, created_at, builder_params) rows desc by created_at,
    or None when the suite DB has no experiments table (Module 2 clone dropdown)."""
    if not os.path.exists(SUITE_DB):
        return None
    try:
        con = duckdb.connect(SUITE_DB, read_only=True)
        try:
            has = con.execute(
                "SELECT 1 FROM information_schema.tables WHERE table_name='experiments'"
            ).fetchone()
            if not has:
                return None
            return con.execute(
                "SELECT exp_id, name, created_at, builder_params "
                "FROM experiments ORDER BY created_at DESC"
            ).fetchall()
        finally:
            con.close()
    except Exception:
        return None


def load_full_experiment(exp_id):
    path = os.path.join(ARTIFACTS_DIR, exp_id, "result.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


# ── Small JSON helper (mirrors Module 2/3 conventions) ────────────────────────
def jsonable(obj):
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)
