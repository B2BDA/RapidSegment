"""Module 1 materialization helpers (mirrors 1_Data_Loader.py)."""
from __future__ import annotations

import os

import duckdb

from storage import DB_FILE, DB_FILE_MOD, PROFILING_JSON, load_state, save_state, table_cols
from utils.data_loader_helpers import is_num


def _column_types():
    con = duckdb.connect(DB_FILE, read_only=True)
    try:
        return {r[0]: r[1] for r in con.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name='udl_data' ORDER BY ordinal_position"
        ).fetchall()}
    finally:
        con.close()


def materialize_modified(positive_value=None):
    """Write a transformed copy to ``module1_data_modified.duckdb``.

    Returns ``(message, new_target_col)``.
    """
    if not os.path.exists(DB_FILE):
        raise ValueError("Load a dataset first.")
    cols = table_cols(DB_FILE)
    if not cols:
        raise ValueError("No data to materialize.")

    col_types = _column_types()
    state = load_state()
    overrides = state.get("type_overrides") or {}
    target_col = state.get("target_col")
    tinfo = state.get("tinfo") or {}

    select_parts = []
    new_target = target_col
    for col in cols:
        if col == target_col and target_col:
            if positive_value is not None:
                pv = str(positive_value).replace("'", "''")
                select_parts.append(f'"{col}"')
                select_parts.append(
                    f"(CASE WHEN LOWER(TRIM(CAST(\"{col}\" AS VARCHAR)))='{pv.lower()}' "
                    f"THEN 1 ELSE 0 END)::INT AS \"{col}__binary\""
                )
                new_target = f"{col}__binary"
            elif tinfo.get("is_binary") and tinfo.get("binary_label"):
                select_parts.append(
                    f"(CASE WHEN TRY_CAST(\"{col}\" AS DOUBLE) IS NOT NULL "
                    f"THEN CAST(TRY_CAST(\"{col}\" AS DOUBLE) AS INT) "
                    f"WHEN LOWER(TRIM(CAST(\"{col}\" AS VARCHAR))) "
                    f"IN ('1','true','yes','y','t') THEN 1 ELSE 0 END)::INT AS \"{col}\""
                )
            else:
                select_parts.append(f'"{col}"')
        else:
            ov = overrides.get(str(col), "AUTO")
            if ov == "CATEGORICAL":
                select_parts.append(f'CAST("{col}" AS VARCHAR) AS "{col}"')
            elif ov == "NUMERIC":
                select_parts.append(f'TRY_CAST("{col}" AS DOUBLE) AS "{col}"')
            else:
                # AUTO: materialize exactly what the quality report *displays* —
                # BOOLEAN and other non-numeric types are shown as CATEGORICAL,
                # and the engine's naive binning cannot quantile-bin BOOLEAN. Cast
                # to VARCHAR so materialization matches the UI metadata (and the
                # engine treats it as a categorical feature).
                inferred_categorical = not is_num(col_types.get(col, ""))
                if inferred_categorical:
                    select_parts.append(f'CAST("{col}" AS VARCHAR) AS "{col}"')
                else:
                    select_parts.append(f'"{col}"')

    select_sql = ", ".join(select_parts)
    con = duckdb.connect(DB_FILE_MOD)
    src = DB_FILE.replace("\\", "/")
    try:
        con.execute(f"ATTACH '{src}' AS raw (READ_ONLY)")
        con.execute("DROP TABLE IF EXISTS udl_data")
        con.execute(f"CREATE TABLE udl_data AS SELECT {select_sql} FROM raw.udl_data")
    finally:
        con.close()

    if new_target != target_col:
        state["target_col"] = new_target
        state["tinfo"] = None
    state["data_modified"] = True
    save_state(state)
    return ("Modified dataset written and now active.", new_target)


def reset_dataset():
    try:
        con = duckdb.connect(DB_FILE)
        con.execute("DROP TABLE IF EXISTS udl_data")
        con.close()
    except Exception:
        pass
    for p in (DB_FILE_MOD, PROFILING_JSON):
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass
    state = load_state()
    state["loaded"] = False
    state["tinfo"] = None
    state["data_modified"] = False
    save_state(state)
