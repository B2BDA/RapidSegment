"""Module 1 router: Data Source & Profiling."""
from __future__ import annotations

import json
import os
import shutil
import tempfile

import duckdb
import pandas as pd

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from storage import (
    DB_FILE, DB_FILE_MOD, PROFILING_JSON, UPLOADS_DIR, active_db, db_query,
    db_scalar, is_loaded, load_state, save_state, table_cols,
)
from utils.data_loader_helpers import (
    detect_format, find_sample_datasets, is_num, persist_file_direct,
    smart_default_hint,
)

router = APIRouter(prefix="/api/m1", tags=["m1"])


class PathLoad(BaseModel):
    path: str
    encoding: str = "Auto-detect"


class SampleLoad(BaseModel):
    name: str


class BigQueryLoad(BaseModel):
    table_ref: str


class Overrides(BaseModel):
    overrides: dict = {}


class Materialize(BaseModel):
    positive_value: str | None = None


class TargetValidate(BaseModel):
    target_col: str


# ── Status ────────────────────────────────────────────────────────────────────
@router.get("/status")
def status():
    state = load_state()
    n_rows = db_scalar(active_db(), "SELECT COUNT(*) FROM udl_data") if is_loaded() else 0
    n_cols = len(table_cols(active_db())) if is_loaded() else 0
    return {
        "loaded": bool(state["loaded"]) and is_loaded(),
        "data_modified": os.path.exists(DB_FILE_MOD),
        "dataset_name": state.get("dataset_name", ""),
        "db_file": DB_FILE,
        "db_file_mod": DB_FILE_MOD,
        "active_db": active_db(),
        "db_mb": round(os.path.getsize(active_db()) / 1024 / 1024, 1) if is_loaded() else 0.0,
        "n_rows": int(n_rows or 0),
        "n_cols": n_cols,
        "target_col": state.get("target_col"),
        "type_overrides": state.get("type_overrides") or {},
        "tinfo": state.get("tinfo"),
        "max_upload_mb": 8000,
    }


# ── Source helpers ────────────────────────────────────────────────────────────
@router.get("/samples")
def samples():
    found = find_sample_datasets()
    return {"samples": [{"name": k, "path": v} for k, v in found.items()]}


@router.get("/smart-defaults")
def smart_defaults():
    return {"hints": smart_default_hint()}


class DatasetName(BaseModel):
    name: str = ""


@router.post("/dataset-name")
def set_dataset_name(body: DatasetName):
    state = load_state()
    state["dataset_name"] = body.name.strip()
    save_state(state)
    return {"dataset_name": state["dataset_name"]}


# ── Load ──────────────────────────────────────────────────────────────────────
@router.post("/load/path")
def load_path(body: PathLoad):
    if not os.path.exists(body.path):
        raise HTTPException(404, f"File not found: {body.path}")
    try:
        persist_file_direct(body.path, body.encoding,
                            dataset_name=os.path.basename(body.path))
    except Exception as exc:
        raise HTTPException(400, str(exc))
    return status()


@router.post("/load/upload")
def load_upload(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False, dir=UPLOADS_DIR)
    try:
        with tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        persist_file_direct(tmp_path, "Auto-detect",
                            dataset_name=file.filename or os.path.basename(tmp_path))
    except Exception as exc:
        raise HTTPException(400, str(exc))
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
    return status()


@router.post("/load/sample")
def load_sample(body: SampleLoad):
    found = find_sample_datasets()
    if body.name not in found:
        raise HTTPException(404, "Sample dataset not found.")
    try:
        persist_file_direct(found[body.name], "Auto-detect", dataset_name=body.name)
    except Exception as exc:
        raise HTTPException(400, str(exc))
    return status()


@router.post("/load/bigquery")
def load_bigquery(body: BigQueryLoad):
    parts = [p.strip() for p in body.table_ref.split(".") if p.strip()]
    if len(parts) not in (2, 3):
        raise HTTPException(400,
                            "Enter the table as 'project_id.dataset_id.table_id' "
                            "(or 'dataset_id.table_id').")
    pid = parts[0] if len(parts) == 3 else None
    did, tid = parts[-2], parts[-1]
    try:
        from rapidsegment.utils.data_loader import UniversalDataLoader
        data = UniversalDataLoader(project_id=pid, dataset_id=did, table_id=tid).load()
        db_write(data)
        state = load_state()
        state["loaded"] = True
        state["tinfo"] = None
        state["data_modified"] = False
        state["dataset_name"] = f"{did}.{tid}"
        save_state(state)
    except ImportError:
        raise HTTPException(400, "google-cloud-bigquery not installed. Run: pip install google-cloud-bigquery")
    except Exception as exc:
        raise HTTPException(400, f"BigQuery load failed: {exc}")
    return status()


@router.post("/browse/bigquery")
def browse_bigquery(body: dict):
    project = body.get("project")
    action = body.get("action", "datasets")
    dataset = body.get("dataset")
    try:
        from google.cloud import bigquery
        client = bigquery.Client(project=project) if project else bigquery.Client()
    except ImportError:
        raise HTTPException(400, "google-cloud-bigquery not installed: pip install google-cloud-bigquery")
    except Exception as exc:
        raise HTTPException(400, f"Browse failed: {exc}")
    try:
        if action == "datasets":
            return {"datasets": sorted(d.dataset_id for d in client.list_datasets())}
        if action == "tables":
            return {"tables": sorted(t.table_id for t in client.list_tables(f"{project}.{dataset}"))}
        if action == "preview":
            df = client.query(f"SELECT * FROM `{project}.{dataset}` LIMIT 1000").to_dataframe()
            return {"rows": df.head(1000).to_dict(orient="records"),
                    "columns": list(df.columns)}
    except Exception as exc:
        raise HTTPException(400, f"Browse failed: {exc}")
    raise HTTPException(400, "Unknown action")


def db_write(arrow_table):
    con = duckdb.connect(DB_FILE)
    con.execute("DROP TABLE IF EXISTS udl_data")
    con.execute("CREATE TABLE udl_data AS SELECT * FROM arrow_table")
    con.close()


# ── Preview / profiling ───────────────────────────────────────────────────────
@router.get("/preview")
def preview():
    if not is_loaded():
        raise HTTPException(404, "No dataset loaded.")
    df = db_query(active_db(), "SELECT * FROM udl_data LIMIT 100")
    return {
        "columns": list(df.columns),
        "rows": df.to_dict(orient="records"),
        "db_file": active_db(),
    }


def _profiling_stamp():
    return {
        "db_mtime": os.path.getmtime(active_db()),
        "target": load_state().get("target_col"),
        "overrides": load_state().get("type_overrides") or {},
    }


def build_metadata(summ):
    state = load_state()
    overrides = state.get("type_overrides") or {}

    def effective_types(summ_):
        out = {}
        for _, row in summ_.iterrows():
            inferred = "NUMERIC" if is_num(row.get("column_type", "")) else "CATEGORICAL"
            ov = overrides.get(str(row["column_name"]), "AUTO")
            out[str(row["column_name"])] = inferred if ov == "AUTO" else ov
        return out

    eff = effective_types(summ)
    stamp = f"{os.path.getmtime(active_db())}:{db_scalar(active_db(), 'SELECT COUNT(*) FROM udl_data')}"

    def top5(col, _stamp):
        return db_query(
            active_db(),
            f'SELECT CAST("{col}" AS VARCHAR) AS value, COUNT(*) AS cnt '
            f'FROM udl_data GROUP BY 1 ORDER BY 2 DESC LIMIT 5',
        )

    rows = []
    for _, row in summ.iterrows():
        col = str(row["column_name"])
        etype = eff[col]
        inferred = "NUMERIC" if is_num(row.get("column_type", "")) else "CATEGORICAL"
        null_pct = float(row.get("null_percentage") or 0)
        unique = int(row.get("approx_unique") or 0)

        if etype == "NUMERIC":
            avg = row.get("avg")
            dist = f"min={row.get('min', '—')}  max={row.get('max', '—')}"
            if avg is not None:
                dist += f"  mean={float(avg):.4g}"
        else:
            t5 = top5(col, stamp)
            dist = ", ".join(f"{str(v)}:{c:,}" for v, c in zip(t5["value"], t5["cnt"])) or "—"

        warn = "✓"
        try:
            if etype == "CATEGORICAL":
                float(str(row.get("min", "")))
                float(str(row.get("max", "")))
                warn = "⚠ looks numeric but has text"
        except (ValueError, TypeError):
            pass
        if etype == "NUMERIC" and unique <= 5:
            warn = "⚠ low cardinality"
        if null_pct > 20:
            warn = "⚠ high nulls (>20%)"

        rows.append({
            "Column": col,
            "Type": inferred if etype == inferred else f"{inferred} → {etype}",
            "Cardinality": unique,
            "Null %": f"{null_pct:.1f}%",
            "Distribution": dist,
            "Warning": warn,
        })
    return pd.DataFrame(rows)


def get_profiling():
    if not is_loaded():
        return None
    stamp = _profiling_stamp()
    if os.path.exists(PROFILING_JSON):
        try:
            with open(PROFILING_JSON, "r", encoding="utf-8") as f:
                blob = json.load(f)
            if blob.get("stamp") == stamp:
                return pd.DataFrame(blob["rows"])
        except Exception:
            pass
    summ = db_query(active_db(), "SUMMARIZE udl_data")
    df = build_metadata(summ)
    try:
        with open(PROFILING_JSON, "w", encoding="utf-8") as f:
            json.dump({"stamp": stamp, "rows": df.to_dict(orient="records")}, f)
    except Exception:
        pass
    return df


@router.get("/quality")
def quality():
    if not is_loaded():
        raise HTTPException(404, "No dataset loaded.")
    n_rows = db_scalar(active_db(), "SELECT COUNT(*) FROM udl_data")
    types = db_query(active_db(),
                     "SELECT column_name, data_type FROM information_schema.columns "
                     "WHERE table_name='udl_data' ORDER BY ordinal_position")
    summ = db_query(active_db(), "SUMMARIZE udl_data")
    state = load_state()
    overrides = state.get("type_overrides") or {}
    eff = {}
    for _, row in summ.iterrows():
        inferred = "NUMERIC" if is_num(row.get("column_type", "")) else "CATEGORICAL"
        ov = overrides.get(str(row["column_name"]), "AUTO")
        eff[str(row["column_name"])] = inferred if ov == "AUTO" else ov
    n_num = sum(1 for t in eff.values() if t == "NUMERIC")
    n_cat = len(eff) - n_num
    db_mb = os.path.getsize(active_db()) / 1024 / 1024
    profiling = get_profiling()
    return {
        "n_rows": int(n_rows or 0),
        "n_cols": len(eff),
        "n_numeric": n_num,
        "n_categorical": n_cat,
        "db_mb": round(db_mb, 1),
        "types": types.to_dict(orient="records"),
        "profiling": profiling.to_dict(orient="records") if profiling is not None else [],
        "summarize": summ.to_dict(orient="records"),
    }


# ── Overrides / materialize / reset ───────────────────────────────────────────
@router.post("/overrides")
def set_overrides(body: Overrides):
    state = load_state()
    state["type_overrides"] = body.overrides or {}
    save_state(state)
    return status()


@router.post("/materialize")
def materialize(body: Materialize):
    if not is_loaded():
        raise HTTPException(400, "Load a dataset first.")
    from services import m1_materialize
    try:
        msg, target = m1_materialize.materialize_modified(body.positive_value)
    except Exception as exc:
        raise HTTPException(400, str(exc))
    state = load_state()
    state["data_modified"] = True
    if target:
        state["target_col"] = target
        state["tinfo"] = None
    save_state(state)
    return {"message": msg, "target_col": state.get("target_col")}


@router.post("/reset")
def reset():
    from services import m1_materialize
    m1_materialize.reset_dataset()
    return status()


# ── Target selection & validation ─────────────────────────────────────────────
@router.post("/target/validate")
def validate_target(body: TargetValidate):
    if not is_loaded():
        raise HTTPException(404, "No dataset loaded.")
    sel_col = body.target_col
    if sel_col not in table_cols(active_db()):
        raise HTTPException(400, f"Column '{sel_col}' not in dataset.")
    n_total = db_scalar(active_db(), "SELECT COUNT(*) FROM udl_data")
    n_notnull = db_scalar(active_db(), f'SELECT COUNT("{sel_col}") FROM udl_data')
    n_distinct = db_scalar(active_db(), f'SELECT COUNT(DISTINCT "{sel_col}") FROM udl_data')
    dist = db_query(
        active_db(),
        f'SELECT CAST("{sel_col}" AS VARCHAR) AS val, COUNT(*) AS cnt '
        f'FROM udl_data GROUP BY "{sel_col}" ORDER BY cnt DESC LIMIT 10',
    )
    vals_nonnull = [v for v in dist["val"].tolist() if v is not None]
    BMAPS = [
        ({"0", "1"}, "0 / 1"),
        ({"true", "false"}, "True / False"),
        ({"yes", "no"}, "Yes / No"),
        ({"y", "n"}, "Y / N"),
    ]
    blabel = next((l for s, l in BMAPS if {str(v).lower() for v in vals_nonnull} == s), None)
    er = None
    if blabel:
        pos = dist.loc[dist["val"].astype(str).str.lower().isin({"1", "true", "yes", "y"}), "cnt"].sum()
        er = int(pos) / n_total if n_total else 0
    tinfo = dict(
        col=sel_col, n_distinct=int(n_distinct or 0), is_binary=blabel is not None,
        binary_label=blabel, event_rate=er,
        null_count=int(n_total or 0) - int(n_notnull or 0),
        null_pct=(int(n_total or 0) - int(n_notnull or 0)) / n_total * 100 if n_total else 0,
        dist=[{"val": str(v), "cnt": int(c)} for v, c in zip(dist["val"], dist["cnt"])],
        n_total=int(n_total or 0),
    )
    state = load_state()
    state["tinfo"] = tinfo
    state["target_col"] = sel_col
    save_state(state)
    return tinfo


@router.get("/profiling-report")
def profiling_report():
    if not is_loaded():
        raise HTTPException(404, "No dataset loaded.")
    df = get_profiling()
    if df is None:
        raise HTTPException(404, "No dataset loaded.")
    state = load_state()
    tinfo = state.get("tinfo")
    return {
        "csv": df.to_csv(index=False),
        "json": json.dumps(
            {"columns": df.to_dict(orient="records"),
             "rows": int(db_scalar(active_db(), "SELECT COUNT(*) FROM udl_data")),
             "target": tinfo["col"] if tinfo else None},
            indent=2,
        ),
    }
