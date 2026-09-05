"""Module 2 router: Workbench (config build / validate / templates / handoff)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import (
    BIN_MAP, BIN_RMAP, CONSERVATIVE, EXPAND_LOG_OPTIONS, METRIC_MAP, METRIC_RMAP,
    N_JOBS_MAP, N_JOBS_RMAP, PRESETS, QUICK_DISCOVERY, SORT_PRIORITY_MAP,
    SORT_PRIORITY_OPTIONS, SORT_PRIORITY_RMAP, build_cfg, delete_template,
    estimate_seconds, leaderboard_for_clone, load_templates, save_template,
    validate_params,
)
from storage import active_db, db_scalar, is_loaded, load_state, table_cols

router = APIRouter(prefix="/api/m2", tags=["m2"])


class ConfigBody(BaseModel):
    cfg: dict


class TemplateBody(BaseModel):
    name: str
    cfg: dict


# ── Option maps & presets ─────────────────────────────────────────────────────
@router.get("/options")
def options():
    return {
        "bin_options": list(BIN_MAP.keys()),
        "bin_map": BIN_MAP,
        "bin_rmap": BIN_RMAP,
        "metric_options": list(METRIC_MAP.keys()),
        "metric_map": METRIC_MAP,
        "metric_rmap": METRIC_RMAP,
        "sort_priority_options": [{"value": v, "label": l} for v, l in SORT_PRIORITY_OPTIONS],
        "sort_priority_map": SORT_PRIORITY_MAP,
        "sort_priority_rmap": SORT_PRIORITY_RMAP,
        "n_jobs_options": list(N_JOBS_MAP.keys()),
        "n_jobs_map": N_JOBS_MAP,
        "n_jobs_rmap": N_JOBS_RMAP,
        "expand_log_options": EXPAND_LOG_OPTIONS,
        "grid_size_range": [1000, 20000],
        "grid_lift_range": [1.0, 10.0],
    }


@router.get("/presets")
def presets():
    return {"presets": PRESETS}


# ── Dataset context ───────────────────────────────────────────────────────────
@router.get("/status")
def status():
    loaded = is_loaded()
    state = load_state()
    n_rows = db_scalar(active_db(), "SELECT COUNT(*) FROM udl_data") if loaded else 0
    return {
        "loaded": loaded,
        "n_rows": int(n_rows or 0),
        "n_cols": len(table_cols(active_db())) if loaded else 0,
        "target_col": state.get("target_col"),
        "tinfo": state.get("tinfo"),
        "dataset_name": state.get("dataset_name", ""),
        "data_table": "udl_data",
    }


@router.get("/columns")
def columns():
    if not is_loaded():
        raise HTTPException(404, "No dataset loaded.")
    cols = table_cols(active_db())
    summ = db_query_summ()
    info = {}
    for _, row in summ.iterrows():
        c = str(row["column_name"])
        u = int(row.get("approx_unique") or 99)
        info[c] = {
            "column_type": str(row.get("column_type")),
            "approx_unique": u,
            "likely_binary": u <= 2,
        }
    return {
        "columns": cols,
        "info": info,
        "target": load_state().get("target_col"),
    }


def db_query_summ():
    import duckdb
    con = duckdb.connect(active_db(), read_only=True)
    try:
        return con.execute("SUMMARIZE udl_data").df()
    finally:
        con.close()


# ── Config build / validate / estimate ────────────────────────────────────────
@router.post("/config/build")
def config_build(body: ConfigBody):
    cfg = build_cfg(body.cfg)
    all_cols = table_cols(active_db()) if is_loaded() else []
    issues = validate_params(cfg, all_cols)
    n_rows = db_scalar(active_db(), "SELECT COUNT(*) FROM udl_data") if is_loaded() else 0
    return {
        "cfg": cfg,
        "issues": issues,
        "estimate_seconds": estimate_seconds(cfg, int(n_rows or 0)),
        "columns": all_cols,
    }


@router.post("/config/validate")
def config_validate(body: ConfigBody):
    cfg = build_cfg(body.cfg)
    all_cols = table_cols(active_db()) if is_loaded() else []
    return {"issues": validate_params(cfg, all_cols)}


@router.post("/estimate")
def estimate(body: ConfigBody):
    cfg = build_cfg(body.cfg)
    n_rows = body.cfg.get("n_rows")
    if n_rows is None:
        n_rows = db_scalar(active_db(), "SELECT COUNT(*) FROM udl_data") if is_loaded() else 0
    return {"estimate_seconds": estimate_seconds(cfg, int(n_rows or 0))}


# ── Templates ─────────────────────────────────────────────────────────────────
@router.get("/templates")
def templates():
    return {"templates": load_templates()}


@router.post("/templates")
def template_save(body: TemplateBody):
    name = body.name.strip() or body.cfg.get("experiment_name", "template")
    if not name.strip():
        raise HTTPException(400, "Enter a template name first.")
    save_template(name.strip(), body.cfg)
    return {"ok": True, "name": name.strip(), "templates": load_templates()}


@router.delete("/templates/{name}")
def template_delete(name: str):
    ok = delete_template(name)
    if not ok:
        raise HTTPException(404, "Template not found.")
    return {"ok": True, "templates": load_templates()}


# ── Clone from Leaderboard ────────────────────────────────────────────────────
@router.get("/leaderboard")
def leaderboard():
    rows = leaderboard_for_clone()
    if rows is None:
        return {"available": False, "rows": []}
    return {"available": True, "rows": rows}


# ── Run handoff (validate before navigating to Module 3) ─────────────────────
@router.post("/run")
def run(body: ConfigBody):
    if not is_loaded():
        raise HTTPException(400, "No dataset loaded — run Module 1 first.")
    cfg = build_cfg(body.cfg)
    all_cols = table_cols(active_db())
    issues = validate_params(cfg, all_cols)
    n_rows = db_scalar(active_db(), "SELECT COUNT(*) FROM udl_data") or 0
    if issues:
        return {"ok": False, "issues": issues}
    return {
        "ok": True,
        "cfg": cfg,
        "estimate_seconds": estimate_seconds(cfg, int(n_rows)),
    }
