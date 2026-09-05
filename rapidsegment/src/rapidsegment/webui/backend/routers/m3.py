"""Module 3 router: Execution & Artifact Console."""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from services.engine import (
    build_sql_script, logs_txt, run_manager,
)
from storage import load_full_experiment, load_state


class StartBody(BaseModel):
    cfg: dict


router = APIRouter(prefix="/api/m3", tags=["m3"])


@router.post("/start")
def start(body: StartBody):
    cfg = body.cfg or {}
    if not cfg.get("target_col"):
        raise HTTPException(400, "Config has no target_col — re-run from the Workbench (Module 2).")
    state = load_state()
    try:
        exp_id = run_manager.submit(
            cfg, dataset_name=state.get("dataset_name", ""),
            event_rate=(state.get("tinfo") or {}).get("event_rate"), m1_state=state,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))
    return {"exp_id": exp_id}


@router.get("/status/{run_id}")
def status(run_id: str, after: int = 0):
    snap = run_manager.status(run_id, after=after)
    if snap is None:
        raise HTTPException(404, "Unknown run.")
    return snap


@router.post("/cancel/{run_id}")
def cancel(run_id: str):
    ok = run_manager.cancel(run_id)
    if not ok:
        raise HTTPException(404, "Unknown run.")
    return {"ok": True}


@router.get("/latest")
def latest():
    exp = run_manager.latest()
    if exp is None:
        # Fall back to the most recent completed run in the DB.
        from storage import read_all_experiments
        rows = read_all_experiments()
        if not rows:
            return {"experiment": None}
        latest_row = rows[0]
        full = load_full_experiment(latest_row["exp_id"])
        return {"experiment": full or latest_row}
    return {"experiment": exp}


@router.get("/experiment/{exp_id}")
def experiment(exp_id: str):
    full = load_full_experiment(exp_id)
    if full is None:
        raise HTTPException(404, "No artifacts for this experiment.")
    return full


@router.get("/export/{run_id}/logs.txt")
def export_logs(run_id: str):
    run = run_manager.get(run_id)
    if run is not None and not run.get("finalized"):
        text = logs_txt(run.get("logs") or [])
    else:
        art = load_full_experiment(run_id)
        text = logs_txt((art or {}).get("logs") or [])
        if not text:
            import os
            from storage import ARTIFACTS_DIR
            p = os.path.join(ARTIFACTS_DIR, run_id, "logs.txt")
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as fh:
                    text = fh.read()
    return PlainTextResponse(text or "— no log output —")


@router.get("/export/{run_id}/sql.sql")
def export_sql(run_id: str):
    art = load_full_experiment(run_id)
    if art is None:
        run = run_manager.get(run_id)
        if run is None:
            raise HTTPException(404, "Unknown run.")
        text = build_sql_script(run.get("segments") or [], run.get("coverage") or [],
                                cfg=run.get("cfg") or {}, exp=run.get("experiment") or {})
    else:
        res = art.get("result") or {}
        text = build_sql_script(res.get("segments") or [], res.get("coverage") or [],
                                cfg=art.get("config") or {}, exp=art)
    return PlainTextResponse(text)


@router.get("/export/{run_id}/config.json")
def export_config(run_id: str):
    art = load_full_experiment(run_id)
    if art is None:
        run = run_manager.get(run_id)
        if run is None:
            raise HTTPException(404, "Unknown run.")
        cfg = run.get("cfg") or {}
    else:
        cfg = art.get("config") or {}
    return json.dumps(cfg, indent=2)
