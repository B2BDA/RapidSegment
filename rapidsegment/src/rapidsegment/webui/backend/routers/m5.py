"""Module 5 router: Leaderboard (Best Experiment per Dataset)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from config import normalize_cfg
from storage import (
    delete_experiment, duplicate_experiment, load_full_experiment,
    read_all_experiments,
)

router = APIRouter(prefix="/api/m5", tags=["m5"])


class ExpIdBody(BaseModel):
    exp_id: str


@router.get("/experiments")
def experiments():
    try:
        rows = read_all_experiments()
    except RuntimeError as exc:
        raise HTTPException(500, str(exc))
    out = []
    for r in rows:
        out.append({
            "exp_id": r["exp_id"], "name": r["name"],
            "created_at": str(r["created_at"]),
            "data_rows": int(r.get("data_rows") or 0),
            "data_cols": int(r.get("data_cols") or 0),
            "status": r["status"],
            "execution_time_sec": float(r.get("execution_time_sec") or 0),
            "target_col": r["target_col"],
            "primary_key": r.get("primary_key") or "",
            "segments_count": int(r.get("segments_count") or 0),
            "avg_lift": float(r.get("avg_lift") or 0),
            "max_lift": float(r.get("max_lift") or 0),
            "coverage_pct": float(r.get("coverage_pct") or 0),
            "cumulative_event_capture": float(r.get("cumulative_event_capture") or 0),
            "baseline_rate": float(r.get("baseline_rate") or 0),
            "error_msg": r.get("error_msg"),
            "dataset_name": r.get("dataset_name") or "(unnamed)",
            "config": normalize_cfg(r.get("builder_params") or {}),
        })
    return {"rows": out}


@router.post("/delete")
def delete(body: ExpIdBody):
    delete_experiment(body.exp_id)
    return {"ok": True}


@router.post("/duplicate")
def duplicate(body: ExpIdBody):
    row = duplicate_experiment(body.exp_id)
    if row is None:
        raise HTTPException(404, "Experiment has no artifacts to duplicate.")
    return {"ok": True, "new_exp_id": row["exp_id"], "name": row["name"]}


@router.get("/export/{exp_id}/json")
def export(exp_id: str):
    full = load_full_experiment(exp_id)
    if full is None:
        raise HTTPException(404, "No artifacts for this experiment.")
    import json
    return Response(json.dumps(full, indent=2, default=str),
                    media_type="application/json")


@router.get("/clone/{exp_id}")
def clone(exp_id: str):
    full = load_full_experiment(exp_id)
    if full is None:
        raise HTTPException(404, "No artifacts for this experiment.")
    cfg = dict(full.get("config") or {})
    cfg["experiment_name"] = f"Clone-{full.get('name', 'exp')}"
    return {"cfg": normalize_cfg(cfg)}


@router.get("/compare")
def compare(run_a: str, run_b: str):
    fa = load_full_experiment(run_a)
    fb = load_full_experiment(run_b)
    if fa is None or fb is None:
        raise HTTPException(404, "One of the runs has no artifacts.")
    keys = sorted(set((fa.get("config") or {})) | set((fb.get("config") or {})))
    diffs = [
        {"parameter": k, "run_a": str((fa.get("config") or {}).get(k)),
         "run_b": str((fb.get("config") or {}).get(k))}
        for k in keys if (fa.get("config") or {}).get(k) != (fb.get("config") or {}).get(k)
    ]
    return {
        "run_a": fa,
        "run_b": fb,
        "diffs": diffs,
        "identical": not diffs,
    }
