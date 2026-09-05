"""Module 6 router: Arena (1v1 Comparison)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from storage import load_full_experiment, read_all_experiments

router = APIRouter(prefix="/api/m6", tags=["m6"])


def _f(x):
    try:
        return float(x or 0)
    except (TypeError, ValueError):
        return 0.0


def _winner(a, b, higher_better=True):
    if a == b:
        return "tie"
    good_a = (a > b) if higher_better else (a < b)
    return "A" if good_a else "B"


@router.get("/experiments")
def experiments():
    rows = read_all_experiments()
    return {"rows": [
        {"exp_id": r["exp_id"], "name": r["name"], "created_at": str(r["created_at"]),
         "status": r["status"]}
        for r in rows
    ]}


@router.get("/compare")
def compare(run_a: str, run_b: str):
    fa = load_full_experiment(run_a)
    fb = load_full_experiment(run_b)
    if fa is None or fb is None:
        raise HTTPException(404, "One of the runs has no artifacts.")

    # KPI face-off
    kpis = [
        {"metric": "Avg Lift", "run_a": round(_f(fa["result"].get("avg_lift")), 3),
         "run_b": round(_f(fb["result"].get("avg_lift")), 3), "higher_better": True},
        {"metric": "Max Lift", "run_a": round(_f(fa["result"].get("max_lift")), 3),
         "run_b": round(_f(fb["result"].get("max_lift")), 3), "higher_better": True},
        {"metric": "Coverage %", "run_a": round(_f(fa["result"].get("coverage_pct")), 3),
         "run_b": round(_f(fb["result"].get("coverage_pct")), 3), "higher_better": True},
        {"metric": "Segments", "run_a": int(fa["result"].get("segments_count") or 0),
         "run_b": int(fb["result"].get("segments_count") or 0), "higher_better": True},
        {"metric": "Cumulative Event Capture %",
         "run_a": round(_f(fa["result"].get("cumulative_event_capture")), 3),
         "run_b": round(_f(fb["result"].get("cumulative_event_capture")), 3), "higher_better": True},
        {"metric": "Data Rows", "run_a": int(fa.get("data_rows") or 0),
         "run_b": int(fb.get("data_rows") or 0), "higher_better": True},
        {"metric": "Exec Time (s)", "run_a": round(_f(fa.get("execution_time_sec")), 3),
         "run_b": round(_f(fb.get("execution_time_sec")), 3), "higher_better": False},
    ]
    for k in kpis:
        k["winner"] = _winner(k["run_a"], k["run_b"], k["higher_better"])

    ca = fa.get("config") or {}
    cb = fb.get("config") or {}
    keys = sorted(set(ca) | set(cb))
    param_diff = [
        {"parameter": k, "run_a": str(ca.get(k)), "run_b": str(cb.get(k)),
         "different": ca.get(k) != cb.get(k)}
        for k in keys
    ]

    segs_a = (fa.get("result") or {}).get("segments") or []
    segs_b = (fb.get("result") or {}).get("segments") or []
    rules_a = {s.get("rule_string", ""): s for s in segs_a}
    rules_b = {s.get("rule_string", ""): s for s in segs_b}
    set_a, set_b = set(rules_a), set(rules_b)
    shared = sorted(set_a & set_b)
    uniq_a = sorted(set_a - set_b)
    uniq_b = sorted(set_b - set_a)
    union = set_a | set_b
    jac = len(shared) / len(union) if union else 0.0

    shared_rows = [
        {"rule": r,
         "a_lift": round(_f(rules_a[r].get("lift")), 3),
         "b_lift": round(_f(rules_b[r].get("lift")), 3),
         "delta_lift": round(_f(rules_b[r].get("lift")) - _f(rules_a[r].get("lift")), 3)}
        for r in shared
    ]
    all_rules = sorted(set_a | set_b)
    sql_rows = [
        {"rule": r,
         "run_a_sql": (rules_a.get(r, {}) or {}).get("sql_filter", ""),
         "run_b_sql": (rules_b.get(r, {}) or {}).get("sql_filter", "")}
        for r in all_rules
    ]
    lift_distribution = {
        "run_a": [{"x": int(s.get("segment_id") or (i + 1)),
                   "y": _f(s.get("lift"))} for i, s in enumerate(segs_a)],
        "run_b": [{"x": int(s.get("segment_id") or (i + 1)),
                   "y": _f(s.get("lift"))} for i, s in enumerate(segs_b)],
    }

    return {
        "run_a": {"exp_id": fa["exp_id"], "name": fa["name"], "status": fa["status"],
                  "created_at": str(fa["created_at"]), "target_col": fa["target_col"]},
        "run_b": {"exp_id": fb["exp_id"], "name": fb["name"], "status": fb["status"],
                  "created_at": str(fb["created_at"]), "target_col": fb["target_col"]},
        "kpis": kpis,
        "param_diff": param_diff,
        "overlap": {
            "shared_count": len(shared), "unique_a": len(uniq_a),
            "unique_b": len(uniq_b), "jaccard": round(jac, 2),
            "shared": shared_rows,
        },
        "sql_diff": sql_rows,
        "lift_distribution": lift_distribution,
        "has_segments": bool(segs_a or segs_b),
    }
