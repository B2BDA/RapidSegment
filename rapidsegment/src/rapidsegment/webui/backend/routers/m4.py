"""Module 4 router: Results Dashboard & Visualization."""
from __future__ import annotations

import io
import json
import os
import zipfile

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel

from services.engine import build_sql_script, logs_txt, run_manager
from services.m4 import (
    build_diag_builder, build_scorecard, eligible_features_from_dataset,
    fig_decile, fig_distribution, fig_feature_importance, fig_scatter,
    fig_sunburst, generate_feature_health_local, map_weights,
    tracked_features_from_artifacts,
)
from storage import (
    ARTIFACTS_DIR, SUITE_DIR, active_db, is_loaded, load_full_experiment,
    load_state, read_all_experiments,
)

router = APIRouter(prefix="/api/m4", tags=["m4"])


class ScoreBody(BaseModel):
    exp_id: str
    cfg: dict = {}


class JourneyBody(BaseModel):
    feature: str
    exp_id: str | None = None
    cfg: dict = {}


class HealthBody(BaseModel):
    features: list[str]
    exp_id: str | None = None
    cfg: dict = {}


class NoSegBody(BaseModel):
    exp_id: str | None = None
    cfg: dict = {}


def _load_experiment(exp_id=None):
    """Return (exp, source_label). Prefers the live run, else latest DB row."""
    live = run_manager.latest()
    if exp_id:
        full = load_full_experiment(exp_id)
        if full:
            return full, "artifacts"
        raise HTTPException(404, f"No artifacts for `{exp_id}`.")
    if live and live.get("result"):
        return live, "live session"
    rows = read_all_experiments()
    if not rows:
        return None, "no experiments"
    latest_row = rows[0]
    full = load_full_experiment(latest_row["exp_id"])
    if full:
        return full, "artifacts (latest)"
    exp = {
        "exp_id": latest_row.get("exp_id"), "name": latest_row.get("name"),
        "created_at": str(latest_row.get("created_at")),
        "status": latest_row.get("status"),
        "execution_time_sec": float(latest_row.get("execution_time_sec") or 0),
        "target_col": latest_row.get("target_col"),
        "primary_key": latest_row.get("primary_key") or "",
        "data_rows": int(latest_row.get("data_rows") or 0),
        "data_cols": int(latest_row.get("data_cols") or 0),
        "config": latest_row.get("builder_params") or {},
        "result": {
            "segments_count": int(latest_row.get("segments_count") or 0),
            "avg_lift": float(latest_row.get("avg_lift") or 0),
            "max_lift": float(latest_row.get("max_lift") or 0),
            "coverage_pct": float(latest_row.get("coverage_pct") or 0),
            "baseline_rate_pct": float(latest_row.get("baseline_rate") or 0),
            "cumulative_event_capture": float(latest_row.get("cumulative_event_capture") or 0),
            "error_msg": latest_row.get("error_msg"),
            "segments": [], "coverage": [], "stop_reason": None,
        },
        "logs": [],
    }
    return exp, "suite_data.db (latest)"


def _segments_coverage(exp):
    segments = (exp.get("result") or {}).get("segments") or []
    coverage = (exp.get("result") or {}).get("coverage") or []
    if segments:
        return segments, coverage
    saved = load_full_experiment(exp.get("exp_id") or "")
    if saved:
        res = saved.get("result") or {}
        return res.get("segments") or [], res.get("coverage") or []
    return segments, coverage


@router.get("/experiments")
def experiments():
    rows = read_all_experiments()
    return {"rows": [
        {"exp_id": r["exp_id"], "name": r["name"], "created_at": str(r["created_at"]),
         "status": r["status"], "target_col": r["target_col"]}
        for r in rows
    ]}


@router.get("/summary")
def summary(exp_id: str | None = None):
    exp, source = _load_experiment(exp_id)
    if exp is None:
        return {"experiment": None, "source": source}
    segments, coverage = _segments_coverage(exp)
    cfg = exp.get("config") or {}
    if not isinstance(cfg, dict):
        cfg = {}
    scorecard = None
    sp = os.path.join(ARTIFACTS_DIR, exp.get("exp_id", ""), "scorecard.json")
    if os.path.exists(sp):
        try:
            with open(sp, "r", encoding="utf-8") as fh:
                scorecard = json.load(fh)
        except Exception:
            scorecard = None
    weights = map_weights(scorecard, segments)
    features = (tracked_features_from_artifacts(exp.get("exp_id"))
                or eligible_features_from_dataset(cfg))
    return {
        "experiment": exp,
        "source": source,
        "segments": segments,
        "coverage": coverage,
        "scorecard": scorecard,
        "weights": weights,
        "features": features,
        "cfg": cfg,
        "dataset_available": is_loaded() and os.path.exists(active_db()),
    }


@router.post("/scorecard")
def scorecard(body: ScoreBody):
    cfg = body.cfg or {}
    exp, _ = _load_experiment(body.exp_id)
    if exp is None:
        raise HTTPException(404, "No experiment found.")
    segments = (exp.get("result") or {}).get("segments") or []
    if not segments:
        segments = _segments_coverage(exp)[0]
    sc = build_scorecard(cfg or exp.get("config") or {}, body.exp_id, segments)
    return {"scorecard": sc,
            "weights": map_weights(sc, segments) if sc else {}}


@router.get("/charts")
def charts(exp_id: str | None = None):
    exp, _ = _load_experiment(exp_id)
    if exp is None:
        raise HTTPException(404, "No experiment found.")
    segments, coverage = _segments_coverage(exp)
    if not segments:
        return {"segments": [], "charts": None}
    cfg = exp.get("config") or {}
    if not isinstance(cfg, dict):
        cfg = {}
    scorecard = None
    sp = os.path.join(ARTIFACTS_DIR, exp.get("exp_id", ""), "scorecard.json")
    if os.path.exists(sp):
        try:
            with open(sp, "r", encoding="utf-8") as fh:
                scorecard = json.load(fh)
        except Exception:
            pass
    feature_groups = cfg.get("feature_groups") or {}
    return {
        "segments": segments,
        "charts": {
            "scatter": fig_scatter(segments),
            "distribution": fig_distribution(segments, cfg.get("target_col", "")),
            "sunburst": fig_sunburst(segments),
            "decile": fig_decile(scorecard),
            "feature_importance": fig_feature_importance(segments, feature_groups),
        },
    }


# ── Diagnostics ───────────────────────────────────────────────────────────────
@router.post("/feature-journey")
def feature_journey(body: JourneyBody):
    exp, _ = _load_experiment(body.exp_id)
    if exp is None:
        raise HTTPException(404, "No experiment found.")
    cfg = body.cfg or exp.get("config") or {}
    if not is_loaded() or not os.path.exists(active_db()):
        raise HTTPException(400, "Dataset not available — load it in Module 1 to enable diagnostics.")
    b = build_diag_builder(cfg, exp.get("exp_id"))
    if b is None:
        raise HTTPException(400, "Could not build diagnostics (no dataset / persisted diagnostics).")
    import contextlib
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            b.explain_feature_journey(body.feature)
    except Exception as exc:
        raise HTTPException(400, str(exc))
    return {"text": buf.getvalue() or "(no journey recorded for this feature)"}


@router.post("/feature-health")
def feature_health(body: HealthBody):
    exp, _ = _load_experiment(body.exp_id)
    if exp is None:
        raise HTTPException(404, "No experiment found.")
    cfg = body.cfg or exp.get("config") or {}
    if not is_loaded() or not os.path.exists(active_db()):
        raise HTTPException(400, "Dataset not available — load it in Module 1 first.")
    type_overrides = (load_state().get("type_overrides")) or {}
    try:
        hr = generate_feature_health_local(
            active_db(), body.features, cfg.get("target_col", ""),
            type_overrides, int(cfg.get("naive_bins", 5) or 5),
        )
    except Exception as exc:
        raise HTTPException(500, f"Health report failed: {exc}")
    return {"rows": hr.to_dict(orient="records"),
            "csv": hr.to_csv(index=False)}


@router.post("/explain-no-segments")
def explain_no_segments(body: NoSegBody):
    exp, _ = _load_experiment(body.exp_id)
    if exp is None:
        raise HTTPException(404, "No experiment found.")
    cfg = body.cfg or exp.get("config") or {}
    if not is_loaded() or not os.path.exists(active_db()):
        raise HTTPException(400, "Dataset not available — reload in Module 1 to enable the full diagnostic.")
    b = build_diag_builder(cfg, exp.get("exp_id"))
    if b is None:
        raise HTTPException(400, "Could not build diagnostics.")
    return {"text": b.explain_no_segments()}


# ── Exports ───────────────────────────────────────────────────────────────────
def _html_report(exp, res, segments, scorecard):
    name = exp.get("name", "Experiment")
    rows = "".join(
        f"<tr><td>{s['segment_id']}</td><td>{s.get('rule_string', '')}</td>"
        f"<td>{s.get('count', 0):,}</td><td>{float(s.get('rate', 0)):.2f}%</td>"
        f"<td>{float(s.get('lift', 0)):.2f}x</td></tr>"
        for s in segments
    )
    sc = scorecard or {}
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>RapidSegment Report - {name}</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:32px;color:#1f2937}}
table{{border-collapse:collapse;width:100%;margin-top:12px}}
th,td{{border:1px solid #d1d5db;padding:6px 10px;text-align:left;font-size:13px}}
h1{{color:#4338ca}} h2{{color:#3730a3;margin-top:28px}}
.kpi{{display:flex;gap:16px;flex-wrap:wrap}} .kpi div{{background:#eef2ff;padding:12px 18px;border-radius:10px}}
</style></head><body>
<h1>RapidSegment - Results Report</h1>
<p><b>Experiment:</b> {name} . <b>ID:</b> {exp.get('exp_id', '')} . <b>Status:</b> {exp.get('status', '')}</p>
<div class="kpi">
<div><b>Segments</b><br>{res.get('segments_count', 0)}</div>
<div><b>Coverage %</b><br>{res.get('coverage_pct', 0):.2f}</div>
<div><b>Avg lift</b><br>{res.get('avg_lift', 0):.2f}x</div>
<div><b>Max lift</b><br>{res.get('max_lift', 0):.2f}x</div>
<div><b>Baseline rate</b><br>{res.get('baseline_rate_pct', 0):.2f}%</div>
</div>
<h2>Segments</h2>
<table><tr><th>ID</th><th>Rule</th><th>Count</th><th>Event Rate</th><th>Lift</th></tr>
{rows}</table>
<h2>Scorecard</h2>
<pre>{json.dumps(sc, indent=2)}</pre>
</body></html>"""


def _export_payload(exp_id):
    exp, _ = _load_experiment(exp_id)
    if exp is None:
        raise HTTPException(404, "No experiment found.")
    segments, coverage = _segments_coverage(exp)
    cfg = exp.get("config") or {}
    if not isinstance(cfg, dict):
        cfg = {}
    scorecard = None
    sp = os.path.join(ARTIFACTS_DIR, exp.get("exp_id", ""), "scorecard.json")
    if os.path.exists(sp):
        try:
            with open(sp, "r", encoding="utf-8") as fh:
                scorecard = json.load(fh)
        except Exception:
            pass
    res = exp.get("result") or {}
    return exp, res, segments, coverage, cfg, scorecard


@router.get("/export/{exp_id}/segments.csv")
def export_segments(exp_id: str):
    _, _, segments, _, _, _ = _export_payload(exp_id)
    import pandas as pd
    csv = pd.DataFrame(segments).to_csv(index=False) if segments else ""
    return Response(csv, media_type="text/csv")


@router.get("/export/{exp_id}/coverage.csv")
def export_coverage(exp_id: str):
    _, _, _, coverage, _, _ = _export_payload(exp_id)
    import pandas as pd
    csv = pd.DataFrame(coverage).to_csv(index=False) if coverage else ""
    return Response(csv, media_type="text/csv")


@router.get("/export/{exp_id}/config.json")
def export_config(exp_id: str):
    exp, _, _, _, cfg, _ = _export_payload(exp_id)
    return json.dumps(cfg, indent=2)


@router.get("/export/{exp_id}/sql.sql")
def export_sql(exp_id: str):
    exp, _, segments, coverage, cfg, _ = _export_payload(exp_id)
    return PlainTextResponse(build_sql_script(segments, coverage, cfg=cfg, exp=exp))


@router.get("/export/{exp_id}/report.html")
def export_report(exp_id: str):
    exp, res, segments, _, _, scorecard = _export_payload(exp_id)
    return Response(_html_report(exp, res, segments, scorecard), media_type="text/html")


@router.get("/export/{exp_id}/scorecard.json")
def export_scorecard(exp_id: str):
    exp, _, _, _, _, scorecard = _export_payload(exp_id)
    if scorecard is None:
        raise HTTPException(404, "Scorecard not available.")
    return json.dumps(scorecard, indent=2)


@router.get("/export/{exp_id}/zip")
def export_zip(exp_id: str):
    exp, res, segments, coverage, cfg, scorecard = _export_payload(exp_id)
    import pandas as pd
    segs_csv = pd.DataFrame(segments).to_csv(index=False) if segments else ""
    cov_csv = pd.DataFrame(coverage).to_csv(index=False) if coverage else ""
    sql_script = build_sql_script(segments, coverage, cfg=cfg, exp=exp)
    html_report = _html_report(exp, res, segments, scorecard)
    sc_json = json.dumps(scorecard, indent=2) if scorecard is not None else None
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("segments.csv", segs_csv)
        zf.writestr("coverage.csv", cov_csv)
        zf.writestr("config.json", json.dumps(cfg, indent=2))
        zf.writestr("segments.sql", sql_script)
        zf.writestr("report.html", html_report)
        if sc_json is not None:
            zf.writestr("scorecard.json", sc_json)
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="rapidsegment_{exp_id}.zip"'})
