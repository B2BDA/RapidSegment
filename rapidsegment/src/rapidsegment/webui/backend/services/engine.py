"""Module 3 run manager: background extraction with live log/step polling.

The Streamlit implementation advances a run by re-rendering the page every 2s
(``_monitor_tick``). FastAPI is stateless, so instead each run gets its own
background *monitor thread* that drains the log/worker queues, updates the 6-step
timeline, kicks off the coverage pass and finalizes the experiment — the UI only
polls ``/status``.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import uuid
from datetime import datetime

import duckdb

from rapidsegment import StrategicSegmentBuilder

from config import normalize_cfg
from storage import (
    SUITE_DIR, active_db, db_scalar, jsonable, table_cols, upsert_experiment,
)

PHASE_NAMES = [
    "Configure & load data",
    "Feature ranking (IV / response rate)",
    "Candidate rule generation",
    "Binning & rule complexity",
    "Residual extraction (per segment)",
    "Final coverage",
]

LEVEL_RANK = {"INFO": 10, "WARNING": 30, "ERROR": 40}


def _ui_log(run, level, msg):
    run["logs"].append({
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "level": level, "msg": msg, "src": "ui",
    })


def _attach_log_handler(run):
    class _QueueHandler(logging.Handler):
        def emit(self, record):
            try:
                run["log_q"].put({
                    "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "level": record.levelname,
                    "msg": record.getMessage(),
                    "src": "lib",
                })
            except Exception:
                pass

    handler = _QueueHandler(level=logging.INFO)
    try:
        logging.getLogger("StrategicEngine").setLevel(logging.INFO)
        logging.getLogger("StrategicEngine").addHandler(handler)
        run["handler"] = handler
    except Exception:
        run["handler"] = None


def _detach_log_handler(run):
    handler = run.get("handler")
    if handler is None:
        return
    try:
        logging.getLogger("StrategicEngine").removeHandler(handler)
    except Exception:
        pass
    run["handler"] = None


def logs_txt(records):
    return "\n".join(
        f"{r.get('ts', '')} | {str(r.get('level', 'INFO')):<7} | {r.get('msg', '')}"
        for r in records
    )


def _log_to_step(line):
    if "Evaluating final hierarchical coverage" in line:
        return 6
    if (
        "🏁 Extraction complete." in line
        or "✅ Segment" in line
        or "📌 Feature Usage Tracker" in line
        or "⏹️" in line
        or "All eligible features exhausted" in line
        or "No candidate" in line
        or "No features had valid binned" in line
        or "No valid binned variables" in line
    ):
        return 5
    if "🔄 Iteration" in line or "📊 Dynamic Grid Search Enabled" in line:
        return 4
    if "📦 Binning method" in line:
        return 3
    if (
        "🔍 Computing IV and bins" in line
        or "⚙️ DuckDB Configured" in line
        or "📊 Sort priority" in line
        or "🔒 Locking Original Base Rate" in line
    ):
        return 2
    if "🚀 Starting hierarchical" in line:
        return 1
    return None


def _build_coverage_sql(segments, target):
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
           SUM(CAST("{target}" AS DOUBLE)) AS target_events,
           (SUM(CAST("{target}" AS DOUBLE)) * 100.0 / COUNT(*)) AS response_rate
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


def build_sql_script(segments, coverage, cfg=None, exp=None):
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


def _current_feature(run):
    builder = run.get("builder")
    if builder is None:
        return "—"
    try:
        diag = builder.diagnostics_ or []
        if diag and diag[-1].get("winning_segment"):
            vars_used = diag[-1]["winning_segment"].get("variables_used") or []
            if vars_used:
                return ", ".join(vars_used)
        counts = builder.feature_usage_counts or {}
        used = [c for c in counts if counts[c] > 0]
        if used:
            return max(used, key=lambda c: counts[c])
    except Exception:
        pass
    return "—"


def _build_experiment(run):
    cfg = run["cfg"]
    segments = run.get("segments") or []
    coverage = run.get("coverage") or []
    cec = max(
        (float(r.get("cumulative_event_capture") or 0) for r in coverage if r.get("segment") != 0),
        default=0.0,
    )
    lifts = [float(s.get("lift") or 0) for s in segments if s.get("lift") is not None]
    cov_pct = sum(
        float(r.get("capture_rate") or 0) for r in coverage if r.get("segment") != 0
    )
    baseline = float(coverage[0].get("base_response_rate") or 0) if coverage else None
    if baseline is None:
        baseline = float((run.get("tinfo") or {}).get("event_rate") or 0) * 100
    result = {
        "segments": segments,
        "coverage": coverage,
        "stop_reason": run.get("stop_reason"),
        "segments_count": len(segments),
        "avg_lift": round(sum(lifts) / len(lifts), 4) if lifts else 0.0,
        "max_lift": round(max(lifts), 4) if lifts else 0.0,
        "coverage_pct": round(cov_pct, 3),
        "cumulative_event_capture": round(cec, 3),
        "baseline_rate_pct": round(baseline, 3),
        "step_reached": run.get("step", 1),
    }
    if run["status"] == "failed":
        result["error_msg"] = run.get("error_msg")
    return {
        "exp_id": run["exp_id"],
        "name": cfg.get("experiment_name", "Experiment"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": run["status"],
        "execution_time_sec": float(run.get("elapsed", 0) or 0),
        "target_col": cfg.get("target_col", ""),
        "primary_key": cfg.get("primary_key", ""),
        "dataset_name": run.get("dataset_name", ""),
        "data_rows": int(run.get("n_rows", 0) or 0),
        "data_cols": int(run.get("n_cols", 0) or 0),
        "config": jsonable(cfg),
        "result": jsonable(result),
        "logs": run.get("logs", []),
    }


def _write_artifacts(run, exp):
    try:
        try:
            b = run.get("builder")
            if b is not None:
                exp["result"]["diagnostics_"] = jsonable(getattr(b, "diagnostics_", []))
                exp["result"]["feature_usage_counts"] = jsonable(
                    getattr(b, "feature_usage_counts", {}))
        except Exception:
            pass
        os.makedirs(run["exp_dir"], exist_ok=True)
        with open(os.path.join(run["exp_dir"], "logs.txt"), "w", encoding="utf-8") as fh:
            fh.write(logs_txt(run.get("logs") or []))
        with open(os.path.join(run["exp_dir"], "config.json"), "w", encoding="utf-8") as fh:
            json.dump(run["cfg"], fh, indent=2)
        with open(os.path.join(run["exp_dir"], "sql.sql"), "w", encoding="utf-8") as fh:
            fh.write(build_sql_script(
                exp["result"].get("segments") or [],
                exp["result"].get("coverage") or [],
                cfg=exp["config"], exp=exp,
            ))
        with open(os.path.join(run["exp_dir"], "result.json"), "w", encoding="utf-8") as fh:
            json.dump(exp, fh, indent=2, default=str)
    except Exception as exc:
        _ui_log(run, "WARNING", f"Could not write artifact files: {exc}")


def _finalize(run, status):
    run["finalized"] = True
    run["status"] = status
    run["elapsed"] = round(time.time() - run["t0"], 2)
    try:
        if status == "cancelled":
            run["segments"] = list(run.get("live_segments") or [])
            run["stop_reason"] = "Cancelled by user"
            if run["segments"]:
                _ui_log(run, "INFO",
                        f"Cancelled — snapshotting {len(run['segments'])} partial segment(s).")
                try:
                    run["coverage"] = run["builder"].evaluate_final_coverage(run["data_path"])
                    run["coverage_done"] = True
                except Exception as exc:
                    run["coverage"] = []
                    _ui_log(run, "WARNING", f"Partial coverage computation failed: {exc}")
            else:
                run["coverage"] = []
                _ui_log(run, "WARNING", "Cancelled before any segment was found.")
        elif status == "failed":
            run["segments"] = []
            run["coverage"] = []
            _ui_log(run, "ERROR", f"Experiment failed: {run.get('error_msg') or 'unknown error'}")
        else:  # completed
            _ui_log(run, "INFO", f"Extraction complete — {len(run['segments'])} segment(s).")
    except Exception as exc:
        _ui_log(run, "ERROR", f"Finalise error: {exc}")
    _detach_log_handler(run)
    try:
        exp = _build_experiment(run)
        run["experiment"] = exp
        run["save_error"] = None
        try:
            upsert_experiment(exp)
        except Exception as exc:
            run["save_error"] = f"Persist failed: {exc}"
            _ui_log(run, "WARNING", f"Could not persist to suite_data.db: {exc}")
        _write_artifacts(run, exp)
    except Exception as exc:
        _ui_log(run, "ERROR", f"Persistence failed: {exc}")


# ── Worker threads ────────────────────────────────────────────────────────────
def _run_extract(run):
    try:
        run["out_q"].put(("phase", 1))
        segments = run["builder"].extract_segments(run["data_path"])
        run["out_q"].put(("extracted", segments, run["builder"].stop_reason))
    except Exception as exc:
        run["out_q"].put(("error", str(exc)))


def _run_coverage(run):
    try:
        coverage = run["builder"].evaluate_final_coverage(run["data_path"])
        run["out_q"].put(("done", coverage))
    except Exception as exc:
        run["out_q"].put(("error", f"Coverage computation failed: {exc}"))


def _monitor_tick(run):
    while True:
        try:
            rec = run["log_q"].get_nowait()
        except queue.Empty:
            break
        run["logs"].append(rec)
        step = _log_to_step(rec.get("msg") or "")
        if step:
            run["step"] = max(run["step"], step)
    while True:
        try:
            msg = run["out_q"].get_nowait()
        except queue.Empty:
            break
        kind = msg[0]
        if kind == "phase":
            run["step"] = max(run["step"], int(msg[1]))
        elif kind == "extracted":
            run["extract_done"] = True
            run["segments"] = list(msg[1])
            run["stop_reason"] = msg[2]
        elif kind == "done":
            run["coverage"] = msg[1]
            run["coverage_done"] = True
            run["step"] = max(run["step"], 6)
        elif kind == "error":
            run["error_msg"] = msg[1]
            run["failed"] = True
    try:
        run["live_segments"] = list(run["builder"].segments or [])
    except Exception:
        pass
    if run.get("extract_done") and run.get("segments") and not run.get("cov_started"):
        run["cov_started"] = True
        run["step"] = max(run["step"], 6)
        _ui_log(run, "INFO", "Extraction done — computing final coverage (in-memory DuckDB)…")
        threading.Thread(target=_run_coverage, args=(run,), daemon=True).start()
    if run.get("failed"):
        _finalize(run, "failed")
    elif run.get("extract_done") and (not run.get("segments") or run.get("coverage_done")):
        _finalize(run, "completed")


def _monitor_loop(run):
    while not run.get("finalized"):
        _monitor_tick(run)
        if run.get("cancel_requested") and not run.get("finalized"):
            _finalize(run, "cancelled")
            break
        if run.get("finalized"):
            break
        time.sleep(1.0)
    _monitor_tick(run)


def start_run(cfg, dataset_name="", event_rate=None, m1_state=None):
    target = cfg.get("target_col")
    if not target:
        raise ValueError("Config has no target_col — re-run from the Workbench (Module 2).")
    data_path = active_db()
    if not os.path.exists(data_path):
        raise ValueError("Cannot find dataset — re-run Module 1 (Data Loader) first.")
    try:
        n_rows = db_scalar(data_path, "SELECT COUNT(*) FROM udl_data")
        n_cols = len(table_cols(data_path))
    except Exception as exc:
        raise ValueError(f"Cannot load dataset from `{data_path}`: {exc}") from exc
    if not n_rows:
        raise ValueError("Dataset is empty — re-run Module 1 with a non-empty file.")

    exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    exp_dir = os.path.join(SUITE_DIR, "artifacts", exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "tmp"), exist_ok=True)
    ignore = list(cfg.get("ignore_features") or [])
    pk = cfg.get("primary_key") or ""
    if pk and pk not in ignore:
        ignore.append(pk)

    builder = StrategicSegmentBuilder(
        target=cfg["target_col"],
        n_jobs=cfg.get("n_jobs", -1),
        min_sample_size=cfg["min_sample_size"],
        min_lift=cfg["min_lift"],
        min_events=cfg["min_events"],
        top_n_vars=cfg["top_n_vars"],
        max_segments=cfg["max_segments"],
        max_feature_reuse=cfg["max_feature_reuse"],
        param_grid=cfg.get("param_grid"),
        enable_diversity=cfg["enable_diversity"],
        enable_1way=cfg["enable_1way"],
        enable_2way=cfg["enable_2way"],
        enable_3way=cfg["enable_3way"],
        feature_groups=cfg.get("feature_groups") or {},
        ignore_features=ignore,
        sort_priority=cfg.get("sort_priority", "rate_lift_count"),
        binning_method=cfg["binning_method"],
        naive_bins=cfg["naive_bins"],
        max_expansion_hops=cfg["max_expansion_hops"],
        selection_metric=cfg["selection_metric"],
        expand_log_mode=cfg.get("expand_log_mode", "none"),
        memory_limit_gb=cfg.get("memory_limit_gb"),
        engine_threads=cfg.get("engine_threads"),
        db_path=os.path.join(exp_dir, "workbench.duckdb"),
        db_temp_dir=os.path.join(exp_dir, "tmp"),
        persist_db=True,
    )

    run = {
        "exp_id": exp_id,
        "exp_dir": exp_dir,
        "t0": time.time(),
        "cfg": cfg,
        "data_path": data_path,
        "n_rows": int(n_rows or 0),
        "n_cols": int(n_cols or 0),
        "dataset_name": dataset_name,
        "tinfo": {"event_rate": event_rate} if event_rate else (m1_state or {}).get("tinfo") or {},
        "builder": builder,
        "out_q": queue.Queue(),
        "log_q": queue.Queue(),
        "logs": [],
        "step": 1,
        "segments": [],
        "live_segments": [],
        "coverage": [],
        "extract_done": False,
        "cov_started": False,
        "coverage_done": False,
        "stop_reason": None,
        "error_msg": None,
        "status": "running",
        "finalized": False,
        "elapsed": 0.0,
        "cancel_requested": False,
        "handler": None,
    }
    _attach_log_handler(run)
    _ui_log(run, "INFO", f"Experiment '{cfg.get('experiment_name', 'Experiment')}' started — exp_id={exp_id}")
    _ui_log(run, "INFO", f"Data: {n_rows:,} rows × {n_cols} cols · target=`{cfg['target_col']}`")
    threading.Thread(target=_run_extract, args=(run,), daemon=True).start()
    threading.Thread(target=_monitor_loop, args=(run,), daemon=True).start()
    return run


def _trim(segs):
    out = []
    for s in segs or []:
        if not isinstance(s, dict):
            continue
        out.append({
            "segment_id": s.get("segment_id"),
            "rule_string": s.get("rule_string", ""),
            "sql_filter": s.get("sql_filter", ""),
            "count": s.get("count", 0),
            "rate": s.get("rate"),
            "lift": s.get("lift"),
        })
    return out


def run_status(run, after=0):
    """Snapshot for the UI: state, step pills, metrics, and new log records."""
    segs = list(run.get("live_segments") or []) if not run.get("finalized") else list(run.get("segments") or [])
    lifts = [float(s.get("lift") or 0) for s in segs if s.get("lift") is not None]
    best = max(segs, key=lambda s: float(s.get("lift") or 0)) if segs else None
    return {
        "exp_id": run["exp_id"],
        "status": run.get("status", "running"),
        "finalized": run.get("finalized", False),
        "step": run.get("step", 1),
        "step_names": PHASE_NAMES,
        "elapsed": round(time.time() - run["t0"], 1) if not run.get("finalized") else run.get("elapsed", 0),
        "n_rows": run.get("n_rows", 0),
        "n_cols": run.get("n_cols", 0),
        "target_col": run.get("cfg", {}).get("target_col", ""),
        "experiment_name": run.get("cfg", {}).get("experiment_name", "Experiment"),
        "segments_found": len(segs),
        "coverage_pct": _coverage_pct(run),
        "avg_lift": round(sum(lifts) / len(lifts), 2) if lifts else None,
        "best_lift": round(float(best["lift"]), 2) if best else None,
        "best_rule": (str(best["rule_string"])[:42] + "…") if best else None,
        "current_feature": _current_feature(run),
        "stop_reason": run.get("stop_reason"),
        "error_msg": run.get("error_msg"),
        "cancel_requested": run.get("cancel_requested", False),
        "save_error": run.get("save_error"),
        "top_candidates": [
            {"rule_string": s["rule_string"], "count": s.get("count", 0),
             "lift": float(s.get("lift") or 0)}
            for s in segs[:3]
        ],
        "segments_preview": _trim(segs[:20]),
        "logs": run.get("logs", [])[after:],
        "log_count": len(run.get("logs", [])),
    }


def _coverage_pct(run):
    if run.get("coverage_done") and run.get("coverage"):
        return sum(
            float(r.get("capture_rate") or 0)
            for r in run["coverage"] if r.get("segment") != 0
        )
    total = sum(float(s.get("count") or 0) for s in (run.get("live_segments") or []))
    return min(100.0, total / max(run.get("n_rows", 1), 1) * 100)


# ── Manager (module-level singleton) ──────────────────────────────────────────
class RunManager:
    def __init__(self):
        self._runs = {}
        self._lock = threading.Lock()

    def submit(self, cfg, dataset_name="", event_rate=None, m1_state=None):
        with self._lock:
            run = start_run(normalize_cfg(cfg), dataset_name, event_rate, m1_state)
            self._runs[run["exp_id"]] = run
            return run["exp_id"]

    def get(self, run_id):
        with self._lock:
            return self._runs.get(run_id)

    def cancel(self, run_id):
        run = self.get(run_id)
        if run is None:
            return False
        run["cancel_requested"] = True
        return True

    def status(self, run_id, after=0):
        run = self.get(run_id)
        if run is None:
            return None
        return run_status(run, after)

    def latest(self):
        with self._lock:
            for run in reversed(list(self._runs.values())):
                if run.get("experiment"):
                    return run["experiment"]
        return None

    def latest_run(self):
        with self._lock:
            runs = list(self._runs.values())
        if not runs:
            return None
        return runs[-1]


run_manager = RunManager()
