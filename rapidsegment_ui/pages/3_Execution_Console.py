"""
RapidSegment — Module 3: Real-Time Execution & Artifact Console
================================================================
Live execution console for StrategicSegmentBuilder experiments handed
off from Module 2 (the Workbench), with a 6-step status timeline, live
KPI metrics, a split-pane log / SQL console, cancel & interrupt support
and an export hub (Logs.txt / SQL.sql / Config.json).

Consumes Module 2 (workbench) output:
    - st.session_state["pending_run"]    dict      validated builder config

Produces (same contract as Module 2):
    - st.session_state["experiment"]     dict      {exp_id, name, created_at, status,
                                                   execution_time_sec, target_col,
                                                   primary_key, data_rows, data_cols,
                                                   config (builder params JSON),
                                                   result (segments + coverage summary),
                                                   logs (captured terminal stream)}
    - st.session_state["last_config"]    dict      last experiment config (clone support)

Files touched:
    read  .rapidsegment_suite/module1_data.duckdb   (udl_data)
    r/w   .rapidsegment_suite/suite_data.db         (experiments table)
    write .rapidsegment_suite/artifacts/<exp_id>/   (workbench.duckdb, tmp/,
                                                    logs.txt, config.json, sql.sql,
                                                    result.json)

Run with:  streamlit run Module_3_execution.py
"""
import html
import json
import logging
import os
import queue
import threading
import time
import uuid
from datetime import datetime

import duckdb
import pandas as pd
import streamlit as st

from rapidsegment import StrategicSegmentBuilder

# ── Constants & storage ───────────────────────────────────────────────────────
SUITE_DIR = os.path.join(os.getcwd(), ".rapidsegment_suite")
os.makedirs(SUITE_DIR, exist_ok=True)
DB_FILE = os.path.join(SUITE_DIR, "module1_data.duckdb")
SUITE_DB = os.path.join(SUITE_DIR, "suite_data.db")

REFRESH_SECONDS = 2.0  # live-metrics refresh cadence (2–5 s per spec)

PHASE_NAMES = [
    "Configure & load data",
    "Feature ranking (IV / response rate)",
    "Candidate rule generation",
    "Binning & rule complexity",
    "Residual extraction (per segment)",
    "Final coverage",
]

LEVEL_RANK = {"INFO": 10, "WARNING": 30, "ERROR": 40}
LEVEL_COLORS = {"INFO": "#c9d1d9", "WARNING": "#f0b429", "ERROR": "#f85149",
                "DEBUG": "#8b949e"}


# ── Small helpers (Module 2 conventions) ─────────────────────────────────────
def rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


def db_query(sql, read_only=True):
    con = duckdb.connect(DB_FILE, read_only=read_only)
    result = con.execute(sql).df()
    con.close()
    return result


def db_scalar(sql):
    con = duckdb.connect(DB_FILE, read_only=True)
    result = con.execute(sql).fetchone()[0]
    con.close()
    return result


def card():
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()


def _jsonable(obj):
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


# ── Log capture (Python logging piped into Streamlit state) ──────────────────
def _ui_log(run, level, msg):
    run["logs"].append({
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "level": level,
        "msg": msg,
        "src": "ui",
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
        # Attach to the engine logger only — child loggers (e.g.
        # "StrategicEngine.DataLoader") propagate into it, so attaching to the
        # root as well would duplicate every record.
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


def _logs_txt(records):
    return "\n".join(
        f"{r.get('ts', '')} | {str(r.get('level', 'INFO')):<7} | {r.get('msg', '')}"
        for r in records
    )


def _filter_logs(records, level):
    if level == "All":
        return records
    min_rank = LEVEL_RANK.get(str(level).upper(), 10)
    return [
        r for r in records
        if LEVEL_RANK.get(str(r.get("level", "INFO")).upper(), 10) >= min_rank
    ]


def _terminal_html(records, height=400):
    parts = []
    for r in records:
        color = LEVEL_COLORS.get(r.get("level", "INFO"), "#c9d1d9")
        parts.append(
            f'<span style="color:#6e7681;">{html.escape(str(r.get("ts", "")))}</span> '
            f'<span style="color:{color};font-weight:600;">'
            f'[{html.escape(str(r.get("level", "")))}]</span> '
            f'<span style="color:#c9d1d9;">{html.escape(str(r.get("msg", "")))}</span>'
        )
    body = "\n".join(parts) if parts else (
        '<span style="color:#6e7681;">— no log output yet —</span>'
    )
    return (
        f'<div style="background:#0d1117;color:#c9d1d9;font-family:Consolas,'
        f'\'Courier New\',monospace;font-size:12px;line-height:1.5;'
        f'padding:10px 12px;border:1px solid #30363d;border-radius:8px;'
        f'height:{height}px;overflow-y:auto;white-space:pre-wrap;'
        f'word-break:break-all;">{body}</div>'
    )


# ── Status timeline (6 extraction steps) ─────────────────────────────────────
def _log_to_step(line):
    """Map captured library log lines onto the 6-step timeline."""
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


def _step_states(step, status):
    if status == "completed":
        return ["done"] * 6
    if status in ("cancelled", "failed"):
        return [
            "done" if i < step else ("error" if i == step else "pending")
            for i in range(1, 7)
        ]
    return [
        "done" if i < step else ("active" if i == step else "pending")
        for i in range(1, 7)
    ]


_PILL_BORDER = {"done": "#22c55e", "active": "#3b82f6",
                "pending": "#4b5563", "error": "#ef4444"}
_PILL_BG = {"done": "rgba(34,197,94,0.14)", "active": "rgba(59,130,246,0.16)",
            "pending": "rgba(75,85,99,0.10)", "error": "rgba(239,68,68,0.14)"}
_PILL_ICON = {"done": "✅", "active": "⏳", "pending": "⏳", "error": "❌"}


def _pill_html(idx, name, state):
    border, bg = _PILL_BORDER[state], _PILL_BG[state]
    return (
        f'<div style="border:1px solid {border};background:{bg};border-radius:8px;'
        f'padding:8px 4px;text-align:center;height:100%;">'
        f'<div style="font-size:16px;line-height:1.2;">{_PILL_ICON[state]}</div>'
        f'<div style="font-size:10px;color:#9aa4b2;margin-top:2px;">Step {idx}</div>'
        f'<div style="font-size:11px;color:#e6edf3;font-weight:600;margin-top:2px;">'
        f'{html.escape(name)}</div></div>'
    )


def _current_feature(run):
    """Best live guess of the feature being processed (diagnostics / usage)."""
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


# ── Coverage + SQL generation (local, in-memory — no library evaluate_* call) ─
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


def compute_coverage_local(segments, df, target):
    """Replicates StrategicSegmentBuilder.evaluate_final_coverage with a fresh,
    in-memory DuckDB connection (the library method is NOT called — it hangs on
    the shared db_path file lock in this environment)."""
    if not segments:
        return []
    con = duckdb.connect()
    try:
        con.execute("SET threads = 4;")
        con.register("input_data_view", df)
        return con.execute(_build_coverage_sql(segments, target)).df().to_dict(orient="records")
    finally:
        con.close()


def _build_sql_script(segments, coverage, cfg=None, exp=None):
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


# ── Persistence (same contract as Module 2) ──────────────────────────────────
def upsert_experiment(exp):
    con = duckdb.connect(SUITE_DB)
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
            error_msg TEXT
        )
        """
    )
    con.execute(
        """
        INSERT OR REPLACE INTO experiments VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, CAST(? AS JSON), ?, ?, ?, ?, ?, ?
        )
        """,
        [
            exp["exp_id"], exp["name"], exp["created_at"],
            exp["data_rows"], exp["data_cols"], exp["status"],
            exp["execution_time_sec"], exp["target_col"], exp["primary_key"],
            json.dumps(exp["config"]),
            exp["result"]["segments_count"], exp["result"]["avg_lift"],
            exp["result"]["max_lift"], exp["result"]["coverage_pct"],
            exp["result"]["baseline_rate_pct"], exp["result"].get("error_msg"),
        ],
    )
    con.close()


def _build_experiment(run):
    cfg = run["cfg"]
    segments = run.get("segments") or []
    coverage = run.get("coverage") or []
    lifts = [float(s.get("lift") or 0) for s in segments if s.get("lift") is not None]
    cov_pct = sum(
        float(r.get("capture_rate") or 0) for r in coverage if r.get("segment") != 0
    )
    baseline = float(coverage[0].get("base_response_rate") or 0) if coverage else None
    if baseline is None:
        baseline = float((st.session_state.get("tinfo") or {}).get("event_rate") or 0) * 100
    result = {
        "segments": segments,
        "coverage": coverage,
        "stop_reason": run.get("stop_reason"),
        "segments_count": len(segments),
        "avg_lift": round(sum(lifts) / len(lifts), 4) if lifts else 0.0,
        "max_lift": round(max(lifts), 4) if lifts else 0.0,
        "coverage_pct": round(cov_pct, 3),
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
        "execution_time_sec": run["elapsed"],
        "target_col": cfg.get("target_col", ""),
        "primary_key": cfg.get("primary_key", ""),
        "data_rows": run.get("n_rows", 0),
        "data_cols": run.get("n_cols", 0),
        "config": _jsonable(cfg),
        "result": _jsonable(result),
        "logs": run.get("logs", []),
    }


def _write_artifacts(run, exp):
    try:
        os.makedirs(run["exp_dir"], exist_ok=True)
        with open(os.path.join(run["exp_dir"], "logs.txt"), "w", encoding="utf-8") as fh:
            fh.write(_logs_txt(run.get("logs") or []))
        with open(os.path.join(run["exp_dir"], "config.json"), "w", encoding="utf-8") as fh:
            json.dump(run["cfg"], fh, indent=2)
        with open(os.path.join(run["exp_dir"], "sql.sql"), "w", encoding="utf-8") as fh:
            fh.write(_build_sql_script(
                exp["result"].get("segments") or [],
                exp["result"].get("coverage") or [],
                cfg=exp["config"], exp=exp,
            ))
        with open(os.path.join(run["exp_dir"], "result.json"), "w", encoding="utf-8") as fh:
            json.dump(exp, fh, indent=2, default=str)
    except Exception as exc:
        _ui_log(run, "WARNING", f"Could not write artifact files: {exc}")


def _finalize(run, status):
    """Graceful termination: snapshot partial state, persist and detach."""
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
                    run["coverage"] = compute_coverage_local(
                        run["segments"], run["df"], run["cfg"]["target_col"])
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
        st.session_state["experiment"] = exp
        st.session_state["last_config"] = _jsonable(run["cfg"])
        try:
            upsert_experiment(exp)
        except Exception:
            _ui_log(run, "WARNING", "Could not persist to suite_data.db.")
        _write_artifacts(run, exp)
    except Exception as exc:
        _ui_log(run, "ERROR", f"Persistence failed: {exc}")


# ── Worker threads (daemon — UI thread only polls) ───────────────────────────
def _run_extract(run):
    try:
        run["out_q"].put(("phase", 1))
        segments = run["builder"].extract_segments(run["df"])
        run["out_q"].put(("extracted", segments, run["builder"].stop_reason))
    except Exception as exc:
        run["out_q"].put(("error", str(exc)))


def _run_coverage(run):
    try:
        coverage = compute_coverage_local(
            run["segments"], run["df"], run["cfg"]["target_col"])
        run["out_q"].put(("done", coverage))
    except Exception as exc:
        run["out_q"].put(("error", f"Coverage computation failed: {exc}"))


def _start_run(cfg):
    if not cfg.get("target_col"):
        st.error("Config has no target_col — re-run from the Workbench (Module 2).")
        return None
    try:
        df = db_query("SELECT * FROM udl_data")
    except Exception as exc:
        st.error(f"Cannot load dataset from `{DB_FILE}`: {exc}\n\n"
                 "Re-run Module 1 (Data Loader) / Module 2 (Workbench) first.")
        return None
    if df.empty:
        st.error("Dataset is empty — re-run Module 1 with a non-empty file.")
        return None
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
        db_path=os.path.join(exp_dir, "workbench.duckdb"),
        db_temp_dir=os.path.join(exp_dir, "tmp"),
    )
    run = {
        "exp_id": exp_id,
        "exp_dir": exp_dir,
        "t0": time.time(),
        "cfg": cfg,
        "df": df,
        "n_rows": len(df),
        "n_cols": df.shape[1],
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
    _ui_log(run, "INFO", f"Data: {len(df):,} rows × {df.shape[1]} cols · target=`{cfg['target_col']}`")
    threading.Thread(target=_run_extract, args=(run,), daemon=True).start()
    st.session_state["m3_run"] = run
    return run


def _monitor_tick(run):
    """Drain log + worker queues, update timeline, start coverage thread."""
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


# ── Rendering ────────────────────────────────────────────────────────────────
def _coverage_pct(run):
    if run.get("coverage_done") and run.get("coverage"):
        return sum(
            float(r.get("capture_rate") or 0)
            for r in run["coverage"] if r.get("segment") != 0
        )
    total = sum(float(s.get("count") or 0) for s in (run.get("live_segments") or []))
    return min(100.0, total / max(run.get("n_rows", 1), 1) * 100)


def _render_console(run, live=False):
    segs = list(run.get("live_segments") or []) if live else list(run.get("segments") or [])
    status = None if live else run.get("status")
    st.markdown("### Status Timeline")
    states = _step_states(run.get("step", 1), status)
    cols = st.columns(6)
    for i, (col, name) in enumerate(zip(cols, PHASE_NAMES)):
        with col:
            st.markdown(_pill_html(i + 1, name, states[i]), unsafe_allow_html=True)
    if live:
        st.caption(
            f"⏱ Elapsed: **{fmt_duration(time.time() - run['t0'])}** · "
            f"Found **{len(segs)}** segment(s) so far… · "
            f"Current feature: **{_current_feature(run)}**"
        )
    else:
        st.caption(
            f"⏱ Elapsed: **{fmt_duration(run.get('elapsed', 0))}** · "
            f"Segments found: **{len(segs)}** · "
            f"Current feature: **{_current_feature(run)}**"
        )
    lifts = [float(s.get("lift") or 0) for s in segs if s.get("lift") is not None]
    best = max(segs, key=lambda s: float(s.get("lift") or 0)) if segs else None
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Segments found", f"{len(segs)}")
    k2.metric("Total coverage %", f"{_coverage_pct(run):.1f}%")
    k3.metric("Average lift", f"{sum(lifts) / len(lifts):.2f}×" if lifts else "—")
    k4.metric("Best segment (lift)", f"{float(best['lift']):.2f}×" if best else "—",
              delta=(str(best["rule_string"])[:42] + "…") if best else None)
    if live:
        st.caption("Coverage % is a live estimate (residual counts ÷ total rows) until the "
                   "final coverage pass completes.")
    st.markdown("**Top candidates**")
    with card():
        if segs:
            for s in segs[:3]:
                st.caption(f"· `{s['rule_string']}` — {s.get('count', 0):,} rows · "
                           f"lift {float(s.get('lift') or 0):.2f}×")
        else:
            st.caption("No segments yet — extraction in progress…")

    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown("#### Log Terminal")
        lvl = st.radio(
            "Level filter", ["All", "Info", "Warning", "Error"], index=0,
            horizontal=True, key=f"m3_log_lvl_{run['exp_id']}",
        )
        st.download_button(
            "⧉ Copy Logs (.txt)",
            _logs_txt(run.get("logs") or []).encode("utf-8"),
            file_name=f"logs_{run['exp_id']}.txt", mime="text/plain",
            key=f"m3_copy_{run['exp_id']}",
        )
        st.markdown(
            _terminal_html(_filter_logs(run.get("logs") or [], lvl)),
            unsafe_allow_html=True,
        )
    with right_col:
        st.markdown("#### SQL Inspector")
        if segs:
            for s in segs:
                st.caption(f"Segment {s['segment_id']} · `{s['rule_string']}`")
                st.code(s.get("sql_filter") or "", language="sql")
                st.download_button(
                    "Copy SQL",
                    (s.get("sql_filter") or "").encode("utf-8"),
                    file_name=f"segment_{s['segment_id']}.sql", mime="text/plain",
                    key=f"m3_sql_{run['exp_id']}_{s['segment_id']}",
                    use_container_width=True,
                )
        else:
            st.caption("Waiting for segments…")


def _render_live(run):
    h1, h2 = st.columns([4, 1])
    with h1:
        st.subheader(f"🚀 {run['cfg'].get('experiment_name', 'Experiment')}")
        st.caption(f"`{run['exp_id']}` · target=`{run['cfg']['target_col']}` · "
                   f"{run['n_rows']:,} rows × {run['n_cols']} cols")
    with h2:
        if st.button(
            "⛔ Cancel Extraction", type="primary", use_container_width=True,
            key=f"m3_cancel_{run['exp_id']}",
        ):
            run["cancel_requested"] = True
    _render_console(run, live=True)


def _render_export_hub(run, exp):
    st.divider()
    st.subheader("Export Hub")
    res = exp.get("result") or {}
    logs_txt = _logs_txt(run.get("logs") or [])
    sql_script = _build_sql_script(
        res.get("segments") or [], res.get("coverage") or [],
        cfg=exp.get("config") or {}, exp=exp,
    )
    cfg_json = json.dumps(exp.get("config") or {}, indent=2)
    e1, e2, e3 = st.columns(3)
    e1.download_button(
        "⬇️ Logs.txt", logs_txt.encode("utf-8"),
        file_name=f"logs_{exp['exp_id']}.txt", mime="text/plain",
        key=f"dl_logs_{exp['exp_id']}", use_container_width=True,
    )
    e2.download_button(
        "⬇️ SQL.sql", sql_script.encode("utf-8"),
        file_name=f"segments_{exp['exp_id']}.sql", mime="text/plain",
        key=f"dl_sql_{exp['exp_id']}", use_container_width=True,
    )
    e3.download_button(
        "⬇️ Config.json", cfg_json.encode("utf-8"),
        file_name=f"config_{exp['exp_id']}.json", mime="application/json",
        key=f"dl_cfg_{exp['exp_id']}", use_container_width=True,
    )


def _render_results(exp):
    st.divider()
    st.subheader("Results Summary")
    res = exp.get("result") or {}
    segments = res.get("segments") or []
    coverage = res.get("coverage") or []
    st.caption(f"stop_reason: {res.get('stop_reason') or '—'}")
    m = st.columns(6)
    m[0].metric("Segments", res.get("segments_count", len(segments)))
    m[1].metric("Coverage %", f"{res.get('coverage_pct', 0):.2f}")
    m[2].metric("Avg lift", f"{res.get('avg_lift', 0):.2f}×")
    m[3].metric("Max lift", f"{res.get('max_lift', 0):.2f}×")
    m[4].metric("Baseline rate", f"{res.get('baseline_rate_pct', 0):.2f}%")
    m[5].metric("Elapsed", fmt_duration(exp.get("execution_time_sec", 0)))
    if segments:
        seg_cols = ["segment_id", "rule_string", "sql_filter", "count", "rate",
                    "lift", "meta_applied_sample_size", "meta_applied_min_lift"]
        seg_df = pd.DataFrame(segments)
        st.markdown("**Segments**")
        st.dataframe(
            seg_df[[c for c in seg_cols if c in seg_df.columns]],
            height=300, use_container_width=True, hide_index=True,
        )
    if coverage:
        st.markdown("**Final coverage (events vs. non-events)**")
        st.dataframe(pd.DataFrame(coverage), height=300,
                     use_container_width=True, hide_index=True)
    else:
        st.caption("No coverage rows — experiment produced no segments.")


def _view_run(exp):
    res = exp.get("result") or {}
    return {
        "exp_id": exp.get("exp_id", "view"),
        "cfg": exp.get("config") or {},
        "logs": exp.get("logs") or [],
        "segments": res.get("segments") or [],
        "live_segments": res.get("segments") or [],
        "coverage": res.get("coverage") or [],
        "status": exp.get("status", "completed"),
        "elapsed": exp.get("execution_time_sec", 0),
        "n_rows": exp.get("data_rows", 0),
        "n_cols": exp.get("data_cols", 0),
        "step": res.get("step_reached") or 6,
        "builder": None,
    }


def _render_view(exp):
    st.subheader(f"📦 {exp.get('name', 'Experiment')}")
    st.caption(
        f"`{exp.get('exp_id', '')}` · {exp.get('created_at', '')} · "
        f"status=`{exp.get('status')}` · target=`{exp.get('target_col')}` · "
        f"{exp.get('data_rows', '?')} rows"
    )
    status = exp.get("status")
    if status == "cancelled":
        st.warning("⚠️ This experiment was cancelled — partial results shown below.")
    elif status == "failed":
        st.error(f"❌ This experiment failed: "
                 f"{(exp.get('result') or {}).get('error_msg') or 'unknown error'}")
    run = _view_run(exp)
    _render_console(run, live=False)
    _render_export_hub(run, exp)
    _render_results(exp)
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        lc = st.session_state.get("last_config")
        if st.button(
            "♻️ Re-run last config", use_container_width=True,
            disabled=not lc, key=f"m3_rerun_{exp.get('exp_id', 'view')}",
        ):
            if lc:
                st.session_state["pending_run"] = dict(lc)
                rerun()
        if not lc:
            st.caption("No stored config to re-run — start from the Workbench.")
    with c2:
        try:
            st.page_link("pages/2_Workbench.py",
                         label="Configure new experiment in Workbench (Module 2)", icon="⚙️")
        except Exception:
            st.caption("Configure a new experiment in the Workbench (Module 2).")


def _render_final(run):
    status = run["status"]
    if status == "cancelled":
        st.warning("⚠️ Extraction cancelled — partial results saved below.")
    elif status == "failed":
        st.error(f"❌ Experiment failed: {run.get('error_msg') or 'unknown error'}")
    _render_console(run, live=False)
    exp = run.get("experiment") or {}
    if exp:
        _render_export_hub(run, exp)
        _render_results(exp)


# ── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="RapidSegment — Execution Console", layout="wide")

st.markdown(
    """
    <style>
    div.stButton > button[kind="primary"] {
        background-color: #d92d20; border-color: #d92d20; color: #ffffff;
        font-weight: 600;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #b42318; border-color: #b42318; color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("RapidSegment — Module 3: Execution & Artifact Console")
st.caption("Real-time extraction monitoring, log / SQL console, cancel & export hub.")

# ── Entry modes ──────────────────────────────────────────────────────────────
pending_run = st.session_state.pop("pending_run", None)
run = st.session_state.get("m3_run")

if isinstance(pending_run, dict):
    st.session_state.pop("m3_run", None)
    run = _start_run(pending_run)
    if run is None:
        st.stop()

if run is not None and not run.get("finalized"):
    _monitor_tick(run)
    if not run.get("finalized"):
        _render_live(run)
        if run.get("cancel_requested") and not run.get("finalized"):
            _finalize(run, "cancelled")
    if run.get("finalized"):
        st.session_state.pop("m3_run", None)
        _render_final(run)
    else:
        time.sleep(REFRESH_SECONDS)
        rerun()
else:
    exp = st.session_state.get("experiment")
    if exp:
        _render_view(exp)
    else:
        st.info(
            "No experiment pending. Configure one in the **Workbench (Module 2)** "
            "and press **Run Experiment**."
        )
        try:
            st.page_link("pages/2_Workbench.py", label="Go to Workbench", icon="⚙️")
        except Exception:
            st.caption("Navigate to the Workbench (Module 2) from the sidebar.")