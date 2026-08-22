"""
RapidSegment — Module 2: The Workbench (Enhanced)
=================================================
Streamlit workbench for configuring all StrategicSegmentBuilder parameters
with smart defaults, interactive validation, parameter presets, grid search
and experiment execution.

Consumes Module 1 (data loader) output:
    - st.session_state["loaded"]        bool      data present in module1_data.duckdb
    - st.session_state["target_col"]    str       validated target column
    - st.session_state["tinfo"]         dict      profiling info incl. event_rate

Hands off to Module 3 (execution console) — the Run button no longer
executes inline; it validates the config, then:
    - st.session_state["pending_run"]    dict      validated config consumed by
                                                  pages/3_Execution_Console.py
                                                  (Module 3), which then writes
    - st.session_state["experiment"]     dict      {exp_id, name, created_at, status,
                                                  execution_time_sec, target_col,
                                                  primary_key, data_rows, data_cols,
                                                  config (builder params JSON),
                                                  result (segments + coverage summary)}
    - st.session_state["last_config"]   dict      last experiment config (clone support)
`run_builder`, `compute_coverage_local` and `upsert_experiment` are kept for
reference / reuse — execution now happens in Module 3.

Files touched:
    read  .rapidsegment_suite/module1_data.duckdb   (udl_data)
    r/w   .rapidsegment_suite/templates.json
    r/w   .rapidsegment_suite/suite_data.db         (experiments table)
    write .rapidsegment_suite/artifacts/<exp_id>/   (builder DuckDB + temp dir)

Run with:  streamlit run Module_2_workbench.py
"""
import json
import os
import queue
import threading
import time
import uuid
from datetime import datetime
import pandas as pd
import duckdb
import streamlit as st

from rapidsegment import StrategicSegmentBuilder

# ── Constants & storage ───────────────────────────────────────────────────────
SUITE_DIR = os.path.join(os.getcwd(), ".rapidsegment_suite")
os.makedirs(SUITE_DIR, exist_ok=True)
DB_FILE = os.path.join(SUITE_DIR, "module1_data.duckdb")
SUITE_DB = os.path.join(SUITE_DIR, "suite_data.db")
TEMPLATES_FILE = os.path.join(SUITE_DIR, "templates.json")

BIN_OPTIONS = ["Optimal (CART)", "Optimal (Quantile)", "Naive"]
BIN_MAP = {
    "Optimal (CART)": "optimal_cart",
    "Optimal (Quantile)": "optimal_quantile",
    "Naive": "naive",
}
BIN_RMAP = {v: k for k, v in BIN_MAP.items()}
METRIC_OPTIONS = ["IV", "Response Rate"]
METRIC_MAP = {"IV": "iv", "Response Rate": "response_rate"}
METRIC_RMAP = {v: k for k, v in METRIC_MAP.items()}

SORT_PRIORITY_OPTIONS = [
    ("rate_lift_count", "Rate → Lift → Count (default)"),
    ("lift_rate_count", "Lift → Rate → Count"),
    ("lift_count_rate", "Lift → Count → Rate"),
    ("count_lift_rate", "Count → Lift → Rate"),
    ("count_rate_lift", "Count → Rate → Lift"),
    ("rate_count_lift", "Rate → Count → Lift"),
    ("events_lift_rate", "Events → Lift → Rate"),
    ("events_rate_lift", "Events → Rate → Lift"),
    ("lift_events_rate", "Lift → Events → Rate"),
    ("rate_events_lift", "Rate → Events → Lift"),
    ("events_count_rate", "Events → Count → Rate"),
    ("events_rate_count", "Events → Rate → Count"),
    ("count_events_rate", "Count → Events → Rate"),
    ("rate_events_count", "Rate → Events → Count"),
]
SORT_PRIORITY_MAP = dict(SORT_PRIORITY_OPTIONS)
SORT_PRIORITY_RMAP = {v: k for k, v in SORT_PRIORITY_OPTIONS}
SORT_PRIORITY_HELP = (
    "Ranking strategy for champion selection — sorted descending, so the first "
    "dimension dominates. E.g. 'Rate → Lift → Count' prefers the highest response "
    "rate, then lift, then volume. All 14 combinations of rate / lift / count / "
    "events are supported by the library."
)

MAX_JOBS = max(1, os.cpu_count() or 4)
N_JOBS_OPTIONS = ["-1 (all but one core)"] + [str(i) for i in range(1, MAX_JOBS + 1)]
N_JOBS_MAP = {opt: -1 if opt.startswith("-1") else int(opt) for opt in N_JOBS_OPTIONS}
N_JOBS_RMAP = {v: k for k, v in N_JOBS_MAP.items()}

EXPAND_LOG_OPTIONS = ["none", "summary", "champion", "full"]

GRID_SIZE_OPTIONS = [250, 500, 750, 1000, 1500, 2000, 2500, 3000, 5000]
GRID_LIFT_OPTIONS = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0]

QUICK_DISCOVERY = {
    "experiment_name": "Quick Discovery",
    "description": "Aggressive discovery: fast naive binning, wide search.",
    "top_n_vars": 20,
    "max_segments": 15,
    "max_feature_reuse": 2,
    "enable_diversity": False,
    "ignore_features": [],
    "binning_method": "naive",
    "naive_bins": 5,
    "max_expansion_hops": 1,
    "enable_1way": True,
    "enable_2way": True,
    "enable_3way": True,
    "selection_metric": "iv",
    "min_sample_size": 500,
    "min_lift": 1.2,
    "min_events": 50,
    "param_grid": None,
    "sort_priority": "rate_lift_count",
    "n_jobs": -1,
    "expand_log_mode": "none",
}

CONSERVATIVE = {
    "experiment_name": "Conservative",
    "description": "Strict constraints, stable optimal quantile binning.",
    "top_n_vars": 10,
    "max_segments": 5,
    "max_feature_reuse": 1,
    "enable_diversity": False,
    "ignore_features": [],
    "binning_method": "optimal_quantile",
    "naive_bins": 5,
    "max_expansion_hops": 0,
    "enable_1way": True,
    "enable_2way": True,
    "enable_3way": False,
    "selection_metric": "response_rate",
    "min_sample_size": 5000,
    "min_lift": 2.0,
    "min_events": 500,
    "param_grid": None,
    "sort_priority": "rate_lift_count",
    "n_jobs": -1,
    "expand_log_mode": "none",
}


# ── Small helpers ─────────────────────────────────────────────────────────────
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


def toggle(label, key, help=None):
    try:
        return st.toggle(label, key=key, help=help)
    except AttributeError:
        return st.checkbox(label, key=key, help=help)


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


def grid_combos(cfg):
    pg = cfg.get("param_grid") or {}
    if not pg:
        return 1
    sizes = len(pg.get("min_sample_size") or [1])
    lifts = len(pg.get("min_lift") or [1])
    return max(1, sizes) * max(1, lifts)


# ── Config build / validation / estimation ───────────────────────────────────
def build_params():
    bm = BIN_MAP.get(st.session_state["wb_binning_method"], "optimal_cart")
    sm = METRIC_MAP.get(st.session_state["wb_selection_metric"], "iv")
    pk = st.session_state["wb_primary_key"]
    grid = None
    if st.session_state["wb_enable_grid"]:
        sizes = list(st.session_state.get("wb_grid_sizes") or [])
        lifts = list(st.session_state.get("wb_grid_lifts") or [])
        if sizes or lifts:
            grid = {
                "min_sample_size": sizes or [st.session_state["wb_min_sample_size"]],
                "min_lift": lifts or [st.session_state["wb_min_lift"]],
            }
    groups = st.session_state.get("wb_groups") or []
    default_name = f"exp_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    return {
        "experiment_name": str(st.session_state["wb_experiment_name"]).strip() or default_name,
        "description": str(st.session_state["wb_description"]),
        "data_table": str(st.session_state["wb_data_table"]),
        "target_col": str(st.session_state["wb_target_col"]),
        "primary_key": "" if pk == "(none)" else str(pk),
        "top_n_vars": int(st.session_state["wb_top_n_vars"]),
        "max_segments": int(st.session_state["wb_max_segments"]),
        "max_feature_reuse": int(st.session_state["wb_max_feature_reuse"]),
        "feature_groups": {
            g["name"]: list(g.get("cols") or []) for g in groups if g.get("name")
        },
        "enable_diversity": bool(st.session_state["wb_enable_diversity"]),
        "ignore_features": list(st.session_state.get("wb_ignore_features") or []),
        "binning_method": bm,
        "naive_bins": int(st.session_state["wb_naive_bins"]),
        "max_expansion_hops": int(st.session_state["wb_max_expansion_hops"]),
        "enable_1way": bool(st.session_state["wb_enable_1way"]),
        "enable_2way": bool(st.session_state["wb_enable_2way"]),
        "enable_3way": bool(st.session_state["wb_enable_3way"]),
        "selection_metric": sm,
        "min_sample_size": int(st.session_state["wb_min_sample_size"]),
        "min_lift": float(st.session_state["wb_min_lift"]),
        "min_events": int(st.session_state["wb_min_events"]),
        "param_grid": grid,
        "sort_priority": SORT_PRIORITY_RMAP[st.session_state["wb_sort_priority"]],
        "n_jobs": N_JOBS_MAP[st.session_state["wb_n_jobs"]],
        "expand_log_mode": st.session_state["wb_expand_log_mode"],
    }


def validate_params(cfg, all_cols):
    issues = []
    if cfg["target_col"] not in all_cols:
        issues.append(f"Target column '{cfg['target_col']}' is not in the dataset.")
    if cfg["primary_key"] and cfg["primary_key"] not in all_cols:
        issues.append(f"Primary key column '{cfg['primary_key']}' is not in the dataset.")
    if not (cfg["enable_1way"] or cfg["enable_2way"] or cfg["enable_3way"]):
        issues.append("Enable at least one rule type (1-way / 2-way / 3-way).")
    if cfg["min_events"] > cfg["min_sample_size"]:
        issues.append(
            f"min_events ({cfg['min_events']}) exceeds min_sample_size "
            f"({cfg['min_sample_size']}) — no rule could ever pass."
        )
    if cfg["target_col"] in cfg["ignore_features"]:
        issues.append("Target column cannot be listed under ignore features.")
    for group, feats in cfg["feature_groups"].items():
        if cfg["target_col"] in feats:
            issues.append(f"Target column cannot be inside feature group '{group}'.")
        for feat in feats:
            if feat not in all_cols:
                issues.append(f"Feature '{feat}' in group '{group}' is not in the dataset.")
    if cfg["param_grid"]:
        if not cfg["param_grid"].get("min_sample_size") and not cfg["param_grid"].get("min_lift"):
            issues.append("Grid search needs at least one min_sample_size or min_lift value.")
    if cfg["binning_method"] == "naive" and cfg["naive_bins"] < 3:
        issues.append("Naive binning needs at least 3 bins.")
    return issues


def estimate_seconds(cfg, n_rows):
    pg = cfg.get("param_grid") or {}
    combos = max(1, len(pg.get("min_sample_size") or [1])) * max(1, len(pg.get("min_lift") or [1]))
    base = (max(n_rows, 1000) / 100_000.0) * 6.0
    base *= 0.45 if cfg["binning_method"] == "naive" else 2.0
    base *= 1.0 + 0.35 * cfg["max_expansion_hops"]
    if cfg["enable_2way"]:
        base *= 1.4
    if cfg["enable_3way"]:
        base *= 2.1
    base *= (cfg["max_segments"] / 10.0) * max(1.0, cfg["top_n_vars"] / 15.0)
    return max(15.0, base * combos)


# ── Templates / leaderboard persistence ──────────────────────────────────────
def load_templates():
    if not os.path.exists(TEMPLATES_FILE):
        return {}
    try:
        with open(TEMPLATES_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_template(name, cfg):
    templates = load_templates()
    templates[name] = cfg
    with open(TEMPLATES_FILE, "w", encoding="utf-8") as fh:
        json.dump(templates, fh, indent=2)


def cfg_from_json(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return {}
    return {}


def read_leaderboard():
    if not os.path.exists(SUITE_DB):
        return None
    try:
        con = duckdb.connect(SUITE_DB, read_only=True)
        has = con.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name='experiments'"
        ).fetchone()
        if not has:
            con.close()
            return None
        rows = con.execute(
            "SELECT exp_id, name, created_at, builder_params "
            "FROM experiments ORDER BY created_at DESC"
        ).fetchall()
        con.close()
        return list(rows)
    except Exception:
        return None


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


def clear_group_keys():
    for key in list(st.session_state):
        if key.startswith("wb_group_cols_") or key.startswith("wb_group_rm_"):
            del st.session_state[key]


def apply_config(cfg):
    mapping = {
        "experiment_name": "wb_experiment_name",
        "description": "wb_description",
        "data_table": "wb_data_table",
        "target_col": "wb_target_col",
        "primary_key": "wb_primary_key",
        "top_n_vars": "wb_top_n_vars",
        "max_segments": "wb_max_segments",
        "max_feature_reuse": "wb_max_feature_reuse",
        "enable_diversity": "wb_enable_diversity",
        "ignore_features": "wb_ignore_features",
        "binning_method": "wb_binning_method",
        "naive_bins": "wb_naive_bins",
        "max_expansion_hops": "wb_max_expansion_hops",
        "enable_1way": "wb_enable_1way",
        "enable_2way": "wb_enable_2way",
        "enable_3way": "wb_enable_3way",
        "selection_metric": "wb_selection_metric",
        "min_sample_size": "wb_min_sample_size",
        "min_lift": "wb_min_lift",
        "min_events": "wb_min_events",
        "sort_priority": "wb_sort_priority",
        "n_jobs": "wb_n_jobs",
        "expand_log_mode": "wb_expand_log_mode",
    }
    for cfg_key, widget_key in mapping.items():
        if cfg_key not in cfg or cfg[cfg_key] is None or cfg[cfg_key] == "":
            continue
        value = cfg[cfg_key]
        if cfg_key == "binning_method":
            value = BIN_RMAP.get(value, value)
        if cfg_key == "selection_metric":
            value = METRIC_RMAP.get(value, value)
        if cfg_key == "sort_priority":
            value = SORT_PRIORITY_MAP.get(value, value)
        if cfg_key == "n_jobs":
            value = N_JOBS_RMAP.get(value, "-1 (all but one core)")
        st.session_state[widget_key] = value
    fg = cfg.get("feature_groups")
    if isinstance(fg, dict):
        st.session_state["wb_groups"] = [
            {"name": name, "cols": list(cols)} for name, cols in fg.items()
        ]
        clear_group_keys()
    pg = cfg.get("param_grid")
    if pg:
        st.session_state["wb_enable_grid"] = True
        if pg.get("min_sample_size"):
            st.session_state["wb_grid_sizes"] = list(pg["min_sample_size"])
        if pg.get("min_lift"):
            st.session_state["wb_grid_lifts"] = list(pg["min_lift"])
    else:
        st.session_state["wb_enable_grid"] = False


# ── Experiment runner (threaded, live phase + elapsed time) ───────────────────
def compute_coverage_local(segments, df, target):
    """Replicates StrategicSegmentBuilder.evaluate_final_coverage with a fresh,
    locally-tuned DuckDB connection (used as timeout fallback)."""
    con = duckdb.connect()
    try:
        con.execute("SET threads = 4;")
        con.register("input_data_view", df)
        case_sql = "\n".join(
            f"WHEN {seg['sql_filter']} THEN {seg['segment_id']}" for seg in segments
        )
        query = f"""
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
        return con.execute(query).df().to_dict(orient="records")
    finally:
        con.close()


def run_builder(cfg, df):
    ignore = list(cfg["ignore_features"])
    if cfg["primary_key"] and cfg["primary_key"] not in ignore:
        ignore.append(cfg["primary_key"])
    exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    exp_dir = os.path.join(SUITE_DIR, "artifacts", exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    builder = StrategicSegmentBuilder(
        target=cfg["target_col"],
        n_jobs=cfg.get("n_jobs", -1),
        min_sample_size=cfg["min_sample_size"],
        min_lift=cfg["min_lift"],
        min_events=cfg["min_events"],
        top_n_vars=cfg["top_n_vars"],
        max_segments=cfg["max_segments"],
        max_feature_reuse=cfg["max_feature_reuse"],
        param_grid=cfg["param_grid"],
        enable_diversity=cfg["enable_diversity"],
        enable_1way=cfg["enable_1way"],
        enable_2way=cfg["enable_2way"],
        enable_3way=cfg["enable_3way"],
        feature_groups=cfg["feature_groups"],
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
    out = queue.Queue()

    def _extract():
        try:
            out.put(("phase", "Extracting segments…"))
            segments = builder.extract_segments(df)
            out.put(("extracted", segments, builder.stop_reason))
        except Exception as exc:
            out.put(("error", str(exc)))

    def _coverage(segments):
        try:
            out.put(("done", compute_coverage_local(segments, df, cfg["target_col"])))
        except Exception as exc:
            out.put(("error", str(exc)))

    threading.Thread(target=_extract, daemon=True).start()
    t0 = time.time()
    segments, coverage, stop_reason, err = [], [], None, None
    phase_text = "Running experiment…"
    try:
        with st.status("Running experiment…", expanded=True) as status:
            while True:
                try:
                    msg = out.get(timeout=0.5)
                except queue.Empty:
                    status.update(
                        label=f"{phase_text} — elapsed {fmt_duration(time.time() - t0)}"
                    )
                    continue
                kind = msg[0]
                if kind == "phase":
                    phase_text = msg[1]
                    status.update(label=f"{phase_text} — elapsed {fmt_duration(time.time() - t0)}")
                elif kind == "extracted":
                    segments, stop_reason = msg[1], msg[2]
                    if not segments:
                        break
                    phase_text = "Extraction done — computing final coverage…"
                    threading.Thread(target=_coverage, args=(segments,), daemon=True).start()
                elif kind == "done":
                    coverage = msg[1]
                    break
                elif kind == "error":
                    err = msg[1]
                    break
            if err:
                status.update(label="Extraction failed", state="error", expanded=True)
            else:
                status.update(
                    label=(
                        f"Extraction complete — {len(segments)} segment(s)"
                        + (" · coverage computed locally" if segments else "")
                    ),
                    state="complete",
                    expanded=False,
                )
    except AttributeError:
        with st.spinner("Running experiment…"):
            segments = builder.extract_segments(df)
            coverage = compute_coverage_local(segments, df, cfg["target_col"]) if segments else []
            stop_reason = builder.stop_reason
    if err:
        return None, [], None, time.time() - t0, err, exp_id
    return segments, coverage, stop_reason, time.time() - t0, None, exp_id


# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="RapidSegment — Workbench", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] .block-container { padding-bottom: 9rem; }
    [data-testid="stVerticalBlockBorderWrapper"]:last-of-type {
        position: fixed; left: 0; right: 0; bottom: 0; z-index: 99;
        background: rgba(13, 16, 23, 0.96);
        border-top: 1px solid rgba(255, 255, 255, 0.12);
        padding: 0.5rem 1.2rem 0.3rem;
    }
    div.stButton > button[kind="primary"] {
        background-color: #000000; border-color: #000000; color: #ffffff;
        font-weight: 600;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #26262b; border-color: #26262b; color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session init & deferred actions ──────────────────────────────────────────
if "wb_groups" not in st.session_state:
    st.session_state["wb_groups"] = []

pending = st.session_state.pop("wb_pending", None)
if isinstance(pending, dict):
    apply_config(pending)
    rerun()

# ── Guard: data must be loaded by Module 1 ───────────────────────────────────
if not st.session_state.get("loaded"):
    if os.path.exists(DB_FILE):
        try:
            if db_scalar("SELECT COUNT(*) FROM udl_data") > 0:
                st.session_state["loaded"] = True
        except Exception:
            pass
    if not st.session_state.get("loaded"):
        st.warning(
            "No dataset loaded yet. Run **Module 1 (Data Loader)** first, "
            "validate a binary target column, then come back to the Workbench."
        )
        st.stop()

all_cols = db_query(
    "SELECT column_name FROM information_schema.columns "
    "WHERE table_name='udl_data' ORDER BY ordinal_position"
)["column_name"].tolist()
n_rows = db_scalar("SELECT COUNT(*) FROM udl_data")
n_cols = len(all_cols)
tinfo = st.session_state.get("tinfo")

preset_target = st.session_state.get("target_col") or (all_cols[0] if all_cols else "")
if preset_target not in all_cols:
    preset_target = all_cols[0] if all_cols else ""

defaults = {
    "wb_experiment_name": f"exp_{datetime.now().strftime('%Y-%m-%d_%H-%M')}",
    "wb_description": "",
    "wb_data_table": "udl_data",
    "wb_target_col": preset_target,
    "wb_primary_key": "(none)",
    "wb_top_n_vars": 15,
    "wb_max_segments": 10,
    "wb_max_feature_reuse": 1,
    "wb_enable_diversity": False,
    "wb_ignore_features": [],
    "wb_binning_method": "Optimal (CART)",
    "wb_naive_bins": 5,
    "wb_max_expansion_hops": 0,
    "wb_enable_1way": True,
    "wb_enable_2way": True,
    "wb_enable_3way": True,
    "wb_selection_metric": "IV",
    "wb_min_sample_size": 1000,
    "wb_min_lift": 1.5,
    "wb_min_events": 100,
    "wb_sort_priority": SORT_PRIORITY_MAP["rate_lift_count"],
    "wb_n_jobs": "-1 (all but one core)",
    "wb_expand_log_mode": "none",
    "wb_enable_grid": False,
    "wb_grid_sizes": [500, 1000, 2000],
    "wb_grid_lifts": [1.5, 2.0, 3.0],
    "wb_template_name": "",
    "wb_preset": "Quick Discovery",
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

if st.session_state["wb_target_col"] not in all_cols:
    st.session_state["wb_target_col"] = preset_target
if st.session_state["wb_primary_key"] not in ["(none)"] + all_cols:
    st.session_state["wb_primary_key"] = "(none)"
valid_sizes = [v for v in st.session_state["wb_grid_sizes"] if v in GRID_SIZE_OPTIONS]
st.session_state["wb_grid_sizes"] = valid_sizes or [500, 1000, 2000]
valid_lifts = [v for v in st.session_state["wb_grid_lifts"] if v in GRID_LIFT_OPTIONS]
st.session_state["wb_grid_lifts"] = valid_lifts or [1.5, 2.0, 3.0]

st.title("RapidSegment — Module 2: The Workbench")
st.caption(
    f"Dataset: **{st.session_state['wb_data_table']}** · "
    f"{n_rows:,} rows · {n_cols} columns"
)

# ── Two-column layout (right column first so presets can re-fill widgets) ─────
left, right = st.columns([3, 1.55], gap="medium")

with right:
    notice = st.empty()

    saved_templates = load_templates()
    preset_names = ["Quick Discovery", "Conservative", "Last experiment"] + list(saved_templates.keys())
    if st.session_state.get("wb_preset") not in preset_names:
        st.session_state["wb_preset"] = "Quick Discovery"
    preset = st.selectbox("Preset templates", preset_names, key="wb_preset")
    if st.button("Apply Preset", use_container_width=True):
        if preset == "Quick Discovery":
            preset_cfg = dict(QUICK_DISCOVERY)
        elif preset == "Conservative":
            preset_cfg = dict(CONSERVATIVE)
        elif preset == "Last experiment":
            previous = st.session_state.get("experiment")
            preset_cfg = (previous or {}).get("config") if previous else None
            if preset_cfg is None:
                leaderboard = read_leaderboard()
                preset_cfg = cfg_from_json(leaderboard[0][3]) if leaderboard else None
            if preset_cfg is None:
                notice.error("No previous experiment found to clone.")
                preset_cfg = False
        else:
            preset_cfg = saved_templates.get(preset)
        if preset_cfg:
            apply_config(preset_cfg)
            rerun()

    st.markdown("#### Real-Time Summary")
    cfg_now = build_params()
    with card():
        st.markdown("📊 **Segment Discovery**")
        st.caption(
            f"top_n_vars={cfg_now['top_n_vars']} · "
            f"max_segments={cfg_now['max_segments']} · "
            f"max_feature_reuse={cfg_now['max_feature_reuse']}"
        )
        st.caption(
            f"diversity={'on' if cfg_now['enable_diversity'] else 'off'} · "
            f"{len(cfg_now['feature_groups'])} group(s) · "
            f"{len(cfg_now['ignore_features'])} ignored"
        )
    with card():
        st.markdown("🔢 **Rule Complexity**")
        bin_label = BIN_RMAP.get(cfg_now["binning_method"], cfg_now["binning_method"])
        bin_extra = (
            f" · {cfg_now['naive_bins']} bins" if cfg_now["binning_method"] == "naive" else ""
        )
        st.caption(f"binning={bin_label}{bin_extra}")
        st.caption(
            f"hops={cfg_now['max_expansion_hops']} · "
            f"1-way={'on' if cfg_now['enable_1way'] else 'off'} · "
            f"2-way={'on' if cfg_now['enable_2way'] else 'off'} · "
            f"3-way={'on' if cfg_now['enable_3way'] else 'off'} · "
            f"metric={cfg_now['selection_metric']}"
        )
    with card():
        st.markdown("⚙️ **Constraints**")
        st.caption(
            f"min_sample_size={cfg_now['min_sample_size']:,} · "
            f"min_lift={cfg_now['min_lift']} · "
            f"min_events={cfg_now['min_events']}"
        )
        if cfg_now["param_grid"]:
            st.caption(
                f"Grid: {len(cfg_now['param_grid']['min_sample_size'])} sizes × "
                f"{len(cfg_now['param_grid']['min_lift'])} lifts = "
                f"{grid_combos(cfg_now)} combinations"
            )

    st.markdown("#### Validation Checklist")
    sel_target = st.session_state["wb_target_col"]
    issues = validate_params(cfg_now, all_cols)
    target_ok = bool(
        tinfo and tinfo.get("is_binary") and tinfo.get("col") == sel_target
    )
    event_rate = tinfo.get("event_rate") if tinfo and tinfo.get("col") == sel_target else None
    imbalance = event_rate is not None and (event_rate < 0.01 or event_rate > 0.99)
    grid_on = bool(cfg_now["param_grid"])
    items = [
        (
            "Target column selected",
            target_ok,
            f"`{sel_target}` — " + ("validated as binary in Module 1" if target_ok else "not validated as binary"),
        ),
        (
            "Data loaded",
            True,
            f"{n_rows:,} rows · {n_cols} columns",
        ),
        (
            "Class imbalance detected",
            not imbalance,
            f"Event rate {event_rate:.2%}" if event_rate is not None else "No validated event rate",
        ),
        (
            "Parameters valid",
            not issues,
            issues[0] if issues else "All checks passed",
        ),
        (
            "Grid search time estimate",
            False,
            f"Evaluating {grid_combos(cfg_now)} combination(s)"
            if grid_on
            else "Grid search disabled",
        ),
    ]
    for label, ok, note in items:
        mark = "✅" if ok else "⚠️"
        st.markdown(f"{mark} **{label}**")
        st.caption(note)

with left:
    with st.expander("Basic Settings", expanded=True):
        st.text_input("Experiment Name", key="wb_experiment_name")
        st.text_area("Description (optional)", key="wb_description", height=80)
        b1, b2, b3 = st.columns(3)
        b1.text_input("Data table name", key="wb_data_table")
        b2.selectbox("Target column", all_cols, key="wb_target_col")
        b3.selectbox(
            "Primary key column (for scorecard)",
            ["(none)"] + all_cols,
            key="wb_primary_key",
        )

    with st.expander("Segment Discovery Strategy", expanded=True):
        d1, d2, d3 = st.columns(3)
        d1.slider("top_n_vars", 5, 50, key="wb_top_n_vars")
        d2.slider("max_segments", 1, 20, key="wb_max_segments")
        d3.slider("max_feature_reuse", 1, 5, key="wb_max_feature_reuse")

        s1, s2 = st.columns(2)
        s1.selectbox(
            "Sort priority (champion ranking)",
            list(SORT_PRIORITY_MAP.values()),
            key="wb_sort_priority",
            help=SORT_PRIORITY_HELP,
        )
        s2.selectbox(
            "Parallel jobs",
            N_JOBS_OPTIONS,
            key="wb_n_jobs",
            help="-1 uses all but one CPU core for IV computation.",
        )

        with st.expander("Feature grouping (business categories)"):
            gname = st.text_input(
                "Business category name",
                key="wb_new_group",
                placeholder="e.g. Delinquency",
            )
            if st.button("Add category", use_container_width=True):
                name = gname.strip()
                groups = st.session_state["wb_groups"]
                if not name or any(g["name"] == name for g in groups):
                    st.session_state["wb_group_msg"] = (
                        "Category name is empty or already exists."
                    )
                else:
                    groups.append({"name": name, "cols": []})
                    st.session_state["wb_groups"] = groups
                    st.session_state.pop("wb_group_msg", None)
                    clear_group_keys()
                    rerun()
            if st.session_state.get("wb_group_msg"):
                st.warning(st.session_state["wb_group_msg"])
            st.caption("Diversity toggle prevents rules that mix features from the same group.")
            for gi, group in enumerate(st.session_state["wb_groups"]):
                with st.expander(group["name"]):
                    picked = st.multiselect(
                        "Columns",
                        all_cols,
                        default=group.get("cols") or [],
                        key=f"wb_group_cols_{gi}",
                    )
                    group["cols"] = list(picked)
                    if st.button(
                        f"Remove category '{group['name']}'",
                        key=f"wb_group_rm_{gi}",
                        use_container_width=True,
                    ):
                        st.session_state["wb_groups"].pop(gi)
                        clear_group_keys()
                        rerun()
        st.multiselect(
            "Ignore features",
            all_cols,
            key="wb_ignore_features",
            help="Columns excluded before IV calculation.",
        )
        toggle(
            "Enable diversity (prevent mixing groups in one rule)",
            "wb_enable_diversity",
        )

    with st.expander("Binning & Rule Complexity", expanded=True):
        st.radio(
            "Binning method",
            BIN_OPTIONS,
            key="wb_binning_method",
            help="'Optimal (CART)' and 'Optimal (Quantile)' use OptBinning (slower, more predictive); "
            "'Naive' uses fast DuckDB quantiles.",
        )
        c1, c2 = st.columns(2)
        if st.session_state["wb_binning_method"] == "Naive":
            c1.slider("Naive bins", 3, 20, key="wb_naive_bins")
        else:
            c1.caption("Naive bins only apply to the 'Naive' binning method.")
        c2.slider("Max expansion hops", 0, 5, key="wb_max_expansion_hops")
        r1, r2, r3 = st.columns(3)
        toggle("Enable 1-way rules", "wb_enable_1way")
        toggle("Enable 2-way rules", "wb_enable_2way")
        toggle("Enable 3-way rules", "wb_enable_3way")
        st.selectbox(
            "Selection metric",
            METRIC_OPTIONS,
            key="wb_selection_metric",
            help="Metric used to rank features for top_n_vars selection.",
        )
        st.selectbox(
            "Expansion log mode",
            EXPAND_LOG_OPTIONS,
            key="wb_expand_log_mode",
            help="Verbosity of adjacent-bin expansion logging: 'none' is quiet; "
            "'champion'/'full' print ranked champion-vs-expanded comparisons in the run log.",
        )

    with st.expander("Hard Constraints", expanded=True):
        h1, h2, h3 = st.columns(3)
        h1.number_input(
            "min_sample_size", min_value=100, max_value=10_000_000, step=100,
            key="wb_min_sample_size",
        )
        h2.number_input(
            "min_lift", min_value=0.5, max_value=20.0, step=0.1, format="%.2f",
            key="wb_min_lift",
            help="Values below 1.0 are accepted — a rule with lift < 1 sits below "
            "the base response rate and will only rank if later dimensions dominate.",
        )
        h3.number_input(
            "min_events", min_value=1, max_value=1_000_000, step=10,
            key="wb_min_events",
        )

    with st.expander("Advanced: Grid Search (Optional)", expanded=True):
        toggle("Enable grid search", "wb_enable_grid")
        if st.session_state["wb_enable_grid"]:
            g1, g2 = st.columns(2)
            g1.multiselect(
                "min_sample_size values",
                GRID_SIZE_OPTIONS,
                key="wb_grid_sizes",
            )
            g2.multiselect(
                "min_lift values",
                GRID_LIFT_OPTIONS,
                key="wb_grid_lifts",
            )
            sizes = st.session_state.get("wb_grid_sizes") or []
            lifts = st.session_state.get("wb_grid_lifts") or []
            combos = max(1, len(sizes)) * max(1, len(lifts))
            st.caption(f"Evaluating **{combos}** combination(s).")
        else:
            st.caption("Grid search disabled — single (min_sample_size, min_lift) pair will be used.")

# ── Sticky action bar ─────────────────────────────────────────────────────────
try:
    footer = st.container(border=True)
except TypeError:
    footer = st.container()

with footer:
    f1, f2, f3 = st.columns([1.35, 1.7, 1.9], gap="medium")
    with f1:
        st.text_input("Template name", key="wb_template_name", placeholder="e.g. FastBinning")
        if st.button("Save as Template", use_container_width=True):
            tname = str(st.session_state["wb_template_name"]).strip()
            if not tname:
                tname = str(st.session_state["wb_experiment_name"]).strip()
            if tname:
                save_template(tname, build_params())
                notice.success(f"Template '{tname}' saved to templates.json")
            else:
                notice.error("Enter a template name first.")

    with f2:
        leaderboard = read_leaderboard()
        if leaderboard is None:
            st.selectbox(
                "Clone from Leaderboard",
                ["(no experiments yet)"],
                disabled=True,
            )
            st.caption("No `suite_data.db` with an experiments table found — run an experiment first.")
        elif not leaderboard:
            st.selectbox(
                "Clone from Leaderboard",
                ["(no experiments yet)"],
                disabled=True,
            )
            st.caption("Leaderboard is empty — run an experiment first.")
        else:
            lb_names = [f"{row[1]} · {str(row[2])[:16]}" for row in leaderboard]
            if st.session_state.get("wb_lb_exp") not in lb_names:
                st.session_state["wb_lb_exp"] = lb_names[0]
            st.selectbox("Clone from Leaderboard", lb_names, key="wb_lb_exp")
            if st.button("Apply Clone", use_container_width=True):
                row = leaderboard[lb_names.index(st.session_state["wb_lb_exp"])]
                cloned_cfg = cfg_from_json(row[3])
                if cloned_cfg:
                    st.session_state["wb_pending"] = cloned_cfg
                    rerun()
                else:
                    notice.error("Selected experiment has no stored parameters.")

    with f3:
        cfg_now = build_params()
        if st.button("Run Experiment", type="primary", use_container_width=True):
            issues = validate_params(cfg_now, all_cols)
            if issues:
                notice.error("Validation failed:\n" + "\n".join(f"• {item}" for item in issues))
            else:
                st.session_state["pending_run"] = _jsonable(cfg_now)
                try:
                    st.switch_page("pages/3_Execution_Console.py")
                except AttributeError:
                    st.experimental_switch_page("pages/3_Execution_Console.py")
                except Exception:
                    notice.success(
                        "Configuration saved — open Module 3 (Execution Console) to run it."
                    )
        estimated = estimate_seconds(cfg_now, n_rows)
        st.caption(f"Estimated time: **{fmt_duration(estimated)}**")

# ── Latest experiment results (preview for Module 3) ─────────────────────────
exp = st.session_state.get("experiment")
if exp:
    st.divider()
    st.subheader("Latest Experiment Results")
    res = exp.get("result") or {}
    st.caption(
        f"`{exp['name']}` · {exp.get('created_at', '')} · "
        f"{exp['execution_time_sec']:.1f}s · "
        f"target=`{exp['target_col']}` · "
        f"stop_reason={res.get('stop_reason') or '—'}"
    )

    segments = res.get("segments") or []
    coverage = res.get("coverage") or []
    if segments:
        seg_cols = ["segment_id", "rule_string", "sql_filter", "count", "rate", "lift",
                    "meta_applied_sample_size", "meta_applied_min_lift"]
        seg_df = pd.DataFrame(segments)
        st.markdown("**Segments**")
        st.dataframe(
            seg_df[[c for c in seg_cols if c in seg_df.columns]],
            height=300, use_container_width=True, hide_index=True,
        )
    if coverage:
        cov_df = pd.DataFrame(coverage)
        st.markdown("**Final coverage (events vs. non-events)**")
        st.dataframe(cov_df, height=300, use_container_width=True, hide_index=True)
    else:
        st.caption("No coverage rows — experiment produced no segments.")