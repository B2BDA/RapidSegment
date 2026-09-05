"""Module 2 (Workbench) configuration helpers: presets, option maps, validation,
estimation and template persistence. Mirrors ``rapidsegment/ui/pages/2_Workbench.py``.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

from storage import TEMPLATES_FILE, read_leaderboard_raw

# ── Option maps ───────────────────────────────────────────────────────────────
BIN_MAP = {"Optimal (CART)": "optimal_cart", "Optimal (Quantile)": "optimal_quantile", "Naive": "naive"}
BIN_RMAP = {v: k for k, v in BIN_MAP.items()}
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

VALID_SORT_PRIORITIES = set(SORT_PRIORITY_MAP)

MAX_JOBS = max(1, os.cpu_count() or 4)
N_JOBS_OPTIONS = ["-1 (all but one core)"] + [str(i) for i in range(1, MAX_JOBS + 1)]
N_JOBS_MAP = {opt: -1 if opt.startswith("-1") else int(opt) for opt in N_JOBS_OPTIONS}
N_JOBS_RMAP = {v: k for k, v in N_JOBS_MAP.items()}

EXPAND_LOG_OPTIONS = ["none", "summary", "champion", "full"]

# ── Presets ───────────────────────────────────────────────────────────────────
QUICK_DISCOVERY = {
    "experiment_name": "Quick Discovery",
    "description": "Aggressive discovery: fast naive binning, wide search.",
    "top_n_vars": 20, "max_segments": 15, "max_feature_reuse": 2,
    "enable_diversity": False, "ignore_features": [],
    "binning_method": "naive", "naive_bins": 5, "max_expansion_hops": 1,
    "enable_1way": True, "enable_2way": True, "enable_3way": True,
    "selection_metric": "iv", "min_sample_size": 500, "min_lift": 1.2,
    "min_events": 50, "param_grid": None, "sort_priority": "rate_lift_count",
    "n_jobs": -1, "expand_log_mode": "none",
}

CONSERVATIVE = {
    "experiment_name": "Conservative",
    "description": "Strict constraints, stable optimal quantile binning.",
    "top_n_vars": 10, "max_segments": 5, "max_feature_reuse": 1,
    "enable_diversity": False, "ignore_features": [],
    "binning_method": "optimal_quantile", "naive_bins": 5, "max_expansion_hops": 0,
    "enable_1way": True, "enable_2way": True, "enable_3way": False,
    "selection_metric": "response_rate", "min_sample_size": 5000, "min_lift": 2.0,
    "min_events": 500, "param_grid": None, "sort_priority": "rate_lift_count",
    "n_jobs": -1, "expand_log_mode": "none",
}

PRESETS = {"Quick Discovery": QUICK_DISCOVERY, "Conservative": CONSERVATIVE}

DEFAULTS = {
    "experiment_name": "",
    "description": "",
    "data_table": "udl_data",
    "target_col": "",
    "primary_key": "",
    "top_n_vars": 15,
    "max_segments": 10,
    "max_feature_reuse": 1,
    "feature_groups": {},
    "enable_diversity": False,
    "ignore_features": [],
    "binning_method": "optimal_cart",
    "naive_bins": 5,
    "max_expansion_hops": 0,
    "enable_1way": True,
    "enable_2way": True,
    "enable_3way": True,
    "selection_metric": "iv",
    "min_sample_size": 1000,
    "min_lift": 1.5,
    "min_events": 100,
    "param_grid": None,
    "sort_priority": "rate_lift_count",
    "n_jobs": -1,
    "expand_log_mode": "none",
}


def build_cfg(body):
    """Coerce a client-supplied config dict into a validated builder config."""
    cfg = dict(DEFAULTS)
    for k, v in (body or {}).items():
        if k in cfg or k == "feature_groups":
            cfg[k] = v
    bm = BIN_MAP.get(str(cfg.get("binning_method", "")), cfg.get("binning_method", "optimal_cart"))
    cfg["binning_method"] = bm
    sm = METRIC_MAP.get(str(cfg.get("selection_metric", "")), cfg.get("selection_metric", "iv"))
    cfg["selection_metric"] = sm
    sp = cfg.get("sort_priority", "rate_lift_count")
    if isinstance(sp, str) and sp in SORT_PRIORITY_MAP and sp not in VALID_SORT_PRIORITIES:
        sp = SORT_PRIORITY_RMAP.get(sp, sp)
    if sp not in VALID_SORT_PRIORITIES:
        sp = "rate_lift_count"
    cfg["sort_priority"] = sp
    if isinstance(cfg.get("n_jobs"), str):
        cfg["n_jobs"] = N_JOBS_MAP.get(cfg["n_jobs"], -1)
    try:
        cfg["n_jobs"] = int(cfg.get("n_jobs") or -1)
    except Exception:
        cfg["n_jobs"] = -1
    if not cfg.get("experiment_name"):
        cfg["experiment_name"] = f"exp_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    cfg.setdefault("data_table", "udl_data")
    return cfg


def validate_params(cfg, all_cols):
    issues = []
    if cfg.get("target_col") not in all_cols:
        issues.append(f"Target column '{cfg.get('target_col')}' is not in the dataset.")
    pk = cfg.get("primary_key") or ""
    if pk and pk not in all_cols:
        issues.append(f"Primary key column '{pk}' is not in the dataset.")
    if not (cfg.get("enable_1way") or cfg.get("enable_2way") or cfg.get("enable_3way")):
        issues.append("Enable at least one rule type (1-way / 2-way / 3-way).")
    if int(cfg.get("min_events", 0) or 0) > int(cfg.get("min_sample_size", 0) or 0):
        issues.append(
            f"min_events ({cfg.get('min_events')}) exceeds min_sample_size "
            f"({cfg.get('min_sample_size')}) — no rule could ever pass.")
    if cfg.get("target_col") in (cfg.get("ignore_features") or []):
        issues.append("Target column cannot be listed under ignore features.")
    for group, feats in (cfg.get("feature_groups") or {}).items():
        if cfg.get("target_col") in feats:
            issues.append(f"Target column cannot be inside feature group '{group}'.")
        for feat in feats:
            if feat not in all_cols:
                issues.append(f"Feature '{feat}' in group '{group}' is not in the dataset.")
    if cfg.get("param_grid"):
        pg = cfg["param_grid"]
        if not pg.get("min_sample_size") and not pg.get("min_lift"):
            issues.append("Grid search needs at least one min_sample_size or min_lift value.")
    if cfg.get("binning_method") == "naive" and int(cfg.get("naive_bins", 5) or 5) < 3:
        issues.append("Naive binning needs at least 3 bins.")
    return issues


def estimate_seconds(cfg, n_rows):
    pg = cfg.get("param_grid") or {}
    combos = max(1, len(pg.get("min_sample_size") or [1])) * max(1, len(pg.get("min_lift") or [1]))
    base = (max(n_rows, 1000) / 100_000.0) * 6.0
    if cfg.get("binning_method") == "naive":
        base *= 0.8
    ways = int(cfg.get("enable_1way") or 0) + int(cfg.get("enable_2way") or 0) + int(cfg.get("enable_3way") or 0)
    base *= max(1, ways)
    hops = max(1, int(cfg.get("max_expansion_hops", 0) or 0) + 1)
    base *= max(1.0, hops * 0.6)
    base *= combos
    return round(base, 1)


# ── Templates (templates.json CRUD) ───────────────────────────────────────────
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


def delete_template(name):
    templates = load_templates()
    if name in templates:
        del templates[name]
        with open(TEMPLATES_FILE, "w", encoding="utf-8") as fh:
            json.dump(templates, fh, indent=2)
        return True
    return False


def normalize_cfg(cfg):
    """Coerce stored config values to builder-accepted forms (Module 4 helper)."""
    cfg = dict(cfg or {})
    bm = cfg.get("binning_method")
    if bm in ("Optimal (CART)", "Optimal (Quantile)", "Naive"):
        bm = BIN_MAP.get(bm)
    if bm not in ("naive", "optimal_cart", "optimal_quantile", "optimal"):
        bm = "optimal_cart"
    cfg["binning_method"] = bm
    sm = cfg.get("selection_metric")
    if sm in ("IV", "Response Rate"):
        sm = METRIC_MAP.get(sm)
    if sm not in ("iv", "response_rate"):
        sm = "iv"
    cfg["selection_metric"] = sm
    try:
        cfg["n_jobs"] = int(cfg.get("n_jobs", -1))
    except Exception:
        cfg["n_jobs"] = -1
    if cfg.get("sort_priority") not in VALID_SORT_PRIORITIES:
        cfg["sort_priority"] = "rate_lift_count"
    if cfg.get("expand_log_mode") not in EXPAND_LOG_OPTIONS:
        cfg["expand_log_mode"] = "none"
    return cfg


def leaderboard_for_clone():
    """Return a light leaderboard snapshot (id/name/created_at) for the clone dropdown."""
    rows = read_leaderboard_raw()
    if rows is None:
        return None
    return [
        {"exp_id": r[0], "name": r[1], "created_at": str(r[2]), "builder_params": r[3]}
        for r in rows
    ]
