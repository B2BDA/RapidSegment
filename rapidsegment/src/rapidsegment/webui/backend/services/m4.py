"""Module 4 (Results Dashboard) computations: scorecard, feature health,
diagnostics builder and Plotly chart data. Mirrors
``rapidsegment/ui/pages/4_Results_Dashboard.py``.
"""
from __future__ import annotations

import json
import os
import shutil

import duckdb
import pandas as pd

from rapidsegment import StrategicSegmentBuilder, StrategicSegmentScore

from config import normalize_cfg
from storage import (
    ARTIFACTS_DIR, SUITE_DIR, active_db, db_query, jsonable, table_cols,
)

SEG_COLORS = [
    "#6366f1", "#f59e0b", "#22c55e", "#ef4444", "#3b82f6",
    "#ec4899", "#14b8a6", "#f97316", "#8b5cf6", "#06b6d4",
]


# ── Rule parsing helpers ──────────────────────────────────────────────────────
def segment_variables(rule_string):
    vars_ = []
    for part in str(rule_string).split("&"):
        if "=" in part:
            vars_.append(part.split("=", 1)[0].strip())
    return vars_


def segment_complexity(rule_string):
    return len([p for p in str(rule_string).split("&") if "=" in p])


def tracked_features_from_artifacts(exp_id):
    if not exp_id:
        return None
    art = os.path.join(ARTIFACTS_DIR, exp_id, "result.json")
    if not os.path.exists(art):
        return None
    try:
        with open(art, "r", encoding="utf-8") as fh:
            saved = json.load(fh)
        diag = (saved.get("result") or {}).get("diagnostics_")
        if diag:
            fs = (diag[-1].get("features_state") or {})
            if fs:
                return sorted(fs.keys())
    except Exception:
        pass
    return None


def eligible_features_from_dataset(cfg):
    data_path = active_db()
    if not os.path.exists(data_path):
        return []
    try:
        cols = table_cols(data_path)
        target = cfg.get("target_col")
        ignore = set(cfg.get("ignore_features") or [])
        pk = cfg.get("primary_key")
        if pk:
            ignore.add(pk)
        return sorted(c for c in cols if c not in ignore and c != target)
    except Exception:
        return []


# ── Scorecard (StrategicSegmentScore) ─────────────────────────────────────────
def build_scorecard(cfg, exp_id, segments):
    data_path = active_db()
    if not segments or not os.path.exists(data_path):
        return None
    target = cfg.get("target_col")
    if not target:
        return None
    seg_cols = [f"seg_{s['segment_id']}" for s in segments]
    case_exprs = ", ".join(
        f"CASE WHEN {s['sql_filter']} THEN 1 ELSE 0 END AS seg_{s['segment_id']}"
        for s in segments
    )
    art_dir = os.path.join(SUITE_DIR, "artifacts", exp_id)
    os.makedirs(art_dir, exist_ok=True)

    flags_db = os.path.join(art_dir, "scored.duckdb")
    if os.path.exists(flags_db):
        os.remove(flags_db)
    src = data_path.replace("\\", "/")
    con = duckdb.connect(flags_db)
    try:
        con.execute(f"ATTACH '{src}' AS src (READ_ONLY)")
        con.execute(
            f"CREATE TABLE df AS SELECT ROW_NUMBER() OVER () AS rs_row_id, "
            f'"{target}" AS "{target}", {case_exprs} FROM src.udl_data'
        )
    finally:
        con.close()

    try:
        scorer = StrategicSegmentScore(
            target_col=target, primary_key="rs_row_id", segment_cols=seg_cols)
        export_path = os.path.join(art_dir, "scorecard.json")
        score_db = os.path.join(art_dir, "score_work.duckdb")
        if os.path.exists(score_db):
            os.remove(score_db)
        artifact = scorer.calculate_and_export_weights(
            flags_db, export_path=export_path, db_path=score_db)
        return jsonable(artifact)
    except Exception:
        return None


def map_weights(scorecard, segments):
    if not scorecard:
        return {}
    out = {}
    for s in segments:
        w = scorecard.get("segment_weights", {}).get(f"seg_{s['segment_id']}", {})
        out[s["segment_id"]] = w.get("weight", 0)
    return out


# ── Visualization builders (Plotly figure dicts) ──────────────────────────────
def fig_scatter(segments):
    data = []
    for i, s in enumerate(segments):
        data.append({
            "type": "scatter", "mode": "markers+text",
            "x": [float(s.get("count") or 0)],
            "y": [float(s.get("lift") or 0)],
            "name": f"Seg {s['segment_id']}",
            "text": [f"Seg {s['segment_id']}"],
            "textposition": "top center",
            "marker": {
                "size": max(12, float(s.get("capture_rate", s.get("coverage", 0)) or 5) * 4),
                "color": SEG_COLORS[i % len(SEG_COLORS)],
                "opacity": 0.75,
                "line": {"width": 1, "color": "#0d1117"},
            },
            "hovertext": s.get("rule_string", ""),
        })
    return {
        "data": data,
        "layout": {
            "title": "Lift vs. Volume", "xaxis_title": "Count (volume)",
            "yaxis_title": "Lift (x)", "height": 420, "template": "plotly_white",
        },
    }


def fig_distribution(segments, target):
    seg_ids, events, nonev = [], [], []
    for s in segments:
        cnt = float(s.get("count") or 0)
        rate = float(s.get("rate") or 0) / 100.0
        seg_ids.append(f"Seg {s['segment_id']}")
        events.append(cnt * rate)
        nonev.append(cnt * (1 - rate))
    data = [
        {"type": "bar", "x": seg_ids, "y": events, "name": "Events", "marker_color": "#ef4444"},
        {"type": "bar", "x": seg_ids, "y": nonev, "name": "Non-Events", "marker_color": "#6366f1"},
    ]
    return {
        "data": data,
        "layout": {
            "barmode": "stack", "title": "Segment Distribution (Events vs Non-Events)",
            "xaxis_title": "Segment", "yaxis_title": "Count", "height": 420,
            "template": "plotly_white",
        },
    }


def fig_sunburst(segments):
    from collections import defaultdict
    grp_children = defaultdict(list)
    for s in segments:
        cx = segment_complexity(s.get("rule_string", ""))
        grp_children[cx].append((s["segment_id"], float(s.get("count") or 1)))
    ids, labels, parents, values, colors = [], [], [], [], []
    total = 0.0
    for cx in sorted(grp_children):
        grp_id = f"grp_{cx}"
        grp_count = sum(c for _, c in grp_children[cx])
        total += grp_count
        ids.append(grp_id)
        labels.append(f"{cx}-way rules")
        parents.append("root")
        values.append(grp_count)
        colors.append(SEG_COLORS[(cx - 1) % len(SEG_COLORS)])
        for seg_i, cnt in grp_children[cx]:
            ids.append(f"seg_{seg_i}")
            labels.append(f"Seg {seg_i}")
            parents.append(grp_id)
            values.append(cnt)
            colors.append(SEG_COLORS[seg_i % len(SEG_COLORS)])
    ids.insert(0, "root")
    labels.insert(0, "All segments")
    parents.insert(0, "")
    values.insert(0, total)
    colors.insert(0, "#0d1117")
    hovertext = []
    for _id in ids:
        if _id.startswith("seg_"):
            si = int(_id.split("_")[1])
            hovertext.append(next((s.get("rule_string", "") for s in segments
                                   if s["segment_id"] == si), ""))
        else:
            hovertext.append("")
    data = [{
        "type": "sunburst", "ids": ids, "labels": labels, "parents": parents,
        "values": values, "marker": {"colors": colors}, "branchvalues": "total",
        "hovertext": hovertext,
    }]
    return {
        "data": data,
        "layout": {"title": "Rule Complexity Breakdown", "height": 460, "template": "plotly_white"},
    }


def fig_decile(scorecard):
    dt = (scorecard or {}).get("decile_min_thresholds") or {}
    if not dt:
        return None
    deciles = sorted(dt.keys(), key=lambda k: int(k))
    xs = [int(k) for k in deciles]
    ys = [float(dt[k]) for k in deciles]
    return {
        "data": [{"type": "scatter", "x": xs, "y": ys, "mode": "lines+markers",
                  "name": "Min score threshold"}],
        "layout": {
            "title": "Decile Thresholds", "xaxis_title": "Decile (1 = best)",
            "yaxis_title": "Min score", "height": 420, "template": "plotly_white",
        },
    }


def fig_feature_importance(segments, feature_groups):
    counts = {}
    for s in segments:
        for v in segment_variables(s.get("rule_string", "")):
            counts[v] = counts.get(v, 0) + 1
    if not counts:
        return None
    items = sorted(counts.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    group_color = {}
    for gi, (g, feats) in enumerate((feature_groups or {}).items()):
        for f in feats:
            group_color[f] = SEG_COLORS[gi % len(SEG_COLORS)]
    cols = [group_color.get(n, "#94a3b8") for n in names]
    grp_label = {n: next((g for g, fs in (feature_groups or {}).items() if n in fs), "ungrouped")
                 for n in names}
    data = [{
        "type": "bar", "x": vals, "y": names, "orientation": "h",
        "marker_color": cols,
        "text": [f" ({grp_label[n]})" for n in names], "textposition": "outside",
    }]
    return {
        "data": data,
        "layout": {
            "title": "Feature Importance (usage count)", "xaxis_title": "# times used",
            "yaxis_title": "Feature", "height": max(360, 40 * len(names)),
            "template": "plotly_white",
        },
    }


# ── Feature health (corrected local implementation) ───────────────────────────
def generate_feature_health_local(data_path, features, target, type_overrides=None, naive_bins=5):
    if not features or not os.path.exists(data_path):
        return pd.DataFrame()
    type_overrides = type_overrides or {}
    con = duckdb.connect(":memory:")
    src = data_path.replace("\\", "/")
    con.execute(f"ATTACH '{src}' AS src (READ_ONLY)")
    con.execute("CREATE VIEW input_df AS SELECT * FROM src.udl_data")
    columns_types = {r[0]: r[1] for r in con.execute("DESCRIBE input_df").fetchall()}

    target_expr = f"""
    (CASE
        WHEN TRY_CAST("{target}" AS DOUBLE) IS NOT NULL THEN TRY_CAST("{target}" AS DOUBLE)
        WHEN LOWER(TRIM(CAST("{target}" AS VARCHAR))) IN ('1','true','yes','y','t') THEN 1.0
        ELSE 0.0
    END)
    """
    num_types = ("INT", "BIGINT", "DOUBLE", "FLOAT", "DECIMAL", "REAL",
                 "NUMERIC", "HUGEINT", "TINYINT", "SMALLINT")
    missing_test = "IN ('','None','nan','NaN','<NA>','null','NULL')"

    rows = []
    for col in features:
        if col not in columns_types:
            continue
        ov = type_overrides.get(str(col), "AUTO")
        duckdb_type = columns_types[col].upper()
        is_num_type = any(t in duckdb_type for t in num_types)
        treat_categorical = (ov == "CATEGORICAL") or (ov == "AUTO" and not is_num_type)

        if treat_categorical:
            q = f"""
            SELECT
                CASE
                    WHEN "{col}" IS NULL OR TRIM(CAST("{col}" AS VARCHAR)) {missing_test} THEN 'Missing'
                    ELSE CAST("{col}" AS VARCHAR)
                END AS bin,
                COUNT(*) AS total_count,
                SUM({target_expr}) AS event_count,
                (SUM({target_expr}) * 100.0 / COUNT(*)) AS response_rate,
                CASE
                    WHEN "{col}" IS NULL OR TRIM(CAST("{col}" AS VARCHAR)) {missing_test} THEN TRUE
                    ELSE FALSE
                END AS is_missing
            FROM input_df
            GROUP BY 1, 5
            ORDER BY is_missing ASC, bin ASC
            """
        else:
            nb = max(2, int(naive_bins))
            q = f"""
            WITH ranked AS (
                SELECT
                    TRY_CAST("{col}" AS DOUBLE) AS val,
                    {target_expr} AS target_val,
                    NTILE({nb}) OVER (ORDER BY TRY_CAST("{col}" AS DOUBLE)) AS tile
                FROM input_df
                WHERE TRY_CAST("{col}" AS DOUBLE) IS NOT NULL
            ),
            num_bins AS (
                SELECT
                    tile,
                    MIN(val) AS bmin, MAX(val) AS bmax,
                    COUNT(*) AS total_count,
                    SUM(target_val) AS event_count,
                    (SUM(target_val) * 100.0 / COUNT(*)) AS response_rate
                FROM ranked
                GROUP BY tile
            )
            SELECT
                CASE
                    WHEN bmin = bmax
                    THEN 'Bin ' || tile || ': ' || ROUND(bmin, 6)
                    ELSE 'Bin ' || tile || ': [' || ROUND(bmin, 6) || ', ' || ROUND(bmax, 6) || ']'
                END AS bin,
                total_count, event_count, response_rate, FALSE AS is_missing, tile
            FROM num_bins
            UNION ALL
            SELECT 'Missing' AS bin, COUNT(*) AS total_count,
                   SUM({target_expr}) AS event_count,
                   (SUM({target_expr}) * 100.0 / NULLIF(COUNT(*), 0)) AS response_rate,
                   TRUE AS is_missing, NULL AS tile
            FROM input_df
            WHERE "{col}" IS NULL
            HAVING COUNT(*) > 0
            ORDER BY is_missing ASC, tile ASC
            """
        for row in con.execute(q).fetchall():
            rows.append({
                "feature": col,
                "bin": row[0],
                "total_count": int(row[1]),
                "event_count": int(row[2] or 0),
                "response_rate_%": round(float(row[3] or 0.0), 4),
                "is_missing": bool(row[4]),
            })
    con.close()
    return pd.DataFrame(rows)


# ── Diagnostics builder (explain_* methods) ───────────────────────────────────
def build_diag_builder(cfg, exp_id):
    full = normalize_cfg(cfg)
    if exp_id:
        art = os.path.join(ARTIFACTS_DIR, exp_id, "result.json")
        if os.path.exists(art):
            try:
                with open(art, "r", encoding="utf-8") as fh:
                    saved = json.load(fh)
                res = saved.get("result") or {}
                diag = res.get("diagnostics_")
                if diag:
                    b = StrategicSegmentBuilder(
                        target=full.get("target_col") or "",
                        n_jobs=full.get("n_jobs", -1),
                        min_sample_size=full.get("min_sample_size", 1000),
                        min_lift=full.get("min_lift", 1.5),
                        min_events=full.get("min_events", 100),
                        top_n_vars=full.get("top_n_vars", 15),
                        max_segments=full.get("max_segments", 10),
                        max_feature_reuse=full.get("max_feature_reuse", 1),
                        enable_diversity=full.get("enable_diversity", False),
                        enable_1way=full.get("enable_1way", True),
                        enable_2way=full.get("enable_2way", True),
                        enable_3way=full.get("enable_3way", True),
                        selection_metric=full.get("selection_metric", "iv"),
                        binning_method=full.get("binning_method", "optimal_cart"),
                        naive_bins=full.get("naive_bins", 5),
                        max_expansion_hops=full.get("max_expansion_hops", 0),
                    )
                    b.diagnostics_ = diag
                    b.segments = res.get("segments") or []
                    b.stop_reason = res.get("stop_reason")
                    b.feature_usage_counts = res.get("feature_usage_counts") or {}
                    return b
            except Exception:
                pass
    data_path = active_db()
    if not os.path.exists(data_path):
        return None
    b = StrategicSegmentBuilder(
        target=full.get("target_col") or "",
        n_jobs=full.get("n_jobs", -1),
        min_sample_size=full.get("min_sample_size", 1000),
        min_lift=full.get("min_lift", 1.5),
        min_events=full.get("min_events", 100),
        top_n_vars=full.get("top_n_vars", 15),
        max_segments=full.get("max_segments", 10),
        max_feature_reuse=full.get("max_feature_reuse", 1),
        enable_diversity=full.get("enable_diversity", False),
        enable_1way=full.get("enable_1way", True),
        enable_2way=full.get("enable_2way", True),
        enable_3way=full.get("enable_3way", True),
        selection_metric=full.get("selection_metric", "iv"),
        binning_method=full.get("binning_method", "optimal_cart"),
        naive_bins=full.get("naive_bins", 5),
        max_expansion_hops=full.get("max_expansion_hops", 0),
    )
    b.extract_segments(data_path)
    return b
