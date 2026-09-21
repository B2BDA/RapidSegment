"""
RapidSegment — Module 4: Results Dashboard & Visualization
===========================================================
Displays extracted segments with rich context, metrics, and actionable export
options, plus a deployable scorecard (StrategicSegmentScore) and a diagnostic
drilldown (feature health report + no-segments explanation).

Consumes (same contract as Module 2 / Module 3):
    - st.session_state["experiment"]     dict      {exp_id, name, created_at, status,
                                                    execution_time_sec, target_col,
                                                    primary_key, data_rows, data_cols,
                                                    config (builder params JSON),
                                                    result (segments + coverage summary),
                                                    logs (captured terminal stream)}

Falls back to the most recent row of `.rapidsegment_suite/suite_data.db` when no
live experiment is in session state (e.g. opened directly from the sidebar).

Files touched:
    read  .rapidsegment_suite/module1_data.duckdb   (udl_data — for scorecard / health)
    r/w   .rapidsegment_suite/suite_data.db         (experiments table — read only here)
    write .rapidsegment_suite/artifacts/<exp_id>/   (scorecard.json)

Run with:  streamlit run Module_4_results.py
"""
import contextlib
import io
import json
import os
import zipfile

import duckdb
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from rapidsegment import StrategicSegmentScore, StrategicSegmentBuilder
from rapidsegment.ui._theme import apply_cyberpunk_theme
from rapidsegment.builder import _quote_sql_ident

from rapidsegment.ui._state import (
    SUITE_DIR, DB_FILE, DB_FILE_MOD, SUITE_DB, ARTIFACTS_DIR,
    active_db, db_query, db_scalar, rerun, card, _jsonable, fmt_duration,
    _build_coverage_sql, _build_sql_script, render_coverage_table,
)

SEG_COLORS = [
    "#6366f1", "#f59e0b", "#22c55e", "#ef4444", "#3b82f6",
    "#ec4899", "#14b8a6", "#f97316", "#8b5cf6", "#06b6d4",
    "#84cc16", "#eab308", "#a855f7", "#10b981", "#fb7185",
]

# ── Experiment loading ───────────────────────────────────────────────────────
def _build_exp_from_row(row):
    """Build an experiment dict from a suite_data.db row.

    Shared by the saved-run loader and the latest-run fallback.
    """
    (
        exp_id, name, created_at, data_rows, data_cols, status,
        execution_time_sec, target_col, primary_key, builder_params,
        segments_count, avg_lift, max_lift, coverage_pct, baseline_rate, error_msg,
    ) = row
    cfg = _jsonable(json.loads(builder_params)) if builder_params else {}
    return {
        "exp_id": exp_id, "name": name, "created_at": str(created_at),
        "status": status, "execution_time_sec": float(execution_time_sec or 0),
        "target_col": target_col, "primary_key": primary_key or "",
        "data_rows": int(data_rows or 0), "data_cols": int(data_cols or 0),
        "config": cfg,
        "result": {
            "segments_count": int(segments_count or 0),
            "avg_lift": float(avg_lift or 0),
            "max_lift": float(max_lift or 0),
            "coverage_pct": float(coverage_pct or 0),
            "baseline_rate_pct": float(baseline_rate or 0),
            "error_msg": error_msg,
            "segments": [], "coverage": [], "stop_reason": None,
        },
        "logs": [],
    }


def load_experiment():
    """Return (exp_dict, source_label).

    Prefers the saved run requested from the Leaderboard (`m4_view_exp`); else
    a live session; else the latest DB row.
    """
    saved_id = st.session_state.get("m4_view_exp")
    if saved_id:
        st.session_state.pop("m4_view_exp", None)
        try:
            con = duckdb.connect(SUITE_DB, read_only=True)
            row = con.execute(
                "SELECT exp_id, name, created_at, data_rows, data_cols, status, "
                "execution_time_sec, target_col, primary_key, builder_params, "
                "segments_count, avg_lift, max_lift, coverage_pct, baseline_rate, error_msg "
                "FROM experiments WHERE exp_id = ?",
                [saved_id],
            ).fetchone()
            con.close()
        except Exception:
            row = None
        if row:
            saved = _build_exp_from_row(row)
            st.session_state["experiment"] = saved
            return saved, "saved run (from Leaderboard)"

    live = st.session_state.get("experiment")
    if isinstance(live, dict) and live.get("result"):
        return live, "live session"
    if not os.path.exists(SUITE_DB):
        return None, "no suite_data.db"
    try:
        con = duckdb.connect(SUITE_DB, read_only=True)
        has = con.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name='experiments'"
        ).fetchone()
        if not has:
            con.close()
            return None, "no experiments table"
        row = con.execute(
            "SELECT exp_id, name, created_at, data_rows, data_cols, status, "
            "execution_time_sec, target_col, primary_key, builder_params, "
            "segments_count, avg_lift, max_lift, coverage_pct, baseline_rate, error_msg "
            "FROM experiments ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        con.close()
        if not row:
            return None, "empty experiments table"
        return _build_exp_from_row(row), "suite_data.db (latest)"
    except Exception as exc:
        return None, f"read error: {exc}"


def load_segments_from_artifacts(exp):
    """Recover full segments/coverage/stop_reason from the saved artifact JSON."""
    exp_id = exp.get("exp_id")
    if not exp_id:
        return None, None
    path = os.path.join(SUITE_DIR, "artifacts", exp_id, "result.json")
    if not os.path.exists(path):
        return None, None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            saved = json.load(fh)
        res = saved.get("result") or {}
        if exp.get("result") and res.get("stop_reason"):
            exp["result"]["stop_reason"] = res["stop_reason"]
        return res.get("segments") or [], res.get("coverage") or []
    except Exception:
        return None, None


def load_dataset():
    """Deprecated: returns the on-disk dataset path instead of a full dataframe.

    Kept only as a thin shim; callers should use ``active_db()`` directly so the
    full dataset is never materialised into Python (pandas) memory.
    """
    return active_db()


# ── Rule parsing helpers ──────────────────────────────────────────────────────
def segment_variables(rule_string):
    vars_ = []
    for part in str(rule_string).split("&"):
        if "=" in part:
            vars_.append(part.split("=", 1)[0].strip())
    return vars_


def segment_complexity(rule_string):
    return len([p for p in str(rule_string).split("&") if "=" in p])


def _tracked_features_from_artifacts(exp_id):
    """Return the engine's tracked feature names (sorted) from persisted diagnostics, or None.

    Reuses the diagnostics_ Module 3 saved into artifacts/<exp_id>/result.json — no
    re-extraction. This is the authoritative feature set (every eligible column the
    builder scored per iteration), matching StrategicSegmentBuilder.features_state.
    """
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


def _eligible_features_from_dataset(data_path, cfg):
    """Fallback feature list: columns of udl_data minus target / primary key / ignored.

    Mirrors builder.extract_segments' eligible_cols when persisted diagnostics are
    unavailable, so the module still surfaces the same eligible set.
    """
    if not data_path or not os.path.exists(data_path):
        return []
    try:
        con = duckdb.connect(data_path, read_only=True)
        try:
            cols = [r[0] for r in con.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='udl_data'").fetchall()]
        finally:
            con.close()
        target = cfg.get("target_col")
        ignore = set(cfg.get("ignore_features") or [])
        pk = cfg.get("primary_key")
        if pk:
            ignore.add(pk)
        return sorted(c for c in cols if c not in ignore and c != target)
    except Exception:
        return []


# ── Scorecard (StrategicSegmentScore) ─────────────────────────────────────────
def build_scorecard(cfg, data_path, segments):
    """Create per-segment flag columns, score the population, return artifact dict.

    Zero-copy: flag columns and the scorecard are computed entirely inside
    DuckDB on the persisted dataset — no full pandas materialisation of the data.
    """
    if not segments or not data_path or not os.path.exists(data_path):
        return None
    target = cfg.get("target_col")
    if not target:
        return None
    seg_cols = [f"seg_{s['segment_id']}" for s in segments]
    case_exprs = ", ".join(
        f"CASE WHEN {s['sql_filter']} THEN 1 ELSE 0 END AS seg_{s['segment_id']}"
        for s in segments
    )
    exp_id = st.session_state.get("experiment", {}).get("exp_id") or "m4"
    art_dir = os.path.join(SUITE_DIR, "artifacts", exp_id)
    os.makedirs(art_dir, exist_ok=True)

    # Build the flagged table in DuckDB (attach source read-only; add a stable
    # row id so the scorer has a primary key). No pandas involved.
    flags_db = os.path.join(art_dir, "scored.duckdb")
    if os.path.exists(flags_db):
        os.remove(flags_db)
    src = data_path.replace("\\", "/")
    # Only keep what the scorer needs (row id, target, segment flags) — NOT a
    # full copy of all 70 columns. Saves ~2 full-dataset copies of disk/IO.
    con = duckdb.connect(flags_db)
    try:
        con.execute(f"ATTACH '{src}' AS src (READ_ONLY)")
        con.execute(
            f"CREATE TABLE df AS SELECT ROW_NUMBER() OVER () AS rs_row_id, "
            f'{_quote_sql_ident(target)} AS {_quote_sql_ident(target)}, {case_exprs} FROM src.udl_data'
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
        # Pass the DuckDB file path (zero-copy); the scorer attaches and reads `df`.
        artifact = scorer.calculate_and_export_weights(
            flags_db, export_path=export_path, db_path=score_db)
        return _jsonable(artifact)
    except Exception as exc:
        st.warning(f"Scorecard computation failed: {exc}")
        return None


def map_weights(scorecard, segments):
    """Return dict segment_id -> weight (int) from scorecard.segment_weights."""
    if not scorecard:
        return {}
    out = {}
    for s in segments:
        w = scorecard.get("segment_weights", {}).get(f"seg_{s['segment_id']}", {})
        out[s["segment_id"]] = w.get("weight", 0)
    return out


# ── Visualization builders ───────────────────────────────────────────────────
def _fig_scatter(segments):
    fig = go.Figure()
    for i, s in enumerate(segments):
        fig.add_trace(go.Scatter(
            x=[float(s.get("count") or 0)],
            y=[float(s.get("lift") or 0)],
            mode="markers+text",
            name=f"Seg {s['segment_id']}",
            text=[f"Seg {s['segment_id']}"],
            textposition="top center",
            marker=dict(
                size=max(12, float(s.get("capture_rate", s.get("coverage", 0)) or 5) * 4),
                color=SEG_COLORS[i % len(SEG_COLORS)],
                opacity=0.75,
                line=dict(width=1, color="#0d1117"),
            ),
            hovertext=s.get("rule_string", ""),
        ))
    fig.update_layout(
        title="Lift vs. Volume", xaxis_title="Count (volume)", yaxis_title="Lift (x)",
        height=420, template="plotly_white",
    )
    return fig


def _fig_distribution(segments, target):
    fig = go.Figure()
    seg_ids, events, nonev = [], [], []
    for s in segments:
        cnt = float(s.get("count") or 0)
        rate = float(s.get("rate") or 0) / 100.0
        seg_ids.append(f"Seg {s['segment_id']}")
        events.append(cnt * rate)
        nonev.append(cnt * (1 - rate))
    fig.add_bar(x=seg_ids, y=events, name="Events", marker_color="#ef4444")
    fig.add_bar(x=seg_ids, y=nonev, name="Non-Events", marker_color="#6366f1")
    fig.update_layout(barmode="stack", title="Segment Distribution (Events vs Non-Events)",
                      xaxis_title="Segment", yaxis_title="Count", height=420, template="plotly_white")
    return fig


def _fig_sunburst(segments):
    # Rule complexity = number of feature predicates ANDed in a rule.
    # Inner ring = complexity groups (1/2/3-way); outer ring = individual segments.
    # branchvalues="total" requires every parent value == sum of its children, so we
    # accumulate group + root counts from the segments.
    from collections import defaultdict
    grp_children = defaultdict(list)  # cx -> [(seg_id, count), ...]
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
    # Root must equal the sum of its children for branchvalues="total"
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
    fig = go.Figure(go.Sunburst(
        ids=ids, labels=labels, parents=parents, values=values,
        marker=dict(colors=colors), branchvalues="total", hovertext=hovertext,
    ))
    fig.update_layout(title="Rule Complexity Breakdown", height=460, template="plotly_white")
    return fig


def _fig_decile(scorecard):
    dt = (scorecard or {}).get("decile_min_thresholds") or {}
    if not dt:
        return None
    deciles = sorted(dt.keys(), key=lambda k: int(k))
    xs = [int(k) for k in deciles]
    ys = [float(dt[k]) for k in deciles]
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines+markers", name="Min score threshold"))
    fig.update_layout(
        title="Decile Thresholds", xaxis_title="Decile (1 = best)", yaxis_title="Min score",
        height=420, template="plotly_white",
    )
    return fig


def _fig_feature_importance(segments, feature_groups):
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
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h", marker_color=cols,
        text=[f" ({grp_label[n]})" for n in names], textposition="outside",
    ))
    fig.update_layout(title="Feature Importance (usage count)", xaxis_title="# times used",
                      yaxis_title="Feature", height=max(360, 40 * len(names)), template="plotly_white")
    return fig



# ── HTML report ───────────────────────────────────────────────────────────────
def _build_html_report(exp, res, segments, coverage, scorecard):
    name = exp.get("name", "Experiment")
    rows = "".join(
        f"<tr><td>{s['segment_id']}</td><td>{s.get('rule_string','')}</td>"
        f"<td>{s.get('count',0):,}</td><td>{float(s.get('rate',0)):.2f}%</td>"
        f"<td>{float(s.get('lift',0)):.2f}x</td></tr>"
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
<p><b>Experiment:</b> {name} . <b>ID:</b> {exp.get('exp_id','')} . <b>Status:</b> {exp.get('status','')}</p>
<div class="kpi">
<div><b>Segments</b><br>{res.get('segments_count',0)}</div>
<div><b>Coverage %</b><br>{res.get('coverage_pct',0):.2f}</div>
<div><b>Avg lift</b><br>{res.get('avg_lift',0):.2f}x</div>
<div><b>Max lift</b><br>{res.get('max_lift',0):.2f}x</div>
<div><b>Baseline rate</b><br>{res.get('baseline_rate_pct',0):.2f}%</div>
<div><b>Elapsed</b><br>{fmt_duration(exp.get('execution_time_sec',0))}</div>
</div>
<h2>Segments</h2>
<table><tr><th>ID</th><th>Rule</th><th>Count</th><th>Event Rate</th><th>Lift</th></tr>
{rows}</table>
<h2>Scorecard</h2>
<pre>{json.dumps(sc, indent=2)}</pre>
</body></html>"""


# ── Rendering ─────────────────────────────────────────────────────────────────
def render_summary_cards(exp, res):
    m = st.columns(6)
    m[0].metric("Segments", res.get("segments_count", len(res.get("segments") or [])))
    m[1].metric("Coverage %", f"{res.get('coverage_pct', 0):.2f}")
    m[2].metric("Avg lift", f"{res.get('avg_lift', 0):.2f}x")
    m[3].metric("Max lift", f"{res.get('max_lift', 0):.2f}x")
    m[4].metric("Baseline rate", f"{res.get('baseline_rate_pct', 0):.2f}%")
    m[5].metric("Elapsed", fmt_duration(exp.get("execution_time_sec", 0)))



def render_visualizations(segments, coverage, scorecard, cfg):
    if not segments:
        st.info("Nothing to visualize - no segments were produced.")
        return
    tabs = st.tabs(["Lift vs Volume", "Distribution", "Rule Complexity", "Decile", "Feature Importance"])
    feature_groups = (cfg.get("config") if isinstance(cfg.get("config"), dict) else cfg).get("feature_groups") or {}
    with tabs[0]:
        st.plotly_chart(_fig_scatter(segments), width='stretch')
    with tabs[1]:
        st.plotly_chart(_fig_distribution(segments, cfg.get("target_col")), width='stretch')
    with tabs[2]:
        st.caption(
            "**Rule complexity** = number of feature conditions AND-ed in a rule. "
            "1-way = single feature, 2-way = two features, 3-way = three. Inner ring groups "
            "segments by complexity; outer ring shows each segment sized by population."
        )
        st.plotly_chart(_fig_sunburst(segments), width='stretch')
    with tabs[3]:
        fig = _fig_decile(scorecard)
        if fig is not None:
            st.plotly_chart(fig, width='stretch')
        else:
            st.caption("No decile thresholds - run the scorecard to generate them.")
    with tabs[4]:
        fig = _fig_feature_importance(segments, feature_groups)
        if fig is not None:
            st.plotly_chart(fig, width='stretch')
        else:
            st.caption("No feature usage to display.")


def render_scorecard_json(scorecard):
    if scorecard is None:
        st.caption("Scorecard not available (no segments, or dataset missing).")
        return
    distinct = len({w.get("weight") for w in scorecard.get("segment_weights", {}).values() if w.get("weight")})
    st.caption(f"Distinct score values: {distinct} "
               f"({'good for deciling' if distinct >= 10 else 'low - increase max_segments for smooth deciles'})")
    sc = st.container(height=420, border=True)
    sc.json(scorecard)


def _normalize_cfg(full):
    """Coerce stored config values to builder-accepted forms. Older runs (or rows
    duplicated from them) may hold human labels like 'Optimal (CART)' instead of the
    canonical value 'optimal_cart'; the constructor raises ValueError on those."""
    cfg = dict(full)
    bm = cfg.get("binning_method")
    if bm in ("Optimal (CART)", "Optimal (Quantile)", "Naive"):
        bm = {"Optimal (CART)": "optimal_cart",
              "Optimal (Quantile)": "optimal_quantile",
              "Naive": "naive"}.get(bm)
    if bm not in ("naive", "optimal_cart", "optimal_quantile", "optimal"):
        bm = "optimal_cart"
    cfg["binning_method"] = bm

    sm = cfg.get("selection_metric")
    if sm in ("IV", "Response Rate"):
        sm = {"IV": "iv", "Response Rate": "response_rate"}.get(sm)
    if sm not in ("iv", "response_rate"):
        sm = "iv"
    cfg["selection_metric"] = sm

    try:
        cfg["n_jobs"] = int(cfg.get("n_jobs", -1))
    except Exception:
        cfg["n_jobs"] = -1

    valid_sp = {
        "rate_lift_count", "lift_rate_count", "lift_count_rate", "count_lift_rate",
        "count_rate_lift", "rate_count_lift", "events_lift_rate", "events_rate_lift",
        "lift_events_rate", "rate_events_lift", "events_count_rate",
        "events_rate_count", "count_events_rate", "rate_events_count",
    }
    if cfg.get("sort_priority") not in valid_sp:
        cfg["sort_priority"] = "rate_lift_count"

    if cfg.get("expand_log_mode") not in ("none", "summary", "champion", "full"):
        cfg["expand_log_mode"] = "none"
    return cfg


def _build_diag_builder(cfg, data_path, exp_id=None):
    """Return a builder with populated diagnostics_ for the explain_* methods.

    Preferred path: reuse diagnostics_ persisted by Module 3 into
    artifacts/<exp_id>/result.json — no re-extraction (fast, and works even
    without the dataset loaded). Falls back to re-running extract_segments once
    (cached on session state) only when no persisted diagnostics are available.
    """
    cached = st.session_state.get("m4_diag_builder")
    if cached is not None and st.session_state.get("m4_diag_exp_id") == exp_id:
        return cached
    full = _normalize_cfg(cfg)

    # 1) Reuse persisted diagnostics (no segmentation re-run)
    if exp_id:
        art = os.path.join(ARTIFACTS_DIR, exp_id, "result.json")
        if os.path.exists(art):
            try:
                with open(art, "r", encoding="utf-8") as fh:
                    saved = json.load(fh)
                res = saved.get("result") or {}
                diag = res.get("diagnostics_")
                if diag:
                    # Prefer the artifact's own config (guaranteed to match the original run)
                    art_cfg = _normalize_cfg(saved.get("config") or cfg)
                    b = StrategicSegmentBuilder(
                        target=art_cfg.get("target_col") or "",
                        n_jobs=art_cfg.get("n_jobs", -1),
                        min_sample_size=art_cfg.get("min_sample_size", 1000),
                        min_lift=art_cfg.get("min_lift", 1.5),
                        min_events=art_cfg.get("min_events", 100),
                        top_n_vars=art_cfg.get("top_n_vars", 15),
                        max_segments=art_cfg.get("max_segments", 10),
                        max_feature_reuse=art_cfg.get("max_feature_reuse", 1),
                        enable_diversity=art_cfg.get("enable_diversity", False),
                        enable_1way=art_cfg.get("enable_1way", True),
                        enable_2way=art_cfg.get("enable_2way", True),
                        enable_3way=art_cfg.get("enable_3way", True),
                        selection_metric=art_cfg.get("selection_metric", "iv"),
                        binning_method=art_cfg.get("binning_method", "optimal_cart"),
                        naive_bins=art_cfg.get("naive_bins", 5),
                        max_expansion_hops=art_cfg.get("max_expansion_hops", 0),
                    )
                    b.diagnostics_ = diag
                    b.segments = res.get("segments") or []
                    b.stop_reason = res.get("stop_reason")
                    b.feature_usage_counts = res.get("feature_usage_counts") or {}
                    st.session_state["m4_diag_builder"] = b
                    st.session_state["m4_diag_exp_id"] = exp_id
                    return b
            except Exception:
                pass

    # 2) Fallback: re-extract once (cached)
    if not data_path or not os.path.exists(data_path):
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
    with st.spinner("Running extraction to collect diagnostics..."):
        b.extract_segments(data_path)
    st.session_state["m4_diag_builder"] = b
    st.session_state["m4_diag_exp_id"] = exp_id
    return b


def generate_feature_health_local(data_path, features, target, type_overrides=None, naive_bins=5):
    """Corrected feature health report (replaces the library version).

    Fixes two bugs in StrategicSegmentBuilder.generate_feature_health_report:
      * UI categorical overrides are respected — a column marked CATEGORICAL
        is binned by distinct value instead of being NTILE'd as numeric (the
        library decides numeric/categorical solely from the DuckDB column type,
        so an overridden-categorical numeric column was wrongly quantile-binned).
      * Numeric bin labels are made unique via the tile index
        ('Bin 1: [a, b]'), so adjacent/low-cardinality tiles never appear
        'repeated' after rounding collapses the [min,max] label.

    Reads the persisted dataset directly from ``data_path`` (a DuckDB file) via a
    lazy view — no full pandas / in-memory materialisation of the data.
    """
    if not features or not data_path or not os.path.exists(data_path):
        return pd.DataFrame()
    type_overrides = type_overrides or {}
    con = duckdb.connect(":memory:")
    src = data_path.replace("\\", "/")
    con.execute(f"ATTACH '{src}' AS src (READ_ONLY)")
    con.execute("CREATE VIEW input_df AS SELECT * FROM src.udl_data")
    columns_types = {r[0]: r[1] for r in con.execute("DESCRIBE input_df").fetchall()}

    target_expr = f"""
    (CASE
        WHEN TRY_CAST({_quote_sql_ident(target)} AS DOUBLE) IS NOT NULL THEN TRY_CAST({_quote_sql_ident(target)} AS DOUBLE)
        WHEN LOWER(TRIM(CAST({_quote_sql_ident(target)} AS VARCHAR))) IN ('1','true','yes','y','t') THEN 1.0
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
                    WHEN {_quote_sql_ident(col)} IS NULL OR TRIM(CAST({_quote_sql_ident(col)} AS VARCHAR)) {missing_test} THEN 'Missing'
                    ELSE CAST({_quote_sql_ident(col)} AS VARCHAR)
                END AS bin,
                COUNT(*) AS total_count,
                SUM({target_expr}) AS event_count,
                (SUM({target_expr}) * 100.0 / COUNT(*)) AS response_rate,
                CASE
                    WHEN {_quote_sql_ident(col)} IS NULL OR TRIM(CAST({_quote_sql_ident(col)} AS VARCHAR)) {missing_test} THEN TRUE
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
                    TRY_CAST({_quote_sql_ident(col)} AS DOUBLE) AS val,
                    {target_expr} AS target_val,
                    NTILE({nb}) OVER (ORDER BY TRY_CAST({_quote_sql_ident(col)} AS DOUBLE)) AS tile
                FROM input_df
                WHERE TRY_CAST({_quote_sql_ident(col)} AS DOUBLE) IS NOT NULL
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
            WHERE {_quote_sql_ident(col)} IS NULL
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


def render_diagnostics(exp, cfg, data_path, segments):
    st.subheader("Diagnostic Drilldown")
    res = exp.get("result") or {}
    stop = res.get("stop_reason")
    if stop:
        st.info(f"**Stop reason:** {stop}")
    else:
        st.caption("No stop reason recorded.")

    # 1) Feature Journey — dedicated space
    # Feature list = exactly what the engine tracked (all eligible columns except
    # target / primary key / ignored), so eligible-but-unused features are visible
    # here too. Mirrors StrategicSegmentBuilder.features_state (builder.py).
    feats = (_tracked_features_from_artifacts(exp.get("exp_id"))
             or _eligible_features_from_dataset(data_path, cfg))

    with st.expander("Feature Journey (audit trail per feature)", expanded=True):
        if not feats:
            st.caption("No features were tracked (no dataset / diagnostics available).")
        else:
            fj = st.selectbox("Choose a feature to trace", feats, key="m4_fj")
            if st.button("Show feature journey", key="m4_fj_btn"):
                if not data_path or not os.path.exists(data_path):
                    st.warning("Dataset not available - load it in Module 1 to enable diagnostics.")
                else:
                    b = _build_diag_builder(cfg, data_path, exp.get("exp_id"))
                    if b is not None:
                        buf = io.StringIO()
                        with contextlib.redirect_stdout(buf):
                            b.explain_feature_journey(fj)
                        st.code(buf.getvalue() or "(no journey recorded for this feature)", language="text")

    # 2) Feature Health Report
    with st.expander("Feature Health Report (bin-level stats)"):
        if not feats:
            st.caption("No features to profile (no dataset / diagnostics available).")
        else:
            sel = st.multiselect("Features to profile", feats, default=feats[:5], key="m4_health_sel")
            if st.button("Generate health report", key="m4_health"):
                if not data_path or not os.path.exists(data_path):
                    st.warning("Dataset not available - load data via Module 1 first.")
                else:
                    try:
                        type_overrides = st.session_state.get("type_overrides") or {}
                        with st.spinner("Profiling features..."):
                            hr = generate_feature_health_local(
                                data_path, list(sel), cfg.get("target_col", ""),
                                type_overrides, int(cfg.get("naive_bins", 5)),
                            )
                        st.dataframe(hr, width='stretch', hide_index=True)
                        csv = hr.to_csv(index=False).encode("utf-8")
                        st.download_button("Download health report (CSV)", csv,
                                            file_name="feature_health.csv", mime="text/csv")
                    except Exception as exc:
                        st.error(f"Health report failed: {exc}")

    # 3) Why did it stop? (no-segments explanation)
    with st.expander("Why did it stop? (no-segments explanation)"):
        if not data_path or not os.path.exists(data_path):
            st.caption("Dataset not available - reload in Module 1 to enable the full diagnostic.")
        else:
            if st.button("Run full diagnostics", key="m4_diag"):
                st.session_state.pop("m4_noseg", None)
                b = _build_diag_builder(cfg, data_path, exp.get("exp_id"))
                if b is not None:
                    st.session_state["m4_noseg"] = b.explain_no_segments()
            if st.session_state.get("m4_noseg"):
                st.code(st.session_state["m4_noseg"], language="text")


def _build_runnable_script(exp, cfg, data_path):
    """Reconstruct a standalone Python script that reproduces the experiment."""
    cfg = cfg or {}
    data_path_str = data_path or "module1_data.duckdb"
    lines = [
        '#!/usr/bin/env python3',
        '"""Runnable script to reproduce this RapidSegment experiment."""',
        '',
        '# 1. Install dependencies (uncomment if needed)',
        '# pip install rapidsegment',
        '',
        'import os',
        'from rapidsegment import StrategicSegmentBuilder',
        '',
        '',
        f'data_path = r"{data_path_str}"',
        '',
        '',
        'builder = StrategicSegmentBuilder(',
        f'    target={cfg.get("target_col")!r},',
        f'    min_sample_size={cfg.get("min_sample_size", 1000)},',
        f'    min_lift={cfg.get("min_lift", 1.5)},',
        f'    min_events={cfg.get("min_events", 100)},',
        f'    max_segments={cfg.get("max_segments", 10)},',
        f'    top_n_vars={cfg.get("top_n_vars", 15)},',
        f'    max_feature_reuse={cfg.get("max_feature_reuse", 1)},',
        f'    enable_diversity={cfg.get("enable_diversity", False)},',
        f'    enable_1way={cfg.get("enable_1way", True)},',
        f'    enable_2way={cfg.get("enable_2way", True)},',
        f'    enable_3way={cfg.get("enable_3way", True)},',
        f'    selection_metric={cfg.get("selection_metric", "iv")!r},',
        f'    binning_method={cfg.get("binning_method", "optimal_cart")!r},',
        f'    naive_bins={cfg.get("naive_bins", 5)},',
        f'    max_expansion_hops={cfg.get("max_expansion_hops", 0)},',
        f'    n_jobs={cfg.get("n_jobs", -1)},',
        f'    sort_priority={cfg.get("sort_priority", "rate_lift_count")!r},',
        ')',
        '',
        '',
        'segments = builder.extract_segments(data_path)',
        '',
        '',
        'print(f"Found {len(segments)} segments.")',
        'for s in segments:',
        '    print(f"  Seg {s[\'segment_id\']}: lift={s[\'lift\"]:.2f}x, '
        'count={s[\'count\']}, rule={s[\'rule_string\"]}")',
    ]
    if cfg.get("primary_key"):
        lines.insert(16, f'    primary_key={cfg["primary_key"]!r},')
    if cfg.get("expand_log_mode"):
        lines.insert(-8, f'    expand_log_mode={cfg["expand_log_mode"]!r},')
    return "\n".join(lines) + "\n"


def render_export_hub(exp, segments, coverage, scorecard, cfg):
    st.subheader("Export Hub")
    res = exp.get("result") or {}
    segs_csv = pd.DataFrame(segments).to_csv(index=False).encode("utf-8") if segments else b""
    cov_csv = pd.DataFrame(coverage).to_csv(index=False).encode("utf-8") if coverage else b""
    cfg_json = json.dumps(exp.get("config") or {}, indent=2).encode("utf-8")
    sql_script = _build_sql_script(segments, coverage, cfg=cfg, exp=exp).encode("utf-8")
    html_report = _build_html_report(exp, res, segments, coverage, scorecard).encode("utf-8")

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Segments (CSV)", segs_csv, file_name=f"segments_{exp.get('exp_id','')}.csv",
                           mime="text/csv")
        st.download_button("Coverage (CSV)", cov_csv, file_name=f"coverage_{exp.get('exp_id','')}.csv",
                           mime="text/csv")
        st.download_button("Config (JSON)", cfg_json, file_name=f"config_{exp.get('exp_id','')}.json",
                           mime="application/json")
    with c2:
        st.download_button("SQL (deployable)", sql_script, file_name=f"segments_{exp.get('exp_id','')}.sql",
                           mime="text/plain")
        st.download_button("Report (HTML)", html_report, file_name=f"report_{exp.get('exp_id','')}.html",
                           mime="text/html")
        st.download_button("Runnable script (.py)",
                           _build_runnable_script(exp, cfg, active_db()).encode("utf-8"),
                           file_name=f"run_{exp.get('exp_id','')}.py", mime="text/x-python")

    if scorecard is not None:
        sc_json = json.dumps(scorecard, indent=2).encode("utf-8")
        st.download_button("Scorecard (JSON)", sc_json, file_name=f"scorecard_{exp.get('exp_id','')}.json",
                           mime="application/json")

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("segments.csv", segs_csv.decode("utf-8", "ignore"))
        zf.writestr("coverage.csv", cov_csv.decode("utf-8", "ignore"))
        zf.writestr("config.json", cfg_json.decode("utf-8", "ignore"))
        zf.writestr("segments.sql", sql_script.decode("utf-8", "ignore"))
        zf.writestr("report.html", html_report.decode("utf-8", "ignore"))
        if scorecard is not None:
            zf.writestr("scorecard.json", sc_json.decode("utf-8", "ignore"))
    st.download_button("Download ALL (ZIP)", zip_buf.getvalue(),
                       file_name=f"rapidsegment_{exp.get('exp_id','')}.zip", mime="application/zip")


# ── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="RapidSegment — Results Dashboard", layout="wide")
apply_cyberpunk_theme()
st.title("RapidSegment — Module 4: Results Dashboard & Visualization")
st.caption("Extracted segments, visualizations, deployable scorecard, and diagnostics.")

exp, source = load_experiment()
if exp is None:
    st.warning("No experiment found. Run an experiment in the Workbench (Module 2) / "
               "Execution Console (Module 3) first, or ensure `suite_data.db` has rows.")
    try:
        st.page_link("pages/2_Workbench.py", label="Go to Workbench", icon="⚙️")
    except Exception:
        st.caption("Navigate to Module 2 from the sidebar.")
    st.stop()

st.caption(f"Source: **{source}** · `{exp.get('exp_id','')}` · "
           f"target=`{exp.get('target_col','')}` · status=`{exp.get('status','')}`")

segments, coverage = load_segments_from_artifacts(exp)
if not segments:
    segments = exp.get("result", {}).get("segments") or []
    coverage = exp.get("result", {}).get("coverage") or []

data_path = active_db()
cfg = exp.get("config") or {}
if not isinstance(cfg, dict):
    cfg = {}

scorecard = None
if segments and data_path and os.path.exists(data_path):
    if st.button("Generate / refresh scorecard", key="m4_score", width='content'):
        with st.spinner("Scoring population (StrategicSegmentScore)..."):
            scorecard = build_scorecard(cfg, data_path, segments)
    else:
        # Try to load a previously saved scorecard artifact
        sp = os.path.join(SUITE_DIR, "artifacts", exp.get("exp_id", ""), "scorecard.json")
        if os.path.exists(sp):
            try:
                with open(sp, "r", encoding="utf-8") as fh:
                    scorecard = json.load(fh)
            except Exception:
                scorecard = None
else:
    if segments and (not data_path or not os.path.exists(data_path)):
        st.info("Dataset (`module1_data.duckdb`) not found - scorecard and feature health "
                 "report require the original data. Reload it in Module 1 to enable them.")

weights = map_weights(scorecard, segments)

st.divider()
render_summary_cards(exp, exp.get("result") or {})

st.subheader("Coverage")
render_coverage_table(segments, coverage, weights)

if segments:
    with st.expander("Expand a segment for full SQL WHERE clause"):
        parts = []
        for s in segments:
            parts.append(f"-- Segment {s['segment_id']} · {s['rule_string']}")
            parts.append(s.get("sql_filter") or "")
            parts.append("")
        st.code("\n".join(parts), language="sql")

st.subheader("Visualizations")
render_visualizations(segments, coverage, scorecard, cfg)

st.subheader("Scorecard")
render_scorecard_json(scorecard)

render_diagnostics(exp, cfg, data_path, segments)

st.divider()
render_export_hub(exp, segments, coverage, scorecard, cfg)

st.divider()
c1, c2 = st.columns(2)
with c1:
    try:
        st.page_link("pages/2_Workbench.py", label="Configure new experiment (Module 2)", icon="⚙️")
    except Exception:
        st.caption("Open the Workbench (Module 2) from the sidebar.")
with c2:
    try:
        st.page_link("pages/3_Execution_Console.py", label="Re-run in Execution Console (Module 3)", icon="🚀")
    except Exception:
        st.caption("Open the Execution Console (Module 3) from the sidebar.")

