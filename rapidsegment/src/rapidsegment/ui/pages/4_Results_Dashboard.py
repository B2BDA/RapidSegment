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

# ── Constants & storage ───────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE) if os.path.basename(_HERE) == "pages" else _HERE
SUITE_DIR = os.path.join(_PROJECT_ROOT, ".rapidsegment_suite")
os.makedirs(SUITE_DIR, exist_ok=True)
DB_FILE = os.path.join(SUITE_DIR, "module1_data.duckdb")
DB_FILE_MOD = os.path.join(SUITE_DIR, "module1_data_modified.duckdb")
SUITE_DB = os.path.join(SUITE_DIR, "suite_data.db")
ARTIFACTS_DIR = os.path.join(SUITE_DIR, "artifacts")


def active_db():
    """Read the materialized *modified* dataset if it exists, else the raw load."""
    return DB_FILE_MOD if os.path.exists(DB_FILE_MOD) else DB_FILE

SEG_COLORS = [
    "#6366f1", "#f59e0b", "#22c55e", "#ef4444", "#3b82f6",
    "#ec4899", "#14b8a6", "#f97316", "#8b5cf6", "#06b6d4",
    "#84cc16", "#eab308", "#a855f7", "#10b981", "#fb7185",
]


# ── Small helpers (consistent with Module 2 / 3) ──────────────────────────────
def rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


def db_query(sql, read_only=True):
    con = duckdb.connect(active_db(), read_only=read_only)
    result = con.execute(sql).df()
    con.close()
    return result


def db_scalar(sql):
    con = duckdb.connect(active_db(), read_only=True)
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


# ── Experiment loading ───────────────────────────────────────────────────────
def load_experiment():
    """Return (exp_dict, source_label). Prefers live session; else latest DB row."""
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
        (
            exp_id, name, created_at, data_rows, data_cols, status,
            execution_time_sec, target_col, primary_key, builder_params,
            segments_count, avg_lift, max_lift, coverage_pct, baseline_rate, error_msg,
        ) = row
        cfg = _jsonable(json.loads(builder_params)) if builder_params else {}
        exp = {
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
        return exp, "suite_data.db (latest)"
    except Exception as exc:
        return None, f"read error: {exc}"


def load_segments_from_artifacts(exp):
    """Recover full segments/coverage from the saved artifact JSON if present."""
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
        return res.get("segments") or [], res.get("coverage") or []
    except Exception:
        return None, None


def load_dataset():
    if not os.path.exists(DB_FILE):
        return None
    try:
        return db_query("SELECT * FROM udl_data")
    except Exception:
        return None


# ── Rule parsing helpers ──────────────────────────────────────────────────────
def segment_variables(rule_string):
    vars_ = []
    for part in str(rule_string).split("&"):
        if "=" in part:
            vars_.append(part.split("=", 1)[0].strip())
    return vars_


def segment_complexity(rule_string):
    return len([p for p in str(rule_string).split("&") if "=" in p])


# ── Scorecard (StrategicSegmentScore) ─────────────────────────────────────────
def build_scorecard(cfg, df, segments):
    """Create per-segment flag columns, score the population, return artifact dict."""
    if not segments or df is None or df.empty:
        return None
    target = cfg.get("target_col")
    if not target or target not in df.columns:
        return None
    seg_cols = [f"seg_{s['segment_id']}" for s in segments]
    case_exprs = ", ".join(
        f"CASE WHEN {s['sql_filter']} THEN 1 ELSE 0 END AS seg_{s['segment_id']}"
        for s in segments
    )
    con = duckdb.connect()
    try:
        con.register("udl", df)
        flags = con.execute(f"SELECT {case_exprs} FROM udl").df()
    finally:
        con.close()
    scored = pd.concat([df.reset_index(drop=True), flags.reset_index(drop=True)], axis=1)

    pk = cfg.get("primary_key") or ""
    if pk not in scored.columns:
        pk = "rs_row_id"
        scored[pk] = range(len(scored))

    try:
        scorer = StrategicSegmentScore(target_col=target, primary_key=pk, segment_cols=seg_cols)
        exp_id = st.session_state.get("experiment", {}).get("exp_id") or "m4"
        art_dir = os.path.join(SUITE_DIR, "artifacts", exp_id)
        os.makedirs(art_dir, exist_ok=True)
        export_path = os.path.join(art_dir, "scorecard.json")
        artifact = scorer.calculate_and_export_weights(scored, export_path=export_path)
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


# ── SQL builder (reuses Module 3 style) ───────────────────────────────────────
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
        lines.append(f"-- Segment {s['segment_id']} . {s['rule_string']}")
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


def render_segments_table(segments, coverage, weights):
    if not segments:
        st.caption("No segments were produced by this experiment.")
        return
    cov_by_seg = {int(r.get("segment")): r for r in (coverage or []) if r.get("segment")}
    rows = []
    for s in segments:
        c = cov_by_seg.get(int(s["segment_id"]), {})
        rows.append({
            "segment_id": s["segment_id"],
            "rule_string": s.get("rule_string", ""),
            "sql_filter": s.get("sql_filter", ""),
            "count": s.get("count", 0),
            "rate": s.get("rate", 0),
            "lift": s.get("lift", 0),
            "capture_rate": c.get("capture_rate", s.get("capture_rate", 0)),
            "weight": weights.get(s["segment_id"], 0),
            "meta_applied_sample_size": s.get("meta_applied_sample_size", ""),
            "meta_applied_min_lift": s.get("meta_applied_min_lift", ""),
        })
    df = pd.DataFrame(rows)
    for col in ("rate", "lift", "capture_rate"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    st.dataframe(df, width='stretch', hide_index=True)
    with st.expander("Expand a segment for full SQL WHERE clause"):
        for s in segments:
            st.markdown(f"**Segment {s['segment_id']}** - `{s.get('rule_string','')}`")
            st.code(s.get("sql_filter", ""), language="sql")


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


def _build_diag_builder(cfg, df, exp_id=None):
    """Return a builder with populated diagnostics_ for the explain_* methods.

    Preferred path: reuse diagnostics_ persisted by Module 3 into
    artifacts/<exp_id>/result.json — no re-extraction (fast, and works even
    without the dataset loaded). Falls back to re-running extract_segments once
    (cached on session state) only when no persisted diagnostics are available.
    """
    cached = st.session_state.get("m4_diag_builder")
    if cached is not None:
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
                    st.session_state["m4_diag_builder"] = b
                    return b
            except Exception:
                pass

    # 2) Fallback: re-extract once (cached)
    if df is None or df.empty:
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
        b.extract_segments(df)
    st.session_state["m4_diag_builder"] = b
    return b


def generate_feature_health_local(df, features, target, type_overrides=None, naive_bins=5):
    """Corrected feature health report (replaces the library version).

    Fixes two bugs in StrategicSegmentBuilder.generate_feature_health_report:
      * UI categorical overrides are respected — a column marked CATEGORICAL
        is binned by distinct value instead of being NTILE'd as numeric (the
        library decides numeric/categorical solely from the DuckDB column type,
        so an overridden-categorical numeric column was wrongly quantile-binned).
      * Numeric bin labels are made unique via the tile index
        ('Bin 1: [a, b]'), so adjacent/low-cardinality tiles never appear
        'repeated' after rounding collapses the [min,max] label.
    """
    if not features:
        return pd.DataFrame()
    type_overrides = type_overrides or {}
    con = duckdb.connect(":memory:")
    con.register("input_data_view", df)
    con.execute("CREATE TABLE input_df AS SELECT * FROM input_data_view")
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


def render_diagnostics(exp, cfg, df, segments):
    st.subheader("Diagnostic Drilldown")
    res = exp.get("result") or {}
    stop = res.get("stop_reason")
    if stop:
        st.info(f"**Stop reason:** {stop}")
    else:
        st.caption("No stop reason recorded.")

    # 1) Feature Journey — dedicated space
    with st.expander("Feature Journey (audit trail per feature)", expanded=True):
        feats = sorted({v for s in segments for v in segment_variables(s.get("rule_string", ""))})
        if not feats:
            st.caption("No features were used (no segments produced).")
        else:
            fj = st.selectbox("Choose a feature to trace", feats, key="m4_fj")
            if st.button("Show feature journey", key="m4_fj_btn"):
                if df is None or df.empty:
                    st.warning("Dataset not available - load it in Module 1 to enable diagnostics.")
                else:
                    b = _build_diag_builder(cfg, df, exp.get("exp_id"))
                    if b is not None:
                        buf = io.StringIO()
                        with contextlib.redirect_stdout(buf):
                            b.explain_feature_journey(fj)
                        st.code(buf.getvalue() or "(no journey recorded for this feature)", language="text")

    # 2) Feature Health Report
    with st.expander("Feature Health Report (bin-level stats)"):
        feats = sorted({v for s in segments for v in segment_variables(s.get("rule_string", ""))})
        if not feats:
            st.caption("No features to profile (no segments).")
        else:
            sel = st.multiselect("Features to profile", feats, default=feats[:5], key="m4_health_sel")
            if st.button("Generate health report", key="m4_health"):
                if df is None or df.empty:
                    st.warning("Dataset not available - load data via Module 1 first.")
                else:
                    try:
                        type_overrides = st.session_state.get("type_overrides") or {}
                        with st.spinner("Profiling features..."):
                            hr = generate_feature_health_local(
                                df, list(sel), cfg.get("target_col", ""),
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
        if df is None or df.empty:
            st.caption("Dataset not available - reload in Module 1 to enable the full diagnostic.")
        else:
            if st.button("Run full diagnostics", key="m4_diag"):
                b = _build_diag_builder(cfg, df, exp.get("exp_id"))
                if b is not None:
                    st.session_state["m4_noseg"] = b.explain_no_segments()
            if st.session_state.get("m4_noseg"):
                st.code(st.session_state["m4_noseg"], language="text")


def render_export_hub(exp, segments, coverage, scorecard, cfg):
    st.subheader("Export Hub")
    res = exp.get("result") or {}
    segs_csv = pd.DataFrame(segments).to_csv(index=False).encode("utf-8") if segments else b""
    cov_csv = pd.DataFrame(coverage).to_csv(index=False).encode("utf-8") if coverage else b""
    cfg_json = json.dumps(exp.get("config") or {}, indent=2).encode("utf-8")
    sql_script = _build_sql_script(segments, coverage, cfg=cfg, exp=exp).encode("utf-8")
    html_report = _build_html_report(exp, res, segments, coverage, scorecard).encode("utf-8")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.download_button("Segments (CSV)", segs_csv, file_name=f"segments_{exp.get('exp_id','')}.csv",
                       mime="text/csv", width='stretch')
    c2.download_button("Coverage (CSV)", cov_csv, file_name=f"coverage_{exp.get('exp_id','')}.csv",
                       mime="text/csv", width='stretch')
    c3.download_button("Config (JSON)", cfg_json, file_name=f"config_{exp.get('exp_id','')}.json",
                       mime="application/json", width='stretch')
    c4.download_button("SQL (deployable)", sql_script, file_name=f"segments_{exp.get('exp_id','')}.sql",
                       mime="text/plain", width='stretch')
    c5.download_button("Report (HTML)", html_report, file_name=f"report_{exp.get('exp_id','')}.html",
                       mime="text/html", width='stretch')

    if scorecard is not None:
        sc_json = json.dumps(scorecard, indent=2).encode("utf-8")
        st.download_button("Scorecard (JSON)", sc_json, file_name=f"scorecard_{exp.get('exp_id','')}.json",
                           mime="application/json", width='stretch')

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
                       file_name=f"rapidsegment_{exp.get('exp_id','')}.zip", mime="application/zip",
                       width='stretch')


# ── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="RapidSegment — Results Dashboard", layout="wide")
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

df = load_dataset()
cfg = exp.get("config") or {}
if not isinstance(cfg, dict):
    cfg = {}

scorecard = None
if segments and df is not None and not df.empty:
    if st.button("Generate / refresh scorecard", key="m4_score", width='content'):
        with st.spinner("Scoring population (StrategicSegmentScore)..."):
            scorecard = build_scorecard(cfg, df, segments)
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
    if segments and (df is None or df.empty):
        st.info("Dataset (`module1_data.duckdb`) not found - scorecard and feature health "
                "report require the original data. Reload it in Module 1 to enable them.")

weights = map_weights(scorecard, segments)

st.divider()
render_summary_cards(exp, exp.get("result") or {})

st.subheader("Segments")
render_segments_table(segments, coverage, weights)

st.subheader("Visualizations")
render_visualizations(segments, coverage, scorecard, cfg)

st.subheader("Scorecard")
render_scorecard_json(scorecard)

render_diagnostics(exp, cfg, df, segments)

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

