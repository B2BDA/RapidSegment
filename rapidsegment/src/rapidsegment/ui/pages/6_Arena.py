"""
RapidSegment — Module 6: Arena (1v1 Comparison)
────────────────────────────────────────────────
Deep side-by-side comparison of two experiments: KPI face-off with winners,
full parameter diff (differing fields highlighted), segment overlap analysis
with overlaid lift distributions, and a SQL diff of matching segments.

Standalone mirror:  Module_6_arena.py   ·   page: pages/6_Arena.py
"""

import json
import os
from datetime import datetime

import duckdb
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Paths (must match modules 1–5) ──────────────────────────────────────────
SUITE_DIR = os.path.join(os.getcwd(), ".rapidsegment_suite")
os.makedirs(SUITE_DIR, exist_ok=True)
SUITE_DB = os.path.join(SUITE_DIR, "suite_data.db")
ARTIFACTS_DIR = os.path.join(SUITE_DIR, "artifacts")

EXP_COLS = [
    "exp_id", "name", "created_at", "data_rows", "data_cols", "status",
    "execution_time_sec", "target_col", "primary_key", "builder_params",
    "segments_count", "avg_lift", "max_lift", "coverage_pct",
    "baseline_rate", "error_msg",
]


def _jsonable(v):
    if isinstance(v, dict):
        return {k: _jsonable(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def read_all_experiments():
    if not os.path.exists(SUITE_DB):
        return []
    try:
        con = duckdb.connect(SUITE_DB, read_only=True)
        has = con.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name='experiments'"
        ).fetchone()
        if not has:
            con.close()
            return []
        rows = con.execute(
            "SELECT {} FROM experiments ORDER BY created_at DESC".format(
                ", ".join(EXP_COLS))
        ).fetchall()
        con.close()
        out = []
        for r in rows:
            d = dict(zip(EXP_COLS, r))
            try:
                d["config"] = _jsonable(json.loads(d["builder_params"])) \
                    if d.get("builder_params") else {}
            except Exception:
                d["config"] = {}
            out.append(d)
        return out
    except Exception as exc:
        st.error(f"Failed to read experiments: {exc}")
        return []


def load_full_experiment(exp_id):
    rows = read_all_experiments()
    exp = next((e for e in rows if e["exp_id"] == exp_id), None)
    if not exp:
        return None
    result = {
        "segments_count": int(exp["segments_count"] or 0),
        "avg_lift": float(exp["avg_lift"] or 0),
        "max_lift": float(exp["max_lift"] or 0),
        "coverage_pct": float(exp["coverage_pct"] or 0),
        "baseline_rate_pct": float(exp["baseline_rate"] or 0),
        "error_msg": exp["error_msg"],
        "segments": [], "coverage": [], "stop_reason": None,
    }
    art = os.path.join(ARTIFACTS_DIR, exp_id, "result.json")
    if os.path.exists(art):
        try:
            with open(art, "r", encoding="utf-8") as fh:
                saved = json.load(fh)
            res = saved.get("result") or {}
            result["segments"] = res.get("segments") or []
            result["coverage"] = res.get("coverage") or []
            result["stop_reason"] = res.get("stop_reason")
        except Exception:
            pass
    return {
        "exp_id": exp["exp_id"], "name": exp["name"],
        "created_at": str(exp["created_at"]), "status": exp["status"],
        "execution_time_sec": float(exp["execution_time_sec"] or 0),
        "target_col": exp["target_col"], "primary_key": exp["primary_key"] or "",
        "data_rows": int(exp["data_rows"] or 0),
        "data_cols": int(exp["data_cols"] or 0),
        "config": exp["config"], "result": result, "logs": [],
    }


def param_diff(cfg_a, cfg_b):
    keys = sorted(set((cfg_a or {})) | set((cfg_b or {})))
    return [(k, (cfg_a or {}).get(k), (cfg_b or {}).get(k))
            for k in keys if (cfg_a or {}).get(k) != (cfg_b or {}).get(k)]


# ── Page rendering ────────────────────────────────────────────────────────────
def _winner(a, b, higher_better=True):
    if a == b:
        return "tie"
    good_a = (a > b) if higher_better else (a < b)
    return "A 🏆" if good_a else "B 🏆"


def _kpi_faceoff(ea, eb):
    rows = [
        ("Avg Lift", float(ea["avg_lift"] or 0), float(eb["avg_lift"] or 0), True),
        ("Max Lift", float(ea["max_lift"] or 0), float(eb["max_lift"] or 0), True),
        ("Coverage %", float(ea["coverage_pct"] or 0), float(eb["coverage_pct"] or 0), True),
        ("Segments", int(ea["segments_count"] or 0), int(eb["segments_count"] or 0), True),
        ("Data Rows", int(ea["data_rows"] or 0), int(eb["data_rows"] or 0), True),
        ("Exec Time (s)", float(ea["execution_time_sec"] or 0),
         float(eb["execution_time_sec"] or 0), False),
    ]
    df = pd.DataFrame([
        {"Metric": m, "Run A": round(av, 3), "Run B": round(bv, 3),
         "Winner": _winner(av, bv, hb)} for m, av, bv, hb in rows])
    st.dataframe(df, use_container_width=True, hide_index=True)


def _param_diff_view(ca, cb):
    keys = sorted(set((ca or {})) | set((cb or {})))
    df = pd.DataFrame([
        {"Parameter": k, "Run A": str((ca or {}).get(k)),
         "Run B": str((cb or {}).get(k)),
         "Different?": "YES" if (ca or {}).get(k) != (cb or {}).get(k) else ""}
        for k in keys])
    st.dataframe(df, use_container_width=True, hide_index=True)


def _segment_comparison(segs_a, segs_b):
    if not segs_a and not segs_b:
        st.info("No segment data available for these runs (artifacts missing).")
        return
    rules_a = {s.get("rule_string", ""): s for s in (segs_a or [])}
    rules_b = {s.get("rule_string", ""): s for s in (segs_b or [])}
    set_a, set_b = set(rules_a), set(rules_b)
    shared = set_a & set_b
    uniq_a = set_a - set_b
    uniq_b = set_b - set_a
    union = set_a | set_b
    jac = len(shared) / len(union) if union else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Shared segments", len(shared))
    c2.metric("Unique to A", len(uniq_a))
    c3.metric("Unique to B", len(uniq_b))
    c4.metric("Jaccard overlap", f"{jac:.2f}")

    # Overlaid lift distribution
    fig = go.Figure()
    if segs_a:
        fig.add_trace(go.Scatter(
            y=[float(s.get("lift") or 0) for s in segs_a],
            mode="markers", name="Run A", marker=dict(color="#58a6ff")))
    if segs_b:
        fig.add_trace(go.Scatter(
            y=[float(s.get("lift") or 0) for s in segs_b],
            mode="markers", name="Run B", marker=dict(color="#3fb950")))
    fig.update_layout(title="Segment Lift Distribution",
                      yaxis_title="lift", xaxis_title="segment index",
                      height=360, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    if shared:
        st.write("**Shared segments — lift by run**")
        st.dataframe(pd.DataFrame([
            {"Rule": r, "A lift": round(float(rules_a[r].get("lift") or 0), 3),
             "B lift": round(float(rules_b[r].get("lift") or 0), 3),
             "Δ lift": round(float(rules_b[r].get("lift") or 0) -
                             float(rules_a[r].get("lift") or 0), 3)}
            for r in sorted(shared)]), use_container_width=True, hide_index=True)
    else:
        st.caption("No shared segment rules between these runs.")


def _sql_diff(segs_a, segs_b):
    rules_a = {s.get("rule_string", ""): s for s in (segs_a or [])}
    rules_b = {s.get("rule_string", ""): s for s in (segs_b or [])}
    all_rules = sorted(set(rules_a) | set(rules_b))
    if not all_rules:
        st.info("No segment SQL available for these runs.")
        return
    df = pd.DataFrame([
        {"Rule": r,
         "Run A SQL": (rules_a.get(r, {}) or {}).get("sql_filter", ""),
         "Run B SQL": (rules_b.get(r, {}) or {}).get("sql_filter", "")}
        for r in all_rules])
    st.dataframe(df, use_container_width=True, hide_index=True)


def main():
    st.title("6 · Arena — 1v1 Comparison")
    st.caption("Pick two experiments and compare them head-to-head across "
               "KPIs, parameters, segments and generated SQL.")

    rows = read_all_experiments()
    if len(rows) < 1:
        st.info("No experiments found. Run a segmentation in the Workbench first.")
        st.page_link("pages/2_Workbench.py", label="Go to Workbench →", icon="⚙️")
        return

    labels = {r["exp_id"]: f"{r['name']}  ({str(r['created_at'])[:19]})"
              for r in rows}
    c1, c2 = st.columns(2)
    a_id = c1.selectbox("Run A", list(labels.keys()),
                        format_func=lambda x: labels[x], key="m6_a")
    b_id = c2.selectbox("Run B", list(labels.keys()),
                        index=min(1, len(labels) - 1),
                        format_func=lambda x: labels[x], key="m6_b")

    ea = next(r for r in rows if r["exp_id"] == a_id)
    eb = next(r for r in rows if r["exp_id"] == b_id)
    full_a = load_full_experiment(a_id) or {"result": {}}
    full_b = load_full_experiment(b_id) or {"result": {}}
    segs_a = (full_a.get("result") or {}).get("segments") or []
    segs_b = (full_b.get("result") or {}).get("segments") or []

    tabs = st.tabs(["KPI Face-off", "Parameter Diff", "Segment Comparison", "SQL Diff"])
    with tabs[0]:
        st.subheader(f"{ea['name']}  vs  {eb['name']}")
        _kpi_faceoff(ea, eb)
    with tabs[1]:
        _param_diff_view(ea["config"], eb["config"])
    with tabs[2]:
        _segment_comparison(segs_a, segs_b)
    with tabs[3]:
        _sql_diff(segs_a, segs_b)


if __name__ == "__main__":
    main()
