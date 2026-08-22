"""
RapidSegment — Module 5: Enhanced Leaderboard (Experiment Tracking)
────────────────────────────────────────────────────────────────────
Reads experiments from `suite_data.db`, ranks them, shows inline
sparklines per experiment, offers row-level actions (clone to workbench,
view results, compare, duplicate, delete, export), and summary stats.

Run from the app root:  streamlit run app.py  (then open "5 · Leaderboard")
Standalone mirror:  Module_5_leaderboard.py
"""

import json
import os
import uuid
from datetime import datetime

import duckdb
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Paths (must match modules 1–4) ──────────────────────────────────────────
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


# ── Data access ──────────────────────────────────────────────────────────────
def read_all_experiments():
    """Return all experiments as a list of dicts (newest first)."""
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
                ", ".join(EXP_COLS)
            )
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


def delete_experiment(exp_id):
    con = duckdb.connect(SUITE_DB)
    con.execute("DELETE FROM experiments WHERE exp_id = ?", [exp_id])
    con.close()


def duplicate_experiment(exp):
    new_id = uuid.uuid4().hex
    created = datetime.now()
    con = duckdb.connect(SUITE_DB)
    con.execute(
        """
        INSERT OR REPLACE INTO experiments
        (exp_id, name, created_at, data_rows, data_cols, status,
         execution_time_sec, target_col, primary_key, builder_params,
         segments_count, avg_lift, max_lift, coverage_pct, baseline_rate, error_msg)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CAST(? AS JSON), ?, ?, ?, ?, ?, ?)
        """,
        [
            new_id, f"{exp['name']} (copy)", created,
            exp["data_rows"], exp["data_cols"], exp["status"],
            exp["execution_time_sec"], exp["target_col"], exp["primary_key"],
            json.dumps(_jsonable(exp["config"])),
            exp["segments_count"], exp["avg_lift"], exp["max_lift"],
            exp["coverage_pct"], exp["baseline_rate"], exp["error_msg"],
        ],
    )
    con.close()
    return new_id


def load_full_experiment(exp_id):
    """Build a full experiment dict (DB row + recovered result.json) for
    handing off to Module 4's 'View Results'."""
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


def clone_to_workbench(exp):
    st.session_state["wb_pending"] = _jsonable(exp["config"])
    st.switch_page("pages/2_Workbench.py")


def view_results(exp_id):
    st.session_state["experiment"] = load_full_experiment(exp_id)
    st.switch_page("pages/4_Results_Dashboard.py")


def param_diff(cfg_a, cfg_b):
    """Return list of (key, val_a, val_b) for keys whose values differ."""
    keys = sorted(set((cfg_a or {})) | set((cfg_b or {})))
    diffs = []
    for k in keys:
        a, b = (cfg_a or {}).get(k), (cfg_b or {}).get(k)
        if a != b:
            diffs.append((k, a, b))
    return diffs


def most_used_params(rows):
    """Aggregate config fields across experiments for the summary panel."""
    agg = {}
    for r in rows:
        cfg = r.get("config") or {}
        for k, v in cfg.items():
            agg.setdefault(k, {})
            agg[k][v] = agg[k].get(v, 0) + 1
    out = {}
    for k, counts in agg.items():
        top_val, top_n = max(counts.items(), key=lambda kv: kv[1])
        out[k] = (top_val, top_n, len(rows))
    return out


# ── Page rendering ────────────────────────────────────────────────────────────
def _sparkline(exp):
    labels = ["Avg Lift", "Max Lift", "Coverage %"]
    vals = [float(exp["avg_lift"] or 0), float(exp["max_lift"] or 0),
            float(exp["coverage_pct"] or 0)]
    fig = go.Figure(go.Bar(x=labels, y=vals,
                           marker_color=["#58a6ff", "#3fb950", "#d29922"]))
    fig.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10),
                     yaxis_title="value", template="plotly_white")
    return fig


def _summary_cards(rows):
    total = len(rows)
    avg_time = (sum(float(r["execution_time_sec"] or 0) for r in rows) / total
                if total else 0.0)
    best = max(rows, key=lambda r: float(r["avg_lift"] or 0)) if rows else None
    used = most_used_params(rows)
    bm = used.get("binning_method")
    bm_txt = f"{bm[0]} ({bm[1]}/{bm[2]})" if bm else "—"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Experiments", total)
    c2.metric("Avg extraction time", f"{avg_time:.1f}s")
    c3.metric("Best avg lift",
              f"{float(best['avg_lift'] or 0):.2f}×" if best else "—",
              help=(best["name"] if best else ""))
    c4.metric("Top binning method", bm_txt)


def main():
    st.title("5 · Leaderboard — Experiment Tracking")
    st.caption("Rank, filter and audit every segmentation run. Clone any "
               "config back to the Workbench, drill into Results, or compare runs.")

    rows = read_all_experiments()

    # Sidebar filters
    with st.sidebar:
        st.header("Filters")
        q = st.text_input("Search name", "")
        status_opts = sorted({r["status"] for r in rows}) if rows else []
        statuses = st.multiselect("Status", status_opts, default=status_opts)
        min_lift = st.slider("Min avg lift", 0.0, 5.0, 0.0, 0.1)
        if rows:
            dates = [pd.to_datetime(r["created_at"]) for r in rows]
            dmin, dmax = min(dates).date(), max(dates).date()
            rng = st.date_input("Date range", (dmin, dmax))
            f_dmin = pd.to_datetime(rng[0]) if len(rng) == 2 else None
            f_dmax = pd.to_datetime(rng[1]) if len(rng) == 2 else None
        else:
            f_dmin = f_dmax = None

    # Apply filters
    if rows:
        filtered = [r for r in rows
                    if q.lower() in r["name"].lower()
                    and r["status"] in statuses
                    and float(r["avg_lift"] or 0) >= min_lift]
        if f_dmin and f_dmax:
            filtered = [r for r in filtered
                        if f_dmin <= pd.to_datetime(r["created_at"]) <= f_dmax]
        rows_view = filtered
    else:
        rows_view = []

    if not rows:
        st.info("No experiments yet. Run a segmentation in the Workbench to "
                "populate the leaderboard.")
        st.page_link("pages/2_Workbench.py", label="Go to Workbench →",
                     icon="⚙️")
        return

    _summary_cards(rows_view)

    # Ranked grid
    st.subheader("Ranked Experiments")
    grid = pd.DataFrame([{
        "Rank": i + 1,
        "Name": r["name"],
        "Created": str(r["created_at"])[:19],
        "Rows": int(r["data_rows"] or 0),
        "Cols": int(r["data_cols"] or 0),
        "Segments": int(r["segments_count"] or 0),
        "Avg Lift": round(float(r["avg_lift"] or 0), 3),
        "Max Lift": round(float(r["max_lift"] or 0), 3),
        "Coverage %": round(float(r["coverage_pct"] or 0), 2),
        "Status": r["status"],
    } for i, r in enumerate(rows_view)])
    if not grid.empty:
        st.dataframe(grid, use_container_width=True, hide_index=True)
    else:
        st.warning("No experiments match the current filters.")
        return

    # Inspect / actions
    st.subheader("Inspect & Actions")
    labels = {r["exp_id"]: f"{r['name']}  ({str(r['created_at'])[:19]})"
              for r in rows_view}
    sel = st.selectbox("Select experiment", list(labels.keys()),
                       format_func=lambda x: labels[x])
    exp = next(r for r in rows_view if r["exp_id"] == sel)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Avg Lift", f"{float(exp['avg_lift'] or 0):.2f}×")
    m2.metric("Max Lift", f"{float(exp['max_lift'] or 0):.2f}×")
    m3.metric("Coverage", f"{float(exp['coverage_pct'] or 0):.1f}%")
    m4.metric("Segments", int(exp["segments_count"] or 0))
    m5.metric("Status", exp["status"])
    st.plotly_chart(_sparkline(exp), use_container_width=True)

    b1, b2, b3, b4, b5 = st.columns(5)
    if b1.button("Clone to Workbench", key="m5_clone", use_container_width=True):
        clone_to_workbench(exp)
    if b2.button("View Results", key="m5_view", use_container_width=True):
        view_results(exp["exp_id"])
    cfg_json = json.dumps(_jsonable(exp["config"]), indent=2)
    b3.download_button("Export Config", cfg_json,
                       file_name=f"{exp['exp_id']}_config.json",
                       mime="application/json", use_container_width=True)
    if b4.button("Duplicate", key="m5_dup", use_container_width=True):
        duplicate_experiment(exp)
        st.rerun()
    if b5.button("Delete", key="m5_del", use_container_width=True,
                 type="primary"):
        delete_experiment(exp["exp_id"])
        st.success("Experiment deleted.")
        st.rerun()

    # Compare (lightweight face-off; full Arena is Module 6)
    st.subheader("Compare Two Runs")
    c_ids = list(labels.keys())
    ca, cb = st.columns(2)
    a_id = ca.selectbox("Run A", c_ids, format_func=lambda x: labels[x],
                        key="m5_a")
    b_id = cb.selectbox("Run B", c_ids, index=min(1, len(c_ids) - 1),
                        format_func=lambda x: labels[x], key="m5_b")
    ea = next(r for r in rows if r["exp_id"] == a_id)
    eb = next(r for r in rows if r["exp_id"] == b_id)
    if a_id != b_id:
        cmp = pd.DataFrame([
            {"Metric": "Avg Lift", "Run A": round(float(ea["avg_lift"] or 0), 3),
             "Run B": round(float(eb["avg_lift"] or 0), 3)},
            {"Metric": "Max Lift", "Run A": round(float(ea["max_lift"] or 0), 3),
             "Run B": round(float(eb["max_lift"] or 0), 3)},
            {"Metric": "Coverage %", "Run A": round(float(ea["coverage_pct"] or 0), 2),
             "Run B": round(float(eb["coverage_pct"] or 0), 2)},
            {"Metric": "Segments", "Run A": int(ea["segments_count"] or 0),
             "Run B": int(eb["segments_count"] or 0)},
            {"Metric": "Rows", "Run A": int(ea["data_rows"] or 0),
             "Run B": int(eb["data_rows"] or 0)},
            {"Metric": "Time (s)", "Run A": round(float(ea["execution_time_sec"] or 0), 1),
             "Run B": round(float(eb["execution_time_sec"] or 0), 1)},
        ])
        st.dataframe(cmp, use_container_width=True, hide_index=True)
        diffs = param_diff(ea["config"], eb["config"])
        if diffs:
            st.write("**Parameter differences**")
            st.dataframe(
                pd.DataFrame(
                    [{"Parameter": k, "Run A": str(v), "Run B": str(w)}
                     for k, v, w in diffs]),
                use_container_width=True, hide_index=True)
        else:
            st.success("Configs are identical.")
    else:
        st.info("Pick two different runs to compare.")


if __name__ == "__main__":
    main()
