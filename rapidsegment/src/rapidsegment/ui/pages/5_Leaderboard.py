"""
RapidSegment — Module 5: Leaderboard (Best Experiment per Dataset)
──────────────────────────────────────────────────────────────────
For a single dataset, ranks every experiment by a chosen performance KPI
(avg lift, max lift, coverage, segment count) and highlights the best
performer. Reads from `suite_data.db`, offers row-level actions (clone to
workbench, view results, compare, duplicate, delete, export) and summary
stats.

Run from the app root:  streamlit run app.py  (then open "5 · Leaderboard")
Standalone mirror:  Module_5_leaderboard.py
"""

import json
import os
import shutil
import uuid
from datetime import datetime

import duckdb
import pandas as pd
import streamlit as st
from rapidsegment.ui._theme import apply_cyberpunk_theme

# ── Paths (must match modules 1–4) ──────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE) if os.path.basename(_HERE) == "pages" else _HERE
SUITE_DIR = os.path.join(_PROJECT_ROOT, ".rapidsegment_suite")
os.makedirs(SUITE_DIR, exist_ok=True)
SUITE_DB = os.path.join(SUITE_DIR, "suite_data.db")
ARTIFACTS_DIR = os.path.join(SUITE_DIR, "artifacts")

EXP_COLS = [
    "exp_id", "name", "created_at", "data_rows", "data_cols", "status",
    "execution_time_sec", "target_col", "primary_key", "builder_params",
    "segments_count", "avg_lift", "max_lift", "coverage_pct", "cumulative_event_capture",
    "baseline_rate", "error_msg", "dataset_name",
]

# Ranking KPIs the user can pick from
KPI_OPTS = {
    "Avg Lift ×": "avg_lift",
    "Max Lift ×": "max_lift",
    "Coverage %": "coverage_pct",
    "Segments": "segments_count",
    "Cumulative Event Capture %": "cumulative_event_capture",
}

apply_cyberpunk_theme()


# ── Column coercion helpers (DB gives Decimal/None) ──────────────────────────
def _s(x):
    return str(x) if x is not None else ""


def _i(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _b(x):
    return bool(x)


def _dataset_name(r):
    """The dataset a run belongs to (set in Module 1)."""
    return _s(r.get("dataset_name")) or "(unnamed)"


# ── Read ─────────────────────────────────────────────────────────────────────
def read_all_experiments():
    """Read every experiment row. Mirrors the columns written by Module 3."""
    st.session_state.pop("m5_read_error", None)
    if not os.path.exists(SUITE_DB):
        return []
    try:
        con = duckdb.connect(SUITE_DB, read_only=False)
        try:
            con.execute("ALTER TABLE experiments ADD COLUMN dataset_name TEXT")
        except Exception:
            pass
        try:
            con.execute("ALTER TABLE experiments ADD COLUMN cumulative_event_capture DOUBLE")
        except Exception:
            pass
        rows = con.execute(
            f"SELECT {','.join(EXP_COLS)} FROM experiments ORDER BY created_at DESC"
        ).fetchall()
        con.close()
    except Exception as e:  # surface, don't crash the page
        st.session_state["m5_read_error"] = str(e)
        return []

    out = []
    for r in rows:
        d = dict(zip(EXP_COLS, r))
        for k in ("builder_params", "error_msg"):
            if isinstance(d.get(k), str) and d[k]:
                try:
                    d[k] = json.loads(d[k])
                except Exception:
                    pass
        out.append(d)
    return out


def load_full_experiment(exp_id):
    p = os.path.join(ARTIFACTS_DIR, exp_id, "result.json")
    if not os.path.exists(p):
        st.error(f"No artifacts for `{exp_id}`.")
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Could not load run artifacts: {e}")
        return None


# ── Write / mutate ────────────────────────────────────────────────────────────
def delete_experiment(exp_id):
    con = duckdb.connect(SUITE_DB)
    con.execute("DELETE FROM experiments WHERE exp_id = ?", [exp_id])
    con.close()
    art = os.path.join(ARTIFACTS_DIR, exp_id)
    if os.path.isdir(art):
        shutil.rmtree(art, ignore_errors=True)


def duplicate_experiment(exp_id):
    full = load_full_experiment(exp_id)
    if not full:
        return
    new_id = str(uuid.uuid4())
    row = dict(full)
    row["exp_id"] = new_id
    row["name"] = f"{full.get('name', 'exp')} (copy)"
    row["created_at"] = datetime.now().isoformat()
    con = duckdb.connect(SUITE_DB)
    placeholders = ",".join(["?"] * len(EXP_COLS))
    con.execute(
        f"INSERT OR REPLACE INTO experiments ({','.join(EXP_COLS)}) VALUES ({placeholders})",
        [row.get(c) for c in EXP_COLS],
    )
    con.close()
    os.makedirs(os.path.join(ARTIFACTS_DIR, new_id), exist_ok=True)
    src = os.path.join(ARTIFACTS_DIR, exp_id, "result.json")
    dst = os.path.join(ARTIFACTS_DIR, new_id, "result.json")
    if os.path.exists(src):
        shutil.copy(src, dst)
    st.success(f"Duplicated → **{row['name']}**")


def clone_to_workbench(exp_id):
    full = load_full_experiment(exp_id)
    if not full:
        return
    cfg = dict(full.get("config") or {})
    cfg["exp_name"] = f"Clone-{full.get('name', 'exp')}"
    st.session_state["m2_exp_cfg"] = cfg
    st.session_state["m2_just_cloned"] = True
    st.success("Cloned into **Module 2 · Workbench**. Switch there to run it.")


def view_results(exp_id):
    st.session_state["m4_view_exp"] = exp_id
    st.success("Open **Module 4 · Results Dashboard** → choose 'View a saved run'.")


def export_run(exp_id):
    full = load_full_experiment(exp_id)
    if not full:
        return
    st.download_button(
        "Download run JSON",
        data=json.dumps(full, indent=2, default=str),
        file_name=f"{exp_id}.json",
        mime="application/json",
        key=f"exp_{exp_id}",
    )


def param_diff(cfg_a, cfg_b):
    keys = sorted(set(cfg_a) | set(cfg_b))
    return [(k, cfg_a.get(k), cfg_b.get(k)) for k in keys if cfg_a.get(k) != cfg_b.get(k)]


# ── Summary ───────────────────────────────────────────────────────────────────
def _summary_cards(rows, sig, best, kpi_label):
    completed = [r for r in rows if r["status"] == "completed"]
    n_comp = len(completed)
    best_lift = max((_f(r.get("avg_lift")) for r in completed), default=0.0)
    best_cov = max((_f(r.get("coverage_pct")) for r in completed), default=0.0)
    cols = st.columns(4)
    cols[0].metric("Experiments", len(rows), help=f"Dataset: {sig}")
    cols[1].metric("Completed", n_comp)
    cols[2].metric("Best avg lift", f"{best_lift:.2f}×")
    cols[3].metric("Best coverage", f"{best_cov:.1f}%")
    if best:
        st.success(
            f"🏆 **Best performer** ({kpi_label}): **{best['name']}** "
            f"— avg lift {_f(best.get('avg_lift')):.2f}×, "
            f"max lift {_f(best.get('max_lift')):.2f}×, "
            f"coverage {_f(best.get('coverage_pct')):.1f}%, "
            f"{_i(best.get('segments_count'))} segments."
        )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    st.title("🏆 Leaderboard — Best Experiment per Dataset")
    st.caption(
        "Rank every experiment run on a dataset by a performance KPI and "
        "spot the winner. No date filter — every saved run counts."
    )

    rows = read_all_experiments()
    read_err = st.session_state.get("m5_read_error")
    if read_err:
        st.error(f"Could not read the experiment database: `{read_err}`")
        st.caption(f"DB path: `{SUITE_DB}`")

    if not rows:
        st.info(
            "No experiments yet. Run a configuration in **Module 3 · Execution "
            "Console** — it is saved automatically and shows up here."
        )
        return

    # Group by dataset name (set in Module 1)
    groups: dict = {}
    for r in rows:
        groups.setdefault(_dataset_name(r), []).append(r)

    ds_list = sorted(groups, key=lambda s: (-len(groups[s]), s))
    default_idx = 0
    sel = st.selectbox(
        "Dataset",
        options=["All datasets"] + ds_list,
        index=default_idx,
        help="Experiments are grouped by the dataset name set in Module 1.",
    )
    view_rows = rows if sel == "All datasets" else groups[sel]

    # Ranking control
    kpi_label = st.radio(
        "Rank by performance KPI", list(KPI_OPTS.keys()),
        horizontal=True, index=0,
    )
    kpi_key = KPI_OPTS[kpi_label]

    # Light filters (no date picker)
    col_f, col_s = st.columns([1, 2])
    with col_f:
        statuses = sorted({r["status"] for r in view_rows})
        sel_status = st.multiselect("Status", statuses, default=statuses)
    with col_s:
        q = st.text_input("Search by name", placeholder="experiment name…")

    filtered = [
        r for r in view_rows
        if r["status"] in sel_status and q.lower() in _s(r["name"]).lower()
    ]

    if not filtered:
        st.warning("No experiments match the current filters.")
        return

    # Best = top *completed* run by the chosen KPI
    completed = [r for r in filtered if r["status"] == "completed"]
    best = None
    if completed:
        best = max(completed, key=lambda r: _f(r.get(kpi_key)) or 0.0)

    _summary_cards(filtered, sel, best, kpi_label)

    # Ranked grid (completed first, then by KPI desc)
    ranked = sorted(
        filtered,
        key=lambda r: (r["status"] != "completed", -(_f(r.get(kpi_key)) or 0.0)),
    )
    grid = pd.DataFrame([
        {
            "#": i,
            "🏆": "🏆 Best" if best and r["exp_id"] == best["exp_id"] else "",
            "Experiment": r["name"],
            "Dataset": _dataset_name(r),
            "Status": r["status"],
            "Avg Lift ×": round(_f(r.get("avg_lift")) or 0.0, 3),
            "Max Lift ×": round(_f(r.get("max_lift")) or 0.0, 3),
            "Coverage %": round(_f(r.get("coverage_pct")) or 0.0, 1),
            "Segments": _i(r.get("segments_count")),
            "Rows": _i(r.get("data_rows")),
            "Time (s)": round(_f(r.get("execution_time_sec")) or 0.0, 2),
        }
        for i, r in enumerate(ranked, 1)
    ])
    st.dataframe(grid, width='stretch', hide_index=True)

    # Row-level actions
    st.divider()
    st.subheader("Actions")
    for r in ranked:
        label = f"{'🏆 ' if best and r['exp_id'] == best['exp_id'] else ''}{r['name']}  ·  {r['status']}"
        with st.expander(label):
            if r["status"] != "completed":
                st.caption(f"⚠️ {_s(r.get('error_msg')) or 'Run did not complete.'}")
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                if st.button("Clone → Workbench", key=f"cl_{r['exp_id']}"):
                    clone_to_workbench(r["exp_id"])
            with c2:
                if st.button("View results", key=f"vw_{r['exp_id']}"):
                    view_results(r["exp_id"])
            with c3:
                export_run(r["exp_id"])
            with c4:
                if st.button("Duplicate", key=f"dp_{r['exp_id']}"):
                    duplicate_experiment(r["exp_id"])
            with c5:
                if st.button("🗑 Delete", key=f"dl_{r['exp_id']}", type="primary"):
                    st.session_state[f"m5_confirm_{r['exp_id']}"] = True
            if st.session_state.get(f"m5_confirm_{r['exp_id']}"):
                st.warning(f"Delete **{r['name']}**? This cannot be undone.")
                cc1, cc2 = st.columns(2)
                with cc1:
                    if st.button("Yes, delete", key=f"dly_{r['exp_id']}", type="primary"):
                        delete_experiment(r["exp_id"])
                        st.session_state.pop(f"m5_confirm_{r['exp_id']}", None)
                        st.rerun()
                with cc2:
                    if st.button("Cancel", key=f"dlc_{r['exp_id']}"):
                        st.session_state.pop(f"m5_confirm_{r['exp_id']}", None)

    # Compare two runs
    st.divider()
    st.subheader("Compare two runs")
    opts = [(r["exp_id"], r["name"]) for r in ranked]
    if len(opts) >= 2:
        a_id = st.selectbox("Run A", opts, format_func=lambda o: o[1], key="cmp_a")
        b_id = st.selectbox(
            "Run B", opts, index=min(1, len(opts) - 1),
            format_func=lambda o: o[1], key="cmp_b",
        )
        if a_id != b_id:
            fa = load_full_experiment(a_id[0])
            fb = load_full_experiment(b_id[0])
            if fa and fb:
                diffs = param_diff(fa.get("config") or {}, fb.get("config") or {})
                if diffs:
                    st.dataframe(
                        pd.DataFrame(
                            [{"Parameter": k, "Run A": str(v), "Run B": str(w)}
                             for k, v, w in diffs]),
                        width='stretch', hide_index=True)
                else:
                    st.success("Configs are identical.")
        else:
            st.info("Pick two different runs to compare.")
    else:
        st.info("Need at least two runs to compare.")


if __name__ == "__main__":
    main()
