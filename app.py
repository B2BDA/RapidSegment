
import solara
import solara.lab
from pathlib import Path
import json


def _prewarm_duckdb():
    """Ensure duckdb's C extension is fully initialized before the app runs.

    Solara's autorouting executes this module (often more than once). A
    partially-initialized ``_duckdb`` left in ``sys.modules`` causes the
    intermittent ``No module named '_duckdb._sqltypes'`` error; this helper
    cleans poisoned modules and retries the import.
    """
    import sys
    for _ in range(3):
        try:
            import duckdb  # noqa: F401
            import _duckdb  # noqa: F401
            if not hasattr(_duckdb, "_sqltypes"):
                raise RuntimeError("_duckdb._sqltypes not attached")
            return duckdb
        except Exception:
            for m in list(sys.modules):
                if m == "_duckdb" or m.startswith("_duckdb.") or m == "duckdb" or m.startswith("duckdb."):
                    sys.modules.pop(m, None)
            continue
    import duckdb  # final attempt
    return duckdb


_prewarm_duckdb()

from rapidsegment_suite.db import SuiteDB
from rapidsegment_suite.builder_runner import RapidSegmentRunner
from rapidsegment_suite.data_profiler_duckdb import DuckDBProfiler
from rapidsegment_suite.module1_data_source import (
    DataSourceModule,
    profile_report as m1_profile_report,
)

db = SuiteDB()
profiler = DuckDBProfiler(db_path=".rapidsegment_suite/profiling.db")
runner = RapidSegmentRunner(db=db)

theme_mode = solara.reactive("light")
tab_index = solara.reactive(0)
data_table_name = solara.reactive("main_data")
target_col = solara.reactive("")
primary_key = solara.reactive("")
exp_name = solara.reactive("exp_first_run")
status_msg = solara.reactive("Ready - load a dataset in Module 1 to begin")
column_list = solara.reactive([])
selected_exp = solara.reactive("")
exp_a = solara.reactive("")
exp_b = solara.reactive("")
exp_list = solara.reactive([])
builder_params = solara.reactive({
    "target": "", "top_n_vars": 15, "max_segments": 5, "max_feature_reuse": 1,
    "enable_diversity": False, "enable_1way": True, "enable_2way": True, "enable_3way": False,
    "min_sample_size": 1000, "min_lift": 2.0, "min_events": 5,
    "binning_method": "optimal", "sort_priority": "rate_lift_count"
})

def refresh_exps():
    try:
        df = db.list_experiments_df()
        exp_list.value = df.to_dict(orient="records") if df is not None and not df.empty else []
    except:
        exp_list.value = []

def proceed_to_workbench(target, table_name, pk):
    """Wired to Module 1's 'Proceed to Workbench' action."""
    data_table_name.value = table_name
    target_col.value = target
    primary_key.value = pk or ""
    column_list.value = [c["name"] for c in profiler._get_columns_meta(table_name)]
    bp = dict(builder_params.value)
    bp["target"] = target
    builder_params.value = bp
    if m1_profile_report.value:
        status_msg.value = (f"Ready to run | target={target} | "
                            f"{m1_profile_report.value['total_rows']:,} rows")
    tab_index.value = 1

def run_experiment():
    try:
        tgt = target_col.value or builder_params.value.get("target")
        if not tgt:
            status_msg.value = "Select target"
            return
        df = profiler.con.execute(f"SELECT * FROM {data_table_name.value}").df()
        params = dict(builder_params.value)
        params["target"] = tgt
        eid = runner.run(name=exp_name.value, params=params, data=df, primary_key=primary_key.value or None, build_scorecard=True)
        status_msg.value = f"Done {eid}"
        selected_exp.value = eid
        refresh_exps()
    except Exception as e:
        status_msg.value = f"Run error: {e}"

def toggle_theme():
    theme_mode.value = "dark" if theme_mode.value == "light" else "light"

@solara.component
def WorkbenchTab():
    with solara.Column():
        solara.Select(label="Target", values=column_list.value, value=target_col)
        solara.InputText(label="Exp name", value=exp_name)
        solara.SliderInt(label="Top N vars", value=solara.reactive(builder_params.value["top_n_vars"]), min=5, max=50, on_value=lambda v: builder_params.set({**builder_params.value, "top_n_vars": v}))
        solara.SliderInt(label="Max segments", value=solara.reactive(builder_params.value["max_segments"]), min=1, max=15, on_value=lambda v: builder_params.set({**builder_params.value, "max_segments": v}))
        solara.SliderFloat(label="Min lift", value=solara.reactive(builder_params.value["min_lift"]), min=1.0, max=10.0, step=0.1, on_value=lambda v: builder_params.set({**builder_params.value, "min_lift": v}))
        solara.Button(label="Run Experiment", on_click=run_experiment, style={"background": "black", "color": "white"})

@solara.component
def LeaderboardTab():
    refresh_exps()
    with solara.Column():
        solara.Button(label="Refresh", on_click=refresh_exps)
        if exp_list.value:
            html = "<table style='width:100%;'><tr><th>ID</th><th>Name</th><th>Lift</th></tr>"
            for r in exp_list.value[:30]:
                html += f"<tr><td>{r.get('exp_id','')[:12]}</td><td>{r.get('name','')}</td><td>{r.get('max_lift',0):.2f}</td></tr>"
            html += "</table>"
            solara.HTML(unsafe_innerHTML=html)

@solara.component
def ConsoleTab():
    with solara.Column():
        solara.Select(label="Exp", values=[r['exp_id'] for r in exp_list.value], value=selected_exp)

@solara.component
def ArenaTab():
    with solara.Column():
        solara.Select(label="A", values=[r['exp_id'] for r in exp_list.value], value=exp_a)
        solara.Select(label="B", values=[r['exp_id'] for r in exp_list.value], value=exp_b)

@solara.component
def Page():
    is_dark = theme_mode.value == "dark"
    bg = "#0e0e0e" if is_dark else "#ffffff"
    txt = "#f5f5f5" if is_dark else "#111111"
    solara.HTML(unsafe_innerHTML=f"<style>body{{background:{bg}!important; color:{txt}!important;}}</style>")
    with solara.Column(style={"max-width": "1200px", "margin": "0 auto", "padding": "12px", "background": bg, "color": txt}):
        with solara.Row(style={"justify-content": "space-between", "border-bottom": "2px solid #000", "padding": "8px 0"}):
            solara.Markdown("## RapidSegment - No-Code Platform")
            with solara.Row():
                solara.Text(f"{profiler.get_db_size_info()['file_size_mb']} MB | {status_msg.value}", style={"font-size": "11px", "max-width": "600px"})
                solara.Button(label="DARK" if theme_mode.value=="light" else "LIGHT", on_click=toggle_theme)
        with solara.lab.Tabs(value=tab_index):
            with solara.lab.Tab("Module 1 - Data Source & Profiling"):
                DataSourceModule(profiler=profiler, on_proceed=proceed_to_workbench)
            with solara.lab.Tab("Workbench"):
                WorkbenchTab()
            with solara.lab.Tab("Leaderboard"):
                LeaderboardTab()
            with solara.lab.Tab("Console"):
                ConsoleTab()
            with solara.lab.Tab("Arena"):
                ArenaTab()
