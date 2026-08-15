
import solara
import solara.lab
from pathlib import Path
import json

from rapidsegment_suite.db import SuiteDB
from rapidsegment_suite.builder_runner import RapidSegmentRunner
from rapidsegment_suite.data_profiler_duckdb import DuckDBProfiler

db = SuiteDB()
profiler = DuckDBProfiler(db_path=".rapidsegment_suite/profiling.db")
runner = RapidSegmentRunner(db=db)

theme_mode = solara.reactive("light")
source_type = solara.reactive("local")
local_file_path = solara.reactive("")
bq_full_path = solara.reactive("")
bq_project = solara.reactive("")
bq_dataset = solara.reactive("")
bq_table = solara.reactive("")
data_table_name = solara.reactive("main_data")
target_col = solara.reactive("")
primary_key = solara.reactive("")
exp_name = solara.reactive("exp_first_run")
status_msg = solara.reactive("Ready - paste file path OR BigQuery table path below")
profiling_report = solara.reactive(None)
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

def parse_bq(s):
    try:
        p = s.strip().replace("bq://","").replace("bigquery://","")
        parts = p.split(".")
        if len(parts) == 3:
            bq_project.value, bq_dataset.value, bq_table.value = parts
        elif len(parts) == 2:
            bq_dataset.value, bq_table.value = parts
    except:
        pass

def load_data():
    try:
        if source_type.value == "local":
            fp = local_file_path.value.strip()
            if not fp:
                status_msg.value = "Enter file path first"
                return
            if not Path(fp).exists():
                status_msg.value = f"File not found: {fp}"
                return
            profiler.load_table(file_path=fp, table_name=data_table_name.value)
        else:
            if bq_full_path.value:
                parse_bq(bq_full_path.value)
            if not bq_dataset.value or not bq_table.value:
                status_msg.value = "Enter BQ dataset.table"
                return
            profiler.load_from_bq(bq_path=bq_full_path.value or None, project_id=bq_project.value or None, dataset_id=bq_dataset.value, table_id=bq_table.value, table_name=data_table_name.value)
        cols = profiler._get_columns_meta(data_table_name.value)
        column_list.value = [c["name"] for c in cols]
        rep = profiler.profile(data_table_name.value, target_col=None)
        profiling_report.value = rep
        status_msg.value = f"Loaded {rep['total_rows']:,} rows x {rep['total_columns']} cols | DB {rep['size_info']['file_size_mb']} MB"
        refresh_exps()
    except Exception as e:
        status_msg.value = f"Load error: {e}"

def calc_event():
    try:
        if not target_col.value:
            status_msg.value = "Select target"
            return
        rep = profiler.profile(data_table_name.value, target_col=target_col.value)
        profiling_report.value = rep
        bp = dict(builder_params.value)
        bp["target"] = target_col.value
        builder_params.value = bp
        ev = rep.get('event_info',{})
        status_msg.value = f"Target {target_col.value}: {ev.get('event_rate_pct','?')}%"
    except Exception as e:
        status_msg.value = f"Error: {e}"

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
def DataTab():
    with solara.Column(style={"gap": "10px"}):
        solara.Select(label="Source type: local file path OR BigQuery path", values=["local", "bigquery"], value=source_type)
        if source_type.value == "local":
            solara.InputText(label="Local file path (e.g. /workspaces/RapidSegment/data.csv)", value=local_file_path, placeholder="/home/user/data.csv")
        else:
            solara.InputText(label="Full BQ path project.dataset.table", value=bq_full_path, placeholder="my-project.analytics.events", on_value=lambda v: parse_bq(v))
            with solara.Row():
                solara.InputText(label="Dataset", value=bq_dataset)
                solara.InputText(label="Table", value=bq_table)
        with solara.Row():
            solara.Button(label="1. Load & Profile (DuckDB disk)", on_click=load_data, style={"background": "black", "color": "white"})
            solara.Select(label="Target variable", values=column_list.value, value=target_col)
            solara.Button(label="2. Calc Event Rate", on_click=calc_event)
        if profiling_report.value:
            rep = profiling_report.value
            solara.Text(f"{rep['total_rows']} rows, {rep['total_columns']} cols, Num {rep['num_numeric']} Cat {rep['num_categorical']}, DB {rep['size_info']['file_size_mb']} MB")

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
        with solara.lab.Tabs():
            with solara.lab.Tab("Data Source & Profiling"):
                DataTab()
            with solara.lab.Tab("Workbench"):
                WorkbenchTab()
            with solara.lab.Tab("Leaderboard"):
                LeaderboardTab()
            with solara.lab.Tab("Console"):
                ConsoleTab()
            with solara.lab.Tab("Arena"):
                ArenaTab()
