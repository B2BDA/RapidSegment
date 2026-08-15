"""
Workbench - Create & Clone (Connected to Real RapidSegment params)
Two-Column Card Layout + State Injection + Black Run Button
"""
import json
import pandas as pd
from pathlib import Path

from .db import SuiteDB
from .builder_runner import RapidSegmentRunner
from .data_loader import SuiteDataLoader

# Default builder params matching library defaults
DEFAULT_BUILDER_PARAMS = {
    "target": "default_flag",
    "n_jobs": -1,
    "min_sample_size": 1000,
    "min_lift": 2.0,
    "min_events": 5,
    "top_n_vars": 15,
    "max_segments": 5,
    "max_feature_reuse": 1,
    "enable_diversity": False,
    "enable_1way": True,
    "enable_2way": True,
    "enable_3way": True,
    "feature_groups": {},
    "ignore_features": [],
    "param_grid": {
        "min_sample_size": [1000, 2500],
        "min_lift": [1.5, 2.5]
    },
    "sort_priority": "rate_lift_count",
    "binning_method": "optimal",
    "naive_bins": 5,
    "max_expansion_hops": 0,
    "selection_metric": "iv",
    "expand_log_mode": "none"
}

class Workbench:
    def __init__(self, db: SuiteDB = None, data: pd.DataFrame = None, data_path: str = None):
        self.db = db or SuiteDB()
        self.runner = RapidSegmentRunner(db=self.db)
        self.loader = SuiteDataLoader()
        
        # Load data
        if data is not None or data_path is not None:
            self.data = self.loader.load(file_path=data_path, df=data)
        else:
            self.data = None

        self.current_params = json.loads(json.dumps(DEFAULT_BUILDER_PARAMS))
        self.current_name = "exp_first_run"
        self._last_exp_id = None
        self.primary_key = None

    def set_data(self, df: pd.DataFrame = None, file_path: str = None, primary_key: str = None):
        self.data = self.loader.load(file_path=file_path, df=df)
        self.primary_key = primary_key
        print(f"Data set: {self.data.shape}, PK={primary_key}")
        return self.loader.preview(self.data)

    def configure(self, params: dict = None, name: str = None, primary_key: str = None) -> pd.DataFrame:
        if params:
            # merge with defaults
            merged = json.loads(json.dumps(DEFAULT_BUILDER_PARAMS))
            for k,v in params.items():
                merged[k] = v
            self.current_params = merged
        if name:
            self.current_name = name
        if primary_key:
            self.primary_key = primary_key
        return self.show_summary()

    def show_summary(self) -> pd.DataFrame:
        """Right column - pandas light view + validation"""
        flat = []
        for k,v in self.current_params.items():
            if k in ["feature_groups", "param_grid", "ignore_features"]:
                flat.append({"group": "advanced" if k=="param_grid" else "rules", "param": k, "value": str(v)[:80]})
            else:
                group = "data_scope" if k in ["target","ignore_features"] else "rules" if k in ["top_n_vars","max_segments","max_feature_reuse","enable_diversity","feature_groups"] else "advanced"
                flat.append({"group": group, "param": k, "value": v})
        df = pd.DataFrame(flat)

        # Validation checks
        checks = []
        if not self.current_params.get("target"):
            checks.append("❌ target required")
        if self.data is not None and self.current_params.get("target") not in self.data.columns:
            checks.append(f"❌ target {self.current_params.get('target')} not in data columns {list(self.data.columns)[:10]}")
        if self.current_params.get("min_sample_size",0) < 100:
            checks.append("⚠️ min_sample_size very small")
        if self.current_params.get("max_segments",0) > 20:
            checks.append("⚠️ max_segments large - may be slow")
        if self.data is None:
            checks.append("⚠️ No data loaded - call set_data()")

        if not checks:
            checks = ["✅ All checks passed - ready to run StrategicSegmentBuilder"]

        print(f"Experiment: {self.current_name} | Target: {self.current_params.get('target')} | Data: {self.data.shape if self.data is not None else 'None'}")
        try:
            from IPython.display import display
            display(df.style.set_caption("Builder Param Summary").hide(axis="index"))
        except:
            print(df.to_string())
        
        print("\nValidation:")
        for c in checks:
            print(c)
        
        if self.data is not None:
            print("\nData Profile (DuckDB quick):")
            prof = self.loader.profile(self.data, self.current_params.get("target","target"))
            print(prof.to_string(index=False))

        return df

    def clone_from_history(self, exp_id: str):
        rec = self.db.get_experiment(exp_id)
        self.current_params = rec['params']
        self.current_name = rec['name'] + "_clone"
        print(f"Cloned {exp_id} ({rec['name']}) -> {self.current_name}")
        return self.show_summary()

    def run(self, name: str = None, data: pd.DataFrame = None) -> str:
        if name:
            self.current_name = name
        if data is not None:
            self.data = data
        if self.data is None:
            raise ValueError("No data - call set_data(df) first")

        exp_id = self.runner.run(
            name=self.current_name,
            params=self.current_params,
            data=self.data,
            primary_key=self.primary_key,
            build_scorecard=True
        )
        self._last_exp_id = exp_id
        print(f"✓ Completed {exp_id} - Check console.show(exp_id) and leaderboard.show()")
        return exp_id

    def ui(self):
        """Interactive two-column UI - Jupyter native"""
        try:
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError:
            print("ipywidgets not available, use .show_summary() and .run()")
            return self.show_summary()

        # Data scope
        target_w = widgets.Text(value=str(self.current_params['target']), description='Target:')
        ignore_w = widgets.Text(value=",".join(self.current_params['ignore_features']), description='Ignore:')
        pk_w = widgets.Text(value=self.primary_key or "", description='Primary Key:')

        # Rules
        top_n_w = widgets.IntSlider(value=self.current_params['top_n_vars'], min=5, max=50, step=1, description='Top N vars:')
        max_seg_w = widgets.IntSlider(value=self.current_params['max_segments'], min=1, max=15, step=1, description='Max seg:')
        max_reuse_w = widgets.IntSlider(value=self.current_params['max_feature_reuse'], min=1, max=5, step=1, description='Max reuse:')
        diversity_w = widgets.Checkbox(value=self.current_params['enable_diversity'], description='Enable diversity')
        enable_1w = widgets.Checkbox(value=self.current_params['enable_1way'], description='1-way')
        enable_2w = widgets.Checkbox(value=self.current_params['enable_2way'], description='2-way')
        enable_3w = widgets.Checkbox(value=self.current_params['enable_3way'], description='3-way')

        # Advanced
        min_sample_w = widgets.IntSlider(value=self.current_params['min_sample_size'], min=100, max=10000, step=100, description='Min sample:')
        min_lift_w = widgets.FloatSlider(value=self.current_params['min_lift'], min=1.0, max=10.0, step=0.1, description='Min lift:')
        min_events_w = widgets.IntSlider(value=self.current_params['min_events'], min=1, max=500, step=5, description='Min events:')
        binning_w = widgets.Dropdown(options=['optimal','optimal_cart','optimal_quantile','naive'], value=self.current_params['binning_method'], description='Binning:')
        sort_w = widgets.Dropdown(options=['rate_lift_count','lift_count_rate','count_lift_rate','lift','rate'], value=self.current_params['sort_priority'], description='Sort:')
        expansion_w = widgets.IntSlider(value=self.current_params['max_expansion_hops'], min=0, max=3, step=1, description='Expansion hops:')

        exp_name_w = widgets.Text(value=self.current_name, description='Exp name:')

        # Clone dropdown
        history = self.db.list_params_for_clone()
        options = [(f"{v['name']} ({k})", k) for k,v in history.items()]
        clone_dd = widgets.Dropdown(options=[("", "")] + options, description='Clone from:')

        # Layout
        data_box = widgets.VBox([widgets.HTML("<b>Data Scope</b>"), target_w, ignore_w, pk_w])
        rules_box = widgets.VBox([widgets.HTML("<b>Segment Rules</b>"), top_n_w, max_seg_w, max_reuse_w, diversity_w, widgets.HBox([enable_1w, enable_2w, enable_3w])])
        adv_box = widgets.VBox([widgets.HTML("<b>Advanced Hyperparameters</b>"), min_sample_w, min_lift_w, min_events_w, binning_w, sort_w, expansion_w])

        left_accordion = widgets.Accordion(children=[data_box, rules_box, adv_box])
        left_accordion.set_title(0, "Data Scope")
        left_accordion.set_title(1, "Segment Rules")
        left_accordion.set_title(2, "Advanced")

        out_summary = widgets.Output()

        def refresh(*args):
            # Build params from widgets
            ignore_list = [x.strip() for x in ignore_w.value.split(",") if x.strip()]
            self.current_params.update({
                "target": target_w.value,
                "ignore_features": ignore_list,
                "top_n_vars": top_n_w.value,
                "max_segments": max_seg_w.value,
                "max_feature_reuse": max_reuse_w.value,
                "enable_diversity": diversity_w.value,
                "enable_1way": enable_1w.value,
                "enable_2way": enable_2w.value,
                "enable_3way": enable_3w.value,
                "min_sample_size": min_sample_w.value,
                "min_lift": min_lift_w.value,
                "min_events": min_events_w.value,
                "binning_method": binning_w.value,
                "sort_priority": sort_w.value,
                "max_expansion_hops": expansion_w.value,
            })
            self.current_name = exp_name_w.value
            self.primary_key = pk_w.value or None
            with out_summary:
                out_summary.clear_output()
                flat = [{"param": k, "value": str(v)[:60]} for k,v in self.current_params.items()]
                display(pd.DataFrame(flat))

        for w in [target_w, ignore_w, pk_w, top_n_w, max_seg_w, max_reuse_w, diversity_w, enable_1w, enable_2w, enable_3w, min_sample_w, min_lift_w, min_events_w, binning_w, sort_w, expansion_w, exp_name_w]:
            w.observe(refresh, names='value')

        def on_clone(change):
            if change['new']:
                rec = self.db.get_experiment(change['new'])
                p = rec['params']
                target_w.value = p.get('target','')
                ignore_w.value = ",".join(p.get('ignore_features',[]))
                top_n_w.value = p.get('top_n_vars',15)
                max_seg_w.value = p.get('max_segments',5)
                max_reuse_w.value = p.get('max_feature_reuse',1)
                diversity_w.value = p.get('enable_diversity',False)
                min_sample_w.value = p.get('min_sample_size',1000)
                min_lift_w.value = p.get('min_lift',2.0)
                min_events_w.value = p.get('min_events',5)
                binning_w.value = p.get('binning_method','optimal')
                sort_w.value = p.get('sort_priority','rate_lift_count')
                expansion_w.value = p.get('max_expansion_hops',0)
                exp_name_w.value = rec['name'] + "_clone"

        clone_dd.observe(on_clone, names='value')

        run_btn = widgets.Button(description="▶ Run Experiment", button_style='primary', layout=widgets.Layout(width='220px', height='44px'))
        run_btn.style.button_color = 'black'
        run_btn.style.text_color = 'white'
        out_run = widgets.Output()

        def on_run(b):
            with out_run:
                out_run.clear_output()
                print(f"Running {exp_name_w.value} with StrategicSegmentBuilder...")
                try:
                    eid = self.run(name=exp_name_w.value)
                    print(f"Done: {eid}")
                except Exception as e:
                    print(f"Error: {e}")

        run_btn.on_click(on_run)

        ui = widgets.VBox([
            clone_dd,
            widgets.HBox([left_accordion, out_summary], layout=widgets.Layout(gap='20px')),
            widgets.HTML("<hr>"),
            exp_name_w,
            run_btn,
            out_run
        ])
        refresh()
        return ui
