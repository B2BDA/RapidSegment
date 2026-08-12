# -*- coding: utf-8 -*-
"""
RapidSegment Experiment Suite
=============================
Pure-ipywidgets experiment workbench for RapidSegment.

Usage inside a Jupyter notebook:
    from rs_experiment_suite import launch_suite
    suite = launch_suite()
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import platform
import re
import shutil
import sys
import time
import traceback
import uuid
import zipfile
from dataclasses import asdict, dataclass, field, fields, MISSING
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import ipywidgets as W
import numpy as np
import pandas as pd
from IPython.display import display

try:
    from rapidsegment import (
        StrategicSegmentBuilder,
        StrategicSegmentScore,
        UniversalDataLoader,
    )
    from rapidsegment import __version__ as RS_VERSION
except ImportError:
    StrategicSegmentBuilder = StrategicSegmentScore = UniversalDataLoader = None  # type: ignore
    RS_VERSION = "not-installed"

# ---------------------------------------------------------------------------
# Paths & helpers
# ---------------------------------------------------------------------------
SUITE_ROOT = Path("./rs_experiments").resolve()
SUITE_ROOT.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_name(name: str) -> str:
    s = re.sub(r"[^\w\-. ]+", "", name.strip())
    s = re.sub(r"\s+", "_", s)
    return s[:80] or "unnamed"


def _short_id() -> str:
    return uuid.uuid4().hex[:8]


def _json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def _json_load(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _file_sha256(path: Path, chunk: int = 1 << 20) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class DataSource:
    kind: str = "local"  # "local" | "bq"
    path: str = ""
    project_id: str = ""
    dataset_id: str = ""
    table_id: str = ""
    target_col: str = ""
    primary_key: str = ""
    ignore_features: List[str] = field(default_factory=list)

    def describe(self) -> str:
        if self.kind == "bq":
            return f"BQ://{self.project_id}.{self.dataset_id}.{self.table_id}"
        return f"file://{self.path}"


@dataclass
class BuilderParams:
    """Mirrors the StrategicSegmentBuilder knobs exposed in the UI."""

    n_jobs: int = -1
    min_sample_size: int = 1000
    min_lift: float = 2.0
    min_events: int = 5
    top_n_vars: int = 20
    max_segments: int = 10
    max_feature_reuse: int = 1
    enable_diversity: bool = False
    enable_1way: bool = True
    enable_2way: bool = True
    enable_3way: bool = True
    sort_priority: str = "lift_rate_count"
    binning_method: str = "optimal"
    naive_bins: int = 5
    max_expansion_hops: int = 0
    selection_metric: str = "iv"
    expand_log_mode: str = "summary"
    param_grid_min_sample: List[int] = field(default_factory=lambda: [1000])
    param_grid_min_lift: List[float] = field(default_factory=lambda: [2.0])
    feature_groups_json: str = "{}"
    ignore_features: List[str] = field(default_factory=list)

    def to_builder_kwargs(self, target: str) -> Dict[str, Any]:
        grid = None
        if self.param_grid_min_sample or self.param_grid_min_lift:
            grid = {
                "min_sample_size": self.param_grid_min_sample or [self.min_sample_size],
                "min_lift": self.param_grid_min_lift or [self.min_lift],
            }
        try:
            feature_groups = json.loads(self.feature_groups_json or "{}")
        except json.JSONDecodeError:
            feature_groups = {}
        return {
            "target": target,
            "n_jobs": self.n_jobs,
            "min_sample_size": self.min_sample_size,
            "min_lift": self.min_lift,
            "min_events": self.min_events,
            "top_n_vars": self.top_n_vars,
            "max_segments": self.max_segments,
            "max_feature_reuse": self.max_feature_reuse,
            "param_grid": grid,
            "enable_diversity": self.enable_diversity,
            "enable_1way": self.enable_1way,
            "enable_2way": self.enable_2way,
            "enable_3way": self.enable_3way,
            "feature_groups": feature_groups or None,
            "ignore_features": list(self.ignore_features),
            "sort_priority": self.sort_priority,
            "binning_method": self.binning_method,
            "naive_bins": self.naive_bins,
            "max_expansion_hops": self.max_expansion_hops,
            "selection_metric": self.selection_metric,
            "expand_log_mode": self.expand_log_mode,
        }


@dataclass
class ExperimentMeta:
    exp_id: str
    name: str
    created_at: str
    updated_at: str
    status: str = "draft"  # draft | queued | running | completed | failed
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    cloned_from: Optional[str] = None
    data_source: Dict[str, Any] = field(default_factory=dict)
    builder_params: Dict[str, Any] = field(default_factory=dict)
    env: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)
    run_info: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Logging capture
# ---------------------------------------------------------------------------
class ListHandler(logging.Handler):
    def __init__(self, level=logging.INFO):
        super().__init__(level)
        self.records: List[str] = []
        self._file = None

    def set_file(self, path: Optional[Path]):
        if self._file:
            self._file.close()
        self._file = open(path, "a", encoding="utf-8") if path else None

    def emit(self, record):
        try:
            msg = self.format(record)
            self.records.append(msg)
            if self._file:
                self._file.write(msg + "\n")
                self._file.flush()
        except Exception:
            self.handleError(record)

    def close(self):
        if self._file:
            self._file.close()
            self._file = None
        super().close()


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------
class ExperimentStore:
    def __init__(self, root: Path = SUITE_ROOT):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, exp_id: str) -> Path:
        return self.root / exp_id

    def list_ids(self) -> List[str]:
        ids = []
        for p in sorted(self.root.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if p.is_dir() and (p / "metadata.json").exists():
                ids.append(p.name)
        return ids

    def load_meta(self, exp_id: str) -> ExperimentMeta:
        raw = _json_load(self._dir(exp_id) / "metadata.json")
        kw: Dict[str, Any] = {}
        for f in fields(ExperimentMeta):
            if f.name in raw:
                kw[f.name] = raw[f.name]
            elif f.default is not MISSING:
                kw[f.name] = f.default
            elif f.default_factory is not MISSING:  # type: ignore
                kw[f.name] = f.default_factory()  # type: ignore
            else:
                kw[f.name] = None
        return ExperimentMeta(**kw)

    def save_meta(self, meta: ExperimentMeta) -> None:
        meta.updated_at = _now_iso()
        _json_dump(asdict(meta), self._dir(meta.exp_id) / "metadata.json")

    def create(
        self,
        name: str,
        notes: str = "",
        tags: Optional[List[str]] = None,
        cloned_from: Optional[str] = None,
    ) -> ExperimentMeta:
        exp_id = f"{_safe_name(name)}_{_short_id()}"
        self._dir(exp_id).mkdir(parents=True, exist_ok=True)
        meta = ExperimentMeta(
            exp_id=exp_id,
            name=name,
            created_at=_now_iso(),
            updated_at=_now_iso(),
            status="draft",
            notes=notes,
            tags=tags or [],
            cloned_from=cloned_from,
            env=self._capture_env(),
        )
        self.save_meta(meta)
        return meta

    def clone(self, source_id: str, new_name: str) -> ExperimentMeta:
        src = self.load_meta(source_id)
        meta = self.create(
            new_name,
            notes=f"Cloned from {src.name} ({source_id})\n{src.notes}",
            tags=list(src.tags),
            cloned_from=source_id,
        )
        meta.data_source = copy.deepcopy(src.data_source)
        meta.builder_params = copy.deepcopy(src.builder_params)
        meta.stats = copy.deepcopy(src.stats)
        meta.status = "draft"
        self.save_meta(meta)
        for fname in ("builder_params.json", "data_source.json", "stats.json"):
            src_p = self._dir(source_id) / fname
            if src_p.exists():
                shutil.copy2(src_p, self._dir(meta.exp_id) / fname)
        return meta

    def delete(self, exp_id: str) -> None:
        d = self._dir(exp_id)
        if d.exists():
            shutil.rmtree(d)

    def path(self, exp_id: str, *parts) -> Path:
        return self._dir(exp_id).joinpath(*parts)

    @staticmethod
    def _capture_env() -> Dict[str, Any]:
        return {
            "python": sys.version,
            "platform": platform.platform(),
            "rapidsegment": RS_VERSION,
            "pandas": getattr(pd, "__version__", "?"),
            "duckdb": getattr(duckdb, "__version__", "?"),
            "numpy": getattr(np, "__version__", "?"),
            "timestamp": _now_iso(),
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
class ExperimentRunner:
    def __init__(self, store: ExperimentStore):
        self.store = store

    def load_data(self, ds: DataSource) -> Any:
        if UniversalDataLoader is None:
            raise RuntimeError("rapidsegment is not installed")
        if ds.kind == "bq":
            loader = UniversalDataLoader(
                project_id=ds.project_id or None,
                dataset_id=ds.dataset_id,
                table_id=ds.table_id,
            )
            return loader.load()
        if not ds.path:
            raise ValueError("Local path is empty")
        loader = UniversalDataLoader(file_path=ds.path)
        return loader.load()

    def profile(self, data: Any, target: str) -> Dict[str, Any]:
        con = duckdb.connect()
        con.register("t", data)
        n_rows = con.execute("SELECT COUNT(*) FROM t").fetchone()[0]
        cols = con.execute("DESCRIBE t").fetchall()
        col_names = [c[0] for c in cols]
        col_types = {c[0]: c[1] for c in cols}

        null_rates = {}
        for c in col_names:
            try:
                nulls = con.execute(
                    f'SELECT SUM(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END) FROM t'
                ).fetchone()[0]
                null_rates[c] = round(100.0 * (nulls or 0) / max(n_rows, 1), 3)
            except Exception:
                null_rates[c] = None

        event_rate = None
        target_balance = None
        if target and target in col_names:
            try:
                ev = con.execute(
                    f'SELECT AVG(CAST("{target}" AS DOUBLE)), '
                    f'SUM(CAST("{target}" AS DOUBLE)), COUNT(*) FROM t'
                ).fetchone()
                event_rate = round(float(ev[0] or 0) * 100, 4)
                target_balance = {
                    "events": int(ev[1] or 0),
                    "non_events": int((ev[2] or 0) - (ev[1] or 0)),
                    "event_rate_pct": event_rate,
                }
            except Exception as e:
                target_balance = {"error": str(e)}

        sample = None
        try:
            sample = con.execute("SELECT * FROM t LIMIT 5").df().to_dict(orient="records")
        except Exception:
            pass
        con.close()
        return {
            "n_rows": int(n_rows),
            "n_columns": len(col_names),
            "columns": col_names,
            "dtypes": col_types,
            "null_rates_pct": null_rates,
            "event_rate_pct": event_rate,
            "target_balance": target_balance,
            "sample_head": sample,
            "profiled_at": _now_iso(),
        }

    def run(self, meta: ExperimentMeta, progress_cb=None, status_cb=None) -> ExperimentMeta:
        def status(msg: str):
            if status_cb:
                status_cb(msg)
            logger = logging.getLogger("StrategicEngine")
            logger.info(msg)

        def progress(p: float):
            if progress_cb:
                progress_cb(max(0.0, min(1.0, p)))

        exp_id = meta.exp_id
        log_path = self.store.path(exp_id, "logs.txt")
        handler = ListHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
        handler.set_file(log_path)

        root_logger = logging.getLogger()
        eng_logger = logging.getLogger("StrategicEngine")
        root_logger.addHandler(handler)
        eng_logger.addHandler(handler)
        eng_logger.setLevel(logging.INFO)

        meta.status = "running"
        meta.run_info = {"started_at": _now_iso()}
        meta.error = None
        self.store.save_meta(meta)

        try:
            if StrategicSegmentBuilder is None:
                raise RuntimeError("rapidsegment package is not available")

            ds = DataSource(**meta.data_source)
            known = {f.name for f in fields(BuilderParams)}
            params = BuilderParams(
                **{k: v for k, v in meta.builder_params.items() if k in known}
            )

            status("Loading data…")
            progress(0.05)
            data = self.load_data(ds)

            status("Profiling dataset…")
            progress(0.15)
            stats = self.profile(data, ds.target_col)
            meta.stats = stats
            _json_dump(stats, self.store.path(exp_id, "stats.json"))
            self.store.save_meta(meta)

            status(
                f"Data ready – {stats['n_rows']:,} rows × {stats['n_columns']} cols | "
                f"event rate {stats.get('event_rate_pct')}%"
            )
            progress(0.25)

            _json_dump(asdict(ds), self.store.path(exp_id, "data_source.json"))
            _json_dump(asdict(params), self.store.path(exp_id, "builder_params.json"))

            if ds.kind == "local" and ds.path:
                meta.env["data_sha256"] = _file_sha256(Path(ds.path))

            status("Building StrategicSegmentBuilder…")
            progress(0.30)
            kwargs = params.to_builder_kwargs(ds.target_col)
            ign = set(kwargs.get("ignore_features") or [])
            if ds.primary_key:
                ign.add(ds.primary_key)
            kwargs["ignore_features"] = list(ign)

            builder = StrategicSegmentBuilder(**kwargs)

            status("Extracting hierarchical segments (this may take a while)…")
            progress(0.40)
            t0 = time.time()
            segments = builder.extract_segments(data)
            extract_secs = time.time() - t0
            progress(0.75)

            status(f"Extracted {len(segments)} segments in {extract_secs:.1f}s – evaluating coverage…")
            coverage = builder.evaluate_final_coverage(data)
            progress(0.90)

            _json_dump(segments, self.store.path(exp_id, "segments.json"))
            cov_clean = []
            for row in coverage:
                cov_clean.append(
                    {
                        k: (
                            float(v)
                            if isinstance(v, (np.floating, float))
                            else int(v)
                            if isinstance(v, (np.integer, int))
                            else v
                        )
                        for k, v in row.items()
                    }
                )
            _json_dump(cov_clean, self.store.path(exp_id, "coverage.json"))

            try:
                _json_dump(builder.diagnostics_, self.store.path(exp_id, "diagnostics.json"))
            except Exception:
                pass

            meta.run_info.update(
                {
                    "finished_at": _now_iso(),
                    "extract_seconds": round(extract_secs, 2),
                    "n_segments": len(segments),
                    "success": True,
                }
            )
            meta.status = "completed"
            status("✅ Experiment completed successfully")
            progress(1.0)

        except Exception as e:
            meta.status = "failed"
            meta.error = f"{type(e).__name__}: {e}"
            meta.run_info["finished_at"] = _now_iso()
            meta.run_info["success"] = False
            tb = traceback.format_exc()
            handler.records.append(tb)
            if handler._file:
                handler._file.write(tb + "\n")
            status(f"❌ Failed: {meta.error}")
            progress(1.0)
        finally:
            root_logger.removeHandler(handler)
            eng_logger.removeHandler(handler)
            handler.close()
            _json_dump({"lines": handler.records}, self.store.path(exp_id, "logs.json"))
            self.store.save_meta(meta)

        return meta


# ---------------------------------------------------------------------------
# UI constants
# ---------------------------------------------------------------------------
SORT_CHOICES = [
    "lift_rate_count",
    "lift_count_rate",
    "count_lift_rate",
    "rate_lift_count",
    "count_rate_lift",
    "rate_count_lift",
    "events_lift_rate",
    "events_rate_lift",
    "lift_events_rate",
    "rate_events_lift",
    "events_count_rate",
    "events_rate_count",
    "count_events_rate",
    "rate_events_count",
]

PRESETS = {
    "Balanced (default)": dict(
        min_sample_size=1000,
        min_lift=2.0,
        max_segments=10,
        top_n_vars=20,
        enable_1way=True,
        enable_2way=True,
        enable_3way=True,
        sort_priority="lift_rate_count",
        binning_method="optimal",
    ),
    "High-lift focus": dict(
        min_sample_size=500,
        min_lift=3.5,
        max_segments=8,
        top_n_vars=15,
        enable_1way=True,
        enable_2way=True,
        enable_3way=False,
        sort_priority="lift_events_rate",
        binning_method="optimal",
    ),
    "Volume-first": dict(
        min_sample_size=2500,
        min_lift=1.5,
        max_segments=12,
        top_n_vars=25,
        enable_1way=True,
        enable_2way=True,
        enable_3way=True,
        sort_priority="count_lift_rate",
        binning_method="naive",
        naive_bins=5,
    ),
    "Fast exploratory": dict(
        min_sample_size=2000,
        min_lift=2.0,
        max_segments=5,
        top_n_vars=10,
        enable_1way=True,
        enable_2way=True,
        enable_3way=False,
        sort_priority="lift_rate_count",
        binning_method="naive",
        naive_bins=4,
        max_expansion_hops=0,
    ),
}


# ---------------------------------------------------------------------------
# Main UI class
# ---------------------------------------------------------------------------
class ExperimentSuiteUI:
    def __init__(self, root: Path = SUITE_ROOT):
        self.store = ExperimentStore(root)
        self.runner = ExperimentRunner(self.store)
        self.queue: List[str] = []
        self._current_meta: Optional[ExperimentMeta] = None
        self._stop_flag = False
        self._build_widgets()
        self._refresh_experiment_list()

    def display(self):
        display(self.root_box)

    # ------------------------------------------------------------------ build
    def _build_widgets(self):
        self.out_log = W.Output(
            layout=W.Layout(height="220px", overflow_y="auto", border="1px solid #ccc")
        )
        self.status_html = W.HTML(value="<i>Ready</i>")
        self.progress = W.FloatProgress(
            value=0, min=0, max=1, description="Progress", layout=W.Layout(width="60%")
        )
        self.progress_label = W.Label(value="")

        # Experiments list
        self.exp_select = W.Select(
            options=[], description="Experiments", layout=W.Layout(width="100%", height="180px")
        )
        self.btn_refresh = W.Button(description="↻ Refresh", button_style="info")
        self.btn_load = W.Button(description="Load into editor", button_style="primary")
        self.btn_clone = W.Button(description="Clone…")
        self.btn_delete = W.Button(description="Delete", button_style="danger")
        self.btn_add_queue = W.Button(description="＋ Add to queue", button_style="success")

        self.btn_refresh.on_click(lambda _: self._refresh_experiment_list())
        self.btn_load.on_click(self._on_load)
        self.btn_clone.on_click(self._on_clone)
        self.btn_delete.on_click(self._on_delete)
        self.btn_add_queue.on_click(self._on_add_queue)

        list_box = W.VBox(
            [
                W.HTML("<b>Saved experiments</b>"),
                self.exp_select,
                W.HBox(
                    [
                        self.btn_refresh,
                        self.btn_load,
                        self.btn_clone,
                        self.btn_delete,
                        self.btn_add_queue,
                    ]
                ),
            ]
        )

        # Identity
        self.w_name = W.Text(value="My experiment", description="Name", layout=W.Layout(width="60%"))
        self.w_notes = W.Textarea(
            value="", description="Notes", layout=W.Layout(width="90%", height="60px")
        )
        self.w_tags = W.Text(
            value="",
            description="Tags",
            placeholder="comma,separated",
            layout=W.Layout(width="60%"),
        )

        # Data source
        self.w_kind = W.ToggleButtons(
            options=[("Local file", "local"), ("BigQuery", "bq")], description="Source", value="local"
        )
        self.w_path = W.Text(
            value="",
            description="File path",
            placeholder="/path/to/data.parquet",
            layout=W.Layout(width="80%"),
        )
        self.w_project = W.Text(value="", description="Project", layout=W.Layout(width="40%"))
        self.w_dataset = W.Text(value="", description="Dataset", layout=W.Layout(width="40%"))
        self.w_table = W.Text(value="", description="Table", layout=W.Layout(width="40%"))
        self.w_target = W.Text(value="", description="Target col", layout=W.Layout(width="40%"))
        self.w_pk = W.Text(value="", description="Primary key", layout=W.Layout(width="40%"))
        self.w_ignore = W.Text(
            value="", description="Ignore cols", placeholder="col1,col2", layout=W.Layout(width="70%")
        )

        self.btn_profile = W.Button(
            description="Load & Profile data", button_style="info", icon="search"
        )
        self.btn_profile.on_click(self._on_profile)
        self.profile_out = W.Output(
            layout=W.Layout(border="1px solid #ddd", max_height="260px", overflow_y="auto")
        )

        self.w_kind.observe(self._toggle_source, names="value")
        self._toggle_source({"new": "local"})

        data_box = W.VBox(
            [
                W.HTML("<b>1 · Data source</b>"),
                self.w_kind,
                self.w_path,
                W.HBox([self.w_project, self.w_dataset, self.w_table]),
                W.HBox([self.w_target, self.w_pk]),
                self.w_ignore,
                self.btn_profile,
                self.profile_out,
            ]
        )

        # Builder params
        self.w_preset = W.Dropdown(
            options=list(PRESETS.keys()), description="Preset", layout=W.Layout(width="50%")
        )
        self.btn_apply_preset = W.Button(description="Apply preset")
        self.btn_apply_preset.on_click(self._on_apply_preset)

        self.w_min_sample = W.IntText(value=1000, description="min_sample_size")
        self.w_min_lift = W.FloatText(value=2.0, description="min_lift")
        self.w_min_events = W.IntText(value=5, description="min_events")
        self.w_top_n = W.IntText(value=20, description="top_n_vars")
        self.w_max_seg = W.IntText(value=10, description="max_segments")
        self.w_reuse = W.IntText(value=1, description="max_feature_reuse")
        self.w_njobs = W.IntText(value=-1, description="n_jobs")

        self.w_1way = W.Checkbox(value=True, description="1-way")
        self.w_2way = W.Checkbox(value=True, description="2-way")
        self.w_3way = W.Checkbox(value=True, description="3-way")
        self.w_diversity = W.Checkbox(value=False, description="enable_diversity")

        self.w_sort = W.Dropdown(
            options=SORT_CHOICES, value="lift_rate_count", description="sort_priority"
        )
        self.w_binning = W.Dropdown(options=["optimal", "naive"], value="optimal", description="binning")
        self.w_naive_bins = W.IntText(value=5, description="naive_bins")
        self.w_hops = W.IntText(value=0, description="max_expansion_hops")
        self.w_sel_metric = W.Dropdown(
            options=["iv", "response_rate"], value="iv", description="selection_metric"
        )
        self.w_expand_log = W.Dropdown(
            options=["none", "summary", "champion", "full"],
            value="summary",
            description="expand_log",
        )

        self.w_grid_sample = W.Text(
            value="1000", description="grid min_sample", placeholder="e.g. 1000,2500,5000"
        )
        self.w_grid_lift = W.Text(
            value="2.0", description="grid min_lift", placeholder="e.g. 2.0,3.5"
        )
        self.w_feat_groups = W.Textarea(
            value="{}",
            description="feature_groups (JSON)",
            layout=W.Layout(width="90%", height="60px"),
        )

        params_box = W.VBox(
            [
                W.HTML("<b>2 · Builder parameters</b>"),
                W.HBox([self.w_preset, self.btn_apply_preset]),
                W.HBox([self.w_min_sample, self.w_min_lift, self.w_min_events]),
                W.HBox([self.w_top_n, self.w_max_seg, self.w_reuse, self.w_njobs]),
                W.HBox([self.w_1way, self.w_2way, self.w_3way, self.w_diversity]),
                W.HBox([self.w_sort, self.w_binning, self.w_naive_bins]),
                W.HBox([self.w_hops, self.w_sel_metric, self.w_expand_log]),
                W.HBox([self.w_grid_sample, self.w_grid_lift]),
                self.w_feat_groups,
            ]
        )

        self.btn_save = W.Button(description="💾 Save experiment (draft)", button_style="primary")
        self.btn_save_queue = W.Button(description="💾 Save & add to queue", button_style="success")
        self.btn_save.on_click(lambda _: self._save_current(add_to_queue=False))
        self.btn_save_queue.on_click(lambda _: self._save_current(add_to_queue=True))

        form_box = W.VBox(
            [
                W.HTML("<b>Experiment identity</b>"),
                self.w_name,
                self.w_notes,
                self.w_tags,
                data_box,
                params_box,
                W.HBox([self.btn_save, self.btn_save_queue]),
            ]
        )

        # Queue
        self.queue_select = W.SelectMultiple(
            options=[], description="Queue", layout=W.Layout(width="100%", height="140px")
        )
        self.btn_q_up = W.Button(description="↑")
        self.btn_q_down = W.Button(description="↓")
        self.btn_q_remove = W.Button(description="Remove", button_style="warning")
        self.btn_q_clear = W.Button(description="Clear queue", button_style="danger")
        self.btn_run = W.Button(
            description="▶ Run queue sequentially",
            button_style="success",
            layout=W.Layout(width="260px"),
        )
        self.btn_stop = W.Button(description="■ Stop after current", button_style="danger")

        self.btn_q_up.on_click(lambda _: self._queue_move(-1))
        self.btn_q_down.on_click(lambda _: self._queue_move(1))
        self.btn_q_remove.on_click(self._queue_remove)
        self.btn_q_clear.on_click(lambda _: self._queue_clear())
        self.btn_run.on_click(self._on_run_queue)
        self.btn_stop.on_click(lambda _: setattr(self, "_stop_flag", True))

        queue_box = W.VBox(
            [
                W.HTML("<b>Experiment queue</b> (runs one after another)"),
                self.queue_select,
                W.HBox([self.btn_q_up, self.btn_q_down, self.btn_q_remove, self.btn_q_clear]),
                W.HBox([self.btn_run, self.btn_stop]),
                W.HBox([self.progress, self.progress_label]),
                self.status_html,
                W.HTML("<b>Live log</b>"),
                self.out_log,
            ]
        )

        # Results
        self.res_select = W.Dropdown(options=[], description="Experiment")
        self.btn_res_load = W.Button(description="Show results", button_style="info")
        self.btn_res_load.on_click(self._on_show_results)
        self.res_out = W.Output(layout=W.Layout(border="1px solid #ccc", min_height="300px"))
        self.btn_compare = W.Button(description="Compare two (select in list)")
        self.btn_compare.on_click(self._on_compare)
        self.btn_export = W.Button(description="Export experiment as ZIP")
        self.btn_export.on_click(self._on_export)

        results_box = W.VBox(
            [
                W.HTML("<b>Results explorer</b>"),
                W.HBox([self.res_select, self.btn_res_load, self.btn_compare, self.btn_export]),
                self.res_out,
            ]
        )

        self.tabs = W.Tab(children=[list_box, form_box, queue_box, results_box])
        self.tabs.set_title(0, "📚 Experiments")
        self.tabs.set_title(1, "✏️ Setup")
        self.tabs.set_title(2, "▶ Queue & Run")
        self.tabs.set_title(3, "📊 Results")

        self.root_box = W.VBox(
            [
                W.HTML(
                    "<h2 style='margin:4px 0'>RapidSegment Experiment Suite</h2>"
                    f"<small>Storage: <code>{SUITE_ROOT}</code></small>"
                ),
                self.tabs,
            ]
        )

    # ------------------------------------------------------------------ helpers
    def _refresh_experiment_list(self):
        ids = self.store.list_ids()
        labels = []
        for i in ids:
            try:
                m = self.store.load_meta(i)
                labels.append((f"{m.name}  [{m.status}]  ({i})", i))
            except Exception:
                labels.append((i, i))
        self.exp_select.options = labels
        self.res_select.options = labels
        self._refresh_queue_widget()

    def _refresh_queue_widget(self):
        opts = []
        for i, eid in enumerate(self.queue):
            try:
                m = self.store.load_meta(eid)
                opts.append((f"{i+1}. {m.name} [{m.status}]", eid))
            except Exception:
                opts.append((f"{i+1}. {eid}", eid))
        self.queue_select.options = opts

    def _toggle_source(self, change):
        is_local = change["new"] == "local"
        self.w_path.layout.display = None if is_local else "none"
        for w in (self.w_project, self.w_dataset, self.w_table):
            w.layout.display = "none" if is_local else None

    def _collect_data_source(self) -> DataSource:
        ign = [x.strip() for x in self.w_ignore.value.split(",") if x.strip()]
        return DataSource(
            kind=self.w_kind.value,
            path=self.w_path.value.strip(),
            project_id=self.w_project.value.strip(),
            dataset_id=self.w_dataset.value.strip(),
            table_id=self.w_table.value.strip(),
            target_col=self.w_target.value.strip(),
            primary_key=self.w_pk.value.strip(),
            ignore_features=ign,
        )

    def _parse_list_num(self, text: str, cast=float) -> List:
        if not text.strip():
            return []
        out = []
        for part in text.split(","):
            part = part.strip()
            if part:
                out.append(cast(part))
        return out

    def _collect_params(self) -> BuilderParams:
        ign = [x.strip() for x in self.w_ignore.value.split(",") if x.strip()]
        return BuilderParams(
            n_jobs=int(self.w_njobs.value),
            min_sample_size=int(self.w_min_sample.value),
            min_lift=float(self.w_min_lift.value),
            min_events=int(self.w_min_events.value),
            top_n_vars=int(self.w_top_n.value),
            max_segments=int(self.w_max_seg.value),
            max_feature_reuse=int(self.w_reuse.value),
            enable_diversity=bool(self.w_diversity.value),
            enable_1way=bool(self.w_1way.value),
            enable_2way=bool(self.w_2way.value),
            enable_3way=bool(self.w_3way.value),
            sort_priority=self.w_sort.value,
            binning_method=self.w_binning.value,
            naive_bins=int(self.w_naive_bins.value),
            max_expansion_hops=int(self.w_hops.value),
            selection_metric=self.w_sel_metric.value,
            expand_log_mode=self.w_expand_log.value,
            param_grid_min_sample=self._parse_list_num(self.w_grid_sample.value, int),
            param_grid_min_lift=self._parse_list_num(self.w_grid_lift.value, float),
            feature_groups_json=self.w_feat_groups.value.strip() or "{}",
            ignore_features=ign,
        )

    def _populate_form(self, meta: ExperimentMeta):
        self.w_name.value = meta.name
        self.w_notes.value = meta.notes or ""
        self.w_tags.value = ",".join(meta.tags or [])
        ds = meta.data_source or {}
        self.w_kind.value = ds.get("kind", "local")
        self.w_path.value = ds.get("path", "")
        self.w_project.value = ds.get("project_id", "")
        self.w_dataset.value = ds.get("dataset_id", "")
        self.w_table.value = ds.get("table_id", "")
        self.w_target.value = ds.get("target_col", "")
        self.w_pk.value = ds.get("primary_key", "")
        self.w_ignore.value = ",".join(ds.get("ignore_features") or [])

        bp = meta.builder_params or {}
        self.w_min_sample.value = bp.get("min_sample_size", 1000)
        self.w_min_lift.value = bp.get("min_lift", 2.0)
        self.w_min_events.value = bp.get("min_events", 5)
        self.w_top_n.value = bp.get("top_n_vars", 20)
        self.w_max_seg.value = bp.get("max_segments", 10)
        self.w_reuse.value = bp.get("max_feature_reuse", 1)
        self.w_njobs.value = bp.get("n_jobs", -1)
        self.w_1way.value = bp.get("enable_1way", True)
        self.w_2way.value = bp.get("enable_2way", True)
        self.w_3way.value = bp.get("enable_3way", True)
        self.w_diversity.value = bp.get("enable_diversity", False)
        self.w_sort.value = bp.get("sort_priority", "lift_rate_count")
        self.w_binning.value = bp.get("binning_method", "optimal")
        self.w_naive_bins.value = bp.get("naive_bins", 5)
        self.w_hops.value = bp.get("max_expansion_hops", 0)
        self.w_sel_metric.value = bp.get("selection_metric", "iv")
        self.w_expand_log.value = bp.get("expand_log_mode", "summary")
        self.w_grid_sample.value = ",".join(
            str(x) for x in bp.get("param_grid_min_sample", [1000])
        )
        self.w_grid_lift.value = ",".join(str(x) for x in bp.get("param_grid_min_lift", [2.0]))
        self.w_feat_groups.value = bp.get("feature_groups_json", "{}")
        self._current_meta = meta

    # ------------------------------------------------------------------ callbacks
    def _on_apply_preset(self, _):
        p = PRESETS.get(self.w_preset.value, {})
        mapping = {
            "min_sample_size": "w_min_sample",
            "min_lift": "w_min_lift",
            "max_segments": "w_max_seg",
            "top_n_vars": "w_top_n",
            "enable_1way": "w_1way",
            "enable_2way": "w_2way",
            "enable_3way": "w_3way",
            "sort_priority": "w_sort",
            "binning_method": "w_binning",
            "naive_bins": "w_naive_bins",
            "max_expansion_hops": "w_hops",
        }
        for k, v in p.items():
            wname = mapping.get(k)
            if wname and hasattr(self, wname):
                getattr(self, wname).value = v

    def _on_profile(self, _):
        self.profile_out.clear_output()
        with self.profile_out:
            try:
                ds = self._collect_data_source()
                if not ds.target_col:
                    print("⚠️ Please set the target column first.")
                    return
                print(f"Loading {ds.describe()} …")
                data = self.runner.load_data(ds)
                stats = self.runner.profile(data, ds.target_col)
                print(f"Rows          : {stats['n_rows']:,}")
                print(f"Columns       : {stats['n_columns']}")
                print(f"Event rate    : {stats.get('event_rate_pct')} %")
                if stats.get("target_balance"):
                    tb = stats["target_balance"]
                    print(f"  events={tb.get('events')}, non-events={tb.get('non_events')}")
                print("\nNull rates (top 15 by null %):")
                nr = sorted(
                    stats["null_rates_pct"].items(), key=lambda x: -(x[1] or 0)
                )[:15]
                for c, r in nr:
                    print(f"  {c:40s} {r}%")
                print("\nSample (first 5 rows):")
                if stats.get("sample_head"):
                    display(pd.DataFrame(stats["sample_head"]))
            except Exception as e:
                print(f"❌ {type(e).__name__}: {e}")
                traceback.print_exc()

    def _save_current(self, add_to_queue: bool = False) -> Optional[str]:
        name = self.w_name.value.strip() or "unnamed"
        tags = [t.strip() for t in self.w_tags.value.split(",") if t.strip()]
        ds = self._collect_data_source()
        params = self._collect_params()

        if self._current_meta and self._current_meta.name == name:
            meta = self._current_meta
            meta.notes = self.w_notes.value
            meta.tags = tags
        else:
            meta = self.store.create(name, notes=self.w_notes.value, tags=tags)

        meta.data_source = asdict(ds)
        meta.builder_params = asdict(params)
        meta.status = "draft"
        self.store.save_meta(meta)
        self._current_meta = meta
        self._refresh_experiment_list()
        if add_to_queue:
            if meta.exp_id not in self.queue:
                self.queue.append(meta.exp_id)
                self._refresh_queue_widget()
            self.status_html.value = f"<b>Saved & queued:</b> {meta.name} ({meta.exp_id})"
        else:
            self.status_html.value = f"<b>Saved draft:</b> {meta.name} ({meta.exp_id})"
        return meta.exp_id

    def _on_load(self, _):
        if not self.exp_select.value:
            return
        meta = self.store.load_meta(self.exp_select.value)
        self._populate_form(meta)
        self.tabs.selected_index = 1
        self.status_html.value = f"Loaded <b>{meta.name}</b> ({meta.status})"

    def _on_clone(self, _):
        if not self.exp_select.value:
            return
        src = self.store.load_meta(self.exp_select.value)
        meta = self.store.clone(src.exp_id, f"{src.name} (clone)")
        self._refresh_experiment_list()
        self._populate_form(meta)
        self.tabs.selected_index = 1
        self.status_html.value = f"Cloned → <b>{meta.name}</b>"

    def _on_delete(self, _):
        if not self.exp_select.value:
            return
        eid = self.exp_select.value
        self.store.delete(eid)
        if eid in self.queue:
            self.queue.remove(eid)
        self._refresh_experiment_list()
        self.status_html.value = f"Deleted {eid}"

    def _on_add_queue(self, _):
        if not self.exp_select.value:
            return
        eid = self.exp_select.value
        if eid not in self.queue:
            self.queue.append(eid)
            self._refresh_queue_widget()
        self.status_html.value = f"Queued {eid}"

    def _queue_move(self, direction: int):
        sel = list(self.queue_select.value)
        if not sel:
            return
        eid = sel[0]
        idx = self.queue.index(eid)
        new_idx = idx + direction
        if 0 <= new_idx < len(self.queue):
            self.queue[idx], self.queue[new_idx] = self.queue[new_idx], self.queue[idx]
            self._refresh_queue_widget()

    def _queue_remove(self, _):
        for eid in list(self.queue_select.value):
            if eid in self.queue:
                self.queue.remove(eid)
        self._refresh_queue_widget()

    def _queue_clear(self):
        self.queue.clear()
        self._refresh_queue_widget()

    def _on_run_queue(self, _):
        if not self.queue:
            self.status_html.value = "Queue is empty"
            return
        self._stop_flag = False
        self.btn_run.disabled = True
        total = len(self.queue)
        self.out_log.clear_output()

        for i, eid in enumerate(list(self.queue)):
            if self._stop_flag:
                with self.out_log:
                    print("⏹ Stopped by user after current experiment boundary.")
                break
            try:
                meta = self.store.load_meta(eid)
            except Exception as e:
                with self.out_log:
                    print(f"Could not load {eid}: {e}")
                continue

            self.status_html.value = f"Running <b>{i+1}/{total}</b>: {meta.name}"
            self.progress_label.value = f"{meta.name}"

            def prog(p, base=i, tot=total):
                self.progress.value = (base + p) / tot

            def stat(msg, name=meta.name):
                self.status_html.value = f"<b>{name}</b>: {msg}"
                with self.out_log:
                    print(msg)

            with self.out_log:
                print(f"\n===== START {meta.name} ({eid}) =====")
            meta = self.runner.run(meta, progress_cb=prog, status_cb=stat)
            with self.out_log:
                print(f"===== END {meta.name} → {meta.status} =====\n")

            if eid in self.queue:
                self.queue.remove(eid)
            self._refresh_queue_widget()
            self._refresh_experiment_list()

        self.btn_run.disabled = False
        self.progress.value = 1.0
        self.status_html.value = "<b>Queue finished</b>"
        self.progress_label.value = "done"

    def _on_show_results(self, _):
        self.res_out.clear_output()
        eid = self.res_select.value
        if not eid:
            return
        with self.res_out:
            meta = self.store.load_meta(eid)
            print(f"Experiment : {meta.name}")
            print(f"ID         : {meta.exp_id}")
            print(f"Status     : {meta.status}")
            print(f"Created    : {meta.created_at}")
            print(f"Notes      : {meta.notes}")
            print(f"Tags       : {meta.tags}")
            if meta.cloned_from:
                print(f"Cloned from: {meta.cloned_from}")
            print(
                f"Data       : {meta.data_source.get('kind')} → "
                f"{meta.data_source.get('path') or meta.data_source.get('table_id')}"
            )
            print(f"Target     : {meta.data_source.get('target_col')}")
            if meta.stats:
                print(
                    f"\nProfile: {meta.stats.get('n_rows'):,} rows × "
                    f"{meta.stats.get('n_columns')} cols | "
                    f"event rate {meta.stats.get('event_rate_pct')}%"
                )
            if meta.run_info:
                print(f"Run      : {meta.run_info}")
            if meta.error:
                print(f"ERROR    : {meta.error}")

            seg_path = self.store.path(eid, "segments.json")
            if seg_path.exists():
                segs = _json_load(seg_path)
                print(f"\n—— Segments ({len(segs)}) ——")
                df = pd.DataFrame(segs)
                cols = [
                    c
                    for c in ["segment_id", "count", "rate", "lift", "sql_filter", "rule_string"]
                    if c in df.columns
                ]
                display(df[cols] if cols else df)
            else:
                print("\n(no segments.json yet)")

            cov_path = self.store.path(eid, "coverage.json")
            if cov_path.exists():
                cov = _json_load(cov_path)
                print("\n—— Coverage (hierarchical evaluation on original data) ——")
                display(pd.DataFrame(cov))
            else:
                print("\n(no coverage.json yet)")

            log_path = self.store.path(eid, "logs.txt")
            if log_path.exists():
                print("\n—— Log tail (last 40 lines) ——")
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                print("\n".join(lines[-40:]))

    def _on_compare(self, _):
        self.res_out.clear_output()
        a = self.res_select.value
        b = self.exp_select.value
        if not a or not b or a == b:
            with self.res_out:
                print(
                    "Select two different experiments "
                    "(one in Results dropdown, one in Experiments list)."
                )
            return
        with self.res_out:
            for label, eid in [("A", a), ("B", b)]:
                meta = self.store.load_meta(eid)
                print(f"=== {label}: {meta.name} ({eid}) status={meta.status} ===")
                cov_path = self.store.path(eid, "coverage.json")
                if cov_path.exists():
                    display(pd.DataFrame(_json_load(cov_path)))
                else:
                    print("(no coverage)")
                print()

    def _on_export(self, _):
        eid = self.res_select.value
        if not eid:
            return
        src = self.store.path(eid)
        zip_path = SUITE_ROOT / f"{eid}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in src.rglob("*"):
                if f.is_file():
                    zf.write(f, arcname=str(f.relative_to(src)))
        with self.res_out:
            print(f"Exported → {zip_path}")


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------
def launch_suite(root: str | Path = SUITE_ROOT) -> ExperimentSuiteUI:
    """Create and display the Experiment Suite UI inside the current notebook."""
    ui = ExperimentSuiteUI(Path(root))
    ui.display()
    return ui
