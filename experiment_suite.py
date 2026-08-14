#!/usr/bin/env python3
"""
experiment_suite.py — configure, clone, edit and run RapidSegment experiments
=============================================================================

A lightweight experiment manager for RapidSegment. Build named experiment
configurations, clone/edit them, then run one or all of them with a single
call. The registry persists to a JSON file so configurations survive across
sessions. Designed so it can later be wrapped in an ipywidgets UI.

Quick start (programmatic)
--------------------------
    from experiment_suite import ExperimentSuite

    suite = ExperimentSuite()                       # loads registry if present
    suite.create("baseline", method="naive", max_segments=5)
    suite.clone("baseline", "deep_hops",
                max_segments=8, max_expansion_hops=2, binning_method="optimal")
    suite.edit("baseline", min_lift=2.0)
    suite.list()

    result = suite.run("deep_hops", data=df)        # one-click run
    print(result.summary_df())

    suite.run_all(data=df)                          # run everything

CLI
---
    python3 experiment_suite.py create baseline --method naive --max_segments 5
    python3 experiment_suite.py clone baseline deep_hops --max_segments 8
    python3 experiment_suite.py edit baseline --min_lift 2.0
    python3 experiment_suite.py list
    python3 experiment_suite.py show baseline
    python3 experiment_suite.py remove baseline
    python3 experiment_suite.py run deep_hops --data Examples/bank-full.csv
    python3 experiment_suite.py run-all --data Examples/bank-full.csv
"""

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import duckdb

from rapidsegment import StrategicSegmentBuilder, StrategicSegmentScore, UniversalDataLoader

# -----------------------------------------------------------------------------
# Parameter schema
# -----------------------------------------------------------------------------
PARAM_DEFAULTS: Dict[str, Any] = {
    # data
    "data_path": None,
    "target": "Target",
    "max_rows": 0,
    # builder
    "n_jobs": -1,
    "min_sample_size": 1000,
    "min_lift": 1.5,
    "min_events": 100,
    "top_n_vars": 15,
    "max_segments": 5,
    "max_feature_reuse": 1,
    "param_grid": None,
    "enable_diversity": False,
    "enable_1way": True,
    "enable_2way": True,
    "enable_3way": True,
    "feature_groups": None,
    "ignore_features": None,
    "sort_priority": "rate_lift_count",
    "binning_method": "optimal",
    "naive_bins": 5,
    "max_expansion_hops": 0,
    "selection_metric": "iv",
    "expand_log_mode": "none",
    # scoring
    "score": False,
    "primary_key": None,
    "score_export_path": None,
}

_DATA_PARAMS = {"data_path", "target", "max_rows"}
_SCORING_PARAMS = {"score", "primary_key", "score_export_path"}
_BUILDER_PARAMS = set(PARAM_DEFAULTS) - _DATA_PARAMS - _SCORING_PARAMS

_TRUE_TOKENS = {"1", "1.0", "true", "yes", "y", "t"}
_FALSE_TOKENS = {"0", "0.0", "false", "no", "n", "f"}

SUPPORTED_DATA_FIELDS = ("data_path", "target", "max_rows")


def _normalize_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Ensure the target column is numeric binary (0/1)."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Available: {list(df.columns)}")
    col = df[target]
    if pd.api.types.is_string_dtype(col) or pd.api.types.is_bool_dtype(col):
        mapping = {}
        for v in col.dropna().unique():
            vs = str(v).strip().lower()
            if vs in _TRUE_TOKENS:
                mapping[v] = 1
            elif vs in _FALSE_TOKENS:
                mapping[v] = 0
        df = df.copy()
        df[target] = col.map(mapping).astype("int64")
    else:
        df = df.copy()
        df[target] = df[target].astype("int64")
    return df


def _load_data(data_path: str, target: str, max_rows: int = 0) -> pd.DataFrame:
    """Load a data file (CSV via pandas, others via UniversalDataLoader)."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if data_path.lower().endswith(".csv"):
        df = pd.read_csv(data_path)
    else:
        table = UniversalDataLoader(file_path=data_path).load()
        df = table.to_pandas()
    if max_rows and max_rows > 0:
        df = df.iloc[:max_rows].copy()
    return _normalize_target(df, target)


class ExperimentConfig:
    """A single named experiment configuration."""

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None) -> None:
        if not name or not str(name).strip():
            raise ValueError("Experiment name cannot be empty.")
        self.name = str(name).strip()
        self.params: Dict[str, Any] = {}
        for key, default in PARAM_DEFAULTS.items():
            self.params[key] = default
        if params:
            self.update(params)

    def update(self, updates: Dict[str, Any]) -> None:
        unknown = set(updates) - set(PARAM_DEFAULTS)
        if unknown:
            raise ValueError(
                f"Unknown experiment parameters: {sorted(unknown)}. "
                f"Valid keys: {sorted(PARAM_DEFAULTS)}"
            )
        self.params.update(updates)

    @property
    def builder_kwargs(self) -> Dict[str, Any]:
        return {
            k: v for k, v in self.params.items()
            if k in _BUILDER_PARAMS and v is not None
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, **self.params}


class ExperimentResult:
    """Result of a single experiment run."""

    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        builder: StrategicSegmentBuilder,
        segments: List[Dict[str, Any]],
        coverage: List[Dict[str, Any]],
        model: Optional[Dict[str, Any]],
        elapsed: float,
    ) -> None:
        self.name = name
        self.config = config
        self.builder = builder
        self.segments = segments
        self.coverage = coverage
        self.model = model
        self.elapsed = elapsed
        self.stop_reason = builder.stop_reason

    def summary_df(self) -> pd.DataFrame:
        rows = []
        for s in self.segments:
            rows.append({
                "segment_id": s["segment_id"],
                "rule_string": s["rule_string"],
                "count": s["count"],
                "rate_%": round(s["rate"], 4),
                "lift": round(s["lift"], 4),
                "sql_filter": s["sql_filter"],
            })
        return pd.DataFrame(rows)

    def coverage_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.coverage)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "config": self.config,
            "stop_reason": self.stop_reason,
            "n_segments": len(self.segments),
            "elapsed_seconds": round(self.elapsed, 2),
            "segments": self.segments,
            "coverage": self.coverage,
            "model": self.model,
        }


class ExperimentSuite:
    """
    Registry of named experiment configurations with run capabilities.

    Args:
        registry_path: Optional path to a JSON file for persisting configs.
    """

    def __init__(self, registry_path: str = "experiment_suite.json") -> None:
        self.registry_path = registry_path
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.results: Dict[str, ExperimentResult] = {}
        if os.path.exists(registry_path):
            self._load(registry_path)

    # ------------------------------------------------------------------ CRUD
    def create(self, name: str, **overrides) -> ExperimentConfig:
        """Create a new experiment from defaults, optionally overriding params."""
        if name in self.experiments:
            raise ValueError(f"Experiment '{name}' already exists.")
        cfg = ExperimentConfig(name, overrides)
        self.experiments[name] = cfg
        return cfg

    def clone(self, source: str, new_name: str, **overrides) -> ExperimentConfig:
        """Clone an existing experiment (editable before insertion)."""
        if source not in self.experiments:
            raise ValueError(f"Unknown experiment '{source}'. Existing: {self.list_names()}")
        if new_name in self.experiments:
            raise ValueError(f"Experiment '{new_name}' already exists.")
        cfg = ExperimentConfig(new_name, self.experiments[source].params)
        cfg.update(overrides)
        self.experiments[new_name] = cfg
        return cfg

    def edit(self, name: str, **updates) -> ExperimentConfig:
        """Edit one or more parameters of an existing experiment."""
        if name not in self.experiments:
            raise ValueError(f"Unknown experiment '{name}'. Existing: {self.list_names()}")
        self.experiments[name].update(updates)
        return self.experiments[name]

    def remove(self, name: str) -> None:
        """Remove an experiment from the registry."""
        if name not in self.experiments:
            raise ValueError(f"Unknown experiment '{name}'. Existing: {self.list_names()}")
        del self.experiments[name]
        self.results.pop(name, None)

    def get(self, name: str) -> ExperimentConfig:
        if name not in self.experiments:
            raise ValueError(f"Unknown experiment '{name}'. Existing: {self.list_names()}")
        return self.experiments[name]

    def show(self, name: str) -> Dict[str, Any]:
        return self.get(name).to_dict()

    def list_names(self) -> List[str]:
        return list(self.experiments.keys())

    def list(self) -> pd.DataFrame:
        """Summary table of all configured experiments."""
        rows = []
        for cfg in self.experiments.values():
            p = cfg.params
            rows.append({
                "name": cfg.name,
                "method": p["binning_method"],
                "max_segments": p["max_segments"],
                "min_lift": p["min_lift"],
                "min_sample_size": p["min_sample_size"],
                "top_n_vars": p["top_n_vars"],
                "hops": p["max_expansion_hops"],
                "naive_bins": p["naive_bins"],
                "score": p["score"],
                "data_path": p["data_path"],
                "target": p["target"],
            })
        return pd.DataFrame(rows)

    # ---------------------------------------------------------------- running
    def run(
        self,
        name: str,
        data: Optional[Union[pd.DataFrame, str]] = None,
        verbose: bool = True,
    ) -> ExperimentResult:
        """
        Run an experiment end-to-end: load data, extract segments, evaluate
        coverage and (optionally) build a scorecard.

        Args:
            name: Experiment name in the registry.
            data: A pandas DataFrame, a path to a data file, or None to use the
                  experiment's configured `data_path`.
            verbose: Whether to print a short result summary.

        Returns:
            ExperimentResult with segments, coverage, model and timing.
        """
        if name not in self.experiments:
            raise ValueError(f"Unknown experiment '{name}'. Existing: {self.list_names()}")
        cfg = self.experiments[name]
        p = cfg.params
        target = p["target"]

        if isinstance(data, str):
            df = _load_data(data, target, p["max_rows"])
        elif data is not None:
            df = _normalize_target(data.copy(), target)
        else:
            if not p["data_path"]:
                raise ValueError(
                    f"Experiment '{name}' has no data_path and no data was provided."
                )
            df = _load_data(p["data_path"], target, p["max_rows"])

        builder = StrategicSegmentBuilder(target=target, **cfg.builder_kwargs)
        t0 = time.time()
        segments = builder.extract_segments(df)
        coverage = builder.evaluate_final_coverage(df)
        model = None
        if p["score"]:
            model = self._score(df, builder, p)

        result = ExperimentResult(
            name=name,
            config=cfg.to_dict(),
            builder=builder,
            segments=segments,
            coverage=coverage,
            model=model,
            elapsed=time.time() - t0,
        )
        self.results[name] = result
        if verbose:
            print(
                f"[{name}] segments={len(segments)} stop_reason={result.stop_reason!r} "
                f"time={result.elapsed:.1f}s"
            )
            if segments:
                top = segments[0]
                print(
                    f"  top segment: count={top['count']} lift={top['lift']:.2f}x "
                    f"rate={top['rate']:.2f}% rule={top['rule_string']}"
                )
        return result

    def run_all(
        self,
        data: Optional[Union[pd.DataFrame, str]] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Run every registered experiment and return a comparison table.
        """
        if not self.experiments:
            raise ValueError("No experiments configured. Use create() first.")
        rows = []
        for name in self.list_names():
            result = self.run(name, data=data, verbose=verbose)
            rows.append({
                "name": name,
                "method": result.config["binning_method"],
                "n_segments": len(result.segments),
                "stop_reason": result.stop_reason,
                "top_count": result.segments[0]["count"] if result.segments else None,
                "top_lift": round(result.segments[0]["lift"], 4) if result.segments else None,
                "top_rule": result.segments[0]["rule_string"] if result.segments else None,
                "coverage_%": round(sum(r["capture_rate"] for r in result.coverage if r["segment"] != 0), 2),
                "elapsed_s": round(result.elapsed, 2),
            })
        return pd.DataFrame(rows)

    def _score(
        self, df: pd.DataFrame, builder: StrategicSegmentBuilder, p: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build a scorecard from the extracted segments."""
        segments = builder.segments
        if not segments:
            raise ValueError(f"Experiment produced no segments; cannot score.")
        export_path = p["score_export_path"] or os.path.join(
            "experiments", f"{time.strftime('%Y%m%d_%H%M%S')}_model.json"
        )
        os.makedirs(os.path.dirname(export_path) or ".", exist_ok=True)

        primary_key = p["primary_key"]
        scoring_df = pd.DataFrame({p["target"]: df[p["target"]]})
        if primary_key is None:
            primary_key = "__rs_pk"
            scoring_df[primary_key] = range(len(df))
        else:
            if primary_key not in df.columns:
                raise ValueError(f"primary_key '{primary_key}' not found in data.")
            scoring_df[primary_key] = df[primary_key]

        segment_cols = []
        con = duckdb.connect()
        con.register("base", df)
        for seg in segments:
            col = f"SEG_{seg['segment_id']}"
            scoring_df[col] = con.execute(
                f"SELECT CAST(({seg['sql_filter']}) AS INTEGER) FROM base"
            ).df().iloc[:, 0].astype(int)
            segment_cols.append(col)
        con.close()

        scorer = StrategicSegmentScore(
            target_col=p["target"],
            primary_key=primary_key,
            segment_cols=segment_cols,
        )
        return scorer.calculate_and_export_weights(scoring_df, export_path)

    # ------------------------------------------------------------ persistence
    def save(self, path: Optional[str] = None) -> str:
        path = path or self.registry_path
        payload = {
            "experiments": [cfg.to_dict() for cfg in self.experiments.values()],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        self.registry_path = path
        return path

    def _load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for entry in payload.get("experiments", []):
            params = {k: v for k, v in entry.items() if k != "name"}
            cfg = ExperimentConfig(entry["name"], params)
            self.experiments[cfg.name] = cfg
        self.registry_path = path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RapidSegment experiment suite")
    p.add_argument("--registry", default="experiment_suite.json", help="registry JSON path")
    sub = p.add_subparsers(dest="command", required=True)

    def _add_overrides(sp):
        for key in sorted(PARAM_DEFAULTS):
            sp.add_argument(f"--{key}", default=None)

    pc = sub.add_parser("create", help="create a new experiment")
    pc.add_argument("name")
    _add_overrides(pc)

    pl = sub.add_parser("clone", help="clone an experiment")
    pl.add_argument("source")
    pl.add_argument("new_name")
    _add_overrides(pl)

    pe = sub.add_parser("edit", help="edit an experiment")
    pe.add_argument("name")
    _add_overrides(pe)

    pr = sub.add_parser("remove", help="remove an experiment")
    pr.add_argument("name")

    sub.add_parser("list", help="list experiments")

    ps = sub.add_parser("show", help="show an experiment config")
    ps.add_argument("name")

    pru = sub.add_parser("run", help="run one experiment")
    pru.add_argument("name")
    pru.add_argument("--data", default=None)
    pru.add_argument("--quiet", action="store_true")

    pra = sub.add_parser("run-all", help="run all experiments")
    pra.add_argument("--data", default=None)
    pra.add_argument("--quiet", action="store_true")
    return p


def _collect_overrides(args) -> Dict[str, Any]:
    out = {}
    for key in PARAM_DEFAULTS:
        val = getattr(args, key, None)
        if val is None:
            continue
        out[key] = _coerce(key, val)
    return out


def _coerce(key: str, val: str) -> Any:
    default = PARAM_DEFAULTS[key]
    if key in ("param_grid", "feature_groups"):
        return json.loads(val) if val not in ("", "None") else None
    if key in ("ignore_features",):
        return json.loads(val) if val not in ("", "None") else None
    if isinstance(default, bool):
        return str(val).lower() in ("1", "true", "yes")
    if isinstance(default, int):
        return int(float(val))
    if isinstance(default, float):
        return float(val)
    return val


def main() -> int:
    args = _build_parser().parse_args()
    suite = ExperimentSuite(args.registry)

    if args.command == "create":
        suite.create(args.name, **_collect_overrides(args))
        suite.save()
        print(f"Created experiment '{args.name}'.")
    elif args.command == "clone":
        suite.clone(args.source, args.new_name, **_collect_overrides(args))
        suite.save()
        print(f"Cloned '{args.source}' -> '{args.new_name}'.")
    elif args.command == "edit":
        suite.edit(args.name, **_collect_overrides(args))
        suite.save()
        print(f"Edited experiment '{args.name}'.")
    elif args.command == "remove":
        suite.remove(args.name)
        suite.save()
        print(f"Removed experiment '{args.name}'.")
    elif args.command == "list":
        df = suite.list()
        print(df.to_string(index=False) if len(df) else "No experiments configured.")
    elif args.command == "show":
        print(json.dumps(suite.show(args.name), indent=2, default=str))
    elif args.command == "run":
        result = suite.run(args.name, data=args.data, verbose=not args.quiet)
        print(result.summary_df().to_string(index=False))
    elif args.command == "run-all":
        cmp_df = suite.run_all(data=args.data, verbose=not args.quiet)
        print(cmp_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
