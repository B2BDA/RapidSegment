#!/usr/bin/env python3
"""
stress_test.py — RapidSegment health-check suite
================================================

Checks the health of the RapidSegment library with clear pass/fail reporting.

Test modes
----------
fast:
    Smoke test on a small sample dataset. Verifies every module runs without
    errors: data loading, segment extraction (naive + optimal), SQL filter
    execution, coverage consistency, scorecard building, feature health
    reports, diagnostics and a quick determinism probe.

deep:
    Full battery used to validate the library end-to-end:
      * determinism across in-process repeats and subprocesses with different
        PYTHONHASHSEED values
      * parameter-matrix stress runs (segment count, expansion hops, bin count,
        grid search, sort priorities, top-N features, small subsets)
      * edge cases (tiny data, NaN/null values, impossible thresholds,
        ignore_features, feature groups/diversity, boolean target)
      * ground-truth metric validation against hierarchical evaluation
      * optimal-binning row-alignment checks (1-way and 2-way)
      * scorer deep validation and data-loader format round-trips

Usage
-----
    python3 stress_test.py --mode fast
    python3 stress_test.py --mode deep
    python3 stress_test.py --mode deep --data path/to/data.csv --target Target
    python3 stress_test.py --mode deep --rows 10000 --out report.json

Exit code is 0 when every check passes, 1 otherwise.
"""

import argparse
import contextlib
import io
import itertools
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import duckdb
import pyarrow as pa
import pyarrow.parquet as pa_pq

from rapidsegment import (
    StrategicSegmentBuilder,
    StrategicSegmentScore,
    UniversalDataLoader,
)

DEFAULT_DATA = "Examples/bank-full.csv"
DEFAULT_TARGET = "Target"

_TRUE_TOKENS = {"1", "1.0", "true", "yes", "y", "t"}
_FALSE_TOKENS = {"0", "0.0", "false", "no", "n", "f"}


# -----------------------------------------------------------------------------
# Canonical signature
# -----------------------------------------------------------------------------
def signature(builder: StrategicSegmentBuilder) -> Dict[str, Any]:
    """Deterministic canonical signature of an extraction run."""
    segs = []
    for s in builder.segments:
        segs.append(
            {
                "segment_id": s["segment_id"],
                "rule_string": s["rule_string"],
                "sql_filter": s["sql_filter"],
                "count": int(s["count"]),
                "rate": round(float(s["rate"]), 6),
                "lift": round(float(s["lift"]), 6),
            }
        )
    diags = []
    for d in builder.diagnostics_:
        diags.append(
            {
                "iteration": d.get("iteration"),
                "winning_segment": d.get("winning_segment"),
                "candidate_funnel": d.get("candidate_funnel"),
                "near_miss": d.get("near_miss"),
            }
        )
    return {
        "stop_reason": builder.stop_reason,
        "n_segments": len(segs),
        "segments": segs,
        "diagnostics": diags,
    }


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------
def prepare_data(path: str, target: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Load a CSV and normalise the target column to numeric binary (0/1)."""
    df = pd.read_csv(path)
    if max_rows and max_rows > 0:
        df = df.iloc[:max_rows].copy()
    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found. Available: {list(df.columns)}"
        )
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


# -----------------------------------------------------------------------------
# Test runner
# -----------------------------------------------------------------------------
class Suite:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.results: List[Dict[str, Any]] = []
        self._section_name = "setup"

    def section(self, name: str):
        self._section_name = name

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.results.append(
            {
                "section": self._section_name,
                "name": name,
                "ok": bool(ok),
                "detail": str(detail)[:400],
            }
        )
        status = "PASS" if ok else "FAIL"
        print(f"[{self.mode.upper()}] {status:4} | {name}")
        if detail:
            print(f"      -> {detail}")
        return bool(ok)

    def check_no_raise(self, name: str, fn, detail_fmt: Optional[str] = None) -> Tuple[bool, Any]:
        t0 = time.time()
        try:
            result = fn()
            ok = True
            detail = detail_fmt(result) if detail_fmt else ""
        except Exception as e:  # noqa: BLE001 - we report any failure
            ok = False
            result = None
            detail = f"{type(e).__name__}: {e}"
        self.check(name, ok, detail)
        self.results[-1]["time"] = round(time.time() - t0, 2)
        return ok, result

    def summary(self) -> Dict[str, Any]:
        total = len(self.results)
        passed = sum(1 for r in self.results if r["ok"])
        failed = total - passed
        print("\n" + "=" * 70)
        print(f"RAPIDSEGMENT STRESS TEST — mode={self.mode.upper()}")
        print(f"  TOTAL: {total}  |  PASS: {passed}  |  FAIL: {failed}")
        if failed:
            print("\nFAILED CHECKS:")
            for r in self.results:
                if not r["ok"]:
                    print(f"  - [{r['section']}] {r['name']}: {r['detail']}")
        print("=" * 70)
        return {
            "mode": self.mode,
            "total": total,
            "passed": passed,
            "failed": failed,
            "results": self.results,
        }


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------
def build_and_extract(data: pd.DataFrame, **kw) -> Tuple[StrategicSegmentBuilder, List[Dict[str, Any]]]:
    builder = StrategicSegmentBuilder(target=DEFAULT_TARGET, **kw)
    segments = builder.extract_segments(data)
    return builder, segments


def hierarchical_ground_truth(data: pd.DataFrame, builder: StrategicSegmentBuilder) -> List[str]:
    """Re-derive each segment's metrics hierarchically on the original data.

    Returns a list of problem descriptions (empty == all metrics correct).
    """
    con = duckdb.connect()
    con.register("orig", data)
    con.execute(f'CREATE OR REPLACE TABLE residual AS SELECT * FROM orig')
    base_rate = data[DEFAULT_TARGET].mean()
    problems: List[str] = []
    for seg in builder.segments:
        row = con.execute(
            f'SELECT COUNT(*), SUM(CAST("{DEFAULT_TARGET}" AS DOUBLE)) '
            f'FROM residual WHERE {seg["sql_filter"]}'
        ).fetchone()
        cnt, evt = row[0], row[1] or 0
        rate = (evt / cnt) * 100.0 if cnt else 0.0
        lift = rate / (base_rate * 100.0) if base_rate else 0.0
        if (
            int(cnt) != int(seg["count"])
            or abs(rate - seg["rate"]) > 1e-6
            or abs(lift - seg["lift"]) > 1e-6
        ):
            problems.append(
                f"seg {seg['segment_id']} reported(count={seg['count']},rate={seg['rate']:.4f},"
                f"lift={seg['lift']:.4f}) vs actual(count={cnt},rate={rate:.4f},lift={lift:.4f}) "
                f"rule={seg['rule_string']}"
            )
        con.execute(f'DELETE FROM residual WHERE {seg["sql_filter"]}')
    con.close()
    return problems


def score_model(data: pd.DataFrame, builder: StrategicSegmentBuilder, export_dir: str) -> Dict[str, Any]:
    """Run the full builder->scorer pipeline. Returns the model artifact."""
    segments = builder.segments
    scoring_df = data[[DEFAULT_TARGET]].copy()
    scoring_df["__pk"] = [f"ROW_{i}" for i in range(len(data))]
    segment_cols = []
    con = duckdb.connect()
    con.register("base", data)
    for seg in segments:
        col = f"SEG_{seg['segment_id']}"
        scoring_df[col] = con.execute(
            f"SELECT CAST(({seg['sql_filter']}) AS INTEGER) FROM base"
        ).df().iloc[:, 0].astype(int)
        segment_cols.append(col)
    con.close()

    export_path = os.path.join(export_dir, "model.json")
    scorer = StrategicSegmentScore(
        target_col=DEFAULT_TARGET,
        primary_key="__pk",
        segment_cols=segment_cols,
    )
    return scorer.calculate_and_export_weights(scoring_df, export_path)


# -----------------------------------------------------------------------------
# Subprocess single-run mode (used by deep determinism + stress battery)
# -----------------------------------------------------------------------------
def subprocess_signature(
    kw: Dict[str, Any],
    hash_seed: Optional[int] = None,
    data_path: str = "",
) -> Dict[str, Any]:
    """Run a single extraction in a subprocess and return its canonical signature."""
    env = os.environ.copy()
    if hash_seed is not None:
        env["PYTHONHASHSEED"] = str(hash_seed)
    cmd = [sys.executable, os.path.abspath(__file__), "--sub", "--data", data_path]
    for k, v in kw.items():
        cmd.append(f"--{k}")
        if not isinstance(v, bool):
            cmd.append(str(v))
        else:
            cmd.append("1" if v else "0")
    proc = subprocess.run(
        cmd, capture_output=True, text=True, env=env, timeout=1800
    )
    try:
        out = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return {"_ok": False, "error": f"rc={proc.returncode}; stderr={proc.stderr[-600:]}"}
    out["_hash_seed"] = hash_seed
    return out


# -----------------------------------------------------------------------------
# Test sections
# -----------------------------------------------------------------------------
class FastTests:
    def __init__(self, suite: Suite, data: pd.DataFrame, scratch: str, data_path: str) -> None:
        self.s = suite
        self.data = data
        self.scratch = scratch
        self.data_path = data_path
        self.features = [c for c in data.columns if c != DEFAULT_TARGET]

    def run_data_loader(self) -> None:
        self.s.section("data_loader")
        csv_path = os.path.join(self.scratch, "input.csv")
        self.data.to_csv(csv_path, index=False)
        self.s.check_no_raise(
            "UniversalDataLoader(file_path) returns pyarrow table",
            lambda: UniversalDataLoader(file_path=csv_path).load(),
            lambda t: f"rows={len(t)} cols={t.num_columns}",
        )
        self.s.check_no_raise(
            "UniversalDataLoader(fallback_data) returns data as-is",
            lambda: UniversalDataLoader().load(fallback_data=self.data),
            lambda t: f"rows={len(t)}",
        )

        def _no_config():
            try:
                UniversalDataLoader().load()
                return "unexpected: no error"
            except ValueError as e:
                return str(e)

        ok, res = self.s.check_no_raise(
            "UniversalDataLoader without config raises ValueError",
            _no_config,
        )
        if ok:
            self.s.check("  ... error message is informative", "Invalid Configuration" in str(res))

    def run_builder_smoke(self) -> None:
        self.s.section("builder_smoke")
        for method in ("naive", "optimal"):
            ok, (builder, segments) = self.s.check_no_raise(
                f"extract_segments works with binning_method={method}",
                lambda m=method: build_and_extract(self.data, binning_method=m, max_segments=2),
                lambda r: f"segments={len(r[1])} stop_reason={r[0].stop_reason!r}",
            )
            if ok:
                self.s.check(
                    f"  {method}: at least one segment generated",
                    len(segments) >= 1,
                    f"segments={len(segments)}",
                )
                self.s.check(
                    f"  {method}: stop_reason recorded",
                    builder.stop_reason is not None,
                    f"stop_reason={builder.stop_reason!r}",
                )

    def run_sql_filters(self) -> None:
        self.s.section("sql_filters")
        builder, segments = build_and_extract(self.data, binning_method="naive", max_segments=2)
        self.s.check("  naive run produced segments for SQL test", len(segments) > 0, f"segs={len(segments)}")
        excluded = []
        for seg in segments:
            where = seg["sql_filter"]
            if excluded:
                where = f"({where}) AND NOT ({' OR '.join(excluded)})"
            con = duckdb.connect()
            con.register("tbl", self.data)
            cnt, _ = con.execute(
                f'SELECT COUNT(*), SUM(CAST("{DEFAULT_TARGET}" AS DOUBLE)) FROM tbl WHERE ({where})'
            ).fetchone()
            con.close()
            self.s.check(
                f"  SQL filter matches hierarchical count for seg {seg['segment_id']}",
                cnt == seg["count"],
                f"sql_count={cnt} vs reported={seg['count']} rule={seg['rule_string']}",
            )
            excluded.append(seg["sql_filter"])

    def run_coverage(self) -> None:
        self.s.section("coverage")
        builder, segments = build_and_extract(self.data, binning_method="naive", max_segments=2)
        ok, cov = self.s.check_no_raise(
            "evaluate_final_coverage returns rows",
            lambda: builder.evaluate_final_coverage(self.data),
            lambda r: f"cov_rows={len(r)} segments={len(segments)}",
        )
        if not ok or not cov:
            return
        seg_map = {s["segment_id"]: s for s in segments}
        all_match = True
        for row in cov:
            if row["segment"] != 0:
                s = seg_map.get(row["segment"])
                if s is None or row["total_count"] != s["count"]:
                    all_match = False
        self.s.check(
            "  per-segment counts match reported counts",
            all_match,
        )
        total_cov = sum(r["total_count"] for r in cov)
        self.s.check(
            "  coverage sums to full population",
            total_cov == len(self.data),
            f"cov_total={total_cov} population={len(self.data)}",
        )

    def run_metric_sanity(self) -> None:
        self.s.section("metric_sanity")
        builder, segments = build_and_extract(self.data, binning_method="naive", max_segments=2)
        sane = all(
            s["count"] > 0 and 0.0 <= s["rate"] <= 100.0 and s["lift"] >= 1.0
            for s in segments
        )
        self.s.check(
            "  all segment metrics within sane ranges",
            sane,
            f"counts={[s['count'] for s in segments]} lifts={[round(s['lift'],2) for s in segments]}",
        )

    def run_scorer(self) -> None:
        self.s.section("scorer")
        builder, segments = build_and_extract(self.data, binning_method="naive", max_segments=2)
        if len(segments) == 0:
            self.s.check("  builder produced segments for scorer test", False, "no segments")
            return
        ok, model = self.s.check_no_raise(
            "StrategicSegmentScore end-to-end produces a model artifact",
            lambda: score_model(self.data, builder, self.scratch),
            lambda r: f"keys={sorted(r.keys())}",
        )
        if ok:
            for key in ("model_metadata", "segment_weights", "decile_min_thresholds"):
                self.s.check(f"  artifact contains '{key}'", key in model, "")
            self.s.check(
                "  10 decile thresholds present",
                len(model.get("decile_min_thresholds", {})) == 10,
                f"n={len(model.get('decile_min_thresholds', {}))}",
            )
            self.s.check(
                "  weights are integers",
                all(isinstance(w["weight"], int) for w in model["segment_weights"].values()),
                str({k: v["weight"] for k, v in model["segment_weights"].items()}),
            )

    def run_health_report(self) -> None:
        self.s.section("health_report")
        builder, segments = build_and_extract(self.data, binning_method="naive", max_segments=2)
        ok, report = self.s.check_no_raise(
            "generate_feature_health_report returns a DataFrame",
            lambda: builder.generate_feature_health_report(self.data, self.features[:4]),
            lambda r: f"rows={len(r)} cols={list(r.columns)}",
        )
        if ok:
            self.s.check("  health report non-empty", len(report) > 0, f"rows={len(report)}")

    def run_diagnostics(self) -> None:
        self.s.section("diagnostics")
        builder, segments = build_and_extract(self.data, binning_method="naive", max_segments=2)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                builder.explain_feature_journey(self.features[0])
                journey_ok = True
                journey_detail = ""
            except Exception as e:  # noqa: BLE001
                journey_ok = False
                journey_detail = f"{type(e).__name__}: {e}"
        self.s.check("  explain_feature_journey does not raise", journey_ok, journey_detail)

        ok, reason = self.s.check_no_raise(
            "  explain_no_segments returns a string",
            lambda: builder.explain_no_segments(),
            lambda r: f"len={len(r)}",
        )
        if ok:
            self.s.check("  explain_no_segments is informative", len(reason) > 0, "")

    def run_determinism_smoke(self) -> None:
        self.s.section("determinism_smoke")
        b1, _ = build_and_extract(self.data, binning_method="naive", max_segments=2)
        b2, _ = build_and_extract(self.data, binning_method="naive", max_segments=2)
        self.s.check(
            "  two identical runs produce identical signatures",
            _canonical_json(signature(b1)) == _canonical_json(signature(b2)),
        )


class DeepTests(FastTests):
    def run_determinism_battery(self) -> None:
        self.s.section("determinism_battery")
        for method in ("naive", "optimal"):
            try:
                inproc = [
                    signature(*build_and_extract(
                        self.data, binning_method=method, max_segments=3
                    ))
                    for _ in range(2)
                ]
                inproc_identical = _canonical_json(inproc[0]) == _canonical_json(inproc[1])
            except Exception as e:  # noqa: BLE001
                self.s.check(
                    f"  {method}: in-process repeats identical",
                    False, f"{type(e).__name__}: {e}",
                )
                continue
            self.s.check(
                f"  {method}: in-process repeats identical",
                inproc_identical,
            )
            proc_refs = [
                subprocess_signature(
                    {"method": method, "max_segments": 3, "n_rows": len(self.data)},
                    hash_seed=seed,
                    data_path=self.data_path,
                )
                for seed in (111, 222)
            ]
            ok = all(p.get("_ok", True) for p in proc_refs)
            identical = (
                ok
                and _canonical_json({k: v for k, v in inproc[0].items() if not k.startswith("_")})
                == _canonical_json({k: v for k, v in proc_refs[0].items() if not k.startswith("_")})
            )
            all_same = identical and (
                _canonical_json({k: v for k, v in proc_refs[0].items() if not k.startswith("_")})
                == _canonical_json({k: v for k, v in proc_refs[1].items() if not k.startswith("_")})
            )
            self.s.check(
                f"  {method}: subprocess runs (2 hash seeds) identical & match in-process",
                all_same,
                f"ok={ok} match_inproc={identical}",
            )

    def run_stress_battery(self) -> None:
        self.s.section("stress_battery")
        cases: List[Tuple[str, str, Dict[str, Any]]] = []
        for m in ("naive", "optimal"):
            for segs in (1, 2, 3):
                cases.append((m, f"{m}_seg{segs}", {"method": m, "max_segments": segs}))
        cases += [
            ("naive", "naive_hops1", {"method": "naive", "max_segments": 2, "max_expansion_hops": 1}),
            ("naive", "naive_hops2", {"method": "naive", "max_segments": 2, "max_expansion_hops": 2}),
            ("optimal", "opt_hops1", {"method": "optimal", "max_segments": 2, "max_expansion_hops": 1}),
            ("naive", "naive_bins10", {"method": "naive", "max_segments": 2, "naive_bins": 10}),
            ("naive", "naive_grid", {"method": "naive", "max_segments": 2, "param_grid": True}),
            ("optimal", "opt_grid", {"method": "optimal", "max_segments": 2, "param_grid": True}),
            ("naive", "naive_small", {"method": "naive", "max_segments": 3, "n_rows": 8000}),
            ("optimal", "opt_small", {"method": "optimal", "max_segments": 3, "n_rows": 8000}),
            ("naive", "naive_sort_events", {"method": "naive", "max_segments": 2, "sort_priority": "events_rate_lift"}),
            ("naive", "naive_top8", {"method": "naive", "max_segments": 2, "top_n_vars": 8}),
        ]
        for _, tag, kw in cases:
            t0 = time.time()
            res = subprocess_signature(kw, hash_seed=None, data_path=self.data_path)
            dt = time.time() - t0
            ok = res.get("_ok", False) if isinstance(res, dict) else False
            nseg = res.get("n_segments", "?") if isinstance(res, dict) else "?"
            err = (res.get("error") or "")[:100] if isinstance(res, dict) else ""
            self.s.check(f"  stress: {tag}", bool(ok), f"segs={nseg} time={dt:.1f}s {err}")

    def run_edge_battery(self) -> None:
        self.s.section("edge_battery")
        df = self.data

        # tiny data -> graceful stop
        ok, (b2, segs2) = self.s.check_no_raise(
            "  tiny data: extract_segments runs",
            lambda: build_and_extract(
                df.iloc[:500].copy(), binning_method="naive",
                max_segments=3, min_sample_size=1000,
            ),
            lambda r: f"segs={len(r[1])}",
        )
        if ok:
            self.s.check(
                "  tiny data stops gracefully with stop_reason",
                len(segs2) == 0 and b2.stop_reason is not None,
                f"segs={len(segs2)} stop={b2.stop_reason!r}",
            )

        # NaN/null values in features and target
        df_nan = df.copy()
        idx = df_nan.index
        df_nan.loc[np.random.RandomState(1).choice(idx, 2000), "balance"] = np.nan
        df_nan.loc[np.random.RandomState(2).choice(idx, 500), "job"] = None
        df_nan[DEFAULT_TARGET] = df_nan[DEFAULT_TARGET].fillna(0).astype(int)
        for method in ("naive", "optimal"):
            ok, (_, segs3) = self.s.check_no_raise(
                f"  NaN/null handling ({method}): extract runs",
                lambda m=method: build_and_extract(df_nan, binning_method=m, max_segments=3),
                lambda r: f"segs={len(r[1])}",
            )
            if ok:
                self.s.check(
                    f"  NaN/null handling ({method}) produces segments",
                    len(segs3) > 0,
                    f"segs={len(segs3)}",
                )

        # impossible thresholds -> graceful stop
        ok, (_, segs4) = self.s.check_no_raise(
            "  impossible min_lift: extract runs",
            lambda: build_and_extract(df, binning_method="naive", max_segments=3, min_lift=10.0),
            lambda r: f"segs={len(r[1])}",
        )
        if ok:
            self.s.check(
                "  impossible min_lift stops gracefully",
                len(segs4) == 0,
                f"segs={len(segs4)}",
            )

        # ignore_features
        ok, (_, segs5) = self.s.check_no_raise(
            "  ignore_features: extract runs",
            lambda: build_and_extract(
                df, binning_method="naive", max_segments=2,
                ignore_features=["duration", "pdays"],
            ),
            lambda r: f"segs={len(r[1])}",
        )
        if ok:
            used = set()
            for s in segs5:
                for part in s["rule_string"].split(" & "):
                    used.add(part.split("=")[0])
            self.s.check(
                "  ignore_features respected",
                not (used & {"duration", "pdays"}),
                f"used={sorted(used)}",
            )

        # feature groups / diversity
        ok, (_, segs6) = self.s.check_no_raise(
            "  diversity mode runs",
            lambda: build_and_extract(
                df,
                binning_method="naive",
                max_segments=2,
                enable_diversity=True,
                feature_groups={
                    "demographic": ["age", "job", "marital", "education", "default", "housing", "loan"],
                    "contact": ["contact", "month", "day", "campaign", "pdays", "previous", "poutcome", "duration", "balance"],
                },
            ),
            lambda r: f"segs={len(r[1])}",
        )

        # boolean target
        df_bool = df.copy()
        df_bool[DEFAULT_TARGET] = df_bool[DEFAULT_TARGET].astype(bool)
        ok, (_, segs7) = self.s.check_no_raise(
            "  boolean target: extract runs",
            lambda: build_and_extract(df_bool, binning_method="naive", max_segments=2),
            lambda r: f"segs={len(r[1])}",
        )
        if ok:
            self.s.check("  boolean target works", len(segs7) > 0, f"segs={len(segs7)}")

        # repeated extract on same builder -> same result
        ok, (b8, segs8) = self.s.check_no_raise(
            "  repeat extract on same builder runs",
            lambda: build_and_extract(df, binning_method="naive", max_segments=2),
        )
        if ok:
            b8.extract_segments(df)
            r1 = signature(b8)
            b8.extract_segments(df)
            r2 = signature(b8)
            self.s.check(
                "  repeat extract on same builder is stateless & identical",
                _canonical_json(r1) == _canonical_json(r2),
            )

    def run_metric_validation(self) -> None:
        self.s.section("metric_validation")
        configs = [
            {"max_segments": 3},
            {"max_segments": 3, "sort_priority": "events_rate_lift"},
            {"max_segments": 2, "max_expansion_hops": 1, "naive_bins": 5},
            {"max_segments": 3, "top_n_vars": 8},
            {"max_segments": 3, "param_grid": {"min_sample_size": [1000, 2000], "min_lift": [1.5, 2.0]}},
        ]
        total_problems = 0
        for cfg in configs:
            for method in ("optimal", "naive"):
                ok, (builder, segments) = self.s.check_no_raise(
                    f"  extract runs: {method} {cfg}",
                    lambda m=method, c=cfg: build_and_extract(self.data, binning_method=m, **c),
                    lambda r: f"segs={len(r[1])}",
                )
                if not ok:
                    continue
                problems = hierarchical_ground_truth(self.data, builder)
                total_problems += len(problems)
                self.s.check(
                    f"  metrics valid: {method} {cfg}",
                    len(problems) == 0,
                    f"segs={len(segments)} mismatches={len(problems)}"
                    + (f" first={problems[0]}" if problems else ""),
                )
        self.s.check("  total metric mismatches == 0", total_problems == 0, f"mismatches={total_problems}")

    def run_alignment_check(self) -> None:
        self.s.section("alignment_check")
        try:
            self._alignment_inner()
        except Exception as e:  # noqa: BLE001
            self.s.check("  alignment check completed", False, f"{type(e).__name__}: {e}")

    def _alignment_inner(self) -> None:
        builder = StrategicSegmentBuilder(
            target=DEFAULT_TARGET, binning_method="optimal", max_segments=4
        )
        con = duckdb.connect()
        con.register("input_data_view", self.data)
        con.execute(
            f'''CREATE OR REPLACE TABLE current_df AS
               SELECT ROW_NUMBER() OVER () AS "__rs_row_id",
                      * REPLACE (CAST("{DEFAULT_TARGET}" AS DOUBLE) AS "{DEFAULT_TARGET}")
               FROM input_data_view'''
        )
        cols_info = con.execute("DESCRIBE current_df").fetchall()
        columns_types = {row[0]: row[1] for row in cols_info}
        eligible = [c for c in columns_types if c not in (DEFAULT_TARGET, "__rs_row_id")]
        ranking, precomputed_bins = builder.compute_iv_ranking_and_bin(con, eligible, columns_types)

        raw_target = con.execute(
            f'SELECT "{DEFAULT_TARGET}" FROM current_df'
        ).fetchnumpy()[DEFAULT_TARGET]
        clean_target = (
            raw_target.filled(0) if isinstance(raw_target, np.ma.MaskedArray) else raw_target
        )
        binned_data = {DEFAULT_TARGET: clean_target}
        valid = []
        for v in ranking:
            col = v["variable"]
            if col in precomputed_bins and len(np.unique(precomputed_bins[col])) > 1:
                binned_data[col] = precomputed_bins[col]
                valid.append(col)
        pd_binned = pd.DataFrame(binned_data)

        con = duckdb.connect()
        con.register("input_data_view", self.data)
        con.execute(
            f'''CREATE OR REPLACE TABLE current_df AS
               SELECT ROW_NUMBER() OVER () AS "__rs_row_id",
                      * REPLACE (CAST("{DEFAULT_TARGET}" AS DOUBLE) AS "{DEFAULT_TARGET}")
               FROM input_data_view'''
        )
        con.close()

        mismatch_1way = 0
        for col in valid[:6]:
            binned_stats = pd_binned.groupby(col)[DEFAULT_TARGET].agg(cnt="count", evt="sum")
            con = duckdb.connect()
            con.register("input_data_view", self.data)
            con.execute(
                f'''CREATE OR REPLACE TABLE current_df AS
                   SELECT ROW_NUMBER() OVER () AS "__rs_row_id",
                          * REPLACE (CAST("{DEFAULT_TARGET}" AS DOUBLE) AS "{DEFAULT_TARGET}")
                   FROM input_data_view'''
            )
            for label in binned_stats.index:
                sql = builder.parse_rule_to_sql(f"{col}={label}")
                res = con.execute(
                    f'SELECT COUNT(*) , SUM("{DEFAULT_TARGET}") FROM current_df WHERE ({sql})'
                ).fetchone()
                b_cnt, b_evt = binned_stats.loc[label]
                a_cnt, a_evt = res[0], res[1] or 0
                if abs(b_cnt - a_cnt) > 0 or abs(b_evt - a_evt) > 1e-9:
                    mismatch_1way += 1
            con.close()
        self.s.check("  1-way binned vs actual alignment", mismatch_1way == 0, f"mismatches={mismatch_1way}")

        mismatch_2way = 0
        con = duckdb.connect()
        con.register("input_data_view", self.data)
        con.execute(
            f'''CREATE OR REPLACE TABLE current_df AS
               SELECT ROW_NUMBER() OVER () AS "__rs_row_id",
                      * REPLACE (CAST("{DEFAULT_TARGET}" AS DOUBLE) AS "{DEFAULT_TARGET}")
                   FROM input_data_view'''
        )
        for c1, c2 in list(itertools.combinations(valid, 2))[:20]:
            grouped = pd_binned.groupby([c1, c2])[DEFAULT_TARGET].agg(cnt="count", evt="sum")
            for (l1, l2), (b_cnt, b_evt) in grouped.iterrows():
                sql = builder.parse_rule_to_sql(f"{c1}={l1} & {c2}={l2}")
                res = con.execute(
                    f'SELECT COUNT(*) , SUM("{DEFAULT_TARGET}") FROM current_df WHERE ({sql})'
                ).fetchone()
                a_cnt, a_evt = res[0], res[1] or 0
                if abs(b_cnt - a_cnt) > 0 or abs(b_evt - a_evt) > 1e-9:
                    mismatch_2way += 1
        con.close()
        self.s.check("  2-way binned vs actual alignment", mismatch_2way == 0, f"mismatches={mismatch_2way}")

    def run_scorer_deep(self) -> None:
        self.s.section("scorer_deep")
        builder, segments = build_and_extract(self.data, binning_method="naive", max_segments=3)
        if len(segments) == 0:
            self.s.check("  builder produced segments for deep scorer test", False, "no segments")
            return
        ok, model = self.s.check_no_raise(
            "  scorer full pipeline (3 segments)",
            lambda: score_model(self.data, builder, self.scratch),
            lambda r: f"segments={len(segments)} keys={sorted(r.keys())}",
        )
        if ok:
            meta = model["model_metadata"]
            self.s.check(
                "  metadata population matches data",
                meta["total_training_population"] == len(self.data),
                f"meta={meta['total_training_population']} actual={len(self.data)}",
            )
            self.s.check(
                "  active scored population > 0",
                meta["active_scored_population"] > 0,
                f"active={meta['active_scored_population']}",
            )
            self.s.check(
                "  baseline event rate sane",
                0.0 < meta["baseline_event_rate"] < 1.0,
                f"rate={meta['baseline_event_rate']}",
            )
            self.s.check(
                "  all weights are integers",
                all(isinstance(w["weight"], int) for w in model["segment_weights"].values()),
            )
            thr = model["decile_min_thresholds"]
            self.s.check(
                "  decile thresholds are monotonic non-increasing",
                all(thr[str(i)] >= thr[str(i + 1)] for i in range(1, 10)),
                str(thr),
            )

    def run_data_loader_formats(self) -> None:
        self.s.section("data_loader_formats")
        csv_path = os.path.join(self.scratch, "input.csv")
        self.data.to_csv(csv_path, index=False)

        ok, t1 = self.s.check_no_raise(
            "  CSV -> pyarrow via loader",
            lambda: UniversalDataLoader(file_path=csv_path).load(),
            lambda t: f"rows={len(t)}",
        )
        if ok:
            self.s.check("  CSV row count preserved", len(t1) == len(self.data), f"{len(t1)} vs {len(self.data)}")

        parquet_path = os.path.join(self.scratch, "input.parquet")
        pa_pq.write_table(pa.Table.from_pandas(self.data), parquet_path)
        ok, t2 = self.s.check_no_raise(
            "  Parquet -> pyarrow via loader",
            lambda: UniversalDataLoader(file_path=parquet_path).load(),
            lambda t: f"rows={len(t)}",
        )
        if ok:
            self.s.check("  Parquet row count preserved", len(t2) == len(self.data), f"{len(t2)} vs {len(self.data)}")

        arrow_path = os.path.join(self.scratch, "input.arrow")
        with pa.memory_map(arrow_path, "w") as sink:
            with pa.ipc.new_file(sink, pa.Table.from_pandas(self.data).schema) as writer:
                writer.write_table(pa.Table.from_pandas(self.data))
        ok, t3 = self.s.check_no_raise(
            "  Arrow/Feather -> pyarrow via loader",
            lambda: UniversalDataLoader(file_path=arrow_path).load(),
            lambda t: f"rows={len(t)}",
        )
        if ok:
            self.s.check("  Arrow row count preserved", len(t3) == len(self.data), f"{len(t3)} vs {len(self.data)}")

    def run_health_report_deep(self) -> None:
        self.s.section("health_report_deep")
        builder, _ = build_and_extract(self.data, binning_method="naive", max_segments=2)
        features = [c for c in self.data.columns if c != DEFAULT_TARGET]
        ok, report = self.s.check_no_raise(
            "  health report for all features",
            lambda: builder.generate_feature_health_report(self.data, features),
            lambda r: f"rows={len(r)} features={r['feature'].nunique() if len(r) else 0}",
        )
        if not ok:
            return
        per_feature = report.groupby("feature")["total_count"].sum()
        bad = per_feature[per_feature != len(self.data)]
        self.s.check(
            "  per-feature bin counts sum to population",
            len(bad) == 0,
            f"bad_features={list(bad.index)}" if len(bad) else "all good",
        )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RapidSegment stress test suite")
    p.add_argument("--mode", choices=["fast", "deep"], default="fast")
    p.add_argument("--data", default=DEFAULT_DATA, help="path to input CSV")
    p.add_argument("--target", default=DEFAULT_TARGET, help="target column name")
    p.add_argument("--rows", type=int, default=0, help="cap number of rows (0 = all)")
    p.add_argument("--out", default=None, help="write JSON report to this path")
    # hidden subprocess mode
    p.add_argument("--sub", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--method", default="naive", help=argparse.SUPPRESS)
    p.add_argument("--max_segments", type=int, default=3, help=argparse.SUPPRESS)
    p.add_argument("--naive_bins", type=int, default=5, help=argparse.SUPPRESS)
    p.add_argument("--max_expansion_hops", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--sort_priority", default="rate_lift_count", help=argparse.SUPPRESS)
    p.add_argument("--top_n_vars", type=int, default=15, help=argparse.SUPPRESS)
    p.add_argument("--n_rows", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--param_grid", type=int, default=0, help=argparse.SUPPRESS)
    return p


def run_subprocess_mode(args) -> None:
    """Single-extraction mode for subprocess isolation; prints JSON signature."""
    data = prepare_data(os.path.abspath(args.data), args.target, args.n_rows or None)
    kw = dict(
        target=args.target,
        binning_method=args.method,
        max_segments=args.max_segments,
        naive_bins=args.naive_bins,
        max_expansion_hops=args.max_expansion_hops,
        sort_priority=args.sort_priority,
        top_n_vars=args.top_n_vars,
    )
    if args.param_grid:
        kw["param_grid"] = {"min_sample_size": [1000, 2000], "min_lift": [1.5, 2.0]}
    try:
        builder = StrategicSegmentBuilder(**kw)
        segments = builder.extract_segments(data)
        out = signature(builder)
        out["_ok"] = True
    except Exception as e:  # noqa: BLE001
        out = {"_ok": False, "error": f"{type(e).__name__}: {e}"}
    print(_canonical_json(out))
    sys.exit(0 if out.get("_ok") else 1)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.sub:
        run_subprocess_mode(args)
        return 0

    data_path = os.path.abspath(args.data)
    if not os.path.exists(data_path):
        print(f"ERROR: data file not found: {data_path}")
        return 1

    # Run inside a scratch dir so all library artifacts (experiments/, *.db)
    # stay out of the caller's working directory.
    scratch = tempfile.mkdtemp(prefix="rs_stress_")
    previous_cwd = os.getcwd()
    os.chdir(scratch)

    start = time.time()
    try:
        data = prepare_data(data_path, args.target, args.rows or None)
        suite = Suite(args.mode)
        tests = DeepTests(suite, data, scratch, data_path) if args.mode == "deep" else FastTests(suite, data, scratch, data_path)

        tests.run_data_loader()
        tests.run_builder_smoke()
        tests.run_sql_filters()
        tests.run_coverage()
        tests.run_metric_sanity()
        tests.run_scorer()
        tests.run_health_report()
        tests.run_diagnostics()
        tests.run_determinism_smoke()

        if args.mode == "deep":
            tests.run_determinism_battery()
            tests.run_stress_battery()
            tests.run_edge_battery()
            tests.run_metric_validation()
            tests.run_alignment_check()
            tests.run_scorer_deep()
            tests.run_data_loader_formats()
            tests.run_health_report_deep()

        summary = suite.summary()
        summary["elapsed_seconds"] = round(time.time() - start, 1)
        if args.out:
            os.chdir(previous_cwd)
            out_path = os.path.abspath(args.out)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"\nReport written to: {out_path}")
        return 1 if summary["failed"] else 0
    finally:
        os.chdir(previous_cwd)


if __name__ == "__main__":
    sys.exit(main())
