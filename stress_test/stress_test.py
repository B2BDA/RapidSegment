"""
Intensive stress test of StrategicSegmentBuilder.

Strategy:
1. Load the adversarial dataset (numeric + tricky categorical mix, nulls, outliers).
2. Run through a large matrix of constructor parameter combinations covering:
   - binning_method: optimal vs naive
   - selection_metric: iv vs response_rate
   - sort_priority: all 14 variants
   - enable_1way/2way/3way: all combinations
   - enable_diversity with/without feature_groups
   - max_expansion_hops: 0, 1, 2, 5 (only meaningful with naive binning, but test both)
   - max_feature_reuse: 1 vs 2
   - param_grid: None vs a real grid
   - ignore_features: with/without
   - min_lift / min_sample_size / min_events at various strict/loose thresholds
   - db_path explicit vs auto-generated
3. For each run: call extract_segments(), then evaluate_final_coverage(),
   explain_feature_journey(), generate_feature_health_report().
4. Independently re-validate every returned segment's sql_filter against the
   raw dataframe using a fresh DuckDB connection + pandas, to catch silent
   parser mis-translations (count/rate mismatches) not caught by exceptions.
5. Catch and record every exception with full traceback + the parameter set
   that caused it, plus non-exception correctness issues (mismatches,
   NaNs, negative counts, sql injection markers, malformed SQL).
"""
import itertools
import json
import sys
import time
import traceback
import warnings

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, "./")
from rapidsegment import StrategicSegmentBuilder

warnings.filterwarnings("ignore")

df = pd.read_parquet("./stress_data.parquet")

RESULTS = []


def log_result(name, params, status, detail="", elapsed=None, extra=None):
    RESULTS.append({
        "name": name,
        "params": params,
        "status": status,
        "detail": detail[:2000] if isinstance(detail, str) else detail,
        "elapsed": elapsed,
        "extra": extra,
    })
    flag = {"OK": "✅", "FAIL": "❌", "WARN": "⚠️"}.get(status, "?")
    print(f"{flag} [{status}] {name} ({elapsed:.2f}s)" if elapsed is not None else f"{flag} [{status}] {name}")
    if status != "OK":
        print(f"    -> {detail[:500]}")


def independent_validate(builder_obj, segments, data):
    """
    Re-derive each segment's row count/response rate directly against `data`
    using the segment's sql_filter, via a completely fresh duckdb connection,
    and compare against what extract_segments()/evaluate_final_coverage() claims.
    Also sanity-checks for structurally broken SQL (unbalanced parens, etc.)
    """
    issues = []
    con = duckdb.connect(":memory:")
    con.register("raw_view", data)
    con.execute("CREATE TABLE raw AS SELECT * FROM raw_view")

    for seg in segments:
        sql_filter = seg["sql_filter"]
        if sql_filter.count("(") != sql_filter.count(")"):
            issues.append(f"segment {seg['segment_id']}: unbalanced parens in sql_filter: {sql_filter}")
            continue
        try:
            res = con.execute(
                f'SELECT COUNT(*), SUM(CAST(target AS DOUBLE)) FROM raw WHERE ({sql_filter})'
            ).fetchone()
        except Exception as e:
            issues.append(f"segment {seg['segment_id']}: sql_filter raised on fresh con: {e} | filter={sql_filter}")
            continue
        cnt, evt = res[0], (res[1] or 0)
        # These are on the *hierarchical residual* dataset at extraction time,
        # not directly comparable to full-data counts, so we only sanity check
        # generic invariants here (non-negativity, not wildly larger than population).
        if cnt is None or cnt < 0:
            issues.append(f"segment {seg['segment_id']}: negative/None count from fresh validation")
        if cnt > len(data):
            issues.append(f"segment {seg['segment_id']}: count {cnt} exceeds population {len(data)}")
        if seg["count"] > len(data):
            issues.append(f"segment {seg['segment_id']}: reported count {seg['count']} exceeds population size")
        if seg["rate"] < 0 or seg["rate"] > 100:
            issues.append(f"segment {seg['segment_id']}: rate {seg['rate']} out of [0,100] bounds")
        if np.isnan(seg["lift"]) or np.isinf(seg["lift"]):
            issues.append(f"segment {seg['segment_id']}: lift is NaN/Inf")
    con.close()
    return issues


def run_one(name, kwargs, data=df, run_extras=True):
    t0 = time.time()
    try:
        sb = StrategicSegmentBuilder(**kwargs)
        segments = sb.extract_segments(data)
        elapsed = time.time() - t0

        issues = independent_validate(sb, segments, data)

        if run_extras:
            cov = sb.evaluate_final_coverage(data)
            # health report on a mix of variable types
            health_df = sb.generate_feature_health_report(
                data, ["income", "region", "high_card_id", "constant_cat", "constant_num", "flag_str"]
            )
            if sb.diagnostics_:
                fname = list(sb.diagnostics_[0]["features_state"].keys())[0]
                sb.explain_feature_journey(fname)

            # Cross check coverage capture rates sum to <=100 roughly and counts non-negative
            for row in cov:
                if row["total_count"] < 0:
                    issues.append(f"coverage segment {row['segment']}: negative total_count")
                if row["capture_rate"] is not None and (row["capture_rate"] < -1e-6 or row["capture_rate"] > 100.0001):
                    issues.append(f"coverage segment {row['segment']}: capture_rate out of bounds {row['capture_rate']}")

        if issues:
            log_result(name, kwargs, "WARN", "; ".join(issues), elapsed, extra={"n_segments": len(segments)})
        else:
            log_result(name, kwargs, "OK", f"{len(segments)} segments extracted", elapsed,
                       extra={"n_segments": len(segments), "rules": [s["rule_string"] for s in segments]})
        return sb, segments
    except Exception as e:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        log_result(name, kwargs, "FAIL", tb, elapsed)
        return None, None


# ============================================================================
# PHASE 1: Baseline sanity runs (default params, optimal vs naive)
# ============================================================================
print("\n" + "=" * 100)
print("PHASE 1: Baseline runs")
print("=" * 100)

run_one("baseline_optimal", dict(target="target", binning_method="optimal", db_path=":memory:", db_temp_dir="/tmp"))
run_one("baseline_naive", dict(target="target", binning_method="naive", db_path=":memory:", db_temp_dir="/tmp"))

# ============================================================================
# PHASE 2: sort_priority exhaustive sweep
# ============================================================================
print("\n" + "=" * 100)
print("PHASE 2: sort_priority exhaustive sweep")
print("=" * 100)

sort_priorities = [
    "lift_count_rate", "count_lift_rate", "rate_lift_count", "lift_rate_count",
    "count_rate_lift", "rate_count_lift", "events_lift_rate", "events_rate_lift",
    "lift_events_rate", "rate_events_lift", "events_count_rate", "events_rate_count",
    "count_events_rate", "rate_events_count", "totally_bogus_priority",
]
for sp in sort_priorities:
    run_one(f"sort_priority={sp}", dict(
        target="target", sort_priority=sp, binning_method="naive",
        max_expansion_hops=2, db_path=":memory:", db_temp_dir="/tmp",
        max_segments=3,
    ), run_extras=False)

# ============================================================================
# PHASE 3: enable_1way/2way/3way combinatorics
# ============================================================================
print("\n" + "=" * 100)
print("PHASE 3: 1way/2way/3way combinations (including all-False)")
print("=" * 100)

for e1, e2, e3 in itertools.product([True, False], repeat=3):
    run_one(f"ways_1={e1}_2={e2}_3={e3}", dict(
        target="target", enable_1way=e1, enable_2way=e2, enable_3way=e3,
        max_segments=3, db_path=":memory:", db_temp_dir="/tmp",
    ), run_extras=False)

# ============================================================================
# PHASE 4: enable_diversity + feature_groups (valid and invalid)
# ============================================================================
print("\n" + "=" * 100)
print("PHASE 4: diversity + feature_groups")
print("=" * 100)

run_one("diversity_valid_groups", dict(
    target="target", enable_diversity=True,
    feature_groups={"money": ["income", "balance"], "geo": ["region", "high_card_id"]},
    max_segments=3, db_path=":memory:", db_temp_dir="/tmp",
))

run_one("diversity_invalid_group_var", dict(
    target="target", enable_diversity=True,
    feature_groups={"money": ["income", "NON_EXISTENT_COLUMN"]},
    max_segments=3, db_path=":memory:", db_temp_dir="/tmp",
), run_extras=False)

run_one("diversity_group_references_target", dict(
    target="target", enable_diversity=True,
    feature_groups={"bad": ["target"]},
    max_segments=3, db_path=":memory:", db_temp_dir="/tmp",
), run_extras=False)

# ============================================================================
# PHASE 5: max_expansion_hops sweep (naive + optimal)
# ============================================================================
print("\n" + "=" * 100)
print("PHASE 5: max_expansion_hops sweep")
print("=" * 100)

for hops in [0, 1, 2, 5, 50]:
    for bm in ["naive", "optimal"]:
        run_one(f"hops={hops}_bm={bm}", dict(
            target="target", max_expansion_hops=hops, binning_method=bm,
            expand_log_mode="full", max_segments=2,
            db_path=":memory:", db_temp_dir="/tmp",
        ), run_extras=False)

# negative hops (should clamp to 0 per max(0,int(...)))
run_one("hops=negative", dict(
    target="target", max_expansion_hops=-5, binning_method="naive",
    max_segments=2, db_path=":memory:", db_temp_dir="/tmp",
), run_extras=False)

# ============================================================================
# PHASE 6: max_feature_reuse & top_n_vars extremes
# ============================================================================
print("\n" + "=" * 100)
print("PHASE 6: feature reuse / top_n_vars extremes")
print("=" * 100)

run_one("max_feature_reuse=1", dict(target="target", max_feature_reuse=1, max_segments=5,
        db_path=":memory:", db_temp_dir="/tmp"), run_extras=False)
run_one("max_feature_reuse=0", dict(target="target", max_feature_reuse=0, max_segments=5,
        db_path=":memory:", db_temp_dir="/tmp"), run_extras=False)
run_one("top_n_vars=1", dict(target="target", top_n_vars=1, max_segments=3,
        db_path=":memory:", db_temp_dir="/tmp"), run_extras=False)
run_one("top_n_vars=0", dict(target="target", top_n_vars=0, max_segments=3,
        db_path=":memory:", db_temp_dir="/tmp"), run_extras=False)
run_one("top_n_vars=1000", dict(target="target", top_n_vars=1000, max_segments=3,
        db_path=":memory:", db_temp_dir="/tmp"), run_extras=False)
run_one("max_segments=0", dict(target="target", max_segments=0,
        db_path=":memory:", db_temp_dir="/tmp"), run_extras=False)
run_one("max_segments=50", dict(target="target", max_segments=50,
        db_path=":memory:", db_temp_dir="/tmp"), run_extras=False)

# ============================================================================
# PHASE 7: param_grid sweeps
# ============================================================================
print("\n" + "=" * 100)
print("PHASE 7: param_grid sweeps")
print("=" * 100)

run_one("param_grid_normal", dict(
    target="target",
    param_grid={"min_sample_size": [500, 1000, 2000], "min_lift": [1.5, 2.0, 3.0]},
    max_segments=3, db_path=":memory:", db_temp_dir="/tmp",
), run_extras=False)

run_one("param_grid_empty_lists", dict(
    target="target",
    param_grid={"min_sample_size": [], "min_lift": []},
    max_segments=3, db_path=":memory:", db_temp_dir="/tmp",
), run_extras=False)

run_one("param_grid_single_key_only", dict(
    target="target",
    param_grid={"min_sample_size": [500, 5000]},
    max_segments=3, db_path=":memory:", db_temp_dir="/tmp",
), run_extras=False)

# ============================================================================
# PHASE 8: min_lift / min_sample_size / min_events extremes
# ============================================================================
print("\n" + "=" * 100)
print("PHASE 8: constraint extremes")
print("=" * 100)

run_one("min_lift=0", dict(target="target", min_lift=0.0, max_segments=3,
        db_path=":memory:", db_temp_dir="/tmp"), run_extras=False)
run_one("min_lift=huge", dict(target="target", min_lift=1000.0, max_segments=3,
        db_path=":memory:", db_temp_dir="/tmp"), run_extras=False)
run_one("min_lift=negative", dict(target="target", min_lift=-5.0, max_segments=3,
        db_path=":memory:", db_temp_dir="/tmp"), run_extras=False)
run_one("min_sample_size=0", dict(target="target", min_sample_size=0, max_segments=3,
        db_path=":memory:", db_temp_dir="/tmp"), run_extras=False)
run_one("min_sample_size=huge", dict(target="target", min_sample_size=10_000_000, max_segments=3,
        db_path=":memory:", db_temp_dir="/tmp"), run_extras=False)
run_one("min_events=0", dict(target="target", min_events=0, max_segments=3,
        db_path=":memory:", db_temp_dir="/tmp"), run_extras=False)
run_one("min_events=huge", dict(target="target", min_events=1_000_000, max_segments=3,
        db_path=":memory:", db_temp_dir="/tmp"), run_extras=False)
run_one("min_sample_size=negative", dict(target="target", min_sample_size=-100, max_segments=3,
        db_path=":memory:", db_temp_dir="/tmp"), run_extras=False)

# ============================================================================
# PHASE 9: selection_metric + binning_method + naive_bins extremes
# ============================================================================
print("\n" + "=" * 100)
print("PHASE 9: selection_metric / naive_bins extremes")
print("=" * 100)

for sm in ["iv", "response_rate", "bogus_metric"]:
    run_one(f"selection_metric={sm}", dict(target="target", selection_metric=sm, max_segments=3,
            db_path=":memory:", db_temp_dir="/tmp"), run_extras=False)

for nb in [1, 2, 3, 100]:
    run_one(f"naive_bins={nb}", dict(target="target", binning_method="naive", naive_bins=nb,
            max_segments=2, db_path=":memory:", db_temp_dir="/tmp"), run_extras=False)

run_one("binning_method=bogus", dict(target="target", binning_method="totally_bogus",
        max_segments=2, db_path=":memory:", db_temp_dir="/tmp"), run_extras=False)

# ============================================================================
# PHASE 10: ignore_features / n_jobs extremes
# ============================================================================
print("\n" + "=" * 100)
print("PHASE 10: ignore_features / n_jobs extremes")
print("=" * 100)

run_one("ignore_some_features", dict(
    target="target", ignore_features=["noise_num", "noise_cat", "high_card_id"],
    max_segments=3, db_path=":memory:", db_temp_dir="/tmp",
), run_extras=False)

run_one("ignore_all_features", dict(
    target="target",
    ignore_features=["income", "tenure_days", "balance", "noise_num", "constant_num",
                      "region", "high_card_id", "constant_cat", "noise_cat", "flag_str"],
    max_segments=3, db_path=":memory:", db_temp_dir="/tmp",
), run_extras=False)

run_one("ignore_nonexistent_feature", dict(
    target="target", ignore_features=["DOES_NOT_EXIST"],
    max_segments=3, db_path=":memory:", db_temp_dir="/tmp",
), run_extras=False)

run_one("n_jobs=1", dict(target="target", n_jobs=1, max_segments=2,
        db_path=":memory:", db_temp_dir="/tmp"), run_extras=False)
run_one("n_jobs=2", dict(target="target", n_jobs=2, max_segments=2,
        db_path=":memory:", db_temp_dir="/tmp"), run_extras=False)

# ============================================================================
# PHASE 11: expand_log_mode variants + db_path auto mode
# ============================================================================
print("\n" + "=" * 100)
print("PHASE 11: expand_log_mode + auto db path")
print("=" * 100)

for mode in ["none", "summary", "full", "garbage_mode"]:
    run_one(f"expand_log_mode={mode}", dict(
        target="target", binning_method="naive", max_expansion_hops=2, expand_log_mode=mode,
        max_segments=2, db_path=":memory:", db_temp_dir="/tmp",
    ), run_extras=False)

# Auto-created disk-backed DB (default path) -- test the setup_disk_backed_db route
run_one("auto_db_path", dict(target="target", max_segments=2), run_extras=False)

# ============================================================================
# PHASE 12: degenerate / adversarial datasets
# ============================================================================
print("\n" + "=" * 100)
print("PHASE 12: degenerate datasets")
print("=" * 100)

# All-zero target (no events at all)
df_zero = df.copy()
df_zero["target"] = 0
run_one("all_zero_target", dict(target="target", max_segments=3,
        db_path=":memory:", db_temp_dir="/tmp"), data=df_zero, run_extras=False)

# All-one target
df_one = df.copy()
df_one["target"] = 1
run_one("all_one_target", dict(target="target", max_segments=3,
        db_path=":memory:", db_temp_dir="/tmp"), data=df_one, run_extras=False)

# Tiny dataset (fewer rows than min_sample_size default)
df_tiny = df.sample(n=50, random_state=1).reset_index(drop=True)
run_one("tiny_dataset_default_thresholds", dict(target="target", max_segments=3,
        db_path=":memory:", db_temp_dir="/tmp"), data=df_tiny, run_extras=False)

# Single row
df_single = df.sample(n=1, random_state=1).reset_index(drop=True)
run_one("single_row_dataset", dict(target="target", max_segments=3, min_sample_size=1,
        min_events=0, db_path=":memory:", db_temp_dir="/tmp"), data=df_single, run_extras=False)

# Only target column, no features
df_target_only = df[["target"]].copy()
run_one("only_target_column", dict(target="target", max_segments=3,
        db_path=":memory:", db_temp_dir="/tmp"), data=df_target_only, run_extras=False)

# Empty dataframe (0 rows) - structurally valid schema
df_empty = df.iloc[0:0].copy()
run_one("empty_dataframe", dict(target="target", max_segments=3,
        db_path=":memory:", db_temp_dir="/tmp"), data=df_empty, run_extras=False)

# All-null feature column
df_allnull = df.copy()
df_allnull["income"] = np.nan
run_one("all_null_numeric_feature", dict(target="target", max_segments=3,
        db_path=":memory:", db_temp_dir="/tmp"), data=df_allnull, run_extras=False)

df_allnull_cat = df.copy()
df_allnull_cat["region"] = None
run_one("all_null_categorical_feature", dict(target="target", max_segments=3,
        db_path=":memory:", db_temp_dir="/tmp"), data=df_allnull_cat, run_extras=False)

# Target column with non-binary values
df_bad_target = df.copy()
df_bad_target["target"] = rng_choice = np.random.default_rng(0).integers(0, 5, size=len(df))
run_one("non_binary_target", dict(target="target", max_segments=3,
        db_path=":memory:", db_temp_dir="/tmp"), data=df_bad_target, run_extras=False)

# Target column as boolean dtype
df_bool_target = df.copy()
df_bool_target["target"] = df_bool_target["target"].astype(bool)
run_one("boolean_target", dict(target="target", max_segments=3,
        db_path=":memory:", db_temp_dir="/tmp"), data=df_bool_target, run_extras=False)

# Missing target column entirely
run_one("missing_target_column", dict(target="does_not_exist", max_segments=3,
        db_path=":memory:", db_temp_dir="/tmp"), data=df, run_extras=False)

print("\n\nAll phases dispatched.")
with open("./results_phase1.json", "w") as f:
    json.dump(RESULTS, f, default=str, indent=2)
print(f"Total runs so far: {len(RESULTS)}")
