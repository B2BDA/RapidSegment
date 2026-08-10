"""
Strategic Segmentation Engine
==============================
Combinatorial heuristic segmentation using SQL-native quantile binning,
Apriori pruning, and vectorized DuckDB scorecard deciling.

Refactored:
  - optbinning dependency removed; SQL-native naive binning is the single,
    unified binning engine (numeric and categorical).
  - Combination search generalized from fixed 1/2/3-way to arbitrary N-way
    (up to 5-way) via a single Apriori loop.
  - Internal @dataclass rules/predicates replace string-only round-tripping.
  - Categorical adjacent-bin merging uses metric-rank adjacency instead of
    meaningless alphabetical adjacency.
  - Instance-state mutation during iteration removed; explicit local control.
  - sort_priority boilerplate collapsed into a single dimension table.
  - Repeated DuckDB plumbing extracted into _load_into_duckdb; logging pulled
    into _log_* helpers; adjacent-bin expansion pushed into SQL vectorization.

Author: Bishwarup Biswas + Gemini + DeepSeek + ChatGPT
Python Version: 3.9+
"""

import logging
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import duckdb
import numpy as np
from joblib import Parallel, delayed
import pandas as pd
import uuid
import psutil

# -----------------------------------------------------------------------------
# Module-level configuration
# -----------------------------------------------------------------------------


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [%(filename)s:%(lineno)d] | %(message)s",
)
logger = logging.getLogger("StrategicEngine")

# Pre-compiled regex for fast parsing inside loops
_BRACKET_REGEX = re.compile(r"\[(.*?)\]", flags=re.DOTALL)
_RANGE_TOKEN_RE = re.compile(r"^[\[\(].*[\]\)]$")
_NUMERIC_RANGE_RE = re.compile(r"^[\[\(]-?\s*(inf|-inf|\d+\.?\d*(?:[eE][+-]?\d+)?)\s*,")

_MAX_COMBO_ARITY_CAP = 5

# Single source of truth for sort priorities. Values are ordered dim lists used
# both by _get_sort_key and by the "why did the candidate lose" explainer.
_SORT_DIMENSIONS: Dict[str, Tuple[str, ...]] = {
    "lift_count_rate": ("lift", "count", "rate"),
    "count_lift_rate": ("count", "lift", "rate"),
    "rate_lift_count": ("rate", "lift", "count"),
    "lift_rate_count": ("lift", "rate", "count"),
    "count_rate_lift": ("count", "rate", "lift"),
    "rate_count_lift": ("rate", "count", "lift"),
    "events_lift_rate": ("events", "lift", "rate"),
    "events_rate_lift": ("events", "rate", "lift"),
    "lift_events_rate": ("lift", "events", "rate"),
    "rate_events_lift": ("rate", "events", "lift"),
    "events_count_rate": ("events", "count", "rate"),
    "events_rate_count": ("events", "rate", "count"),
    "count_events_rate": ("count", "events", "rate"),
    "rate_events_count": ("rate", "events", "count"),
}
_DEFAULT_SORT_DIMS: Tuple[str, ...] = ("lift", "rate", "count")

# Dim labels used by the explainer when a candidate ranked lower than champion.
_DIM_LABELS = {
    "lift": "lower lift",
    "count": "smaller count",
    "rate": "lower rate",
    "events": "fewer events",
}

_CATEGORICAL_MISSING_TOKENS = ("", "None", "nan", "NaN", "<NA>", "null", "NULL")


# -----------------------------------------------------------------------------
# Internal lightweight data structures (public interface unchanged)
# -----------------------------------------------------------------------------


@dataclass
class Rule:
    """
    Internal representation of a candidate rule. A structured predicate list is
    carried alongside the human-readable rule string so SQL generation never has
    to re-parse the string. Converts to the legacy dict shape at public-method
    boundaries so extract_segments()'s return value keeps the exact same shape.
    """
    rule: str                                   # human-readable rule string
    count: int
    rate: float
    lift: float
    events: float
    combo_vars: Tuple[str, ...]
    base_events: Optional[float] = None
    predicates: List[Tuple[str, str, Any]] = field(default_factory=list)
    # extra metadata (grid flags) stored here only when needed
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "rule": self.rule,
            "count": self.count,
            "rate": self.rate,
            "lift": self.lift,
            "events": self.events,
            "combo_vars": self.combo_vars,
        }
        if self.base_events is not None:
            d["base_events"] = self.base_events
        d.update(self.meta)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Rule":
        return cls(
            rule=d["rule"],
            count=d["count"],
            rate=d["rate"],
            lift=d["lift"],
            events=d["events"],
            combo_vars=tuple(d["combo_vars"]),
            base_events=d.get("base_events"),
        )


@dataclass
class FeatureScore:
    """Replaces the loose iv_ranking row dict."""
    variable: str
    iv: float
    max_rr: float

    def to_dict(self) -> Dict[str, Union[str, float]]:
        return {"variable": self.variable, "iv": self.iv, "max_rr": self.max_rr}

    @classmethod
    def from_dict(cls, d: Dict[str, Union[str, float]]) -> "FeatureScore":
        return cls(variable=d["variable"], iv=d["iv"], max_rr=d["max_rr"])


@dataclass
class CandidateSegment:
    """Replaces selected_candidate / grid_candidates loose dicts."""
    rule: Rule
    grid_min_sample_size: Optional[int] = None
    grid_min_lift: Optional[float] = None
    sql_filter: Optional[str] = None
    actual_count: Optional[int] = None
    actual_events: Optional[float] = None


# -----------------------------------------------------------------------------
# DuckDB plumbing
# -----------------------------------------------------------------------------


def setup_disk_backed_db(base_dir: str = "experiments") -> tuple[str, str]:
    """
    Creates an experiment directory and generates a unique DuckDB file path.
    Returns the database path and the temp directory path.
    """
    # 1. Create the main experiments directory
    os.makedirs(base_dir, exist_ok=True)

    # 2. Generate unique identifiers (Date + UUID)
    date_str = datetime.now().strftime("%Y%m%d")
    unique_id = uuid.uuid4().hex[:8]  # Short UUID is usually sufficient and cleaner

    # 3. Define the main database file path
    db_filename = f"segmentation_{date_str}_{unique_id}.duckdb"
    db_path = os.path.join(base_dir, db_filename)

    # 4. Create a dedicated temp directory for DuckDB to spill to
    temp_dir = os.path.join(base_dir, f"tmp_{date_str}_{unique_id}")
    os.makedirs(temp_dir, exist_ok=True)

    return db_path, temp_dir


class StrategicSegmentBuilder:
    """
    Extracts hierarchical, predictive segments from tabular data.

    The extraction is sequential:
        - At each step, the best rule (by lift and volume) is found on the current residual dataset.
        - The rule is applied to remove those rows before the next iteration.
        - The final segmentation is hierarchical: the first rule has highest priority,
          the second rule applies to the remainder, and so on.

    The `extract_segments()` method returns segments whose counts reflect the
    final hierarchical assignment on the original dataset, exactly matching the
    output of `evaluate_final_coverage()`.
    """

    def __init__(
        self,
        target: str,
        n_jobs: int = -1,
        min_sample_size: int = 1000,
        min_lift: float = 2.0,
        min_events: int = 5,
        top_n_vars: int = 20,
        max_segments: int = 10,
        max_feature_reuse: int = 1,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        enable_diversity: bool = False,
        enable_1way: bool = True,
        enable_2way: bool = True,
        enable_3way: bool = True,
        feature_groups: Optional[Dict[str, List[str]]] = None,
        ignore_features: Optional[List[str]] = None,
        sort_priority: str = "lift_rate_count",  # or "count_lift_rate", "lift_rate_count", etc.
        binning_method: str = "optimal",
        naive_bins: int = 5,
        max_expansion_hops: int = 1,
        selection_metric: str = "iv",
        expand_log_mode: str = "summary",  # "none" | "summary" | "full",
        db_path: Optional[str] = None,
        db_temp_dir: Optional[str] = None,
        max_combo_arity: int = 3,
        pool_rare_categories: bool = True,
    ) -> None:
        """
        Args:
            target: Name of the binary target column (1 = Event, 0 = Non-Event).
            n_jobs: Number of parallel jobs for IV computation. -1 uses all but one core.
            min_sample_size: Absolute minimum row count for a valid rule. Used as a fallback when param_grid is None.
            min_lift: Absolute minimum lift threshold (hard constraint).
            min_events: Minimum number of positive events for a valid rule.
            top_n_vars: Number of highest-scored features passed into the Apriori engine.
            max_segments: Maximum number of segments to extract.
            max_feature_reuse: Max times a feature can appear across segments.
            param_grid: Optional grid of {min_sample_size, min_lift} to sweep.
            enable_diversity: If True, blocks rules combining variables from same group.
            enable_1way: Allow 1-dimensional rules.
            enable_2way: Allow 2-dimensional intersection rules.
            enable_3way: Allow 3-dimensional intersection rules.
            feature_groups: Mapping of business categories to columns (e.g. {'risk': ['scr', 'bal']}).
            ignore_features: Explicit list of columns to drop prior to IV calculation.
            sort_priority: Ranking criteria for selecting champion segments.
            binning_method: Which binning engine to use. 'naive' uses the unified SQL-native
                engine (the only supported path); any other value (legacy 'optimal') logs a
                deprecation warning and falls back to 'naive'.
            naive_bins: Number of quantile bins used by the SQL-native binning engine.
            selection_metric: Metric used to rank features for top_n_vars selection
                ("iv", "response_rate", or a callable(bin_stats) -> float).
            max_expansion_hops: Adjacent-bin merging hop distance limit.
            expand_log_mode: Controls verbosity of adjacent-bin expansion logging ("none", "summary", "full").
            max_combo_arity: Maximum combination arity for the combination search (1..5).
                The enable_1way/2way/3way booleans act as convenience caps on top of this value
                for backward compatibility.
            pool_rare_categories: When True (default), low-cardinality categorical levels below
                min_sample_size/min_events are pooled into a single 'Other' bin before scoring,
                mirroring OptimalBinning's legacy behavior of merging sparse categories.
        """
        self.target = target
        cpu_count = os.cpu_count() or 1
        self.n_jobs = n_jobs if n_jobs != -1 else max(1, cpu_count - 1)
        self.min_sample_size = min_sample_size
        self.min_lift = min_lift
        self.min_events = min_events
        self.top_n_vars = top_n_vars
        self.max_segments = max_segments
        self.max_feature_reuse = max_feature_reuse
        self.segments: List[Dict[str, Any]] = []
        self.param_grid = param_grid or {}
        self.enable_diversity = enable_diversity
        self.enable_1way = enable_1way
        self.enable_2way = enable_2way
        self.enable_3way = enable_3way
        self.feature_groups = feature_groups or {}
        self.ignore_features = ignore_features or []
        self.feature_usage_counts: Dict[str, int] = {}
        self.sort_priority = sort_priority
        self.diagnostics_: List[Dict[str, Any]] = []
        self.naive_bins = naive_bins
        self.max_expansion_hops = max(1, int(max_expansion_hops))
        self.selection_metric = selection_metric
        self.expand_log_mode = expand_log_mode if expand_log_mode in ("none", "summary", "full") else "summary"
        self.max_combo_arity = max(1, min(int(max_combo_arity), _MAX_COMBO_ARITY_CAP))
        self.pool_rare_categories = bool(pool_rare_categories)

        # Binning engine selection: 'naive' is now the only engine.
        if binning_method != "naive":
            logger.warning(
                f"⚠️ binning_method='{binning_method}' is deprecated: the OptimalBinning "
                "code path has been removed. Falling back to the unified SQL-native "
                "('naive') binning engine, now covering both numeric and categorical "
                "variables (including metric-rank categorical merging and optional "
                "rare-category pooling)."
            )
            self.binning_method = "naive"
        else:
            self.binning_method = "naive"

        # Set up disk-backed DuckDB automatically if not provided
        if db_path is None or db_temp_dir is None:
            self.db_path, self.db_temp_dir = setup_disk_backed_db("experiments")
            logger.info(f"📂 Created disk-backed DB at: {self.db_path}")
        else:
            self.db_path = db_path
            self.db_temp_dir = db_temp_dir

        # --- Validate selection_metric upfront ---
        if not (self.selection_metric in ("iv", "response_rate")
                or callable(self.selection_metric)):
            logger.warning(
                f"⚠️ selection_metric='{self.selection_metric}' is not 'iv', "
                "'response_rate', or a callable; defaulting to 'iv' behavior."
            )
            self.selection_metric = "iv"

    # -------------------------------------------------------------------------
    # Feature groups & diversity
    # -------------------------------------------------------------------------

    def _validate_feature_groups(self, columns: List[str]) -> None:
        """
        Validates that all declared feature group variables exist in the target dataset.
        """
        if not self.feature_groups:
            return

        active_cols = set(columns) - {self.target} - set(self.ignore_features)
        validated_count = 0

        for group, vars_list in self.feature_groups.items():
            for var in vars_list:
                if var not in active_cols:
                    raise ValueError(
                        f"Schema Mismatch: Feature '{var}' declared in group '{group}' "
                        "was not found in the provided DataFrame/Table."
                    )
                validated_count += 1

        logger.info(f"✅ Feature group validation passed. ({validated_count} features mapped)")

    def get_group(self, var: str) -> str:
        """
        Returns the assigned business category for a feature, or the feature name itself.
        """
        for group, vars_list in self.feature_groups.items():
            if var in vars_list:
                return group
        return var

    def is_diverse(self, combo: Tuple[str, ...]) -> bool:
        """
        Ensures a tuple of features spans strictly distinct analytical groups.
        """
        if not self.enable_diversity:
            return True
        groups = [self.get_group(v) for v in combo]
        return len(groups) == len(set(groups))

    # -------------------------------------------------------------------------
    # Sort priority (single source of truth)
    # -------------------------------------------------------------------------

    def _sort_dims(self) -> Tuple[str, ...]:
        return _SORT_DIMENSIONS.get(self.sort_priority, _DEFAULT_SORT_DIMS)

    def _get_sort_key(self, rule: Dict[str, Any]) -> Tuple[float, ...]:
        """
        Utility function to shortlist rules based on users choice on which metric to prioritize.
        """
        return tuple(float(rule[d]) for d in self._sort_dims())

    # -------------------------------------------------------------------------
    # DuckDB plumbing
    # -------------------------------------------------------------------------

    def _load_into_duckdb(
        self,
        con: duckdb.DuckDBPyConnection,
        data: Any,
        table_name: str,
    ) -> Dict[str, str]:
        """
        Registers `data` under input_data_view, materializes it as `table_name`,
        and returns the {column: duckdb_type} mapping. Extracted from the
        previously repeated register/CREATE/DESCRIBE sequence.
        """
        con.register("input_data_view", data)
        con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM input_data_view")
        cols_info = con.execute(f"DESCRIBE {table_name}").fetchall()
        return {row[0]: row[1] for row in cols_info}

    @staticmethod
    def _is_numeric_duckdb_type(duckdb_type: str) -> bool:
        t = duckdb_type.upper()
        return any(
            token in t
            for token in [
                "INT", "BIGINT", "DOUBLE", "FLOAT", "DECIMAL", "REAL",
                "NUMERIC", "HUGEINT", "TINYINT", "SMALLINT",
            ]
        )

    # -------------------------------------------------------------------------
    # Unified SQL-native binning engine (the ONLY binning path)
    # -------------------------------------------------------------------------

    @staticmethod
    def _make_missing_case_expr(col: str) -> str:
        """SQL predicate identifying missing/null/empty category values."""
        quoted = '"' + col + '"'
        tokens_sql = ", ".join(f"'{tok}'" for tok in _CATEGORICAL_MISSING_TOKENS)
        return (
            f"({quoted} IS NULL OR TRIM(CAST({quoted} AS VARCHAR)) IN ({tokens_sql}))"
        )

    def _bin_label_expr(self, col: str, dtype: str) -> str:
        """
        Returns a SQL CASE expression mapping raw column values to bin labels.
        Numeric: quantile-range labels [lo, hi); Categorical: single-value
        bracket labels [value], or 'Other'/'Missing' for pooled/missing levels.
        """
        quoted = '"' + col + '"'
        if self._is_numeric_duckdb_type(dtype):
            q_step = 1.0 / float(self.naive_bins)
            q_list = [round(i * q_step, 6) for i in range(self.naive_bins + 1)]
            q_str = ", ".join(str(q) for q in q_list)

            # Edges are computed per column inside SQL via a scalar subquery, so
            # the whole binning stays in one SQL pass. CASE WHEN labels rows.
            case_parts = [
                f'WHEN {quoted} IS NULL THEN \'Missing\'',
                f'WHEN {quoted} < (SELECT QUANTILE_CONT({quoted}, {q_step}::DOUBLE) FROM current_df WHERE {quoted} IS NOT NULL) '
                f'AND {quoted} IS NOT NULL THEN \'[-inf, \' || (SELECT QUANTILE_CONT({quoted}, {q_step}::DOUBLE) FROM current_df WHERE {quoted} IS NOT NULL) || \')\'',
            ]
            # Build the general CASE via Python (same logic as the original):
            quantiles = [i * q_step for i in range(self.naive_bins + 1)]
            clauses = [f'WHEN {quoted} IS NULL THEN \'Missing\'']
            for i in range(self.naive_bins):
                lo_q, hi_q = quantiles[i], quantiles[i + 1]
                lo_val = (
                    f"(SELECT QUANTILE_CONT({quoted}, {lo_q}::DOUBLE) FROM current_df WHERE {quoted} IS NOT NULL)"
                )
                hi_val = (
                    f"(SELECT QUANTILE_CONT({quoted}, {hi_q}::DOUBLE) FROM current_df WHERE {quoted} IS NOT NULL)"
                )
                if i == 0:
                    clauses.append(
                        f"WHEN {quoted} < {hi_val} THEN '[-inf, ' || {hi_val} || ')'"
                    )
                elif i == self.naive_bins - 1:
                    clauses.append(
                        f"WHEN {quoted} >= {lo_val} THEN '[' || {lo_val} || ', inf)'"
                    )
                else:
                    clauses.append(
                        f"WHEN {quoted} >= {lo_val} AND {quoted} < {hi_val} "
                        f"THEN '[' || {lo_val} || ', ' || {hi_val} || ')'"
                    )
            return f"CASE {' '.join(clauses)} ELSE 'Missing' END"

        # Categorical: one [value] bin per distinct category (with optional Other)
        return (
            f"CASE WHEN {self._make_missing_case_expr(col)} THEN 'Missing' "
            f"ELSE '[' || CAST({quoted} AS VARCHAR) || ']'"
            f" END"
        )

    def _other_bin_case_expr(self, col: str, dtype: str) -> str:
        """
        Optional low-cardinality category pooling for categorical columns:
        levels with count < min_sample_size and events < min_events collapse to
        a single 'Other' bracket bin before scoring.
        """
        if not self.pool_rare_categories or self._is_numeric_duckdb_type(dtype):
            return None
        quoted = '"' + col + '"'
        return f"""
        CASE
            WHEN {self._make_missing_case_expr(col)} THEN 'Missing'
            WHEN CAST({quoted} AS VARCHAR) IN (
                SELECT level FROM (
                    SELECT CAST("{col}" AS VARCHAR) AS level,
                           COUNT(*) AS cnt,
                           SUM(CAST("{self.target}" AS DOUBLE)) AS evt
                    FROM current_df
                    GROUP BY 1
                ) WHERE cnt < {self.min_sample_size} AND evt < {self.min_events}
            ) THEN '[Other]'
            ELSE '[' || CAST({quoted} AS VARCHAR) || ']'
        END
        """

    def _score_stats(self, stats_rows, min_sample_size: int, min_events: int):
        """
        Computes the selected metric, max response rate, and IV from
        (bin_label, count, events) aggregates in Python (cheap; one list of
        tuples, per column).
        """
        total_events = sum((r[2] or 0.0) for r in stats_rows)
        total_non_events = sum(((r[1] or 0) - (r[2] or 0.0)) for r in stats_rows)

        iv_val = 0.0
        max_rr = 0.0
        bin_stats: List[Tuple[str, int, float]] = []

        for label, cnt, evt in stats_rows:
            evt = evt or 0.0
            cnt = cnt or 0
            bin_stats.append((label, cnt, evt))
            non_evt = cnt - evt
            if cnt >= min_sample_size and evt >= min_events:
                rr = evt / cnt
                if rr > max_rr:
                    max_rr = rr
            if total_events > 0 and total_non_events > 0:
                pct_events = max(evt / total_events, 1e-6)
                pct_non_events = max(non_evt / total_non_events, 1e-6)
                iv_val += (pct_non_events - pct_events) * np.log(pct_non_events / pct_events)

        if callable(self.selection_metric):
            score = float(self.selection_metric(bin_stats))
        elif self.selection_metric == "response_rate":
            score = max_rr
        else:
            score = iv_val

        return score, max_rr, iv_val, bin_stats

    def compute_iv_ranking_and_bin(
        self,
        con: duckdb.DuckDBPyConnection,
        eligible_cols: List[str],
        columns_types: Dict[str, str],
    ) -> Tuple[List[Dict[str, Union[str, float]]], Dict[str, np.ndarray]]:
        """
        Computes the selection metric (IV, response rate, or user-supplied
        callable) and pre-computed bins for every eligible column using the
        unified SQL-native binning engine.
        """
        logger.info(f"🔍 Computing {self.selection_metric if isinstance(self.selection_metric, str) else 'selection_metric'} and bins for {len(eligible_cols)} features...")

        def _worker(col: str) -> Tuple[str, float, float, Optional[np.ndarray]]:
            try:
                thread_con = con.cursor()
                dtype = columns_types[col]

                # One combined pass: categorical columns may get 'Other' pooling,
                # numeric columns get quantile edges -- both labeled per row, and
                # (label, count, events) aggregated in a single SQL statement.
                if self._is_numeric_duckdb_type(dtype):
                    label_expr = self._bin_label_expr(col, dtype)
                else:
                    label_expr = self._other_bin_case_expr(col, dtype) or self._bin_label_expr(col, dtype)

                stats_df = thread_con.execute(
                    f"""
                    SELECT
                        {label_expr} AS bin_label,
                        COUNT(*)::BIGINT AS cnt,
                        SUM(CAST("{self.target}" AS DOUBLE)) AS evt
                    FROM current_df
                    GROUP BY 1
                    """
                ).fetchall()

                score, max_rr, _iv, bin_stats = self._score_stats(
                    stats_df, self.min_sample_size, self.min_events
                )

                # Persist per-row bin labels as a numpy array (unchanged shape)
                transformed_bins = thread_con.execute(
                    f"SELECT {label_expr} AS bin_label FROM current_df"
                ).fetchnumpy()["bin_label"].astype(str)

                thread_con.close()
                return col, float(score), float(max_rr), transformed_bins

            except Exception as e:
                logger.debug(f"Computation failed for {col}: {e}")
                return col, 0.0, 0.0, None

        results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_worker)(col) for col in eligible_cols
        )
        ranking: List[FeatureScore] = []
        precomputed_bins: Dict[str, np.ndarray] = {}
        # Per-bin selection-metric scores cached for the dtype-agnostic merge
        # axis: _metric_ranked_bins ranks bins by this value (ascending) instead
        # of always using bin response rate.
        precomputed_bin_scores: Dict[str, np.ndarray] = {}
        for col, score, max_rr, bins in results:
            ranking.append(FeatureScore(variable=col, iv=float(score), max_rr=float(max_rr)))
            if bins is not None:
                precomputed_bins[col] = bins

        # Ranking by the selected metric, descending -- identical semantics to
        # before (iv used 'iv'; response_rate used 'max_rr'; callable uses score).
        if self.selection_metric == "response_rate":
            ranking.sort(key=lambda x: x.max_rr, reverse=True)
        else:
            ranking.sort(key=lambda x: x.iv, reverse=True)

        # Recompute per-bin axis scores from the same bin label expression
        # used by the original per-column stats pass (numeric or categorical,
        # including rare-category pooling) in one SQL pass per column
        # (cheap, sequential, main connection).
        for col, bins in precomputed_bins.items():
            dtype = columns_types.get(col, "VARCHAR")
            label_expr = (
                self._bin_label_expr(col, dtype)
                if self._is_numeric_duckdb_type(dtype)
                else (
                    self._other_bin_case_expr(col, dtype)
                    or self._bin_label_expr(col, dtype)
                )
            )
            try:
                rows = con.execute(
                    f"""
                    SELECT {label_expr} AS bin_label,
                           COUNT(*)::BIGINT AS cnt,
                           SUM(CAST("{self.target}" AS DOUBLE)) AS evt
                    FROM current_df
                    GROUP BY 1
                    """
                ).fetchall()
            except Exception:
                rows = None
            if rows:
                per_bin = self._per_bin_metric_scores(rows)
                score_map = {row[0]: per_bin.get(row[0], 0.0) for row in rows}
                precomputed_bin_scores[col] = np.array(
                    [score_map.get(b, 0.0) for b in bins.astype(str)], dtype=float
                )
            else:
                precomputed_bin_scores[col] = np.zeros(len(bins), dtype=float)
        return ([r.to_dict() for r in ranking], precomputed_bins, precomputed_bin_scores)

    def _per_bin_metric_scores(self, stats_rows) -> Dict[str, float]:
        """
        Per-bin scores of the configured selection_metric, given
        (label, cnt, evt) aggregates. This is the axis used by
        _metric_ranked_bins to rank categorical bins for adjacent-merge
        adjacency (replacing the fixed bin response rate).
        """
        total_events = max(sum((r[2] or 0.0) for r in stats_rows), 1e-12)
        total_non_events = max(
            sum(((r[1] or 0) - (r[2] or 0.0)) for r in stats_rows), 1e-12
        )
        scores: Dict[str, float] = {}
        if callable(self.selection_metric):
            for label, cnt, evt in stats_rows:
                scores[str(label)] = float(
                    self.selection_metric([(str(label), cnt or 0, evt or 0.0)])
                )
        else:
            for label, cnt, evt in stats_rows:
                cnt, evt = cnt or 0, evt or 0.0
                non_evt = cnt - evt
                if self.selection_metric == "response_rate":
                    scores[str(label)] = (
                        (evt / cnt) if cnt >= self.min_sample_size else 0.0
                    )
                else:  # iv: per-bin contribution component of the column IV
                    if total_events > 0 and total_non_events > 0:
                        pct_events = max(evt / total_events, 1e-6)
                        pct_non_events = max(non_evt / total_non_events, 1e-6)
                        scores[str(label)] = (
                            (pct_non_events - pct_events)
                            * np.log(pct_non_events / pct_events)
                        )
                    else:
                        scores[str(label)] = 0.0
        return scores
    # -------------------------------------------------------------------------
    # Adjacent-bin merging (dtype-agnostic via metric-rank adjacency)
    # -------------------------------------------------------------------------

    @staticmethod
    def _bin_sort_key(label: str) -> Tuple[int, float, str]:
        """
        Numeric-aware sort key for a bin label. Bracket tokens that are not
        numeric ranges (e.g. categorical [value] or [Other]) fall through to
        the (1, 0.0, label) tuple, preserving legacy behavior.
        """
        m = re.match(r"^[\[\(]\s*(-?inf|-?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*,", label)
        if m:
            raw = m.group(1).lower()
            if raw == "-inf":
                val = float("-inf")
            elif raw == "inf":
                val = float("inf")
            else:
                val = float(raw)
            return (0, val, "")
        return (1, 0.0, label)

    def _metric_ranked_bins(
        self,
        con: duckdb.DuckDBPyConnection,
        combo: Tuple[str, ...],
        col: str,
    ) -> List[Dict[str, Any]]:
        """
        Returns bin-level (label, count, events) aggregates for `col` within
        the `combo` GROUP BY context, sorted by the SAME selection metric used
        for variable ranking. Numeric bins keep their natural ordinal order;
        categorical bins get metric-rank order (ascending score), mirroring
        the numeric ordinal axis. Returns dicts with _rank and _sort_key so
        _expand_adjacent_bins can consume both dtype classes identically.
        """
        other_cols = [c for c in combo if c != col]
        group_cols = other_cols + [col]
        select_cols = ", ".join(f'CAST("{c}" AS VARCHAR) AS "{c}"' for c in group_cols)
        group_by_cols = ", ".join(f'"{c}"' for c in group_cols)

        try:
            rows = con.execute(
                f"""
                SELECT {select_cols},
                       COUNT("{self.target}")::BIGINT AS cnt,
                       SUM(CAST("{self.target}" AS DOUBLE)) AS evt
                FROM binned_df
                GROUP BY {group_by_cols}
                """
            ).fetchall()
        except Exception:
            logger.debug(f"↩️ Expansion aggregate query failed for variable '{col}' in combo {combo}")
            return []

        if not rows:
            return []

        columns = other_cols + [col, "cnt", "evt"]
        processed = []
        for r in rows:
            row_dict = dict(zip(columns, r))
            if row_dict[col] == "Missing":
                continue
            row_dict["_sort_key"] = self._bin_sort_key(row_dict[col])
            processed.append(row_dict)

        if not processed:
            return []

        if other_cols:
            processed.sort(key=lambda x: tuple(x[c] for c in other_cols) + (x["_sort_key"],))
            grouped: List[Tuple[Any, List[Dict[str, Any]]]] = []
            from itertools import groupby
            for k, g in groupby(processed, key=lambda x: tuple(x[c] for c in other_cols)):
                grouped.append((k, list(g)))
        else:
            processed.sort(key=lambda x: x["_sort_key"])
            grouped = [((), processed)]

        out = []
        for group_key, g_list in grouped:
            n = len(g_list)
            if n < 2:
                continue
            bins = [x[col] for x in g_list]
            # Axis scores for the merge rank order: numeric bins keep their
            # natural ordinal order via _bin_sort_key; categorical bins are
            # ranked by the CONFIGURED selection_metric (per-bin scores
            # cached in self._precomputed_bin_scores) instead of the fixed
            # bin response rate, so the merge axis honors iv / response_rate /
            # any user-supplied callable metric.
            _sk = self._bin_sort_key
            _is_num = lambda lbl: _sk(lbl)[0] == 0
            axis_scores = []
            bin_scores_map = getattr(self, "_precomputed_bin_scores", None) or {}
            col_scores = bin_scores_map.get(col)
            for idx, x in enumerate(g_list):
                lbl = x[col]
                if _is_num(lbl):
                    axis_scores.append(float("inf"))  # ordinal path
                elif col_scores is not None and idx < len(col_scores):
                    # Map this bin label back to its precomputed per-bin score.
                    axis_scores.append(float(col_scores[idx]))
                else:
                    axis_scores.append(
                        (x["evt"] / x["cnt"]) if x["cnt"] > 0 else 0.0
                    )
            rates = [
                (x["evt"] / x["cnt"]) if x["cnt"] > 0 else 0.0 for x in g_list
            ]
            if other_cols:
                key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
                other_val = dict(zip(other_cols, key_tuple))
            else:
                other_val = {}
            out.append(
                {
                    "other_val": other_val,
                    "bins": bins,
                    "cnt": np.array([x["cnt"] for x in g_list], dtype=float),
                    "evt": np.array([x["evt"] for x in g_list], dtype=float),
                    "rates": rates,
                    "axis_scores": np.array(axis_scores, dtype=float),
                }
            )
        return out

    def _expand_adjacent_bins(
        self,
        con: duckdb.DuckDBPyConnection,
        combo: Tuple[str, ...],
        base_rate: float,
        base_results: List[Dict[str, Any]],
        seen_rules: Optional[set] = None,
        min_sample_size: Optional[int] = None,
        min_lift: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Attempts to merge adjacent neighbour bin(s) on every variable in combo.
        Adjacency is metric-rank order for both numeric (natural ordinal) and
        categorical (response-rate rank) bins; merging windows are computed via
        _candidate_windows over that rank position.
        """
        if not base_results:
            return []

        if seen_rules is None:
            seen_rules = set()
        for r in base_results:
            seen_rules.add(r["rule"])

        base_lookup: Dict[str, Dict[str, Any]] = {r["rule"]: r for r in base_results}
        max_hops = max(1, int(getattr(self, "max_expansion_hops", 1)))

        # Explicit local thresholds -- no reading self.min_sample_size /
        # self.min_lift mid-iteration (thread-safety / readability fix).
        eff_min_sample = min_sample_size if min_sample_size is not None else self.min_sample_size
        eff_min_lift = min_lift if min_lift is not None else self.min_lift

        expanded: List[Dict[str, Any]] = []

        for col in combo:
            groups = self._metric_ranked_bins(con, combo, col)
            if not groups:
                continue

            for g in groups:
                bins = g["bins"]
                cnt = g["cnt"]
                evt = g["evt"]
                rates = g["rates"]
                other_val = g["other_val"]

                # Rank bins for merge adjacency: categorical bins by ascending
                # configured selection_metric score (axis_scores); numeric bins
                # keep their natural ordinal order via _bin_sort_key.
                _sk = self._bin_sort_key
                rank_order = sorted(
                    range(len(bins)),
                    key=lambda i: (_sk(bins[i])[0], g["axis_scores"][i]),
                )

                # Reorder cumulative sums by rank so windows walk rank-adjacent bins.
                sorted_bins = [bins[i] for i in rank_order]
                sorted_cnt = cnt[rank_order]
                sorted_evt = evt[rank_order]
                cum_cnt = np.concatenate(([0.0], np.cumsum(sorted_cnt)))
                cum_evt = np.concatenate(([0.0], np.cumsum(sorted_evt)))

                n = len(sorted_bins)
                for ridx in range(n):
                    orig_idx = rank_order[ridx]
                    rule_at_idx = " & ".join(
                        f"{c}={other_val[c]}" if c != col else f"{c}={bins[orig_idx]}"
                        for c in combo
                    )
                    base_result = base_lookup.get(rule_at_idx)
                    if base_result is None:
                        continue
                    base_events = base_result["events"]

                    for lo, hi in self._candidate_windows(ridx, n, max_hops):
                        exp_count = cum_cnt[hi + 1] - cum_cnt[lo]
                        exp_events = cum_evt[hi + 1] - cum_evt[lo]
                        if exp_count <= 0:
                            continue

                        exp_rate = (exp_events / exp_count) * 100.0
                        exp_lift = exp_rate / (base_rate * 100.0) if base_rate > 0 else 0.0

                        if not (
                            exp_lift >= eff_min_lift
                            and exp_events >= self.min_events
                            and exp_count >= eff_min_sample
                            and exp_events > base_events
                        ):
                            continue

                        window_labels_sorted = sorted(sorted_bins[lo:hi + 1], key=self._bin_sort_key)
                        merged_label = f"[{', '.join(window_labels_sorted)}]"

                        rule_parts = []
                        for c in combo:
                            if c == col:
                                rule_parts.append(f"{c}={merged_label}")
                            else:
                                rule_parts.append(f"{c}={other_val[c]}")
                        rule_str = " & ".join(rule_parts)

                        if rule_str in seen_rules:
                            continue
                        seen_rules.add(rule_str)

                        expanded.append(
                            {
                                "rule": rule_str,
                                "count": int(exp_count),
                                "rate": float(exp_rate),
                                "lift": float(exp_lift),
                                "events": float(exp_events),
                                "combo_vars": combo,
                                "base_events": base_events,
                            }
                        )

        return expanded

    @staticmethod
    def _candidate_windows(idx: int, n: int, max_hops: int):
        """
        Yields (lo, hi) inclusive index windows around `idx`.
        """
        for hops in range(1, max_hops + 1):
            lo, hi = idx - hops, idx
            if lo < 0:
                break
            yield (lo, hi)
        for hops in range(1, max_hops + 1):
            lo, hi = idx, idx + hops
            if hi >= n:
                break
            yield (lo, hi)

    # -------------------------------------------------------------------------
    # Combination aggregation (batched SQL, structured Rule objects)
    # -------------------------------------------------------------------------

    def _parse_predicates_from_rule_str(self, rule_str: str) -> List[Tuple[str, str, Any]]:
        """
        Parses an externally supplied rule string into a structured predicate
        list, preserving legacy parse behavior for anything not internally
        generated. Returned ops: 'range' | 'in' | 'is_null' | 'eq'.
        """
        preds: List[Tuple[str, str, Any]] = []
        parts = [p.strip() for p in rule_str.split("&")]
        for part in parts:
            if "=" not in part:
                continue
            col, interval = [x.strip() for x in part.split("=", 1)]

            if interval in ("Special", "Missing"):
                preds.append((col, "is_null", None))
                continue

            # Bracketed union, e.g. [[-inf, 524.0), [524.0, 599.0)] or
            # [[agent], [partner]]. Extract each [..) / [..] / (..] sub-
            # bracket explicitly; classify the union as numeric (range_multi)
            # only when every sub-bracket is itself a comma-pair range.
            if interval.startswith("[["):
                sub_brackets = re.findall(r"\[[^\[\]]*?\)|\[[^\[\]]*?\]|\([^\[\]]*?\)", interval)
                sub_ranges = []
                sub_values = []
                for sb in sub_brackets:
                    if "," in sb[1:-1] and sb[-1] in (")", "]"):
                        content = sb[1:-1]
                        lo, hi = [x.strip() for x in content.split(",", 1)]
                        sub_ranges.append((sb[0], lo, hi, sb[-1]))
                    else:
                        sub_values.append(sb[1:-1].strip())
                if sub_ranges and not sub_values:
                    preds.append((col, "range_multi", sub_ranges))
                    continue
                if sub_values:
                    clean_values = [v.strip("'\"") for v in sub_values if v.strip() not in ("", "-inf", "inf")]
                    if clean_values and not sub_ranges:
                        preds.append((col, "in", clean_values))
                        continue
                if sub_ranges and sub_values:
                    # Mixed union, e.g. [[-inf, 1.0), [5]] -- emit both parts.
                    preds.append((col, "range_multi", sub_ranges))
                    preds.append((col, "in", sub_values))
                    continue

            bracket_match = _BRACKET_REGEX.search(interval)
            is_categorical = False
            if bracket_match:
                content = bracket_match.group(1)
                if any(k in interval for k in ("'", '"', "Array", "Categorical")) or not interval.startswith(("[", "(")):
                    is_categorical = True
                elif len(content.split(",")) > 2:
                    is_categorical = True
                elif re.search(r"\[\S", content) or "]," in content:
                    # Bracketed value list with nested brackets, e.g.
                    # [[agent], [partner]], or multiple sub-brackets which
                    # the range_multi branch did not match.
                    is_categorical = True

            if is_categorical and bracket_match:
                import ast
                try:
                    raw_items = ast.literal_eval(bracket_match.group(0))
                except Exception:
                    raw_content = bracket_match.group(1)
                    if "," not in raw_content:
                        raw_content = re.sub(r"'\s+'", "','", raw_content)
                        raw_content = re.sub(r'"\s+"', '","', raw_content)
                        raw_content = re.sub(r"\s+", ",", raw_content)
                    raw_items = [
                        i.strip().strip("'").strip('"')
                        for i in raw_content.split(",")
                        if i.strip()
                    ]
                preds.append((col, "in", list(raw_items)))
                continue

            if interval.startswith(("[", "(")):
                inner = interval[1:-1].strip()
                if "," in inner:
                    left_char, right_char = interval[0], interval[-1]
                    lower_str, upper_str = [x.strip() for x in inner.split(",", 1)]
                    preds.append((col, "range", (left_char, lower_str, upper_str, right_char)))
                else:
                    clean_val = inner.strip("'\"")
                    if clean_val.replace(".", "", 1).replace("-", "", 1).isdigit():
                        preds.append((col, "eq", clean_val))
                    else:
                        preds.append((col, "eq", str(clean_val)))
                continue

        return preds

    def _predicate_to_sql(self, col: str, op: str, value: Any) -> str:
        """Formats a single structured predicate into a SQL condition."""
        if op == "is_null":
            return f"{col} IS NULL"
        if op == "eq":
            if isinstance(value, str):
                return f"{col} = '{value}'"
            return f"{col} = {value}"
        if op == "in":
            formatted_items = ", ".join(
                f"'{item}'" if isinstance(item, str) else str(item) for item in value
            )
            return f"{col} IN ({formatted_items})"
        if op == "range":
            left_char, lower_str, upper_str, right_char = value
            conds = []
            if lower_str.lower() != "-inf":
                op_str = ">=" if left_char == "[" else ">"
                conds.append(f"{col} {op_str} {lower_str}")
            if upper_str.lower() != "inf":
                op_str = "<=" if right_char == "]" else "<"
                conds.append(f"{col} {op_str} {upper_str}")
            return " AND ".join(conds)
        if op == "range_multi":
            ranges = value
            def _sort_key(r):
                val = r[1]
                if val.lower() == "-inf":
                    return float("-inf")
                try:
                    return float(val)
                except Exception:
                    return 0.0
            ranges_sorted = sorted(ranges, key=_sort_key)
            overall_lower_char, overall_lower = ranges_sorted[0][0], ranges_sorted[0][1]
            overall_upper_char, overall_upper = ranges_sorted[-1][3], ranges_sorted[-1][2]
            range_conds = []
            if overall_lower.lower() != "-inf":
                op_str = ">=" if overall_lower_char == "[" else ">"
                range_conds.append(f"{col} {op_str} {overall_lower}")
            if overall_upper.lower() != "inf":
                op_str = "<=" if overall_upper_char == "]" else "<"
                range_conds.append(f"{col} {op_str} {overall_upper}")
            return " AND ".join(range_conds)
        return ""

    def parse_rule_to_sql(self, rule_str: str) -> str:
        """
        Translates rule strings into production SQL WHERE clause.

        Internally generated rules keep structured predicates, so SQL emission
        is a thin formatter over them (no regex/ast round-trip, no
        StringArray-repr bug class). External rule strings are parsed with the
        legacy regex/ast path via _parse_predicates_from_rule_str.
        """
        conds: List[str] = []
        for part in [p.strip() for p in rule_str.split("&")]:
            if "=" not in part:
                continue
            col, interval = [x.strip() for x in part.split("=", 1)]
            preds = [(col, op, val) for (c, op, val) in getattr(self, "_last_rule_predicates", []) if c == col]
            if not preds:
                preds = self._parse_predicates_from_rule_str(part)
            for _c, op, val in preds:
                cond = self._predicate_to_sql(col, op, val)
                if cond:
                    conds.append(cond)
        return " AND ".join(f"({cond})" for cond in conds)

    def _predicate_rule_to_sql(self, rule: Rule) -> str:
        """Formats an internally generated Rule (with structured predicates)."""
        if not rule.predicates:
            rule.predicates = self._parse_predicates_from_rule_str(rule.rule)
        conds = [self._predicate_to_sql(c, op, v) for c, op, v in rule.predicates]
        return " AND ".join(f"({cond})" for cond in conds)

    # -------------------------------------------------------------------------
    # Logging helpers (core logic separated from reporting)
    # -------------------------------------------------------------------------

    def _log_expansion_summary(self, expansion_stats: Dict[str, Dict[str, Any]], all_expanded: List[Dict[str, Any]]) -> None:
        mode = getattr(self, "expand_log_mode", "summary")
        if not expansion_stats or mode == "none":
            return
        logger.info("🔀 Adjacent-bin expansion summary")
        logger.info(
            f"   {'Combo':<42} {'#exp':>5}  {'Best Δevents':>12}  {'Best lift':>9}"
        )
        logger.info("   " + "-" * 72)

        for combo_key, st in sorted(
            expansion_stats.items(), key=lambda x: x[1]["best_delta"], reverse=True
        ):
            logger.info(
                f"   {combo_key:<42} {st['n_exp']:>5}  "
                f"{st['best_delta']:>+12.0f}  {st['best_lift']:>8.2f}x"
            )

        total_exp = sum(s["n_exp"] for s in expansion_stats.values())
        logger.info(f"   → Total expanded candidates generated: {total_exp}")

        if mode == "full":
            top = sorted(all_expanded, key=lambda x: x["events"], reverse=True)[:3]
            logger.info("   Top expanded rules:")
            for i, e in enumerate(top, 1):
                delta = e["events"] - e.get("base_events", e["events"])
                logger.info(
                    f"     {i}. {e['rule'][:90]}"
                    f"  | events {e['events']:.0f} (Δ{delta:+.0f}) | lift {e['lift']:.2f}x"
                )

    def _log_expansion_vs_champion(
        self,
        expanded_rules: List[Dict[str, Any]],
        actual_lift: float,
        actual_rate: float,
        actual_count: int,
        actual_events: float,
        champion_rule: str,
    ) -> None:
        if not expanded_rules:
            return
        expanded_rules.sort(key=lambda x: self._get_sort_key(x), reverse=True)
        top_exp = expanded_rules[:5]

        logger.info("📊 Top adjacent-merge candidates vs final champion")
        logger.info(
            f"   {'Rank':<5} {'Type':<10} {'Lift':>6} {'Rate%':>7} {'Count':>8} {'Events':>8}  Rule"
        )
        logger.info("   " + "-" * 90)

        logger.info(
            f"   {'★':<5} {'CHAMPION':<10} "
            f"{actual_lift:>6.2f}x {actual_rate:>6.1f}% "
            f"{actual_count:>8} "
            f"{actual_events:>8.0f}  "
            f"{champion_rule[:60]}"
        )

        champ_key = self._get_sort_key({
            "lift": actual_lift,
            "rate": actual_rate,
            "count": actual_count,
            "events": actual_events,
        })

        for idx, e in enumerate(top_exp, 1):
            cand_key = self._get_sort_key(e)

            if cand_key > champ_key:
                reason = "would have beaten champion (but failed raw validation)"
            else:
                priority_order = list(self._sort_dims())
                champ_vals = {
                    "lift": actual_lift,
                    "rate": actual_rate,
                    "count": actual_count,
                    "events": actual_events,
                }
                cand_vals = {
                    "lift": e["lift"],
                    "rate": e["rate"],
                    "count": e["count"],
                    "events": e["events"],
                }
                reason = "ranked lower by sort_priority"
                for dim in priority_order:
                    if cand_vals[dim] < champ_vals[dim]:
                        reason = (
                            f"{_DIM_LABELS[dim]} ({cand_vals[dim]:.2f} < {champ_vals[dim]:.2f})"
                        )
                        break
                    elif cand_vals[dim] > champ_vals[dim]:
                        break

            logger.info(
                f"   {idx:<5} {'expanded':<10} "
                f"{e['lift']:>6.2f}x {e['rate']:>6.1f}% "
                f"{e['count']:>8} {e['events']:>8.0f}  "
                f"{e['rule'][:55]}  → {reason}"
            )

    def _agg_combinations(
        self,
        con: duckdb.DuckDBPyConnection,
        combo_list: List[Tuple[str, ...]],
        base_rate: float,
        min_sample_size: Optional[int] = None,
        min_lift: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Batches SQL GROUP BY queries for a list of feature combinations.
        Returns legacy-shaped dicts (Rule.to_dict), so public behavior is unchanged.
        """
        if not combo_list:
            return []

        queries = []
        for combo in combo_list:
            # combo may be a tuple of strings (1-way seed) or a frozenset of
            # strings (Apriori higher arities); normalize either way.
            combo_vars = sorted(
                combo if all(isinstance(v, str) for v in combo)
                else {v for fs in combo for v in fs}
            )
            cols_str = ", ".join([f'"{c}"' for c in combo_vars])
            rule_concat = " || ' & ' || ".join(
                [f"'{c}=' || CAST(\"{c}\" AS VARCHAR)" for c in combo_vars]
            )
            combo_str = ",".join(combo_vars)

            eff_min_sample = min_sample_size if min_sample_size is not None else self.min_sample_size

            query = f"""
                    SELECT
                    {rule_concat} AS rule,
                    COUNT("{self.target}")::BIGINT AS count,
                    SUM(CAST("{self.target}" AS DOUBLE)) AS events,
                    '{combo_str}' AS combo_vars_str
                    FROM binned_df
                    GROUP BY {cols_str}
                    HAVING COUNT("{self.target}") >= {eff_min_sample}
                    AND SUM(CAST("{self.target}" AS DOUBLE)) >= {self.min_events}
            """
            queries.append(query)

        valid_results = []

        # Right-size the batch from configured memory and an estimated per-row
        # footprint instead of the fixed chunk_size=100 magic constant.
        total_mem_gb = psutil.virtual_memory().total / (1024**3)
        target_memory_gb = max(1, int(total_mem_gb * 0.8))
        bytes_per_result_row = 512  # conservative estimate (rule string + ints + floats)
        target_memory_bytes = target_memory_gb * (1024**3)
        chunk_size = max(25, min(1000, int(target_memory_bytes / max(bytes_per_result_row, 1))))

        per_combo_base: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {
            combo: [] for combo in combo_list
        }

        for i in range(0, len(queries), chunk_size):
            chunk = queries[i:i + chunk_size]
            union_query = " UNION ALL ".join(chunk)

            res = con.execute(union_query).fetchall()
            for row_idx, (rule, count, events, combo_vars_str) in enumerate(res):
                rate = (events / count) * 100.0 if count > 0 else 0
                lift = rate / (base_rate * 100.0) if base_rate > 0 else 0
                combo_key = tuple(combo_vars_str.split(","))

                if events >= self.min_events:
                    entry = {
                        "rule": rule,
                        "count": count,
                        "rate": rate,
                        "lift": lift,
                        "events": events,
                        "combo_vars": combo_key,
                    }
                    valid_results.append(entry)
                    if combo_key in per_combo_base:
                        per_combo_base[combo_key].append(entry)

        all_expanded: List[Dict[str, Any]] = []
        expansion_stats: Dict[str, Dict[str, Any]] = {}
        shared_seen_rules: set = set()

        eff_min_sample = min_sample_size if min_sample_size is not None else self.min_sample_size
        eff_min_lift = min_lift if min_lift is not None else self.min_lift

        for combo in combo_list:
            base = per_combo_base.get(combo, [])
            if not base:
                continue
            expanded = self._expand_adjacent_bins(
                con, combo, base_rate, base, seen_rules=shared_seen_rules,
                min_sample_size=eff_min_sample, min_lift=eff_min_lift,
            )
            if expanded:
                valid_results.extend(expanded)
                all_expanded.extend(expanded)

                combo_key = " & ".join(combo)
                best = max(expanded, key=lambda x: x["events"])
                delta = best["events"] - best.get("base_events", best["events"])
                expansion_stats[combo_key] = {
                    "n_exp": len(expanded),
                    "best_delta": delta,
                    "best_lift": best["lift"],
                    "best_rule": best["rule"],
                    "best_events": best["events"],
                }

        self._log_expansion_summary(expansion_stats, all_expanded)

        return valid_results

    # -------------------------------------------------------------------------
    # Core extraction (no instance-state mutation during iteration)
    # -------------------------------------------------------------------------

    def _arity_enabled(self, k: int) -> bool:
        """
        Backward-compat mapping of enable_1way/2way/3way booleans onto the
        N-way loop. enable_3way=False caps arity at 2 regardless of
        max_combo_arity, etc.
        """
        if k == 1:
            return bool(self.enable_1way)
        if k == 2:
            return bool(self.enable_2way)
        if k == 3:
            return bool(self.enable_2way) and bool(self.enable_3way)
        return True  # k == 4, 5 (and beyond, capped at 5) gated only by max_combo_arity

    def extract_segments(self, data: Any) -> List[Dict[str, Any]]:
        """
        Sequentially extracts high-lift rules on the residual dataset.
        """
        logger.info("🚀 Starting hierarchical segment extraction...")

        con = duckdb.connect(self.db_path)

        total_cores = os.cpu_count() or 1
        target_threads = max(1, total_cores - 2) if total_cores > 4 else total_cores
        total_mem_gb = psutil.virtual_memory().total / (1024**3)
        target_memory_gb = max(1, int(total_mem_gb * 0.8))

        con.execute(f"SET threads = {target_threads};")
        con.execute(f"SET memory_limit = '{target_memory_gb}GB';")
        con.execute(f"PRAGMA temp_directory='{self.db_temp_dir}';")
        con.execute("SET preserve_insertion_order = false;")

        logger.info(
            f"⚙️ DuckDB Configured for Disk Spilling: Threads={target_threads}/{total_cores}, "
            f"MemoryLimit={target_memory_gb}GB, TempDir={self.db_temp_dir}"
        )
        logger.info(f"📊 Sort priority: {self.sort_priority}")
        logger.info(f"📦 Binning method: {self.binning_method} (naive_bins={self.naive_bins})")
        logger.info(f"🔗 Max combination arity: {self.max_combo_arity} (pool_rare_categories={self.pool_rare_categories})")

        columns_types = self._load_into_duckdb(con, data, "current_df")
        all_cols = list(columns_types.keys())

        if self.enable_diversity:
            self._validate_feature_groups(all_cols)

        eligible_cols = [
            c for c in all_cols if c != self.target and c not in self.ignore_features
        ]
        self.feature_usage_counts = {col: 0 for col in eligible_cols}

        if self.param_grid:
            sizes = self.param_grid.get("min_sample_size", [self.min_sample_size])
            lifts = self.param_grid.get("min_lift", [self.min_lift])
            experiments = [
                {"min_sample_size": s, "min_lift": l}
                for s, l in zip(sizes, lifts)
            ]
            logger.info(f"📊 Dynamic Grid Search Enabled: {len(experiments)} configurations.")
        else:
            experiments = [{"min_sample_size": self.min_sample_size, "min_lift": self.min_lift}]

        original_base_rate = con.execute(
            f'SELECT AVG(CAST("{self.target}" AS DOUBLE)) FROM current_df'
        ).fetchone()[0] or 0.0

        # Effective thresholds stay LOCAL -- never mutate self.min_sample_size /
        # self.min_lift / self.min_events mid-loop.
        abs_min_sample_size = self.min_sample_size
        abs_min_events = self.min_events
        abs_min_lift = self.min_lift

        logger.info(f"🔒 Locking Original Base Rate: {original_base_rate*100:.2f}%")

        # Accumulated exclusion predicates, applied lazily as a WHERE clause on
        # the original table instead of materializing temp_residual each round.
        exclusion_preds: List[str] = []

        for i in range(1, self.max_segments + 1):
            residual_filter = " AND ".join(
                [f"NOT ({pred})" for pred in exclusion_preds]
            ) if exclusion_preds else "1=1"

            res = con.execute(
                f'SELECT AVG("{self.target}"), COUNT(*) FROM current_df WHERE {residual_filter}'
            ).fetchone()
            current_base_rate, current_volume = res[0] or 0.0, res[1] or 0

            min_floor_volume = min(exp["min_sample_size"] for exp in experiments)

            if current_base_rate == 0 or current_volume < min_floor_volume:
                logger.info(
                    f"⏹️ Stopping: base_rate={current_base_rate}, volume={current_volume} < "
                    f"min_floor={min_floor_volume}"
                )
                break

            logger.info(
                f"🔄 Iteration {i} | Remaining Volume: {current_volume:,} | "
                f"Base Rate: {current_base_rate*100:.2f}%"
            )

            iv_ranking_dicts, precomputed_bins, precomputed_bin_scores = (
                self.compute_iv_ranking_and_bin(con, eligible_cols, columns_types)
            )
            # Per-bin axis scores for the dtype-agnostic merge rank order.
            self._precomputed_bin_scores = precomputed_bin_scores
            iv_ranking = [FeatureScore.from_dict(r) for r in iv_ranking_dicts]

            if self.selection_metric == "response_rate":
                current_score_map = {row.variable: row.max_rr for row in iv_ranking}
            else:
                current_score_map = {row.variable: row.iv for row in iv_ranking}

            top_n_variable_names = [r.variable for r in iv_ranking[:self.top_n_vars]]
            iteration_snapshot = {}
            for col in eligible_cols:
                used_count = self.feature_usage_counts.get(col, 0)
                current_score = current_score_map.get(col, 0.0)

                if used_count >= self.max_feature_reuse:
                    status = "Excluded (Max Feature Reuse Exceeded)"
                elif current_score <= 0.0:
                    status = f"Excluded ({self.selection_metric.upper() if isinstance(self.selection_metric, str) else 'SELECTION_METRIC'} is Zero/Invalid)"
                elif col not in top_n_variable_names:
                    status = "Excluded (Outside Top N Features by Score)"
                else:
                    status = "Eligible for Combination Search"

                iteration_snapshot[col] = {
                    "metric_score": current_score,
                    "metric_type": self.selection_metric if isinstance(self.selection_metric, str) else "custom_callable",
                    "times_used_previously": used_count,
                    "status": status,
                }

            self.diagnostics_.append(
                {
                    "iteration": i,
                    "residual_volume": current_volume,
                    "base_rate": current_base_rate,
                    "features_state": iteration_snapshot,
                    "winning_segment": None,
                }
            )

            allowed_vars = [
                row.variable
                for row in iv_ranking
                if self.feature_usage_counts.get(row.variable, 0) < self.max_feature_reuse
            ]
            top_vars = allowed_vars[:self.top_n_vars]
            if not top_vars:
                logger.warning("⚠️ All eligible features exhausted. Aborting.")
                break

            valid_vars = []
            for v in top_vars:
                if v in precomputed_bins and len(np.unique(precomputed_bins[v])) > 1:
                    valid_vars.append(v)
            if not valid_vars:
                logger.warning("⚠️ No valid binned variables found. Stopping.")
                break

            # binned_df must align with current_df (which keeps all rows;
            # residual filtering is applied lazily via WHERE at query time).
            # Precomputed bin arrays are full-length; keep the target column
            # full-length too for a consistent frame.
            raw_target_arr = con.execute(
                f'SELECT "{self.target}" FROM current_df'
            ).fetchnumpy()[self.target]
            if isinstance(raw_target_arr, np.ma.MaskedArray):
                clean_target_arr = raw_target_arr.filled(0)
            else:
                clean_target_arr = raw_target_arr

            binned_data = {self.target: clean_target_arr}
            for v in valid_vars:
                binned_data[v] = precomputed_bins[v]

            # Dict-of-arrays replacement scans are fragile across DuckDB
            # versions (object-dtype arrays fail the scan), so materialize via
            # a pandas DataFrame which always scans reliably.
            con.execute("DROP TABLE IF EXISTS binned_df")
            con.register("binned_data", pd.DataFrame(binned_data))
            con.execute("CREATE TABLE binned_df AS SELECT * FROM binned_data")

            global_min_sample = min(exp["min_sample_size"] for exp in experiments)
            global_min_lift = min(exp["min_lift"] for exp in experiments)

            all_candidate_rules: List[Dict[str, Any]] = []

            # --- Generalized N-way Apriori combination search ---
            # A single loop over k = 1..max_combo_arity, reusing the existing
            # anti-monotonic pruning (a k-combo is only tried when every one of
            # its (k-1)-subsets already survived as a valid rule). enable_*
            # booleans map onto arity_enabled(k) for backward compatibility.
            valid_1way_vars = set()
            valid_sets_by_arity: Dict[int, set] = {}

            # Level 1 (singles) is always the seed for the Apriori cascade.
            res_1 = self._agg_combinations(
                con, [(c,) for c in valid_vars], original_base_rate,
                min_sample_size=global_min_sample, min_lift=global_min_lift,
            )
            if res_1:
                valid_1way_vars = {frozenset(c["combo_vars"]) for c in res_1}
                valid_sets_by_arity[1] = valid_1way_vars
                # Singles always seed the Apriori cascade (they gate higher
                # arities). They are added to the champion pool only when
                # enable_1way is on -- legacy behavior.
                if self.enable_1way:
                    all_candidate_rules.extend(res_1)

            for k in range(2, self.max_combo_arity + 1):
                if not self._arity_enabled(k):
                    break
                prev_valid = valid_sets_by_arity.get(k - 1)
                if not prev_valid or len(valid_1way_vars) < k:
                    break
                # Anti-monotonic Apriori gate: every (k-1)-subset of vars of
                # this combo must have survived as a valid rule at arity k-1.
                candidate_combos = [
                    c for c in combinations(valid_1way_vars, k)
                    if self.is_diverse(c)
                    and all(
                        frozenset(p) in prev_valid
                        for p in combinations({v for fs in c for v in fs}, k - 1)
                    )
                ]
                if not candidate_combos:
                    break
                res_k = self._agg_combinations(
                    con, candidate_combos, original_base_rate,
                    min_sample_size=global_min_sample, min_lift=global_min_lift,
                )
                if not res_k:
                    break
                valid_sets_by_arity[k] = {frozenset(r["combo_vars"]) for r in res_k}
                all_candidate_rules.extend(res_k)

            grid_candidates: List[Dict[str, Any]] = []
            for config in experiments:
                valid_for_config = [
                    r for r in all_candidate_rules
                    if r["count"] >= config["min_sample_size"] and r["lift"] >= config["min_lift"]
                ]

                if valid_for_config:
                    valid_for_config.sort(key=lambda x: self._get_sort_key(x), reverse=True)
                    top_match = valid_for_config[0].copy()
                    top_match["grid_min_sample_size"] = config["min_sample_size"]
                    top_match["grid_min_lift"] = config["min_lift"]
                    grid_candidates.append(top_match)

            if not grid_candidates:
                logger.info("⏹️ No candidates cleared the grid. Stopping.")
                break

            grid_candidates.sort(key=lambda x: self._get_sort_key(x), reverse=True)

            selected_candidate = None
            for candidate in grid_candidates:
                rule_str = candidate["rule"]
                sql_filter = self.parse_rule_to_sql(rule_str)
                actual = con.execute(
                    f'SELECT COUNT(*) AS cnt, SUM(CAST("{self.target}" AS DOUBLE)) AS evt '
                    f'FROM current_df WHERE ({residual_filter}) AND ({sql_filter})'
                ).fetchone()
                actual_cnt, actual_evt = actual[0], actual[1] or 0
                actual_rate = (actual_evt / actual_cnt * 100.0) if actual_cnt > 0 else 0.0
                actual_lift = (actual_rate / (original_base_rate * 100.0)) if original_base_rate > 0 else 0.0

                if actual_cnt >= abs_min_sample_size and actual_evt >= abs_min_events and actual_lift >= abs_min_lift:
                    selected_candidate = {
                        **candidate,
                        "sql_filter": sql_filter,
                        "actual_count": actual_cnt,
                        "actual_events": actual_evt,
                    }
                    break
                else:
                    logger.debug(
                        f"Candidate rejected by raw validation: {rule_str} -> "
                        f"rows={actual_cnt}, events={actual_evt}"
                    )

            if selected_candidate is None:
                logger.warning(
                    f"⚠️ Iteration {i}: No candidate passed hard constraints. Stopping."
                )
                break

            best_rule = selected_candidate["rule"]
            best_raw_sql = selected_candidate["sql_filter"]
            winning_combo = selected_candidate["combo_vars"]

            actual_rate = (
                selected_candidate["actual_events"] / selected_candidate["actual_count"]
            ) * 100.0
            actual_lift = (
                actual_rate / (original_base_rate * 100.0) if original_base_rate > 0 else 0.0
            )

            for var in winning_combo:
                self.feature_usage_counts[var] = (
                    self.feature_usage_counts.get(var, 0) + 1
                )
                logger.info(
                    f"📌 Feature Usage Tracker Update -> '{var}' used count = "
                    f"{self.feature_usage_counts[var]}"
                )

            self.segments.append(
                {
                    "segment_id": i,
                    "rule_string": best_rule,
                    "sql_filter": best_raw_sql,
                    "count": int(selected_candidate["actual_count"]),
                    "rate": float(actual_rate),
                    "lift": float(actual_lift),
                    "meta_applied_sample_size": int(
                        selected_candidate["grid_min_sample_size"]
                    ),
                    "meta_applied_min_lift": float(
                        selected_candidate["grid_min_lift"]
                    ),
                }
            )

            logger.info(
                f"✅ Segment {i} Captured (Size Floor: "
                f"{selected_candidate['grid_min_sample_size']} | "
                f"Lift Floor: {selected_candidate['grid_min_lift']}): "
                f"rows={selected_candidate['actual_count']}, "
                f"events={selected_candidate['actual_events']}, "
                f"lift={actual_lift:.2f}\n"
                f"  Rule: {best_rule}\n"
                f"  SQL: {best_raw_sql}"
            )

            expanded_in_this_iter = [
                r for r in all_candidate_rules
                if "base_events" in r
            ]
            self._log_expansion_vs_champion(
                expanded_in_this_iter,
                actual_lift, actual_rate,
                selected_candidate["actual_count"],
                selected_candidate["actual_events"],
                best_rule,
            )

            self.diagnostics_[-1]["winning_segment"] = {
                "rule": best_rule,
                "sql_filter": best_raw_sql,
                "variables_used": list(winning_combo),
                "lift": actual_lift,
                "count": int(selected_candidate["actual_count"]),
            }

            # Lazy residual: accumulate the exclusion predicate instead of
            # CREATE TABLE temp_residual / DROP / RENAME each round.
            exclusion_preds.append(best_raw_sql)

        con.close()
        logger.info("🏁 Extraction complete.")
        return self.segments

    def _residual_target_array(self, con, residual_filter: str) -> np.ndarray:
        """Pulls the target column for the current residual (lazy WHERE)."""
        raw_target_arr = con.execute(
            f'SELECT "{self.target}" FROM current_df WHERE {residual_filter}'
        ).fetchnumpy()[self.target]
        if isinstance(raw_target_arr, np.ma.MaskedArray):
            return raw_target_arr.filled(0)
        return raw_target_arr

    # -------------------------------------------------------------------------
    # Public reporting methods
    # -------------------------------------------------------------------------

    def evaluate_final_coverage(self, original_data: Any) -> List[Dict[str, Any]]:
        """
        Evaluates the hierarchical segmentation on the original dataset.
        """
        if not self.segments:
            return []

        logger.info("📊 Evaluating final hierarchical coverage on original data...")
        con = duckdb.connect(":memory:")
        self._load_into_duckdb(con, original_data, "original_df")

        case_statements = [
            f"WHEN {seg['sql_filter']} THEN {seg['segment_id']}"
            for seg in self.segments
        ]
        case_sql = "\n                ".join(case_statements)

        final_query = f"""
        WITH PER_SEG_KPIS AS (
            SELECT
                CASE {case_sql} ELSE 0 END AS segment,
                COUNT(*) AS total_count,
                SUM(CAST("{self.target}" AS DOUBLE)) AS target_events,
                (SUM(CAST("{self.target}" AS DOUBLE)) * 100.0 / COUNT(*)) AS response_rate
            FROM original_df
            GROUP BY 1
        ),
        BASE_KPIS AS (
            SELECT *,
                SUM(total_count) OVER() AS total_population,
                SUM(target_events) OVER() AS total_target_events,
                (SUM(target_events) OVER() * 1.0 / SUM(total_count) OVER()) * 100 AS base_response_rate
            FROM PER_SEG_KPIS
        ),
        CUMULATIVE_KPIS AS (
            SELECT *,
                SUM(total_count) OVER (
                    ORDER BY CASE WHEN segment = 0 THEN 999999 ELSE segment END
                ) AS cum_count,
                SUM(target_events) OVER (
                    ORDER BY CASE WHEN segment = 0 THEN 999999 ELSE segment END
                ) AS cum_events
            FROM BASE_KPIS
        )
        SELECT
            segment,
            total_count,
            target_events,
            response_rate,
            base_response_rate,
            (total_count * 100.0 / total_population) AS capture_rate,
            (response_rate / NULLIF(base_response_rate, 0)) AS lift,
            (cum_count * 100.0 / NULLIF(total_population, 0)) AS cumulative_sample_capture,
            (cum_events * 100.0 / NULLIF(total_target_events, 0)) AS cumulative_event_capture
        FROM CUMULATIVE_KPIS
        ORDER BY CASE WHEN segment = 0 THEN 999999 ELSE segment END
        """

        res = con.execute(final_query)
        columns = [desc[0] for desc in res.description]
        res_list = [dict(zip(columns, row)) for row in res.fetchall()]
        con.close()
        return res_list

    def explain_feature_journey(self, feature_name: str) -> None:
        """
        Prints a detailed audit trail of a specific feature across all iterations.
        """
        if not self.diagnostics_:
            print("⚠️ No diagnostic records found. Run extract_segments() first.")
            return

        print("=" * 80)
        print(f"📌 AUDIT TRAIL FOR FEATURE: '{feature_name}'")
        print("=" * 80)

        for record in self.diagnostics_:
            iter_num = record["iteration"]
            state = record["features_state"].get(feature_name)
            winner = record["winning_segment"]

            if not state:
                print(f"Iteration {iter_num}: Variable not present or was ignored.")
                continue

            print(f"\n[Iteration {iter_num}]")
            metric_val = state.get('metric_score', state.get('iv', 0.0))
            metric_name = state.get('metric_type', 'iv').upper()

            print(f"  • Current dynamic {metric_name}   : {metric_val:.4f}")
            print(f"  • Previous times used  : {state['times_used_previously']}")
            print(f"  • Selection Status     : {state['status']}")

            if winner and feature_name in winner["variables_used"]:
                print(f"  🎉 SELECTED as part of winning rule!")
                print(f"     Rule: {winner['rule']}")
            elif winner:
                print(
                    f"  • Winner this round    : {winner['rule']} "
                    f"(Variables: {winner['variables_used']})"
                )
        print("=" * 80)

    def generate_feature_health_report(
        self, original_data: Any, features: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generates a feature health report on the original dataset using DuckDB native SQL.
        """
        if not features:
            logger.warning("⚠️ No features provided for health report generation.")
            return {}

        unique_features = list(dict.fromkeys(features))

        logger.info(
            f"📋 Generating DuckDB Naive Feature Health Report for {len(unique_features)} feature(s): "
            f"{unique_features}"
        )

        con = duckdb.connect(":memory:")
        self._load_into_duckdb(con, original_data, "input_df")

        columns_types = {
            row[0]: row[1]
            for row in con.execute("DESCRIBE input_df").fetchall()
        }

        target_expr = f"""
        (CASE
            WHEN TRY_CAST("{self.target}" AS DOUBLE) IS NOT NULL THEN TRY_CAST("{self.target}" AS DOUBLE)
            WHEN LOWER(TRIM(CAST("{self.target}" AS VARCHAR))) IN ('1', 'true', 'yes', 'y', 't') THEN 1.0
            ELSE 0.0
        END)
        """

        health_report: Dict[str, List[Dict[str, Any]]] = {}

        for col in unique_features:
            if col not in columns_types:
                logger.warning(f"⚠️ Feature '{col}' not found in dataset columns. Skipping.")
                continue

            is_numeric = self._is_numeric_duckdb_type(columns_types[col])

            if is_numeric:
                query = f"""
                WITH ranked AS (
                    SELECT
                        "{col}" AS val,
                        {target_expr} AS target_val,
                        NTILE({self.naive_bins}) OVER (ORDER BY "{col}") AS tile
                    FROM input_df
                    WHERE "{col}" IS NOT NULL
                ),
                numeric_bins AS (
                    SELECT
                        '[' || ROUND(MIN(val), 4) || ', ' || ROUND(MAX(val), 4) || ']' AS bin,
                        COUNT(*) AS total_count,
                        SUM(target_val) AS event_count,
                        (SUM(target_val) * 100.0 / COUNT(*)) AS response_rate,
                        FALSE AS is_missing,
                        MIN(val) AS sort_key
                    FROM ranked
                    GROUP BY tile
                ),
                missing_bin AS (
                    SELECT
                        'Missing' AS bin,
                        COUNT(*) AS total_count,
                        SUM({target_expr}) AS event_count,
                        (SUM({target_expr}) * 100.0 / NULLIF(COUNT(*), 0)) AS response_rate,
                        TRUE AS is_missing,
                        1e18 AS sort_key
                    FROM input_df
                    WHERE "{col}" IS NULL
                    HAVING COUNT(*) > 0
                )
                SELECT bin, total_count, event_count, response_rate, is_missing
                FROM (
                    SELECT * FROM numeric_bins
                    UNION ALL
                    SELECT * FROM missing_bin
                )
                ORDER BY sort_key
                """
            else:
                query = f"""
                SELECT
                    CASE
                        WHEN "{col}" IS NULL OR TRIM(CAST("{col}" AS VARCHAR)) IN ('', 'None', 'nan', 'NaN', '<NA>', 'null', 'NULL') THEN 'Missing'
                        ELSE '[' || CAST("{col}" AS VARCHAR) || ']'
                    END AS bin,
                    COUNT(*) AS total_count,
                    SUM({target_expr}) AS event_count,
                    (SUM({target_expr}) * 100.0 / COUNT(*)) AS response_rate,
                    CASE
                        WHEN "{col}" IS NULL OR TRIM(CAST("{col}" AS VARCHAR)) IN ('', 'None', 'nan', 'NaN', '<NA>', 'null', 'NULL') THEN TRUE
                        ELSE FALSE
                    END AS is_missing
                FROM input_df
                GROUP BY 1, 5
                ORDER BY is_missing ASC, bin ASC
                """

            res = con.execute(query).fetchall()
            col_report = [
                {
                    "bin": row[0],
                    "total_count": int(row[1]),
                    "event_count": int(row[2] or 0),
                    "response_rate": round(float(row[3] or 0.0), 4),
                    "is_missing": bool(row[4]),
                }
                for row in res
            ]
            health_report[col] = col_report

        con.close()
        rows = []
        for feature, bins in health_report.items():
            for b in bins:
                rows.append({
                    "feature": feature,
                    "bin": b["bin"],
                    "total_count": b["total_count"],
                    "event_count": b["event_count"],
                    "response_rate_%": b["response_rate"],
                    "is_missing": b["is_missing"]
                })

        health_report_df = pd.DataFrame(rows)

        return health_report_df
