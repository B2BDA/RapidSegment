"""
Strategic Segmentation Engine
==============================
Combinatorial heuristic segmentation using Optimal Binning, Apriori pruning,
and vectorized DuckDB scorecard deciling.

Author: Bishwarup Biswas
Python Version: 3.9+
"""

import logging
import os
import re
import ast
from datetime import datetime
from itertools import combinations, product, groupby
from typing import Any, Dict, List, Optional, Tuple, Union
import duckdb
import numpy as np
from joblib import Parallel, delayed
from optbinning import OptimalBinning
import pandas as pd
import uuid
import psutil
import shutil

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
        min_lift: float = 1.5,
        min_events: int = 100,
        top_n_vars: int = 15,
        max_segments: int = 10,
        max_feature_reuse: int = 1,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        enable_diversity: bool = False,
        enable_1way: bool = True,
        enable_2way: bool = True,
        enable_3way: bool = True,
        feature_groups: Optional[Dict[str, List[str]]] = None,
        ignore_features: Optional[List[str]] = None,
        sort_priority: str = "rate_lift_count",
        binning_method: str = "optimal_cart",  # "naive" | "optimal_cart" | "optimal_quantile" | "optimal" (alias of cart)        
        naive_bins: int = 5 ,
        max_expansion_hops: int = 0,
        selection_metric: str = "iv",
        expand_log_mode: str = "none",
        memory_limit_gb: Optional[float] = None,
        engine_threads: Optional[int] = None,
        db_path: Optional[str] = None,
        db_temp_dir: Optional[str] = None,
        persist_db: bool = False,
    ) -> None:
        """
        Args:
            target: Name of the binary target column (1 = Event, 0 = Non-Event).
            n_jobs: Number of parallel jobs for IV computation. -1 uses all but one core.
            min_sample_size: Absolute minimum row count for a valid rule. Used as a fallback when param_grid is None.
            min_lift: Absolute minimum lift threshold (hard constraint).
            min_events: Minimum number of positive events for a valid rule.
            top_n_vars: Number of highest‑IV features passed into the Apriori engine.
            max_segments: Maximum number of segments to extract.
            max_feature_reuse: Max times a feature can appear across segments.
            param_grid: Optional grid of {min_sample_size, min_lift} to sweep.
            enable_diversity: If True, blocks rules combining variables from same group.
            enable_1way: Allow 1‑dimensional rules.
            enable_2way: Allow 2‑dimensional intersection rules.
            enable_3way: Allow 3‑dimensional intersection rules.
            feature_groups: Mapping of business categories to columns (e.g. {'risk': ['scr', 'bal']}).
            ignore_features: Explicit list of columns to drop prior to IV calculation.
            sort_priority: Ranking criteria for selecting champion segments.
            binning_method: Binning engine:
                - "naive": DuckDB equal-frequency quantiles (supports adjacent-bin expansion).
                - "optimal_cart": OptBinning with CART prebinning (target-aware cuts).
                - "optimal_quantile": OptBinning with quantile prebinning (more stable).
                - "optimal": alias for "optimal_cart" (backward compatible).
            naive_bins: Number of quantile bins when binning_method is "naive".
            selection_metric: Metric used to rank features for top_n_vars selection ("iv" or "response_rate").
            max_expansion_hops: Adjacent-bin merging hop distance limit (0 disables expansion).
            expand_log_mode: Controls verbosity of adjacent-bin expansion logging ("none", "summary", "champion", "full").
                - "none": No expansion logging.
                - "summary": Show summary statistics of adjacent-bin expansions.
                - "champion": Display champion segment with contenders in a formatted table.
                - "full": Display summary plus detailed top expanded rules.
            memory_limit_gb: Hard cap on DuckDB's RAM buffer pool (GB). None => default
                utilises the host: int(80% of total RAM). On a 32 GB machine this is
                ~25 GB. Pass a lower value (e.g. 4) on memory-constrained hardware.
            engine_threads: Max DuckDB execution threads. None => default utilises the
                host: all-but-two cores (or all cores if <=4). Pass a lower value on
                CPU-constrained hardware.
        """
        self.target = target
        self.memory_limit_gb = memory_limit_gb
        self.engine_threads = engine_threads
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
        # Reverse map for O(1) group lookups; rebuilt by _validate_feature_groups
        self._feature_to_group: Dict[str, str] = {
            var: group
            for group, vars_list in self.feature_groups.items()
            for var in vars_list
        }
        self.sort_priority = sort_priority
        self.diagnostics_: List[Dict[str, Any]] = []
        self.stop_reason: Optional[str] = None
        self.binning_method = binning_method
        # Normalize aliases
        _bm = (binning_method or "optimal_cart").lower().strip()
        if _bm == "optimal":
            _bm = "optimal_cart"
        if _bm not in ("naive", "optimal_cart", "optimal_quantile"):
            raise ValueError(
                f"binning_method must be one of 'naive', 'optimal_cart', "
                f"'optimal_quantile', or 'optimal' (alias of cart); got {_bm!r}"
            )
        self.binning_method = _bm
        self.naive_bins = naive_bins     
        self.max_expansion_hops = max(0, int(max_expansion_hops))
        self.selection_metric = selection_metric
        self.expand_log_mode = expand_log_mode if expand_log_mode in ("none", "summary", "champion", "full") else "summary"   
        self._columns_types: Dict[str,str] = {}
        self._categorical_cols: set[str] = set()
        # Set up disk-backed DuckDB automatically if not provided
        self.db_path = db_path
        self.db_temp_dir = db_temp_dir
        # When True the auto-created DuckDB file + temp dir are kept alive across
        # extract -> evaluate -> health calls (single shared artifact) and must be
        # released via close() / context manager. When False (default) the legacy
        # behaviour of deleting the temp DB in finally is preserved.
        self.persist_db = persist_db

    # -------------------------------------------------------------------------
    # Lifecycle helpers for the opt-in persistent connection. When `persist_db`
    # is True the auto-created DuckDB file + temp dir are reused across method
    # calls; call `close()` (or use the context manager) to release them.
    # -------------------------------------------------------------------------
    def close(self) -> None:
        """Releases the auto-created persistent DuckDB artifact, if any."""
        if self.db_path and os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception as e:
                logger.debug(f"Cleanup failed for {self.db_path}: {e}")
        if self.db_temp_dir and os.path.exists(self.db_temp_dir):
            try:
                shutil.rmtree(self.db_temp_dir)
            except Exception as e:
                logger.debug(f"Cleanup failed for {self.db_temp_dir}: {e}")
        self.db_path = None
        self.db_temp_dir = None

    def __enter__(self) -> "StrategicSegmentBuilder":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @staticmethod
    def _resolve_optb_dtype(duckdb_type: str) -> str:
        """
        Determines the correct OptBinning data type flag from a DuckDB type string.
        """
        dtype_upper = duckdb_type.upper()
        if any(t in dtype_upper for t in ["VARCHAR", "CHAR", "STRING", "TEXT", "UUID"]):
            return "categorical"
        return "numerical"

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
        # Rebuild reverse map after validation passes
        self._feature_to_group = {
            var: group
            for group, vars_list in self.feature_groups.items()
            for var in vars_list
        }
        logger.info(f"✅ Feature group validation passed. ({validated_count} features mapped)")

    def get_group(self, var: str) -> str:
        """
        Returns the assigned business category for a feature, or the feature name itself.
        """
        return self._feature_to_group.get(var,var)

    def is_diverse(self, combo: Tuple[str, ...]) -> bool:
        """
        Ensures a tuple of features spans strictly distinct analytical groups.
        """
        if not self.enable_diversity:
            return True
        groups = [self.get_group(v) for v in combo]
        return len(groups) == len(set(groups))

    def _get_sort_key(self, rule: Dict[str, Any]) -> Tuple[float, ...]:
        """
        Create a sortable tuple for a candidate rule based on the configured priority.

        The returned tuple keeps the ordering deterministic across the many ranking
        strategies supported by the builder. Each branch mirrors a different user-facing
        priority such as lift-first, count-first, or event-first selection.
        """
        priority = self.sort_priority
        if priority == "lift_count_rate":
            key = (rule["lift"], rule["count"], rule["rate"])
        elif priority == "count_lift_rate":
            key = (rule["count"], rule["lift"], rule["rate"])
        elif priority == "rate_lift_count":
            key = (rule["rate"], rule["lift"], rule["count"])
        elif priority == "lift_rate_count":
            key = (rule["lift"], rule["rate"], rule["count"])
        elif priority == "count_rate_lift":
            key = (rule["count"], rule["rate"], rule["lift"])
        elif priority == "rate_count_lift":
            key = (rule["rate"], rule["count"], rule["lift"])
        elif priority == "events_lift_rate":
            key = (rule["events"], rule["lift"], rule["rate"])
        elif priority == "events_rate_lift":
            key = (rule["events"], rule["rate"], rule["lift"])
        elif priority == "lift_events_rate":
            key = (rule["lift"], rule["events"], rule["rate"])
        elif priority == "rate_events_lift":
            key = (rule["rate"], rule["events"], rule["lift"])
        elif priority == "events_count_rate":
            key = (rule["events"], rule["count"], rule["rate"])
        elif priority == "events_rate_count":
            key = (rule["events"], rule["rate"], rule["count"])
        elif priority == "count_events_rate":
            key = (rule["count"], rule["events"], rule["rate"])
        elif priority == "rate_events_count":
            key = (rule["rate"], rule["events"], rule["count"])
        else:
            key = (rule["lift"], rule["rate"], rule["count"])
        # Deterministic tie-breaker: when every configured priority dimension
        # is exactly equal between two candidates, the winner must not depend
        # on incoming list order (which itself can vary run-to-run due to
        # Python's per-process string hash randomization affecting set/dict
        # iteration order, and unordered SQL GROUP BY results). Sorting on
        # the rule string last makes ties resolve identically every run.
        return key + (rule.get("rule", ""),)

    def compute_iv_ranking_and_bin(
        self,
        con: duckdb.DuckDBPyConnection,
        eligible_cols: List[str],
        columns_types: Dict[str, str],
    ) -> Tuple[List[Dict[str, Union[str, float]]], Dict[str, np.ndarray]]:
        """
        Computes Information Value (IV) and pre-computed bins using SQL-native DuckDB execution.

        Memory note: the per-feature bin-label arrays (length = residual rows) are only
        materialised for the ``top_n_vars`` features that are actually used downstream.
        Every eligible feature is still *fit* (so the IV ranking is complete), but the
        full-length numpy pulls are bounded to ``top_n_vars`` instead of all columns.
        """
        logger.info(f"🔍 Computing IV and bins for {len(eligible_cols)} features...")

        # ------------------------------------------------------------------
        # Phase 1 — fit + rank. Produces the IV/response-rate ranking and keeps
        # only the lightweight objects needed to transform later (bin edges or the
        # fitted OptBinning model). No full-length bin arrays are retained here.
        # ------------------------------------------------------------------
        def _fit_worker(col: str) -> Tuple[str, float, float, Optional[Dict[str, Any]]]:
            thread_con = con.cursor()
            try:
                dtype = self._resolve_optb_dtype(columns_types[col])

                if self.binning_method == "naive":
                    if dtype == "numerical":
                        # Compute quantiles natively in DuckDB SQL
                        q_step = 1.0 / float(self.naive_bins)
                        q_list = [round(i * q_step, 6) for i in range(self.naive_bins + 1)]
                        q_str = ", ".join(str(q) for q in q_list)
                        
                        quantiles = thread_con.execute(
                            f'SELECT QUANTILE_CONT("{col}", [{q_str}]) FROM current_df WHERE "{col}" IS NOT NULL'
                        ).fetchone()[0]

                        if quantiles is None or len(quantiles) == 0:
                            edges = np.array([-np.inf, np.inf])
                        else:
                            edges = np.unique(np.array(quantiles, dtype=float))
                            if len(edges) < 2:
                                edges = np.array([-np.inf, np.inf])
                            else:
                                edges[0] = -np.inf
                                edges[-1] = np.inf

                        case_clauses = []
                        for i in range(len(edges) - 1):
                            lower, upper = edges[i], edges[i + 1]
                            lower_is_ninf = np.isinf(lower) and lower < 0
                            upper_is_pinf = np.isinf(upper) and upper > 0

                            # Human-readable label (strings are fine)
                            lower_str = "-inf" if lower_is_ninf else str(lower)
                            upper_str = "inf" if upper_is_pinf else str(upper)
                            label = f"[{lower_str}, {upper_str})"

                            if lower_is_ninf and upper_is_pinf:
                                # Whole domain → every non-null value belongs here
                                case_clauses.append(f'WHEN "{col}" IS NOT NULL THEN \'{label}\'')
                            elif lower_is_ninf:
                                case_clauses.append(f'WHEN "{col}" < {upper} THEN \'{label}\'')
                            elif upper_is_pinf:
                                case_clauses.append(f'WHEN "{col}" >= {lower} THEN \'{label}\'')
                            else:
                                case_clauses.append(
                                    f'WHEN "{col}" >= {lower} AND "{col}" < {upper} THEN \'{label}\''
                                )

                        case_expr = f"""
                        CASE 
                            WHEN "{col}" IS NULL THEN 'Missing'
                            {' '.join(case_clauses)}
                            ELSE 'Missing'
                        END
                        """
                    else:
                        case_expr = f"""
                        CASE 
                            WHEN "{col}" IS NULL OR TRIM(CAST("{col}" AS VARCHAR)) IN ('', 'None', 'nan', 'NaN', '<NA>', 'null', 'NULL') THEN 'Missing'
                            ELSE '[' || CAST("{col}" AS VARCHAR) || ']'
                        END
                        """

                    # Aggregate binned metrics in DuckDB
                    stats_df = thread_con.execute(
                        f"""
                        WITH binned AS (
                            SELECT 
                                {case_expr} AS bin_label,
                                CAST("{self.target}" AS DOUBLE) AS target
                            FROM current_df
                        )
                        SELECT 
                            bin_label,
                            COUNT(*) AS cnt,
                            SUM(target) AS evt
                        FROM binned
                        GROUP BY bin_label
                        """
                    ).fetchall()

                    # Information Value is a weighted log-ratio between event and non-event
                    # distributions across the bins. These totals give the denominator for that
                    # calculation and allow us to compare each binned segment against the global base.
                    total_events = sum(r[2] or 0.0 for r in stats_df)
                    total_non_events = sum((r[1] - (r[2] or 0.0)) for r in stats_df)

                    iv_val = 0.0
                    max_rr = 0.0

                    for label, cnt, evt in stats_df:
                        evt = evt or 0.0
                        cnt = cnt or 0
                        non_evt = cnt - evt

                        if cnt >= self.min_sample_size and evt >= self.min_events:
                            rr = evt / cnt
                            if rr > max_rr:
                                max_rr = rr

                        if total_events > 0 and total_non_events > 0:
                            pct_events = max(evt / total_events, 1e-6)
                            pct_non_events = max(non_evt / total_non_events, 1e-6)
                            iv_val += (pct_non_events - pct_events) * np.log(pct_non_events / pct_events)

                    # Keep only the case expression so Phase 2 can materialise the
                    # bin-label array lazily (and only for the top features).
                    return col, float(iv_val), float(max_rr), {
                        "method": "naive",
                        "case_expr": case_expr,
                    }

                else:
                    # Optimal binning fallback
                    data_dict = thread_con.execute(
                        f'''
                        SELECT
                            "__rs_row_id",
                            "{col}",
                            "{self.target}"
                        FROM current_df
                        ORDER BY "__rs_row_id"
                        '''
                    ).fetchnumpy()

                    col_arr_raw = data_dict[col]
                    target_arr_raw = data_dict[self.target]

                    col_arr = col_arr_raw.filled(np.nan if np.issubdtype(col_arr_raw.dtype, np.number) else None) if isinstance(col_arr_raw, np.ma.MaskedArray) else col_arr_raw
                    target_arr = target_arr_raw.filled(0) if isinstance(target_arr_raw, np.ma.MaskedArray) else target_arr_raw

                    # Deterministic row order before CART prebinning.
                    # DuckDB SELECT has no ORDER BY; OptBinning's default prebinning
                    # (CART) can split differently when tied samples arrive in a
                    # different order, which changes bins/IV and cascades into
                    # different segments across runs.
                    #
                    # IMPORTANT: only the *fitting* input is sorted. The transform
                    # below must run on the ORIGINAL (unsorted) arrays so the
                    # returned bin labels stay aligned with "__rs_row_id" / the
                    # row order of current_df. Otherwise binned_df pairs the wrong
                    # target values with the wrong bin labels, corrupting both the
                    # event counts of 1-way rules and the row alignment of 2/3-way
                    # rules (which is what produced spurious candidates whose SQL
                    # re-validation then failed against the real table).
                    try:
                        col_arr = np.asarray(col_arr)
                        target_arr = np.asarray(target_arr)
                        if np.issubdtype(col_arr.dtype, np.number):
                            order = np.lexsort((
                                target_arr,
                                np.nan_to_num(col_arr.astype(float, copy=False), nan=np.inf),
                            ))
                        else:
                            col_as_str = col_arr.astype(str)
                            order = np.lexsort((target_arr, col_as_str))
                        col_fit = col_arr[order]
                        target_fit = target_arr[order]
                    except Exception:
                        col_fit = col_arr
                        target_fit = target_arr

                    # Fit an Optimal Binning model to the current feature/target pair so we can
                    # create monotonic, high-signal bins without relying on a fixed quantile grid.
                    prebinning_method = (
                        "quantile" if self.binning_method == "optimal_quantile" else "cart"
                    )
                    optb = OptimalBinning(
                        name=col,
                        dtype=dtype,
                        prebinning_method=prebinning_method,
                        random_state=42,
                    )
                    optb.fit(col_fit, target_fit)

                    bin_table = optb.binning_table.build()
                    iv_val = float(bin_table["IV"].values[-1])

                    valid_bins = bin_table[
                        (bin_table["Count"] >= self.min_sample_size) & 
                        (bin_table["Event"] >= self.min_events)
                    ]
                    max_rr = float(valid_bins["Event rate"].max()) if not valid_bins.empty else 0.0

                    # Keep only the fitted model + dtype so Phase 2 can transform
                    # lazily (and only for the top features).
                    return col, float(iv_val), float(max_rr), {
                        "method": "optimal",
                        "dtype": dtype,
                        "optb": optb,
                    }

            except Exception as e:
                logger.debug(f"Computation failed for {col}: {e}")
                return col, 0.0, 0.0, None
            finally:
                try:
                    thread_con.close()
                except Exception:
                    pass

        fit_results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_fit_worker)(col) for col in eligible_cols
        )

        ranking = []
        fit_cache: Dict[str, Dict[str, Any]] = {}
        for col, iv, rr, info in fit_results:
            ranking.append({"variable": col, "iv": iv, "max_rr": rr})
            if info is not None:
                fit_cache[col] = info
        # Stable sort: primary metric descending, variable name ascending on ties
        # so top_n_vars is identical across runs when scores collide.
        if self.selection_metric == "response_rate":
            ranking.sort(key=lambda x: (-x["max_rr"], x["variable"]))
        else:
            ranking.sort(key=lambda x: (-x["iv"], x["variable"]))

        # ------------------------------------------------------------------
        # Phase 2 — transform. Materialise the bin-label arrays for *only* the
        # top_n_vars features that downstream candidate generation consumes. This
        # bounds peak memory to top_n_vars full-length arrays instead of one per
        # eligible feature.
        # ------------------------------------------------------------------
        top_n_variable_names = [r["variable"] for r in ranking[: self.top_n_vars]]
        precomputed_bins: Dict[str, np.ndarray] = {}

        def _transform_worker(col: str) -> Tuple[str, Optional[np.ndarray]]:
            info = fit_cache[col]
            thread_con = con.cursor()
            try:
                if info["method"] == "naive":
                    transformed_bins = thread_con.execute(
                        f"""
                        SELECT {info['case_expr']} AS bin_label
                        FROM current_df
                        ORDER BY "__rs_row_id"
                        """
                    ).fetchnumpy()["bin_label"].astype(str)
                    return col, transformed_bins

                # Optimal path: re-pull the (original-order) column and transform
                # with the already-fitted model so labels stay aligned to __rs_row_id.
                data_dict = thread_con.execute(
                    f'''
                    SELECT
                        "__rs_row_id",
                        "{col}",
                        "{self.target}"
                    FROM current_df
                    ORDER BY "__rs_row_id"
                    '''
                ).fetchnumpy()

                col_arr_raw = data_dict[col]
                target_arr_raw = data_dict[self.target]
                col_arr = col_arr_raw.filled(np.nan if np.issubdtype(col_arr_raw.dtype, np.number) else None) if isinstance(col_arr_raw, np.ma.MaskedArray) else col_arr_raw
                target_arr = target_arr_raw.filled(0) if isinstance(target_arr_raw, np.ma.MaskedArray) else target_arr_raw
                col_arr = np.asarray(col_arr)
                target_arr = np.asarray(target_arr)
                optb = info["optb"]

                if info["dtype"] == "categorical":
                    bin_table = optb.binning_table.build()
                    raw_cells = bin_table["Bin"].tolist()
                    clean_labels = {
                        i: self._sanitize_bin_label(cell)
                        for i, cell in enumerate(raw_cells)
                    }
                    idx = optb.transform(col_arr, metric="indices")
                    try:
                        idx_arr = np.asarray(idx)
                    except Exception:
                        idx_arr = np.array(list(idx))

                    idx_int = []
                    for v in idx_arr:
                        try:
                            if hasattr(v, "item"):
                                v_val = v.item()
                            else:
                                v_val = v
                            idx_int.append(int(v_val))
                        except Exception:
                            idx_int.append(-1)

                    transformed_bins = np.array(
                        [clean_labels.get(i, "Missing") for i in idx_int], dtype=str
                    )
                    logger.debug(f"{col}: optimal categorical bins unique={np.unique(transformed_bins).tolist()[:20]}")
                else:
                    transformed_bins = np.asarray(
                        optb.transform(col_arr, metric="bins"), dtype=str
                    )
                return col, transformed_bins

            except Exception as e:
                logger.debug(f"Transform failed for {col}: {e}")
                return col, None
            finally:
                try:
                    thread_con.close()
                except Exception:
                    pass

        if top_n_variable_names:
            transform_results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(_transform_worker)(col) for col in top_n_variable_names
            )
            for col, bins in transform_results:
                if bins is not None:
                    precomputed_bins[col] = bins

        return ranking, precomputed_bins

    @staticmethod
    def _bin_sort_key(label: str) -> Tuple[int, float, str]:
        """
        Numeric-aware sort key for a bin label.
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

    def _expand_adjacent_bins(
        self,
        con: duckdb.DuckDBPyConnection,
        combo: Tuple[str, ...],
        base_rate: float,
        base_results: List[Dict[str, Any]],
        seen_rules: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        """
        Attempts to merge adjacent neighbour bin(s) on every variable in combo.
        Uses native Python groupby processing (no Pandas).
        """
        if not base_results:
            return []

        if seen_rules is None:
            seen_rules = set()
        for r in base_results:
            seen_rules.add(r["rule"])

        base_lookup: Dict[str, Dict[str, Any]] = {r["rule"]: r for r in base_results}
        max_hops = max(0, int(getattr(self, "max_expansion_hops", 0)))

        expanded: List[Dict[str, Any]] = []

        for col in combo:
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
                continue

            if not rows:
                continue

            columns = other_cols + [col, "cnt", "evt"]
            
            processed = []
            for r in rows:
                row_dict = dict(zip(columns, r))
                if row_dict[col] == "Missing":
                    continue
                row_dict["_sort_key"] = self._bin_sort_key(row_dict[col])
                processed.append(row_dict)

            if not processed:
                continue

            if other_cols:
                processed.sort(key=lambda x: tuple(x[c] for c in other_cols) + (x["_sort_key"],))
                group_iter = []
                for k, g in groupby(processed, key=lambda x: tuple(x[c] for c in other_cols)):
                    group_iter.append((k, list(g)))
            else:
                processed.sort(key=lambda x: x["_sort_key"])
                group_iter = [((), processed)]

            for group_key, g_list in group_iter:
                if other_cols:
                    key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
                    other_val = dict(zip(other_cols, key_tuple))
                else:
                    other_val = {}

                n = len(g_list)
                if n < 2:
                    continue

                # Grouped rows are already sorted by the current feature's bin order. We can
                # evaluate sliding windows over the cumulative counts/events to generate merged
                # candidates without re-running a query for every possible interval.
                bins = [x[col] for x in g_list]
                cnt = np.array([x["cnt"] for x in g_list], dtype=float)
                evt = np.array([x["evt"] for x in g_list], dtype=float)
                cum_cnt = np.concatenate(([0.0], np.cumsum(cnt)))
                cum_evt = np.concatenate(([0.0], np.cumsum(evt)))

                for idx in range(n):
                    rule_at_idx = " & ".join(
                        f"{c}={other_val[c]}" if c != col else f"{c}={bins[idx]}"
                        for c in combo
                    )
                    base_result = base_lookup.get(rule_at_idx)
                    if base_result is None:
                        continue

                    base_events = base_result["events"]

                    for lo, hi in self._candidate_windows(idx, n, max_hops):
                        exp_count = cum_cnt[hi + 1] - cum_cnt[lo]
                        exp_events = cum_evt[hi + 1] - cum_evt[lo]
                        if exp_count <= 0:
                            continue

                        exp_rate = (exp_events / exp_count) * 100.0
                        exp_lift = exp_rate / (base_rate * 100.0) if base_rate > 0 else 0.0

                        if not (
                            exp_lift >= self.min_lift
                            and exp_events >= self.min_events
                            and exp_count >= self.min_sample_size
                            and exp_events > base_events
                        ):
                            continue

                        window_labels_sorted = sorted(bins[lo:hi + 1], key=self._bin_sort_key)
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
    
        Windows that would span the variable's entire domain (all n bins) are excluded:
        such a merge removes the variable's filtering power entirely (matches every row),
        producing a degenerate, non-informative candidate. Left unguarded, expansions on
        different variables can each independently collapse to this "no constraint" state
        and surface as distinct rule strings with identical count/events/lift downstream.
        """
        for hops in range(1, max_hops + 1):
            lo, hi = idx - hops, idx
            if lo < 0:
                break
            if hi - lo + 1 >= n:
                break
            yield (lo, hi)
        for hops in range(1, max_hops + 1):
            lo, hi = idx, idx + hops
            if hi >= n:
                break
            if hi - lo + 1 >= n:
                break
            yield (lo, hi)
    @staticmethod
    def _sanitize_bin_label(label: Any) -> str:
        """
        Defends against an OptBinning + pandas>=3.0 interaction where a categorical
        bin's label comes back as a raw array-like (e.g. pandas ArrowStringArray)
        instead of a joined string. Rebuilds the "[cat1, cat2]" format explicitly.
        """
        if isinstance(label, str):
            return label
        try:
            return "[" + ", ".join(str(i) for i in list(label)) + "]"
        except TypeError:
            return str(label)

    def _agg_combinations(
        self,
        con: duckdb.DuckDBPyConnection,
        combo_list: List[Tuple[str, ...]],
        base_rate: float,
    ) -> List[Dict[str, Any]]:
        """
        Aggregate candidate rules for one or more feature combinations.

        The method builds a SQL GROUP BY query for each combination, executes them in
        chunks, and turns the raw aggregate rows into rule dictionaries that later feed
        the ranking and validation stages. For the naive path, the resulting base rules
        can also be expanded into adjacent-bin merges.
        """
        if not combo_list:
            return []

        # Build one GROUP BY query per combination. Chunking keeps the DuckDB work bounded
        # while still letting us evaluate hundreds of candidate rules efficiently.
        queries = []
        for combo in combo_list:
            cols_str = ", ".join([f'"{c}"' for c in combo])
            rule_concat = " || ' & ' || ".join(
                [f"'{c}=' || CAST(\"{c}\" AS VARCHAR)" for c in combo]
            )
            combo_str = ",".join(combo)

            query = f"""
                    SELECT
                    {rule_concat} AS rule,
                    COUNT("{self.target}")::BIGINT AS count,
                    SUM(CAST("{self.target}" AS DOUBLE)) AS events,
                    '{combo_str}' AS combo_vars_str
                    FROM binned_df
                    GROUP BY {cols_str}
                    HAVING COUNT("{self.target}") >= {self.min_sample_size}
                    AND SUM(CAST("{self.target}" AS DOUBLE)) >= {self.min_events}
            """
            queries.append(query)
            
        def _process_rows(rows):
            # Convert the SQL aggregate rows into the internal rule dictionary format used
            # by the rest of the extraction pipeline.
            for rule, count, events, combo_vars_str in rows:
                rate = (events / count) * 100.0 if count > 0 else 0
                lift = rate/ (base_rate * 100.0) if base_rate > 0 else 0
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

        valid_results = []
        chunk_size = 100

        per_combo_base: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {
            combo: [] for combo in combo_list
        }

        for i in range(0, len(queries), chunk_size):
            chunk = queries[i:i + chunk_size]
            chunk_combos = combo_list[i:i + chunk_size]
            union_query = " UNION ALL ".join(chunk)
            try:
                res = con.execute(union_query).fetchall()
                _process_rows(res)
            except Exception as batch_err:
                logger.debug(
                f"Batch query failed ({len(chunk)} combos), retrying individually: {batch_err}"
                )
                for single_query, single_combo in zip(chunk, chunk_combos):
                    try:
                        res = con.execute(single_query).fetchall()
                        _process_rows(res)
                    except Exception as e:
                        logger.debug(f"Single combo query failed for {single_combo}: {e}")

        all_expanded: List[Dict[str, Any]] = []
        expansion_stats: Dict[str, Dict[str, Any]] = {}
        shared_seen_rules: set = set()

        for combo in combo_list:
            base = per_combo_base.get(combo, [])
            if not base:
                continue
            if getattr(self, "binning_method", "naive") == "naive":
                expanded = self._expand_adjacent_bins(
                    con, combo, base_rate, base, seen_rules=shared_seen_rules
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

        mode = getattr(self, "expand_log_mode", "summary")
        if expansion_stats and mode in ("summary", "full"):
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
                        f"     {i}. {e['rule']}"
                        f"  | events {e['events']:.0f} (Δ{delta:+.0f}) | lift {e['lift']:.2f}x"
                    )

        return valid_results

    def parse_rule_to_sql(self, rule_str: str) -> str:
        """
        Translate a human-readable rule string into a SQL WHERE predicate.

        Rules are stored in a compact bracketed form (for example, ``income=[10000, 20000)``)
        and must be converted back into SQL that can be evaluated against the residual table.
        This method handles numeric ranges, categorical sets, and missing-value tokens.
        """
        parts = [p.strip() for p in rule_str.split("&")]
        sql_conditions: List[str] = []

        def _quote_sql_ident(ident: str) -> str:
            # Quote every emitted column identifier so DuckDB reserved words
            # (e.g. "default") and exotic column names do not break the predicate.
            return '"' + str(ident).replace('"', '""') + '"'

        def _quote_sql_string(val: Any) -> str:
            txt = str(val)
            # escape single quotes for SQL and wrap in single quotes
            return "'" + txt.replace("'", "''") + "'"

        def _strip_wrapping_quotes(s: str) -> str:
            # Only strip a matching outer quote pair; never strip interior/unbalanced quotes
            # (Bug 4: str.strip("'\"") mangles values like Say "hi")
            s = s.strip()
            if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
                return s[1:-1]
            return s
        
        def _is_categorical_col(col_name: str) -> bool:
            return col_name in self._categorical_cols
        
        # Each rule is composed of one or more ``column=interval`` parts joined by ``&``.
        # We convert each part independently and later combine them with logical AND.
        for part in parts:
            if "=" not in part:
                continue
        
            col, interval = [x.strip() for x in part.split("=", 1)]
            bracket_match = _BRACKET_REGEX.search(interval)
            col_is_categorical = _is_categorical_col(col)
        
            # 1. Multi-range / merged adjacent NUMERIC ranges like [[10, 20), [20, 30)]
            # Real numeric range tokens always end in ')' or ']', so adjacent tokens are
            # joined by "),"/"]," + "[". Categorical merges (see 1b) join bracketed single
            # values like "[female],[male]" -- joined by "],", never "),". Require the
            # numeric-specific separator so we don't misfire on merged categoricals
            is_numeric_multirange = (
                not col_is_categorical
                and interval.startswith("[[")
                and re.search(r"\),\s*\[", interval) is not None
            )
            if is_numeric_multirange:
                inner = interval.strip()
                if inner.startswith("[") and inner.endswith("]"):
                    inner = inner[1:-1]
        
                raw_tokens = re.split(r"\),\s*", inner)
                ranges = []
                for tok in raw_tokens:
                    tok = tok.strip()
                    if not tok:
                        continue
                    if not tok.endswith(")"):
                        tok = tok + ")"
                    if tok[0] in ("[", "(") and tok[-1] in ("]", ")"):
                        left_char = tok[0]
                        right_char = tok[-1]
                        content = tok[1:-1]
                        if "," in content:
                            lo, hi = [x.strip() for x in content.split(",", 1)]
                            ranges.append((left_char, lo, hi, right_char))
        
                if ranges:
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
                        op = ">=" if overall_lower_char == "[" else ">"
                        range_conds.append(f"{_quote_sql_ident(col)} {op} {overall_lower}")
                    if overall_upper.lower() != "inf":
                        op = "<=" if overall_upper_char == "]" else "<"
                        range_conds.append(f"{_quote_sql_ident(col)} {op} {overall_upper}")
        
                    if range_conds:
                        sql_conditions.append(" AND ".join(range_conds))
                    continue
        
            # 1b. Merged categorical lists like [[male], [female]] or [[male],[female], [others]],
            # produced when _expand_adjacent_bins merges adjacent single-bracket categorical bins.
            # Guard: a single-category value that itself contains brackets (e.g. category "[VIP]"
            # becomes the label "[[VIP]]") must NOT be treated as a multi-value merge
            # (Bug 3 partial mitigation — full fix needs escaped rule grammar).
            if interval.startswith("[[") and bracket_match:
                cat_tokens = re.findall(r"\[([^\[\]]+)\]", interval)
                # If the outer form is exactly one nested pair with no sibling tokens
                # (e.g. "[[VIP]]"), treat as a single literal category value instead.
                is_single_nested = (
                    len(cat_tokens) == 1
                    and re.fullmatch(r"\[\[[^\[\]]+\]\]", interval.strip()) is not None
                )
                if is_single_nested:
                    cleaned = _strip_wrapping_quotes(interval[1:-1])
                    sql_conditions.append(f"{_quote_sql_ident(col)} = {_quote_sql_string(cleaned)}")
                    continue
                if cat_tokens:
                    cleaned_items = [_strip_wrapping_quotes(t) for t in cat_tokens]
                    formatted_items = ", ".join(_quote_sql_string(item) for item in cleaned_items if item)
                    if formatted_items:
                        sql_conditions.append(f"{_quote_sql_ident(col)} IN ({formatted_items})")
                    continue
        
            # 2. Explicit Categoricals
            def _is_numeric_token(tok: str) -> bool:
                tok = tok.strip().strip("'\"")
                if tok.lower() in ("-inf", "inf", "+inf"):
                    return True
                try:
                    float(tok)
                    return True
                except ValueError:
                    return False
        
            is_categorical = False
            if bracket_match:
                content = bracket_match.group(1)
                tokens = [t for t in content.split(",") if t.strip()]
                if any(k in interval for k in ("'", '"', "Array", "Categorical")) or not interval.startswith(("[", "(")):
                    is_categorical = True
                elif len(tokens) > 2:
                    is_categorical = True
                elif not all(_is_numeric_token(t) for t in tokens):
                    is_categorical = True
        
            if is_categorical and bracket_match:
                try:
                    raw_items = ast.literal_eval(bracket_match.group(0))
                    if not isinstance(raw_items, (list, tuple)):
                        raw_items = [raw_items]
                except Exception:
                    raw_content = bracket_match.group(1)
                    if "," not in raw_content:
                        raw_content = re.sub(r"'\s+'", "','", raw_content)
                        raw_content = re.sub(r'"\s+"', '","', raw_content)
                        raw_content = re.sub(r"\s+", ",", raw_content)
                    raw_items = [
                        _strip_wrapping_quotes(i)
                        for i in raw_content.split(",")
                        if i.strip()
                    ]
                formatted_items = ", ".join(
                    [
                        _quote_sql_string(item)
                        if (col_is_categorical or isinstance(item, str))
                        else str(item)
                        for item in raw_items
                    ]
                )
                if formatted_items:
                    sql_conditions.append(f"{_quote_sql_ident(col)} IN ({formatted_items})")
                continue

            # 3. Special or Missing values
            if interval in ["Special", "Missing"]:
                sql_conditions.append(f"{_quote_sql_ident(col)} IS NULL")
                continue
        
            # 4. Standard Intervals [lo, hi) or Single Bracket Values [val]
            if interval.startswith(("[", "(")):
                inner = interval[1:-1].strip()
        
                if "," in inner:
                    if col_is_categorical:
                        raw_items = [_strip_wrapping_quotes(x) for x in inner.split(",") if x.strip()]
                        formatted_items = ", ".join(_quote_sql_string(item) for item in raw_items)
                        if formatted_items:
                            sql_conditions.append(f"{_quote_sql_ident(col)} IN ({formatted_items})")
                        continue
                    left_char, right_char = interval[0], interval[-1]
                    lower_str, upper_str = [x.strip() for x in inner.split(",", 1)]
        
                    range_conds = []
                    if lower_str.lower() != "-inf":
                        op = ">=" if left_char == "[" else ">"
                        range_conds.append(f"{_quote_sql_ident(col)} {op} {lower_str}")
                    if upper_str.lower() != "inf":
                        op = "<=" if right_char == "]" else "<"
                        range_conds.append(f"{_quote_sql_ident(col)} {op} {upper_str}")
        
                    if range_conds:
                        sql_conditions.append(" AND ".join(range_conds))
                else:
                    # Single value inside brackets, e.g. [123] or [Value]
                    clean_val = _strip_wrapping_quotes(inner)
        
                    if col_is_categorical:
                        sql_conditions.append(f"{_quote_sql_ident(col)} = {_quote_sql_string(clean_val)}")
                    elif clean_val.replace(".", "", 1).replace("-", "", 1).isdigit():
                        sql_conditions.append(f"{_quote_sql_ident(col)} = {clean_val}")
                    else:
                        sql_conditions.append(f"{_quote_sql_ident(col)} = {_quote_sql_string(clean_val)}")
        
        return " AND ".join(f"({cond})" for cond in sql_conditions)

    def extract_segments(self, data: Any) -> List[Dict[str, Any]]:
        """
        Sequentially extracts high‑lift rules on the residual dataset.
        """
        logger.info("🚀 Starting hierarchical segment extraction...")
        # Reset all accumulated state so repeated calls produce independent results
        self.stop_reason = None
        self.segments = []
        self.diagnostics_ = []
        self.feature_usage_counts = {}
        
        # Snapshot threshold before any grid search mutation inside the loop
        abs_min_sample_size = self.min_sample_size
        abs_min_events = self.min_events
        abs_min_lift = self.min_lift

        auto_created_db = False
        db_path = self.db_path
        db_temp_dir = self.db_temp_dir

        if db_path is None or db_temp_dir is None:
            db_path, db_temp_dir = setup_disk_backed_db("experiments")
            auto_created_db = True
            # When persistence is requested, remember the path so later calls
            # (evaluate_final_coverage / generate_feature_health_report) reuse the
            # same single artifact instead of rebuilding it from scratch.
            if self.persist_db:
                self.db_path = db_path
                self.db_temp_dir = db_temp_dir
            logger.info(f"📂 Created temporary disk-backed DB at: {db_path}")

        con = duckdb.connect(db_path)

        total_cores = os.cpu_count() or 1
        if self.engine_threads is not None:
            target_threads = max(1, int(self.engine_threads))
        else:
            target_threads = max(1, total_cores - 2) if total_cores > 4 else total_cores
        total_mem_gb = psutil.virtual_memory().total / (1024**3)
        if self.memory_limit_gb is not None:
            target_memory_gb = max(1, int(self.memory_limit_gb))
        else:
            # Default: utilise the host. On a 32 GB / 16-core box this yields
            # ~25 GB buffer and ~14 threads. Override via memory_limit_gb /
            # engine_threads on memory-constrained hardware.
            target_memory_gb = max(1, int(total_mem_gb * 0.8))

        con.execute(f"SET threads = {target_threads};")
        con.execute(f"SET memory_limit = '{target_memory_gb}GB';")
        if db_temp_dir:
            con.execute(f"PRAGMA temp_directory='{db_temp_dir}';")
        # con.execute("SET preserve_insertion_order = false;")
        
        logger.info(
            f"⚙️ DuckDB Configured for Disk Spilling: Threads={target_threads}/{total_cores}, "
            f"MemoryLimit={target_memory_gb}GB, TempDir={self.db_temp_dir}"
        )
        logger.info(f"📊 Sort priority: {self.sort_priority}")
        logger.info(
            f"📦 Binning method: {self.binning_method}"
            + (f" (naive_bins={self.naive_bins})" if self.binning_method == "naive" else "")
        )
        try:
            if isinstance(data, str):
                # Zero-copy path: `data` is a path to a DuckDB database file.
                # Attach it read-only and expose its `udl_data` table as the input
                # view so the dataset is never materialised into Python (pandas/arrow).
                src_path = data.replace("\\", "/")
                con.execute(f"ATTACH '{src_path}' AS __rs_src (READ_ONLY)")
                # Lazy VIEW (not a materialised copy) so the dataset is never
                # duplicated on disk when we already have it as a file.
                con.execute(
                    "CREATE OR REPLACE VIEW input_data_view AS SELECT * FROM __rs_src.udl_data"
                )
            else:
                con.register("input_data_view", data)
            # Cast the target to DOUBLE up front so boolean targets work with AVG,
            # and fail early with a clear message if the target column is missing.
            cols_info_probe = con.execute("DESCRIBE input_data_view").fetchall()
            probe_cols = {row[0] for row in cols_info_probe}
            if self.target not in probe_cols:
                raise ValueError(
                    f"Target column '{self.target}' not found in input data. "
                    f"Available columns: {sorted(probe_cols)}"
                )
            con.execute(
                f'''
                CREATE OR REPLACE TABLE current_df_base AS
                SELECT
                    ROW_NUMBER() OVER () AS "__rs_row_id",
                    * REPLACE (
                        CAST("{self.target}" AS DOUBLE) AS "{self.target}"
                    )
                FROM input_data_view
                '''
            )
            # Track excluded rows with a single mutable flag instead of rewriting the
            # whole residual table on every iteration. `current_df` is a filtered view
            # over the base so all downstream reads see only the live residual.
            con.execute(
                'ALTER TABLE current_df_base ADD COLUMN "__rs_excluded" BOOLEAN DEFAULT FALSE'
            )
            con.execute(
                'CREATE OR REPLACE VIEW current_df AS '
                'SELECT * FROM current_df_base WHERE "__rs_excluded" IS NOT TRUE'
            )

            cols_info = con.execute("DESCRIBE current_df").fetchall()
            columns_types = {row[0]: row[1] for row in cols_info}
            self._columns_types = columns_types
            self._categorical_cols = {
                col_name
                for col_name, dtype in columns_types.items()
                if self._resolve_optb_dtype(dtype) == "categorical"
            }
            # Drop internal bookkeeping columns so they never leak into feature logic.
            internal_cols = {"__rs_row_id", "__rs_excluded"}
            all_cols = [c for c in columns_types.keys() if c not in internal_cols]

            if self.enable_diversity:
                self._validate_feature_groups(all_cols)

            eligible_cols = sorted(
                c
                for c in all_cols
                if (
                    c != self.target
                    and c != "__rs_row_id"
                    and c != "__rs_excluded"
                    and c not in self.ignore_features
                )
)
            self.feature_usage_counts = {col: 0 for col in eligible_cols}

            if self.param_grid:
                sizes = self.param_grid.get("min_sample_size", [self.min_sample_size]) or [self.min_sample_size]
                lifts = self.param_grid.get("min_lift", [self.min_lift]) or [self.min_lift]
                experiments = [
                    {"min_sample_size": s, "min_lift": l}
                    for s, l in product(sizes, lifts)
                ]
                logger.info(f"📊 Dynamic Grid Search Enabled: {len(experiments)} configurations.")
            else:
                experiments = [{"min_sample_size": self.min_sample_size, "min_lift": self.min_lift}]

            original_base_rate = con.execute(
                f'SELECT AVG(CAST("{self.target}" AS DOUBLE)) FROM current_df'
            ).fetchone()[0] or 0.0

            logger.info(f"🔒 Locking Original Base Rate: {original_base_rate*100:.2f}%")

            for i in range(1, self.max_segments + 1):
                res = con.execute(
                    f'SELECT AVG(CAST("{self.target}" AS DOUBLE)), COUNT(*) FROM current_df'
                ).fetchone()
                current_base_rate, current_volume = res[0] or 0.0, res[1] or 0

                # Guard the minimum-volume check against an empty experiments list in case
                # a custom grid is passed with no valid values.
                min_floor_volume = (
                    min(exp["min_sample_size"] for exp in experiments)
                    if experiments
                    else self.min_sample_size
                )

                if current_base_rate == 0 or current_volume < min_floor_volume:
                    self.stop_reason = (
                        f"Insufficient remaining volume ({current_volume:,}) or base rate is zero."
                    )
                    logger.info(
                        f"⏹️ Stopping: base_rate={current_base_rate}, volume={current_volume} < "
                        f"min_floor={min_floor_volume}"
                    )
                    break

                logger.info(
                    f"🔄 Iteration {i} | Remaining Volume: {current_volume:,} | "
                    f"Base Rate: {current_base_rate*100:.2f}%"
                )

                iv_ranking, precomputed_bins = self.compute_iv_ranking_and_bin(
                    con, eligible_cols, columns_types
                )

                if self.selection_metric == "response_rate":
                    current_score_map = {row["variable"]: row["max_rr"] for row in iv_ranking}
                else:
                    current_score_map = {row["variable"]: row["iv"] for row in iv_ranking}
                    
                top_n_variable_names = [r["variable"] for r in iv_ranking[:self.top_n_vars]]
                iteration_snapshot = {}
                for col in eligible_cols:
                    used_count = self.feature_usage_counts.get(col, 0)
                    current_score = current_score_map.get(col, 0.0)
                    
                    if used_count >= self.max_feature_reuse:
                        status = "Excluded (Max Feature Reuse Exceeded)"
                    elif current_score <= 0.0:
                        status = f"Excluded ({self.selection_metric.upper()} is Zero/Invalid)"
                    elif col not in top_n_variable_names:
                        status = "Excluded (Outside Top N Features by Score)"
                    else:
                        status = "Eligible for Combination Search"
                        
                    iteration_snapshot[col] = {
                        "metric_score": current_score,
                        "metric_type": self.selection_metric,
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
                    row["variable"]
                    for row in iv_ranking
                    if self.feature_usage_counts.get(row["variable"], 0) < self.max_feature_reuse
                ]
                top_vars = allowed_vars[:self.top_n_vars]
                if not top_vars:
                    self.stop_reason = "All eligible features exhausted or failed dynamic score threshold."
                    logger.warning("⚠️ All eligible features exhausted. Aborting.")
                    break

                raw_target_arr = con.execute(
                    f'''
                    SELECT "{self.target}"
                    FROM current_df
                    ORDER BY "__rs_row_id"
                    '''
                ).fetchnumpy()[self.target]

                if isinstance(raw_target_arr, np.ma.MaskedArray):
                    clean_target_arr = raw_target_arr.filled(0)
                else:
                    clean_target_arr = raw_target_arr

                binned_data = {self.target: clean_target_arr}
                valid_vars = []
                for v in top_vars:
                    if v in precomputed_bins and len(np.unique(precomputed_bins[v])) > 1:
                        binned_data[v] = precomputed_bins[v]
                        valid_vars.append(v)
                if not valid_vars:
                    self.stop_reason = "No features had valid binned variation to construct candidate rules."
                    logger.warning("⚠️ No valid binned variables found. Stopping.")
                    break

                con.execute("DROP TABLE IF EXISTS binned_df")

                con.register("binned_data_view", binned_data)

                con.execute("""
                    CREATE TABLE binned_df AS
                    SELECT *
                    FROM binned_data_view
                """)

                con.unregister("binned_data_view")

                global_min_sample = (
                    min(exp["min_sample_size"] for exp in experiments)
                    if experiments
                    else self.min_sample_size
                )
                global_min_lift = (
                    min(exp["min_lift"] for exp in experiments)
                    if experiments
                    else self.min_lift
                )
                
                # Apply the current grid-search thresholds for this iteration so the candidate
                # pool is filtered consistently. The original values are restored after the loop.
                self.min_sample_size = global_min_sample
                self.min_lift = global_min_lift
                
                all_candidate_rules: List[Dict[str, Any]] = []

                # Level 1 (Singles)
                res_1 = self._agg_combinations(con, [(c,) for c in valid_vars], original_base_rate)
                valid_1way_vars = set()
                if res_1:
                    valid_1way_vars = {c["combo_vars"][0] for c in res_1}
                    if self.enable_1way:
                        all_candidate_rules.extend(res_1)

                # Level 2 (Pairs)
                valid_2way_sets = set()
                if len(valid_1way_vars) >= 2 and (self.enable_2way or self.enable_3way):
                    combos_2 = [c for c in combinations(sorted(valid_1way_vars), 2) if self.is_diverse(c)]
                    if combos_2:
                        res_2 = self._agg_combinations(con, combos_2, original_base_rate)
                        if res_2:
                            valid_2way_sets = {frozenset(c["combo_vars"]) for c in res_2}
                            if self.enable_2way:
                                all_candidate_rules.extend(res_2)

                # Level 3 (Triplets)
                if self.enable_3way and len(valid_1way_vars) >= 3 and valid_2way_sets:
                    combos_3 = [
                        c for c in combinations(sorted(valid_1way_vars), 3)
                        if self.is_diverse(c) and all(frozenset(p) in valid_2way_sets for p in combinations(c, 2))
                    ]
                    if combos_3:
                        res_3 = self._agg_combinations(con, combos_3, original_base_rate)
                        if res_3:
                            all_candidate_rules.extend(res_3)

                # Build a shortlist of candidates for each grid configuration and keep the top
                # rule from each configuration for later raw validation against the real table.
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
                    self.diagnostics_[-1]["candidate_funnel"] = {
                        "total_candidates_before_grid": len(all_candidate_rules),
                        "candidates_after_grid": 0,
                    }
                    self.stop_reason = "No candidate rules met the minimum grid thresholds (sample size / lift)."
                    logger.info("⏹️ No candidates cleared the grid. Stopping.")
                    break

                grid_candidates.sort(key=lambda x: self._get_sort_key(x), reverse=True)
                self.diagnostics_[-1]["candidate_funnel"] = {
                    "total_candidates_before_grid": len(all_candidate_rules),
                    "candidates_after_grid": len(grid_candidates),
                }

                # Re-run the strongest candidates against the real residual table using SQL.
                # This step enforces the hard constraints on actual data counts/events rather than
                # only the binned approximation used during candidate generation.
                selected_candidate = None
                closest_miss = None
                for candidate in grid_candidates:
                    rule_str = candidate["rule"]
                    sql_filter = self.parse_rule_to_sql(rule_str)
                    actual = con.execute(
                        f'SELECT COUNT(*) AS cnt, SUM(CAST("{self.target}" AS DOUBLE)) AS evt '
                        f'FROM current_df WHERE ({sql_filter})'
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
                        # NEW CODE: 3B - Capture the highest-ranked failed candidate
                        if closest_miss is None:
                            closest_miss = {
                                "rule": candidate["rule"],
                                "gaps": {
                                    "sample_size": {"actual": actual_cnt, "required": abs_min_sample_size, "ok": actual_cnt >= abs_min_sample_size},
                                    "events": {"actual": actual_evt, "required": abs_min_events, "ok": actual_evt >= abs_min_events},
                                    "lift": {"actual": actual_lift, "required": abs_min_lift, "ok": actual_lift >= abs_min_lift},
                                }
                            }
                        
                        logger.debug(
                            f"Candidate rejected by raw validation: {rule_str} -> "
                            f"rows={actual_cnt}, events={actual_evt}"
                        )

                if selected_candidate is None:
                    self.diagnostics_[-1]["near_miss"] = closest_miss
                    self.stop_reason = "Candidates existed but all failed raw SQL validation (sample size, min events, or lift)."
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
                    if "base_events" in r and r["rule"] != best_rule
                ]
                
                expand_mode = getattr(self, "expand_log_mode", "summary")
                should_show_champion = expand_mode in ("champion", "full") and expanded_in_this_iter
                
                if should_show_champion:
                    expanded_in_this_iter.sort(key=lambda x: self._get_sort_key(x), reverse=True)
                    top_exp = expanded_in_this_iter[:5]
            
                    logger.info("📊 Top adjacent-merge candidates vs final champion")
                    logger.info(
                        f"   {'Rank':<5} {'Type':<10} {'Lift':>6} {'Rate%':>7} {'Count':>8} {'Events':>8}  Rule"
                    )
                    logger.info("   " + "-" * 90)
            
                    logger.info(
                        f"   {'★':<5} {'CHAMPION':<10} "
                        f"{actual_lift:>6.2f}x {actual_rate:>6.1f}% "
                        f"{selected_candidate['actual_count']:>8} "
                        f"{selected_candidate['actual_events']:>8.0f}  "
                        f"{best_rule}"
                    )
            
                    for idx, e in enumerate(top_exp, 1):
                        champ_key = self._get_sort_key({
                            "lift": actual_lift,
                            "rate": actual_rate,
                            "count": selected_candidate["actual_count"],
                            "events": selected_candidate["actual_events"],
                        })    
                        cand_key = self._get_sort_key(e)
            
                        if cand_key > champ_key:
                            reason = "would have beaten champion (but failed raw validation)"
                        else:
                            _dim_order = {
                                "lift_count_rate":   ["lift", "count", "rate"],
                                "count_lift_rate":   ["count", "lift", "rate"],
                                "rate_lift_count":   ["rate", "lift", "count"],
                                "lift_rate_count":   ["lift", "rate", "count"],
                                "count_rate_lift":   ["count", "rate", "lift"],
                                "rate_count_lift":   ["rate", "count", "lift"],
                                "events_lift_rate":  ["events", "lift", "rate"],
                                "events_rate_lift":  ["events", "rate", "lift"],
                                "lift_events_rate":  ["lift", "events", "rate"],
                                "rate_events_lift":  ["rate", "events", "lift"],
                                "events_count_rate": ["events", "count", "rate"],
                                "events_rate_count": ["events", "rate", "count"],
                                "count_events_rate": ["count", "events", "rate"],
                                "rate_events_count": ["rate", "events", "count"],
                            }
                            priority_order = _dim_order.get(
                                self.sort_priority, ["lift", "rate", "count"])
                            champ_vals = {
                                "lift": actual_lift,
                                "rate": actual_rate,
                                "count": selected_candidate["actual_count"],
                                "events": selected_candidate["actual_events"],
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
                                    label = {
                                        "lift": "lower lift",
                                        "count": "smaller count",
                                        "rate": "lower rate",
                                        "events": "fewer events",
                                    }[dim]
                                    reason = (
                                        f"{label} ({cand_vals[dim]:.2f} < {champ_vals[dim]:.2f})"
                                    )
                                    break
                                elif cand_vals[dim] > champ_vals[dim]:
                                    break
                
                        logger.info(
                            f"   {idx:<5} {'expanded':<10} "
                            f"{e['lift']:>6.2f}x {e['rate']:>6.1f}% "
                            f"{e['count']:>8} {e['events']:>8.0f}  "
                            f"{e['rule']}  → {reason}"
                        )

                self.diagnostics_[-1]["winning_segment"] = {
                    "rule": best_rule,
                    "sql_filter": best_raw_sql,
                    "variables_used": list(winning_combo),
                    "lift": actual_lift,
                    "count": int(selected_candidate["actual_count"]),
                }

                # Mark matched rows as excluded in place. NULL rows evaluate
                # `(sql) IS TRUE` to NULL/false and therefore stay in the residual,
                # preserving the original NULL handling. A view (`current_df`) keeps
                # all reads pointed at the live residual without rewriting the table.
                con.execute(
                    f"""
                    UPDATE current_df_base
                    SET "__rs_excluded" = TRUE
                    WHERE ({best_raw_sql}) IS TRUE
                """
                )
        finally:
            try:
                con.close()
            except Exception:
                pass
            if auto_created_db and not self.persist_db:
                if os.path.exists(db_path):
                    try:
                        os.remove(db_path)
                    except Exception as e:
                        logger.debug(f"Cleanup failed for {db_path}: {e}")
                if os.path.exists(db_temp_dir):
                    try:
                        shutil.rmtree(db_temp_dir)
                    except Exception as e:
                        logger.debug(f"Cleanup failed for {db_temp_dir}: {e}")

        self.min_sample_size = abs_min_sample_size
        self.min_events = abs_min_events
        self.min_lift = abs_min_lift 
        if not self.stop_reason and len(self.segments) == self.max_segments:
            self.stop_reason = f"Reached max_segments limit ({self.max_segments})."
        logger.info("🏁 Extraction complete.")
        return self.segments

    def evaluate_final_coverage(self, original_data: Any) -> List[Dict[str, Any]]:
        """
        Evaluates the hierarchical segmentation on the original dataset.
        """
        if not self.segments:
            return []

        logger.info("📊 Evaluating final hierarchical coverage on original data...")
        target_db = self.db_path if self.db_path else ":memory:"
        con = duckdb.connect(target_db)
        if self.db_temp_dir and os.path.exists(self.db_temp_dir):
            con.execute(f"PRAGMA temp_directory='{self.db_temp_dir}';")

        # Reuse the single materialised original dataset when present (persistent
        # mode), otherwise register the caller's data and materialise it here.
        try:
            con.execute("SELECT 1 FROM original_df LIMIT 1")
            reused_original = True
        except Exception:
            reused_original = False

        if not reused_original:
            con.register("input_data_view", original_data)
            con.execute("CREATE OR REPLACE TABLE original_df AS SELECT * FROM input_data_view")

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
        
    def explain_no_segments(self) -> str:
        """
        Produces a human-readable diagnostic report explaining why
        extract_segments() returned zero segments or stopped early.
        """
        if not self.diagnostics_:
            if self.stop_reason:
                return f"Extraction produced 0 segment(s).\nStop reason: {self.stop_reason}"
            return "No diagnostics available -- call extract_segments() first."

        lines: List[str] = []
        n_found = len(self.segments)
        last = self.diagnostics_[-1]

        lines.append("=" * 80)
        lines.append("SEGMENT EXTRACTION DIAGNOSTIC REPORT")
        lines.append("=" * 80)
        lines.append(f"Segments Extracted : {n_found} / {self.max_segments}")
        lines.append(f"Iterations Run     : {len(self.diagnostics_)}")
        lines.append(f"Stop Reason        : {self.stop_reason or 'Unknown'}")
        lines.append("-" * 80)

        # Configured thresholds
        lines.append("Active Constraints:")
        lines.append(f"  - min_sample_size : {self.min_sample_size:,}")
        lines.append(f"  - min_lift        : {self.min_lift:.2f}x")
        lines.append(f"  - min_events      : {self.min_events}")
        lines.append(f"  - selection_metric: {self.selection_metric}")
        lines.append("")

        # Feature eligibility snapshot
        features_state = last.get("features_state", {})
        if features_state:
            status_counts: Dict[str, int] = {}
            for v in features_state.values():
                status_counts[v["status"]] = status_counts.get(v["status"], 0) + 1

            lines.append(f"Feature Eligibility Summary (Iteration {last['iteration']}):")
            for status, cnt in sorted(status_counts.items(), key=lambda x: -x[1]):
                lines.append(f"  - {cnt:<3} feature(s): {status}")
            lines.append("")

        # Candidate generation funnel
        funnel = last.get("candidate_funnel")
        if funnel:
            lines.append(f"Candidate Funnel (Iteration {last['iteration']}):")
            lines.append(f"  - 1-way candidates passing base criteria : {funnel.get('1way_candidates', 0):,}")
            lines.append(f"  - 2-way candidates passing base criteria : {funnel.get('2way_candidates', 0):,}")
            lines.append(f"  - 3-way candidates passing base criteria : {funnel.get('3way_candidates', 0):,}")
            lines.append(f"  - Total candidates before grid search   : {funnel.get('total_candidates_before_grid', 0):,}")
            lines.append(f"  - Candidates clearing grid filter       : {funnel.get('candidates_after_grid', 0):,}")
            lines.append("")

        # Closest near miss that failed raw SQL evaluation
        near_miss = last.get("near_miss")
        if near_miss:
            lines.append("Closest Candidate Reject (Failed Raw Validation):")
            lines.append(f"  Rule: {near_miss['rule']}")
            for dim, info in near_miss["gaps"].items():
                status = "OK" if info["ok"] else "FAIL"
                lines.append(f"    • {dim:<12}: actual={info['actual']:.4g} | required={info['required']:.4g} [{status}]")
            lines.append("")

        lines.append("=" * 80)
        return "\n".join(lines)

    def generate_feature_health_report(
        self, original_data: Any, features: List[str]
    ) -> pd.DataFrame:
        """
        Generates a feature health report on the original dataset using DuckDB native SQL.
        """
        if not features:
            logger.warning("⚠️ No features provided for health report generation.")
            return pd.DataFrame()

        unique_features = list(dict.fromkeys(features))

        logger.info(
            f"📋 Generating DuckDB Naive Feature Health Report for {len(unique_features)} feature(s): "
            f"{unique_features}"
        )

        target_db = self.db_path if self.db_path else ":memory:"
        con = duckdb.connect(target_db)
        if self.db_temp_dir and os.path.exists(self.db_temp_dir):
            con.execute(f"PRAGMA temp_directory='{self.db_temp_dir}';")

        # Reuse the single materialised original dataset when present (persistent
        # mode), otherwise register the caller's data and materialise it here.
        try:
            con.execute("SELECT 1 FROM original_df LIMIT 1")
            # View (not a table) so we don't duplicate the full dataset again.
            con.execute("CREATE OR REPLACE VIEW input_df AS SELECT * FROM original_df")
            reused_original = True
        except Exception:
            reused_original = False

        if not reused_original:
            con.register("input_data_view", original_data)
            con.execute("CREATE TABLE input_df AS SELECT * FROM input_data_view")

        cols_info = con.execute("DESCRIBE input_df").fetchall()
        columns_types = {row[0]: row[1] for row in cols_info}

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

            duckdb_type = columns_types[col].upper()
            is_numeric = any(
                t in duckdb_type
                for t in [
                    "INT",
                    "BIGINT",
                    "DOUBLE",
                    "FLOAT",
                    "DECIMAL",
                    "REAL",
                    "NUMERIC",
                    "HUGEINT",
                    "TINYINT",
                    "SMALLINT",
                ]
            )

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
