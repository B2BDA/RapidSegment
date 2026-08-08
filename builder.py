"""
Strategic Segmentation Engine
==============================
Combinatorial heuristic segmentation using Optimal Binning, Apriori pruning,
and vectorized DuckDB scorecard deciling.

Author: Bishwarup Biswas + Gemini + DeepSeek + ChatGPT
Python Version: 3.9+
"""

import logging
import os
import re
from datetime import datetime
from itertools import combinations, product
from typing import Any, Dict, List, Optional, Tuple, Union
import duckdb
import numpy as np
import psutil
from joblib import Parallel, delayed
from optbinning import OptimalBinning
import pandas as pd

# -----------------------------------------------------------------------------
# Module-level configuration
# -----------------------------------------------------------------------------
now = datetime.now()
timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [%(filename)s:%(lineno)d] | %(message)s",
)
logger = logging.getLogger("StrategicEngine")

# Pre-compiled regex for fast parsing inside loops
_BRACKET_REGEX = re.compile(r"\[(.*?)\]", flags=re.DOTALL)


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
        naive_bins: int = 5 ,
        selection_metric: str = "iv",
        expand_log_mode: str = "summary",  # "none" | "summary" | "full"
    ) -> None:
        """
        Args:
            target: Name of the binary target column (1 = Event, 0 = Non-Event).
            n_jobs: Number of parallel jobs for IV computation. -1 uses all but one core.
            min_sample_size: Absolute minimum row count for a valid rule. Used as a fallback when param_grid is None.
            min_lift: Absolute minimum lift threshold (hard constraint).
                Enforced on all final segments, calculated relative to the locked original base response rate. 
                Not relaxed during param_grid exploration.: Minimum lift threshold. Used as a fallback when param_grid is None.
            min_events: Minimum number of positive events for a valid rule. Used as a fallback when param_grid is None.
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
                    Can be a predefined shortcut string:
                    - 'lift_rate_count': Lift → Response Rate → Sample Size
                    - 'count_lift_rate': Sample Size → Lift → Response Rate
                    - 'rate_lift_count': Response Rate → Lift → Sample Size
                    - 'rate_count_lift': Response Rate → Sample Size → Lift
                    - 'lift_count_rate': Lift → Sample Size → Response Rate
                    - 'count_rate_lift': Sample Size → Response Rate → Lift
            binning_method: Which binning engine to use for feature discretization and IV computation.
                    -  'optimal' for OptBinning 
                    -  'naive' for simple quantile/category heuristics.
            naive_bins: Number of quantile bins used when binning_method is 'naive'.
            selection_metric: Metric used to rank features for top_n_vars selection. Support "iv" or "response_rate".
            expand_log_mode: Controls verbosity of adjacent-bin expansion logging.
                    - "summary" (default): neat table summary per iteration
                    - "full": table + top expanded candidates at INFO level
                    - "none": only DEBUG messages
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
        # Diagnostic repository (feature journey tracking)
        self.diagnostics_: List[Dict[str, Any]] = []
        self.binning_method = binning_method  
        self.naive_bins = naive_bins     
        self.selection_metric = selection_metric
        self.expand_log_mode = expand_log_mode if expand_log_mode in ("none", "summary", "full") else "summary"      

    @staticmethod
    def _resolve_optb_dtype(duckdb_type: str) -> str:
        """
        Determines the correct OptBinning data type flag from a DuckDB type string.

        Args:
            duckdb_type: DuckDB column type as string.

        Returns:
            'categorical' or 'numerical'.
        """
        dtype_upper = duckdb_type.upper()
        if any(t in dtype_upper for t in ["VARCHAR", "CHAR", "STRING", "TEXT", "UUID"]):
            return "categorical"
        return "numerical"

    def _validate_feature_groups(self, columns: List[str]) -> None:
        """
        Validates that all declared feature group variables exist in the target dataset.

        Args:
            columns: List of all column names from the table.

        Raises:
            ValueError: If any feature in a group is not found.
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

        Args:
            var: Feature name.

        Returns:
            Group name or the feature name.
        """
        for group, vars_list in self.feature_groups.items():
            if var in vars_list:
                return group
        return var

    def is_diverse(self, combo: Tuple[str, ...]) -> bool:
        """
        Ensures a tuple of features spans strictly distinct analytical groups.

        Args:
            combo: Tuple of feature names.

        Returns:
            True if all features belong to different groups (or diversity is disabled).
        """
        if not self.enable_diversity:
            return True
        groups = [self.get_group(v) for v in combo]
        return len(groups) == len(set(groups))

    def _get_sort_key(self, rule: Dict[str, Any]) -> Tuple[float, ...]:
        """
        Utility function to shorlist rules based on users choice on which metric to prioritize.

        Args:
            rule: Dictionary of Rules that was captured and passed hard constraints.

        Returns:
            Returns a tuple for sorting rules based on self.sort_priority.
        """
        priority = self.sort_priority
        if priority == "lift_count_rate":
            return (rule["lift"], rule["count"], rule["rate"])
        elif priority == "count_lift_rate":
            return (rule["count"], rule["lift"], rule["rate"])
        elif priority == "rate_lift_count":
            return (rule["rate"], rule["lift"], rule["count"])
        elif priority == "lift_rate_count":
            return (rule["lift"], rule["rate"], rule["count"])
        elif priority == "count_rate_lift":
            return (rule["count"], rule["rate"], rule["lift"])
        elif priority == "rate_count_lift":
            return (rule["rate"], rule["count"], rule["lift"])
        else:
            # Fallback: lift, count, rate
            return (rule["lift"], rule["count"], rule["rate"])

    def compute_iv_ranking_and_bin(
        self,
        con: duckdb.DuckDBPyConnection,
        eligible_cols: List[str],
        columns_types: Dict[str, str],
    ) -> Tuple[List[Dict[str, Union[str, float]]], Dict[str, np.ndarray]]:
        """
        Computes Information Value (IV) and pre‑computed bins in a single parallel pass.

        Args:
            con: DuckDB connection.
            eligible_cols: List of feature columns to evaluate.
            columns_types: Mapping of column name to its DuckDB type.

        Returns:
            - Ranking: List of dicts with 'variable' and 'iv' (IV * 100).
            - Precomputed bins: Dict mapping feature name to transformed bin array.
        """
        logger.info(f"🔍 Computing IV and bins for {len(eligible_cols)} features...")

        def _worker(col: str) -> Tuple[str, float, float, Optional[np.ndarray]]:
            try:
                thread_con = con.cursor()
                data_dict = thread_con.execute(
                    f'SELECT "{col}", "{self.target}" FROM current_df'
                ).fetchnumpy()

                col_arr_raw = data_dict[col]
                target_arr_raw = data_dict[self.target]

                # Unmask DuckDB MaskedArrays
                if isinstance(col_arr_raw, np.ma.MaskedArray):
                    col_arr = col_arr_raw.filled(np.nan if np.issubdtype(col_arr_raw.dtype, np.number) else None)
                else:
                    col_arr = col_arr_raw

                if isinstance(target_arr_raw, np.ma.MaskedArray):
                    target_arr = target_arr_raw.filled(0)
                else:
                    target_arr = target_arr_raw

                dtype = self._resolve_optb_dtype(columns_types[col])
                
                iv_val = 0.0
                max_rr = 0.0

                if self.binning_method == "naive":
                    total_events = np.sum(target_arr)
                    total_non_events = len(target_arr) - total_events
                    transformed_bins = np.empty(len(col_arr), dtype=object)
                    
                    # Helper function to compute both metrics per bin
                    def _process_naive_bin_stats(mask, current_iv, current_max_rr):
                        bin_events = np.sum(target_arr[mask])
                        bin_total = np.sum(mask)
                        bin_non_events = bin_total - bin_events
                        
                        if bin_total > 0:
                            # Update max response rate (enforcing hard constraints)
                            if bin_total >= self.min_sample_size and bin_events >= self.min_events:
                                rr = bin_events / bin_total
                                if rr > current_max_rr:
                                    current_max_rr = rr
                                    
                            # Update IV chunk
                            if bin_events > 0 or bin_non_events > 0:
                                pct_events = max((bin_events / total_events) if total_events > 0 else 0, 1e-6)
                                pct_non_events = max((bin_non_events / total_non_events) if total_non_events > 0 else 0, 1e-6)
                                current_iv += (pct_non_events - pct_events) * np.log(pct_non_events / pct_events)
                                
                        return current_iv, current_max_rr

                    if dtype == "numerical":
                        col_arr_float = col_arr.astype(float, copy=False)
                        valid_mask = ~np.isnan(col_arr_float)
                        if not np.any(valid_mask):
                            raise ValueError("Column contains only NaNs")
                            
                        q = np.linspace(0, 100, self.naive_bins + 1)
                        edges = np.unique(np.percentile(col_arr_float[valid_mask], q))
                        
                        if len(edges) < 2:
                            edges = np.array([-np.inf, np.inf])
                        else:
                            edges[0], edges[-1] = -np.inf, np.inf
                            
                        bin_indices = np.digitize(col_arr_float, edges[1:-1], right=False)

                        for i in range(len(edges) - 1):
                            lower, upper = edges[i], edges[i+1]
                            lower_str = "-inf" if np.isinf(lower) and lower < 0 else str(lower)
                            upper_str = "inf" if np.isinf(upper) and upper > 0 else str(upper)
                            
                            mask = (bin_indices == i) & valid_mask
                            transformed_bins[mask] = f"[{lower_str}, {upper_str})"
                            iv_val, max_rr = _process_naive_bin_stats(mask, iv_val, max_rr)

                        # Handle Missing for numerical
                        missing_mask = ~valid_mask
                        if np.any(missing_mask):
                            transformed_bins[missing_mask] = "Missing"
                            iv_val, max_rr = _process_naive_bin_stats(missing_mask, iv_val, max_rr)

                    else:
                        # Handle categorical arrays
                        missing_mask = np.array([
                            x is None 
                            or (isinstance(x, float) and np.isnan(x)) 
                            or (isinstance(x, str) and x.strip() in ("", "None", "nan", "NaN", "<NA>", "null", "NULL"))
                            for x in col_arr
                        ])
                        valid_mask = ~missing_mask

                        if np.any(valid_mask):
                            valid_vals = col_arr[valid_mask].astype(str)
                            unique_vals = np.unique(valid_vals)
                            
                            for val in unique_vals:
                                mask = valid_mask & (col_arr.astype(str) == val)
                                transformed_bins[mask] = f"['{val}']"
                                iv_val, max_rr = _process_naive_bin_stats(mask, iv_val, max_rr)

                        if np.any(missing_mask):
                            transformed_bins[missing_mask] = "Missing"
                            iv_val, max_rr = _process_naive_bin_stats(missing_mask, iv_val, max_rr)

                    transformed_bins = np.asarray(transformed_bins, dtype=str)

                else:
                    # Optimal binning fallback
                    optb = OptimalBinning(name=col, dtype=dtype)
                    optb.fit(col_arr, target_arr)
                    
                    bin_table = optb.binning_table.build()
                    iv_val = bin_table["IV"].values[-1]
                    transformed_bins = np.asarray(optb.transform(col_arr, metric="bins"), dtype=str)
                    
                    # Extract max response rate enforcing hard constraints
                    valid_bins = bin_table[
                        (bin_table["Count"] >= self.min_sample_size) & 
                        (bin_table["Event"] >= self.min_events)
                    ]
                    if not valid_bins.empty:
                        max_rr = valid_bins["Event rate"].max()

                thread_con.close()
                return col, float(iv_val) * 100, float(max_rr), transformed_bins
            except Exception as e:
                logger.debug(f"Computation failed for {col}: {e}")
                return col, 0.0, 0.0, None

        results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_worker)(col) for col in eligible_cols
        )

        ranking = []
        precomputed_bins = {}
        for col, iv, rr, bins in results:
            ranking.append({"variable": col, "iv": iv, "max_rr": rr})
            if bins is not None:
                precomputed_bins[col] = bins

        # Dynamic sorting based on user selection
        if self.selection_metric == "response_rate":
            ranking.sort(key=lambda x: x["max_rr"], reverse=True)
        else:
            ranking.sort(key=lambda x: x["iv"], reverse=True)
            
        return ranking, precomputed_bins

    def _expand_adjacent_bins(
        self,
        con: duckdb.DuckDBPyConnection,
        combo: Tuple[str, ...],
        base_rate: float,
        base_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        For each qualifying bin result, attempts to merge it with its adjacent
        neighbour bin on every variable in the combo, producing expanded candidates
        that capture more events while still clearing min_lift and min_events.

        The expansion is per-variable: for a combo (A, B), for each qualifying
        (bin_A, bin_B) result, we try expanding bin_A to include its neighbour,
        and separately try expanding bin_B to include its neighbour, keeping the
        other variable's bin fixed. Expanded candidates that improve event capture
        over the base result while maintaining lift are added as additional
        candidates.

        Args:
            con:          DuckDB connection.
            combo:        Tuple of feature names for this combination.
            base_rate:    Global event rate (used to compute lift).
            base_results: Already-qualifying results from the base GROUP BY pass.

        Returns:
            List of additional expanded rule dicts (same schema as _agg_combinations
            output). May be empty if no expansion improves results.
        """
        if not base_results:
            return []

        # Build a lookup: variable -> sorted list of unique bin labels in binned_df.
        # We only do this once per combo and only for variables that appear in combo.
        bin_labels: Dict[str, List[str]] = {}
        for col in combo:
            rows = con.execute(
                f'SELECT DISTINCT CAST("{col}" AS VARCHAR) AS b '
                f'FROM binned_df ORDER BY b'
            ).fetchall()
            bin_labels[col] = [r[0] for r in rows]

        expanded: List[Dict[str, Any]] = []
        seen_rules: set = set()

        # Index base results by their rule string to avoid exact duplicates
        for r in base_results:
            seen_rules.add(r["rule"])

        for result in base_results:
            # Parse the current bin assignment per variable from the rule string.
            # Rule format: "colA=<bin> & colB=<bin> & ..."
            current_bins: Dict[str, str] = {}
            for part in result["rule"].split(" & "):
                if "=" not in part:
                    continue
                col, bin_val = part.split("=", 1)
                current_bins[col.strip()] = bin_val.strip()

            # For each variable in the combo, try expanding to include the
            # adjacent neighbour bin (both directions: left and right).
            for col in combo:
                if col not in current_bins or col not in bin_labels:
                    continue

                labels = bin_labels[col]
                current_bin = current_bins[col]

                if current_bin not in labels:
                    continue

                idx = labels.index(current_bin)
                neighbours = []
                if idx > 0:
                    neighbours.append(labels[idx - 1])   # left neighbour
                if idx < len(labels) - 1:
                    neighbours.append(labels[idx + 1])   # right neighbour

                for neighbour_bin in neighbours:
                    # Build the expanded bin set for this variable:
                    # the original bin + the neighbour bin.
                    if current_bin == "Missing" or neighbour_bin == "Missing":
                        continue
                    expanded_bins_for_col = [current_bin, neighbour_bin]

                    # Construct the WHERE clause for the expanded query.
                    # For the expanding variable: bin IN (current, neighbour).
                    # For all other variables: bin = their fixed value.
                    where_parts = []
                    rule_parts = []
                    for c in combo:
                        if c == col:
                            in_list = ", ".join(
                                f"'{b}'" for b in expanded_bins_for_col
                            )
                            where_parts.append(
                                f'CAST("{c}" AS VARCHAR) IN ({in_list})'
                            )
                            # Rule label: show both bins joined
                            rule_parts.append(
                                f"{c}=[{', '.join(sorted(expanded_bins_for_col))}]"
                            )
                        else:
                            fixed_bin = current_bins[c]
                            where_parts.append(
                                f'CAST("{c}" AS VARCHAR) = \'{fixed_bin}\''
                            )
                            rule_parts.append(f"{c}={fixed_bin}")

                    rule_str = " & ".join(rule_parts)

                    # Skip if we've already produced this rule
                    if rule_str in seen_rules:
                        continue
                    seen_rules.add(rule_str)

                    where_clause = " AND ".join(where_parts)
                    try:
                        row = con.execute(
                            f"""
                            SELECT
                                COUNT("{self.target}")::BIGINT AS cnt,
                                SUM(CAST("{self.target}" AS DOUBLE)) AS evt
                            FROM binned_df
                            WHERE {where_clause}
                            """
                        ).fetchone()
                    except Exception:
                        continue

                    if row is None:
                        continue

                    exp_count, exp_events = row[0] or 0, row[1] or 0.0
                    if exp_count == 0:
                        continue

                    exp_rate = (exp_events / exp_count) * 100.0
                    exp_lift = exp_rate / (base_rate * 100.0) if base_rate > 0 else 0.0

                    # Only keep if it still clears hard constraints AND captures
                    # more events than the base result it was derived from.
                    if (
                        exp_lift >= self.min_lift
                        and exp_events >= self.min_events
                        and exp_count >= self.min_sample_size
                        and exp_events > result["events"]
                    ):
                        expanded.append(
                            {
                                "rule": rule_str,
                                "count": exp_count,
                                "rate": exp_rate,
                                "lift": exp_lift,
                                "events": exp_events,
                                "combo_vars": combo,
                                "base_events": result["events"],  # for Δ calculation
                            }
                        )

        if not expanded:
            logger.debug(f"↩️ No expansion candidates found for combo {combo}")

        return expanded

    def _agg_combinations(
        self,
        con: duckdb.DuckDBPyConnection,
        combo_list: List[Tuple[str, ...]],
        base_rate: float,
    ) -> List[Dict[str, Any]]:
        """
        Batches SQL GROUP BY queries for a list of feature combinations.

        Args:
            con: DuckDB connection.
            combo_list: List of tuples, each tuple is a combination of features.
            base_rate: Global event rate (used to compute lift).

        Returns:
            List of rule dictionaries with keys: rule, count, events, rate, lift, combo_vars.
        """
        if not combo_list:
            return []

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

        # Map each query back to its combo so we can run expansion per combo.
        combo_for_query = []
        for combo in combo_list:
            combo_for_query.append(combo)

        valid_results = []
        chunk_size = 100

        # We need per-combo base results for the expansion step, so collect them
        # before merging into valid_results.
        per_combo_base: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {
            combo: [] for combo in combo_list
        }

        for i in range(0, len(queries), chunk_size):
            chunk = queries[i:i + chunk_size]
            # combos_chunk = combo_for_query[i:i + chunk_size]
            union_query = " UNION ALL ".join(chunk)

            res = con.execute(union_query).fetchall()
            for row_idx, (rule, count, events, combo_vars_str) in enumerate(res):
                rate = (events / count) * 100.0 if count > 0 else 0
                lift = rate / (base_rate * 100.0) if base_rate > 0 else 0
                combo_key = tuple(combo_vars_str.split(","))

                if lift >= self.min_lift and events >= self.min_events:
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

        # --- Adjacent bin expansion ---
        # For each combo that produced at least one qualifying result, try to
        # find expanded candidates that capture more events while keeping lift.
        all_expanded: List[Dict[str, Any]] = []
        expansion_stats: Dict[str, Dict[str, Any]] = {}  # combo_str -> stats

        for combo in combo_list:
            base = per_combo_base.get(combo, [])
            if not base:
                continue
            # Only perform adjacent bin expansion if we are using naive binning
            if getattr(self, "binning_method", "naive") == "naive":
                expanded = self._expand_adjacent_bins(con, combo, base_rate, base)
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

        # ---- Nice expansion logging ----
        mode = getattr(self, "expand_log_mode", "summary")
        if expansion_stats and mode != "none":
            # Table header
            logger.info("🔀 Adjacent-bin expansion summary")
            logger.info(
                f"   {'Combo':<42} {'#exp':>5}  {'Best Δevents':>12}  {'Best lift':>9}"
            )
            logger.info("   " + "-" * 72)

            # Sort by best event gain descending
            for combo_key, st in sorted(
                expansion_stats.items(), key=lambda x: x[1]["best_delta"], reverse=True
            ):
                logger.info(
                    f"   {combo_key:<42} {st['n_exp']:>5}  "
                    f"{st['best_delta']:>+12.0f}  {st['best_lift']:>8.2f}x"
                )

            total_exp = sum(s["n_exp"] for s in expansion_stats.values())
            logger.info(f"   → Total expanded candidates generated: {total_exp}")

            # Full mode: also show the top 3 concrete rules
            if mode == "full":
                top = sorted(all_expanded, key=lambda x: x["events"], reverse=True)[:3]
                logger.info("   Top expanded rules:")
                for i, e in enumerate(top, 1):
                    delta = e["events"] - e.get("base_events", e["events"])
                    logger.info(
                        f"     {i}. {e['rule'][:90]}"
                        f"  | events {e['events']:.0f} (Δ{delta:+.0f}) | lift {e['lift']:.2f}x"
                    )

        return valid_results

    def parse_rule_to_sql(self, rule_str: str) -> str:
        """
        Translates OptBinning / naive / expanded rule strings into a production SQL WHERE clause.

        Handles:
        - Normal numerical ranges:  col=[10.0, 20.0)
        - Expanded adjacent bins:  col=[[10.0, 20.0), [20.0, 30.0)]
        - Categorical:             col=['A', 'B']  or  col=['success']
        - Missing / Special
        """
        parts = [p.strip() for p in rule_str.split("&")]
        sql_conditions: List[str] = []

        for part in parts:
            if "=" not in part:
                continue

            col, interval = [x.strip() for x in part.split("=", 1)]
            bracket_match = _BRACKET_REGEX.search(interval)

            # ------------------------------------------------------------------
            # 1. Detect expanded numerical form produced by _expand_adjacent_bins
            #    Example: [[2.0, 3.0), [3.0, inf)]
            # ------------------------------------------------------------------
            if interval.startswith("[[") or (interval.startswith("[") and ")," in interval and interval.count("[") >= 2):
                # Extract all individual ranges
                # Remove outer [ ] first
                inner = interval.strip()
                if inner.startswith("[") and inner.endswith("]"):
                    inner = inner[1:-1]

                # Split on "), " while keeping the closing )
                raw_tokens = re.split(r"\),\s*", inner)
                ranges = []
                for tok in raw_tokens:
                    tok = tok.strip()
                    if not tok:
                        continue
                    if not tok.endswith(")"):
                        tok = tok + ")"
                    # Now tok is something like "[2.0, 3.0)" or "[3.0, inf)"
                    if tok[0] in ("[", "(") and tok[-1] in ("]", ")"):
                        left_char = tok[0]
                        right_char = tok[-1]
                        content = tok[1:-1]
                        lo, hi = [x.strip() for x in content.split(",", 1)]
                        ranges.append((left_char, lo, hi, right_char))

                if ranges:
                    # Take outermost bounds
                    # Sort by lower bound so we get the true min/max even if expansion order was wrong
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
                        range_conds.append(f"{col} {op} {overall_lower}")
                    if overall_upper.lower() != "inf":
                        op = "<" if overall_upper_char == ")" else "<="
                        range_conds.append(f"{col} {op} {overall_upper}")

                    if range_conds:
                        sql_conditions.append(" AND ".join(range_conds))
                    continue

            # ------------------------------------------------------------------
            # 2. Categorical detection
            # ------------------------------------------------------------------
            is_categorical = False
            if bracket_match:
                content = bracket_match.group(1)
                if any(k in interval for k in ("'", '"', "Array", "Categorical")) or not interval.startswith(("[", "(")):
                    is_categorical = True
                elif len(content.split(",")) > 2:
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

                formatted_items = ", ".join(
                    [f"'{item}'" if isinstance(item, str) else str(item) for item in raw_items]
                )
                if formatted_items:
                    sql_conditions.append(f"{col} IN ({formatted_items})")
                continue

            # ------------------------------------------------------------------
            # 3. Missing / Special
            # ------------------------------------------------------------------
            if interval in ["Special", "Missing"]:
                sql_conditions.append(f"{col} IS NULL")
                continue

            # ------------------------------------------------------------------
            # 4. Normal continuous range  e.g. [10.0, 20.0)  or  (-inf, 5.0]
            # ------------------------------------------------------------------
            if interval.startswith(("[", "(")):
                left_char, right_char = interval[0], interval[-1]
                lower_str, upper_str = [x.strip() for x in interval[1:-1].split(",", 1)]

                range_conds = []
                if lower_str.lower() != "-inf":
                    op = ">=" if left_char == "[" else ">"
                    range_conds.append(f"{col} {op} {lower_str}")
                if upper_str.lower() != "inf":
                    op = "<=" if right_char == "]" else "<"
                    range_conds.append(f"{col} {op} {upper_str}")

                if range_conds:
                    sql_conditions.append(" AND ".join(range_conds))

        return " AND ".join(f"({cond})" if "AND" in cond else cond for cond in sql_conditions)

    def extract_segments(self, data: Any) -> List[Dict[str, Any]]:
        """
        Sequentially extracts high‑lift rules on the residual dataset.

        After extraction, the stored counts reflect the final hierarchical segmentation
        on the original dataset, ensuring consistency with `evaluate_final_coverage()`.

        Args:
            data: Input data (will be loaded into DuckDB).

        Returns:
            List of segment dictionaries with keys: segment_id, rule_string, sql_filter,
            count, rate, lift, meta_applied_sample_size, meta_applied_min_lift.
        """
        logger.info("🚀 Starting hierarchical segment extraction...")

        # Use in-memory for constrained environments
        con = duckdb.connect(":memory:")

        total_cores = os.cpu_count() or 1
        target_threads = max(1, total_cores - 2) if total_cores > 4 else total_cores
        total_memory_bytes = psutil.virtual_memory().total
        target_memory_gb = max(1, int((total_memory_bytes * 0.6) / (1024 ** 3)))

        con.execute(f"SET threads = {target_threads};")
        con.execute(f"SET memory_limit = '{target_memory_gb}GB';")
        con.execute("SET preserve_insertion_order = false;")
        logger.info(
            f"⚙️ DuckDB Configured: Threads={target_threads}/{total_cores}, "
            f"MemoryLimit={target_memory_gb}GB"
        )
        
        logger.info(f"📊 Sort priority: {self.sort_priority}")
        logger.info(f"📦 Binning method: {self.binning_method} (naive_bins={self.naive_bins})")
        
        con.execute("CREATE OR REPLACE TABLE current_df AS SELECT * FROM data")

        cols_info = con.execute("DESCRIBE current_df").fetchall()
        columns_types = {row[0]: row[1] for row in cols_info}
        all_cols = list(columns_types.keys())

        if self.enable_diversity:
            self._validate_feature_groups(all_cols)

        eligible_cols = [
            c for c in all_cols if c != self.target and c not in self.ignore_features
        ]
        self.feature_usage_counts = {col: 0 for col in eligible_cols}

        # Build parameter grid experiments
        if self.param_grid:
            sizes = self.param_grid.get("min_sample_size", [self.min_sample_size])
            lifts = self.param_grid.get("min_lift", [self.min_lift])
            experiments = [
                {"min_sample_size": s, "min_lift": l}
                for s, l in product(sizes, lifts)
            ]
            logger.info(
                f"📊 Dynamic Grid Search Enabled: {len(experiments)} configurations."
            )
        else:
            experiments = [{"min_sample_size": self.min_sample_size, "min_lift": self.min_lift}]

        # Cache original base_rate and absolute constraints
        original_base_rate = con.execute(
            f'SELECT AVG(CAST("{self.target}" AS DOUBLE)) FROM current_df'
        ).fetchone()[0] or 0.0
        abs_min_sample_size = self.min_sample_size
        abs_min_events = self.min_events
        abs_min_lift = self.min_lift

        logger.info(f"🔒 Locking Original Base Rate: {original_base_rate*100:.2f}%")

        for i in range(1, self.max_segments + 1):
            res = con.execute(
                f'SELECT AVG("{self.target}"), COUNT(*) FROM current_df'
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

            iv_ranking, precomputed_bins = self.compute_iv_ranking_and_bin(
                con, eligible_cols, columns_types
            )

            # --- Diagnostic snapshot ---
            # --- Diagnostic snapshot ---
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
                logger.warning("⚠️ All eligible features exhausted. Aborting.")
                break

            # Build binned table (ensure target array is standard ndarray)
            raw_target_arr = con.execute(
                f'SELECT "{self.target}" FROM current_df'
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
                logger.warning("⚠️ No valid binned variables found. Stopping.")
                break

            con.execute("DROP TABLE IF EXISTS binned_df")
            con.execute("CREATE TABLE binned_df AS SELECT * FROM binned_data")

            # --- Grid search over parameter configurations ---
            grid_candidates: List[Dict[str, Any]] = []
            # Build binned table (ensure target array is standard ndarray)
            raw_target_arr = con.execute(
                f'SELECT "{self.target}" FROM current_df'
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
                logger.warning("⚠️ No valid binned variables found. Stopping.")
                break

            con.execute("DROP TABLE IF EXISTS binned_df")
            con.execute("CREATE TABLE binned_df AS SELECT * FROM binned_data")

            # -------------------------------------------------------------------------
            # THE FIX: Hoist rule generation outside the grid loop to prevent duplicate 
            # SQL execution and redundant _expand_adjacent_bins calls.
            # -------------------------------------------------------------------------
            # 1. Determine global floor constraints across all experiments
            global_min_sample = min(exp["min_sample_size"] for exp in experiments)
            global_min_lift = min(exp["min_lift"] for exp in experiments)
            
            # Temporarily set instance variables for the SQL generation pass
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
                combos_2 = [c for c in combinations(valid_1way_vars, 2) if self.is_diverse(c)]
                if combos_2:
                    res_2 = self._agg_combinations(con, combos_2, original_base_rate)
                    if res_2:
                        valid_2way_sets = {frozenset(c["combo_vars"]) for c in res_2}
                        if self.enable_2way:
                            all_candidate_rules.extend(res_2)

            # Level 3 (Triplets)
            if self.enable_3way and len(valid_1way_vars) >= 3 and valid_2way_sets:
                combos_3 = [
                    c for c in combinations(valid_1way_vars, 3)
                    if self.is_diverse(c) and all(frozenset(p) in valid_2way_sets for p in combinations(c, 2))
                ]
                if combos_3:
                    res_3 = self._agg_combinations(con, combos_3, original_base_rate)
                    if res_3:
                        all_candidate_rules.extend(res_3)

            # 2. Grid search over parameter configurations (In-Memory Filtering)
            grid_candidates: List[Dict[str, Any]] = []
            for config in experiments:
                # Filter the globally generated rules against this config's strict thresholds
                valid_for_config = [
                    r for r in all_candidate_rules
                    if r["count"] >= config["min_sample_size"] and r["lift"] >= config["min_lift"]
                ]
                
                if valid_for_config:
                    # Sort by priority and take the top match for this grid config
                    valid_for_config.sort(key=lambda x: self._get_sort_key(x), reverse=True)
                    top_match = valid_for_config[0].copy()
                    top_match["grid_min_sample_size"] = config["min_sample_size"]
                    top_match["grid_min_lift"] = config["min_lift"]
                    grid_candidates.append(top_match)

            if not grid_candidates:
                logger.info("⏹️ No candidates cleared the grid. Stopping.")
                break

            # 3. Rank the final candidates out of the grid
            grid_candidates.sort(
                key=lambda x: self._get_sort_key(x), reverse=True
            )
            # ... Resume raw validation (selected_candidate loop) ...

            # Rank candidates by (lift, count, rate)
            grid_candidates.sort(
                key=lambda x: self._get_sort_key(x), reverse=True
            )

            selected_candidate = None
            for candidate in grid_candidates:
                rule_str = candidate["rule"]
                sql_filter = self.parse_rule_to_sql(rule_str)
                # Validate on raw current_df (residual)
                actual = con.execute(
                    f'SELECT COUNT(*) AS cnt, SUM(CAST("{self.target}" AS DOUBLE)) AS evt '
                    f'FROM current_df WHERE ({sql_filter})'
                ).fetchone()
                actual_cnt, actual_evt = actual[0], actual[1] or 0
                # Calculate actual lift for this candidate
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

            # Compute metrics from the residual counts
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

            # Store the raw rule and raw SQL (no exclusions)
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
            # ------------------------------------------------------------------
            # Clear comparison of top expanded candidates vs the final champion
            # ------------------------------------------------------------------
            # Collect the top expanded candidates that were available in this iteration
            expanded_in_this_iter = [
                r for r in all_candidate_rules
                if "base_events" in r          # only expanded rules carry this key
            ]
            if expanded_in_this_iter:
                # Sort them the same way the engine ranked everything
                expanded_in_this_iter.sort(key=lambda x: self._get_sort_key(x), reverse=True)
                top_exp = expanded_in_this_iter[:5]   # show top 5
        
                logger.info("📊 Top adjacent-merge candidates vs final champion")
                logger.info(
                    f"   {'Rank':<5} {'Type':<10} {'Lift':>6} {'Rate%':>7} {'Count':>8} {'Events':>8}  Rule"
                )
                logger.info("   " + "-" * 90)
        
                # First line = the actual winner
                logger.info(
                    f"   {'★':<5} {'CHAMPION':<10} "
                    f"{actual_lift:>6.2f}x {actual_rate:>6.1f}% "
                    f"{selected_candidate['actual_count']:>8} "
                    f"{selected_candidate['actual_events']:>8.0f}  "
                    f"{best_rule[:60]}"
                )
        
                for idx, e in enumerate(top_exp, 1):
                        # Decide why it lost (or would have won)
                        champ_key = self._get_sort_key({
                            "lift": actual_lift,
                            "rate": actual_rate,
                            "count": selected_candidate["actual_count"]
                        })
                        cand_key = self._get_sort_key(e)
            
                        if cand_key > champ_key:
                            reason = "would have beaten champion (but failed raw validation)"
                        else:
                            # Walk the sort priority tuple to find the first deciding dimension
                            priority_order = self.sort_priority.split("_")
                            champ_vals = {
                                "lift": actual_lift,
                                "rate": actual_rate,
                                "count": selected_candidate["actual_count"],
                            }
                            cand_vals = {
                                "lift": e["lift"],
                                "rate": e["rate"],
                                "count": e["count"],
                            }
                            reason = "ranked lower by sort_priority"  # fallback
                            for dim in priority_order:
                                if cand_vals[dim] < champ_vals[dim]:
                                    label = {
                                        "lift": "lower lift",
                                        "count": "smaller count",
                                        "rate": "lower rate",
                                    }[dim]
                                    reason = (
                                        f"{label} ({cand_vals[dim]:.2f} < {champ_vals[dim]:.2f})"
                                    )
                                    break
                                elif cand_vals[dim] > champ_vals[dim]:
                                    # candidate beats champ on this dim — next dim decided it
                                    break
            
                        logger.info(
                            f"   {idx:<5} {'expanded':<10} "
                            f"{e['lift']:>6.2f}x {e['rate']:>6.1f}% "
                            f"{e['count']:>8} {e['events']:>8.0f}  "
                            f"{e['rule'][:55]}  → {reason}"
                        )

            self.diagnostics_[-1]["winning_segment"] = {
                "rule": best_rule,
                "sql_filter": best_raw_sql,
                "variables_used": list(winning_combo),
                "lift": actual_lift,
                "count": int(selected_candidate["actual_count"]),
            }

            # Remove rows matching the raw rule from the residual
            # NULL‑safe: keep rows where the rule evaluates to NULL
            con.execute(
                f"""
                CREATE TABLE temp_residual AS
                SELECT * FROM current_df
                WHERE NOT ({best_raw_sql}) OR ({best_raw_sql}) IS NULL
            """
            )
            con.execute("DROP TABLE current_df")
            con.execute("ALTER TABLE temp_residual RENAME TO current_df")

        # Restore original config
        self.min_sample_size = abs_min_sample_size
        self.min_events = abs_min_events
        self.min_lift = abs_min_lift 
        con.close()
        logger.info("🏁 Extraction complete.")
        return self.segments

    def evaluate_final_coverage(self, original_data: Any) -> List[Dict[str, Any]]:
        """
        Evaluates the hierarchical segmentation on the original dataset.

        The rules are applied in the order they were extracted (first rule gets
        highest priority). This yields the true hierarchical segmentation.

        Args:
            original_data: The original (unfiltered) dataset.

        Returns:
            List of dictionaries with segment KPIs: segment, total_count,
            target_events, response_rate, base_response_rate, capture_rate, lift.
        """
        if not self.segments:
            return []

        logger.info("📊 Evaluating final hierarchical coverage on original data...")
        con = duckdb.connect(":memory:")
        con.execute("CREATE OR REPLACE TABLE original_df AS SELECT * FROM original_data")

        # Build CASE statement with raw SQL filters in order
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

        Args:
            feature_name: Name of the feature to trace.
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
            # Default fallback to 'iv' for backward compatibility on older states
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
        Generates a feature health report on the original dataset for an explicitly provided
        list of features using native DuckDB SQL naive binning (NTILE quantiles for numeric
        features and direct grouping for categoricals). Robustly handles string targets.

        Args:
            original_data: The original (unfiltered) dataset.
            features: List of feature column names to evaluate.

        Returns:
            Dictionary mapping each specified feature to its bin health details.
        """
        if not features:
            logger.warning("⚠️ No features provided for health report generation.")
            return {}

        # Deduplicate features while preserving order
        unique_features = list(dict.fromkeys(features))

        logger.info(
            f"📋 Generating DuckDB Naive Feature Health Report for {len(unique_features)} feature(s): "
            f"{unique_features}"
        )

        con = duckdb.connect(":memory:")
        con.execute("CREATE TABLE input_df AS SELECT * FROM original_data")

        cols_info = con.execute("DESCRIBE input_df").fetchall()
        columns_types = {row[0]: row[1] for row in cols_info}

        # Safe target conversion expression for DuckDB SQL (handles numeric, string 'Yes'/'No', 'True'/'False')
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
                # DuckDB SQL Quantile binning via NTILE()
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
                # Direct SQL grouping for categorical values
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
