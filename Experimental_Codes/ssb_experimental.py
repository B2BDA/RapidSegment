"""Strategic Segmentation & Scorecard Engine (SSB.py).

This module provides a high-performance, industrial-grade framework for combinatorial 
heuristic segmentation and financial scorecard development. It minimizes memory 
footprints and engineering bottlenecks by migrating heavy row-wise operations, 
multi-dimensional groupings, and matrix transformations away from Pandas into 
highly vectorized multi-threaded environments powered by DuckDB and NumPy BLAS execution.

Author: Bishwarup Biswas + Gemini
Python Version: 3.9+
"""

import json
import logging
import multiprocessing
import re
import itertools
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple, Union

import duckdb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from optbinning import OptimalBinning

# Configure Production Module Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [%(filename)s:%(lineno)d] | %(message)s",
)
logger = logging.getLogger("StrategicEngine")

# Pre-compile regex at module load for O(1) performance during string parsing loops
_BRACKET_REGEX = re.compile(r"\[(.*)\]")


class StrategicSegmentBuilder:
    """Extracts mutually exclusive, predictive segments from tabular data.

    Utilizes Optimal Binning to discretize continuous features into monotonic
    Information Value (IV) bins, applying an Apriori-style combinatorial prune
    to surface multi-way rules meeting defined Lift and Volume thresholds.

    Attributes:
        target (str): Dependent binary target column name (1 = Event, 0 = Non-Event).
        n_jobs (int): Number of CPU cores allocated to parallelized search jobs.
        min_sample_size (int): Absolute minimum row count required for a valid rule.
        min_lift (float): Minimum lift cutoff (Segment Rate / Population Base Rate).
        top_n_vars (int): Number of highest-IV features passed into the Apriori engine.
        max_segments (int): Hard stopping ceiling for extracted mutually exclusive segments.
        max_feature_reuse (int): Structural limit for tracking and restricting feature dominance.
        enable_diversity (bool): If True, blocks rules combining variables from the same business group.
        enable_1way (bool): Allow 1-dimensional rules in final pool.
        enable_2way (bool): Allow 2-dimensional intersection rules in final pool.
        enable_3way (bool): Allow 3-dimensional intersection rules in final pool.
        feature_groups (Dict[str, List[str]]): Mapping of business categories to columns.
        ignore_features (List[str]): Explicit list of columns to drop prior to IV calculation.
        feature_usage_counts (Dict[str, int]): State tracker tracking feature allocation rules.
        segments (List[Dict[str, Any]]): Captured operational metadata for extracted segments.
    """

    def __init__(
        self,
        target: str,
        n_jobs: int = -1,
        min_sample_size: int = 1000,
        min_lift: float = 2.0,
        top_n_vars: int = 20,
        max_segments: int = 10,
        max_feature_reuse: int = 1,
        enable_diversity: bool = False,
        enable_1way: bool = True,
        enable_2way: bool = True,
        enable_3way: bool = True,
        feature_groups: Optional[Dict[str, List[str]]] = None,
        ignore_features: Optional[List[str]] = None,
    ) -> None:
        """Initializes the Segment Builder engine with performance safety bounds."""
        self.target = target
        self.n_jobs = (
            n_jobs if n_jobs != -1 else max(1, multiprocessing.cpu_count() - 1)
        )
        self.min_sample_size = min_sample_size
        self.min_lift = min_lift
        self.top_n_vars = top_n_vars
        self.max_segments = max_segments
        self.max_feature_reuse = max_feature_reuse
        self.segments: List[Dict[str, Any]] = []

        self.enable_diversity = enable_diversity
        self.enable_1way = enable_1way
        self.enable_2way = enable_2way
        self.enable_3way = enable_3way
        self.feature_groups = feature_groups or {}
        self.ignore_features = ignore_features or []
        self.feature_usage_counts: Dict[str, int] = {}

    @staticmethod
    def _resolve_optb_dtype(series: pd.Series) -> str:
        """Determines the correct OptBinning data type flag for a Pandas Series.

        Args:
            series (pd.Series): Target input variable array.

        Returns:
            str: 'categorical' or 'numerical' dtype string mapping indicator.
        """
        if str(series.dtype) in ["object", "category", "string", "str"]:
            return "categorical"
        return "numerical"

    @staticmethod
    def _is_numeric_string(val: str) -> bool:
        """Safely evaluates if a raw string represents a float/int.

        Args:
            val (str): Element value string to parse.

        Returns:
            bool: True if parsing succeeds, otherwise False.
        """
        try:
            float(val)
            return True
        except ValueError:
            return False

    def _validate_feature_groups(self, df: pd.DataFrame) -> None:
        """Validates that all declared feature group variables exist in the schema.

        Args:
            df (pd.DataFrame): Training or reference pool DataFrame.

        Raises:
            ValueError: If a mapped variable is missing from the DataFrame schema.
        """
        if not self.feature_groups:
            return

        active_cols = set(df.columns) - {self.target} - set(self.ignore_features)
        validated_count = 0

        for group, vars_list in self.feature_groups.items():
            for var in vars_list:
                if var not in active_cols:
                    raise ValueError(
                        f"Schema Mismatch: Feature '{var}' declared in group '{group}' "
                        "was not found in the provided DataFrame."
                    )
                validated_count += 1

        logger.info(f"Feature group validation passed. ({validated_count} features mapped)")

    def get_group(self, var: str) -> str:
        """Returns the assigned business category group name for a given feature.

        Args:
            var (str): Column name string.

        Returns:
            str: Assigned group key, or the variable name itself if unmapped.
        """
        for group, vars_list in self.feature_groups.items():
            if var in vars_list:
                return group
        return var

    def is_diverse(self, combo: Tuple[str, ...]) -> bool:
        """Ensures a tuple of features spans strictly distinct analytical groups.

        Args:
            combo (Tuple[str, ...]): Candidate feature combination set.

        Returns:
            bool: True if combinations are structurally diverse or constraint is disabled.
        """
        if not self.enable_diversity:
            return True
        groups = [self.get_group(v) for v in combo]
        return len(groups) == len(set(groups))

    def compute_iv_ranking(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates Information Value (IV) using optimized NumPy array passing.

        Extracts underlying numpy ndarrays before serialization to prevent heavy 
        Joblib pickling and system overhead overhead.

        Args:
            df (pd.DataFrame): Target transaction or scorecard dataframe.

        Returns:
            pd.DataFrame: Ranked dataframe sorting features by structural IV descending.
        """
        target_vals = df[self.target].values

        def _worker(col: str, col_vals: np.ndarray, dtype: str) -> Dict[str, Union[str, float]]:
            try:
                optb = OptimalBinning(name=col, dtype=dtype)
                optb.fit(col_vals, target_vals)
                iv_val = optb.binning_table.build().IV.iloc[-1]
                return {"variable": col, "iv": float(iv_val) * 100}
            except Exception as e:
                logger.debug(f"IV computation failed for {col}: {e}")
                return {"variable": col, "iv": 0.0}

        eligible_cols = [c for c in df.columns if c != self.target and c not in self.ignore_features]
        
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_worker)(col, df[col].values, self._resolve_optb_dtype(df[col])) 
            for col in eligible_cols
        )

        return (
            pd.DataFrame(results)
            .sort_values("iv", ascending=False)
            .reset_index(drop=True)
        )

    def create_binned_df(self, df: pd.DataFrame, variables: List[str]) -> pd.DataFrame:
        """Transforms continuous data into discrete optimal binned strings.

        Args:
            df (pd.DataFrame): Source raw tabular dataframe.
            variables (List[str]): List of column names to discretize.

        Returns:
            pd.DataFrame: Transformed pandas dataframe consisting of optimized categorical bins.
        """
        binned_df = pd.DataFrame(index=df.index)

        for col in variables:
            dtype = self._resolve_optb_dtype(df[col])
            optb = OptimalBinning(name=col, dtype=dtype)
            optb.fit(df[col].values, df[self.target].values)

            transformed_bins = optb.transform(df[col].values, metric="bins")
            binned_df[col] = pd.Categorical(transformed_bins)

        binned_df[self.target] = df[self.target].values
        return binned_df

    def _agg_combinations(
        self,
        binned_df: pd.DataFrame,
        combo_list: List[Tuple[str, ...]],
        base_rate: float,
    ) -> pd.DataFrame:
        """High-speed native C++ aggregation replacing Pandas groupby using DuckDB UNION ALL sets.

        Bypasses python loop constraints and row-wise Pandas concatenation functions by 
        delegating calculations directly to DuckDB engine.

        Args:
            binned_df (pd.DataFrame): Dataframe consisting of optimal binned columns.
            combo_list (List[Tuple[str, ...]]): Feature combination tuples generated via Apriori layer.
            base_rate (float): Base global population performance rate.

        Returns:
            pd.DataFrame: Candidate segments tracking rules, counts, rates, and historical lifts.
        """
        if not combo_list:
            return pd.DataFrame()

        con = duckdb.connect()
        con.register('binned_df', binned_df)
        
        batch_size = 150
        results = []

        for i in range(0, len(combo_list), batch_size):
            batch = combo_list[i:i + batch_size]
            union_queries = []
            
            for combo in batch:
                rule_sql = " || ' & ' || ".join([f"'{c}=' || CAST(\"{c}\" AS VARCHAR)" for c in combo])
                group_cols = ", ".join([f'"{c}"' for c in combo])
                combo_array_str = ", ".join([f"'{c}'" for c in combo])
                
                query = f"""
                SELECT 
                    {rule_sql} AS rule,
                    COUNT(*) AS count,
                    SUM("{self.target}") AS events,
                    ARRAY[{combo_array_str}] AS combo_vars
                FROM binned_df 
                GROUP BY {group_cols}
                """
                union_queries.append(query)

            full_query = " UNION ALL ".join(union_queries)
            batch_res = con.execute(full_query).df()
            results.append(batch_res)
            
        con.close()

        if not results:
            return pd.DataFrame()

        summary = pd.concat(results, ignore_index=True)
        summary = summary[summary["count"] >= self.min_sample_size]
        
        if summary.empty:
            return pd.DataFrame()

        # Vectorized calculations via NumPy array extraction
        summary["rate"] = (summary["events"].values / summary["count"].values) * 100.0
        summary["lift"] = summary["rate"].values / (base_rate * 100.0)

        summary = summary[summary["lift"] >= self.min_lift]
        
        if summary.empty:
            return pd.DataFrame()

        summary["combo_vars"] = summary["combo_vars"].apply(tuple)
        return summary[["rule", "count", "rate", "lift", "combo_vars"]]

    def parse_rule_to_sql(self, rule_str: str) -> str:
        """Translates OptBinning string syntax into a production SQL WHERE clause.

        Args:
            rule_str (str): Rule criteria string extracted from the combinatorial engine.

        Returns:
            str: Formatted production ANSI SQL string condition block.
        """
        parts = [p.strip() for p in rule_str.split("&")]
        sql_conditions: List[str] = []

        for part in parts:
            if "=" not in part:
                continue

            col, interval = [x.strip() for x in part.split("=", 1)]
            bracket_match = _BRACKET_REGEX.search(interval)

            is_categorical = False
            if bracket_match:
                content = bracket_match.group(1)
                if any(
                    k in interval for k in ("'", '"', "Array", "Categorical")
                ) or not interval.startswith(("[", "(")):
                    is_categorical = True
                elif len(content.split(",")) > 2:
                    is_categorical = True

            if is_categorical and bracket_match:
                raw_items = [
                    i.strip().strip("'").strip('"')
                    for i in bracket_match.group(1).split(",")
                    if i.strip()
                ]
                formatted_items = ", ".join(
                    [
                        item if self._is_numeric_string(item) else f"'{item}'"
                        for item in raw_items
                    ]
                )
                sql_conditions.append(f"{col} IN ({formatted_items})")
                continue

            if interval in ["Special", "Missing"]:
                sql_conditions.append(f"{col} IS NULL")
                continue

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

        return " AND ".join(
            f"({cond})" if "AND" in cond else cond for cond in sql_conditions
        )

    def extract_segments(self, df: pd.DataFrame, param_grid: Optional[Dict[str, List[Any]]] = None) -> pd.DataFrame:
        """Sequentially extracts high-lift segments using an iterative Multi-Threshold Grid Search.

        Employs structural tracking mechanisms to prevent single-feature model dominance
        and uses native DuckDB filtering arrays for updating iteration states.

        Args:
            df (pd.DataFrame): Training collection dataframe pool.
            param_grid (Optional[Dict[str, List[Any]]]): Matrix settings dictionary tracking hyperparameters.

        Returns:
            pd.DataFrame: Extracted operational segment metadata table.
        """
        if self.enable_diversity:
            self._validate_feature_groups(df)

        current_df = df.copy()
        
        eligible_cols = [c for c in df.columns if c != self.target and c not in self.ignore_features]
        self.feature_usage_counts = {col: 0 for col in eligible_cols}
        
        if param_grid:
            logger.info(
                f"Dynamic Grid Search Enabled: {len(param_grid.get('min_sample_size', [self.min_sample_size])) * len(param_grid.get('min_lift', [self.min_lift]))} total configurations."
            )
            sizes = param_grid.get("min_sample_size", [self.min_sample_size])
            lifts = param_grid.get("min_lift", [self.min_lift])
            experiments = [
                {"min_sample_size": s, "min_lift": l}
                for s, l in itertools.product(sizes, lifts)
            ]
        else:
            experiments = [{"min_sample_size": self.min_sample_size, "min_lift": self.min_lift}]

        for i in range(1, self.max_segments + 1):
            base_rate = current_df[self.target].mean()
            min_floor_volume = min(exp["min_sample_size"] for exp in experiments)
            
            if base_rate == 0 or len(current_df) < min_floor_volume:
                break

            logger.info(
                f"Iteration {i} | Remaining Volume: {len(current_df):,} | Base Rate: {base_rate*100:.2f}%"
            )

            iv_ranking = self.compute_iv_ranking(current_df)
            
            allowed_vars = [
                row["variable"] for _, row in iv_ranking.iterrows()
                if self.feature_usage_counts.get(row["variable"], 0) < self.max_feature_reuse
            ]
            
            top_vars = allowed_vars[:self.top_n_vars]
            if not top_vars:
                logger.warning("All eligible features have been exhausted via max_feature_reuse filters. Aborting.")
                break

            binned_df = self.create_binned_df(current_df, top_vars)
            valid_vars = [v for v in top_vars if binned_df[v].nunique() > 1]
            
            grid_candidates: List[pd.Series] = []

            for config in experiments:
                self.min_sample_size = config["min_sample_size"]
                self.min_lift = config["min_lift"]

                all_rules: List[pd.DataFrame] = []

                res_1 = self._agg_combinations(binned_df, [(c,) for c in valid_vars], base_rate)
                valid_1way_vars = set()

                if not res_1.empty:
                    valid_1way_vars = {c[0] for c in res_1["combo_vars"]}
                    if self.enable_1way:
                        all_rules.append(res_1)

                if not valid_1way_vars:
                    continue

                valid_2way_sets = set()
                if len(valid_1way_vars) >= 2 and (self.enable_2way or self.enable_3way):
                    combos_2 = [
                        c for c in combinations(valid_1way_vars, 2) if self.is_diverse(c)
                    ]
                    if combos_2:
                        res_2 = self._agg_combinations(binned_df, combos_2, base_rate)
                        if not res_2.empty:
                            valid_2way_sets = {frozenset(c) for c in res_2["combo_vars"]}
                            if self.enable_2way:
                                all_rules.append(res_2)

                if self.enable_3way and len(valid_1way_vars) >= 3 and valid_2way_sets:
                    combos_3 = [
                        c
                        for c in combinations(valid_1way_vars, 3)
                        if self.is_diverse(c)
                        and all(
                            frozenset(p) in valid_2way_sets for p in combinations(c, 2)
                        )
                    ]
                    if combos_3:
                        res_3 = self._agg_combinations(binned_df, combos_3, base_rate)
                        if not res_3.empty:
                            all_rules.append(res_3)

                if all_rules:
                    shortlisted_config = (
                        pd.concat(all_rules, ignore_index=True)
                        .sort_values(["lift", "rate", "count"], ascending=False)
                        .reset_index(drop=True)
                    )
                    top_match = shortlisted_config.iloc[0].copy()
                    top_match["grid_min_sample_size"] = config["min_sample_size"]
                    top_match["grid_min_lift"] = config["min_lift"]
                    grid_candidates.append(top_match)

            if not grid_candidates:
                logger.info("No active candidates cleared criteria pool across grid variations. Stopping.")
                break

            grid_results = pd.DataFrame(grid_candidates).sort_values(
                ["lift", "count", "rate"], ascending=False
            ).reset_index(drop=True)

            best_match = grid_results.iloc[0]
            best_rule = best_match["rule"]
            best_sql = self.parse_rule_to_sql(best_rule)
            winning_combo = best_match["combo_vars"]

            for var in winning_combo:
                self.feature_usage_counts[var] = self.feature_usage_counts.get(var, 0) + 1
                logger.info(f"Feature Usage Tracker Update -> '{var}' used count = {self.feature_usage_counts[var]}")

            self.segments.append(
                {
                    "segment_id": i,
                    "rule_string": best_rule,
                    "sql_filter": best_sql,
                    "count": int(best_match["count"]),
                    "rate": float(best_match["rate"]),
                    "lift": float(best_match["lift"]),
                    "meta_applied_sample_size": int(best_match["grid_min_sample_size"]),
                    "meta_applied_min_lift": float(best_match["grid_min_lift"])
                }
            )

            logger.info(f"Segment {i} Captured (Size Floor: {best_match['grid_min_sample_size']} | Lift Floor: {best_match['grid_min_lift']}): {best_sql}")
            
            # Maintain residual tracking inside native DuckDB context
            current_df = duckdb.query(
                f"SELECT * FROM current_df WHERE NOT ({best_sql})"
            ).df()

        return pd.DataFrame(self.segments)

    def evaluate_final_coverage(self, original_df: pd.DataFrame) -> pd.DataFrame:
        """Executes a full CASE WHEN query over the source dataset to map mutually exclusive coverage.

        Args:
            original_df (pd.DataFrame): Non-mutated base validation baseline matrix.

        Returns:
            pd.DataFrame: Summary portfolio analytics mapping volumes, event tracking rates and final lifts.
        """
        if not self.segments:
            return pd.DataFrame()

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
                SUM("{self.target}") AS target_events,
                (SUM("{self.target}") * 100.0 / COUNT(*)) AS response_rate
            FROM original_df
            GROUP BY 1
        ),
        BASE_KPIS AS (
            SELECT *,
                SUM(total_count) OVER() AS total_population,
                (SUM(target_events) OVER() * 1.0 / SUM(total_count) OVER()) * 100 AS base_response_rate 
            FROM PER_SEG_KPIS
        )
        SELECT 
            PER_SEG_KPIS.*, 
            BASE_KPIS.base_response_rate,
            (PER_SEG_KPIS.total_count * 1.0 / BASE_KPIS.total_population) * 100 AS capture_rate,
            (PER_SEG_KPIS.response_rate / BASE_KPIS.base_response_rate) AS lift
        FROM PER_SEG_KPIS
        LEFT JOIN BASE_KPIS ON PER_SEG_KPIS.segment = BASE_KPIS.segment
        ORDER BY segment
        """
        return duckdb.query(final_query).df()


class StrategicSegmentScore:
    """High-Throughput Vectorized Scorecard Engine.

    Computes segment weights via Harmonic Mean and applies dot-product deciling 
    over large datasets using optimized DuckDB aggregations and NumPy BLAS operations.

    Attributes:
        target_col (str): Target dependent variable representation column name.
        primary_key (str): Row level indexing hash or tracking sequence column label.
        segment_cols (List[str]): Extracted target categorical segment features.
        model_artifact (Dict[str, Any]): Production tracking schema container exporting parameters.
    """

    def __init__(self, target_col: str, primary_key: str, segment_cols: List[str]) -> None:
        """Initializes the scoring class wrapper."""
        self.target_col = target_col
        self.primary_key = primary_key
        self.segment_cols = segment_cols
        self.model_artifact: Dict[str, Any] = {}

    def calculate_and_export_weights(
        self, df: pd.DataFrame, export_path: str = "scorecard_model.json"
    ) -> Dict[str, Any]:
        """Calculates harmonic weights and derives decile boundaries via vectorized execution.

        Args:
            df (pd.DataFrame): Inputs data profile training frame.
            export_path (str): Target filesystem serialization destination folder path.

        Returns:
            Dict[str, Any]: Production tracking artifact representation mappings.

        Raises:
            RuntimeError: Database driver failures or transaction disruptions.
            ValueError: Structural validation tracking parameters metrics anomalies.
        """
        logger.info(f"Initializing DuckDB scorecard engine for {len(df):,} records...")

        ctx = duckdb.connect()

        # Step 1: Baseline metrics + Vectorized multi-segment aggregation (O(1) Scan)
        agg_expressions = [
            f'COUNT(CASE WHEN "{col}" = 1 THEN 1 END) AS "{col}_cnt", '
            f'SUM(CASE WHEN "{col}" = 1 THEN "{self.target_col}" ELSE 0 END) AS "{col}_ev"'
            for col in self.segment_cols
        ]

        master_sql = f"""
            SELECT 
                COUNT(*) AS total_pop, 
                SUM("{self.target_col}") AS total_ev,
                {', '.join(agg_expressions)}
            FROM df
        """

        master_res = ctx.execute(master_sql).fetchone()
        if not master_res:
            raise RuntimeError("Database engine failed to return aggregations.")

        total_population, total_events = master_res[0], master_res[1]

        if total_population == 0 or total_events == 0:
            raise ValueError(
                "Invalid Dataset: Population and total events must be greater than zero."
            )

        baseline_rate = total_events / total_population
        zero_inflation_rate = 1.0 - baseline_rate

        # Step 2: Unpack vectorized SQL aggregations into weight lookup
        logger.info("Unpacking calculations into harmonic weights...")
        weights_lookup: Dict[str, Dict[str, Union[int, float]]] = {}

        for idx, seg_col in enumerate(self.segment_cols):
            seg_count = master_res[2 + (idx * 2)] or 0
            seg_events = master_res[2 + (idx * 2) + 1] or 0

            if seg_count == 0 or seg_events == 0:
                logger.warning(f"Segment '{seg_col}' has zero volume. Setting weight=0.")
                weights_lookup[seg_col] = {
                    "weight": 0,
                    "lift": 0.0,
                    "response_rate": 0.0,
                    "capture_rate": 0.0,
                }
                continue

            response_rate = seg_events / seg_count
            capture_rate = seg_events / total_events
            lift = response_rate / baseline_rate

            harmonic_mean = 2 * ((response_rate * capture_rate) / (response_rate + capture_rate))
            raw_weight = lift * harmonic_mean * 100.0

            weights_lookup[seg_col] = {
                "weight": int(np.round(raw_weight)),
                "lift": round(lift, 4),
                "response_rate": round(response_rate, 4),
                "capture_rate": round(capture_rate, 4),
            }

        # Step 3: BLAS Matrix Dot-Product Scoring (Extracted numpy views bypass pandas parsing rules)
        logger.info("Scoring training dataset via NumPy Linear Algebra engine...")
        scored_cols = list(weights_lookup.keys())
        weights_vector = np.array(
            [weights_lookup[c]["weight"] for c in scored_cols], dtype=np.float64
        )

        # Runs matrix dot-products at optimized raw C speeds
        train_scores = df[scored_cols].to_numpy(dtype=np.float64) @ weights_vector

        logger.info(f"Dataset Zero-Inflation Rate: {zero_inflation_rate:.2%}")

        # NumPy boolean slicing ignores zero tracking overhead safely
        if zero_inflation_rate >= 0.80:
            logger.info("High Zero-Inflation (>=80%). Isolating Active Population...")
            active_scores = train_scores[train_scores > 0]
        else:
            logger.info("Normal Distribution (<80%). Deciling across full dataset...")
            active_scores = train_scores

        if len(active_scores) == 0:
            raise ValueError("Scorecard Failure: 0 customers triggered any segment rules.")

        # Step 4: High-speed NumPy sorting over Pandas Series operations
        logger.info(f"Calibrating deciles across {len(active_scores):,} target customers...")
        sorted_scores = np.sort(active_scores)[::-1]
        active_pop_size = len(sorted_scores)

        decile_thresholds: Dict[str, int] = {}
        for d in range(1, 11):
            row_idx = int((d / 10.0) * active_pop_size) - 1
            row_idx = max(0, min(active_pop_size - 1, row_idx))
            decile_thresholds[str(d)] = int(sorted_scores[row_idx].item())

        self.model_artifact = {
            "model_metadata": {
                "total_training_population": int(total_population),
                "active_scored_population": int(active_pop_size),
                "active_population_pct": round((active_pop_size / total_population) * 100.0, 2),
                "baseline_event_rate": round(baseline_rate, 4),
            },
            "segment_weights": weights_lookup,
            "decile_min_thresholds": decile_thresholds,
        }

        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(self.model_artifact, f, indent=4)

        return self.model_artifact
