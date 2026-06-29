# Strategic Segmentation & Scorecard Engine (SSE)

A high-performance combinatorial heuristic engine and vectorized scoring framework for extracting highly predictive, mutually exclusive sub-populations from tabular data using Optimal Binning, Apriori pruning, and DuckDB linear algebra.

---

## 1. Executive Summary

### WHAT is it?
The **Strategic Segmentation & Scorecard Engine (SSE)** is an automated rule-induction and customer-deciling library. It searches through massive feature spaces to discover statistically robust sub-populations, extracts transparent multidimensional SQL rules, and calibrates high-throughput production scorecards. Instead of black-box predictions, SSE surfaces human-readable logic:

`utilization_avg_3m >= 0.84 AND max_dpd_12m >= 36.09 AND risk_segment IN ('HighRisk')`

### WHY do we need it?
In credit risk, marketing analytics, and financial crime, identifying concentrated risk pockets or high-value cohorts traditionally demands tedious manual slicing, subjective binning, or brittle decision trees.
* **The Manual Exploration Pitfall:** Analysts frequently miss complex non-linear intersections of three or more variables, or accidentally draft overlapping rules where a single customer qualifies for multiple conflicting treatments.
* **The Machine Learning Pitfall:** Black-box models (XGBoost, LightGBM) capture these non-linear patterns effortlessly, but they cannot be natively exported into transparent, deterministic SQL WHERE clauses required by legacy credit-decisioning engines and core banking frameworks.
* **The Big Data Pitfall:** Iterating over millions of records to compute multi-column permutations or scoring dynamic customer portfolios creates massive memory bottlenecks in standard Pandas workflows.
* **The Solution:** SSE bridges this gap by pairing algorithmic optimal binning with an **O(1) DuckDB execution engine** and **NumPy BLAS matrix operations**, delivering pure SQL transparency at compiled C speeds.

### HOW does it work?
The framework operates across two dedicated pipelines:

1. **Strategic Segment Builder (`StrategicSegmentBuilder`):**
   * **Rank Significance:** Evaluates monotonic Information Value (IV) to rank top predictors.
   * **Discretize:** Groups continuous metrics into optimal intervals via `OptimalBinning`.
   * **Combinatorial Prune:** Evaluates feature pairs and triplets using an Apriori heuristic to isolate massive predictive spikes (Lift).
   * **Extract & Deduplicate:** Translates the winning candidate into production SQL and executes an in-memory elimination query to guarantee 100% mutual exclusivity across sequential segments.

2. **Vectorized Scorecard Engine (`StrategicSegmentScore`):**
   * **Harmonic Weighting:** Aggregates segment volumes in a single database pass and assigns mathematical weights balancing segment Response Rate and overall Capture Rate.
   * **BLAS Scoring:** Converts customer rules into a binary matrix and executes compiled dot-product scoring across millions of rows instantly.
   * **Zero-Inflation Routing:** Automatically detects zero-inflated target distributions (>= 80% non-events) to calibrate decile boundaries strictly across active populations.

---

## 2. Statistical Foundation: WOE & IV

Before evaluating multidimensional intersections, the engine evaluates individual continuous and categorical features using two foundational credit-scoring metrics: **Weight of Evidence (WOE)** and **Information Value (IV)**.

### Mathematical Formulas

$$WOE = \\ln\\left(\\frac{\\text{\\% of Non-Events}}{\\text{\\% of Events}}\\right)$$

$$IV = \\sum \\left(\\text{\\% of Non-Events} - \\text{\\% of Events}\\right) \\times WOE$$

### Step-by-Step Working Example
Trace how the engine evaluates a continuous metric, **`utilization_avg_3m`**, against a binary outcome (`1 = Default`, `0 = Good`).

#### 1. Raw Distribution Data
Given a baseline population of **10,000 customers** (8,000 Goods / 2,000 Defaults), optimal discretization cuts the variable into three monotonic bins:

| Bin (Utilization Range) | Non-Events (Goods) | Events (Defaults) | Total Volume |
| :--- | :---: | :---: | :---: |
| **Low** `[0.00, 0.40)` | 5,000 | 300 | 5,300 |
| **Mid** `[0.40, 0.84)` | 2,500 | 700 | 3,200 |
| **High** `[0.84, inf)` | 500 | 1,000 | 1,500 |
| **Total Baseline** | **8,000** | **2,000** | **10,000** |

#### 2. Distribution Percentages & WOE Derivation
* **Low Bin [0.00, 0.40) Non-Event Rate:** 5,000 / 8,000 = **62.5%**
* **Low Bin [0.00, 0.40) Event Rate:** 300 / 2,000 = **15.0%**
* **Low Bin WOE:** $\\ln(0.625 / 0.150) = \\mathbf{+1.427}$ *(Positive value indicates lower credit risk)*
* **High Bin [0.84, inf) Non-Event Rate:** 500 / 8,000 = **6.25%**
* **High Bin [0.84, inf) Event Rate:** 1,000 / 2,000 = **50.0%**
* **High Bin WOE:** $\\ln(0.0625 / 0.500) = \\mathbf{-2.079}$ *(Negative value indicates elevated credit risk)*

#### 3. Information Value (IV) Contribution
* **Low Bin IV Contribution:** $(0.625 - 0.150) \\times 1.427 = \\mathbf{0.678}$
* **High Bin IV Contribution:** $(0.0625 - 0.500) \\times (-2.079) = \\mathbf{0.910}$

Summing contributions across all internal bins yields the total variable IV. Features exceeding an IV benchmark of **0.30** are flagged as strong predictors and prioritized during combinatorial search.

---

## 3. Algorithmic & Mathematical Architecture

### Apriori Combinatorial Pruning
Evaluating multidimensional intersections across dozens of variables creates an immediate combinatorial explosion. A standard 3-way grid search across 20 features requires evaluating **1,350 unique aggregations** per iteration. 

SSE bypasses brute force by implementing **Apriori itemset pruning**: *If an individual feature fails to clear volume and lift benchmarks, any higher-order n-way intersection containing that feature is mathematically guaranteed to fail.*

```text
[Top 20 Features] ──(Level 1 Prune)──► [6 Surviving Base Vars]
                                                │
[15 Candidate Pairs] ◄──(Level 2 Pairwise Combinations)─┘
         │
  (Level 2 Prune)
         ▼
[4 Surviving Pairs] ──(Level 3 Triplet Synthesis)──► [2 Viable Triplets]
```
# Scorecard Harmonic Weighting

When scoring customers across discovered segments, relying purely on segment Lift ignores cohort volume. A micro-segment of 50 people might have a Lift of 8.0, while a core segment of 5,000 people has a Lift of 3.0.To prevent overfitting to micro-pockets, StrategicSegmentScore calculates a balanced Harmonic Mean Weight:

$$HarmonicWeight = Lift \times \left(2 \times \frac{ResponseRate \times CaptureRate}{ResponseRate + CaptureRate}\right) \times 100$$

## High-Throughput Vectorized Execution
Standard scoring loops iterate row-by-row or column-by-column in Python, creating severe latency. SSE implements two hardware-level optimizations:
* Single-Pass O(1) Database Scans: Instead of dispatching separate SQL queries to aggregate every segment, DuckDB compiles dynamic conditional statements (COUNT(CASE WHEN seg_1 = 1...)) into a single master execution tree. The data pages are read from disk/memory exactly once.
* BLAS Linear Algebra Scoring: Customer segment flags are cast into a contiguous 64-bit floating-point matrix
  ($\mathbf{X}$).
  Weights are aligned into a vector ($\mathbf{w}$). Scoring executes via native C/Fortran dot-product operations:
  $$\mathbf{Scores} = \mathbf{X} \cdot \mathbf{w}$$

## 4. System Architecture & Process Flow
```text
 [Raw Tabular Input Data]
                             │
                             ▼
              ┌──────────────────────────────┐
              │  Rank Features via Monotonic │
              │    Information Value (IV)    │
              └──────────────┬───────────────┘
                             ▼
              ┌──────────────────────────────┐
              │    Apriori Rule Induction    │
              │  (1-Way ► 2-Way ► 3-Way Search)│
              └──────────────┬───────────────┘
                             ▼
              ┌──────────────────────────────┐
              │ Parse Winning Interval Rule  │
              │   to Production SQL WHERE    │
              └──────────────┬───────────────┘
                             ▼
              ┌──────────────────────────────┐
              │ Deduplicate Data via DuckDB  │◄── Ensures 100% Mutual Exclusivity
              │  NOT (winning_sql_filter)    │
              └──────────────┬───────────────┘
                             │
               [Repeat Until Max Segments Met]
                             │
                             ▼
              ┌──────────────────────────────┐
              │ Compile Binary Segment Matrix│
              └──────────────┬───────────────┘
                             ▼
              ┌──────────────────────────────┐
              │ BLAS Matrix Dot-Product Score│◄── Evaluates Millions of Rows/Sec
              └──────────────┬───────────────┘
                             ▼
              ┌──────────────────────────────┐
              │ Zero-Inflation Decile Routing│◄── Calibrates Active Portfolios
              └──────────────┬───────────────┘
                             ▼
               [Export scorecard_model.json]
```
# 5. Class Reference & Parameter Guide

 `StrategicSegmentBuilder`
      
ParameterTypeDefaultDescriptiontargetstrRequiredBinary dependent variable column name (1 = Event, 0 = Non-Event).n_jobsint-1CPU cores allocated for parallel joblib search runs (-1 uses all available cores minus 1).min_sample_sizeint1000Absolute minimum volume required for an induced segment rule to be valid.min_liftfloat2.0Minimum predictive lift benchmark ($\text{Segment Rate} / \text{Base Rate}$).top_n_varsint20Ceiling for top-ranked IV features passed into the Apriori combinatorial search.max_segmentsint10Hard stopping threshold for sequentially extracted mutually exclusive segments.enable_diversityboolFalseBlocks multidimensional rules combining features from the same declared domain group.enable_1wayboolTrueInclude single-variable interval rules in candidate extraction pool.enable_2wayboolTrueInclude two-variable intersection rules in candidate extraction pool.enable_3wayboolTrueInclude three-variable intersection rules in candidate extraction pool.feature_groupsdict{}Domain mapping dictionary (e.g., {'liquidity': ['bal_1m'], 'risk': ['dpd_3m']}).ignore_featureslist[]Explicit list of metadata or identifier columns to exclude from IV calculations.

`StrategicSegmentScore`

Parameter / Method	Type	Description
target_col	str	Binary target variable column name.
primary_key	str	Unique entity identifier column (e.g., customer_id).
segment_cols	list	List of binary indicator columns representing induced segment membership.
calculate_and_export_weights(df, export_path)	Method	Derives harmonic weights, calculates vectorized BLAS scores, calibrates deciles, and writes JSON configuration.

# 6. Quick Start Guide

pip install strategic-segment-engine (coming soon ....)

# ## Example 1: Extracting Mutually Exclusive Segments
import pandas as pd
from strategic_segment_engine import StrategicSegmentBuilder

# 1. Load historical analytical data
df = pd.read_csv("enterprise_credit_risk.csv")

# 2. Define analytical feature groups to enforce rule diversity
domain_groups = {
    "delinquency": ["dpd_avg_3m", "dpd_avg_6m", "max_dpd_12m"],
    "liquidity": ["avg_bal_1m", "avg_bal_3m", "utilization_ratio"],
    "velocity": ["txn_cnt_1m", "spend_velocity_3m"]
}

# 3. Instantiate Builder
builder = StrategicSegmentBuilder(
    target="default_flag",
    min_sample_size=1500,
    min_lift=2.2,
    top_n_vars=15,
    max_segments=5,
    enable_diversity=True,
    feature_groups=domain_groups,
    ignore_features=["customer_id", "origination_date"]
)

# 4. Run automated combinatorial pruning
induced_segments = builder.extract_segments(df)

# 5. Inspect generated production SQL logic
for idx, row in induced_segments.iterrows():
    print(f"Segment {row['segment_id']} [Lift: {row['lift']:.2f}x | Vol: {row['count']}]")
    print(f"SQL WHERE: {row['sql_filter']}\n")

# 6. Generate full database coverage audit
coverage_report = builder.evaluate_final_coverage(df)
print(coverage_report[["segment", "total_count", "response_rate", "capture_rate", "lift"]])

# ## Example 2: Scorecard Calibration & JSON Deployment
import pandas as pd
from strategic_segment_engine import StrategicSegmentScore

# 1. Load training data enriched with binary segment indicator flags (seg_1, seg_2, etc.)
scored_df = pd.read_parquet("training_matrix_with_segment_flags.parquet")

active_segment_columns = ["seg_1", "seg_2", "seg_3", "seg_4", "seg_5"]

# 2. Instantiate Vectorized Scorer
scorer = StrategicSegmentScore(
    target_col="default_flag",
    primary_key="customer_id",
    segment_cols=active_segment_columns
)

# 3. Calculate weights, calibrate decile boundaries, and export deployment artifact
model_artifact = scorer.calculate_and_export_weights(
    df=scored_df,
    export_path="production_scorecard_v1.json"
)

# 4. Review calibrated decile thresholds
print("Calibrated Active Population Decile Thresholds:")
for decile, cutoff in model_artifact["decile_min_thresholds"].items():
    print(f"Decile {decile:2s} Minimum Score Benchmark: {cutoff:>6,d}")
