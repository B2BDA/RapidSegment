# Strategic Segment Builder (SSB)

A combinatorial heuristic engine for extracting highly predictive, mutually exclusive segments from tabular data using Optimal Binning and Apriori-style pruning.

---

## 1. Executive Summary

### WHAT is it?
The **Strategic Segment Builder (SSB)** is an automated rule-induction engine designed to search through large feature spaces to find high-performing, distinct, and statistically reliable sub-populations (segments) within a dataset. Instead of identifying broad trends, SSB extracts precise multidimensional rules—such as:
`utilization_avg_3m >= 0.84 AND max_dpd_12m >= 36.09 AND risk_segment IN ('HighRisk')`

### WHY do we need it?
In risk management, marketing analytics, and fraud detection, identifying specific risk pockets or high-value customer clusters usually demands extensive manual profiling, subjective visual binning, and trial-and-error. 
* **The Rules Engine Pitfall:** Manual exploration frequently misses non-linear intersections of three or more variables, or it accidentally creates overlapping rules where a single customer qualifies for multiple segments.
* **The Machine Learning Pitfall:** While black-box models (e.g., XGBoost, LightGBM) surface these patterns automatically, they cannot be natively translated into transparent, hard-coded SQL expressions required by production billing or legacy credit decisioning engines.
* **The Solution:** SSB bridges this gap by combining the automated discovery power of algorithmic optimization with the absolute transparency of pure SQL logical outputs.

### HOW does it work?
The engine runs recursively in a five-stage loop:
1. **Rank Feature Significance:** Uses Information Value (IV) to surface individual features with the highest correlation to your target variable.
2. **Optimal Discretization:** Groups continuous metrics into optimal categorical bins using the `OptimalBinning` framework.
3. **Combinatorial Search:** Pairs and triplets features using an Apriori heuristic to see if their intersection yields a massive spike in predictive power (Lift).
4. **Isolate and Extract:** Selects the single strongest rule, translates it instantly into production-ready SQL, and commits it to the segment array.
5. **Deduplicate Pool:** Leverages an in-memory SQL layer (`duckdb`) to remove the matching records entirely, ensuring that subsequent iterations evaluate only clean, unsegmented data to enforce absolute mutual exclusivity.

---

## 2. Statistical Foundation: WOE & IV

Before evaluating multidimensional intersections, the engine measures individual features using two core credit-scoring metrics: **Weight of Evidence (WOE)** and **Information Value (IV)**.

* **WOE (Weight of Evidence):** Measures the predictive power of a specific bin relative to the overall population. It quantifies how much a given value shifts the log-odds of an event occurring.
* **IV (Information Value):** Summarizes the overall predictive power of the *entire variable* across all its bins.

### Mathematical Formulas
$$ WOE = ln \\Bigg(\\frac{\\text{percent of non-events}}{\\text{percent of events}}\\Bigg)$$

IV = ∑ (% of non-events - % of events) * WOE
### Step-by-Step Working Example
Let's trace how the engine handles a single continuous variable, **`utilization_avg_3m`**, against a binary target where `1 = Default (Event)` and `0 = Good (Non-Event)`.

#### 1. Raw Distribution Data
Given a baseline population of **10,000 customers** (8,000 Goods / 2,000 Defaults), the optimization layer cuts the continuous variable into three distinct bins:

| Bin (Utilization Range) | Non-Events (Goods) | Events (Defaults) | Total Count |
| :--- | :---: | :---: | :---: |
| **Low** `[0.00, 0.40)` | 5,000 | 300 | 5,300 |
| **Mid** `[0.40, 0.84)` | 2,500 | 700 | 3,200 |
| **High** `[0.84, inf)` | 500 | 1,000 | 1,500 |
| **Total Baseline** | **8,000** | **2,000** | **10,000** |

#### 2. Convert Counts to Distribution Percentages
* **Low Bin [0.00, 0.40):**
  * Non-Event Distribution = 5,000 / 8,000 = 0.625 (62.5%)
  * Event Distribution = 300 / 2,000 = 0.150 (15.0%)
* **High Bin [0.84, inf):**
  * Non-Event Distribution = 500 / 8,000 = 0.0625 (6.25%)
  * Event Distribution = 1,000 / 2,000 = 0.500 (50.0%)

#### 3. Calculate Weight of Evidence (WOE)
* **Low Bin WOE:** ln(0.625 / 0.150) = ln(4.167) = +1.427 (Positive WOE indicates lower risk)
* **High Bin WOE:** ln(0.0625 / 0.500) = ln(0.125) = -2.079 (Negative WOE indicates elevated risk)

#### 3. Calculate Weight of Evidence (WOE)
* **Low Bin WOE:** $\ln(0.625 / 0.150) = \ln(4.167) = \mathbf{+1.427}$  *(Positive WOE indicates lower risk)*
* **High Bin WOE:** $\ln(0.0625 / 0.500) = \ln(0.125) = \mathbf{-2.079}$ *(Negative WOE indicates elevated risk)*

#### 4. Calculate Information Value (IV) Contribution
* **Low Bin IV:** $(0.625 - 0.150) \times 1.427 = 0.475 \times 1.427 = \mathbf{0.678}$
* **High Bin IV:** $(0.0625 - 0.500) \times (-2.079) = -0.4375 \times (-2.079) = \mathbf{0.910}$

Summing the IV contributions of all bins yields the variable's total IV. If a variable's total IV exceeds $0.3$, it is flagged as a strong predictor and prioritized during rule induction.

---

## 3. Algorithmic Design: Apriori vs. Grid Search

Evaluating multidimensional intersections across many variables can easily lead to a combinatorial explosion. SSB overcomes this bottleneck by applying a modified **Apriori pruning technique**.

### Standard Grid Search (Brute Force)
If you select the top 20 predictive variables and want to evaluate combinations up to 3 dimensions deep, a standard grid search calculates **every single mathematical permutation**:
* **1-Way Combinations:** 20 checks
* **2-Way Combinations:** $\binom{20}{2} = 190$ checks
* **3-Way Combinations:** $\binom{20}{3} = 1,140$ checks
* **Total Evaluated Configurations:** **1,350 aggregations** *per iteration*.

Performing thousands of multi-key grouping calculations repeatedly over millions of rows causes severe processing delays.

### The Apriori Solution
The Apriori principle leverages a core pruning property: **If an individual item fails to meet a performance threshold, any higher-order combination containing that item is guaranteed to fail.**

SSB implements this layer step-by-step:
1. **Level 1 (1-Way):** Group by and test all 20 individual variables against your volume (`min_sample_size`) and risk (`min_lift`) benchmarks. Assume only **6 variables** pass.
2. **Level 2 (2-Way):** Instead of pairing all 20 initial metrics, the engine **only generates pairs using the 6 surviving variables**. This prunes candidates down from 190 to just $\binom{6}{2} = \mathbf{15}$ pairs. Assume only **4 pairs** pass.
3. **Level 3 (3-Way):** Instead of testing 1,140 triplets, the engine evaluates only triplets where **all three internal pairs were successful** in Level 2. This narrows the candidate space down to just **2 or 3 highly reliable combinations**.

### Why it Matters:
* **Computational Efficiency:** Reduces required multi-key aggregations by up to $90\%$, cutting runtimes from hours to seconds.
* **Anti-Overfitting Guardrails:** Ensures that high-order 3-way interactions reflect robust, scalable interactions rather than random noise in small data subsets.

---

## 4. System Architecture & Process Flow

```text
       [Tabular Input DataFrame]
                    │
                    ▼
   ┌──────────────────────────────────┐
   │  Compute Information Value (IV)  │ ◄── Identifies and ranks top features
   └──────────────────────────────────┘
                    │
                    ▼
   ┌──────────────────────────────────┐
   │     Optimal Discretization       │ ◄── Drops uninformative single-bin variables
   └──────────────────────────────────┘
                    │
                    ▼
   ┌──────────────────────────────────┐
   │    Apriori Heuristic Engine      │
   │  • Level 1: Find valid base vars │
   │  • Level 2: Construct valid pairs│ ◄── Prunes invalid search spaces in parallel
   │  • Level 3: Form valid triplets  │
   └──────────────────────────────────┘
                    │
                    ▼
   ┌──────────────────────────────────┐
   │   Rule Selection (Max Lift)      │ ◄── Picks the top performing rule
   └──────────────────────────────────┘
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
┌──────────────────┐  ┌──────────────────┐
│ Parse to Pure    │  │ Slice Out Target │
│ Production SQL   │  │ Cohort (DuckDB)  │ ◄── Enforces mutual exclusivity
└──────────────────┘  └──────────────────┘
                               │
                               ▼
                    [Loop to Next Iteration]

```

### Execution Detail:

1. **IV Filter:** The engine filters input metrics down to the `top_n_vars` with the strongest predictive power.
2. **Bin Isolation:** Continuous metrics are transformed into mathematical intervals (e.g., `[0.84, inf)`). Variables that fail to split into at least 2 distinct bins are dropped immediately, preventing non-informative ranges from polluting the pipeline.
3. **Parallel Evaluation:** Combination candidates are distributed across CPU cores using `joblib` to calculate sample size, event rates, and lift concurrently.
4. **The Robust SQL Parsing Layer:** An internal regular expression engine parses string array representations (such as `<StringArray>['HighRisk']`) and numeric boundaries, converting raw rules into standardized SQL WHERE clauses: `col IN ('HighRisk')` or `col >= 0.84`.
5. **Deduplication:** The winning segment's SQL filter is compiled into a destructive elimination query run via `duckdb`:
`SELECT * FROM current_df WHERE NOT (winning_sql_filter)`
This updates the data pool in place, ensuring subsequent search loops focus exclusively on the unsegmented population.

---

## 5. Class Attributes & Parameter Reference

### Technical Configuration

* **`target`** *(str)*: Name of the dependent binary column (`1` = event, `0` = non-event) you intend to predict or optimize.
* **`n_jobs`** *(int, default: -1)*: Number of parallel CPU workers used. If set to `-1`, it automatically allocates all available threads minus one (`multiprocessing.cpu_count() - 1`).

### Rule & Heuristic Constraints

* **`min_sample_size`** *(int, default: 1000)*: Minimum row count required to validate a segment. Stops the engine from generating fragile micro-segments.
* **`min_lift`** *(float, default: 2.0)*: Minimum lift threshold required for a rule to be considered. Calculated as: $\text{Segment Rate} / \text{Baseline Population Rate}$.
* **`top_n_vars`** *(int, default: 20)*: Maximum number of high-IV features selected during each iteration. Caps computational complexity.
* **`max_segments`** *(int, default: 10)*: Maximum number of sequential, mutually exclusive segments the engine will extract before halting.

### Dimensional Controls

* **`enable_1way`** *(bool, default: True)*: Allows or restricts single-variable segment generation.
* **`enable_2way`** *(bool, default: True)*: Allows or restricts two-variable interaction segment generation.
* **`enable_3way`** *(bool, default: True)*: Allows or restricts three-variable interaction segment generation.

### Domain Knowledge Controls

* **`feature_groups`** *(Dict[str, List[str]], default: None)*: Optional dictionary categorizing raw columns into distinct business categories.
* *Example:* `{'liquidity': ['bal_1m', 'bal_3m'], 'delinquency': ['pay_0', 'max_dpd_12m']}`


* **`enable_diversity`** *(bool, default: False)*: When set to `True` (and given `feature_groups`), prevents multidimensional rules from using features within the same group. This forces the search engine to mix different analytical dimensions (e.g., combining a delinquency feature with a liquidity feature rather than pairing two highly correlated delinquency metrics).

---

## 6. Quick Start Guide

### Installation (Coming soon..........)

```bash
pip install strategic-segment-builder

```

### Basic Usage Example

```python
import pandas as pd
from strategic_segment_builder import StrategicSegmentBuilder

# 1. Load your raw transactional or analytical data
df = pd.read_csv("credit_risk_dataset.csv")

# 2. Define business groups (Optional)
groups = {
    "delinquency_vars": ["dpd_avg_3m", "dpd_avg_6m", "dpd_avg_12m", "max_dpd_12m"],            
    "transaction_vars": ["txn_count_avg_3m", "txn_count_avg_6m", "txn_count_avg_12m"],
    "spend_vars": ["spend_avg_3m", "spend_avg_6m", "spend_avg_12m"],
    "repayment_vars": ["payment_ratio_avg_3m", "payment_ratio_avg_6m", "payment_ratio_avg_12m"],
    "card_utilization_vars": ["utilization_avg_3m", "utilization_avg_6m", "utilization_avg_12m", "utilization_max_12m"]
}

# 3. Initialize the Strategic Segment Builder engine
builder = StrategicSegmentBuilder(
    target="default_flag",
    min_sample_size=1500,
    min_lift=2.5,
    enable_diversity=True,
    feature_groups=groups
)

# 4. Extract sequential segments
segments_df = builder.extract_segments(df)

# 5. Review the extracted production-ready rules
print(segments_df[["segment_id", "sql_filter", "count", "lift"]])

# 6. Evaluate full cascading database coverage report
coverage_df = builder.evaluate_final_coverage(df)
print(coverage_df)

```

```

```

