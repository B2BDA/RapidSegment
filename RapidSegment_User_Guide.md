# RapidSegment – Strategic Segment Builder  
## Easy Step-by-Step User Guide (Data Flow & Internals)

This guide explains, in plain language, exactly what happens when you call `extract_segments()` on the `StrategicSegmentBuilder`.  
It focuses on the **naive binning path** (the one that supports adjacent-bin merging).

---

## 1. High-Level Idea

The engine finds **high-lift customer segments** one after another in a hierarchical way:

1. Look at the current data (first time = full data).
2. Find the single best rule that still meets your volume and lift requirements.
3. Remove the people who match that rule.
4. Repeat on the remaining people.
5. Stop when you reach `max_segments` or no more good rules can be found.

The final segments are ordered by priority (Segment 1 is the strongest / first discovered).

---

## 2. What Happens When You Call `extract_segments(data)`

### Step 0 – Setup (runs once)

- The data is loaded into DuckDB.
- The **original base response rate** is calculated on the full dataset and locked forever.
  - Example: if 11.7% of customers responded, every lift calculation from now on uses 11.7% as the denominator.
- Absolute hard floors are also locked:
  - `min_sample_size`
  - `min_events`
  - `min_lift`
- These floors are never relaxed, even if you use a parameter grid.

---

### Step 1 – Start of Each Iteration (Residual Loop)

At the beginning of every iteration the engine works on the **current residual** (the people not yet assigned to any previous segment).

#### 1.1 Feature Ranking & Binning

For every eligible feature (excluding the target and any features you ignored):

- The feature is **binned**:
  - **Numerical** → quantile bins (number of bins = `naive_bins`)
  - **Categorical** → one bin per unique value (+ a “Missing” bin if needed)
- While binning, two scores are calculated:
  - **Information Value (IV)**
  - **Max response rate** among bins that already pass the hard size/event floors

Features are then ranked according to `selection_metric`:
- `"iv"` (default) → highest Information Value first
- `"response_rate"` → highest max response rate first

Only the top `top_n_vars` features that have not yet reached `max_feature_reuse` are kept for this iteration.

**Important**: Binning and ranking are **re-done on every residual**. They are not calculated only once on the original data.

#### 1.2 Build the Temporary Binned Table

A temporary table (`binned_df`) is created that contains:
- The target column
- The top-ranked features, already converted into their bin labels

All subsequent combination searches run on this binned table (very fast).

---

### Step 2 – Rule Discovery (1-way → 2-way → 3-way)

The engine searches for good rules in a controlled combinatorial way (Apriori-style):

1. **1-way** (single features)  
2. **2-way** (pairs) – only pairs where both features already produced at least one valid 1-way rule  
3. **3-way** (triplets) – only if enabled and the pairwise building blocks exist

For every combination it runs a fast GROUP BY and keeps only those that satisfy:

- `count >= min_sample_size`
- `events >= min_events`
- `lift >= min_lift` (lift is always vs the **original** base rate)

---

### Step 3 – Adjacent Bin Expansion (the “Merge” Step)

This only happens when `binning_method = "naive"`.

For every rule that already passed the checks above, the engine tries to **merge neighbouring bins** on each variable in the rule:

- It looks at the left and right neighbouring bins of the current bin.
- It creates a new candidate that covers both the original bin and the neighbour.
- It re-calculates count, events, rate and lift for the expanded version.
- It keeps the expanded version **only if**:
  - It still meets all hard constraints (`min_sample_size`, `min_events`, `min_lift`)
  - **and** it captures **strictly more events** than the original rule

**Key point**: Expansion is only allowed to increase event volume while protecting the lift floor. It does **not** try to increase response rate or lift.

You will see a clean summary table in the logs (controlled by `expand_log_mode`):

```
🔀 Adjacent-bin expansion summary
   Combo                                       #exp  Best Δevents  Best lift
   ------------------------------------------------------------------------
   duration                                       5         +822      2.57x
   previous & pdays                              10         +275      3.04x
   → Total expanded candidates generated: 27
```

---

### Step 4 – Choosing the Champion of the Iteration

All surviving rules (original + expanded) are ranked according to your `sort_priority`:

| sort_priority       | Meaning                                      |
|---------------------|----------------------------------------------|
| `lift_rate_count`   | Highest lift → then rate → then size         |
| `count_lift_rate`   | Largest size → then lift → then rate         |
| `rate_lift_count`   | Highest response rate → then lift → then size|
| …                   | (other combinations available)               |

The top-ranked rule is then **validated on the raw residual data** (not the binned version) to make sure the numbers are accurate.

If it still passes the absolute hard floors, it becomes the official segment for this iteration.

---

### Step 5 – Create the Next Residual

All rows that match the winning rule are removed from the current residual.  
The next iteration starts again from Step 1 on the smaller remaining population.

---

## 3. Important Design Choices (Why Things Work This Way)

| Design Choice                        | Why it exists |
|--------------------------------------|---------------|
| Lift always vs original base rate    | Keeps the meaning of “2× lift” consistent across all segments |
| Hard floors never relaxed            | Prevents weak rules from appearing late in the process |
| Re-binning on every residual         | Features that were weak early can become strong later (and vice-versa) |
| Expansion only increases events      | Protects the quality (lift) while allowing larger, more useful segments |
| Hierarchical removal                 | Guarantees segments do not overlap |

---

## 4. Quick Mental Model

```
Full data
   │
   ▼
Rank features + create bins on residual
   │
   ▼
Search 1/2/3-way combinations
   │
   ▼
Try merging neighbouring bins (only if more events & still good lift)
   │
   ▼
Pick best rule according to sort_priority
   │
   ▼
Validate on raw data → accept as Segment N
   │
   ▼
Remove matching rows → new residual
   │
   └── repeat until max_segments or no more good rules
```

---

## 5. Useful Controls You Can Tune

- `min_lift` / `min_sample_size` / `min_events` → quality & size floors
- `sort_priority` → whether you prefer pure lift, pure volume, or pure rate
- `naive_bins` → more bins = more chances for useful merges (but slower)
- `top_n_vars` → how many features are allowed into the combination search
- `max_feature_reuse` → how many times the same feature can appear across segments
- `expand_log_mode` → `"summary"` (default), `"full"`, or `"none"`

---

This is the complete data-flow picture of the current engine on the Test branch.  
If anything is still unclear, just ask and we can zoom into that specific part.
