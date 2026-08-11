"""
Generates a stress-test binary classification dataset for RapidSegment.

Design goals:
- Strong numeric predictors (continuous, with a clean threshold effect) to
  stress the numeric interval / multi-range parsing branches.
- Strong categorical predictors, including tricky category values:
    * names containing commas               -> "New York, NY"
    * names containing single quotes        -> "O'Brien"
    * names containing double quotes        -> 'Say "hi"'
    * names containing brackets             -> "[VIP]"
    * names that look numeric               -> "007"
    * names that are 'Missing'/'Special'    -> literal collision with sentinel tokens
    * very high cardinality column          -> stresses IN(...) clause length
    * a single-category (constant) column   -> degenerate binning
- Nulls sprinkled into both numeric and categorical columns (missing-value branch).
- An engineered joint interaction between one numeric + one categorical column
  so 2-way/3-way combos have real signal (stresses _agg_combinations combos).
- A few pure noise columns (numeric + categorical) with zero signal, to check
  IV=0/eligibility filtering and top_n_vars truncation.
- Row count large enough (50k) to make min_sample_size/min_events thresholds
  meaningful and to exercise DuckDB disk-spill config paths realistically.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N = 50_000

def make_dataset(n=N, seed=42):
    rng = np.random.default_rng(seed)

    # ---- Strong numeric predictor with clean threshold effect ----
    income = rng.normal(60000, 20000, n).round(2)
    income = np.clip(income, 5000, 250000)

    # ---- Strong numeric predictor, skewed ----
    tenure_days = rng.exponential(400, n).round(1)

    # ---- Numeric predictor with extreme outliers ----
    balance = rng.normal(1000, 500, n)
    # inject extreme outliers
    outlier_idx = rng.choice(n, size=int(n * 0.002), replace=False)
    balance[outlier_idx] = rng.choice([1e7, -1e7, 1e9], size=len(outlier_idx))
    balance = balance.round(2)

    # ---- Pure noise numeric ----
    noise_num = rng.normal(0, 1, n).round(4)

    # ---- Constant numeric column (degenerate binning) ----
    constant_num = np.full(n, 42.0)

    # ---- Strong categorical predictor: tricky strings ----
    tricky_categories = [
        "New York, NY",      # comma
        "O'Brien",           # single quote
        'Say "hi"',          # double quote
        "[VIP]",             # brackets
        "007",                # numeric-looking string
        "Missing",            # collides with sentinel
        "Special",            # collides with sentinel
        "Normal_Segment_A",
        "Normal_Segment_B",
    ]
    # Assign non-uniform probabilities so some are rare (stresses min_sample_size)
    cat_probs = np.array([0.05, 0.05, 0.02, 0.03, 0.02, 0.02, 0.02, 0.40, 0.39])
    cat_probs = cat_probs / cat_probs.sum()
    region = rng.choice(tricky_categories, size=n, p=cat_probs)

    # ---- High cardinality categorical (stresses IN(...) length / bin count) ----
    high_card = rng.integers(0, 500, n).astype(str)
    high_card = np.array(["ID_" + x for x in high_card])

    # ---- Single-category constant categorical ----
    constant_cat = np.full(n, "ONLY_VALUE")

    # ---- Pure noise categorical ----
    noise_cat = rng.choice(["A", "B", "C", "D"], size=n)

    # ---- Binary-looking categorical (edge case for numeric-token detection) ----
    flag_str = rng.choice(["0", "1"], size=n)  # string but numeric-looking single-token

    # ---- Build the TRUE signal (binary target) ----
    # Strong interaction: high income AND region in {VIP, Segment_A} AND long tenure => high event rate
    z = (
        -6.5
        + 4.5 * (income > 90000)
        + 3.0 * (tenure_days > 600)
        + 3.5 * np.isin(region, ["[VIP]", "Normal_Segment_A"])
        + 2.0 * ((income > 90000) & (tenure_days > 600))  # extra interaction lift
        + 0.000001 * np.clip(balance, -2000, 5000)  # weak, mostly-noise numeric signal
    )
    prob = 1 / (1 + np.exp(-z))
    target = rng.binomial(1, prob)

    df = pd.DataFrame({
        "income": income,
        "tenure_days": tenure_days,
        "balance": balance,
        "noise_num": noise_num,
        "constant_num": constant_num,
        "region": region,
        "high_card_id": high_card,
        "constant_cat": constant_cat,
        "noise_cat": noise_cat,
        "flag_str": flag_str,
        "target": target,
    })

    # ---- Sprinkle NULLs into numeric and categorical columns ----
    null_frac = 0.03
    for col in ["income", "tenure_days", "balance", "noise_num"]:
        idx = rng.choice(n, size=int(n * null_frac), replace=False)
        df.loc[idx, col] = np.nan

    for col in ["region", "high_card_id", "noise_cat", "flag_str"]:
        idx = rng.choice(n, size=int(n * null_frac), replace=False)
        df.loc[idx, col] = None

    return df


if __name__ == "__main__":
    df = make_dataset()
    print(df.shape)
    print(df.dtypes)
    print(df["target"].mean())
    print(df.isna().mean())
    df.to_parquet("./stress_data.parquet", index=False)
    print("saved.")
