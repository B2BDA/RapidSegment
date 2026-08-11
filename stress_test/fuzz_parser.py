import sys, warnings, traceback
warnings.filterwarnings("ignore")
sys.path.insert(0, "/.")
import pandas as pd
import numpy as np
import duckdb
from builder import StrategicSegmentBuilder

df = pd.read_parquet("/./stress_data.parquet")

sb = StrategicSegmentBuilder(target="target", db_path=":memory:", db_temp_dir="/tmp")
sb._categorical_cols = {"region", "high_card_id", "constant_cat", "noise_cat", "flag_str"}

con = duckdb.connect(":memory:")
con.register("v", df)
con.execute("CREATE TABLE raw AS SELECT * FROM v")

cases = []

# --- Numeric standard interval branches ---
cases.append(("numeric_closed_open", "income=[50000.0, 100000.0)",
              lambda d: (d["income"] >= 50000.0) & (d["income"] < 100000.0)))
cases.append(("numeric_open_closed", "income=(50000.0, 100000.0]",
              lambda d: (d["income"] > 50000.0) & (d["income"] <= 100000.0)))
cases.append(("numeric_neg_inf_lower", "income=[-inf, 50000.0)",
              lambda d: (d["income"] < 50000.0)))
cases.append(("numeric_inf_upper", "income=[50000.0, inf)",
              lambda d: (d["income"] >= 50000.0)))
cases.append(("numeric_single_value", "constant_num=[42.0]",
              lambda d: (d["constant_num"] == 42.0)))
cases.append(("numeric_negative_bounds", "balance=[-5000.0, -100.0)",
              lambda d: (d["balance"] >= -5000.0) & (d["balance"] < -100.0)))

# --- Numeric multi-range merged (branch 1) ---
cases.append(("numeric_multirange_2", "income=[[50000.0, 75000.0), [75000.0, 100000.0)]",
              lambda d: (d["income"] >= 50000.0) & (d["income"] < 100000.0)))
cases.append(("numeric_multirange_3_with_inf", "income=[[-inf, 30000.0), [30000.0, 60000.0), [60000.0, 90000.0)]",
              lambda d: (d["income"] < 90000.0)))
cases.append(("numeric_multirange_open_end", "income=[[60000.0, 90000.0), [90000.0, inf)]",
              lambda d: (d["income"] >= 60000.0)))

# --- Categorical single value ---
cases.append(("cat_single_simple", "region=[Normal_Segment_A]",
              lambda d: d["region"] == "Normal_Segment_A"))
cases.append(("cat_single_numeric_looking", "region=[007]",
              lambda d: d["region"] == "007"))
cases.append(("cat_single_with_brackets_in_value", "region=[[VIP]]",
              lambda d: d["region"] == "[VIP]"))
cases.append(("cat_single_sentinel_collision_missing", "region=[Missing]",
              # ambiguous: could mean the literal string "Missing" category OR the NULL sentinel;
              # ground truth = literal category value, since interval isn't exactly "Missing" alone
              lambda d: d["region"] == "Missing"))
cases.append(("cat_flag_str_zero", "flag_str=[0]",
              lambda d: d["flag_str"] == "0"))

# --- Categorical list (comma separated, plain) ---
cases.append(("cat_list_plain", "region=[Normal_Segment_A, Normal_Segment_B]",
              lambda d: d["region"].isin(["Normal_Segment_A", "Normal_Segment_B"])))

# --- Categorical merged-list (branch 1b, from _expand_adjacent_bins naive merges) ---
cases.append(("cat_merged_list_2", "region=[[Normal_Segment_A],[Normal_Segment_B]]",
              lambda d: d["region"].isin(["Normal_Segment_A", "Normal_Segment_B"])))
cases.append(("cat_merged_list_with_comma_value", "region=[[New York, NY],[O'Brien]]",
              lambda d: d["region"].isin(["New York, NY", "O'Brien"])))
cases.append(("cat_merged_list_with_quote_value", "region=[[Say \"hi\"],[Normal_Segment_A]]",
              lambda d: d["region"].isin(['Say "hi"', "Normal_Segment_A"])))
cases.append(("cat_merged_list_3", "region=[[007],[Missing],[Special]]",
              lambda d: d["region"].isin(["007", "Missing", "Special"])))

# --- Special / Missing sentinel (NULL) ---
cases.append(("sentinel_missing", "income=Missing",
              lambda d: d["income"].isna()))
cases.append(("sentinel_special", "income=Special",
              lambda d: d["income"].isna()))
cases.append(("sentinel_missing_categorical", "region=Missing",
              lambda d: d["region"].isna()))

# --- High cardinality IN clause ---
some_ids = [f"ID_{i}" for i in range(0, 100, 3)]
cases.append(("high_card_in_list", "high_card_id=[" + ", ".join(some_ids) + "]",
              lambda d, ids=some_ids: d["high_card_id"].isin(ids)))

# --- Multi-part AND rule (2-way / 3-way combos) ---
cases.append(("two_way_and", "income=[80000.0, inf) & region=[Normal_Segment_A]",
              lambda d: (d["income"] >= 80000.0) & (d["region"] == "Normal_Segment_A")))
cases.append(("three_way_and", "income=[80000.0, inf) & region=[Normal_Segment_A] & flag_str=[1]",
              lambda d: (d["income"] >= 80000.0) & (d["region"] == "Normal_Segment_A") & (d["flag_str"] == "1")))

# --- Adversarial: category value that itself equals literal "Special"/"Missing" combined with real NULLs ---
cases.append(("cat_value_equals_special_literal_vs_null", "region=[Special]",
              lambda d: d["region"] == "Special"))

print(f"Running {len(cases)} ground-truth validation cases...\n")

n_pass, n_fail, n_crash = 0, 0, 0
for name, rule, truth_fn in cases:
    try:
        sql = sb.parse_rule_to_sql(rule)
        if not sql.strip():
            print(f"❌ CRASH/EMPTY [{name}]: rule={rule!r} -> empty SQL produced")
            n_crash += 1
            continue
        got = con.execute(f"SELECT COUNT(*) FROM raw WHERE ({sql})").fetchone()[0]
        expected = int(truth_fn(df).sum())
        status = "✅ PASS" if got == expected else "❌ MISMATCH"
        if got == expected:
            n_pass += 1
        else:
            n_fail += 1
        print(f"{status} [{name}]\n    rule={rule!r}\n    sql={sql!r}\n    got={got} expected={expected}\n")
    except Exception as e:
        n_crash += 1
        print(f"💥 CRASH [{name}]: rule={rule!r}\n    {type(e).__name__}: {e}\n")

print("=" * 80)
print(f"PASS={n_pass}  MISMATCH={n_fail}  CRASH={n_crash}  TOTAL={len(cases)}")
