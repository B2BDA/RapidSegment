import numpy as np
import pyarrow as pa
import duckdb
import os
from segmentiq import StrategicSegmentBuilder
from segmentiq import StrategicSegmentScore

def test_pipeline():
    print("--- Starting Pipeline Integration Test (PyArrow) ---")
    
    if os.path.exists("scorecard_analytics.db"):
        os.remove("scorecard_analytics.db")

    # 1. DATA LOADER: Generate Synthetic Data using PyArrow
    print("Step 1: Generating synthetic data via PyArrow...")
    np.random.seed(42)
    n_records = 20000
    
    # Generate dictionary of numpy arrays
    data_dict = {
        "cust_id": [f"CUST_{i:05d}" for i in range(n_records)],
        "max_dpd_12m": np.random.choice([0, 15, 30, 60, 90], size=n_records),
        "utilization": np.random.uniform(0.0, 1.0, size=n_records),
        "spend": np.random.exponential(scale=1000, size=n_records),
        "default_flag": np.random.choice([0, 1], size=n_records, p=[0.9, 0.1])
    }
    
    # Create Table
    data = pa.Table.from_pydict(data_dict)

    # 2. BUILDER: Extract Segments
    print("Step 2: Running StrategicSegmentBuilder...")
    builder = StrategicSegmentBuilder(
        target="default_flag",
        max_segments=3,
        min_sample_size=500
    )
    segments = builder.extract_segments(data)
    
    # 3. INTEGRATION: Apply SQL Rules
    print("Step 3: Applying SQL segments...")
    con = duckdb.connect("scorecard_analytics.db")
    con.execute("CREATE TABLE df_working AS SELECT * FROM data")
    
    segment_cols = []
    for seg in segments:
        col_name = f"SEGMENT_{seg['segment_id']}"
        con.execute(f"""
            ALTER TABLE df_working ADD COLUMN {col_name} INTEGER;
            UPDATE df_working SET {col_name} = 1 WHERE {seg['sql_filter']};
            UPDATE df_working SET {col_name} = 0 WHERE {col_name} IS NULL;
        """)
        segment_cols.append(col_name)
    
    # DuckDB returns pyarrow.Table by default for .arrow()
    scored_table = con.execute("SELECT * FROM df_working").arrow()
 

    # 4. SCORER: Run Scorecard Engine
    print("Step 4: Running StrategicSegmentScore...")
    scorer = StrategicSegmentScore(
        target_col="default_flag",
        primary_key="cust_id",
        segment_cols=segment_cols,
    
    )
    
    model_artifact = scorer.calculate_and_export_weights(scored_table)
    con.close()
    print("Integration Test Passed!")

if __name__ == "__main__":
    test_pipeline()