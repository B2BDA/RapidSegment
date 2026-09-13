> [!IMPORTANT]
> **Legal Disclaimer**  
> This open‑source library (`RapidSegment`) is an independent, community‑driven predictive analytics framework. It is **completely unaffiliated** with any commercial products, SaaS platforms, or enterprise solutions of the same or similar name. Any overlap in nomenclature is purely coincidental.

<p align="center">
  <img width="300" height="500" alt="adfd9cdf-251f-44e4-af79-20802d4a7a01" src="https://github.com/user-attachments/assets/803790d0-7ebe-4e79-a415-65f1daaea40b" />
</p>

# 🚀 RapidSegment – Strategic Segmentation & Scorecard Engine

[![PyPI version](https://img.shields.io/pypi/v/rapidsegment.svg)](https://pypi.org/project/rapidsegment/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**RapidSegment** is an industrial‑grade, combinatorial heuristic engine for discovering high‑lift predictive segments and compiling them into transparent, production‑ready scorecards. It bridges the gap between black‑box ML and legacy SQL rules engines.


## Rapid Segment Interactive Explainer (Must Try)
[Explainer](https://b2bda.github.io/RapidSegment/)

---

> 📘 **Full documentation, architecture diagrams, statistical background, configuration reference, and FAQs live in the [repo-level README](https://github.com/B2BDA/RapidSegment/blob/main/README.md).** This page only covers install + a quick usage example.
 
---
 
## 📦 Installation
 
```bash
pip install rapidsegment
```
 
With the optional no-code Web UI:
 
```bash
pip install "rapidsegment[ui]"
rapidsegment-ui          # opens http://localhost:8501
```
 
---
 
## ⚡ Quick Start
 
```python
import numpy as np
import pandas as pd
import duckdb
from rapidsegment import StrategicSegmentBuilder, StrategicSegmentScore
 
# 1. Synthetic data (or use your own)
np.random.seed(42)
n = 50_000
data = pd.DataFrame({
    "cust_id": [f"CUST_{i:05d}" for i in range(n)],
    "max_dpd_12m": np.random.choice([0, 15, 30, 60, 90], n, p=[0.7, 0.15, 0.08, 0.05, 0.02]),
    "utilization_avg_3m": np.random.uniform(0, 1.2, n),
    "risk_segment": np.random.choice(["Low", "Medium", "High"], n, p=[0.6, 0.3, 0.1]),
    "default_flag": np.random.choice([0, 1], n, p=[0.95, 0.05]),
})
 
# 2. Configure and run the builder
builder = StrategicSegmentBuilder(
    target="default_flag",
    top_n_vars=15,
    max_segments=5,
    ignore_features=["cust_id"],
)
segments = builder.extract_segments(data)
print(pd.DataFrame(segments)[["segment_id", "count", "lift", "sql_filter"]])
 
# 3. Build a scorecard
segment_cols = []
scoring_df = data[["cust_id", "default_flag"]].copy()
for seg in segments:
    col = f"SEG_{seg['segment_id']}"
    scoring_df[col] = duckdb.sql(f"SELECT ({seg['sql_filter']}) FROM data").df().astype(int)
    segment_cols.append(col)
 
scorer = StrategicSegmentScore(target_col="default_flag", primary_key="cust_id", segment_cols=segment_cols)
model = scorer.calculate_and_export_weights(scoring_df, "model.json")
print("Deciles:", model["decile_min_thresholds"])
```
 
## 🧩 How It Fits Together
 
```
Raw Data → StrategicSegmentBuilder → Segments (SQL rules) → StrategicSegmentScore → JSON Scorecard
```
 
`StrategicSegmentBuilder` finds high‑lift, hierarchical rules on your data; `StrategicSegmentScore` turns those rules into a weighted, deployable scorecard. `UniversalDataLoader` and `BigQueryFeatureSelector` are available for ingestion and feature screening — see the full README for details on every component, the extraction algorithm, statistical foundations, and the complete configuration reference.
 
---
 
## 📄 License
 
MIT — see [LICENSE](https://github.com/B2BDA/RapidSegment/blob/main/rapidsegment/LICENSE).
 
**Built with ❤️ by Bishwarup Biswas.** Special thanks to [Guillermo Navas Palencia](https://github.com/guillermo-navas-palencia) for [Optbinning](https://github.com/guillermo-navas-palencia/optbinning).
 
👉 **For everything else — features, architecture, config reference, FAQs — see the [repo README](https://github.com/B2BDA/RapidSegment/blob/main/README.md).**






