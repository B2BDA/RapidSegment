"""
Capture-optimization sweep for RapidSegment on bank-full.csv.

Goal: find a config that beats the decision-tree baseline's 83.85% event capture.
Prints one compact row per config.
"""

import logging
import time
from pathlib import Path

from rapidsegment import StrategicSegmentBuilder, UniversalDataLoader

logging.getLogger("StrategicEngine").setLevel(logging.WARNING)
logging.getLogger("StrategicEngine.DataLoader").setLevel(logging.WARNING)

DATA_PATH = Path(__file__).resolve().parent / "bank-full.csv"
TARGET = "Target"

CONFIGS = [
    {
        "name": "base (rate_lift_count, reuse2, lift1.3)",
        "sort_priority": "rate_lift_count",
        "max_feature_reuse": 2,
        "min_lift": 1.3,
        "min_events": 100,
        "param_grid": None,
    },
    {
        "name": "lift_events_rate, reuse5, lift1.3 grid",
        "sort_priority": "lift_events_rate",
        "max_feature_reuse": 5,
        "min_lift": 1.3,
        "min_events": 100,
        "param_grid": {"min_sample_size": [1000, 2000], "min_lift": [1.3, 1.2, 1.1]},
    },
    {
        "name": "rate_events_lift, reuse5, lift1.3 grid",
        "sort_priority": "rate_events_lift",
        "max_feature_reuse": 5,
        "min_lift": 1.3,
        "min_events": 100,
        "param_grid": {"min_sample_size": [1000, 2000], "min_lift": [1.3, 1.2, 1.1]},
    },
    {
        "name": "lift_events_rate, reuse5, lift1.0",
        "sort_priority": "lift_events_rate",
        "max_feature_reuse": 5,
        "min_lift": 1.0,
        "min_events": 100,
        "param_grid": None,
    },
    {
        "name": "lift_events_rate, reuse5, lift1.0, ev50",
        "sort_priority": "lift_events_rate",
        "max_feature_reuse": 5,
        "min_lift": 1.0,
        "min_events": 50,
        "param_grid": None,
    },
    {
        "name": "lift_events_rate, reuse5, lift1.05, naive bins",
        "sort_priority": "lift_events_rate",
        "max_feature_reuse": 5,
        "min_lift": 1.05,
        "min_events": 100,
        "param_grid": None,
        "binning_method": "naive",
    },
]


def main() -> None:
    arrow_table = UniversalDataLoader(file_path=str(DATA_PATH)).load()
    frame = arrow_table.to_pandas()
    frame[TARGET] = (frame[TARGET] == "yes").astype(int)
    total_events = int(frame[TARGET].sum())
    total_pop = int(frame.shape[0])
    print(f"rows={total_pop} events={total_events} base={total_events/total_pop*100:.2f}%")
    print(f"DT baseline: capture 83.85% / pop 28.04% / wlift 2.99")

    for cfg in CONFIGS:
        builder = StrategicSegmentBuilder(
            target=TARGET,
            min_sample_size=1000,
            min_lift=cfg["min_lift"],
            min_events=cfg["min_events"],
            top_n_vars=15,
            max_segments=12,
            max_feature_reuse=cfg["max_feature_reuse"],
            param_grid=cfg["param_grid"],
            sort_priority=cfg["sort_priority"],
            binning_method=cfg.get("binning_method", "optimal_cart"),
        )
        start = time.time()
        segments = builder.extract_segments(frame)
        coverage = builder.evaluate_final_coverage(frame)
        elapsed = time.time() - start

        rows = [c for c in coverage if c["segment"] != 0]
        captured = sum(int(c["target_events"]) for c in rows)
        pop = sum(int(c["total_count"]) for c in rows)
        wlift = sum(int(c["total_count"]) * c["lift"] for c in rows) / pop
        print(
            f"\n[{cfg['name']}] n={len(rows)} capture={captured/total_events*100:.2f}% "
            f"pop={pop/total_pop*100:.2f}% wlift={wlift:.2f} bestlift={max(c['lift'] for c in rows):.2f} "
            f"time={elapsed:.1f}s stop={builder.stop_reason}"
        )
        for c in rows:
            print(
                f"  seg{c['segment']}: rows={int(c['total_count']):>6} ev={int(c['target_events']):>4} "
                f"rate={c['response_rate']:.2f} lift={c['lift']:.2f}"
            )


if __name__ == "__main__":
    main()
