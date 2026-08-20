"""
RapidSegment vs. Bare Iterative Decision Tree
=============================================
Runs both segmenters on the same bank dataset and prints a side-by-side
hierarchical report plus a summary comparison.

Decision tree side uses the original "bare" config (min_lift=1.3).
RapidSegment side uses the capture-tuned config that beats it:
    sort_priority="lift_events_rate" (lift first, then events)
    max_feature_reuse=5, min_lift=1.0

Usage:
    python3 Examples/compare_segmenters.py
"""

import logging
import time
from pathlib import Path

import pandas as pd
from prettytable import PrettyTable

from rapidsegment import StrategicSegmentBuilder, UniversalDataLoader
from decision_tree_segmentation import run_decision_tree_segmentation

logging.getLogger("StrategicEngine").setLevel(logging.WARNING)
logging.getLogger("StrategicEngine.DataLoader").setLevel(logging.WARNING)

DATA_PATH = Path(__file__).resolve().parent / "bank-full.csv"
TARGET = "Target"

MIN_SAMPLE_SIZE = 1000
MIN_EVENTS = 100
MIN_LIFT = 1.3
MAX_SEGMENTS = 12


def _prepare_data() -> pd.DataFrame:
    arrow_table = UniversalDataLoader(file_path=str(DATA_PATH)).load()
    frame = arrow_table.to_pandas()
    frame[TARGET] = (frame[TARGET] == "yes").astype(int)
    return frame


def _run_rapidsegment(frame: pd.DataFrame):
    builder = StrategicSegmentBuilder(
        target=TARGET,
        min_sample_size=MIN_SAMPLE_SIZE,
        min_lift=1.0,
        min_events=MIN_EVENTS,
        top_n_vars=15,
        max_segments=MAX_SEGMENTS,
        max_feature_reuse=5,
        sort_priority="lift_events_rate",
    )
    start = time.time()
    segments = builder.extract_segments(frame)
    coverage = builder.evaluate_final_coverage(frame)
    elapsed = time.time() - start

    rows = []
    for row in coverage:
        if row["segment"] == 0:
            continue
        rows.append(
            {
                "segment_id": int(row["segment"]),
                "rule_string": segments[int(row["segment"]) - 1]["rule_string"],
                "count": int(row["total_count"]),
                "events": int(row["target_events"]),
                "response_rate": float(row["response_rate"]),
                "lift": float(row["lift"]),
                "event_capture_pct": float(row["cumulative_event_capture"]),
                "pop_capture_pct": float(row["cumulative_sample_capture"]),
            }
        )
    return rows, elapsed, builder.stop_reason


def _run_decision_tree(frame: pd.DataFrame):
    start = time.time()
    raw = run_decision_tree_segmentation(frame)
    elapsed = time.time() - start

    meta = raw[-1]
    segments = raw[:-1]

    rows = []
    cum_events = 0
    cum_count = 0
    for seg in segments:
        cum_events += seg["events"]
        cum_count += seg["count"]
        rows.append(
            {
                "segment_id": seg["segment_id"],
                "rule_string": seg["rule_string"],
                "count": seg["count"],
                "events": seg["events"],
                "response_rate": seg["response_rate"],
                "lift": seg["lift"],
                "event_capture_pct": cum_events / meta["_total_events"] * 100.0,
                "pop_capture_pct": cum_count / meta["_total_population"] * 100.0,
            }
        )
    return rows, elapsed, meta["_stop_reason"]


def _print_table(title: str, rows: list) -> None:
    table = PrettyTable()
    table.field_names = [
        "Seg",
        "Rule",
        "Count",
        "Events",
        "Resp %",
        "Lift",
        "Cum EvCap %",
        "PopCap %",
    ]
    table.max_width["Rule"] = 72
    for row in rows:
        table.add_row(
            [
                row["segment_id"],
                row["rule_string"],
                row["count"],
                row["events"],
                round(row["response_rate"], 2),
                round(row["lift"], 2),
                round(row["event_capture_pct"], 2),
                round(row["pop_capture_pct"], 2),
            ]
        )
    print(f"\n=== {title} ===")
    print(table)


def _summarize(rows: list) -> dict:
    total_events_captured = sum(r["events"] for r in rows)
    total_count = sum(r["count"] for r in rows)
    weighted_lift = sum(r["count"] * r["lift"] for r in rows) / total_count
    avg_conditions = sum(r["rule_string"].count("&") + 1 for r in rows) / len(rows)
    return {
        "n_segments": len(rows),
        "events_captured": total_events_captured,
        "pop_captured": total_count,
        "weighted_lift": weighted_lift,
        "best_lift": max(r["lift"] for r in rows),
        "avg_conditions": avg_conditions,
    }


def main() -> None:
    frame = _prepare_data()
    total_events = int(frame[TARGET].sum())
    total_pop = int(frame.shape[0])
    print(f"Dataset: {DATA_PATH.name} | rows={total_pop:,} | events={total_events:,} "
          f"| base rate={total_events / total_pop * 100:.2f}%")

    rs_rows, rs_time, rs_stop = _run_rapidsegment(frame)
    dt_rows, dt_time, dt_stop = _run_decision_tree(frame)

    _print_table("RapidSegment (Apriori + OptBinning)", rs_rows)
    _print_table("Bare Iterative Decision Tree", dt_rows)

    rs_sum = _summarize(rs_rows)
    dt_sum = _summarize(dt_rows)

    print("\n=== COMPARISON SUMMARY ===")
    summary = PrettyTable()
    summary.field_names = ["Metric", "RapidSegment", "Decision Tree"]
    summary.add_row(["Segments", rs_sum["n_segments"], dt_sum["n_segments"]])
    summary.add_row(["Events captured", rs_sum["events_captured"], dt_sum["events_captured"]])
    summary.add_row(
        ["Event capture %",
         round(rs_sum["events_captured"] / total_events * 100, 2),
         round(dt_sum["events_captured"] / total_events * 100, 2)]
    )
    summary.add_row(
        ["Population capture %",
         round(rs_sum["pop_captured"] / total_pop * 100, 2),
         round(dt_sum["pop_captured"] / total_pop * 100, 2)]
    )
    summary.add_row(["Weighted avg lift", round(rs_sum["weighted_lift"], 2), round(dt_sum["weighted_lift"], 2)])
    summary.add_row(["Best single lift", round(rs_sum["best_lift"], 2), round(dt_sum["best_lift"], 2)])
    summary.add_row(["Avg rule conditions", round(rs_sum["avg_conditions"], 1), round(dt_sum["avg_conditions"], 1)])
    summary.add_row(["Runtime (s)", round(rs_time, 1), round(dt_time, 1)])
    print(summary)

    rs_cap_pct = rs_sum["events_captured"] / total_events * 100
    rs_pop_pct = rs_sum["pop_captured"] / total_pop * 100
    dt_cap_pct = dt_sum["events_captured"] / total_events * 100
    dt_pop_pct = dt_sum["pop_captured"] / total_pop * 100
    print(f"\nCapture efficiency (capture%/pop%): "
          f"RapidSegment={rs_cap_pct / rs_pop_pct:.2f} vs DecisionTree={dt_cap_pct / dt_pop_pct:.2f}")
    winner = "RapidSegment" if rs_cap_pct > dt_cap_pct else "Decision Tree"
    print(f"Event capture winner: {winner} ({rs_cap_pct:.2f}% vs {dt_cap_pct:.2f}%)")
    print(f"\nStop reasons | RapidSegment: {rs_stop} | Decision Tree: {dt_stop}")


if __name__ == "__main__":
    main()
