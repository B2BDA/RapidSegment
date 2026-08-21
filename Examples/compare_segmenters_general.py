#!/usr/bin/env python3
"""
Generalized RapidSegment vs. Decision-Tree comparison for ANY dataset.

Run it interactively:
    python3 Examples/compare_segmenters_general.py --data path/to/data.csv

Or fully headless:
    python3 Examples/compare_segmenters_general.py \
        --data data.csv --target Response --positive yes \
        --ignore-columns id customer_id \
        --plot-out comparison.png --report-out comparison.md

Behavior
--------
* Loads the file with RapidSegment's UniversalDataLoader (csv / parquet /
  arrow / feather / xlsx), falling back to pandas if that fails.
* Auto-detects the target column from common names; if ambiguous (or not
  found), asks you to pick one.
* Checks whether the target is already binary 0/1 (or boolean). If not, it
  asks which class value should be treated as the positive event (1).
* Optionally drops columns listed via --ignore-columns (IDs, PII, leakage
  features, etc.) before fitting either model. RapidSegment also receives
  them as ignore_features so they never enter IV ranking.
* Runs the bare iterative decision tree and RapidSegment with comparable
  constraints, then prints per-method hierarchical tables, a summary, and
  optionally saves a capture-curve chart and/or a markdown report.

Use --no-input to disable every prompt (errors out instead of asking).
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from prettytable import PrettyTable

from rapidsegment import StrategicSegmentBuilder, UniversalDataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decision_tree_segmentation import run_decision_tree_segmentation

logging.getLogger("StrategicEngine").setLevel(logging.WARNING)
logging.getLogger("StrategicEngine.DataLoader").setLevel(logging.WARNING)

PREFERRED_TARGETS = [
    "target", "Target", "TARGET", "label", "Label", "y", "Y",
    "class", "Class", "CLASS", "response", "Response", "churn", "Churn",
    "exited", "Exited", "default", "Default", "survived", "Survived",
    "converted", "Converted",
]


# --------------------------------------------------------------------------- #
# Input helpers
# --------------------------------------------------------------------------- #
def ask_choice(prompt: str, options, allow_input: bool):
    """Print numbered options and return the selected one (or None)."""
    if not allow_input:
        raise SystemExit(f"Input required but --no-input was set. {prompt}")
    print(prompt)
    for i, opt in enumerate(options, start=1):
        print(f"  [{i}] {opt}")
    while True:
        try:
            raw = input("Choice: ").strip()
        except EOFError:
            raise SystemExit("No input available; re-run with --no-input or supply flags.")
        if not raw:
            continue
        try:
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        except ValueError:
            pass
        if raw in [str(o) for o in options]:
            return raw
        print("Invalid choice, try again.")


# --------------------------------------------------------------------------- #
# Loading + target handling
# --------------------------------------------------------------------------- #
def load_dataframe(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    try:
        arrow_table = UniversalDataLoader(file_path=str(path)).load()
        return arrow_table.to_pandas()
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "UniversalDataLoader failed (%s); falling back to pandas.", exc
        )
    if ext == ".csv":
        return pd.read_csv(path, sep=None, engine="python")
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if ext in (".arrow", ".feather"):
        return pd.read_feather(path)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file format: '{ext}'.")


def detect_target(frame: pd.DataFrame, explicit, allow_input: bool) -> str:
    if explicit:
        if explicit not in frame.columns:
            raise SystemExit(f"Target column '{explicit}' not found. Available: {sorted(frame.columns)}")
        return explicit
    hits = [c for c in PREFERRED_TARGETS if c in frame.columns]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        return ask_choice(
            f"Multiple likely target columns found: {hits}. Which is the target?",
            hits, allow_input,
        )
    return ask_choice(
        "Could not auto-detect the target column. Pick one:",
        sorted(frame.columns), allow_input,
    )


def ensure_binary(frame: pd.DataFrame, target: str, positive, allow_input: bool):
    """
    Guarantee the target is numeric 0/1.

    Returns (frame_with_0_1_target, note) where note describes any mapping that
    was applied (None if the target was already 0/1).
    """
    frame = frame.copy()
    if frame[target].isna().any():
        n_dropped = int(frame[target].isna().sum())
        print(f"  Dropping {n_dropped:,} rows with null target.")
        frame = frame[frame[target].notna()]
    ser = frame[target]
    uniq = sorted(ser.unique(), key=lambda v: (str(v), v))

    if all(isinstance(v, bool) for v in uniq):
        frame[target] = ser.astype(int)
        return frame, "boolean mapped to 0/1 (True=1)"

    if all(
        isinstance(v, (int, float)) and not isinstance(v, bool)
        and float(v) in (0.0, 1.0)
        for v in uniq
    ):
        frame[target] = ser.astype(int)
        return frame, None  # already binary 0/1

    if positive is None:
        positive = ask_choice(
            f"Target '{target}' is not binary 0/1. Classes: {uniq}\n"
            "Which value should be treated as the POSITIVE event (1)?",
            uniq, allow_input,
        )
    if positive not in uniq:
        raise SystemExit(
            f"--positive value {positive!r} is not a class of '{target}'. "
            f"Classes are: {uniq}"
        )
    frame[target] = (ser == positive).astype(int)
    return frame, f"'{positive}' mapped to 1; everything else to 0"


def resolve_ignore_columns(frame: pd.DataFrame, ignore_columns, target: str):
    """
    Validate and normalize the list of columns to ignore.

    Returns a list of column names that exist in the frame and are not the
    target. Missing names are reported and skipped.
    """
    if not ignore_columns:
        return []
    resolved = []
    missing = []
    for col in ignore_columns:
        if col == target:
            print(f"  Warning: ignore column '{col}' is the target — skipping.")
            continue
        if col not in frame.columns:
            missing.append(col)
            continue
        resolved.append(col)
    if missing:
        print(f"  Warning: ignore columns not found in data (skipped): {missing}")
    if resolved:
        print(f"  Ignoring columns: {resolved}")
    return resolved


def apply_ignore_columns(frame: pd.DataFrame, ignore_columns):
    """Return a copy of frame with ignore_columns dropped."""
    if not ignore_columns:
        return frame
    return frame.drop(columns=list(ignore_columns))


# --------------------------------------------------------------------------- #
# Running both segmenters
# --------------------------------------------------------------------------- #
def run_rapidsegment(frame, target, args, ignore_features):
    builder = StrategicSegmentBuilder(
        target=target,
        min_sample_size=args.min_sample_size,
        min_lift=args.rs_min_lift,
        min_events=args.min_events,
        top_n_vars=args.top_n_vars,
        max_segments=args.max_segments,
        max_feature_reuse=args.max_feature_reuse,
        sort_priority=args.rs_sort_priority,
        ignore_features=list(ignore_features) if ignore_features else [],
    )
    start = time.time()
    segments = builder.extract_segments(frame)
    coverage = builder.evaluate_final_coverage(frame)
    elapsed = time.time() - start

    rows = []
    for row in coverage:
        if row["segment"] == 0:
            continue
        rows.append({
            "segment_id": int(row["segment"]),
            "rule": segments[int(row["segment"]) - 1]["rule_string"],
            "count": int(row["total_count"]),
            "events": int(row["target_events"]),
            "response_rate": float(row["response_rate"]),
            "lift": float(row["lift"]),
            "cum_evcap_pct": float(row["cumulative_event_capture"]),
            "cum_popcap_pct": float(row["cumulative_sample_capture"]),
        })
    return rows, elapsed, builder.stop_reason


def run_decision_tree(frame, target, args, ignore_columns):
    # DT has no ignore_features param; drop columns before fitting so the
    # comparison is fair (same feature space as RapidSegment).
    dt_frame = apply_ignore_columns(frame, ignore_columns)
    start = time.time()
    raw = run_decision_tree_segmentation(
        dt_frame, target_col=target,
        max_segments=args.max_segments,
        min_sample_size=args.dt_min_sample_size,
        min_events=args.dt_min_events,
        min_lift=args.dt_min_lift,
    )
    elapsed = time.time() - start
    meta = raw[-1]
    segments = raw[:-1]

    rows = []
    cum_events = 0
    cum_count = 0
    for seg in segments:
        cum_events += seg["events"]
        cum_count += seg["count"]
        rows.append({
            "segment_id": seg["segment_id"],
            "rule": seg["rule_string"],
            "count": seg["count"],
            "events": seg["events"],
            "response_rate": seg["response_rate"],
            "lift": seg["lift"],
            "cum_evcap_pct": cum_events / meta["_total_events"] * 100.0,
            "cum_popcap_pct": cum_count / meta["_total_population"] * 100.0,
        })
    return rows, elapsed, meta["_stop_reason"]


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def print_table(title: str, rows) -> None:
    table = PrettyTable()
    table.field_names = ["Seg", "Rule", "Count", "Events", "Resp %", "Lift",
                         "Cum EvCap %", "PopCap %"]
    table.max_width["Rule"] = 72
    for r in rows:
        table.add_row([r["segment_id"], r["rule"], f"{r['count']:,}", r["events"],
                       f"{r['response_rate']:.2f}", f"{r['lift']:.2f}",
                       f"{r['cum_evcap_pct']:.2f}", f"{r['cum_popcap_pct']:.2f}"])
    print(f"\n=== {title} ===")
    print(table)


def summarize(rows) -> dict:
    cap = sum(r["events"] for r in rows)
    pop = sum(r["count"] for r in rows)
    wlift = sum(r["count"] * r["lift"] for r in rows) / pop if pop else 0.0
    n_cond = sum(r["rule"].count("&") + 1 for r in rows) / len(rows) if rows else 0.0
    return {
        "n": len(rows), "cap": cap, "pop": pop, "wlift": wlift,
        "best_lift": max((r["lift"] for r in rows), default=0.0),
        "avg_cond": n_cond,
    }


def build_summary_lines(rs_rows, dt_rows, rs_time, dt_time, rs_stop, dt_stop,
                        total_events, total_pop, note):
    rs = summarize(rs_rows)
    dt = summarize(dt_rows)
    rs_cap_pct = rs["cap"] / total_events * 100 if total_events else 0.0
    dt_cap_pct = dt["cap"] / total_events * 100 if total_events else 0.0
    rs_pop_pct = rs["pop"] / total_pop * 100 if total_pop else 0.0
    dt_pop_pct = dt["pop"] / total_pop * 100 if total_pop else 0.0
    rs_eff = rs_cap_pct / rs_pop_pct if rs_pop_pct else 0.0
    dt_eff = dt_cap_pct / dt_pop_pct if dt_pop_pct else 0.0

    table = PrettyTable()
    table.field_names = ["Metric", "RapidSegment", "Decision Tree"]
    table.add_row(["Segments", rs["n"], dt["n"]])
    table.add_row(["Events captured", f"{rs['cap']:,}", f"{dt['cap']:,}"])
    table.add_row(["Event capture %", f"{rs_cap_pct:.2f}", f"{dt_cap_pct:.2f}"])
    table.add_row(["Population capture %", f"{rs_pop_pct:.2f}", f"{dt_pop_pct:.2f}"])
    table.add_row(["Weighted avg lift", f"{rs['wlift']:.2f}", f"{dt['wlift']:.2f}"])
    table.add_row(["Best single lift", f"{rs['best_lift']:.2f}", f"{dt['best_lift']:.2f}"])
    table.add_row(["Avg rule conditions", f"{rs['avg_cond']:.1f}", f"{dt['avg_cond']:.1f}"])
    table.add_row(["Runtime (s)", f"{rs_time:.1f}", f"{dt_time:.1f}"])
    lines = [f"Base rate (positive events): {total_events/total_pop*100:.2f}% ({total_events:,}/{total_pop:,})"]
    if note:
        lines.append(f"Target encoding: {note}")
    lines.append(str(table))
    lines.append(
        f"\nCapture efficiency (capture% / pop%): RapidSegment={rs_eff:.2f} vs "
        f"DecisionTree={dt_eff:.2f}"
    )
    winner = "RapidSegment" if rs_cap_pct > dt_cap_pct else "Decision Tree"
    lines.append(f"Event capture winner: {winner} ({rs_cap_pct:.2f}% vs {dt_cap_pct:.2f}%)")
    lines.append(f"Stop reasons | RapidSegment: {rs_stop} | Decision Tree: {dt_stop}")
    return lines


def plot_comparison(rs_rows, dt_rows, base_rate, out_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rs_cum_pop = [0] + [r["cum_popcap_pct"] for r in rs_rows]
    rs_cum_ev = [0] + [r["cum_evcap_pct"] for r in rs_rows]
    dt_cum_pop = [0] + [r["cum_popcap_pct"] for r in dt_rows]
    dt_cum_ev = [0] + [r["cum_evcap_pct"] for r in dt_rows]

    x_idx = np.arange(max(len(rs_rows), len(dt_rows)))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes[0, 0]
    ax.step(dt_cum_pop, dt_cum_ev, where="post", color="#d62728", lw=2.2, label="Decision Tree")
    ax.step(rs_cum_pop, rs_cum_ev, where="post", color="#1f77b4", lw=2.2, label="RapidSegment")
    ax.plot(dt_cum_pop, dt_cum_pop, "--", color="gray", lw=1, alpha=.6, label="Random targeting")
    ax.set_xlabel("Cumulative population (%)")
    ax.set_ylabel("Cumulative event capture (%)")
    ax.set_title("Capture curve (higher = better)")
    ax.legend()
    ax.grid(alpha=.3)

    ax = axes[0, 1]
    ax.bar(x_idx[:len(dt_rows)] - 0.2, [r["lift"] for r in dt_rows], width=0.4,
           color="#d62728", alpha=.85, label="Decision Tree")
    ax.bar(x_idx[:len(rs_rows)] + 0.2, [r["lift"] for r in rs_rows], width=0.4,
           color="#1f77b4", alpha=.85, label="RapidSegment")
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.set_xlabel("Segment (by extraction order)")
    ax.set_ylabel("Lift")
    ax.set_title("Lift per segment")
    ax.legend()
    ax.grid(alpha=.3)

    ax = axes[1, 0]
    ax.bar(x_idx[:len(dt_rows)] - 0.2, [r["events"] for r in dt_rows], width=0.4,
           color="#d62728", alpha=.85, label="Decision Tree")
    ax.bar(x_idx[:len(rs_rows)] + 0.2, [r["events"] for r in rs_rows], width=0.4,
           color="#1f77b4", alpha=.85, label="RapidSegment")
    ax.set_xlabel("Segment (by extraction order)")
    ax.set_ylabel("Events captured")
    ax.set_title("Events captured per segment")
    ax.legend()
    ax.grid(alpha=.3)

    ax = axes[1, 1]
    ax.bar(x_idx[:len(dt_rows)] - 0.2, [r["response_rate"] for r in dt_rows], width=0.4,
           color="#d62728", alpha=.85, label="Decision Tree")
    ax.bar(x_idx[:len(rs_rows)] + 0.2, [r["response_rate"] for r in rs_rows], width=0.4,
           color="#1f77b4", alpha=.85, label="RapidSegment")
    ax.axhline(base_rate, color="gray", ls="--", lw=1, label=f"Base rate {base_rate:.1f}%")
    ax.set_xlabel("Segment (by extraction order)")
    ax.set_ylabel("Response rate (%)")
    ax.set_title("Response rate per segment")
    ax.legend()
    ax.grid(alpha=.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved to {out_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="RapidSegment vs Decision Tree comparison")
    p.add_argument("--data", help="Path to the dataset (csv/parquet/arrow/feather/xlsx)")
    p.add_argument("--target", help="Target column name")
    p.add_argument("--positive", help="Which class value is the positive event (1)")
    p.add_argument("--ignore-columns", nargs="*", default=None,
                   help="Column names to ignore / drop before fitting "
                        "(IDs, PII, leakage features). Passed to RapidSegment as "
                        "ignore_features and dropped from the Decision Tree frame.")
    p.add_argument("--no-input", action="store_true",
                   help="Never prompt; error out when input would be required")

    p.add_argument("--min-sample-size", type=int, default=1000)
    p.add_argument("--min-events", type=int, default=100)
    p.add_argument("--max-segments", type=int, default=12)
    p.add_argument("--top-n-vars", type=int, default=15)
    p.add_argument("--max-feature-reuse", type=int, default=5)
    p.add_argument("--rs-sort-priority", default="lift_events_rate")
    p.add_argument("--rs-min-lift", type=float, default=1.0)
    p.add_argument("--dt-min-lift", type=float, default=1.3)
    p.add_argument("--dt-min-events", type=int, default=100)
    p.add_argument("--dt-min-sample-size", type=int, default=1000)

    p.add_argument("--plot-out", help="Save comparison chart to this PNG path")
    p.add_argument("--report-out", help="Save a markdown report to this path")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    allow_input = not args.no_input
    np.random.seed(args.seed)

    if not args.data:
        if not allow_input:
            raise SystemExit("--data is required when using --no-input.")
        args.data = input("Path to dataset (csv/parquet/arrow/feather/xlsx): ").strip()
    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"File not found: {data_path}")

    print(f"Loading {data_path} ...")
    frame = load_dataframe(data_path)
    print(f"Loaded: {frame.shape[0]:,} rows x {frame.shape[1]:,} columns")

    target = detect_target(frame, args.target, allow_input)
    frame, note = ensure_binary(frame, target, args.positive, allow_input)
    total_events = int(frame[target].sum())
    total_pop = int(frame.shape[0])
    print(f"Target: '{target}' | positive events: {total_events:,} / {total_pop:,} "
          f"({total_events/total_pop*100:.2f}%)")
    if note:
        print(f"  ({note})")

    ignore_columns = resolve_ignore_columns(frame, args.ignore_columns, target)

    print("\nRunning RapidSegment (this can take a while) ...")
    rs_rows, rs_time, rs_stop = run_rapidsegment(frame, target, args, ignore_columns)
    print(f"  -> {len(rs_rows)} segments in {rs_time:.1f}s")

    print("Running decision tree ...")
    dt_rows, dt_time, dt_stop = run_decision_tree(frame, target, args, ignore_columns)
    print(f"  -> {len(dt_rows)} segments in {dt_time:.1f}s")

    print_table(f"RapidSegment ({args.rs_sort_priority}, reuse={args.max_feature_reuse}, "
                f"min_lift={args.rs_min_lift})", rs_rows)
    print_table(f"Decision Tree (min_lift={args.dt_min_lift})", dt_rows)

    lines = build_summary_lines(rs_rows, dt_rows, rs_time, dt_time, rs_stop, dt_stop,
                                total_events, total_pop, note)
    print("\n=== COMPARISON SUMMARY ===")
    print("\n".join(lines))

    if args.plot_out:
        plot_comparison(rs_rows, dt_rows, total_events / total_pop * 100, args.plot_out)

    if args.report_out:
        report = [
            "# RapidSegment vs Decision Tree comparison",
            "",
            f"- Dataset: `{data_path}`",
            f"- Target: `{target}`" + (f" (positive class = `{args.positive}`)" if note else ""),
            f"- Rows: {total_pop:,} | positive events: {total_events:,} "
            f"({total_events/total_pop*100:.2f}%)",
            f"- Ignored columns: {ignore_columns if ignore_columns else '(none)'}",
            f"- RapidSegment config: sort_priority=`{args.rs_sort_priority}`, "
            f"max_feature_reuse={args.max_feature_reuse}, min_lift={args.rs_min_lift}, "
            f"top_n_vars={args.top_n_vars}, max_segments={args.max_segments}",
            f"- Decision tree config: min_lift={args.dt_min_lift}, "
            f"min_events={args.dt_min_events}, min_sample_size={args.dt_min_sample_size}",
            "",
            "## Summary",
            "",
            "```",
        ]
        report += lines
        report.append("```")
        if args.plot_out:
            report += ["", f"![comparison]({Path(args.plot_out).name})"]
        Path(args.report_out).write_text("\n".join(report) + "\n")
        print(f"Report saved to {args.report_out}")


if __name__ == "__main__":
    main()
