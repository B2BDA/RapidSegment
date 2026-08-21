"""
Decision-Tree Hierarchical Segmentation
=======================================
End-to-end greedy decision-tree segment extraction on the bank dataset.

Workflow:
    1. Load the full dataset (no train/test split).
    2. Fit a decision tree on the full population and enumerate every
       root-to-leaf decision path.
    3. Pick the path with the maximum event capture (subject to minimum
       count / events / lift constraints).
    4. Remove the rows matching that path from the working population.
    5. Repeat on the residual until no valid decision path remains.
    6. Print a hierarchical report of all segments (event capture,
       response rate, lift, cumulative capture).
"""

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_string_dtype
from prettytable import PrettyTable
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from rapidsegment import UniversalDataLoader

DATA_PATH = "Examples/bank-full.csv"
TARGET_COL = "Target"

MAX_SEGMENTS = 12
MAX_DEPTH = 6
MIN_SAMPLES_LEAF = 1000
MIN_SAMPLE_SIZE = 1000
MIN_EVENTS = 100
MIN_LIFT = 1.3
CLASS_WEIGHT = "balanced"


def _simplify_conditions(conds):
    """
    Collapse redundant conditions from a root-to-leaf path. Multiple splits on
    the same numeric feature become a single tightest range; repeated
    categorical splits become the intersection of the allowed value sets.
    """
    num_bounds = {}
    cat_sets = {}
    for col, kind, op, value in conds:
        if kind == "num":
            bounds = num_bounds.setdefault(col, {"lo": float("-inf"), "hi": float("inf")})
            if op == "gt" and value > bounds["lo"]:
                bounds["lo"] = value
            elif op == "le" and value < bounds["hi"]:
                bounds["hi"] = value
        else:
            cat_sets.setdefault(col, set()).update(value)

    simplified = []
    for col, bounds in num_bounds.items():
        if bounds["lo"] > float("-inf"):
            simplified.append((col, "num", "gt", bounds["lo"]))
        if bounds["hi"] < float("inf"):
            simplified.append((col, "num", "le", bounds["hi"]))
    for col, values in cat_sets.items():
        if values:
            simplified.append((col, "cat", "in", values))
    return simplified


def _decode_conditions(frame: pd.DataFrame, conds):
    """
    Evaluate a list of node conditions against a DataFrame and return a boolean mask.
    Each condition is (col, kind, op, value) where kind is 'num' or 'cat'.
    """
    mask = np.ones(len(frame), dtype=bool)
    for col, kind, op, value in conds:
        if kind == "num":
            col_vals = frame[col].to_numpy()
            if op == "le":
                mask &= col_vals <= value
            else:
                mask &= col_vals > value
        else:
            allowed = value
            col_vals = frame[col].to_numpy()
            mask &= np.isin(col_vals, list(allowed))
    return mask


def _format_rule(conds, encoders) -> str:
    """
    Render a list of conditions as a compact, readable rule string.
    """
    parts = []
    for col, kind, op, value in conds:
        if kind == "cat":
            mapping = encoders[col]
            items = ", ".join(str(mapping[enc]) for enc in sorted(value))
            parts.append(f"{col} IN ({items})")
        else:
            parts.append(f"{col} {'<=' if op == 'le' else '>'} {value:g}")
    return " & ".join(parts)


def _extract_leaf_paths(tree, feature_names, encoders):
    """
    Enumerate all root-to-leaf decision paths of a fitted sklearn tree.

    Returns a list of paths, each a list of (col, kind, op, value) tuples
    where numeric values are split thresholds and categorical values are
    the encoded integer sets decoded to the categories on each side.
    """
    left = tree.children_left
    right = tree.children_right
    features = tree.feature
    thresholds = tree.threshold

    paths = []
    stack = [(0, [])]
    while stack:
        node, conds = stack.pop()
        if left[node] == right[node]:
            paths.append(conds)
            continue

        col = feature_names[features[node]]
        kind = "cat" if col in encoders else "num"
        thr = thresholds[node]

        if kind == "cat":
            mapping = encoders[col]
            enc_val = int(np.floor(thr))
            le_set = {enc for enc in mapping if enc <= enc_val}
            gt_set = {enc for enc in mapping if enc > enc_val}
            stack.append((left[node], conds + [(col, "cat", "le", le_set)]))
            stack.append((right[node], conds + [(col, "cat", "gt", gt_set)]))
        else:
            stack.append((left[node], conds + [(col, "num", "le", thr)]))
            stack.append((right[node], conds + [(col, "num", "gt", thr)]))

    return paths


def _best_path(frame, target, clf, base_rate, feature_names, encoders, constraints):
    """
    Fit a tree on the current population, evaluate every leaf path, and return
    the path with the maximum event capture among valid candidates.
    """
    X = frame.drop(columns=[target])
    y = frame[target].to_numpy()

    if X.shape[0] < MIN_SAMPLE_SIZE or y.sum() == 0:
        return None

    clf.fit(X, y)

    best = None
    for conds in _extract_leaf_paths(clf.tree_, feature_names, encoders):
        conds = _simplify_conditions(conds)
        mask = _decode_conditions(frame, conds)
        count = int(mask.sum())
        events = int(frame.loc[mask, target].sum())
        if count < constraints["min_sample_size"]:
            continue
        if events < constraints["min_events"]:
            continue
        rate = events / count if count else 0.0
        lift = rate / base_rate if base_rate > 0 else 0.0
        if lift < constraints["min_lift"]:
            continue
        candidate = {
            "conds": conds,
            "rule": _format_rule(conds, encoders),
            "count": count,
            "events": events,
            "rate": rate,
            "lift": lift,
        }
        if best is None or (candidate["events"], candidate["lift"], candidate["count"]) > (
            best["events"],
            best["lift"],
            best["count"],
        ):
            best = candidate

    return best


def run_decision_tree_segmentation(
    data: pd.DataFrame,
    target_col: str = TARGET_COL,
    max_segments: int = MAX_SEGMENTS,
    max_depth: int = MAX_DEPTH,
    min_samples_leaf: int = MIN_SAMPLES_LEAF,
    min_sample_size: int = MIN_SAMPLE_SIZE,
    min_events: int = MIN_EVENTS,
    min_lift: float = MIN_LIFT,
    class_weight: str = CLASS_WEIGHT,
) -> list:
    """
    Run greedy hierarchical decision-path extraction and return the segment list.

    The target column is expected to be binary. String targets are handled
    automatically only when the two classes are literally {"yes", "no"};
    boolean targets are mapped to 0/1. For any other encoding, binarize the
    target yourself (0/1) before calling this function.
    """
    frame = data.copy()

    if target_col in frame.columns:
        if is_string_dtype(frame[target_col]):
            unique_values = {v for v in frame[target_col].dropna().unique()}
            if unique_values <= {"yes", "no"}:
                frame[target_col] = (frame[target_col] == "yes").astype(int)
            else:
                raise ValueError(
                    f"Target column '{target_col}' has string values "
                    f"{sorted(unique_values)}. Convert it to 0/1 first "
                    f"(e.g. via compare_segmenters_general.py)."
                )
        elif is_bool_dtype(frame[target_col]):
            frame[target_col] = frame[target_col].astype(int)
    target = target_col

    categorical_cols = [c for c in frame.columns if is_string_dtype(frame[c])]
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        frame[col] = le.fit_transform(frame[col].astype(str))
        encoders[col] = {enc: orig for enc, orig in enumerate(le.classes_)}

    feature_names = [c for c in frame.columns if c != target]
    base_rate = frame[target].mean()
    total_events = int(frame[target].sum())

    constraints = {
        "min_sample_size": min_sample_size,
        "min_events": min_events,
        "min_lift": min_lift,
    }

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
    )

    segments = []
    remaining = np.ones(len(frame), dtype=bool)
    stop_reason = "Reached max_segments"

    for seg_id in range(1, max_segments + 1):
        residual = frame.loc[remaining]
        if residual.shape[0] < min_sample_size:
            stop_reason = "Residual population smaller than min_sample_size"
            break

        residual_base = residual[target].mean()
        best = _best_path(
            residual, target, clf, base_rate, feature_names, encoders, constraints
        )
        if best is None:
            stop_reason = "No valid decision path found on the residual"
            break

        matched = _decode_conditions(frame, best["conds"]) & remaining
        count = int(matched.sum())
        events = int(frame.loc[matched, target].sum())
        rate = events / count if count else 0.0
        lift = rate / base_rate if base_rate > 0 else 0.0

        segments.append(
            {
                "segment_id": seg_id,
                "rule_string": best["rule"],
                "count": count,
                "events": events,
                "response_rate": rate * 100.0,
                "lift": lift,
                "residual_base_rate": residual_base * 100.0,
            }
        )
        remaining &= ~matched

        if remaining.sum() < min_sample_size:
            stop_reason = "Remaining rows below min_sample_size"
            break

    segments.append(
        {
            "_stop_reason": stop_reason,
            "_total_events": total_events,
            "_total_population": int(frame.shape[0]),
        }
    )
    return segments


def print_report(segments: list) -> None:
    """
    Print the hierarchical segmentation report as a formatted table.
    """
    meta = segments[-1]
    segments = segments[:-1]
    total_events = meta["_total_events"]
    total_population = meta["_total_population"]

    table = PrettyTable()
    table.field_names = [
        "Segment",
        "Rule",
        "Count",
        "Events",
        "Resp Rate %",
        "Lift",
        "Residual Base %",
        "Cum Events",
        "Event Capture %",
        "Pop Capture %",
    ]

    cum_events = 0
    cum_count = 0
    for seg in segments:
        cum_events += seg["events"]
        cum_count += seg["count"]
        table.add_row(
            [
                seg["segment_id"],
                seg["rule_string"],
                seg["count"],
                seg["events"],
                round(seg["response_rate"], 2),
                round(seg["lift"], 2),
                round(seg["residual_base_rate"], 2),
                cum_events,
                round(cum_events / total_events * 100.0, 2),
                round(cum_count / total_population * 100.0, 2),
            ]
        )

    print(table)
    print(f"\nStop reason: {meta['_stop_reason']}")
    print(f"Total population: {total_population:,} | Total events: {total_events}")
    print(f"Total event capture: {cum_events} ({cum_events / total_events * 100.0:.2f}%)")


def main() -> None:
    """
    Load bank data, extract decision-tree segments, and print the report.
    """
    arrow_table = UniversalDataLoader(file_path=DATA_PATH).load()
    frame = arrow_table.to_pandas()

    segments = run_decision_tree_segmentation(frame)
    print_report(segments)


if __name__ == "__main__":
    main()
