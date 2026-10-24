"""Smoke tests for the RapidSegment engine.

These guard against dependency regressions (notably DuckDB / optbinning /
pandas / numpy) and provide a fast safety net when upgrading packages. They use
a tiny synthetic dataset so they run in CI in seconds without shipping real
data.
"""
import os
import tempfile
import warnings

import pandas as pd
import pytest


def test_core_imports():
    # Core library must import without the optional UI (streamlit) extra.
    import importlib.util

    import rapidsegment  # noqa: F401
    from rapidsegment import (  # noqa: F401
        StrategicSegmentBuilder,
        StrategicSegmentScore,
        UniversalDataLoader,
    )

    # The UI subpackage is only importable when streamlit is installed.
    if importlib.util.find_spec("streamlit") is not None:
        from rapidsegment.ui import run_ui  # noqa: F401


def test_duckdb_roundtrip():
    import duckdb

    con = duckdb.connect(":memory:")
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    con.register("t", df)
    rows = con.execute(
        "SELECT a, SUM(b) AS s FROM t GROUP BY a ORDER BY a"
    ).fetchall()
    assert rows == [(1, 4), (2, 5), (3, 6)]


def test_streamed_duckdb_arrow_roundtrip(tmp_path):
    """The persisted-loader Arrow path uses DuckDB 2.x relations without warnings."""
    from rapidsegment import UniversalDataLoader

    frame = pd.DataFrame({"id": [1, 2, 3], "target": [0, 1, 0]})
    db_path = UniversalDataLoader().stream_to_duckdb(
        str(tmp_path / "source.duckdb"),
        data=frame,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        table = UniversalDataLoader.duckdb_to_arrow(db_path)

    assert not any(
        "Arrow output on a DB-API result makes an extra copy" in str(item.message)
        for item in caught
    )
    assert table.column_names == ["id", "target"]
    assert table.num_rows == 3
    assert table["target"].to_pylist() == [0.0, 1.0, 0.0]


def test_tiny_segment_extraction():
    from rapidsegment import StrategicSegmentBuilder

    df = pd.DataFrame(
        {
            "target": ([1, 0] * 100),
            "age": ([20, 55, 23, 60, 25, 61, 30, 58, 28, 62] * 20),
            "balance": (
                [100, 5000, 200, 6000, 150, 5500, 300, 4800, 250, 5900] * 20
            ),
        }
    )

    with tempfile.TemporaryDirectory() as d:
        builder = StrategicSegmentBuilder(
            target="target",
            min_sample_size=50,
            min_lift=1.0,
            min_events=5,
            max_segments=3,
            top_n_vars=5,
            db_path=os.path.join(d, "smoke.duckdb"),
            db_temp_dir=os.path.join(d, "tmp"),
            expand_log_mode="none",
        )
        segments = builder.extract_segments(df)
        assert isinstance(segments, list)
        # If any segments were found, they must carry the expected keys.
        if segments:
            keys = set(segments[0].keys())
            assert {
                "segment_id",
                "rule_string",
                "sql_filter",
                "count",
                "rate",
                "lift",
            } <= keys


def test_scorer_weights(tmp_path, monkeypatch):
    from rapidsegment import StrategicSegmentScore

    df = pd.DataFrame(
        {
            "target": ([1, 0] * 50),
            "id": list(range(100)),
            "seg1": ([1, 0] * 50),
        }
    )

    # StrategicSegmentScore writes a transient .db in the CWD; isolate it.
    monkeypatch.chdir(tmp_path)
    scorer = StrategicSegmentScore(
        target_col="target", primary_key="id", segment_cols=["seg1"]
    )
    out = tmp_path / "score.json"
    scorer.calculate_and_export_weights(df, export_path=str(out))
    assert out.exists()
