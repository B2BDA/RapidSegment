"""Unit tests for pure functions and validation logic in RapidSegment.

These complement the smoke tests by exercising the rule-grammar→SQL
translator, the 14 sort-priority variants, strict sort_priority
validation, the SQL identifier quoting helper, the diagnostics funnel,
and the scorer's decile-reversal logic.
"""
import ast
import json
import logging
import os
import re
import tempfile

import pandas as pd
import pytest

from rapidsegment.builder import (
    StrategicSegmentBuilder,
    _quote_sql_ident,
    _quote_sql_string,
    setup_disk_backed_db,
)
from rapidsegment.scorer import StrategicSegmentScore


# ---------------------------------------------------------------------------
# SQL identifier / string quoting helpers
# ---------------------------------------------------------------------------
class TestQuoteHelpers:
    def test_basic_identifier(self):
        assert _quote_sql_ident("age") == '"age"'

    def test_reserved_word(self):
        assert _quote_sql_ident("default") == '"default"'

    def test_embedded_double_quote(self):
        assert _quote_sql_ident('a"b') == '"a""b"'

    def test_non_string_input(self):
        assert _quote_sql_ident(42) == '"42"'

    def test_string_literal_basic(self):
        assert _quote_sql_string("hello") == "'hello'"

    def test_string_literal_escape(self):
        assert _quote_sql_string("it's") == "'it''s'"


# ---------------------------------------------------------------------------
# sort_priority validation
# ---------------------------------------------------------------------------
class TestSortPriorityValidation:
    def test_default_accepted(self):
        b = StrategicSegmentBuilder(target="y")
        assert b.sort_priority == "rate_lift_count"

    @pytest.mark.parametrize(
        "priority",
        [
            "lift_count_rate",
            "count_lift_rate",
            "rate_lift_count",
            "lift_rate_count",
            "count_rate_lift",
            "rate_count_lift",
            "events_lift_rate",
            "events_rate_lift",
            "lift_events_rate",
            "rate_events_lift",
            "events_count_rate",
            "events_rate_count",
            "count_events_rate",
            "rate_events_count",
        ],
    )
    def test_all_valid_priorities(self, priority):
        b = StrategicSegmentBuilder(target="y", sort_priority=priority)
        assert b.sort_priority == priority

    def test_invalid_priority_raises(self):
        with pytest.raises(ValueError, match="sort_priority must be one of"):
            StrategicSegmentBuilder(target="y", sort_priority="bogus")

    def test_invalid_priority_empty_string(self):
        with pytest.raises(ValueError, match="sort_priority must be one of"):
            StrategicSegmentBuilder(target="y", sort_priority="")


# ---------------------------------------------------------------------------
# _get_sort_key — all 14 priority variants
# ---------------------------------------------------------------------------
class TestGetSortKey:
    RULE = {
        "lift": 2.5,
        "count": 1000,
        "rate": 5.0,
        "events": 50,
        "rule": "age=[20, 30)",
    }

    @pytest.mark.parametrize(
        "priority, expected_key",
        [
            ("lift_count_rate", (2.5, 1000, 5.0)),
            ("count_lift_rate", (1000, 2.5, 5.0)),
            ("rate_lift_count", (5.0, 2.5, 1000)),
            ("lift_rate_count", (2.5, 5.0, 1000)),
            ("count_rate_lift", (1000, 5.0, 2.5)),
            ("rate_count_lift", (5.0, 1000, 2.5)),
            ("events_lift_rate", (50, 2.5, 5.0)),
            ("events_rate_lift", (50, 5.0, 2.5)),
            ("lift_events_rate", (2.5, 50, 5.0)),
            ("rate_events_lift", (5.0, 50, 2.5)),
            ("events_count_rate", (50, 1000, 5.0)),
            ("events_rate_count", (50, 5.0, 1000)),
            ("count_events_rate", (1000, 50, 5.0)),
            ("rate_events_count", (5.0, 50, 1000)),
        ],
    )
    def test_sort_key_per_priority(self, priority, expected_key):
        b = StrategicSegmentBuilder(target="y", sort_priority=priority)
        key = b._get_sort_key(self.RULE)
        # The rule string is appended as a deterministic tie-breaker.
        assert key == expected_key + ("age=[20, 30)",)

    def test_sort_key_deterministic_tiebreaker(self):
        """Two rules with identical metrics but different rule strings
        must produce different sort keys (stable tie-breaking)."""
        b = StrategicSegmentBuilder(target="y", sort_priority="lift_count_rate")
        rule_a = {**self.RULE, "rule": "a=[1, 2)"}
        rule_b = {**self.RULE, "rule": "b=[1, 2)"}
        assert b._get_sort_key(rule_a) != b._get_sort_key(rule_b)


# ---------------------------------------------------------------------------
# parse_rule_to_sql — rule grammar → SQL predicate translator
# ---------------------------------------------------------------------------
class TestParseRuleToSql:
    def _make_builder(self, categorical_cols=None):
        b = StrategicSegmentBuilder(target="y")
        b._categorical_cols = set(categorical_cols or [])
        return b

    def test_numeric_range_half_open(self):
        b = self._make_builder()
        sql = b.parse_rule_to_sql("age=[20, 30)")
        assert ">= 20" in sql
        assert "< 30" in sql

    def test_numeric_range_closed(self):
        b = self._make_builder()
        sql = b.parse_rule_to_sql("age=[20, 30]")
        assert ">= 20" in sql
        assert "<= 30" in sql

    def test_numeric_range_open_left(self):
        b = self._make_builder()
        sql = b.parse_rule_to_sql("age=(20, 30]")
        assert "> 20" in sql
        assert "<= 30" in sql

    def test_single_numeric_value(self):
        b = self._make_builder()
        sql = b.parse_rule_to_sql("age=[42]")
        assert '"age"' in sql
        assert "42" in sql

    def test_categorical_single_value(self):
        b = self._make_builder(categorical_cols=["gender"])
        sql = b.parse_rule_to_sql("gender=[female]")
        assert "'female'" in sql

    def test_categorical_set(self):
        b = self._make_builder(categorical_cols=["city"])
        sql = b.parse_rule_to_sql("city=[NYC, LA, SF]")
        assert "IN" in sql
        assert "'NYC'" in sql and "'LA'" in sql and "'SF'" in sql

    def test_special_missing(self):
        b = self._make_builder()
        sql = b.parse_rule_to_sql("age=Missing")
        assert "IS NULL" in sql

    def test_special_special(self):
        b = self._make_builder()
        sql = b.parse_rule_to_sql("age=Special")
        assert "IS NULL" in sql

    def test_nested_single_category_guard(self):
        """[[VIP]] must be treated as a single literal category value,
        not a multi-value merge."""
        b = self._make_builder(categorical_cols=["tier"])
        sql = b.parse_rule_to_sql("tier=[[VIP]]")
        # Should not produce an IN clause (it's a single literal value).
        assert "IN" not in sql

    def test_merged_categorical_list(self):
        b = self._make_builder(categorical_cols=["tier"])
        sql = b.parse_rule_to_sql("tier=[[male],[female]]")
        assert "IN" in sql
        assert "'male'" in sql and "'female'" in sql

    def test_multi_range_numeric(self):
        b = self._make_builder()
        sql = b.parse_rule_to_sql("income=[[10000, 20000), [20000, 30000)]")
        assert ">= 10000" in sql
        assert "30000" in sql

    def test_combined_rules(self):
        b = self._make_builder(categorical_cols=["gender"])
        sql = b.parse_rule_to_sql("age=[20, 30) & gender=[female]")
        assert ">= 20" in sql and "< 30" in sql
        assert "'female'" in sql

    def test_reserved_word_column(self):
        """A column named 'default' (a SQL reserved word) must be
        properly quoted so the generated SQL is valid."""
        b = self._make_builder()
        sql = b.parse_rule_to_sql("default=[1, 5)")
        assert '"default"' in sql

    def test_column_with_embedded_quote(self):
        """A column name containing a double quote must be escaped."""
        b = self._make_builder()
        sql = b.parse_rule_to_sql('a"b=[1, 5)')
        assert '"a""b"' in sql


# ---------------------------------------------------------------------------
# setup_disk_backed_db — uses system temp dir by default
# ---------------------------------------------------------------------------
class TestSetupDiskBackedDb:
    def test_default_uses_temp_dir(self):
        cwd_before = os.getcwd()
        db_path, temp_dir = setup_disk_backed_db()
        try:
            # Both paths must be under the system temp dir, not the CWD.
            assert db_path.startswith(tempfile.gettempdir())
            assert temp_dir.startswith(tempfile.gettempdir())
            # The CWD must not contain an experiments/ directory.
            assert not os.path.exists(os.path.join(cwd_before, "experiments"))
            # The temp directory must actually be created.
            assert os.path.isdir(temp_dir)
        finally:
            import shutil
            base = os.path.dirname(temp_dir)
            if os.path.exists(base):
                shutil.rmtree(base, ignore_errors=True)

    def test_custom_base_dir(self):
        with tempfile.TemporaryDirectory() as d:
            db_path, temp_dir = setup_disk_backed_db(d)
            assert db_path.startswith(d)
            assert temp_dir.startswith(d)
            assert os.path.isdir(temp_dir)


# ---------------------------------------------------------------------------
# Diagnostics funnel — per-level candidate counts
# ---------------------------------------------------------------------------
class TestCandidateFunnel:
    def test_funnel_has_per_level_counts(self):
        """After extraction, every diagnostic iteration that ran candidate
        generation must have a funnel dict with the 1-way / 2-way / 3-way
        counts that explain_no_segments() reads."""
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
                min_sample_size=10,
                min_lift=1.0,
                min_events=1,
                max_segments=1,
                top_n_vars=5,
                binning_method="naive",
                naive_bins=5,
                db_path=os.path.join(d, "funnel.duckdb"),
                db_temp_dir=os.path.join(d, "tmp"),
                expand_log_mode="none",
            )
            builder.extract_segments(df)
            assert builder.diagnostics_, "Diagnostics should be populated"
            # Find the first iteration that reached candidate generation
            # (iterations that hit "features exhausted" early don't have a funnel).
            funnel = None
            for diag in builder.diagnostics_:
                if "candidate_funnel" in diag:
                    funnel = diag["candidate_funnel"]
                    break
            assert funnel is not None, (
                "At least one iteration should have a candidate_funnel"
            )
            # The per-level keys must be present (not silently 0/0/0).
            assert "1way_candidates" in funnel
            assert "2way_candidates" in funnel
            assert "3way_candidates" in funnel
            # At least the 1-way count should be > 0 when features are binned.
            assert funnel["1way_candidates"] > 0, (
                "1-way candidates should be > 0 when features are binned"
            )


# ---------------------------------------------------------------------------
# Scorer — decile reversal logic
# ---------------------------------------------------------------------------
class TestScorerDecileReversal:
    def test_decile_1_is_highest_threshold(self, tmp_path):
        """Decile 1 (top 10%) must have the highest score threshold."""
        import numpy as np

        rng = np.random.RandomState(42)
        n = 1000
        df = pd.DataFrame(
            {
                "target": rng.randint(0, 2, n),
                "id": list(range(n)),
                "seg1": rng.randint(0, 2, n),
                "seg2": rng.randint(0, 2, n),
                "seg3": rng.randint(0, 2, n),
            }
        )

        scorer = StrategicSegmentScore(
            target_col="target",
            primary_key="id",
            segment_cols=["seg1", "seg2", "seg3"],
        )
        out = tmp_path / "score.json"
        scorer.calculate_and_export_weights(df, export_path=str(out))

        with open(out) as f:
            artifact = json.load(f)
        thresholds = artifact["decile_min_thresholds"]
        assert len(thresholds) == 10
        # Decile 1 (top) must have the highest threshold.
        assert int(thresholds["1"]) >= int(thresholds["2"])
        assert int(thresholds["2"]) >= int(thresholds["3"])
        # Decile 10 (bottom) must have the lowest.
        assert int(thresholds["10"]) <= int(thresholds["9"])


# ---------------------------------------------------------------------------
# No logging.basicConfig at import time (library anti-pattern)
# ---------------------------------------------------------------------------
class TestNoBasicConfig:
    """Verify that no module calls logging.basicConfig() at import time.
    Uses AST to find actual function calls, not just string matching
    (which would false-positive on comments that mention the function name).
    """

    @staticmethod
    def _has_basic_config_call(filepath: str) -> bool:
        tree = ast.parse(open(filepath, encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if (
                    isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "logging"
                    and func.attr == "basicConfig"
                ):
                    return True
        return False

    def test_builder_no_basic_config(self):
        import rapidsegment.builder as mod
        assert not self._has_basic_config_call(mod.__file__), (
            "builder.py must not call logging.basicConfig()"
        )

    def test_scorer_no_basic_config(self):
        import rapidsegment.scorer as mod
        assert not self._has_basic_config_call(mod.__file__), (
            "scorer.py must not call logging.basicConfig()"
        )

    def test_bq_selector_no_basic_config(self):
        import rapidsegment.utils.on_gcp_feature_selection as mod
        assert not self._has_basic_config_call(mod.__file__), (
            "on_gcp_feature_selection.py must not call logging.basicConfig()"
        )


# ---------------------------------------------------------------------------
# Broken webui entry point removed from pyproject.toml
# ---------------------------------------------------------------------------
class TestNoBrokenEntryPoint:
    def test_webui_entry_point_removed(self):
        import rapidsegment

        # rapidsegment.__file__ is at src/rapidsegment/__init__.py;
        # pyproject.toml is two directories up from the package dir.
        toml_path = os.path.normpath(
            os.path.join(
                os.path.dirname(rapidsegment.__file__),
                "..",
                "..",
                "pyproject.toml",
            )
        )
        with open(toml_path) as f:
            content = f.read()
        assert "rapidsegment-webui" not in content, (
            "The broken rapidsegment-webui entry point must be removed"
        )


# ---------------------------------------------------------------------------
# Progress logging is visible by default WITHOUT hijacking the root logger
# ---------------------------------------------------------------------------
class TestDefaultLogging:
    """Verify progress logging is enabled by default on the library's OWN
    ``StrategicEngine`` logger, and that the host application's root logger is
    never touched."""

    def test_handler_attached_by_default(self):
        import rapidsegment

        log = logging.getLogger("StrategicEngine")
        owned = [h for h in log.handlers if getattr(h, "_rapidsegment_owned", False)]
        assert owned, "RapidSegment should attach a progress handler by default"
        assert isinstance(owned[0], logging.StreamHandler)
        assert log.level <= logging.INFO, "INFO progress must be visible by default"

    def test_root_logger_not_touched(self):
        # The whole point of the fix: we attach to our OWN named logger, never
        # the root, so host apps keep full control of their own logging.
        import rapidsegment  # noqa: F401

        root = logging.getLogger()
        owned_on_root = [
            h for h in root.handlers if getattr(h, "_rapidsegment_owned", False)
        ]
        assert not owned_on_root, (
            "RapidSegment must not attach handlers to the root logger"
        )

    def test_no_basic_config_in_init(self):
        # __init__ must configure logging via its own helper, not basicConfig.
        import rapidsegment

        tree = ast.parse(open(rapidsegment.__file__, encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if (
                    isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "logging"
                    and func.attr == "basicConfig"
                ):
                    pytest.fail("__init__.py must not call logging.basicConfig()")
        assert hasattr(rapidsegment, "enable_logging") and hasattr(
            rapidsegment, "disable_logging"
        )

    def test_enable_logging_is_idempotent(self):
        import rapidsegment

        log = logging.getLogger("StrategicEngine")
        before = len(log.handlers)
        rapidsegment.enable_logging()  # second call
        rapidsegment.enable_logging(level=logging.DEBUG)
        after = len(log.handlers)
        assert after == before, "enable_logging must not duplicate its handler"

    def test_disable_logging_removes_owned_handler(self):
        import rapidsegment

        rapidsegment.enable_logging()  # ensure on
        log = logging.getLogger("StrategicEngine")
        assert any(getattr(h, "_rapidsegment_owned", False) for h in log.handlers)
        rapidsegment.disable_logging()
        assert not any(
            getattr(h, "_rapidsegment_owned", False) for h in log.handlers
        )
        assert log.level >= logging.WARNING
        # re-enable for any subsequent tests
        rapidsegment.enable_logging()

    def test_host_handler_survives_disable(self):
        # A handler added by the host app (e.g. the UI Execution Console
        # capture handler) must survive rapidsegment.disable_logging().
        import rapidsegment

        log = logging.getLogger("StrategicEngine")
        host_handler = logging.StreamHandler()
        log.addHandler(host_handler)
        try:
            rapidsegment.disable_logging()
            assert host_handler in log.handlers, (
                "disable_logging must only remove its own owned handler"
            )
        finally:
            log.removeHandler(host_handler)
            rapidsegment.enable_logging()

    def test_records_actually_emit(self):
        import rapidsegment  # noqa: F401

        log = logging.getLogger("StrategicEngine")
        captured = []

        class _Capture(logging.Handler):
            def emit(self, record):
                captured.append(record.getMessage())

        cap = _Capture()
        log.addHandler(cap)
        try:
            log.info("progress-test-message")
            assert any("progress-test-message" in m for m in captured), (
                "INFO progress records must emit to the StrategicEngine logger"
            )
        finally:
            log.removeHandler(cap)


# ---------------------------------------------------------------------------
# _state.py — shared helpers (require streamlit[ui] extra)
# ---------------------------------------------------------------------------

try:
    import streamlit as _st  # noqa: F401
    _has_streamlit = True
except ImportError:
    _has_streamlit = False


@pytest.mark.skipif(not _has_streamlit, reason="requires streamlit[ui] extra")
class TestStateHelpers:
    def test_jsonable_dict(self):
        from rapidsegment.ui._state import _jsonable
        assert _jsonable({"a": 1, "b": "x"}) == {"a": 1, "b": "x"}

    def test_jsonable_nested(self):
        from rapidsegment.ui._state import _jsonable
        assert _jsonable({"a": [1, 2], "b": {"c": 3}}) == {"a": [1, 2], "b": {"c": 3}}

    def test_jsonable_none(self):
        from rapidsegment.ui._state import _jsonable
        assert _jsonable(None) is None

    def test_jsonable_numpy_scalar(self):
        import numpy as np
        from rapidsegment.ui._state import _jsonable
        result = _jsonable(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_jsonable_unknown_type(self):
        from rapidsegment.ui._state import _jsonable
        result = _jsonable(object())
        assert isinstance(result, str)

    def test_fmt_duration_seconds(self):
        from rapidsegment.ui._state import fmt_duration
        assert fmt_duration(30) == "30s"

    def test_fmt_duration_minutes(self):
        from rapidsegment.ui._state import fmt_duration
        assert fmt_duration(125) == "2m 05s"

    def test_fmt_duration_hours(self):
        from rapidsegment.ui._state import fmt_duration
        assert fmt_duration(3661) == "1h 01m"

    def test_fmt_duration_zero(self):
        from rapidsegment.ui._state import fmt_duration
        assert fmt_duration(0) == "0s"

    def test_exp_cols_includes_dataset_name(self):
        from rapidsegment.ui._state import EXP_COLS
        assert "dataset_name" in EXP_COLS

    def test_suite_dir_created(self):
        from rapidsegment.ui._state import SUITE_DIR
        import os
        assert os.path.isdir(SUITE_DIR)


# ---------------------------------------------------------------------------
# _state.py SQL quoting used in UI
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _has_streamlit, reason="requires streamlit[ui] extra")
class TestUISQLQuoting:
    """Verify that UI pages import _quote_sql_ident and use it correctly.

    These tests require streamlit (UI extra) to be installed.
    """

    def test_page1_imports_quote_ident(self):
        pytest.importorskip("streamlit", reason="requires streamlit[ui] extra")
        import importlib
        mod = importlib.import_module("rapidsegment.ui.pages.1_Data_Loader")
        assert hasattr(mod, "_quote_sql_ident")

    def test_page3_imports_quote_ident(self):
        pytest.importorskip("streamlit", reason="requires streamlit[ui] extra")
        import importlib
        mod = importlib.import_module("rapidsegment.ui.pages.3_Execution_Console")
        assert hasattr(mod, "_quote_sql_ident")

    def test_page4_imports_quote_ident(self):
        pytest.importorskip("streamlit", reason="requires streamlit[ui] extra")
        import importlib
        mod = importlib.import_module("rapidsegment.ui.pages.4_Results_Dashboard")
        assert hasattr(mod, "_quote_sql_ident")
