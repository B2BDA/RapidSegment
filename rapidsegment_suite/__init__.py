"""
RapidSegment Suite - Solara Package init.

Heavy modules (duckdb/solara-backed) are loaded lazily via __getattr__ so that
``import rapidsegment_suite`` stays light and does not trigger an eager
``import duckdb`` during solara's autorouting module execution (which can race
and produce ``ModuleNotFoundError: No module named '_duckdb._sqltypes'``).
"""
__version__ = "0.4.0-solara-nocode"

_LAZY_EXPORTS = {
    "SuiteDB": "db",
    "SuiteDataLoader": "data_loader",
    "RapidSegmentRunner": "builder_runner",
    "HAS_RAPIDSEGMENT": "builder_runner",
    "DuckDBProfiler": "data_profiler_duckdb",
    "Workbench": "workbench",
    "ArtifactConsole": "console",
    "Leaderboard": "leaderboard",
    "Arena": "arena",
    "DataSourceModule": "module1_data_source",
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        import importlib
        mod = importlib.import_module(f".{_LAZY_EXPORTS[name]}", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
