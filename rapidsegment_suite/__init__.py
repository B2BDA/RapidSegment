"""
RapidSegment Suite - Solara Package init
"""
from .db import SuiteDB
from .data_loader import SuiteDataLoader
from .builder_runner import RapidSegmentRunner, HAS_RAPIDSEGMENT
from .data_profiler_duckdb import DuckDBProfiler
from .workbench import Workbench
from .console import ArtifactConsole
from .leaderboard import Leaderboard
from .arena import Arena

__all__ = [
    "SuiteDB", "SuiteDataLoader", "RapidSegmentRunner", 
    "DuckDBProfiler", "Workbench", "ArtifactConsole", 
    "Leaderboard", "Arena", "HAS_RAPIDSEGMENT"
]
__version__ = "0.3.0-solara-nocode"
