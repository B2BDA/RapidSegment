from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("rapidsegment")
except PackageNotFoundError:
    # Fallback when running from source without install
    __version__ = "0.0.0+dev"

__author__ = "Bishwarup Biswas <bishwarup1429@gmail.com>"

from .utils import UniversalDataLoader, duckdb_to_arrow
from .builder import StrategicSegmentBuilder
from .scorer import StrategicSegmentScore

__all__ = ["UniversalDataLoader", "duckdb_to_arrow", "StrategicSegmentBuilder", "StrategicSegmentScore"]