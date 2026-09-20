from importlib.metadata import version, PackageNotFoundError
import logging
import os

try:
    __version__ = version("rapidsegment")
except PackageNotFoundError:
    # Fallback when running from source without install
    __version__ = "0.0.0+dev"

__author__ = "Bishwarup Biswas <bishwarup1429@gmail.com>"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# RapidSegment shows its progress logs (segment extraction, scoring, etc.) by
# default so interactive users can follow long-running jobs. This is done by
# attaching a handler to the library's OWN "StrategicEngine" logger — the root
# logger is NEVER touched, so host applications keep full control of their own
# logging configuration.
#
# To silence:        rapidsegment.disable_logging()
# To re-enable:      rapidsegment.enable_logging()
# To change level:   rapidsegment.enable_logging(level=logging.DEBUG)
# Or via env var:    RAPIDSEGMENT_LOG_LEVEL=WARNING
#                    (one of DEBUG/INFO/WARNING/ERROR/DISABLE)
_LOGGER_NAME = "StrategicEngine"
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | [%(filename)s:%(lineno)d] | %(message)s"


def enable_logging(level: int = logging.INFO) -> None:
    """Attach RapidSegment's progress handler to the ``StrategicEngine`` logger.

    Idempotent — calling repeatedly will not duplicate the handler. The handler
    is marked (``_rapidsegment_owned``) so :func:`disable_logging` removes only
    this one without touching handlers a host application may have added (e.g.
    the Streamlit Execution Console capture handler in ``ui/pages``).
    """
    log = logging.getLogger(_LOGGER_NAME)
    log.setLevel(level)
    if not any(getattr(h, "_rapidsegment_owned", False) for h in log.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        handler._rapidsegment_owned = True
        log.addHandler(handler)


def disable_logging() -> None:
    """Remove RapidSegment's default progress handler and quiet the logger.

    Bumps the ``StrategicEngine`` logger to ``WARNING`` so only warnings/errors
    surface afterwards. Re-enable with :func:`enable_logging`.
    """
    log = logging.getLogger(_LOGGER_NAME)
    log.handlers = [
        h for h in log.handlers if not getattr(h, "_rapidsegment_owned", False)
    ]
    log.setLevel(logging.WARNING)


# Enable by default, honouring an optional env-var override. This runs before
# the submodule imports below so the handler is in place as soon as the
# library's loggers are created.
_env_level = os.environ.get("RAPIDSEGMENT_LOG_LEVEL", "INFO").upper()
if _env_level in ("DISABLE", "QUIET", "OFF", "NONE"):
    disable_logging()
else:
    enable_logging(getattr(logging, _env_level, logging.INFO))


from .utils import UniversalDataLoader, duckdb_to_arrow
from .builder import StrategicSegmentBuilder
from .scorer import StrategicSegmentScore

__all__ = [
    "UniversalDataLoader",
    "duckdb_to_arrow",
    "StrategicSegmentBuilder",
    "StrategicSegmentScore",
    "enable_logging",
    "disable_logging",
]
