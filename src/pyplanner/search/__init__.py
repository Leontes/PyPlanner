"""Búsqueda heurística hacia delante: configuración, motor y resultados."""

from pyplanner.search.config import (
    DEFAULT_PORTFOLIO,
    PRESETS,
    ConfigError,
    SearchConfig,
)
from pyplanner.search.engine import search
from pyplanner.search.result import SearchResult, SearchStatistics, SearchStatus, StopSignal

__all__ = [
    "DEFAULT_PORTFOLIO",
    "PRESETS",
    "ConfigError",
    "SearchConfig",
    "SearchResult",
    "SearchStatistics",
    "SearchStatus",
    "StopSignal",
    "search",
]
