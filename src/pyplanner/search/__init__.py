"""Búsqueda heurística hacia delante: configuración, motor y resultados."""

from pyplanner.search.config import (
    DEFAULT_CORES,
    DEFAULT_PORTFOLIO,
    PORTFOLIOS,
    PRESETS,
    ConfigError,
    SearchConfig,
    portfolio_for_cores,
)
from pyplanner.search.engine import search
from pyplanner.search.result import SearchResult, SearchStatistics, SearchStatus, StopSignal

__all__ = [
    "DEFAULT_CORES",
    "DEFAULT_PORTFOLIO",
    "PORTFOLIOS",
    "PRESETS",
    "ConfigError",
    "SearchConfig",
    "SearchResult",
    "SearchStatistics",
    "SearchStatus",
    "StopSignal",
    "portfolio_for_cores",
    "search",
]
