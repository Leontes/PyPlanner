"""Resultado de una búsqueda y estadísticas."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol


class SearchStatus(Enum):
    """Cómo terminó una búsqueda."""

    SOLVED = "solved"
    """Se ha encontrado un plan."""
    UNSOLVABLE = "unsolvable"
    """Se ha demostrado que no hay plan (se agotó el espacio de búsqueda)."""
    LIMIT_REACHED = "limit_reached"
    """Se ha alcanzado un límite de tiempo o de expansiones."""
    CANCELLED = "cancelled"
    """Se ha detenido desde fuera (por ejemplo, otro miembro del portfolio encontró plan)."""
    ERROR = "error"
    """Ha fallado (solo en el portfolio: la excepción se captura en el proceso hijo)."""


class StopSignal(Protocol):
    """Señal de parada: ``threading.Event`` y ``multiprocessing.Event`` la cumplen."""

    def is_set(self) -> bool: ...


@dataclass
class SearchStatistics:
    """Contadores de una búsqueda."""

    expanded: int = 0
    generated: int = 0
    evaluated: int = 0
    reopened: int = 0
    dead_ends: int = 0
    evaluation_batches: int = 0
    time: float = 0.0


@dataclass
class SearchResult:
    """Resultado de una búsqueda.

    Attributes:
        status: cómo terminó.
        plan: índices de las acciones del plan (en ``GroundTask.actions``), si hay plan.
        cost: coste del plan, si hay plan.
        config_name: configuración que lo produjo.
        message: detalle adicional (por ejemplo, el error en el portfolio).
    """

    status: SearchStatus
    plan: tuple[int, ...] | None = None
    cost: float | None = None
    statistics: SearchStatistics = field(default_factory=SearchStatistics)
    config_name: str = ""
    message: str = ""

    @property
    def solved(self) -> bool:
        """``True`` si se ha encontrado un plan."""
        return self.status is SearchStatus.SOLVED
