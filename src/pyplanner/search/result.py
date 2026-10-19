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
    FAILED = "failed"
    """Una búsqueda incompleta (IW) ha terminado sin plan y sin demostrar que no lo haya."""
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
    pruned: int = 0
    evaluation_batches: int = 0
    largest_batch: int = 0
    gpu_batches: int = 0
    time: float = 0.0

    def add(self, part: "SearchStatistics") -> None:
        """Suma los contadores de otra búsqueda (el tiempo no: lo mide quien las encadena)."""
        for name in ("expanded", "generated", "evaluated", "reopened", "dead_ends", "pruned"):
            setattr(self, name, getattr(self, name) + getattr(part, name))
        self.evaluation_batches += part.evaluation_batches
        self.largest_batch = max(self.largest_batch, part.largest_batch)
        self.gpu_batches += part.gpu_batches


@dataclass
class SearchResult:
    """Resultado de una búsqueda.

    Attributes:
        status: cómo terminó.
        plan: índices de las acciones del plan (en ``GroundTask.actions``), si hay plan.
        cost: coste del plan, si hay plan.
        config_name: configuración que lo produjo.
        message: detalle adicional (por ejemplo, el error en el portfolio).
        lower_bound: en una búsqueda *anytime*, coste por debajo del cual se ha demostrado
            que no hay plan (``None``: no se ha demostrado nada).
        interrupted: la búsqueda se detuvo antes de terminar (límite o parada externa) y
            el plan es el mejor que llevaba (su incumbente).
    """

    status: SearchStatus
    plan: tuple[int, ...] | None = None
    cost: float | None = None
    statistics: SearchStatistics = field(default_factory=SearchStatistics)
    config_name: str = ""
    message: str = ""
    lower_bound: float | None = None
    interrupted: bool = False

    @property
    def solved(self) -> bool:
        """``True`` si se ha encontrado un plan."""
        return self.status is SearchStatus.SOLVED

    @property
    def optimal(self) -> bool:
        """``True`` si se ha demostrado que el plan es óptimo."""
        return (
            self.cost is not None
            and self.lower_bound is not None
            and self.cost <= self.lower_bound + 1e-6
        )
