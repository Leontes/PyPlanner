"""Configuración de una búsqueda.

Una búsqueda queda completamente definida por un :class:`SearchConfig`. Es un objeto
inmutable y serializable (se envía tal cual a los procesos del portfolio), y se valida al
crearlo, de modo que una configuración incoherente falla antes de empezar a buscar.
"""

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

from pyplanner.errors import PDDLError
from pyplanner.heuristics import DEVICES, HEURISTICS, PREFERRED_HEURISTICS, TENSOR_HEURISTICS

Algorithm = Literal["gbfs", "astar", "wastar"]
TieBreaking = Literal["fifo", "lifo", "random"]


class ConfigError(PDDLError):
    """Configuración de búsqueda no válida."""


@dataclass(frozen=True)
class SearchConfig:
    """Parámetros de una búsqueda.

    Attributes:
        name: nombre para identificarla (en el portfolio y en las estadísticas).
        algorithm: ``"gbfs"`` (voraz, prioridad h), ``"astar"`` (g + h) o ``"wastar"``
            (g + weight · h).
        heuristic: nombre de la heurística (ver :data:`pyplanner.heuristics.HEURISTICS`).
        weight: peso de la heurística en WA* (≥ 1).
        preferred_operators: usa una segunda cola con los sucesores generados por acciones
            útiles y alterna entre las dos. Necesita una heurística que las calcule.
        lazy: evaluación perezosa. Los sucesores entran en la cola con el valor heurístico
            del padre y solo se evalúan al sacarlos. No es compatible con A*.
        tie_breaking: desempate entre nodos con la misma prioridad: ``"fifo"``,
            ``"lifo"`` o ``"random"`` (con ``seed``).
        seed: semilla del desempate aleatorio.
        batch_size: número de nodos que se expanden antes de evaluar juntos sus sucesores.
            Con 1, la búsqueda es la clásica.
        device: dónde se evalúa la heurística: ``"python"``, ``"cpu"`` o ``"cuda"``.
        max_time: límite de tiempo en segundos (``None``: sin límite).
        max_expansions: límite de expansiones (``None``: sin límite).
    """

    name: str = "gbfs-ff"
    algorithm: Algorithm = "gbfs"
    heuristic: str = "ff"
    weight: float = 1.0
    preferred_operators: bool = False
    lazy: bool = False
    tie_breaking: TieBreaking = "fifo"
    seed: int = 0
    batch_size: int = 1
    device: str = "python"
    max_time: float | None = None
    max_expansions: int | None = None

    def __post_init__(self) -> None:
        if self.algorithm not in ("gbfs", "astar", "wastar"):
            raise ConfigError(f"{self.name}: algoritmo desconocido '{self.algorithm}'")
        if self.heuristic not in HEURISTICS:
            raise ConfigError(f"{self.name}: heurística desconocida '{self.heuristic}'")
        if self.device not in DEVICES:
            raise ConfigError(f"{self.name}: dispositivo desconocido '{self.device}'")
        if self.device != "python" and self.heuristic not in TENSOR_HEURISTICS:
            raise ConfigError(
                f"{self.name}: la heurística '{self.heuristic}' no tiene implementación en PyTorch"
            )
        if self.tie_breaking not in ("fifo", "lifo", "random"):
            raise ConfigError(f"{self.name}: desempate desconocido '{self.tie_breaking}'")
        if self.algorithm == "wastar" and self.weight < 1:
            raise ConfigError(f"{self.name}: el peso de WA* tiene que ser ≥ 1")
        if self.preferred_operators and self.heuristic not in PREFERRED_HEURISTICS:
            raise ConfigError(
                f"{self.name}: la heurística '{self.heuristic}' no calcula acciones útiles"
            )
        if self.lazy and self.algorithm == "astar":
            raise ConfigError(f"{self.name}: A* no admite evaluación perezosa")
        if self.batch_size < 1:
            raise ConfigError(f"{self.name}: batch_size tiene que ser ≥ 1")
        if self.max_time is not None and self.max_time <= 0:
            raise ConfigError(f"{self.name}: max_time tiene que ser positivo")
        if self.max_expansions is not None and self.max_expansions < 0:
            raise ConfigError(f"{self.name}: max_expansions no puede ser negativo")

    @property
    def is_optimal(self) -> bool:
        """``True`` si la configuración garantiza planes óptimos (A* con h admisible)."""
        return self.algorithm == "astar" and self.heuristic in ("blind", "hmax")

    def with_changes(self, **changes: Any) -> "SearchConfig":
        """Copia de la configuración con algunos campos cambiados (se vuelve a validar)."""
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """La configuración como diccionario (para mostrarla o guardarla en JSON)."""
        return asdict(self)


#: Configuraciones predefinidas, por nombre.
PRESETS: dict[str, SearchConfig] = {
    config.name: config
    for config in (
        SearchConfig("gbfs-ff", "gbfs", "ff"),
        SearchConfig("gbfs-ff-pref", "gbfs", "ff", preferred_operators=True),
        SearchConfig("lazy-ff-pref", "gbfs", "ff", preferred_operators=True, lazy=True),
        SearchConfig("gbfs-add", "gbfs", "hadd"),
        SearchConfig("wastar-add-3", "wastar", "hadd", weight=3.0),
        SearchConfig("gbfs-goalcount-rnd", "gbfs", "goalcount", tie_breaking="random", seed=1),
        SearchConfig("astar-hmax", "astar", "hmax"),
        SearchConfig("astar-blind", "astar", "blind"),
    )
}

#: Portfolio por defecto: configuraciones diversas y no óptimas, pensadas para encontrar
#: un plan cuanto antes.
DEFAULT_PORTFOLIO = (
    "gbfs-ff-pref",
    "lazy-ff-pref",
    "gbfs-add",
    "wastar-add-3",
    "gbfs-goalcount-rnd",
)
