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
Exploration = Literal["none", "type-based"]


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
        exploration: ``"type-based"`` añade una cola de exploración por tipos (h, g) que
            se alterna con las demás; ayuda a salir de mesetas.
        seed: semilla del desempate aleatorio y de la exploración.
        batch_size: número de nodos que se expanden antes de evaluar juntos sus sucesores.
            Con 1, la búsqueda es la clásica.
        device: dónde se evalúa la heurística: ``"auto"`` (Numba si está instalado, si no
            Python), ``"python"``, ``"numba"``, ``"cpu"`` o ``"cuda"`` (PyTorch).
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
    exploration: Exploration = "none"
    seed: int = 0
    batch_size: int = 1
    device: str = "auto"
    max_time: float | None = None
    max_expansions: int | None = None

    def __post_init__(self) -> None:
        if self.algorithm not in ("gbfs", "astar", "wastar"):
            raise ConfigError(f"{self.name}: algoritmo desconocido '{self.algorithm}'")
        if self.heuristic not in HEURISTICS:
            raise ConfigError(f"{self.name}: heurística desconocida '{self.heuristic}'")
        if self.device not in DEVICES:
            raise ConfigError(f"{self.name}: dispositivo desconocido '{self.device}'")
        if self.device in ("cpu", "cuda") and self.heuristic not in TENSOR_HEURISTICS:
            raise ConfigError(
                f"{self.name}: la heurística '{self.heuristic}' no tiene implementación en PyTorch"
            )
        if self.tie_breaking not in ("fifo", "lifo", "random"):
            raise ConfigError(f"{self.name}: desempate desconocido '{self.tie_breaking}'")
        if self.exploration not in ("none", "type-based"):
            raise ConfigError(f"{self.name}: exploración desconocida '{self.exploration}'")
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
        SearchConfig(
            "gbfs-ff-pref-tb", "gbfs", "ff", preferred_operators=True, exploration="type-based"
        ),
        SearchConfig("lazy-ff-pref", "gbfs", "ff", preferred_operators=True, lazy=True),
        SearchConfig(
            "lazy-ff-pref-tb",
            "gbfs",
            "ff",
            preferred_operators=True,
            lazy=True,
            exploration="type-based",
            seed=1,
        ),
        SearchConfig("gbfs-add", "gbfs", "hadd"),
        SearchConfig("wastar-add-3", "wastar", "hadd", weight=3.0),
        SearchConfig("gbfs-goalcount-rnd", "gbfs", "goalcount", tie_breaking="random", seed=1),
        SearchConfig("astar-hmax", "astar", "hmax"),
        SearchConfig("astar-blind", "astar", "blind"),
    )
}

#: Portfolios por número de núcleos. Cada uno contiene al anterior y añade las
#: configuraciones que más diversidad aportan (búsqueda inmediata o perezosa, exploración,
#: heurística):
#:
#: * 1 núcleo: la configuración individual más robusta. La exploración por tipos evita que
#:   la búsqueda voraz se atasque en mesetas y zonas con muchos callejones sin salida;
#: * 2 núcleos: + la búsqueda perezosa sin exploración, la más rápida en tareas grandes y
#:   fáciles;
#: * 4 núcleos: + la más barata y aleatoria (objetivos pendientes) y la perezosa con
#:   exploración, con otra semilla;
#: * 5 núcleos: + WA* con h_add, que tiende a planes más cortos.
PORTFOLIOS: dict[int, tuple[str, ...]] = {
    1: ("gbfs-ff-pref-tb",),
    2: ("gbfs-ff-pref-tb", "lazy-ff-pref"),
    4: ("gbfs-ff-pref-tb", "lazy-ff-pref", "gbfs-goalcount-rnd", "lazy-ff-pref-tb"),
    5: (
        "gbfs-ff-pref-tb",
        "lazy-ff-pref",
        "gbfs-goalcount-rnd",
        "lazy-ff-pref-tb",
        "wastar-add-3",
    ),
}

#: Núcleos que usa el planificador si no se indica otra cosa (máxima robustez).
DEFAULT_CORES = 5

#: Portfolio por defecto (el de :data:`DEFAULT_CORES` núcleos).
DEFAULT_PORTFOLIO = PORTFOLIOS[DEFAULT_CORES]


def portfolio_for_cores(cores: int) -> tuple[SearchConfig, ...]:
    """Configuraciones que se ejecutan con ``cores`` núcleos.

    Se elige el portfolio más grande que cabe en los núcleos disponibles, de modo que nunca
    hay más búsquedas que núcleos. Con más núcleos que el portfolio más grande, se usa ese.

    Raises:
        ConfigError: si ``cores`` es menor que 1.
    """
    if cores < 1:
        raise ConfigError(f"el número de núcleos tiene que ser ≥ 1 (se ha pedido {cores})")
    size = max(size for size in PORTFOLIOS if size <= cores)
    return tuple(PRESETS[name] for name in PORTFOLIOS[size])
