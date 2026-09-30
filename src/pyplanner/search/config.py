"""Configuración de una búsqueda.

Una búsqueda queda completamente definida por un :class:`SearchConfig`. Es un objeto
inmutable y serializable (se envía tal cual a los procesos del portfolio), y se valida al
crearlo, de modo que una configuración incoherente falla antes de empezar a buscar.
"""

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

from pyplanner.errors import PDDLError
from pyplanner.heuristics import DEVICES, HEURISTICS, PREFERRED_HEURISTICS, TENSOR_HEURISTICS

Algorithm = Literal["gbfs", "astar", "wastar", "iw"]
TieBreaking = Literal["fifo", "lifo", "random"]
Exploration = Literal["none", "type-based"]


class ConfigError(PDDLError):
    """Configuración de búsqueda no válida."""


@dataclass(frozen=True)
class SearchConfig:
    """Parámetros de una búsqueda.

    Attributes:
        name: nombre para identificarla (en el portfolio y en las estadísticas).
        algorithm: ``"gbfs"`` (voraz, prioridad h), ``"astar"`` (g + h), ``"wastar"``
            (g + weight · h) o ``"iw"`` (búsqueda en anchura que poda los estados de
            novedad mayor que ``novelty``; incompleta).
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
        novelty: anchura de la novedad (0: no se usa; 1 o 2). Con ``"gbfs"`` es BFWS: la
            prioridad es (novedad, h), con la novedad medida entre los estados del mismo
            valor h. Con ``"iw"`` es la k de IW(k). Ver :mod:`pyplanner.search.novelty`.
        seed: semilla del desempate aleatorio y de la exploración.
        batch_size: número de nodos que se expanden antes de evaluar juntos sus sucesores.
            Con 1, la búsqueda es la clásica. Con ``adaptive_batch``, es el tamaño mínimo.
        adaptive_batch: tamaño de lote adaptativo: vuelve a ``batch_size`` cuando mejora la
            mejor h vista y se duplica (hasta ``max_batch_size``) tras ``plateau_patience``
            lotes seguidos sin mejorarla. Lotes pequeños mientras la heurística guía y
            grandes en las mesetas, donde aporta poco y conviene evaluar mucho de golpe.
        max_batch_size: tamaño máximo del lote adaptativo.
        plateau_patience: lotes sin mejora antes de duplicar el tamaño.
        device: dónde se evalúa la heurística: ``"auto"`` (Numba si está instalado, si no
            Python), ``"python"``, ``"numba"``, ``"cpu"``, ``"cuda"`` (PyTorch) o
            ``"hybrid"`` (cada lote en Numba o en la GPU según su tamaño; ver
            :mod:`pyplanner.heuristics.hybrid`).
        max_time: límite de tiempo en segundos (``None``: sin límite).
        max_expansions: límite de expansiones (``None``: sin límite).
        max_memory: límite de memoria residente del proceso, en MB (``None``: sin límite).
            Al superarlo, la búsqueda termina con ``LIMIT_REACHED``.
    """

    name: str = "gbfs-ff"
    algorithm: Algorithm = "gbfs"
    heuristic: str = "ff"
    weight: float = 1.0
    preferred_operators: bool = False
    lazy: bool = False
    tie_breaking: TieBreaking = "fifo"
    exploration: Exploration = "none"
    novelty: int = 0
    seed: int = 0
    batch_size: int = 1
    adaptive_batch: bool = False
    max_batch_size: int = 256
    plateau_patience: int = 8
    device: str = "auto"
    max_time: float | None = None
    max_expansions: int | None = None
    max_memory: int | None = None

    def __post_init__(self) -> None:
        if self.algorithm not in ("gbfs", "astar", "wastar", "iw"):
            raise ConfigError(f"{self.name}: algoritmo desconocido '{self.algorithm}'")
        if self.heuristic not in HEURISTICS:
            raise ConfigError(f"{self.name}: heurística desconocida '{self.heuristic}'")
        if self.device not in DEVICES:
            raise ConfigError(f"{self.name}: dispositivo desconocido '{self.device}'")
        if self.device in ("cpu", "cuda", "hybrid") and self.heuristic not in TENSOR_HEURISTICS:
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
        if self.novelty not in (0, 1, 2):
            raise ConfigError(f"{self.name}: la novedad tiene que ser 0, 1 o 2")
        if self.algorithm == "iw" and not self.novelty:
            raise ConfigError(f"{self.name}: IW necesita novelty = 1 o 2 (la k de IW(k))")
        if self.novelty and self.algorithm not in ("gbfs", "iw"):
            raise ConfigError(f"{self.name}: la novedad solo se usa con GBFS (BFWS) o con IW")
        if self.novelty and self.lazy:
            raise ConfigError(f"{self.name}: la novedad necesita evaluación inmediata")
        if self.batch_size < 1:
            raise ConfigError(f"{self.name}: batch_size tiene que ser ≥ 1")
        if self.max_batch_size < self.batch_size:
            raise ConfigError(f"{self.name}: max_batch_size tiene que ser ≥ batch_size")
        if self.plateau_patience < 1:
            raise ConfigError(f"{self.name}: plateau_patience tiene que ser ≥ 1")
        if self.max_time is not None and self.max_time <= 0:
            raise ConfigError(f"{self.name}: max_time tiene que ser positivo")
        if self.max_expansions is not None and self.max_expansions < 0:
            raise ConfigError(f"{self.name}: max_expansions no puede ser negativo")
        if self.max_memory is not None and self.max_memory <= 0:
            raise ConfigError(f"{self.name}: max_memory tiene que ser positivo")

    @property
    def is_complete(self) -> bool:
        """``True`` si encuentra plan siempre que lo haya (todas salvo IW, que poda)."""
        return self.algorithm != "iw"

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
        SearchConfig("bfws-ff", "gbfs", "ff", novelty=2),
        SearchConfig("bfws-ff-pref", "gbfs", "ff", preferred_operators=True, novelty=2),
        SearchConfig("bfws-goalcount", "gbfs", "goalcount", novelty=2),
        SearchConfig(
            "gbfs-ff-pref-ab", "gbfs", "ff", preferred_operators=True, adaptive_batch=True
        ),
        SearchConfig(
            "gbfs-ff-pref-hybrid",
            "gbfs",
            "ff",
            preferred_operators=True,
            adaptive_batch=True,
            device="hybrid",
        ),
        SearchConfig(
            "bfws-ff-pref-hybrid",
            "gbfs",
            "ff",
            preferred_operators=True,
            novelty=2,
            adaptive_batch=True,
            device="hybrid",
        ),
        SearchConfig("iw1", "iw", "blind", novelty=1),
        SearchConfig("iw2", "iw", "blind", novelty=2),
        SearchConfig("astar-hmax", "astar", "hmax"),
        SearchConfig("astar-blind", "astar", "blind"),
    )
}

#: Portfolios por número de núcleos. Cada uno contiene al anterior y añade las
#: configuraciones que más aportan. Elegidos en la iteración 3b con los tiers 1 a 3 y las
#: instancias de driverlog y zenotravel temporales hasta la 17 (ver motor-busqueda.md):
#:
#: * 1 núcleo: BFWS con h_FF y acciones útiles, la configuración individual más robusta y
#:   rápida en esas pruebas;
#: * 2 núcleos: + la búsqueda perezosa, la más rápida en tareas grandes y fáciles;
#: * 4 núcleos: + la perezosa con exploración por tipos (otra semilla) y la más barata y
#:   aleatoria (objetivos pendientes);
#: * 5 núcleos: + GBFS con exploración por tipos, la más rápida en slitherlink;
#: * 6 núcleos (por defecto): + GBFS con lote adaptativo y evaluación híbrida (Numba o
#:   GPU según el lote), el único miembro con GPU: en las pruebas es la mejor sexta
#:   configuración (de 100,9 s a 93,8 s). Sin GPU, evalúa todo con Numba;
#: * 7 núcleos: + WA* con h_add, que no mejora los tiempos pero tiende a planes más
#:   cortos: diversidad a cambio de un núcleo más.
#:
#: Los portfolios de 1 a 5 núcleos no usan la GPU: funcionan igual en cualquier máquina y
#: no pagan el arranque de PyTorch y CUDA (unos segundos) en las tareas pequeñas.
PORTFOLIOS: dict[int, tuple[str, ...]] = {
    1: ("bfws-ff-pref",),
    2: ("bfws-ff-pref", "lazy-ff-pref"),
    4: ("bfws-ff-pref", "lazy-ff-pref", "lazy-ff-pref-tb", "gbfs-goalcount-rnd"),
    5: (
        "bfws-ff-pref",
        "lazy-ff-pref",
        "lazy-ff-pref-tb",
        "gbfs-goalcount-rnd",
        "gbfs-ff-pref-tb",
    ),
    6: (
        "bfws-ff-pref",
        "lazy-ff-pref",
        "lazy-ff-pref-tb",
        "gbfs-goalcount-rnd",
        "gbfs-ff-pref-tb",
        "gbfs-ff-pref-hybrid",
    ),
    7: (
        "bfws-ff-pref",
        "lazy-ff-pref",
        "lazy-ff-pref-tb",
        "gbfs-goalcount-rnd",
        "gbfs-ff-pref-tb",
        "gbfs-ff-pref-hybrid",
        "wastar-add-3",
    ),
}

#: Núcleos que usa el planificador si no se indica otra cosa (máxima robustez).
DEFAULT_CORES = 6

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
