"""Evaluación híbrida: cada lote en Numba (CPU) o en PyTorch (GPU), según su tamaño.

Las implementaciones de Numba y de PyTorch dan exactamente los mismos valores y acciones
útiles (ver :mod:`pyplanner.heuristics.relaxation`), así que elegir una u otra para cada
lote no cambia la búsqueda: solo su velocidad.

**Calibración.** El coste por estado de Numba no depende del tamaño de lote, pero crece
con el de la tarea; el de la GPU tiene un coste fijo alto por llamada que se amortiza con
lotes grandes, y crece menos con la tarea. En la máquina de desarrollo (una GTX 970) la
GPU no compensa nunca en tareas pequeñas como slitherlink, y en las grandes (driverlog o
zenotravel temporales con miles de acciones) es del orden del doble de rápida a partir de
lotes de unos 64 estados. Por eso el umbral no es fijo: al crear la heurística se mide el
coste de las dos en la propia tarea, con estados de un paseo aleatorio, y se fija el
tamaño de lote a partir del cual la GPU es más rápida (con un margen). Si no lo es nunca,
todo va a Numba.

Solo tiene sentido con lotes grandes: con el lote adaptativo de la búsqueda
(``adaptive_batch``), que los hace crecer en las mesetas.
"""

import math
import random
import time
from collections.abc import Sequence

from pyplanner.grounding import GroundTask, State
from pyplanner.heuristics.base import Evaluation, Heuristic

#: Tamaños de lote que se prueban al calibrar.
CALIBRATION_SIZES = (32, 64, 128, 256, 512)
#: La GPU tiene que ser al menos este factor más rápida por estado para usarla.
MARGIN = 0.8
#: Estados de muestra para calibrar.
SAMPLE_STATES = 256


class HybridHeuristic(Heuristic):
    """Heurística que reparte los lotes entre Numba y PyTorch en GPU.

    Attributes:
        threshold: tamaño de lote a partir del cual se usa la GPU (``inf``: nunca).
        gpu_batches: lotes evaluados en la GPU.
    """

    device = "hybrid"

    def __init__(self, task: GroundTask, fast: Heuristic, wide: Heuristic | None) -> None:
        super().__init__(task)
        self.fast = fast
        self.wide = wide
        self.name = fast.name
        self.provides_preferred = fast.provides_preferred
        self.gpu_batches = 0
        self.threshold = math.inf if wide is None else self._calibrate()

    def _calibrate(self) -> float:
        assert self.wide is not None
        states = _sample_states(self.task, SAMPLE_STATES)
        self.fast.evaluate_batch(states[:8])  # carga los núcleos compilados
        start = time.perf_counter()
        self.fast.evaluate_batch(states[:64])
        cpu = (time.perf_counter() - start) / min(64, len(states))
        self.wide.evaluate_batch(states[:8])  # inicializa CUDA y la tarea en la GPU
        for size in CALIBRATION_SIZES:
            batch = (states * (size // len(states) + 1))[:size]
            self.wide.evaluate_batch(batch)  # primera llamada con este tamaño
            start = time.perf_counter()
            self.wide.evaluate_batch(batch)
            gpu = (time.perf_counter() - start) / size
            if gpu < MARGIN * cpu:
                return size
        return math.inf

    def evaluate_batch(self, states: Sequence[State]) -> list[Evaluation]:
        if len(states) >= self.threshold:
            assert self.wide is not None
            self.gpu_batches += 1
            return self.wide.evaluate_batch(states)
        return self.fast.evaluate_batch(states)


def _sample_states(task: GroundTask, count: int, seed: int = 0) -> list[State]:
    """Estados de un paseo aleatorio desde el inicial (para calibrar)."""
    rng = random.Random(seed)
    states = [task.init]
    state = task.init
    attempts = 0
    while len(states) < count and attempts < 20 * count:
        attempts += 1
        actions = list(task.applicable(state))
        successor = task.apply(state, rng.choice(actions)) if actions else None
        if successor is None:
            state = task.init
            continue
        state = successor
        states.append(state)
    return states
