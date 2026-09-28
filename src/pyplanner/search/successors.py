"""Generación de sucesores: qué acciones son aplicables en un estado.

Dos implementaciones con el **mismo orden canónico** (ver
:meth:`pyplanner.grounding.GroundTask.applicable`), así que la búsqueda expande los mismos
nodos con cualquiera de las dos:

* ``python``: :meth:`GroundTask.applicable`, la referencia;
* ``numba``: un núcleo compilado que recorre los hechos del estado y comprueba las
  precondiciones de las acciones candidatas con operaciones sobre palabras de 64 bits.

La búsqueda usa la compilada si Numba está instalado y el dispositivo de la heurística no
es ``"python"`` (con ``"python"`` todo el planificador es la implementación de referencia).
"""

from collections.abc import Callable, Sequence

import numpy as np

from pyplanner.grounding import GroundAction, GroundTask, State
from pyplanner.heuristics import numba_available

#: Función que devuelve las acciones aplicables en un estado, en el orden canónico.
SuccessorGenerator = Callable[[State], Sequence[GroundAction]]


def create_successor_generator(task: GroundTask, device: str = "auto") -> SuccessorGenerator:
    """Generador de sucesores para ``task``: compilado si se puede, de referencia si no."""
    if device != "python" and numba_available():
        return CompiledSuccessors(task)
    return lambda state: list(task.applicable(state))


class CompiledSuccessors:
    """Generador de sucesores compilado con Numba."""

    def __init__(self, task: GroundTask) -> None:
        from pyplanner.search import compiled_successors as kernels

        self._kernel = kernels.applicable
        self.actions = task.actions
        self.num_words = max(1, (len(task.facts) + 63) // 64)
        num_bytes = 8 * self.num_words

        def words(mask: int) -> np.ndarray:
            return np.frombuffer(mask.to_bytes(num_bytes, "little"), dtype=np.uint64)

        empty = np.zeros((0, self.num_words), dtype=np.uint64)
        self.pre = np.stack([words(a.pre) for a in task.actions]) if task.actions else empty
        self.neg = np.stack([words(a.neg) for a in task.actions]) if task.actions else empty
        self.always = np.array([a.index for a in task.actions if not a.pre_facts], dtype=np.int64)
        by_fact: list[list[int]] = [[] for _ in range(len(task.facts))]
        for action in task.actions:
            if action.pre_facts:
                by_fact[action.pre_facts[0]].append(action.index)
        self.trigger_ptr = np.zeros(len(by_fact) + 1, dtype=np.int64)
        self.trigger_ptr[1:] = np.cumsum([len(items) for items in by_fact])
        self.trigger_idx = np.array([a for items in by_fact for a in items], dtype=np.int64)
        self._out = np.empty(len(task.actions), dtype=np.int64)
        self._num_bytes = num_bytes

    def __call__(self, state: State) -> list[GroundAction]:
        words = np.frombuffer(state.to_bytes(self._num_bytes, "little"), dtype=np.uint64)
        count = self._kernel(
            words,
            self.pre,
            self.neg,
            self.always,
            self.trigger_ptr,
            self.trigger_idx,
            self._out,
        )
        actions = self.actions
        return [actions[i] for i in self._out[:count].tolist()]
