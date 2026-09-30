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
    """Generador de sucesores para ``task``: compilado si se puede, de referencia si no.

    El compilado solo admite precondiciones numéricas lineales (las habituales); si alguna
    acción tiene una no lineal, se usa la referencia.
    """
    linear_only = all(c.linear is not None for a in task.actions for c in a.numeric_pre)
    if device != "python" and numba_available() and linear_only:
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
        # Precondiciones numéricas lineales en formato CSR: por acción, sus condiciones; por
        # condición, sus términos (variable, peso), su constante y su comparador.
        cond_ptr = [0]
        term_ptr = [0]
        term_var: list[int] = []
        term_weight: list[float] = []
        constant: list[float] = []
        comparator: list[int] = []
        codes = {">=": kernels.GE, ">": kernels.GT, "=": kernels.EQ}
        for action in task.actions:
            for condition in action.numeric_pre:
                form = condition.linear
                assert form is not None
                for variable, weight in form.terms:
                    term_var.append(variable)
                    term_weight.append(weight)
                term_ptr.append(len(term_var))
                constant.append(form.constant)
                comparator.append(codes[condition.comparator])
            cond_ptr.append(len(constant))
        self.cond_ptr = np.array(cond_ptr, dtype=np.int64)
        self.term_ptr = np.array(term_ptr, dtype=np.int64)
        self.term_var = np.array(term_var, dtype=np.int64)
        self.term_weight = np.array(term_weight, dtype=np.float64)
        self.constant = np.array(constant, dtype=np.float64)
        self.comparator = np.array(comparator, dtype=np.int64)
        self._out = np.empty(len(task.actions), dtype=np.int64)
        self._num_bytes = num_bytes
        self._no_values = np.zeros(0)
        # Sin precondiciones numéricas se usa el núcleo con menos argumentos.
        self._propositional = not constant
        self._facts_kernel = kernels.applicable_facts

    def __call__(self, state: State) -> list[GroundAction]:
        words = np.frombuffer(state.facts.to_bytes(self._num_bytes, "little"), dtype=np.uint64)
        if self._propositional:
            count = self._facts_kernel(
                words, self.pre, self.neg, self.always, self.trigger_ptr, self.trigger_idx,
                self._out,
            )  # fmt: skip
            actions = self.actions
            return [actions[i] for i in self._out[:count].tolist()]
        values = (
            np.array([np.nan if v is None else v for v in state.values], dtype=np.float64)
            if state.values
            else self._no_values
        )
        count = self._kernel(
            words,
            values,
            self.pre,
            self.neg,
            self.always,
            self.trigger_ptr,
            self.trigger_idx,
            self.cond_ptr,
            self.term_ptr,
            self.term_var,
            self.term_weight,
            self.constant,
            self.comparator,
            self._out,
        )
        actions = self.actions
        return [actions[i] for i in self._out[:count].tolist()]
