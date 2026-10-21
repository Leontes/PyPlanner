"""Proyección de las invariantes en las tareas con procesos y eventos.

Una condición ``over all`` hay que **mantenerla** mientras dura la acción, y la relajación de
las heurísticas no lo ve: ignora lo que empeora las condiciones (el consumo de combustible de
un generador, por ejemplo), así que para ella basta con esperar hasta el final. La búsqueda
voraz avanza entonces esperando hasta que la invariante se rompe, y no vuelve atrás lo
bastante para repostar a tiempo.

:class:`ProjectionHeuristic` envuelve una heurística y le suma, en cada estado, el tiempo que
una espera sin hacer nada no cubriría (:meth:`DiscreteTask.shortfall
<pyplanner.grounding.discrete.DiscreteTask.shortfall>`): se simula el paso del tiempo hasta el
final obligatorio de las acciones durativas en curso y, si la invariante de una se rompe
antes, cuenta lo que le faltaba. Las acciones que retrasan la ruptura (empezar a repostar)
bajan el valor, y las que la adelantan lo suben. Las acciones útiles son las de la base.

**Anticipación.** Si solo contaran las acciones en curso, empezar una con invariante (el
generador) subiría el valor de golpe, y la búsqueda voraz se quedaría explorando los estados
de antes (repostar y dejar de repostar da estados nuevos sin fin). Por eso también se simula
empezar cada acción durativa con duración mínima e invariante que la base da como útil (está
en su plan relajado): el valor es el mayor de los faltantes.

**Acciones útiles.** Las de la base no arreglan un faltante (para la relajación basta con
esperar), y la búsqueda perezosa, que pone a los sucesores el valor del padre, bajaría
esperando hasta que ya es tarde para repostar. Por eso, en los estados que se expanden con
un faltante, las útiles son las que lo reducen (si alguna lo hace). Y si lo es la espera de
δ, también la espera hasta el siguiente cambio (``_wait-until``), que la relajación no
tiene.
"""

from collections.abc import Sequence

from pyplanner.grounding import GroundAction, State, mask_of
from pyplanner.grounding.discrete import START_SNAP, WAIT_SNAP, DiscreteTask
from pyplanner.heuristics.base import Evaluation, Heuristic


class ProjectionHeuristic(Heuristic):
    """La heurística ``base`` más lo que no cubre una espera sin hacer nada (ver la
    documentación del módulo)."""

    def __init__(self, task: DiscreteTask, base: Heuristic) -> None:
        super().__init__(task)
        self.base = base
        self.name = base.name
        self.device = base.device
        self.provides_preferred = base.provides_preferred
        self.discrete = task
        self.projects = needs_projection(task)
        # Inicios de las acciones durativas que pueden quedarse cortas.
        self.starts = frozenset(
            action.index
            for action, kind, durative in zip(
                task.actions, task.kinds, task.durative_of, strict=True
            )
            if kind == START_SNAP and _guarded(task, durative)
        )
        self.candidates = _candidates(task)

    def evaluate_batch(self, states: Sequence[State]) -> list[Evaluation]:
        results = []
        for state, evaluation in zip(states, self.base.evaluate_batch(states), strict=True):
            if evaluation.is_dead_end:
                results.append(evaluation)
                continue
            shortfall = self.shortfall(state, evaluation.preferred)
            if shortfall:
                evaluation = Evaluation(evaluation.value + shortfall, evaluation.preferred)
            results.append(evaluation)
        return results

    def shortfall(self, state: State, preferred: Sequence[int]) -> float:
        """El mayor faltante: el del estado y el de empezar cada acción útil de
        :attr:`starts`."""
        task = self.discrete
        if not self.projects:
            return 0.0
        shortfall = task.shortfall(state)
        for index in preferred:
            if index in self.starts:
                successor = task.apply(state, task.actions[index])
                if successor is not None:
                    shortfall = max(shortfall, task.shortfall(successor))
        return shortfall

    def helpful(self, state: State, preferred: frozenset[int]) -> frozenset[int]:
        """Con un faltante, las acciones que lo reducen; si no, las de la base. Con la espera
        de δ, también la espera hasta el siguiente cambio."""
        task = self.discrete
        shortfall = task.shortfall(state) if self.projects else 0.0
        if shortfall:
            preferred = self._reducing(state, shortfall) or preferred
        if task.wait.index in preferred:
            preferred = preferred | {task.jump.index}
        return preferred

    def _reducing(self, state: State, shortfall: float) -> frozenset[int]:
        """Las acciones candidatas aplicables que reducen el faltante."""
        task = self.discrete
        reducing = []
        for action in self.candidates:
            if action.applicable(state):
                successor = task.apply(state, action)
                if successor is not None and task.shortfall(successor) < shortfall - task.epsilon:
                    reducing.append(action.index)
        return frozenset(reducing)


def _candidates(task: DiscreteTask) -> tuple[GroundAction, ...]:
    """Las acciones que pueden cambiar cómo evoluciona el estado: las que cambian valores
    o hechos de las condiciones de los procesos y los eventos (no las esperas)."""
    clauses = [c for p in task.processes for c in p.conditions]
    relevant = mask_of([f for c in clauses for f in (*c.pos_facts, *c.neg_facts)])
    relevant |= mask_of([f for e in task.events for f in (*e.pre_facts, *e.neg_facts)])
    return tuple(
        action
        for action, kind in zip(task.actions, task.kinds, strict=True)
        if kind != WAIT_SNAP
        and (action.numeric or action.conditional or (action.add | action.delete) & relevant)
    )


def _guarded(task: DiscreteTask, durative: int) -> bool:
    """``True`` si la acción durativa tiene invariante y duración mínima."""
    running = task.durative[durative]
    return running.invariant is not None and bool(running.minimum_bounds or running.minimums)


def needs_projection(task: DiscreteTask) -> bool:
    """``True`` si alguna acción durativa tiene invariantes que vigilar."""
    return any(r.invariant is not None for r in task.durative)
