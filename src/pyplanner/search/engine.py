"""Búsqueda heurística hacia delante: GBFS, A* y WA*, voraz o perezosa, por lotes.

Hay dos variantes, que comparten la tabla de nodos, las colas y el control de límites:

* **Evaluación inmediata** (:class:`EagerSearch`): cada sucesor se evalúa al generarlo y
  entra en la cola con su propio valor heurístico. Admite A* y reapertura de nodos.
* **Evaluación perezosa** (:class:`LazySearch`): los sucesores entran en la cola con el
  valor del padre y solo se evalúan al sacarlos. Se evalúan muchos menos estados, a cambio
  de una guía algo peor.

**Lotes.** Con ``batch_size = k`` se sacan hasta *k* entradas de la cola, se expanden y
todos los estados nuevos se evalúan con una sola llamada a
:meth:`~pyplanner.heuristics.Heuristic.evaluate_batch`. En la búsqueda inmediata, un nodo
objetivo solo se acepta si es el primero del lote. Si no, vuelve a la cola, de modo que A*
con una heurística admisible sigue devolviendo planes óptimos.

**Acciones útiles.** Con ``preferred_operators`` hay dos colas: la normal, con todos los
sucesores, y la preferente, solo con los generados por acciones útiles. Se alterna entre
ellas. Como todos los sucesores están en la cola normal, la búsqueda sigue siendo completa.

**Exploración por tipos.** Con ``exploration="type-based"`` se alterna además con una
cola que saca un nodo al azar de un tipo (h, g) al azar (ver :class:`_TypeBuckets`). Ayuda
en dominios con mesetas o muchos callejones sin salida, donde la búsqueda voraz se atasca.

**Callejones sin salida.** Los estados con valor heurístico infinito se descartan. Es
seguro con las heurísticas disponibles, porque solo dan infinito si el objetivo es
inalcanzable incluso en la relajación.
"""

import heapq
import itertools
import random
import time
from collections.abc import Iterator

from pyplanner.grounding import GroundTask, State
from pyplanner.heuristics import Evaluation, Heuristic, create_heuristic
from pyplanner.search.config import SearchConfig
from pyplanner.search.result import SearchResult, SearchStatistics, SearchStatus, StopSignal
from pyplanner.search.successors import create_successor_generator

#: Cada cuántas iteraciones se consulta la señal de parada (consultarla tiene un coste).
STOP_CHECK_INTERVAL = 32

_NO_PARENT = -1

#: Entrada de una cola: (prioridad, desempate, datos del nodo).
_Entry = tuple[tuple[float, ...], float, tuple[int, ...]]


def search(
    task: GroundTask,
    config: SearchConfig,
    heuristic: Heuristic | None = None,
    stop: StopSignal | None = None,
) -> SearchResult:
    """Busca un plan para ``task`` con la configuración dada.

    Args:
        task: tarea instanciada.
        config: configuración de la búsqueda.
        heuristic: heurística ya creada; si es ``None`` se crea a partir de ``config``.
        stop: señal de parada opcional; si se activa, la búsqueda termina con
            :attr:`SearchStatus.CANCELLED`.
    """
    if heuristic is None:
        heuristic = create_heuristic(config.heuristic, task, config.device)
    engine_class = LazySearch if config.lazy else EagerSearch
    return engine_class(task, config, heuristic, stop).run()


class _TypeBuckets:
    """Lista abierta para la exploración por tipos (Xie, Müller, Holte e Imai, 2014).

    Agrupa las entradas por su *tipo* (el par h, g). Al sacar, elige un tipo al azar con la
    misma probabilidad para todos y, dentro de él, una entrada al azar. Así la búsqueda
    visita de vez en cuando zonas que la heurística considera peores, lo que ayuda a salir
    de mesetas y regiones con muchos callejones sin salida. Todas las operaciones son O(1).
    """

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self._buckets: dict[tuple[float, ...], list[tuple[int, ...]]] = {}
        self._types: list[tuple[float, ...]] = []
        self._position: dict[tuple[float, ...], int] = {}

    def push(self, type_key: tuple[float, ...], item: tuple[int, ...]) -> None:
        bucket = self._buckets.get(type_key)
        if bucket is None:
            bucket = self._buckets[type_key] = []
            self._position[type_key] = len(self._types)
            self._types.append(type_key)
        bucket.append(item)

    def pop(self) -> tuple[int, ...]:
        type_key = self._types[self._rng.randrange(len(self._types))]
        bucket = self._buckets[type_key]
        i = self._rng.randrange(len(bucket))
        bucket[i], bucket[-1] = bucket[-1], bucket[i]
        item = bucket.pop()
        if not bucket:  # se quita el tipo intercambiándolo con el último
            del self._buckets[type_key]
            index = self._position.pop(type_key)
            last = self._types.pop()
            if last != type_key:
                self._types[index] = last
                self._position[last] = index
        return item

    def __bool__(self) -> bool:
        return bool(self._types)


class _OpenLists:
    """Colas que se usan por turnos: la normal, la preferente (acciones útiles) y la de
    exploración por tipos, según la configuración."""

    def __init__(self, config: SearchConfig) -> None:
        self.queues: list[list[_Entry]] = [[]]
        if config.preferred_operators:
            self.queues.append([])
        self.types = _TypeBuckets(config.seed) if config.exploration == "type-based" else None
        self._slots = len(self.queues) + (1 if self.types is not None else 0)
        self._turn = 0
        if config.tie_breaking == "random":
            rng = random.Random(config.seed)
            self._tie: Iterator[float] = iter(rng.random, None)
        elif config.tie_breaking == "lifo":
            self._tie = (-float(i) for i in itertools.count())
        else:
            self._tie = (float(i) for i in itertools.count())

    def push(
        self,
        key: tuple[float, ...],
        item: tuple[int, ...],
        preferred: bool,
        type_key: tuple[float, ...],
    ) -> None:
        entry: _Entry = (key, next(self._tie), item)
        heapq.heappush(self.queues[0], entry)
        if preferred and len(self.queues) > 1:
            heapq.heappush(self.queues[1], entry)
        if self.types is not None:
            self.types.push(type_key, item)

    def pop(self) -> tuple[int, ...] | None:
        """Saca una entrada de la cola a la que le toca (o de la siguiente si está vacía)."""
        for offset in range(self._slots):
            slot = (self._turn + offset) % self._slots
            if slot < len(self.queues):
                queue = self.queues[slot]
                if queue:
                    self._turn = (slot + 1) % self._slots
                    return heapq.heappop(queue)[2]
            elif self.types:
                self._turn = (slot + 1) % self._slots
                return self.types.pop()
        return None

    def __bool__(self) -> bool:
        return any(self.queues) or bool(self.types)


class _BestFirstSearch:
    """Parte común: tabla de nodos, prioridades, límites y reconstrucción del plan."""

    def __init__(
        self,
        task: GroundTask,
        config: SearchConfig,
        heuristic: Heuristic,
        stop: StopSignal | None,
    ) -> None:
        self.task = task
        self.config = config
        self.heuristic = heuristic
        self.successors = create_successor_generator(task, config.device)
        self.stop = stop
        self.stats = SearchStatistics()
        self.open = _OpenLists(config)
        # Tabla de nodos (listas paralelas indexadas por el identificador del nodo).
        self.index: dict[State, int] = {}
        self.states: list[State] = []
        self.g: list[float] = []
        self.h: list[float | None] = []
        self.parent: list[int] = []
        self.action: list[int] = []
        self.preferred: list[frozenset[int]] = []
        self.closed: list[bool] = []
        self._start = 0.0
        self._iterations = 0

    # --- Nodos ----------------------------------------------------------------------

    def new_node(self, state: State, g: float, parent: int, action: int) -> int:
        node = len(self.states)
        self.index[state] = node
        self.states.append(state)
        self.g.append(g)
        self.h.append(None)
        self.parent.append(parent)
        self.action.append(action)
        self.preferred.append(frozenset())
        self.closed.append(False)
        return node

    def evaluate(self, nodes: list[int]) -> list[Evaluation]:
        """Evalúa los nodos en un solo lote y guarda h y las acciones útiles."""
        if not nodes:
            return []
        evaluations = self.heuristic.evaluate_batch([self.states[n] for n in nodes])
        self.stats.evaluated += len(nodes)
        self.stats.evaluation_batches += 1
        for node, evaluation in zip(nodes, evaluations, strict=True):
            self.h[node] = evaluation.value
            self.preferred[node] = frozenset(evaluation.preferred)
        return evaluations

    def priority(self, g: float, h: float) -> tuple[float, ...]:
        algorithm = self.config.algorithm
        if algorithm == "gbfs":
            return (h,)
        if algorithm == "astar":
            return (g + h, h)
        return (g + self.config.weight * h, h)

    def extract_plan(self, node: int) -> SearchResult:
        plan: list[int] = []
        while self.parent[node] != _NO_PARENT:
            plan.append(self.action[node])
            node = self.parent[node]
        plan.reverse()
        cost = sum(self.task.actions[a].cost for a in plan)
        return self.finish(SearchStatus.SOLVED, tuple(plan), cost)

    # --- Control --------------------------------------------------------------------

    def interrupted(self) -> SearchStatus | None:
        """Devuelve el motivo si hay que parar (límite o señal externa)."""
        self._iterations += 1
        config = self.config
        if config.max_expansions is not None and self.stats.expanded >= config.max_expansions:
            return SearchStatus.LIMIT_REACHED
        if config.max_time is not None and time.perf_counter() - self._start >= config.max_time:
            return SearchStatus.LIMIT_REACHED
        if (
            self.stop is not None
            and self._iterations % STOP_CHECK_INTERVAL == 1
            and self.stop.is_set()
        ):
            return SearchStatus.CANCELLED
        return None

    def finish(
        self, status: SearchStatus, plan: tuple[int, ...] | None = None, cost: float | None = None
    ) -> SearchResult:
        self.stats.time = time.perf_counter() - self._start
        return SearchResult(status, plan, cost, self.stats, self.config.name)

    def run(self) -> SearchResult:
        self._start = time.perf_counter()
        return self._run()

    def _run(self) -> SearchResult:
        raise NotImplementedError


class EagerSearch(_BestFirstSearch):
    """Búsqueda de primero el mejor con evaluación inmediata de los sucesores."""

    def _run(self) -> SearchResult:
        root = self.new_node(self.task.init, 0, _NO_PARENT, _NO_PARENT)
        [evaluation] = self.evaluate([root])
        if evaluation.is_dead_end:
            return self.finish(SearchStatus.UNSOLVABLE)
        self.push(root, preferred=False)
        reopen = self.config.algorithm != "gbfs"

        while True:
            reason = self.interrupted()
            if reason is not None:
                return self.finish(reason)
            batch = self.pop_batch()
            if batch is None:
                return self.finish(SearchStatus.UNSOLVABLE)
            if isinstance(batch, int):
                return self.extract_plan(batch)

            pending: list[int] = []
            pending_preferred: dict[int, bool] = {}
            for node in batch:
                self.stats.expanded += 1
                state, g = self.states[node], self.g[node]
                useful = self.preferred[node]
                for action in self.successors(state):
                    self.stats.generated += 1
                    successor = self.task.apply(state, action)
                    new_g = g + action.cost
                    is_preferred = action.index in useful
                    known = self.index.get(successor)
                    if known is None:
                        child = self.new_node(successor, new_g, node, action.index)
                        pending.append(child)
                        pending_preferred[child] = is_preferred
                    elif new_g < self.g[known]:
                        self.g[known] = new_g
                        self.parent[known] = node
                        self.action[known] = action.index
                        if self.h[known] is None:  # pendiente de evaluar en este lote
                            pending_preferred[known] |= is_preferred
                        elif reopen and self.h[known] != float("inf"):
                            if self.closed[known]:
                                self.closed[known] = False
                                self.stats.reopened += 1
                            self.push(known, is_preferred)
                    elif self.h[known] is None and is_preferred:
                        pending_preferred[known] = True

            for child, evaluation in zip(pending, self.evaluate(pending), strict=True):
                if evaluation.is_dead_end:
                    self.stats.dead_ends += 1
                    self.closed[child] = True
                else:
                    self.push(child, pending_preferred[child])

    def push(self, node: int, preferred: bool) -> None:
        h = self.h[node]
        assert h is not None
        g = self.g[node]
        self.open.push(self.priority(g, h), (node,), preferred, (h, g))

    def pop_batch(self) -> list[int] | int | None:
        """Saca hasta ``batch_size`` nodos por expandir.

        Devuelve un nodo objetivo (``int``) si es el primero que sale, la lista de nodos a
        expandir, o ``None`` si la cola se ha agotado.
        """
        batch: list[int] = []
        while len(batch) < self.config.batch_size:
            item = self.open.pop()
            if item is None:
                break
            (node,) = item
            if self.closed[node]:
                continue  # entrada obsoleta: el nodo ya se expandió
            if self.task.is_goal(self.states[node]):
                if not batch:
                    return node
                self.push(node, preferred=False)  # se acepta en la siguiente iteración
                break
            self.closed[node] = True
            batch.append(node)
        return batch or None


class LazySearch(_BestFirstSearch):
    """Búsqueda de primero el mejor con evaluación perezosa (GBFS o WA*).

    Las entradas de la cola son pares (padre, acción) con la prioridad del padre. El
    sucesor se genera y se evalúa al sacar la entrada, y no se vuelve a abrir un estado ya
    visitado.
    """

    def _run(self) -> SearchResult:
        root = self.new_node(self.task.init, 0, _NO_PARENT, _NO_PARENT)
        [evaluation] = self.evaluate([root])
        if evaluation.is_dead_end:
            return self.finish(SearchStatus.UNSOLVABLE)
        if self.task.is_goal(self.task.init):
            return self.extract_plan(root)
        self.expand(root)

        while True:
            reason = self.interrupted()
            if reason is not None:
                return self.finish(reason)
            if not self.open:
                return self.finish(SearchStatus.UNSOLVABLE)

            # Saca hasta batch_size entradas cuyos sucesores sean estados nuevos.
            children: list[int] = []
            while len(children) < self.config.batch_size:
                item = self.open.pop()
                if item is None:
                    break
                parent, action_index = item
                action = self.task.actions[action_index]
                successor = self.task.apply(self.states[parent], action)
                if successor in self.index:
                    continue
                children.append(
                    self.new_node(successor, self.g[parent] + action.cost, parent, action_index)
                )

            for child, evaluation in zip(children, self.evaluate(children), strict=True):
                if evaluation.is_dead_end:
                    self.stats.dead_ends += 1
                    continue
                if self.task.is_goal(self.states[child]):
                    return self.extract_plan(child)
                self.expand(child)

    def expand(self, node: int) -> None:
        """Mete en la cola las entradas (nodo, acción) de todas sus acciones aplicables."""
        self.stats.expanded += 1
        self.closed[node] = True
        h = self.h[node]
        assert h is not None
        useful = self.preferred[node]
        for action in self.successors(self.states[node]):
            self.stats.generated += 1
            key = self.priority(self.g[node] + action.cost, h)
            g = self.g[node] + action.cost
            self.open.push(key, (node, action.index), action.index in useful, (h, g))
