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

**Lote adaptativo** (``adaptive_batch``): el tamaño de lote vuelve al mínimo cada vez que
mejora la mejor h vista, y se duplica tras ``plateau_patience`` lotes seguidos sin
mejorarla. Mientras la heurística guía, la búsqueda se comporta como la clásica; en las
mesetas, donde la guía aporta poco, expande muchos nodos de golpe y los evalúa juntos
(lo que aprovecha la GPU con el dispositivo ``"hybrid"``).

**Novedad** (``novelty``, solo con evaluación inmediata; ver
:mod:`pyplanner.search.novelty`):

* **BFWS** (GBFS con novedad): la prioridad es (novedad, h), con la novedad medida entre
  los estados del mismo valor h. Entre estados igual de buenos, se prefieren los que hacen
  cierto algo nuevo. Es completa: no se poda nada.
* **IW(k)** (``algorithm="iw"``): búsqueda en anchura que descarta los estados de novedad
  mayor que k. Es incompleta: si agota la cola habiendo podado algo, termina con
  :attr:`SearchStatus.FAILED`, no con ``UNSOLVABLE``.

**Callejones sin salida.** Los estados con valor heurístico infinito se descartan. Es
seguro con las heurísticas disponibles, porque solo dan infinito si el objetivo es
inalcanzable incluso en la relajación.

**Sin plan: ``UNSOLVABLE`` o ``FAILED``.** ``UNSOLVABLE`` es una demostración: el estado
inicial es un callejón, o la búsqueda ha agotado la cola sin perder nada. Si ha perdido
algo, termina con ``FAILED`` (no ha encontrado plan, pero puede haberlo): IW poda estados,
y en una tarea temporal dos estados iguales con distinta línea temporal son el mismo nodo
(memoización al estilo de POPF), así que al descartar un repetido se pierde un camino que
quizá cumplía unos plazos que el otro no cumple.

**Penalizaciones.** Con preferencias, el paso que incumple una restricción-preferencia
cuesta además su peso (ver :mod:`pyplanner.grounding.preferences`): g lo incluye.

**Cota** (``bound``, para la búsqueda *anytime*: ver :mod:`pyplanner.search.anytime`): solo
se buscan planes de coste menor. Se descartan los nodos cuyo g llega a la cota; con una
heurística admisible (A* con ``blind`` o ``hmax``), también los de g + h.

**Incumbente** (con preferencias y ``on_plan``): cada estado expandido en el que se cumple
el objetivo duro es un plan posible, porque la fase final de las preferencias se puede
recorrer desde él con un coste conocido (los pesos de las que no cumple). Si ese plan
mejora la cota, se apunta, la cota baja hasta él y se informa con ``on_plan``; la búsqueda
sigue. Si termina sin otro plan, devuelve el incumbente. Así una búsqueda voraz que avanza
cumpliendo preferencias va dejando planes cada vez mejores, aunque nunca llegue a la fase
final por sí misma. Si la búsqueda
agota la cola, no hay plan mejor que la cota que ella pueda encontrar; con A* o WA*
(inmediata, con reapertura), no hay ninguno.
"""

import heapq
import random
import time
from array import array
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace

from pyplanner.grounding import GroundTask, State
from pyplanner.heuristics import Evaluation, Heuristic, create_heuristic
from pyplanner.heuristics.landmarks import LandmarkGraph
from pyplanner.search.config import SearchConfig
from pyplanner.search.memory import resident_memory_mb
from pyplanner.search.novelty import NoveltyTable, negated_facts
from pyplanner.search.result import SearchResult, SearchStatistics, SearchStatus, StopSignal
from pyplanner.search.successors import create_successor_generator

#: Cada cuántas iteraciones se consulta la señal de parada (consultarla tiene un coste).
STOP_CHECK_INTERVAL = 32

_NO_PARENT = -1
#: Acciones útiles de un nodo sin ellas (compartido, para no crear un conjunto por nodo).
_NO_PREFERRED: frozenset[int] = frozenset()

#: Entrada de una cola: (prioridad, desempate, datos del nodo).
_Entry = tuple[tuple[float, ...], float, tuple[int, ...]]
#: Entrada de varias acciones de un nodo con la misma prioridad (búsqueda perezosa):
#: (prioridad, desempate de la siguiente, nodo, acciones, posición de la siguiente).
_Group = tuple[tuple[float, ...], float, int, tuple[int, ...], int]

#: Margen de la cota: un plan tiene que mejorarla al menos en esto (los costes son sumas de
#: números reales, con su error de redondeo).
BOUND_TOLERANCE = 1e-9


def search(
    task: GroundTask,
    config: SearchConfig,
    heuristic: Heuristic | None = None,
    stop: StopSignal | None = None,
    bound: float | None = None,
    on_plan: Callable[[SearchResult], None] | None = None,
) -> SearchResult:
    """Busca un plan para ``task`` con la configuración dada.

    Args:
        task: tarea instanciada.
        config: configuración de la búsqueda.
        heuristic: heurística ya creada; si es ``None`` se crea a partir de ``config``.
        stop: señal de parada opcional; si se activa, la búsqueda termina con
            :attr:`SearchStatus.CANCELLED`.
        bound: si se da, solo se buscan planes de coste menor (ver la documentación del
            módulo); si no los hay, la búsqueda termina con ``UNSOLVABLE``.
        on_plan: con preferencias, activa el **incumbente** (ver la documentación del
            módulo): se llama con cada plan mejor que encuentra.

    Si se agota la memoria (el límite duro de :mod:`pyplanner.search.memory`, o la de la
    máquina), termina con ``LIMIT_REACHED``.
    """
    try:
        if heuristic is None:
            heuristic = create_heuristic(config.heuristic, task, config.device)
        engine_class = LazySearch if config.lazy else EagerSearch
        return engine_class(task, config, heuristic, stop, bound, on_plan).run()
    except MemoryError:
        pass  # el resultado se crea fuera: la traza retiene la memoria de la búsqueda
    return SearchResult(
        SearchStatus.LIMIT_REACHED,
        config_name=config.name,
        message="límite de memoria: se ha agotado la memoria permitida (límite duro)",
    )


class _TypeBuckets:
    """Lista abierta para la exploración por tipos (Xie, Müller, Holte e Imai, 2014).

    Agrupa las entradas por su *tipo* (el par h, g). Al sacar, elige un tipo al azar con la
    misma probabilidad para todos y, dentro de él, una entrada al azar. Así la búsqueda
    visita de vez en cuando zonas que la heurística considera peores, lo que ayuda a salir
    de mesetas y regiones con muchos callejones sin salida. Todas las operaciones son O(1).

    Las entradas son ``(nodo,)`` o ``(nodo, acción)`` y se guardan como un entero de 64 bits
    en un ``array`` (8 bytes en lugar de una tupla): la búsqueda perezosa mete una por
    sucesor.
    """

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self._buckets: dict[tuple[float, ...], array[int]] = {}
        self._types: list[tuple[float, ...]] = []
        self._position: dict[tuple[float, ...], int] = {}

    def push(self, type_key: tuple[float, ...], item: tuple[int, ...]) -> None:
        bucket = self._buckets.get(type_key)
        if bucket is None:
            bucket = self._buckets[type_key] = array("q")
            self._position[type_key] = len(self._types)
            self._types.append(type_key)
        # nodo en los 32 bits altos; acción + 1 en los bajos (0: sin acción)
        bucket.append(item[0] << 32 | (item[1] + 1 if len(item) > 1 else 0))

    def pop(self) -> tuple[int, ...]:
        type_key = self._types[self._rng.randrange(len(self._types))]
        bucket = self._buckets[type_key]
        i = self._rng.randrange(len(bucket))
        bucket[i], bucket[-1] = bucket[-1], bucket[i]
        code = bucket.pop()
        if not bucket:  # se quita el tipo intercambiándolo con el último
            del self._buckets[type_key]
            index = self._position.pop(type_key)
            last = self._types.pop()
            if last != type_key:
                self._types[index] = last
                self._position[last] = index
        action = code & 0xFFFFFFFF
        return (code >> 32, action - 1) if action else (code >> 32,)

    def __bool__(self) -> bool:
        return bool(self._types)


class _OpenLists:
    """Colas que se usan por turnos: la normal, la preferente (acciones útiles) y la de
    exploración por tipos, según la configuración. Con landmarks, la normal y la preferente
    se repiten para h_LM, con la prioridad que le da esa heurística.

    El desempate FIFO o LIFO es un contador (0, 1, 2… o 0, −1, −2…). Con él, la búsqueda
    perezosa mete las acciones de un nodo con las mismas prioridades en **una sola
    entrada** por cola (:meth:`push_successors`): reserva tantos desempates seguidos como
    acciones y, al sacar una, la entrada vuelve con el siguiente. Salen exactamente en el
    mismo orden que si fueran entradas sueltas (el desempate solo decide entre entradas de
    igual prioridad, y las de otros nodos tienen desempates fuera del bloque), pero sin una
    entrada por sucesor: en satellite con TIL, con cientos de sucesores por expansión, las
    entradas sueltas eran casi toda la memoria de la búsqueda.
    """

    def __init__(self, config: SearchConfig) -> None:
        self.queues: list[list[_Entry | _Group]] = [[]]
        self._preferred = config.preferred_operators
        if self._preferred:
            self.queues.append([])
        self._second = len(self.queues) if config.landmarks else None
        if config.landmarks:
            self.queues.extend([] for _ in range(len(self.queues)))
        self.types = _TypeBuckets(config.seed) if config.exploration == "type-based" else None
        self._slots = len(self.queues) + (1 if self.types is not None else 0)
        self._turn = 0
        self._last = 0
        self._random: Iterator[float] | None = None
        self._lifo = config.tie_breaking == "lifo"
        self._count = 0
        if config.tie_breaking == "random":
            rng = random.Random(config.seed)
            self._random = iter(rng.random, None)

    @property
    def groups(self) -> bool:
        """``True`` si se pueden meter varias acciones en una entrada (desempate FIFO o
        LIFO; con el aleatorio, cada entrada tiene el suyo)."""
        return self._random is None

    def _tie(self) -> float:
        if self._random is not None:
            return next(self._random)
        tie = -float(self._count) if self._lifo else float(self._count)
        self._count += 1
        return tie

    def push(
        self,
        key: tuple[float, ...],
        item: tuple[int, ...],
        preferred: bool,
        type_key: tuple[float, ...],
        second_key: tuple[float, ...] | None = None,
    ) -> None:
        tie = self._tie()
        self._push_into(0, (key, tie, item), preferred)
        if second_key is not None and self._second is not None:
            self._push_into(self._second, (second_key, tie, item), preferred)
        if self.types is not None:
            self.types.push(type_key, item)

    def _push_into(self, first: int, entry: _Entry, preferred: bool) -> None:
        """Mete la entrada en la cola normal ``first`` y, si es preferente, en la siguiente."""
        heapq.heappush(self.queues[first], entry)
        if preferred and self._preferred:
            heapq.heappush(self.queues[first + 1], entry)

    def push_successors(
        self,
        key: tuple[float, ...],
        node: int,
        actions: list[int],
        preferred: list[int],
        second_key: tuple[float, ...] | None = None,
    ) -> None:
        """Las entradas (``node``, acción) de ``actions``, todas con prioridad ``key`` (y
        ``second_key`` en las colas de h_LM), como si se metieran una a una en ese orden
        (``preferred``: las que van también a las colas preferentes). Solo con
        :attr:`groups`; la cola por tipos se llena aparte (:meth:`push_type`)."""
        first = self._count
        self._count += len(actions)
        targets = [(0, key)]
        if second_key is not None and self._second is not None:
            targets.append((self._second, second_key))
        for queue, priority in targets:
            self._push_group(self.queues[queue], priority, first, node, actions)
            if preferred and self._preferred:
                self._push_group(self.queues[queue + 1], priority, first, node, preferred)

    def _push_group(
        self,
        queue: list[_Entry | _Group],
        key: tuple[float, ...],
        first: int,
        node: int,
        actions: list[int],
    ) -> None:
        if self._lifo:  # sale primero la última, con el desempate más bajo del bloque
            group: _Group = (key, -float(first + len(actions) - 1), node, tuple(actions[::-1]), 0)
        else:
            group = (key, float(first), node, tuple(actions), 0)
        heapq.heappush(queue, group)

    def push_type(self, type_key: tuple[float, ...], item: tuple[int, ...]) -> None:
        """Mete ``item`` en la cola de exploración por tipos, si la hay."""
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
                    self._last = slot
                    entry = heapq.heappop(queue)
                    if len(entry) == 3:
                        return entry[2]
                    key, tie, node, actions, position = entry
                    if position + 1 < len(actions):
                        heapq.heappush(queue, (key, tie + 1, node, actions, position + 1))
                    return (node, actions[position])
            elif self.types:
                self._turn = (slot + 1) % self._slots
                self._last = slot
                return self.types.pop()
        return None

    def retry(self) -> None:
        """La próxima entrada sale de la misma cola que la última: la que salió no cuenta
        como turno (un nodo descartado al expandir, que nunca debió estar en la cola)."""
        self._turn = self._last

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
        bound: float | None = None,
        on_plan: Callable[[SearchResult], None] | None = None,
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
        self.width: list[int] = []
        # Landmarks (ver SearchConfig.landmarks): los aceptados por el camino de cada nodo
        # (bits) y su valor h_LM.
        self.landmarks = LandmarkGraph(task) if config.landmarks else None
        self.accepted: list[int] = []
        self.h_lm: list[float] = []
        self.novelty = (
            NoveltyTable(
                config.novelty, len(task.facts), negated_facts(task), task.continuous_variables
            )
            if config.novelty
            else None
        )
        self._start = 0.0
        self._iterations = 0
        self._message = ""
        self.batch_size = config.batch_size
        # En las tareas clásicas sin axiomas se aplica la acción directamente (una llamada
        # menos por sucesor); las temporales redefinen GroundTask.apply, y con axiomas hay que
        # recalcular los hechos derivados.
        self._direct_apply = (
            type(task).apply is GroundTask.apply and not task.axioms and not task.monitors
        )
        # En las tareas temporales, dos estados iguales pueden tener distinta línea temporal
        # (memoización al estilo de POPF): ver la actualización de padres en EagerSearch.run.
        self._temporal = type(task).apply is not GroundTask.apply
        self._best_h = float("inf")
        self._stalled = 0
        self._dead_end_check: Callable[[Sequence[State]], list[bool]] | None = None
        #: Nodos descartados al expandir por la comprobación aplazada.
        self._discarded: set[int] = set()
        # Cota: los nodos con g desde ``limit`` se descartan.
        self.limit = float("inf") if bound is None else bound - BOUND_TOLERANCE
        self._admissible_bound = bound is not None and config.is_optimal
        # Penalización de las transiciones (restricciones-preferencia incumplidas).
        preferences = task.preferences
        self.penalty = (
            preferences.penalty if preferences is not None and preferences.violations else None
        )
        # Incumbente: el mejor plan que termina en un estado expandido.
        self.on_plan = on_plan
        self.completes = (
            on_plan is not None and preferences is not None and bool(preferences.end_actions)
        )
        self.incumbent: tuple[float, tuple[int, ...]] | None = None
        self._lower_bound: float | None = None
        #: En una tarea temporal, si se ha descartado algún estado repetido (memoización al
        #: estilo de POPF): entonces agotar la cola no demuestra que no haya plan.
        self.lost_paths = False

    # --- Nodos ----------------------------------------------------------------------

    def new_node(self, state: State, g: float, parent: int, action: int) -> int:
        node = len(self.states)
        self.index[state] = node
        self.states.append(state)
        self.g.append(g)
        self.h.append(None)
        self.parent.append(parent)
        self.action.append(action)
        self.preferred.append(_NO_PREFERRED)
        self.closed.append(False)
        if self.novelty is not None:
            self.width.append(0)
        if self.landmarks is not None:
            inherited = self.accepted[parent] if parent != _NO_PARENT else 0
            self.accepted.append(self.landmarks.accept(inherited, state))
            self.h_lm.append(0.0)
        return node

    def merge_landmarks(self, known: int, parent: int, state: State) -> None:
        """Un estado conocido al que se llega por otro camino: sus landmarks aceptados son
        los que lo están por los dos caminos (como en LAMA)."""
        if self.landmarks is not None:
            self.accepted[known] &= self.landmarks.accept(self.accepted[parent], state)

    def helpful(self, node: int) -> frozenset[int]:
        """Las acciones útiles del nodo que se expande (las afina la heurística). Con
        landmarks, también las que consiguen uno de los siguientes; se calculan aquí, solo
        para los nodos que se expanden."""
        if not self.config.preferred_operators:
            return self.preferred[node]
        useful = self.heuristic.helpful(self.states[node], self.preferred[node])
        if self.landmarks is not None:
            useful = useful | self.landmarks.preferred(self.accepted[node])
        return useful

    def evaluate(self, nodes: list[int]) -> list[Evaluation]:
        """Evalúa los nodos en un solo lote y guarda h y las acciones útiles."""
        if not nodes:
            return []
        states = self.states
        evaluations = self.heuristic.evaluate_batch([states[n] for n in nodes])
        stats = self.stats
        stats.evaluated += len(nodes)
        stats.evaluation_batches += 1
        if len(nodes) > stats.largest_batch:
            stats.largest_batch = len(nodes)
        if self.config.adaptive_batch:
            self.adapt(evaluations)
        h, preferred = self.h, self.preferred
        graph = self.landmarks
        for node, evaluation in zip(nodes, evaluations, strict=True):
            h[node] = evaluation.value
            if evaluation.preferred:
                preferred[node] = frozenset(evaluation.preferred)
            if graph is not None:
                self.h_lm[node] = graph.value(self.accepted[node], states[node])
        return evaluations

    def adapt(self, evaluations: list[Evaluation]) -> None:
        """Ajusta el tamaño de lote: al mínimo si mejora la mejor h; si no, lo duplica tras
        ``plateau_patience`` lotes sin mejora."""
        config = self.config
        best = min((e.value for e in evaluations), default=float("inf"))
        if best < self._best_h:
            self._best_h = best
            self.batch_size = config.batch_size
            self._stalled = 0
            return
        self._stalled += 1
        if self._stalled >= config.plateau_patience:
            self.batch_size = min(2 * self.batch_size, config.max_batch_size)
            self._stalled = 0

    def measure_novelty(self, node: int) -> bool:
        """Calcula la novedad del nodo (si la búsqueda la usa). Devuelve ``False`` si IW lo
        poda."""
        if self.novelty is None:
            return True
        h = self.h[node]
        partition = None if self.config.algorithm == "iw" else h
        width = self.novelty.novelty(self.states[node], partition)
        self.width[node] = width
        if self.config.algorithm == "iw" and width > self.config.novelty:
            self.stats.pruned += 1
            return False
        return True

    def exhausted(self) -> SearchResult:
        """La cola se ha agotado. Si no se ha perdido nada por el camino, no hay plan
        (``UNSOLVABLE``). Si no, la búsqueda solo sabe que no lo ha encontrado
        (``FAILED``): IW ha podado estados, o una tarea temporal ha descartado estados
        repetidos, que con otra línea temporal podrían haber llegado al objetivo.

        Con una cota, una búsqueda inmediata con reapertura (A* o WA*) demuestra que no hay
        plan mejor que ella (``lower_bound``)."""
        status = SearchStatus.UNSOLVABLE
        caveat = self.task.unsolvability_caveat(self.lost_paths)
        if self.stats.pruned:
            status = SearchStatus.FAILED
        elif caveat is not None:
            status = SearchStatus.FAILED
            self._message = caveat
        exact = not self.config.lazy and self.config.algorithm in ("astar", "wastar")
        if status is SearchStatus.UNSOLVABLE and exact and self.limit < float("inf"):
            self._lower_bound = self.limit + BOUND_TOLERANCE
        return self.finish(status)

    def complete(self, node: int) -> None:
        """Incumbente: si en el nodo se puede pasar a la fase final de las preferencias y
        el plan que termina en él mejora la cota, lo apunta, aprieta la cota e informa."""
        preferences = self.task.preferences
        assert preferences is not None and self.on_plan is not None
        g, state = self.g[node], self.states[node]
        if g >= self.limit or not preferences.can_end(self.task, state):
            return
        cost = g + preferences.end_cost(state)
        if cost >= self.limit:
            return
        plan = (*self.path(node), *preferences.completion(self.task, state))
        self.incumbent = (cost, plan)
        self.limit = cost - BOUND_TOLERANCE
        self.stats.time = time.perf_counter() - self._start
        self.on_plan(
            SearchResult(SearchStatus.SOLVED, plan, cost, replace(self.stats), self.config.name)
        )

    def priority(self, g: float, h: float, width: int = 0) -> tuple[float, ...]:
        algorithm = self.config.algorithm
        if algorithm == "iw":
            return (g,)
        if algorithm == "gbfs":
            return (width, h) if self.novelty is not None else (h,)
        if algorithm == "astar":
            return (g + h, h)
        return (g + self.config.weight * h, h)

    def path(self, node: int) -> list[int]:
        """Las acciones desde la raíz hasta el nodo."""
        plan: list[int] = []
        while self.parent[node] != _NO_PARENT:
            plan.append(self.action[node])
            node = self.parent[node]
        plan.reverse()
        return plan

    def extract_plan(self, node: int) -> SearchResult:
        return self.finish(SearchStatus.SOLVED, tuple(self.path(node)), self.g[node])

    # --- Control --------------------------------------------------------------------

    def interrupted(self) -> SearchStatus | None:
        """Devuelve el motivo si hay que parar (límite o señal externa)."""
        self._iterations += 1
        config = self.config
        if config.max_expansions is not None and self.stats.expanded >= config.max_expansions:
            return SearchStatus.LIMIT_REACHED
        if config.max_time is not None and time.perf_counter() - self._start >= config.max_time:
            return SearchStatus.LIMIT_REACHED
        if self._iterations % STOP_CHECK_INTERVAL == 1:
            if self.stop is not None and self.stop.is_set():
                return SearchStatus.CANCELLED
            if config.max_memory is not None:
                used = resident_memory_mb()
                if used is not None and used > config.max_memory:
                    self._message = (
                        f"límite de memoria: {used:.0f} MB usados (máximo {config.max_memory})"
                    )
                    return SearchStatus.LIMIT_REACHED
        return None

    def finish(
        self, status: SearchStatus, plan: tuple[int, ...] | None = None, cost: float | None = None
    ) -> SearchResult:
        self.stats.time = time.perf_counter() - self._start
        self.stats.gpu_batches = getattr(self.heuristic, "gpu_batches", 0)
        message = self._message
        interrupted = False
        if status is not SearchStatus.SOLVED and self.incumbent is not None:
            # Sin plan propio, el resultado es el incumbente.
            cost, plan = self.incumbent
            interrupted = status in (SearchStatus.LIMIT_REACHED, SearchStatus.CANCELLED)
            status = SearchStatus.SOLVED
        return SearchResult(
            status,
            plan,
            cost,
            self.stats,
            self.config.name,
            message,
            self._lower_bound,
            interrupted,
        )

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
        self.measure_novelty(root)
        self.push(root, preferred=False)
        reopen = self.config.algorithm in ("astar", "wastar")
        limit, penalty = self.limit, self.penalty
        # Sin cota, penalizaciones ni incumbente, una sola comprobación por sucesor.
        checked = penalty is not None or self.completes or limit < float("inf")
        # La comprobación de callejones que se puede aplazar (los plazos de los TIL) se hace
        # al sacar cada nodo de la cola: la mayoría de los sucesores nunca se expanden.
        self._dead_end_check = self.heuristic.defer_dead_end_check()

        while True:
            reason = self.interrupted()
            if reason is not None:
                return self.finish(reason)
            batch = self.pop_batch()
            if batch is None:
                return self.exhausted()
            if isinstance(batch, int):
                return self.extract_plan(batch)

            pending: list[int] = []
            pending_preferred: dict[int, bool] = {}
            for node in batch:
                self.stats.expanded += 1
                if self.completes:
                    self.complete(node)
                    limit = self.limit
                state, g = self.states[node], self.g[node]
                useful = self.helpful(node)
                for action in self.successors(state):
                    self.stats.generated += 1
                    successor = (
                        action.apply(state)
                        if self._direct_apply
                        else self.task.apply(state, action)
                    )
                    if successor is None:
                        continue  # efecto numérico indefinido: la acción no se puede aplicar
                    new_g = g + action.cost
                    if checked:
                        if penalty is not None:
                            new_g += penalty(state, successor)
                        if new_g >= limit:
                            continue  # no puede mejorar la cota
                    is_preferred = action.index in useful
                    known = self.index.get(successor)
                    if known is None:
                        child = self.new_node(successor, new_g, node, action.index)
                        pending.append(child)
                        pending_preferred[child] = is_preferred
                        continue
                    self.merge_landmarks(known, node, successor)
                    if self._temporal:
                        # Uno de los dos caminos (y su línea temporal) se pierde.
                        self.lost_paths = True
                    if new_g < self.g[known] and not (self._temporal and self.closed[known]):
                        # Con un camino mejor se cambia el padre. En una tarea temporal, solo
                        # si el nodo no se ha expandido, y cambiando también el estado: su línea
                        # temporal (y la de sus hijos) tiene que ser la del camino del plan.
                        # Con un nodo ya expandido se pierde la mejora: sus hijos se
                        # calcularon con la línea temporal del camino anterior.
                        if self._temporal:
                            self.states[known] = successor
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
                elif self._admissible_bound and self.g[child] + evaluation.value >= self.limit:
                    self.closed[child] = True  # con h admisible, no puede mejorar la cota
                elif not self.measure_novelty(child):
                    self.closed[child] = True
                else:
                    self.push(child, pending_preferred[child])

    def push(self, node: int, preferred: bool) -> None:
        h = self.h[node]
        assert h is not None
        g = self.g[node]
        width = self.width[node] if self.novelty is not None else 0
        second = None if self.landmarks is None else self.priority(g, self.h_lm[node], width)
        self.open.push(self.priority(g, h, width), (node,), preferred, (h, g), second)

    def pop_batch(self) -> list[int] | int | None:
        """Saca hasta ``batch_size`` nodos por expandir.

        Devuelve un nodo objetivo (``int``) si es el primero que sale, la lista de nodos a
        expandir, o ``None`` si la cola se ha agotado.
        """
        batch: list[int] = []
        while len(batch) < self.batch_size:
            item = self.open.pop()
            if item is None:
                break
            (node,) = item
            if self.closed[node]:
                if self._discarded and node in self._discarded:
                    self.open.retry()  # otra entrada de un nodo descartado: no cuenta
                continue  # entrada obsoleta: el nodo ya se expandió
            if self.g[node] >= self.limit:
                self.closed[node] = True  # la cota ha bajado (incumbente): no la mejora
                continue
            if self.task.is_goal(self.states[node]):
                if not batch:
                    return node
                self.push(node, preferred=False)  # se acepta en la siguiente iteración
                break
            self.closed[node] = True
            check = self._dead_end_check
            if check is not None and not check([self.states[node]])[0]:
                # Callejón descubierto al expandir: se descarta sin gastar el turno de su
                # cola, así que el orden de los nodos vivos es el mismo que si se hubiera
                # descartado al generarlo.
                self.stats.dead_ends += 1
                self._discarded.add(node)
                self.open.retry()
                continue
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
                return self.exhausted()

            # Saca hasta batch_size entradas cuyos sucesores sean estados nuevos.
            children: list[int] = []
            while len(children) < self.batch_size:
                item = self.open.pop()
                if item is None:
                    break
                parent, action_index = item
                action = self.task.actions[action_index]
                parent_state = self.states[parent]
                successor = (
                    action.apply(parent_state)
                    if self._direct_apply
                    else self.task.apply(parent_state, action)
                )
                if successor is None:
                    continue
                known = self.index.get(successor)
                if known is not None:
                    self.merge_landmarks(known, parent, successor)
                    if self._temporal:
                        self.lost_paths = True  # el camino nuevo (y su línea temporal) se pierde
                    continue
                g = self.g[parent] + action.cost
                if self.penalty is not None:
                    g += self.penalty(parent_state, successor)
                if g >= self.limit:
                    continue  # no puede mejorar la cota
                children.append(self.new_node(successor, g, parent, action_index))

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
        if self.completes:
            self.complete(node)
        h = self.h[node]
        assert h is not None
        useful = self.helpful(node)
        limit = self.limit
        open_lists = self.open
        landmarks = self.landmarks is not None
        if not open_lists.groups:
            for action in self.successors(self.states[node]):
                self.stats.generated += 1
                g = self.g[node] + action.cost
                if g >= limit:
                    continue  # no puede mejorar la cota
                key = self.priority(g, h)
                second = self.priority(g, self.h_lm[node]) if landmarks else None
                item = (node, action.index)
                open_lists.push(key, item, action.index in useful, (h, g), second)
            return
        # Una entrada por prioridades (ver _OpenLists.push_successors). Con landmarks, que
        # solo van con GBFS (prioridad sin g), es una por nodo.
        groups: dict[
            tuple[tuple[float, ...], tuple[float, ...] | None], tuple[list[int], list[int]]
        ] = {}
        for action in self.successors(self.states[node]):
            self.stats.generated += 1
            g = self.g[node] + action.cost
            if g >= limit:
                continue  # no puede mejorar la cota
            keys = (self.priority(g, h), self.priority(g, self.h_lm[node]) if landmarks else None)
            group = groups.get(keys)
            if group is None:
                group = groups[keys] = ([], [])
            group[0].append(action.index)
            if action.index in useful:
                group[1].append(action.index)
            open_lists.push_type((h, g), (node, action.index))
        for (key, second), (actions, preferred) in groups.items():
            open_lists.push_successors(key, node, actions, preferred, second)
