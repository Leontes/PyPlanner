"""Búsquedas locales: *enforced hill-climbing* (FF) y paseos aleatorios (Arvand).

Dan diversidad al portfolio (trabajo opcional, punto 5): no mantienen una cola global como
las búsquedas de primero el mejor, sino que avanzan desde el mejor estado encontrado.

**Enforced hill-climbing** (``algorithm="ehc"``; Hoffmann y Nebel, 2001). Desde el estado
actual, una búsqueda en anchura hasta el primer estado de h menor (o un objetivo), que
pasa a ser el actual. Con ``preferred_operators``, la búsqueda en anchura solo usa las
acciones útiles de cada estado; si así no encuentra salida, lo intenta con todas. Si
tampoco, la búsqueda sigue con GBFS (la misma heurística y el tiempo que queda), como FF:
así es completa. Los sucesores se evalúan en lotes de ``batch_size`` (con 1, es la de FF).

**Paseos aleatorios** (``algorithm="mrw"``; *Monte-Carlo random walks*, Nakhost y Müller,
2009). Desde el estado actual se dan paseos aleatorios de unos pocos pasos y solo se evalúa
el estado al que llega cada uno (en lotes). En cuanto uno mejora la menor h vista, la
búsqueda salta a él; si en :data:`WALKS_PER_STEP` paseos ninguno la mejora, salta al mejor
de todos igualmente, y tras :data:`MAX_STALLED_STEPS` saltos así vuelve a empezar desde el
estado inicial. Si los paseos no mejoran, se alargan (:data:`EXTEND_PERIOD`,
:data:`EXTEND_RATE`). Con ``preferred_operators``, cada paso elige las acciones con
probabilidad proporcional a exp(Q/τ), donde Q cuenta las veces que la acción ha sido útil
en los estados evaluados (MHA, *monte-carlo helpful actions*). Es incompleta: sin plan,
termina por el límite de tiempo.

**Planes.** Los dos dejan ciclos en el plan (volver a un estado ya visitado), sobre todo
los paseos. Se quitan al final si el plan sin ellos sigue siendo válido y no cuesta más
(:meth:`_LocalSearch.without_cycles`).

**Preferencias, cota e incumbente**: como en :mod:`pyplanner.search.engine` (se descartan
los pasos cuyo g llega a la cota, y cada estado al que se avanza puede ser un incumbente).
"""

import math
import random
import time
from dataclasses import replace

from pyplanner.grounding import GroundAction, State
from pyplanner.heuristics import Evaluation
from pyplanner.search.engine import (
    _NO_PARENT,
    BOUND_TOLERANCE,
    EagerSearch,
    _BestFirstSearch,
)
from pyplanner.search.result import SearchResult, SearchStatus

#: Paseos por salto (Arvand: 2000).
WALKS_PER_STEP = 2000
#: Longitud inicial de los paseos (Arvand: 10).
WALK_LENGTH = 10
#: Paseos sin mejorar tras los que se alargan, y cuánto (Arvand: 300 y 1,5).
EXTEND_PERIOD = 300
EXTEND_RATE = 1.5
#: Saltos seguidos sin mejorar tras los que se vuelve a empezar (Arvand: 7).
MAX_STALLED_STEPS = 7
#: Temperatura de MHA (Arvand: 10).
MHA_TEMPERATURE = 10.0
#: Paseos cuyos estados finales se evalúan juntos (como mínimo; o ``batch_size``).
WALK_BATCH = 32


class _LocalSearch(_BestFirstSearch):
    """Parte común: transiciones con su coste y la cota, y planes sin ciclos."""

    def transition(self, state: State, action: GroundAction) -> tuple[State, float] | None:
        """El sucesor y lo que cuesta llegar a él (con la penalización de las preferencias);
        ``None`` si la acción no se puede aplicar."""
        if self._direct_apply:
            successor, cost = action.apply(state), action.cost
        else:
            successor, cost = self.task.transition(state, action)
        if successor is None:
            return None
        if self.penalty is not None:
            cost += self.penalty(state, successor)
        return successor, cost

    def step(self, state: State, action: GroundAction, g: float) -> tuple[State, float] | None:
        """Un sucesor generado por la búsqueda y su g; ``None`` si la acción no se puede
        aplicar o si g llega a la cota."""
        self.stats.generated += 1
        result = self.transition(state, action)
        if result is None or g + result[1] >= self.limit:
            return None
        return result[0], g + result[1]

    def alive(self, state: State) -> bool:
        """``False`` si la comprobación aplazada de callejones (los plazos de los TIL) lo
        descarta."""
        check = self._dead_end_check
        return check is None or check([state])[0]

    def without_cycles(self, plan: list[int]) -> tuple[tuple[int, ...], float] | None:
        """El plan sin los tramos que vuelven a un estado ya visitado, con su coste; ``None``
        si el plan sin ellos no es válido (en una tarea temporal, por ejemplo, los estados
        iguales pueden tener distinta línea temporal)."""
        actions = self.task.actions
        states = [self.task.init]
        position = {self.task.init: 0}
        kept: list[int] = []
        state = self.task.init
        for index in plan:
            step = self.transition(state, actions[index])
            if step is None:
                return None
            state = step[0]
            if state in position:  # ciclo: se vuelve al estado de entonces
                cut = position[state]
                for dropped in states[cut + 1 :]:
                    del position[dropped]
                del states[cut + 1 :]
                del kept[cut:]
                continue
            position[state] = len(states)
            states.append(state)
            kept.append(index)
        return self.replay(kept)

    def replay(self, plan: list[int]) -> tuple[tuple[int, ...], float] | None:
        """El plan y su coste si es válido y llega al objetivo; si no, ``None``."""
        state, g = self.task.init, 0.0
        for index in plan:
            step = self.transition(state, self.task.actions[index])
            if step is None:
                return None
            state, g = step[0], g + step[1]
        return (tuple(plan), g) if self.task.is_goal(state) else None

    def solved(self, plan: list[int], cost: float) -> SearchResult:
        """El resultado con el plan sin ciclos, si es válido y no cuesta más."""
        shorter = self.without_cycles(plan)
        if shorter is not None and shorter[1] <= cost + BOUND_TOLERANCE:
            return self.finish(SearchStatus.SOLVED, shorter[0], shorter[1])
        return self.finish(SearchStatus.SOLVED, tuple(plan), cost)


class EnforcedHillClimbing(_LocalSearch):
    """*Enforced hill-climbing* (ver la documentación del módulo)."""

    def _run(self) -> SearchResult:
        root = self.new_node(self.task.init, 0, _NO_PARENT, _NO_PARENT)
        [evaluation] = self.evaluate([root])
        if evaluation.is_dead_end:
            return self.finish(SearchStatus.UNSOLVABLE)
        self._dead_end_check = self.heuristic.defer_dead_end_check()
        current = root
        while not self.task.is_goal(self.states[current]):
            outcome = self.climb(current, self.config.preferred_operators)
            if outcome is None and self.config.preferred_operators:
                outcome = self.climb(current, helpful_only=False)
            if isinstance(outcome, SearchStatus):
                return self.finish(outcome)
            if outcome is None:
                return self.fall_back()
            current = outcome
        return self.solved(self.path(current), self.g[current])

    def climb(self, current: int, helpful_only: bool) -> int | SearchStatus | None:
        """Búsqueda en anchura desde ``current`` hasta el primer estado de h menor o un
        objetivo. Devuelve su nodo, ``None`` si se agota o el motivo si hay que parar."""
        h0 = self.h[current]
        assert h0 is not None
        seen = {self.states[current]}
        layer = [current]
        while layer:
            following: list[int] = []
            pending: list[int] = []
            for node in layer:
                reason = self.interrupted()
                if reason is not None:
                    return reason
                self.stats.expanded += 1
                if self.completes:
                    self.complete(node)
                state, g = self.states[node], self.g[node]
                useful = self.helpful(node) if helpful_only else None
                for action in self.successors(state):
                    if useful is not None and action.index not in useful:
                        continue
                    step = self.step(state, action, g)
                    if step is None or step[0] in seen:
                        continue
                    successor, new_g = step
                    seen.add(successor)
                    child = self.new_node(successor, new_g, node, action.index)
                    if self.task.is_goal(successor):
                        return child
                    pending.append(child)
                    if len(pending) >= self.batch_size:
                        better = self.better(pending, h0, following)
                        if better is not None:
                            return better
                        pending = []
            better = self.better(pending, h0, following)
            if better is not None:
                return better
            layer = following
        return None

    def better(self, pending: list[int], h0: float, following: list[int]) -> int | None:
        """Evalúa los nodos: devuelve el primero de h menor que ``h0``; los demás que no son
        callejones pasan a ``following`` (la siguiente capa)."""
        for node, evaluation in zip(pending, self.evaluate(pending), strict=True):
            if evaluation.is_dead_end or not self.alive(self.states[node]):
                self.stats.dead_ends += 1
                continue
            if evaluation.value < h0:
                return node
            following.append(node)
        return None

    def fall_back(self) -> SearchResult:
        """Sin salida (un callejón que la heurística no reconoce): GBFS con la misma
        heurística y el tiempo que queda, como FF."""
        config = self.config
        elapsed = time.perf_counter() - self._start
        max_time = None if config.max_time is None else config.max_time - elapsed
        if max_time is not None and max_time <= 0:
            return self.finish(SearchStatus.LIMIT_REACHED)
        gbfs = config.with_changes(algorithm="gbfs", max_time=max_time)
        bound = None if self.limit == float("inf") else self.limit + BOUND_TOLERANCE
        result = EagerSearch(self.task, gbfs, self.heuristic, self.stop, bound, self.on_plan).run()
        statistics = replace(self.stats)
        statistics.add(result.statistics)
        statistics.time = time.perf_counter() - self._start
        if not result.solved and self.incumbent is not None:
            self.stats = statistics
            return self.finish(result.status)
        message = "EHC sin salida: sigue con GBFS" + (
            f"; {result.message}" if result.message else ""
        )
        return replace(result, statistics=statistics, config_name=config.name, message=message)


class RandomWalks(_LocalSearch):
    """Paseos aleatorios de Arvand (ver la documentación del módulo)."""

    def _run(self) -> SearchResult:
        root = self.new_node(self.task.init, 0, _NO_PARENT, _NO_PARENT)
        [evaluation] = self.evaluate([root])
        if evaluation.is_dead_end:
            return self.finish(SearchStatus.UNSOLVABLE)
        if self.task.is_goal(self.task.init):
            return self.finish(SearchStatus.SOLVED, (), 0.0)
        self._dead_end_check = self.heuristic.defer_dead_end_check()
        self.random = random.Random(self.config.seed)
        #: Valor de cada acción para MHA (veces que ha sido útil en un estado evaluado).
        self.values: dict[int, float] = {}
        self.segment: dict[int, tuple[int, ...]] = {}  # nodo → acciones desde su padre
        self.learn(evaluation)
        chunk = max(WALK_BATCH, self.config.batch_size)
        current = root
        best_h = evaluation.value
        stalled = 0
        while True:
            length = WALK_LENGTH
            walks = since_extension = 0
            best: tuple[float, State, float, list[int]] | None = None
            jumped = False
            while walks < WALKS_PER_STEP:
                reason = self.interrupted()
                if reason is not None:
                    return self.finish(reason)
                ends = []
                for _ in range(chunk):
                    end = self.walk(current, length)
                    if end is None:
                        continue
                    state, g, actions, goal = end
                    if goal:
                        plan = [*self.path(current), *actions]
                        return self.solved(plan, g)
                    ends.append((state, g, actions))
                walks += chunk
                since_extension += chunk
                for (state, g, actions), result in zip(
                    ends, self.evaluate_states(ends), strict=True
                ):
                    if result.is_dead_end or not self.alive(state):
                        self.stats.dead_ends += 1
                        continue
                    self.learn(result)
                    if best is None or result.value < best[0]:
                        best = (result.value, state, g, actions)
                if best is not None and best[0] < best_h:
                    jumped = True
                    break
                if since_extension >= EXTEND_PERIOD:
                    length = math.ceil(length * EXTEND_RATE)
                    since_extension = 0
            if best is None:  # todos los paseos acaban en callejones
                stalled = MAX_STALLED_STEPS
            elif jumped:
                stalled = 0
            else:
                stalled += 1
            if stalled >= MAX_STALLED_STEPS:
                current, best_h, stalled = 0, self.h[0] or 0.0, 0  # vuelve a empezar
                continue
            assert best is not None
            value, state, g, actions = best
            current = self.jump(current, state, g, actions, value)
            best_h = value

    def walk(self, start: int, length: int) -> tuple[State, float, list[int], bool] | None:
        """Un paseo de ``length`` pasos desde el nodo ``start``: el estado al que llega, su
        g, las acciones y si es un objetivo (entonces se para ahí). ``None`` si se queda sin
        acciones aplicables."""
        state, g = self.states[start], self.g[start]
        actions: list[int] = []
        rng = self.random
        mha = self.config.preferred_operators
        for _ in range(length):
            candidates = list(self.successors(state))
            self.stats.expanded += 1
            while candidates:
                if mha and self.values:
                    weights = [
                        math.exp(min(self.values.get(a.index, 0.0) / MHA_TEMPERATURE, 50.0))
                        for a in candidates
                    ]
                    position = rng.choices(range(len(candidates)), weights)[0]
                else:
                    position = rng.randrange(len(candidates))
                action = candidates[position]
                step = self.step(state, action, g)
                if step is not None:
                    break
                candidates[position] = candidates[-1]  # no se puede aplicar: otra
                candidates.pop()
            else:
                return None
            state, g = step
            actions.append(action.index)
            if self.task.is_goal(state):
                return state, g, actions, True
        return state, g, actions, False

    def evaluate_states(self, ends: list[tuple[State, float, list[int]]]) -> list[Evaluation]:
        """Evalúa los estados finales de los paseos en un lote (sin crear nodos)."""
        if not ends:
            return []
        evaluations = self.heuristic.evaluate_batch([state for state, _, _ in ends])
        stats = self.stats
        stats.evaluated += len(ends)
        stats.evaluation_batches += 1
        stats.largest_batch = max(stats.largest_batch, len(ends))
        return evaluations

    def learn(self, evaluation: Evaluation) -> None:
        """MHA: cada acción útil en un estado evaluado vale una unidad más."""
        if self.config.preferred_operators:
            values = self.values
            for action in evaluation.preferred:
                values[action] = values.get(action, 0.0) + 1.0

    def jump(self, parent: int, state: State, g: float, actions: list[int], value: float) -> int:
        """Avanza al estado final de un paseo: un nodo nuevo, hijo de ``parent`` por todas
        las acciones del paseo."""
        node = self.new_node(state, g, parent, actions[-1] if actions else _NO_PARENT)
        self.h[node] = value
        self.segment[node] = tuple(actions)
        if self.completes:
            self.complete(node)
        return node

    def path(self, node: int) -> list[int]:
        """Las acciones desde la raíz: cada salto aporta las de su paseo."""
        segments: list[tuple[int, ...]] = []
        while self.parent[node] != _NO_PARENT:
            segments.append(self.segment[node])
            node = self.parent[node]
        return [a for segment in reversed(segments) for a in segment]
