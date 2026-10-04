"""Plazos de los literales iniciales temporizados en la relajación (iteración 4).

La heurística de las tareas temporales (h_FF sobre las acciones de inicio y fin) no sabe
nada del tiempo: no ve que un TIL que borra un hecho obliga a darse prisa. Este módulo
añade una comprobación que **descarta estados sin salida por los plazos**, y deja el valor
en manos de la heurística base (:class:`DeadlineHeuristic` la envuelve).

**Instantes más tempranos en la relajación.** Es un Dijkstra como el de h_max, pero el
coste es el tiempo absoluto en el que, como pronto, puede ser cierto cada hecho relajado
(hechos, complementarios y condiciones numéricas de
:class:`~pyplanner.heuristics.relaxation.RelaxedTask`):

* un hecho cierto en el estado lo es desde el último paso que lo cambió en la línea
  temporal (0 si ninguno): un paso nuevo que lo lea no puede ir antes;
* una acción relajada ocurre en el máximo de sus precondiciones; el fin de una acción
  durativa, además, no antes de su inicio más la duración mínima (la precondición
  ``running@a`` pesa esa duración), y el de una acción en curso, no antes de su inicio
  real más la duración mínima;
* un TIL pendiente ocurre en su instante;
* los axiomas y las acciones instantáneas no duran.

**Ventanas.** Un hecho que solo cambian los TIL (``visible`` en satellite: ninguna acción
lo añade) solo es cierto en los intervalos que marcan el estado y los TIL pendientes. Una
precondición sobre él retrasa la acción hasta el primer intervalo que la cumple, y si no
hay ninguno la acción no sirve. Lo mismo con su complementario si ninguna acción lo borra.
Una condición ``over all`` exige además que la acción entera (inicio + duración mínima)
quepa en el intervalo.

Todo son **cotas inferiores** (duraciones mínimas, intervalos cerrados sin la separación ε,
una repetición de cada acción para las condiciones numéricas, invariantes disyuntivas
ignoradas), así que si el objetivo no se alcanza en este tiempo relajado, el estado no
tiene solución: nunca se descarta uno que la tenga. El cálculo para en cuanto una cláusula
del objetivo tiene todos sus hechos con instante definitivo y cabe en sus ventanas.

Solo se usa en las tareas con TIL: sin plazos, no detecta nada que la relajación sin
tiempo no detecte ya.

**Implementaciones.** Las entradas de cada estado (instante inicial de cada hecho relajado
e intervalos de las ventanas) se preparan con NumPy. El Dijkstra tiene una versión de
referencia en Python (:meth:`DeadlineRelaxation.reachable_reference`) y otra compilada con
Numba (:mod:`pyplanner.heuristics.compiled_deadlines`), que dan el mismo resultado.
"""

import heapq
import math
from collections.abc import Callable, Sequence

import numpy as np

from pyplanner.grounding import State
from pyplanner.grounding.numeric import evaluate, variables_of
from pyplanner.grounding.temporal import (
    END_SNAP,
    START_SNAP,
    TIL_SNAP,
    TemporalState,
    TemporalTask,
    Timeline,
)
from pyplanner.heuristics.base import DEAD_END, Evaluation, Heuristic
from pyplanner.heuristics.relaxation import RelaxedTask

INF = math.inf


def _csr(lists: Sequence[Sequence[float]], dtype: type) -> tuple[np.ndarray, np.ndarray]:
    pointers = np.zeros(len(lists) + 1, dtype=np.int64)
    pointers[1:] = np.cumsum([len(items) for items in lists])
    values: np.ndarray = np.array([v for items in lists for v in items], dtype=dtype)
    return pointers, values


class DeadlineRelaxation:
    """Estructuras para calcular instantes más tempranos con ventanas, una vez por tarea.

    Attributes:
        windowed: nodos relajados con ventanas, como ``(hecho, positivo)`` (``positivo``
            es ``False`` si el nodo es el complementario del hecho).
        window_of: índice en ``windowed`` de cada nodo relajado, o -1.
    """

    def __init__(self, task: TemporalTask) -> None:
        self.task = task
        relaxed_task = task.relaxation
        self.relaxed = relaxed = RelaxedTask(relaxed_task)
        num_facts, num_actions = relaxed.num_facts, relaxed.num_actions

        # Hechos con ventanas: los que cambian los TIL y ninguna otra acción añade (o borra,
        # para su complementario).
        til_actions = [i for i, s in enumerate(task.snaps) if s.kind == TIL_SNAP]
        til_changed: set[int] = set()
        for i in til_actions:
            action = task.actions[i]
            til_changed |= {*action.add_facts, *action.del_facts}
        til_changed = {f for f in til_changed if not task.til_done_mask >> f & 1}
        added: set[int] = {axiom.head for axiom in task.axioms}
        deleted: set[int] = set()
        for action, info in zip(relaxed_task.actions, task.snaps, strict=True):
            if info.kind == TIL_SNAP:
                continue
            added |= {*action.add_facts, *(f for e in action.conditional for f in e.add_facts)}
            deleted |= {*action.del_facts, *(f for e in action.conditional for f in e.del_facts)}
        complement = relaxed.complement
        self.windowed: list[tuple[int, bool]] = []
        window_of = [-1] * num_facts
        for fact in sorted(til_changed):
            if fact not in added:
                window_of[fact] = len(self.windowed)
                self.windowed.append((fact, True))
            if fact not in deleted and fact in complement:
                window_of[complement[fact]] = len(self.windowed)
                self.windowed.append((fact, False))
        self.window_of = window_of
        # Cambios de cada TIL (en orden) sobre los hechos con ventanas.
        windowed_facts = sorted({fact for fact, _ in self.windowed})
        self.windowed_facts = np.array(windowed_facts, dtype=np.int64)
        position = {fact: i for i, fact in enumerate(windowed_facts)}
        self.til_changes: list[tuple[float, tuple[tuple[int, bool], ...]]] = []
        for k, i in enumerate(til_actions):
            action = task.actions[i]
            changes = tuple(
                [(position[f], True) for f in action.add_facts if f in position]
                + [(position[f], False) for f in action.del_facts if f in position]
            )
            self.til_changes.append((task.til_times[k], changes))
        self._node_of = [
            (position[fact], positive) for fact, positive in self.windowed
        ]  # (posición del hecho, positivo) de cada nodo con ventanas

        # Por acción relajada: precondiciones que cuentan en el Dijkstra (con su peso),
        # precondiciones con ventanas, instante mínimo (TIL) y, para los inicios con
        # invariantes con ventanas, la duración que tiene que caber.
        self.duration = [_min_duration(task, d) for d in range(len(task.durative))]
        counted: list[list[tuple[int, float]]] = []
        window_pre: list[list[int]] = []
        span_pre: list[list[int]] = []
        self.release: list[float] = []
        self.span: list[float] = []
        effects: list[list[tuple[int, int]]] = []
        for a in range(num_actions):
            operator = relaxed.operator[a]
            snap = task.snaps[operator] if 0 <= operator < len(task.actions) else None
            weight_of: dict[int, float] = {}
            if snap is not None and snap.kind == END_SNAP:
                weight_of[task.durative[snap.durative].running] = self.duration[snap.durative]
            counted.append([(p, weight_of.get(p, 0.0)) for p in relaxed.pre[a] if window_of[p] < 0])
            window_pre.append([window_of[p] for p in relaxed.pre[a] if window_of[p] >= 0])
            release = 0.0
            if snap is not None and snap.kind == TIL_SNAP:
                release = task.til_times[til_actions.index(operator)]
            self.release.append(release)
            span = 0.0
            nodes: list[int] = []
            if snap is not None and snap.kind == START_SNAP:
                invariant = task.durative[snap.durative].invariant
                if invariant is not None and len(invariant) == 1:
                    (clause,) = invariant
                    invariant_nodes = [*clause.pos_facts]
                    invariant_nodes += [complement[f] for f in clause.neg_facts if f in complement]
                    nodes = [window_of[n] for n in invariant_nodes if window_of[n] >= 0]
                    if nodes:
                        span = self.duration[snap.durative]
            self.span.append(span)
            span_pre.append(nodes)
            effects.append(
                [(q, -1) for q in relaxed.add[a] if window_of[q] < 0]
                + [(relaxed.edges[e][1], e) for e in relaxed.edges_of[a]]
            )
        pre_of: list[list[tuple[int, float]]] = [[] for _ in range(num_facts)]
        for a, items in enumerate(counted):
            for p, weight in items:
                pre_of[p].append((a, weight))
        goal_counted = [[f for f in c if window_of[f] < 0] for c in relaxed.goal]
        goal_window = [[window_of[f] for f in c if window_of[f] >= 0] for c in relaxed.goal]
        goal_of: list[list[int]] = [[] for _ in range(num_facts)]
        for c, facts in enumerate(goal_counted):
            for f in facts:
                goal_of[f].append(c)

        # Versión en listas (referencia en Python) y en arrays CSR (Numba).
        self.counted_len = [len(items) for items in counted]
        self.pre_of = pre_of
        self.window_pre = window_pre
        self.span_pre = span_pre
        self.effects = effects
        self.goal_counted = goal_counted
        self.goal_window = goal_window
        self.goal_of = goal_of
        self.arrays = (
            np.array(self.counted_len, dtype=np.int64),
            *_csr([[a for a, _ in items] for items in pre_of], np.int64),
            _csr([[w for _, w in items] for items in pre_of], np.float64)[1],
            *_csr([[q for q, _ in items] for items in effects], np.int64),
            _csr([[e for _, e in items] for items in effects], np.int64)[1],
            np.array(self.release, dtype=np.float64),
            np.array(self.span, dtype=np.float64),
            *_csr(window_pre, np.int64),
            *_csr(span_pre, np.int64),
            *_csr(goal_counted, np.int64),
            *_csr(goal_window, np.int64),
            *_csr(goal_of, np.int64),
        )

        # Para preparar las entradas de cada estado.
        self.num_base = relaxed.num_base
        self.num_bytes = max(1, (relaxed.num_base + 7) // 8)
        self.negated = np.array(relaxed.negated, dtype=np.int64)
        self.windowed_nodes = np.array(
            [n for n in range(num_facts) if window_of[n] >= 0], dtype=np.int64
        )
        self.running_fact = np.array([d.running for d in task.durative], dtype=np.int64)

    # --- Entradas de cada estado ------------------------------------------------------

    def inputs(
        self, states: Sequence[State]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Entradas del Dijkstra para un lote: instante inicial de cada nodo relajado (lote
        × nodos, ``inf`` si no es cierto), intervalos de las ventanas (punteros lote ×
        (ventanas + 1) sobre ``inicio`` y ``fin``) y aristas numéricas útiles (lote ×
        aristas)."""
        relaxed = self.relaxed
        batch = len(states)
        times = np.full((batch, relaxed.num_facts), INF)
        usable = np.ones((batch, len(relaxed.edges)), dtype=np.bool_)
        num_windows = len(self.windowed)
        pointers = np.zeros((batch, num_windows + 1), dtype=np.int64)
        starts: list[float] = []
        ends: list[float] = []
        raw = b"".join(s.facts.to_bytes(self.num_bytes, "little") for s in states)
        bits_of = np.unpackbits(
            np.frombuffer(raw, dtype=np.uint8).reshape(batch, self.num_bytes),
            axis=1,
            bitorder="little",
        )[:, : self.num_base].astype(np.bool_)
        for row, state in enumerate(states):
            timeline = state.peek_timeline() if isinstance(state, TemporalState) else Timeline()
            facts = bits_of[row]
            since, variable_since = self._since(timeline)
            row_times = times[row]
            row_times[: self.num_base][facts] = since[facts]
            for durative, (step, _, _) in timeline.running.items():
                row_times[self.running_fact[durative]] = timeline.earliest[step]
            false_negated = ~facts[self.negated]
            complements = self.num_base + np.flatnonzero(false_negated)
            row_times[complements] = since[self.negated[false_negated]]
            if relaxed.numeric:
                offset = relaxed.numeric_offset
                for j, condition in enumerate(relaxed.numeric):
                    if condition.holds(state.values):
                        keys = variables_of(condition.expression)
                        row_times[offset + j] = max(
                            (variable_since.get(v, 0.0) for v in keys), default=0.0
                        )
                if relaxed.edges:
                    usable[row] = np.array(relaxed.multipliers(state)) != DEAD_END
            row_times[self.windowed_nodes] = INF
            for w, (start_list, end_list) in enumerate(self._windows(state, facts)):
                pointers[row, w] = len(starts)
                starts.extend(start_list)
                ends.extend(end_list)
            pointers[row, num_windows] = len(starts)
        return (
            times,
            pointers,
            np.array(starts, dtype=np.float64),
            np.array(ends, dtype=np.float64),
            usable,
        )

    def _since(self, timeline: Timeline) -> tuple[np.ndarray, dict[int, float]]:
        """Desde cuándo tiene su valor actual cada hecho (array) y cada variable numérica
        que ha cambiado (diccionario): el instante del último paso que lo cambió, o 0."""
        since = np.zeros(self.num_base)
        variables: dict[int, float] = {}
        changer = timeline.changer
        if changer:
            earliest = np.asarray(timeline.earliest)
            keys = np.fromiter(changer.keys(), dtype=np.int64, count=len(changer))
            steps = np.fromiter(changer.values(), dtype=np.int64, count=len(changer))
            is_fact = keys >= 0
            since[keys[is_fact]] = earliest[steps[is_fact]]
            for key, step in zip(keys[~is_fact].tolist(), steps[~is_fact].tolist(), strict=True):
                variables[-key - 1] = timeline.earliest[step]
        return since, variables

    def _windows(self, state: State, facts: np.ndarray) -> list[tuple[list[float], list[float]]]:
        """Intervalos (inicios y fines) de cada nodo con ventanas, según el estado y los TIL
        pendientes."""
        done = (state.facts & self.task.til_done_mask).bit_count()
        value = facts[self.windowed_facts].tolist()
        since = [-INF] * len(value)
        true: list[tuple[list[float], list[float]]] = [([], []) for _ in value]
        false: list[tuple[list[float], list[float]]] = [([], []) for _ in value]
        for time, changes in self.til_changes[done:]:
            for i, new in changes:
                if new == value[i]:
                    continue
                target = true[i] if value[i] else false[i]
                target[0].append(since[i])
                target[1].append(time)
                value[i], since[i] = new, time
        for i, current in enumerate(value):
            target = true[i] if current else false[i]
            target[0].append(since[i])
            target[1].append(INF)
        return [true[i] if positive else false[i] for i, positive in self._node_of]

    # --- Dijkstra de referencia -------------------------------------------------------

    def reachable_reference(self, states: Sequence[State]) -> list[bool]:
        """Para cada estado, ``False`` si el objetivo no se puede alcanzar respetando los
        plazos (implementación de referencia en Python)."""
        times, pointers, starts, ends, usable = self.inputs(states)
        return [
            self._reachable(
                times[row].tolist(),
                pointers[row].tolist(),
                starts.tolist(),
                ends.tolist(),
                usable[row].tolist(),
            )
            for row in range(len(states))
        ]

    def _reachable(
        self,
        time: list[float],
        pointers: list[int],
        starts: list[float],
        ends: list[float],
        usable: list[bool],
    ) -> bool:
        def fit(moment: float, nodes: Sequence[int], span: float) -> float:
            """Primer instante ``≥ moment`` en el que ``[instante, instante + span]`` cabe en
            un intervalo de cada nodo (``inf`` si no hay ninguno)."""
            changed = True
            while changed and moment < INF:
                changed = False
                for w in nodes:
                    fitted = INF
                    for i in range(pointers[w], pointers[w + 1]):
                        candidate = max(moment, starts[i])
                        if candidate + span <= ends[i]:
                            fitted = candidate
                            break
                    if fitted != moment:
                        moment = fitted
                        changed = True
            return moment

        def goal_reached(c: int) -> bool:
            moment = max((time[f] for f in self.goal_counted[c]), default=0.0)
            return fit(moment, self.goal_window[c], 0.0) < INF

        heap = [(t, f) for f, t in enumerate(time) if t < INF]
        heapq.heapify(heap)
        unsatisfied = self.counted_len.copy()
        accumulated = [0.0] * len(unsatisfied)
        pending_goal = [len(c) for c in self.goal_counted]
        if any(n == 0 and goal_reached(c) for c, n in enumerate(pending_goal)):
            return True
        done = [False] * len(time)

        def trigger(a: int) -> None:
            moment = fit(max(self.release[a], accumulated[a]), self.window_pre[a], 0.0)
            if self.span_pre[a]:
                moment = fit(moment, self.span_pre[a], self.span[a])
            if moment == INF:
                return
            for q, edge in self.effects[a]:
                if (edge < 0 or usable[edge]) and moment < time[q]:
                    time[q] = moment
                    heapq.heappush(heap, (moment, q))

        for a, count in enumerate(unsatisfied):
            if count == 0:
                trigger(a)
        while heap:
            t, f = heapq.heappop(heap)
            if done[f] or t != time[f]:
                continue
            done[f] = True
            for c in self.goal_of[f]:
                pending_goal[c] -= 1
                if pending_goal[c] == 0 and goal_reached(c):
                    return True
            for a, weight in self.pre_of[f]:
                if t + weight > accumulated[a]:
                    accumulated[a] = t + weight
                unsatisfied[a] -= 1
                if unsatisfied[a] == 0:
                    trigger(a)
        return False


def _min_duration(task: TemporalTask, durative: int) -> float:
    """Duración mínima de una acción durativa en cualquier estado: la de sus restricciones
    ``=`` y ``>=`` de valor constante (las que dependen del estado no cuentan), o ε."""
    instance = task.durative[durative]
    lower = task.epsilon
    for bound in (*instance.start_bounds, *instance.end_bounds):
        if bound.comparator in ("=", ">=") and not variables_of(bound.value):
            value = evaluate(bound.value, ())
            if value is not None:
                lower = max(lower, value)
    return lower


class DeadlineHeuristic(Heuristic):
    """Envuelve una heurística y declara callejón sin salida los estados que no pueden
    alcanzar el objetivo respetando los plazos de los TIL (ver la documentación del
    módulo). El resto de estados conservan el valor y las acciones útiles de la base.

    Con ``compiled`` (por defecto, si Numba está instalado), el Dijkstra va compilado.
    """

    def __init__(self, task: TemporalTask, base: Heuristic, compiled: bool = True) -> None:
        super().__init__(task)
        self.base = base
        self.name = base.name
        self.device = base.device
        self.provides_preferred = base.provides_preferred
        self.deadlines = DeadlineRelaxation(task)
        self.compiled = compiled
        #: ``True`` si la búsqueda hace la comprobación al expandir (ver
        #: :meth:`defer_dead_end_check`).
        self.deferred = False

    def reachable(self, states: Sequence[State]) -> list[bool]:
        """``False`` para los estados que no pueden cumplir los plazos.

        Los estados sin TIL pendientes no se comprueban: ya no hay plazos, las ventanas no
        cambian, y la relajación de la heurística base detecta los mismos callejones."""
        task = self.task
        assert isinstance(task, TemporalTask)
        pending = len(task.til_times)
        mask = task.til_done_mask
        checked = [s for s in states if (s.facts & mask).bit_count() < pending]
        if not checked:
            return [True] * len(states)
        if self.compiled:
            from pyplanner.heuristics.compiled_deadlines import reachable

            results = iter(reachable(self.deadlines, checked))
        else:
            results = iter(self.deadlines.reachable_reference(checked))
        return [next(results) if (s.facts & mask).bit_count() < pending else True for s in states]

    def defer_dead_end_check(self) -> Callable[[Sequence[State]], list[bool]]:
        self.deferred = True
        return self.reachable

    def evaluate_batch(self, states: Sequence[State]) -> list[Evaluation]:
        if self.deferred:
            return self.base.evaluate_batch(states)
        live = self.reachable(states)
        kept = [s for s, ok in zip(states, live, strict=True) if ok]
        evaluations = iter(self.base.evaluate_batch(kept) if kept else [])
        return [next(evaluations) if ok else Evaluation(DEAD_END) for ok in live]
