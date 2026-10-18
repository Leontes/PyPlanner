"""h_TFF: plan relajado temporal (trabajo opcional, punto 3: parte B de la heurística temporal).

h_FF no sabe nada del tiempo: en satellite con ventanas de visibilidad, su plan relajado
usa antenas que ya no van a ver el objetivo. La parte A
(:mod:`pyplanner.heuristics.deadlines`) solo descarta los estados que no pueden cumplir los
plazos. h_TFF extrae además un **plan relajado que respeta los tiempos**, como el grafo de
planificación temporal de POPF y OPTIC, y lo usa como heurística.

**Instantes más tempranos.** Los de la parte A (:class:`DeadlineRelaxation`): el instante en
que, como pronto, puede ser cierto cada nodo relajado (hechos, complementarios y
condiciones numéricas), con las duraciones mínimas, los TIL pendientes y las ventanas de
los hechos que solo cambian los TIL. Una acción relajada ocurre en el máximo de sus
precondiciones (más la duración mínima en el fin de una durativa), retrasada hasta que
quepa en sus ventanas.

**Soportador de cada hecho**, independiente del orden del cálculo, para que las tres
implementaciones den lo mismo: la clave de un hecho es (instante, nivel), donde un hecho
cierto en el estado tiene nivel 0 y una acción da a sus efectos el nivel 1 + el máximo
nivel de sus precondiciones. El soportador es la acción relajada de menor índice entre las
que dan al hecho su clave mínima. El nivel crece en cada paso, así que el plan relajado no
tiene ciclos (aunque haya acciones instantáneas). Es el mismo desempate que h_FF, con el
instante en lugar del coste h_add.

**Cláusula del objetivo:** la que, como pronto, se cumple antes (instante de sus hechos y
sus ventanas; la primera si hay empate). Si ninguna se puede cumplir, el estado es un
callejón sin salida: es exactamente la comprobación de la parte A.

**Plan relajado y valor**, como en h_FF: desde los hechos de la cláusula que no son ciertos
en el estado, el soportador de cada uno y, de él, sus precondiciones que no son ciertas. El
valor suma, por cada operador del plan, su coste por el máximo de repeticiones de sus
acciones relajadas (las aristas numéricas). Las acciones útiles son los operadores de las
acciones del plan con todas sus precondiciones ciertas en el estado (también las de los
hechos con ventanas).

**Implementaciones:** la referencia en Python (:meth:`TemporalRelaxedPlan.evaluate_reference`,
un Dijkstra por (instante, nivel)), Numba (:mod:`pyplanner.heuristics.compiled_temporal_ff`)
y PyTorch (:mod:`pyplanner.heuristics.tensor_temporal_ff`, con puntos fijos sobre el lote),
con exactamente el mismo resultado.

**Solo en las tareas con TIL.** Sin plazos, el único cambio frente a h_FF es elegir los
soportadores por instante más temprano en lugar de por coste, y eso da peores planes
relajados (como pasa con h_max frente a h_add): en driverlog y zenotravel temporales
resolvía menos instancias. Con TIL, en cambio, respeta las ventanas: en satellite resuelve
más y con muchas menos expansiones. Quitar a h_FF solo las acciones que no caben en el
tiempo, manteniendo los soportadores por coste, no sirve: casi nunca son las que elige. Así
que ``tff`` es h_TFF en las tareas con TIL y h_FF en las demás (ver
:func:`pyplanner.heuristics.create_heuristic`); :class:`TemporalFFHeuristic` funciona en
cualquier tarea temporal.
"""

import heapq
from collections.abc import Sequence

import numpy as np

from pyplanner.grounding import State
from pyplanner.grounding.temporal import TemporalTask
from pyplanner.heuristics.base import DEAD_END, Evaluation, Heuristic
from pyplanner.heuristics.deadlines import INF, DeadlineRelaxation, _csr


class TemporalRelaxedPlan:
    """Estructuras del plan relajado temporal, una vez por tarea (sobre las de la parte A).

    Attributes:
        deadlines: la relajación con instantes y ventanas de la parte A.
        counted: por acción relajada, sus precondiciones sin ventanas (las que hay que
            conseguir).
    """

    def __init__(self, task: TemporalTask) -> None:
        self.task = task
        self.deadlines = d = DeadlineRelaxation(task)
        relaxed = d.relaxed
        self.counted = [
            [p for p in relaxed.pre[a] if d.window_of[p] < 0] for a in range(relaxed.num_actions)
        ]
        self.operator = relaxed.operator
        self.operator_cost = relaxed.operator_cost
        self.axiom_operator = relaxed.axiom_operator
        self.num_operators = relaxed.num_operators
        self.arrays = (
            *_csr(self.counted, np.int64),
            np.array(self.operator, dtype=np.int64),
            np.array(self.operator_cost, dtype=np.float64),
        )

    def evaluate_reference(self, states: Sequence[State]) -> list[Evaluation]:
        """h_TFF de cada estado (implementación de referencia en Python)."""
        times, pointers, starts, ends, usable, multipliers, window_true = (
            self.deadlines.plan_inputs(states)
        )
        return [
            self._evaluate(
                times[row].tolist(),
                pointers[row].tolist(),
                starts.tolist(),
                ends.tolist(),
                usable[row].tolist(),
                multipliers[row].tolist(),
                window_true[row].tolist(),
            )
            for row in range(len(states))
        ]

    def _evaluate(
        self,
        initial: list[float],
        pointers: list[int],
        starts: list[float],
        ends: list[float],
        usable: list[bool],
        multipliers: list[float],
        window_true: list[bool],
    ) -> Evaluation:
        d = self.deadlines

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

        num_facts = len(initial)
        time = list(initial)
        never = num_facts + 1  # nivel de un hecho todavía sin alcanzar
        level = [0 if t < INF else never for t in time]
        supporter = [-1] * num_facts
        repeats_of = [1.0] * num_facts  # repeticiones de la arista del soportador
        heap = [(t, 0, f) for f, t in enumerate(time) if t < INF]
        heapq.heapify(heap)
        unsatisfied = d.counted_len.copy()
        accumulated = [0.0] * len(unsatisfied)
        action_level = [0] * len(unsatisfied)
        pending_goal = [len(c) for c in d.goal_counted]
        best_moment, best_clause = INF, -1

        def consider(c: int) -> None:
            nonlocal best_moment, best_clause
            moment = max((time[f] for f in d.goal_counted[c]), default=0.0)
            moment = fit(moment, d.goal_window[c], 0.0)
            # Las cláusulas se completan en cualquier orden: con empate, la de menor índice.
            if moment < INF and (
                moment < best_moment or (moment == best_moment and c < best_clause)
            ):
                best_moment, best_clause = moment, c

        def trigger(a: int) -> None:
            moment = fit(max(d.release[a], accumulated[a]), d.window_pre[a], 0.0)
            if d.span_pre[a]:
                moment = fit(moment, d.span_pre[a], d.span[a])
            if moment == INF:
                return
            lev = action_level[a] + 1
            for q, edge in d.effects[a]:
                if edge >= 0 and not usable[edge]:
                    continue
                m = 1.0 if edge < 0 else multipliers[edge]
                if moment < time[q] or (moment == time[q] and lev < level[q]):
                    time[q], level[q], supporter[q], repeats_of[q] = moment, lev, a, m
                    heapq.heappush(heap, (moment, lev, q))
                elif moment == time[q] and lev == level[q] and a < supporter[q]:
                    supporter[q], repeats_of[q] = a, m

        for c, count in enumerate(pending_goal):
            if count == 0:
                consider(c)
        for a, count in enumerate(unsatisfied):
            if count == 0:
                trigger(a)
        done = [False] * num_facts
        # Se para cuando ninguna cláusula pendiente puede cumplirse antes que la mejor: sus
        # hechos que faltan no son ciertos antes del siguiente instante del montículo.
        while heap and heap[0][0] <= best_moment:
            t, lev, f = heapq.heappop(heap)
            if done[f] or t != time[f] or lev != level[f]:
                continue  # entrada obsoleta
            done[f] = True
            for c in d.goal_of[f]:
                pending_goal[c] -= 1
                if pending_goal[c] == 0:
                    consider(c)
            for a, weight in d.pre_of[f]:
                if t + weight > accumulated[a]:
                    accumulated[a] = t + weight
                if lev > action_level[a]:
                    action_level[a] = lev
                unsatisfied[a] -= 1
                if unsatisfied[a] == 0:
                    trigger(a)
        if best_clause < 0:
            return Evaluation(DEAD_END)
        return self._relaxed_plan(
            d.goal_counted[best_clause], initial, supporter, repeats_of, window_true
        )

    def _relaxed_plan(
        self,
        clause: Sequence[int],
        initial: list[float],
        supporter: list[int],
        repeats_of: list[float],
        window_true: list[bool],
    ) -> Evaluation:
        """Extrae el plan relajado desde los hechos de ``clause`` (ver la documentación del
        módulo)."""
        repeats: dict[int, float] = {}  # acción relajada → repeticiones necesarias
        marked: set[int] = set()
        stack = [g for g in clause if initial[g] == INF]
        while stack:
            f = stack.pop()
            if f in marked:
                continue
            marked.add(f)
            best = supporter[f]
            if best not in repeats:
                repeats[best] = repeats_of[f]
                stack.extend(p for p in self.counted[best] if initial[p] == INF)
            elif repeats_of[f] > repeats[best]:
                repeats[best] = repeats_of[f]
        per_operator: dict[int, float] = {}
        preferred: set[int] = set()
        window_pre = self.deadlines.window_pre
        for a, times in repeats.items():
            operator = self.operator[a]
            if times > per_operator.get(operator, 0.0):
                per_operator[operator] = times
            if (
                operator != self.axiom_operator
                and all(initial[p] < INF for p in self.counted[a])
                and all(window_true[w] for w in window_pre[a])
            ):
                preferred.add(operator)
        # Suma en orden de operador: el mismo orden de sumas que en Numba y PyTorch.
        value = sum(self.operator_cost[o] * times for o, times in sorted(per_operator.items()))
        return Evaluation(value, tuple(sorted(preferred)))


class TemporalFFHeuristic(Heuristic):
    """h_TFF (ver la documentación del módulo) en una tarea temporal.

    Args:
        device: ``"python"`` (referencia), ``"numba"``, ``"cpu"`` o ``"cuda"``.
    """

    name = "tff"
    provides_preferred = True

    def __init__(self, task: TemporalTask, device: str = "python") -> None:
        super().__init__(task)
        self.device = device
        self.plan = TemporalRelaxedPlan(task)
        self._tensors = None
        if device in ("cpu", "cuda"):
            from pyplanner.heuristics.tensor_temporal_ff import TensorTemporalRelaxedPlan

            self._tensors = TensorTemporalRelaxedPlan(self.plan, device)

    def evaluate_batch(self, states: Sequence[State]) -> list[Evaluation]:
        if self.device == "numba":
            from pyplanner.heuristics.compiled_temporal_ff import relaxed_plans

            return relaxed_plans(self.plan, states)
        if self._tensors is not None:
            return self._tensors.evaluate(states)
        return self.plan.evaluate_reference(states)
