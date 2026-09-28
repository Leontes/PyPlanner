"""Heurísticas de la relajación sin borrados: h_max, h_add y h_FF (implementación de referencia).

En la relajación sin borrados se ignoran los efectos de borrado, así que un hecho, una vez
alcanzado, sigue siendo cierto. El coste de alcanzar cada hecho se define así:

* h_max: coste(a) + **máximo** de los costes de las precondiciones de la acción ``a``;
* h_add: coste(a) + **suma** de los costes de las precondiciones (Bonet y Geffner, 2001).

El coste de un hecho es el mínimo sobre las acciones que lo añaden (0 si es cierto en el
estado). Se calcula con un Dijkstra generalizado: los hechos se extraen de un montículo por
orden de coste, y una acción se "dispara" cuando se han extraído todas sus precondiciones.

h_FF (Hoffmann y Nebel, 2001) extrae un **plan relajado** a partir de los costes de h_add:
para cada objetivo pendiente elige su mejor *soportador* y repite con sus precondiciones.
El valor es el coste de los operadores del plan relajado, y las acciones útiles son las del
plan que son aplicables en el estado.

**Tarea relajada** (:class:`RelaxedTask`), a partir de la tarea ADL normalizada:

* **Hechos complementarios.** Cada hecho *p* que aparece negado en alguna condición tiene
  un hecho "no-*p*". Es cierto si *p* es falso en el estado, y lo hacen cierto las acciones
  que borran *p*. Así las precondiciones negativas pasan a ser positivas dentro de la
  relajación, como hace FF. El estado de la búsqueda no cambia.
* **Acciones relajadas.** Cada operador da una acción relajada para sus efectos
  incondicionales y otra por cada efecto condicional, cuya precondición incluye la
  condición del efecto (como en Fast Downward). Cada una recuerda su operador: h_FF cuenta
  el coste de cada operador **una sola vez**, aunque use varios de sus efectos.
* **Objetivo disyuntivo.** El valor es el mínimo entre las cláusulas. h_FF extrae el plan
  relajado de la mejor cláusula; si hay empate, de la de menor índice. Sin cláusulas, el
  objetivo es imposible.

Todos los costes de acción son positivos. Por eso, cuando se han extraído todos los hechos
del objetivo, los costes que necesita la extracción del plan relajado ya son definitivos, y
el cálculo puede terminar antes.
"""

import heapq
from collections.abc import Sequence

from pyplanner.grounding import GroundTask, State, bits, mask_of
from pyplanner.heuristics.base import DEAD_END, Evaluation, Heuristic


class RelaxedTask:
    """Estructuras de la relajación, precalculadas una vez por tarea.

    Las comparten la implementación de referencia y la de PyTorch.

    Attributes:
        num_facts: hechos de la tarea más hechos complementarios.
        negated: hechos de la tarea que tienen complementario; el complementario de
            ``negated[i]`` es el hecho ``len(task.facts) + i``.
        pre, add, cost, operator: precondiciones, añadidos, coste y operador de cada acción
            relajada, en orden determinista (el índice decide los empates de h_FF).
        operator_cost: coste de cada operador de la tarea.
        goal: hechos de cada cláusula del objetivo, ya con complementarios.
    """

    def __init__(self, task: GroundTask) -> None:
        self.task = task
        base_facts = len(task.facts)
        negated = sorted(
            {f for a in task.actions for f in a.neg_facts}
            | {f for a in task.actions for e in a.conditional for f in e.condition.neg_facts}
            | {f for clause in task.goal for f in clause.neg_facts}
        )
        complement = {f: base_facts + i for i, f in enumerate(negated)}
        self.num_facts = base_facts + len(negated)
        self.negated = tuple(negated)
        self.negated_mask = mask_of(negated)
        self._complement = complement

        def literals(pos: Sequence[int], neg: Sequence[int]) -> tuple[int, ...]:
            return tuple(sorted({*pos, *(complement[f] for f in neg)}))

        def effects(adds: Sequence[int], deletes: Sequence[int]) -> tuple[int, ...]:
            return tuple(sorted({*adds, *(complement[f] for f in deletes if f in complement)}))

        self.pre: list[tuple[int, ...]] = []
        self.add: list[tuple[int, ...]] = []
        self.cost: list[int] = []
        self.operator: list[int] = []
        for action in task.actions:
            base_pre = literals(action.pre_facts, action.neg_facts)
            base_add = effects(action.add_facts, action.del_facts)
            if base_add:
                self._append(base_pre, base_add, action.cost, action.index)
            for effect in action.conditional:
                condition = literals(effect.condition.pos_facts, effect.condition.neg_facts)
                effect_add = effects(effect.add_facts, effect.del_facts)
                if effect_add:
                    pre = tuple(sorted({*base_pre, *condition}))
                    self._append(pre, effect_add, action.cost, action.index)
        self.num_actions = len(self.pre)
        self.operator_cost = [a.cost for a in task.actions]
        self.goal = [literals(c.pos_facts, c.neg_facts) for c in task.goal]
        self.goal_facts = frozenset(f for clause in self.goal for f in clause)

        self.pre_of: list[list[int]] = [[] for _ in range(self.num_facts)]
        self.achievers: list[list[int]] = [[] for _ in range(self.num_facts)]
        for a in range(self.num_actions):
            for f in self.pre[a]:
                self.pre_of[f].append(a)
            for f in self.add[a]:
                self.achievers[f].append(a)
        self.no_pre = [a for a in range(self.num_actions) if not self.pre[a]]
        self._pre_len = [len(p) for p in self.pre]

    def _append(self, pre: tuple[int, ...], add: tuple[int, ...], cost: int, operator: int) -> None:
        self.pre.append(pre)
        self.add.append(add)
        self.cost.append(cost)
        self.operator.append(operator)

    def true_facts(self, state: State) -> list[int]:
        """Hechos relajados ciertos en ``state``: los del estado y los complementarios de
        los hechos negables que son falsos."""
        complement = self._complement
        return [*bits(state), *(complement[f] for f in bits(~state & self.negated_mask))]

    def fact_costs(self, state: State, combine_max: bool) -> list[float]:
        """Costes h_max (``combine_max``) o h_add de cada hecho relajado desde ``state``.

        Se detiene cuando se han extraído todos los hechos del objetivo. Los hechos no
        extraídos pueden quedar con un coste provisional o infinito; la extracción del
        plan relajado no los necesita.
        """
        return self._dijkstra(state, combine_max, None)

    def fact_costs_and_supporters(self, state: State) -> tuple[list[float], list[int]]:
        """Costes h_add y mejor soportador de cada hecho extraído (para h_FF).

        El soportador se anota durante el propio Dijkstra: la acción de menor índice entre
        las que alcanzan el coste mínimo. Como los costes son positivos, todas esas
        acciones se disparan antes de extraer el hecho, así que la regla coincide con la de
        la implementación en PyTorch.
        """
        supporters = [-1] * self.num_facts
        return self._dijkstra(state, False, supporters), supporters

    def _dijkstra(
        self, state: State, combine_max: bool, supporters: list[int] | None
    ) -> list[float]:
        # Bucle crítico para el rendimiento: variables locales y sin llamadas a funciones
        # auxiliares dentro del bucle interior.
        cost: list[float] = [DEAD_END] * self.num_facts
        heap: list[tuple[float, int]] = []
        for f in self.true_facts(state):
            cost[f] = 0
            heap.append((0, f))
        heapq.heapify(heap)
        push, pop = heapq.heappush, heapq.heappop
        unsatisfied = self._pre_len.copy()
        accumulated = [0.0] * self.num_actions
        action_cost, adds, pre_of = self.cost, self.add, self.pre_of
        goal_facts = self.goal_facts
        track = supporters is not None
        support: list[int] = supporters if supporters is not None else []

        def relax(a: int, value: float) -> None:
            for q in adds[a]:
                current = cost[q]
                if value < current:
                    cost[q] = value
                    push(heap, (value, q))
                    if track:
                        support[q] = a
                elif track and value == current and a < support[q]:
                    support[q] = a

        for a in self.no_pre:
            relax(a, action_cost[a])
        pending_goals = sum(1 for g in goal_facts if cost[g] != 0)
        done = [False] * self.num_facts
        while heap and pending_goals:
            c, f = pop(heap)
            if done[f]:
                continue  # entrada obsoleta: el hecho ya se extrajo con su coste mínimo
            done[f] = True
            if c != 0 and f in goal_facts:
                pending_goals -= 1
            for a in pre_of[f]:
                remaining = unsatisfied[a] - 1
                unsatisfied[a] = remaining
                if combine_max:
                    if c > accumulated[a]:
                        accumulated[a] = c
                else:
                    accumulated[a] += c
                if remaining == 0:
                    value = action_cost[a] + accumulated[a]
                    for q in adds[a]:
                        current = cost[q]
                        if value < current:
                            cost[q] = value
                            push(heap, (value, q))
                            if track:
                                support[q] = a
                        elif track and value == current and a < support[q]:
                            support[q] = a
        return cost

    def clause_values(self, cost: Sequence[float], combine_max: bool) -> list[float]:
        """Valor h_max o h_add de cada cláusula del objetivo."""
        values: list[float] = []
        for clause in self.goal:
            costs = [cost[f] for f in clause]
            values.append((max(costs) if combine_max else sum(costs)) if costs else 0)
        return values

    def goal_value(self, cost: Sequence[float], combine_max: bool) -> float:
        """Valor h_max o h_add del objetivo: el mínimo entre sus cláusulas."""
        return min(self.clause_values(cost, combine_max), default=DEAD_END)

    def relaxed_plan(self, cost: Sequence[float], supporters: Sequence[int]) -> Evaluation:
        """Extrae el plan relajado de h_FF a partir de los costes h_add de los hechos.

        Se usa la cláusula del objetivo de menor valor (la de menor índice si hay empate).
        El mejor soportador de un hecho es la acción relajada de **menor índice** entre las
        que lo añaden con coste mínimo. La regla es determinista, y así las dos
        implementaciones dan el mismo plan relajado. Los ``supporters`` son los que
        calcula :meth:`fact_costs_and_supporters`.
        """
        values = self.clause_values(cost, False)
        if not values or min(values) == DEAD_END:
            return Evaluation(DEAD_END)
        clause = self.goal[values.index(min(values))]
        plan: set[int] = set()
        marked: set[int] = set()
        stack = [g for g in clause if cost[g] != 0]
        while stack:
            f = stack.pop()
            if f in marked:
                continue
            marked.add(f)
            best = supporters[f]
            if best not in plan:
                plan.add(best)
                stack.extend(p for p in self.pre[best] if cost[p] != 0)
        operators = {self.operator[a] for a in plan}
        value = sum(self.operator_cost[o] for o in operators)
        preferred = {self.operator[a] for a in plan if all(cost[p] == 0 for p in self.pre[a])}
        return Evaluation(value, tuple(sorted(preferred)))


class _RelaxationHeuristic(Heuristic):
    combine_max = False

    def __init__(self, task: GroundTask, relaxed: RelaxedTask | None = None) -> None:
        super().__init__(task)
        self.relaxed = relaxed or RelaxedTask(task)


class HMaxHeuristic(_RelaxationHeuristic):
    """h_max: admisible, poco informada."""

    name = "hmax"
    combine_max = True

    def evaluate_batch(self, states: Sequence[State]) -> list[Evaluation]:
        r = self.relaxed
        return [Evaluation(r.goal_value(r.fact_costs(s, True), True)) for s in states]


class HAddHeuristic(_RelaxationHeuristic):
    """h_add: no admisible, bien informada."""

    name = "hadd"

    def evaluate_batch(self, states: Sequence[State]) -> list[Evaluation]:
        r = self.relaxed
        return [Evaluation(r.goal_value(r.fact_costs(s, False), False)) for s in states]


class FFHeuristic(_RelaxationHeuristic):
    """h_FF: coste del plan relajado, con acciones útiles."""

    name = "ff"
    provides_preferred = True

    def evaluate_batch(self, states: Sequence[State]) -> list[Evaluation]:
        r = self.relaxed
        results = []
        for state in states:
            cost, supporters = r.fact_costs_and_supporters(state)
            results.append(r.relaxed_plan(cost, supporters))
        return results
