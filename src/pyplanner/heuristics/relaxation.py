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
El valor es el coste de las acciones del plan relajado, y las acciones útiles son las del
plan que son aplicables en el estado.

Todos los costes de acción son positivos. Por eso, cuando se han extraído todos los
objetivos, los costes que necesita la extracción del plan relajado ya son definitivos, y el
cálculo puede terminar antes.
"""

import heapq
from collections.abc import Sequence

from pyplanner.grounding import GroundTask, State, bits
from pyplanner.heuristics.base import DEAD_END, Evaluation, Heuristic


class RelaxedTask:
    """Estructuras de la relajación, precalculadas una vez por tarea.

    Las comparten la implementación de referencia y la de PyTorch.
    """

    def __init__(self, task: GroundTask) -> None:
        self.task = task
        self.num_facts = len(task.facts)
        self.num_actions = len(task.actions)
        self.pre = [a.pre_facts for a in task.actions]
        self.add = [a.add_facts for a in task.actions]
        self.cost = [a.cost for a in task.actions]
        self.goal = task.goal_facts
        self.goal_set = frozenset(task.goal_facts)
        self.pre_of: list[list[int]] = [[] for _ in range(self.num_facts)]
        self.achievers: list[list[int]] = [[] for _ in range(self.num_facts)]
        for a, action in enumerate(task.actions):
            for f in action.pre_facts:
                self.pre_of[f].append(a)
            for f in action.add_facts:
                self.achievers[f].append(a)
        self.no_pre = [a for a in range(self.num_actions) if not self.pre[a]]

    def fact_costs(self, state: State, combine_max: bool) -> list[float]:
        """Costes h_max (``combine_max``) o h_add de cada hecho desde ``state``.

        Se detiene cuando se han extraído todos los objetivos. Los hechos no extraídos
        pueden quedar con un coste provisional o infinito; la extracción del plan relajado
        no los necesita.
        """
        cost: list[float] = [DEAD_END] * self.num_facts
        heap: list[tuple[float, int]] = []
        for f in bits(state):
            cost[f] = 0
            heap.append((0, f))
        heapq.heapify(heap)
        unsatisfied = [len(p) for p in self.pre]
        accumulated = [0.0] * self.num_actions

        def trigger(a: int) -> None:
            value = self.cost[a] + accumulated[a]
            for q in self.add[a]:
                if value < cost[q]:
                    cost[q] = value
                    heapq.heappush(heap, (value, q))

        for a in self.no_pre:
            trigger(a)
        pending_goals = sum(1 for g in self.goal if cost[g] != 0)
        done = [False] * self.num_facts
        while heap and pending_goals:
            c, f = heapq.heappop(heap)
            if done[f] or c > cost[f]:
                continue
            done[f] = True
            if c != 0 and f in self.goal_set:
                pending_goals -= 1
            for a in self.pre_of[f]:
                unsatisfied[a] -= 1
                if combine_max:
                    accumulated[a] = max(accumulated[a], c)
                else:
                    accumulated[a] += c
                if unsatisfied[a] == 0:
                    trigger(a)
        return cost

    def goal_value(self, cost: Sequence[float], combine_max: bool) -> float:
        """Valor h_max o h_add del objetivo a partir de los costes de los hechos."""
        values = [cost[g] for g in self.goal]
        if not values:
            return 0
        return max(values) if combine_max else sum(values)

    def relaxed_plan(self, state: State, cost: Sequence[float]) -> Evaluation:
        """Extrae el plan relajado de h_FF a partir de los costes h_add de los hechos.

        El mejor soportador de un hecho es la acción de **menor índice** entre las que lo
        añaden con coste mínimo. La regla es determinista, y así las dos implementaciones
        dan el mismo plan relajado.
        """
        if any(cost[g] == DEAD_END for g in self.goal):
            return Evaluation(DEAD_END)
        plan: set[int] = set()
        marked: set[int] = set()
        stack = [g for g in self.goal if cost[g] != 0]
        while stack:
            f = stack.pop()
            if f in marked:
                continue
            marked.add(f)
            best = self._best_supporter(f, cost)
            if best not in plan:
                plan.add(best)
                stack.extend(p for p in self.pre[best] if cost[p] != 0)
        value = sum(self.cost[a] for a in plan)
        preferred = tuple(sorted(a for a in plan if all(cost[p] == 0 for p in self.pre[a])))
        return Evaluation(value, preferred)

    def _best_supporter(self, fact: int, cost: Sequence[float]) -> int:
        target = cost[fact]
        for a in self.achievers[fact]:  # ordenados por índice
            if self.cost[a] + sum(cost[p] for p in self.pre[a]) == target:
                return a
        raise AssertionError(f"el hecho {fact} no tiene soportador con coste {target}")


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
        return [r.relaxed_plan(s, r.fact_costs(s, False)) for s in states]
