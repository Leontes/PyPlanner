"""Heurísticas triviales: ciega y número de objetivos pendientes."""

from collections.abc import Sequence

from pyplanner.grounding import State
from pyplanner.heuristics.base import DEAD_END, Evaluation, Heuristic


class BlindHeuristic(Heuristic):
    """0 en los estados objetivo y el coste mínimo de una acción en el resto.

    Es admisible: con A* equivale a una búsqueda de coste uniforme. Si el objetivo es
    imposible (no tiene cláusulas), todos los estados son callejones sin salida.
    """

    name = "blind"

    def evaluate_batch(self, states: Sequence[State]) -> list[Evaluation]:
        if not self.task.goal:
            return [Evaluation(DEAD_END) for _ in states]
        min_cost = min((a.cost for a in self.task.actions), default=0)
        return [Evaluation(0 if self.task.is_goal(s) else min_cost) for s in states]


class GoalCountHeuristic(Heuristic):
    """Número de literales del objetivo que todavía no se cumplen (incluidas las
    condiciones numéricas). No es admisible.

    Con un objetivo disyuntivo se toma la cláusula con menos literales pendientes.
    """

    name = "goalcount"

    def evaluate_batch(self, states: Sequence[State]) -> list[Evaluation]:
        goal = self.task.goal
        if len(goal) == 1 and not goal[0].numeric:
            # Caso habitual (un objetivo conjuntivo sin condiciones numéricas): sin
            # generadores, que costaban un 20% de la búsqueda con esta heurística.
            pos, neg = goal[0].pos, goal[0].neg
            return [
                Evaluation((pos & ~s.facts).bit_count() + (neg & s.facts).bit_count())
                for s in states
            ]
        clauses = [(c.pos, c.neg, c.numeric) for c in goal]
        results = []
        for state in states:
            facts, values = state.facts, state.values
            pending = (
                (pos & ~facts).bit_count()
                + (neg & facts).bit_count()
                + sum(not c.holds(values) for c in numeric)
                for pos, neg, numeric in clauses
            )
            results.append(Evaluation(min(pending, default=DEAD_END)))
        return results
