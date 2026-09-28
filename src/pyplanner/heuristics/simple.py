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
    """Número de literales del objetivo que todavía no se cumplen. No es admisible.

    Con un objetivo disyuntivo se toma la cláusula con menos literales pendientes.
    """

    name = "goalcount"

    def evaluate_batch(self, states: Sequence[State]) -> list[Evaluation]:
        clauses = [(c.pos, c.neg) for c in self.task.goal]
        return [
            Evaluation(
                min(
                    ((pos & ~s).bit_count() + (neg & s).bit_count() for pos, neg in clauses),
                    default=DEAD_END,
                )
            )
            for s in states
        ]
