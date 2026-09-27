"""Heurísticas triviales: ciega y número de objetivos pendientes."""

from collections.abc import Sequence

from pyplanner.grounding import State
from pyplanner.heuristics.base import Evaluation, Heuristic


class BlindHeuristic(Heuristic):
    """0 en los estados objetivo y el coste mínimo de una acción en el resto.

    Es admisible: con A* equivale a una búsqueda de coste uniforme.
    """

    name = "blind"

    def evaluate_batch(self, states: Sequence[State]) -> list[Evaluation]:
        min_cost = min((a.cost for a in self.task.actions), default=0)
        return [Evaluation(0 if self.task.is_goal(s) else min_cost) for s in states]


class GoalCountHeuristic(Heuristic):
    """Número de hechos del objetivo que todavía no se cumplen. No es admisible."""

    name = "goalcount"

    def evaluate_batch(self, states: Sequence[State]) -> list[Evaluation]:
        goal = self.task.goal
        return [Evaluation((goal & ~s).bit_count()) for s in states]
