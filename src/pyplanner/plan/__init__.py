"""Planes: representación, validación interna y salida en formato IPC o JSON.

El formato IPC (una acción por línea, ``(accion arg1 arg2)``) es el que leen VAL y el resto
de herramientas de la comunidad. El JSON sigue el esquema de :mod:`pyplanner.schema`
(``"kind": "plan"``) y es el que usará el proyecto en el que se integre el planificador.
"""

import json
from dataclasses import dataclass
from typing import Any

from pyplanner.grounding import GroundTask
from pyplanner.schema import SCHEMA_VERSION


@dataclass(frozen=True)
class PlanStep:
    """Una acción del plan: nombre del esquema y argumentos."""

    name: str
    args: tuple[str, ...]

    def __str__(self) -> str:
        return f"({' '.join([self.name, *self.args])})"


@dataclass(frozen=True)
class Plan:
    """Un plan secuencial para una tarea."""

    domain: str
    problem: str
    steps: tuple[PlanStep, ...]
    cost: float
    #: Valor de la métrica del problema (``None`` si no la hay o los costes no la representan).
    metric: float | None = None
    #: ``True`` si el coste es el número de acciones (sin métrica que sirva de coste).
    unit_cost: bool = True

    @classmethod
    def from_indices(cls, task: GroundTask, indices: tuple[int, ...]) -> "Plan":
        """Construye el plan a partir de índices de acciones de ``task``."""
        actions = [task.actions[i] for i in indices]
        cost = sum(a.cost for a in actions)
        offset = task.metric_offset
        return cls(
            domain=task.domain_name,
            problem=task.problem_name,
            steps=tuple(PlanStep(a.name, a.args) for a in actions),
            cost=cost,
            metric=offset + cost if task.metric_is_cost and offset is not None else None,
            unit_cost=not task.metric_is_cost,
        )

    def __len__(self) -> int:
        return len(self.steps)

    def to_ipc(self) -> str:
        """El plan en formato IPC, con el coste (y la métrica) en comentarios finales."""
        lines = [str(step) for step in self.steps]
        lines.append(f"; cost = {_number(self.cost)}" + (" (unit cost)" if self.unit_cost else ""))
        if self.metric is not None:
            lines.append(f"; metric = {_number(self.metric)}")
        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict[str, Any]:
        """El plan como documento JSON (``"kind": "plan"``)."""
        document: dict[str, Any] = {
            "version": SCHEMA_VERSION,
            "kind": "plan",
            "domain": self.domain,
            "problem": self.problem,
            "cost": _number(self.cost),
        }
        if self.metric is not None:
            document["metric"] = _number(self.metric)
        document["steps"] = [{"name": s.name, "args": list(s.args)} for s in self.steps]
        return document

    def to_json(self, indent: int | None = 2) -> str:
        """El plan como texto JSON."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def simulate(task: GroundTask, indices: tuple[int, ...]) -> str | None:
    """Ejecuta el plan sobre la tarea y comprueba que alcanza el objetivo.

    Es una validación interna, más rápida que VAL pero sobre la tarea ya instanciada.

    Returns:
        ``None`` si el plan es válido, o una descripción del primer fallo.
    """
    state = task.init
    for step, index in enumerate(indices, start=1):
        action = task.actions[index]
        if not action.applicable(state):
            return f"paso {step}: {action} no es aplicable"
        successor = task.apply(state, action)
        if successor is None:
            return f"paso {step}: {action} tiene un efecto numérico indefinido"
        state = successor
    if not task.is_goal(state):
        return "el plan no alcanza el objetivo"
    return None


def _number(value: float) -> int | float:
    return int(value) if float(value).is_integer() else value


__all__ = ["Plan", "PlanStep", "simulate"]
