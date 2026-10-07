"""Planes: representación, validación interna y salida en formato IPC o JSON.

El formato IPC (una acción por línea, ``(accion arg1 arg2)``) es el que leen VAL y el resto
de herramientas de la comunidad. En un plan temporal cada línea lleva además el tiempo de
inicio y la duración: ``12.010: (accion arg1 arg2) [20.000]``. El JSON sigue el esquema de
:mod:`pyplanner.schema` (``"kind": "plan"``) y es el que usará el proyecto en el que se
integre el planificador.
"""

import json
from dataclasses import dataclass
from typing import Any

from pyplanner.grounding import GroundTask
from pyplanner.grounding.discrete import DiscreteTask
from pyplanner.grounding.preferences import final_state
from pyplanner.grounding.temporal import TemporalTask
from pyplanner.schema import SCHEMA_VERSION


@dataclass(frozen=True)
class PlanStep:
    """Una acción del plan: nombre del esquema y argumentos.

    En un plan temporal lleva también su tiempo de inicio y su duración (``None`` en las
    acciones instantáneas).
    """

    name: str
    args: tuple[str, ...]
    time: float | None = None
    duration: float | None = None

    def __str__(self) -> str:
        return f"({' '.join([self.name, *self.args])})"

    def to_ipc(self) -> str:
        """La línea del plan en formato IPC."""
        if self.time is None:
            return str(self)
        line = f"{self.time:.3f}: {self}"
        return line if self.duration is None else f"{line} [{self.duration:.3f}]"


@dataclass(frozen=True)
class Plan:
    """Un plan para una tarea: secuencial, o temporal si tiene ``makespan``."""

    domain: str
    problem: str
    steps: tuple[PlanStep, ...]
    cost: float
    #: Valor de la métrica del problema (``None`` si no la hay o los costes no la representan).
    metric: float | None = None
    #: ``True`` si el coste es el número de acciones (sin métrica que sirva de coste).
    unit_cost: bool = True
    #: Duración total de un plan temporal (``None`` en uno secuencial).
    makespan: float | None = None
    #: Instancias de preferencias del objetivo y de las restricciones que incumple
    #: (``nombre arg1 arg2``).
    violated: tuple[str, ...] = ()

    @property
    def is_temporal(self) -> bool:
        return self.makespan is not None

    @classmethod
    def from_indices(cls, task: GroundTask, indices: tuple[int, ...]) -> "Plan":
        """Construye el plan a partir de índices de acciones de ``task``.

        En una tarea temporal, los índices son pasos (inicios, fines y acciones
        instantáneas) y el plan se construye con su planificación en el tiempo. En una con
        procesos y eventos (PDDL+), además esperas, que no salen en el plan.
        """
        if isinstance(task, TemporalTask | DiscreteTask):
            scheduled, makespan = task.schedule(indices)
            makespan = _clean(makespan)
            return cls(
                domain=task.domain_name,
                problem=task.problem_name,
                steps=tuple(
                    PlanStep(
                        s.name,
                        s.args,
                        _clean(s.time),
                        None if s.duration is None else _clean(s.duration),
                    )
                    for s in scheduled
                ),
                cost=len(scheduled),
                metric=makespan if task.metric_is_makespan else None,
                makespan=makespan,
            )
        preferences = task.preferences
        if preferences is None:
            actions = [task.actions[i] for i in indices]
            cost = sum(a.cost for a in actions)
            metric = cost
            violated: list[str] = []
        else:
            # Sin la fase final de las preferencias; el coste es la contribución de las
            # acciones a la métrica, y la métrica suma además las preferencias incumplidas.
            actions = [task.actions[i] for i in preferences.plan_actions(indices)]
            cost = _clean(sum(a.cost - preferences.step_cost for a in actions))
            final = final_state(task, indices)
            metric = _clean(preferences.metric(task, indices, final))
            violated = preferences.violated(task, indices, final)
        offset = task.metric_offset
        return cls(
            domain=task.domain_name,
            problem=task.problem_name,
            steps=tuple(PlanStep(a.name, a.args) for a in actions),
            cost=cost,
            metric=offset + metric if task.metric_is_cost and offset is not None else None,
            unit_cost=not task.metric_is_cost,
            violated=tuple(violated),
        )

    def __len__(self) -> int:
        return len(self.steps)

    def to_ipc(self) -> str:
        """El plan en formato IPC, con el coste (y la métrica) en comentarios finales. En un
        plan temporal, el comentario da la duración total en lugar del coste."""
        lines = [step.to_ipc() for step in self.steps]
        if self.makespan is not None:
            lines.append(f"; makespan = {_number(self.makespan)}")
        else:
            unit = " (unit cost)" if self.unit_cost else ""
            lines.append(f"; cost = {_number(self.cost)}{unit}")
        if self.metric is not None:
            lines.append(f"; metric = {_number(self.metric)}")
        if self.violated:
            lines.append(f"; violated preferences: {', '.join(self.violated)}")
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
        if self.makespan is not None:
            document["makespan"] = _number(self.makespan)
        if self.violated:
            document["violated"] = list(self.violated)
        steps = []
        for step in self.steps:
            node: dict[str, Any] = {"name": step.name, "args": list(step.args)}
            if step.time is not None:
                node["time"] = _number(step.time)
            if step.duration is not None:
                node["duration"] = _number(step.duration)
            steps.append(node)
        document["steps"] = steps
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


def _clean(value: float) -> float:
    """Quita el ruido de coma flotante de los tiempos (sumas de ε)."""
    return round(value, 6)


__all__ = ["Plan", "PlanStep", "simulate"]
