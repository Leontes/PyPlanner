"""Interfaz común de las heurísticas.

Toda heurística evalúa **lotes** de estados (:meth:`Heuristic.evaluate_batch`). Evaluar un
solo estado es un lote de tamaño 1. Así la búsqueda llama siempre igual, y una
implementación que aproveche el paralelismo (PyTorch en CPU o GPU) se puede cambiar por la
de referencia sin tocar la búsqueda.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pyplanner.grounding import GroundTask, State

#: Valor heurístico de un estado sin solución (callejón sin salida).
DEAD_END = math.inf


@dataclass(frozen=True)
class Evaluation:
    """Resultado de evaluar un estado.

    Attributes:
        value: estimación del coste hasta el objetivo; :data:`DEAD_END` si no hay solución.
        preferred: índices de las acciones útiles (*preferred operators*) del estado, si la
            heurística las calcula.
    """

    value: float
    preferred: tuple[int, ...] = ()

    @property
    def is_dead_end(self) -> bool:
        """``True`` si la heurística ha demostrado que el estado no tiene solución."""
        return self.value == DEAD_END


class Heuristic(ABC):
    """Heurística sobre una tarea instanciada.

    Attributes:
        name: nombre corto (``"ff"``, ``"hadd"``…).
        provides_preferred: ``True`` si calcula acciones útiles.
        device: dónde se evalúa (``"python"``, ``"cpu"`` o ``"cuda"``).
    """

    name: str = ""
    provides_preferred: bool = False
    device: str = "python"

    def __init__(self, task: GroundTask) -> None:
        self.task = task

    @abstractmethod
    def evaluate_batch(self, states: Sequence[State]) -> list[Evaluation]:
        """Evalúa varios estados a la vez. El resultado sigue el orden de ``states``."""

    def evaluate(self, state: State) -> Evaluation:
        """Evalúa un solo estado (lote de tamaño 1)."""
        return self.evaluate_batch([state])[0]

    def helpful(self, state: State, preferred: frozenset[int]) -> frozenset[int]:
        """Las acciones útiles de un estado que se va a expandir: por defecto, las de su
        evaluación (``preferred``). Una heurística puede afinarlas aquí, donde solo se paga
        por los estados que se expanden (no por todos los que se evalúan)."""
        return preferred

    def defer_dead_end_check(self) -> Callable[[Sequence[State]], list[bool]] | None:
        """Aplaza la parte de la heurística que solo detecta callejones sin salida, si la
        tiene, y la devuelve para que la búsqueda la aplique al expandir.

        Las búsquedas inmediatas evalúan todos los sucesores, y la mayoría nunca se
        expanden: una comprobación cara que solo descarta estados sale más barata al sacar
        cada nodo de la cola. A partir de la llamada, :meth:`evaluate_batch` ya no la hace.
        Devuelve ``None`` si no hay nada que aplazar."""
        return None
