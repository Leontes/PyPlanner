"""Heurísticas y su creación a partir de un nombre y un dispositivo.

Dispositivos:

* ``"python"``: implementación de referencia en Python puro; siempre disponible.
* ``"numba"``: la misma implementación compilada con Numba (``pip install -e ".[fast]"``);
  unas 10 veces más rápida. Da exactamente los mismos resultados que la referencia.
* ``"cpu"`` o ``"cuda"``: implementación en PyTorch, evaluando por lotes. Solo está
  disponible si PyTorch está instalado (``pip install -e ".[gpu]"``); compensa con lotes
  grandes, sobre todo en GPU.
* ``"auto"`` (por defecto): ``"numba"`` si Numba está instalado y ``"python"`` si no.

Las heurísticas triviales (``blind``, ``goalcount``) usan siempre la implementación en
Python con ``"auto"`` y ``"numba"``, porque no hay nada que acelerar.
"""

from functools import cache

from pyplanner.errors import PDDLError
from pyplanner.grounding import GroundTask
from pyplanner.heuristics.base import DEAD_END, Evaluation, Heuristic
from pyplanner.heuristics.relaxation import FFHeuristic, HAddHeuristic, HMaxHeuristic
from pyplanner.heuristics.simple import BlindHeuristic, GoalCountHeuristic

#: Heurísticas disponibles, por nombre.
HEURISTICS = ("blind", "goalcount", "hmax", "hadd", "ff")

#: Heurísticas con implementación en PyTorch.
TENSOR_HEURISTICS = ("hmax", "hadd", "ff")

#: Heurísticas con implementación compilada con Numba.
COMPILED_HEURISTICS = ("hmax", "hadd", "ff")

#: Dispositivos de evaluación.
DEVICES = ("auto", "python", "numba", "cpu", "cuda", "hybrid")

#: Heurísticas que calculan acciones útiles.
PREFERRED_HEURISTICS = ("ff",)


class HeuristicError(PDDLError):
    """Heurística o dispositivo no válidos."""


@cache
def numba_available() -> bool:
    """``True`` si Numba está instalado y se puede importar."""
    try:
        import numba  # noqa: F401
    except ImportError:
        return False
    return True


def create_heuristic(name: str, task: GroundTask, device: str = "auto") -> Heuristic:
    """Crea la heurística ``name`` para ``task`` en el dispositivo indicado.

    En las tareas temporales con literales iniciales temporizados, la heurística va envuelta
    en :class:`~pyplanner.heuristics.deadlines.DeadlineHeuristic`, que descarta los estados
    que no pueden cumplir los plazos. En las de procesos y eventos con invariantes, en
    :class:`~pyplanner.heuristics.projection.ProjectionHeuristic`, que suma lo que una espera
    sin hacer nada no cubriría (salvo a ``blind``, que tiene que seguir siendo admisible).

    Raises:
        HeuristicError: si la heurística no existe, no tiene implementación para ese
            dispositivo o PyTorch no está instalado.
    """
    heuristic = _create(name, task, device)
    from pyplanner.grounding.discrete import DiscreteTask
    from pyplanner.grounding.temporal import TemporalTask

    if isinstance(task, TemporalTask) and task.til_times:
        from pyplanner.heuristics.deadlines import DeadlineHeuristic

        compiled = numba_available() and device != "python"
        return DeadlineHeuristic(task, heuristic, compiled=compiled)
    if isinstance(task, DiscreteTask) and name != "blind":
        from pyplanner.heuristics.projection import ProjectionHeuristic, needs_projection

        if needs_projection(task):
            return ProjectionHeuristic(task, heuristic)
    return heuristic


def _create(name: str, task: GroundTask, device: str) -> Heuristic:
    if name not in HEURISTICS:
        raise HeuristicError(
            f"heurística desconocida '{name}' (disponibles: {', '.join(HEURISTICS)})"
        )
    if device not in DEVICES:
        raise HeuristicError(
            f"dispositivo desconocido '{device}' (disponibles: {', '.join(DEVICES)})"
        )
    # Las tareas temporales con variables continuas tienen una versión propia para la
    # heurística (mismos hechos, variables e índices de acciones).
    task = task.relaxation
    if device == "auto":
        device = "numba" if numba_available() else "python"
    if device == "numba" and name not in COMPILED_HEURISTICS:
        device = "python"  # nada que acelerar
    if device == "hybrid":
        return _hybrid(name, task)
    if device == "python":
        reference: dict[str, type[Heuristic]] = {
            "blind": BlindHeuristic,
            "goalcount": GoalCountHeuristic,
            "hmax": HMaxHeuristic,
            "hadd": HAddHeuristic,
            "ff": FFHeuristic,
        }
        return reference[name](task)
    if device == "numba":
        if not numba_available():
            raise HeuristicError(
                "el dispositivo 'numba' necesita Numba; instálalo con pip install -e \".[fast]\""
            )
        from pyplanner.heuristics import compiled

        compiled_classes: dict[str, type[compiled._CompiledHeuristic]] = {
            "hmax": compiled.CompiledHMaxHeuristic,
            "hadd": compiled.CompiledHAddHeuristic,
            "ff": compiled.CompiledFFHeuristic,
        }
        return compiled_classes[name](task)
    if name not in TENSOR_HEURISTICS:
        raise HeuristicError(f"la heurística '{name}' no tiene implementación en PyTorch")
    try:
        from pyplanner.heuristics import tensor
    except ImportError as exc:
        raise HeuristicError(
            f"el dispositivo '{device}' necesita PyTorch; "
            'instálalo con pip install -e ".[gpu]" (ver README)'
        ) from exc
    classes: dict[str, type[tensor._TensorHeuristic]] = {
        "hmax": tensor.TensorHMaxHeuristic,
        "hadd": tensor.TensorHAddHeuristic,
        "ff": tensor.TensorFFHeuristic,
    }
    return classes[name](task, device)


def _hybrid(name: str, task: GroundTask) -> Heuristic:
    """Heurística híbrida: Numba para los lotes pequeños y PyTorch en GPU para los grandes
    (ver :mod:`pyplanner.heuristics.hybrid`). Sin GPU, todo va a Numba."""
    from pyplanner.heuristics.hybrid import HybridHeuristic

    if name not in TENSOR_HEURISTICS or not numba_available():
        raise HeuristicError(
            f"el dispositivo 'hybrid' necesita Numba y una heurística con implementación en "
            f"PyTorch, y '{name}' no la tiene"
            if numba_available()
            else "el dispositivo 'hybrid' necesita Numba"
        )
    fast = _create(name, task, "numba")
    wide = None
    try:
        import torch

        if torch.cuda.is_available():
            wide = _create(name, task, "cuda")
    except ImportError:
        pass
    return HybridHeuristic(task, fast, wide)


__all__ = [
    "COMPILED_HEURISTICS",
    "DEAD_END",
    "DEVICES",
    "HEURISTICS",
    "PREFERRED_HEURISTICS",
    "TENSOR_HEURISTICS",
    "Evaluation",
    "Heuristic",
    "HeuristicError",
    "create_heuristic",
    "numba_available",
]
