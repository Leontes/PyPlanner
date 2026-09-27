"""Heurísticas y su creación a partir de un nombre y un dispositivo.

Dispositivos:

* ``"python"``: implementación de referencia en Python puro; siempre disponible.
* ``"cpu"`` o ``"cuda"``: implementación en PyTorch, evaluando por lotes. Solo está
  disponible si PyTorch está instalado (``pip install -e ".[gpu]"``) y solo para las
  heurísticas que se pueden expresar con tensores.
"""

from pyplanner.errors import PDDLError
from pyplanner.grounding import GroundTask
from pyplanner.heuristics.base import DEAD_END, Evaluation, Heuristic
from pyplanner.heuristics.relaxation import FFHeuristic, HAddHeuristic, HMaxHeuristic
from pyplanner.heuristics.simple import BlindHeuristic, GoalCountHeuristic

#: Heurísticas disponibles, por nombre.
HEURISTICS = ("blind", "goalcount", "hmax", "hadd", "ff")

#: Heurísticas con implementación en PyTorch.
TENSOR_HEURISTICS = ("hmax", "hadd", "ff")

#: Dispositivos de evaluación.
DEVICES = ("python", "cpu", "cuda")

#: Heurísticas que calculan acciones útiles.
PREFERRED_HEURISTICS = ("ff",)


class HeuristicError(PDDLError):
    """Heurística o dispositivo no válidos."""


def create_heuristic(name: str, task: GroundTask, device: str = "python") -> Heuristic:
    """Crea la heurística ``name`` para ``task`` en el dispositivo indicado.

    Raises:
        HeuristicError: si la heurística no existe, no tiene implementación para ese
            dispositivo o PyTorch no está instalado.
    """
    if name not in HEURISTICS:
        raise HeuristicError(
            f"heurística desconocida '{name}' (disponibles: {', '.join(HEURISTICS)})"
        )
    if device not in DEVICES:
        raise HeuristicError(
            f"dispositivo desconocido '{device}' (disponibles: {', '.join(DEVICES)})"
        )
    if device == "python":
        reference: dict[str, type[Heuristic]] = {
            "blind": BlindHeuristic,
            "goalcount": GoalCountHeuristic,
            "hmax": HMaxHeuristic,
            "hadd": HAddHeuristic,
            "ff": FFHeuristic,
        }
        return reference[name](task)
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


__all__ = [
    "DEAD_END",
    "DEVICES",
    "HEURISTICS",
    "PREFERRED_HEURISTICS",
    "TENSOR_HEURISTICS",
    "Evaluation",
    "Heuristic",
    "HeuristicError",
    "create_heuristic",
]
