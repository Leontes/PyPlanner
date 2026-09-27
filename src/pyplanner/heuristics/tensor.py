"""Implementación en PyTorch de h_max, h_add y h_FF, con evaluación por lotes en CPU o GPU.

Este módulo solo se importa si se pide un dispositivo de PyTorch (``"cpu"`` o ``"cuda"``);
el resto del planificador no depende de PyTorch.

La relajación se calcula como un **punto fijo** sobre todo el lote a la vez. En cada
iteración:

1. coste de cada acción = coste(a) + suma (h_add) o máximo (h_max) de los costes de sus
   precondiciones;
2. coste de cada hecho = mínimo entre su coste actual y el de las acciones que lo añaden.

Las relaciones acción-hecho se guardan como **listas de aristas** (pares de índices), no
como matrices densas, y se procesan con ``index_add_`` y ``scatter_reduce``. Así la memoria
crece con el número de relaciones que existen y no con acciones × hechos.

El punto fijo es el mismo que calcula el Dijkstra de la implementación de referencia, así
que los valores coinciden. Los costes son enteros pequeños y se representan sin error en
``float32``.

h_FF también se extrae con tensores, para todo el lote a la vez
(:meth:`TensorRelaxation.relaxed_plans`):

1. **Mejor soportador** de cada hecho: entre las acciones que lo añaden con coste igual al
   del hecho, la de menor índice. Es la misma regla que la referencia, así que el plan
   relajado es idéntico.
2. **Extracción hacia atrás por capas:** empezando por los objetivos pendientes, se
   seleccionan sus soportadores. Sus precondiciones que no son ciertas en el estado pasan
   a ser hechos necesarios, y así hasta que no aparece ninguno nuevo. Como cada soportador
   tiene precondiciones de coste estrictamente menor, el número de capas está acotado por
   la profundidad del plan relajado.
3. **Valor y acciones útiles:** el valor es la suma de costes de las acciones seleccionadas;
   las acciones útiles son las seleccionadas con todas sus precondiciones ciertas.
"""

from collections.abc import Sequence

import torch

from pyplanner.errors import PDDLError
from pyplanner.grounding import GroundTask, State
from pyplanner.heuristics.base import DEAD_END, Evaluation, Heuristic
from pyplanner.heuristics.relaxation import RelaxedTask


class DeviceError(PDDLError):
    """El dispositivo pedido no existe o PyTorch no puede usarlo."""


def resolve_device(device: str) -> torch.device:
    """Convierte ``"cpu"`` o ``"cuda"`` en un ``torch.device``, comprobando que existe."""
    if device == "cuda" and not torch.cuda.is_available():
        raise DeviceError("se ha pedido el dispositivo 'cuda', pero PyTorch no detecta ninguna GPU")
    if device not in ("cpu", "cuda"):
        raise DeviceError(f"dispositivo de PyTorch desconocido: '{device}'")
    return torch.device(device)


class TensorRelaxation:
    """La relajación sin borrados de una tarea, compilada a tensores en un dispositivo."""

    def __init__(self, relaxed: RelaxedTask, device: torch.device) -> None:
        self.relaxed = relaxed
        self.device = device
        self.num_facts = relaxed.num_facts
        self.num_bytes = (relaxed.num_facts + 7) // 8

        def edges(pairs: list[tuple[int, int]]) -> tuple[torch.Tensor, torch.Tensor]:
            tensor = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2)
            return tensor[:, 0].to(device), tensor[:, 1].to(device)

        self.pre_action, self.pre_fact = edges(
            [(a, f) for a, facts in enumerate(relaxed.pre) for f in facts]
        )
        self.add_action, self.add_fact = edges(
            [(a, f) for a, facts in enumerate(relaxed.add) for f in facts]
        )
        self.cost = torch.tensor(relaxed.cost, dtype=torch.float32, device=device)
        self.goal = torch.tensor(relaxed.goal, dtype=torch.long, device=device)
        self._bit_shifts = torch.arange(8, dtype=torch.uint8, device=device)

    def states_to_tensor(self, states: Sequence[State]) -> torch.Tensor:
        """Convierte un lote de bitsets en una matriz booleana lote × hechos."""
        if self.num_bytes == 0:
            return torch.zeros(len(states), 0, dtype=torch.bool, device=self.device)
        raw = b"".join(s.to_bytes(self.num_bytes, "little") for s in states)
        packed = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(self.device)
        packed = packed.view(len(states), self.num_bytes)
        unpacked = (packed.unsqueeze(-1) >> self._bit_shifts) & 1
        return unpacked.view(len(states), -1)[:, : self.num_facts].bool()

    def fact_costs(self, states: Sequence[State], combine_max: bool) -> torch.Tensor:
        """Matriz lote × hechos con los costes h_max o h_add (``inf`` si inalcanzable)."""
        batch = len(states)
        inf = torch.tensor(float("inf"), device=self.device)
        h = torch.where(self.states_to_tensor(states), torch.zeros((), device=self.device), inf)
        pre_index = self.pre_action.expand(batch, -1)
        add_index = self.add_fact.expand(batch, -1)
        num_actions = self.cost.shape[0]
        for _ in range(self.num_facts + 1):
            pre_values = h[:, self.pre_fact]
            if combine_max:
                accumulated = torch.zeros(batch, num_actions, device=self.device)
                accumulated = accumulated.scatter_reduce(
                    1, pre_index, pre_values, reduce="amax", include_self=True
                )
            else:
                accumulated = torch.zeros(batch, num_actions, device=self.device)
                accumulated.index_add_(1, self.pre_action, pre_values)
            action_cost = self.cost + accumulated
            updated = h.scatter_reduce(
                1, add_index, action_cost[:, self.add_action], reduce="amin", include_self=True
            )
            if torch.equal(updated, h):
                break
            h = updated
        return h

    def goal_values(self, costs: torch.Tensor, combine_max: bool) -> list[float]:
        """Valor h_max o h_add del objetivo para cada estado del lote."""
        if self.goal.numel() == 0:
            return [0.0] * costs.shape[0]
        goal_costs = costs[:, self.goal]
        values = goal_costs.amax(dim=1) if combine_max else goal_costs.sum(dim=1)
        return [DEAD_END if v == float("inf") else v for v in values.tolist()]

    def relaxed_plans(self, states: Sequence[State]) -> list[Evaluation]:
        """h_FF (valor y acciones útiles) de cada estado del lote, extraído con tensores."""
        batch = len(states)
        num_actions = self.cost.shape[0]
        device = self.device
        h = self.fact_costs(states, combine_max=False)
        pending = h > 0  # hechos que no son ciertos en el estado

        # Coste de cada acción con los costes definitivos de los hechos.
        action_cost = self.cost.expand(batch, -1).clone()
        action_cost.index_add_(1, self.pre_action, h[:, self.pre_fact])

        # Mejor soportador: la acción de menor índice que alcanza el coste del hecho. El
        # índice num_actions hace de "ninguno" (columna extra que luego se descarta).
        achieves = action_cost[:, self.add_action] == h[:, self.add_fact]
        candidates = torch.where(achieves, self.add_action, num_actions)
        best = torch.full((batch, self.num_facts), num_actions, dtype=torch.long, device=device)
        best = best.scatter_reduce(
            1, self.add_fact.expand(batch, -1), candidates, reduce="amin", include_self=True
        )

        # Estados en los que algún objetivo es inalcanzable: callejones sin salida.
        dead = torch.zeros(batch, dtype=torch.bool, device=device)
        frontier = torch.zeros(batch, self.num_facts, dtype=torch.bool, device=device)
        if self.goal.numel():
            goal_costs = h[:, self.goal]
            dead = torch.isinf(goal_costs).any(dim=1)
            frontier[:, self.goal] = goal_costs > 0
            frontier &= ~dead.unsqueeze(1)
        needed = frontier.clone()
        selected = torch.zeros(batch, num_actions + 1, dtype=torch.bool, device=device)
        pre_pending = pending[:, self.pre_fact]

        while frontier.any():
            # Soportadores de los hechos de la frontera.
            chosen = torch.zeros(batch, num_actions + 1, dtype=torch.uint8, device=device)
            chosen = chosen.scatter_reduce(
                1, best, frontier.to(torch.uint8), reduce="amax", include_self=True
            ).bool()
            new_actions = chosen & ~selected
            selected |= new_actions
            # Sus precondiciones pendientes son los nuevos hechos necesarios.
            required = new_actions[:, self.pre_action] & pre_pending
            reached = torch.zeros(batch, self.num_facts, dtype=torch.uint8, device=device)
            reached = reached.scatter_reduce(
                1, self.pre_fact.expand(batch, -1), required.to(torch.uint8), "amax"
            ).bool()
            frontier = reached & ~needed
            needed |= frontier

        plan = selected[:, :num_actions]
        values = (plan * self.cost).sum(dim=1)
        unsatisfied = torch.zeros(batch, num_actions, device=device)
        unsatisfied.index_add_(1, self.pre_action, pre_pending.float())
        preferred = torch.nonzero(plan & (unsatisfied == 0)).tolist()

        per_state: list[list[int]] = [[] for _ in range(batch)]
        for row, action in preferred:
            per_state[row].append(action)
        return [
            Evaluation(DEAD_END) if is_dead else Evaluation(value, tuple(actions))
            for is_dead, value, actions in zip(
                dead.tolist(), values.tolist(), per_state, strict=True
            )
        ]


class _TensorHeuristic(Heuristic):
    combine_max = False

    def __init__(self, task: GroundTask, device: str, relaxed: RelaxedTask | None = None) -> None:
        super().__init__(task)
        self.device = device
        self.tensors = TensorRelaxation(relaxed or RelaxedTask(task), resolve_device(device))


class TensorHMaxHeuristic(_TensorHeuristic):
    """h_max evaluada por lotes con PyTorch."""

    name = "hmax"
    combine_max = True

    def evaluate_batch(self, states: Sequence[State]) -> list[Evaluation]:
        costs = self.tensors.fact_costs(states, True)
        return [Evaluation(v) for v in self.tensors.goal_values(costs, True)]


class TensorHAddHeuristic(_TensorHeuristic):
    """h_add evaluada por lotes con PyTorch."""

    name = "hadd"

    def evaluate_batch(self, states: Sequence[State]) -> list[Evaluation]:
        costs = self.tensors.fact_costs(states, False)
        return [Evaluation(v) for v in self.tensors.goal_values(costs, False)]


class TensorFFHeuristic(_TensorHeuristic):
    """h_FF por lotes con PyTorch: costes h_add y extracción del plan relajado en tensores."""

    name = "ff"
    provides_preferred = True

    def evaluate_batch(self, states: Sequence[State]) -> list[Evaluation]:
        return self.tensors.relaxed_plans(states)
