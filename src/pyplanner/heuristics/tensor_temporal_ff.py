"""h_TFF en PyTorch, para todo el lote a la vez (CPU o GPU).

Calcula lo mismo que la referencia (:mod:`pyplanner.heuristics.temporal_ff`), con puntos
fijos en lugar de un Dijkstra:

1. **Instantes.** Punto fijo sobre el lote: cada acción relajada ocurre en el máximo de sus
   precondiciones (con su peso) y de su instante mínimo, retrasada hasta caber en sus
   ventanas; cada hecho toma el mínimo de lo que le ofrecen sus acciones. Ajustar a las
   ventanas es otro punto fijo pequeño: con los intervalos de cada estado en un tensor
   relleno (lote × nodos con ventanas × intervalos), el primer intervalo en el que cabe es
   el de menor instante entre los que caben, porque están ordenados y no se solapan. Las
   operaciones son las mismas que en la referencia (sumas, máximos y comparaciones), así
   que los instantes son exactamente los mismos.
2. **Niveles.** Con los instantes definitivos, una arista es *ajustada* si da a su hecho
   justo su instante. El nivel de un hecho es 0 si es cierto en el estado y conserva su
   instante; si no, el mínimo, entre sus aristas ajustadas, de 1 + el máximo nivel de las
   precondiciones de la acción. Es un segundo punto fijo. No vale el truco de h_FF en
   PyTorch (la última iteración en que bajó el coste): con máximos, una precondición con
   holgura puede alcanzarse por un camino más corto que no es el más temprano.
3. **Soportador:** la acción de menor índice entre las aristas ajustadas con el nivel del
   hecho.
4. **Cláusula, plan relajado, valor y acciones útiles** como en h_FF en PyTorch: la
   cláusula que antes se cumple (la primera si hay empate) y la extracción por capas hacia
   atrás.
"""

from collections.abc import Sequence

import numpy as np
import torch

from pyplanner.grounding import State
from pyplanner.heuristics.base import DEAD_END, Evaluation
from pyplanner.heuristics.temporal_ff import TemporalRelaxedPlan
from pyplanner.heuristics.tensor import CHUNK_ELEMENTS, DTYPE, resolve_device

INF = float("inf")


class TensorTemporalRelaxedPlan:
    """El plan relajado temporal de una tarea, compilado a tensores en un dispositivo."""

    def __init__(self, plan: TemporalRelaxedPlan, device: str) -> None:
        self.plan = plan
        self.device = dev = resolve_device(device)
        d = plan.deadlines
        relaxed = d.relaxed
        self.num_facts = relaxed.num_facts
        self.num_actions = relaxed.num_actions
        self.num_operators = plan.num_operators
        self.num_clauses = len(d.goal_counted)
        self.num_windows = len(d.windowed)

        def longs(values: Sequence[int]) -> torch.Tensor:
            return torch.tensor(list(values), dtype=torch.long, device=dev)

        def floats(values: Sequence[float]) -> torch.Tensor:
            return torch.tensor(list(values), dtype=DTYPE, device=dev)

        pre = [(a, f, w) for f, items in enumerate(d.pre_of) for a, w in items]
        self.pre_action = longs([a for a, _, _ in pre])
        self.pre_fact = longs([f for _, f, _ in pre])
        self.pre_weight = floats([w for _, _, w in pre])
        effects = [(a, q, e) for a, items in enumerate(d.effects) for q, e in items]
        self.eff_action = longs([a for a, _, _ in effects])
        self.eff_fact = longs([q for _, q, _ in effects])
        edge = [e for _, _, e in effects]
        self.eff_numeric = longs([i for i, e in enumerate(edge) if e >= 0])
        self.eff_edge = longs([e for e in edge if e >= 0])
        self.release = floats(d.release)
        self.span = floats(d.span)
        windows = [(a, w) for a, nodes in enumerate(d.window_pre) for w in nodes]
        self.window_action = longs([a for a, _ in windows])
        self.window_node = longs([w for _, w in windows])
        spans = [(a, w) for a, nodes in enumerate(d.span_pre) for w in nodes]
        self.span_action = longs([a for a, _ in spans])
        self.span_node = longs([w for _, w in spans])
        goals = [(c, f) for c, facts in enumerate(d.goal_counted) for f in facts]
        self.goal_clause = longs([c for c, _ in goals])
        self.goal_fact = longs([f for _, f in goals])
        goal_windows = [(c, w) for c, nodes in enumerate(d.goal_window) for w in nodes]
        self.goal_window_clause = longs([c for c, _ in goal_windows])
        self.goal_window_node = longs([w for _, w in goal_windows])
        self.operator = longs(plan.operator)
        self.operator_cost = floats(plan.operator_cost)
        relations = max(
            1, len(pre) + len(effects) + self.num_facts + (len(windows) + len(spans)) * 4
        )
        self.chunk_size = max(16, CHUNK_ELEMENTS // relations)

    def evaluate(self, states: Sequence[State]) -> list[Evaluation]:
        """h_TFF de cada estado, por trozos de como mucho ``chunk_size`` estados."""
        results: list[Evaluation] = []
        for start in range(0, len(states), self.chunk_size):
            results.extend(self._evaluate(states[start : start + self.chunk_size]))
        return results

    # --- Entradas y ventanas ----------------------------------------------------------

    def _windows(
        self, pointers: np.ndarray, starts: np.ndarray, ends: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Intervalos de cada nodo con ventanas en tensores rellenos (lote × nodos ×
        intervalos): los que sobran empiezan en ``inf`` y nunca valen."""
        counts = pointers[:, 1:] - pointers[:, :-1]
        width = max(1, int(counts.max(initial=0)))
        offsets = np.arange(width)
        index = pointers[:, :-1, None] + offsets
        valid = offsets < counts[..., None]
        index = np.where(valid, index, 0)
        if starts.size == 0:
            starts, ends = np.zeros(1), np.zeros(1)
        window_start = np.where(valid, starts[index], INF)
        window_end = np.where(valid, ends[index], -INF)
        return (
            torch.from_numpy(window_start).to(self.device),
            torch.from_numpy(window_end).to(self.device),
        )

    def _fit(
        self,
        moment: torch.Tensor,
        owner: torch.Tensor,
        node: torch.Tensor,
        span: torch.Tensor | None,
        window_start: torch.Tensor,
        window_end: torch.Tensor,
    ) -> torch.Tensor:
        """Como ``fit`` de la referencia para cada fila de ``moment`` (lote × dueños), con
        los nodos de ``node`` de cada dueño ``owner``: el menor instante ≥ el dado en el que
        cada nodo tiene un intervalo con sitio para ``span`` (``None``: 0)."""
        if not owner.numel():
            return moment
        batch = moment.shape[0]
        starts = window_start[:, node, :]
        ends = window_end[:, node, :]
        extra = 0.0 if span is None else span[owner][None, :, None]
        index = owner.expand(batch, -1)
        while True:
            candidate = torch.maximum(moment[:, owner][..., None], starts)
            fits = candidate + extra <= ends
            first = torch.where(fits, candidate, INF).amin(dim=2)
            updated = moment.scatter_reduce(1, index, first, reduce="amax", include_self=True)
            if torch.equal(updated, moment):
                return moment
            moment = updated

    # --- Cálculo ----------------------------------------------------------------------

    def _moments(
        self,
        time: torch.Tensor,
        window_start: torch.Tensor,
        window_end: torch.Tensor,
    ) -> torch.Tensor:
        """Instante de cada acción relajada (lote × acciones) con los instantes ``time``."""
        batch = time.shape[0]
        accumulated = self.release.expand(batch, -1).clone()
        if self.pre_action.numel():
            accumulated = accumulated.scatter_reduce(
                1,
                self.pre_action.expand(batch, -1),
                time[:, self.pre_fact] + self.pre_weight,
                reduce="amax",
                include_self=True,
            )
        moment = self._fit(
            accumulated, self.window_action, self.window_node, None, window_start, window_end
        )
        return self._fit(
            moment, self.span_action, self.span_node, self.span, window_start, window_end
        )

    def _offers(self, moment: torch.Tensor, usable: torch.Tensor) -> torch.Tensor:
        """Instante que ofrece cada arista de efecto (lote × aristas; ``inf`` si la arista
        numérica no sirve en el estado)."""
        offers = moment[:, self.eff_action]
        if self.eff_numeric.numel():
            blocked = ~usable[:, self.eff_edge]
            offers[:, self.eff_numeric] = torch.where(blocked, INF, offers[:, self.eff_numeric])
        return offers

    def _evaluate(self, states: Sequence[State]) -> list[Evaluation]:
        batch = len(states)
        if self.num_clauses == 0:
            return [Evaluation(DEAD_END) for _ in states]
        device = self.device
        times, pointers, starts, ends, usable_np, multipliers_np, window_true_np = (
            self.plan.deadlines.plan_inputs(states)
        )
        initial = torch.from_numpy(times).to(device)
        usable = torch.from_numpy(usable_np).to(device)
        multipliers = torch.from_numpy(multipliers_np).to(device)
        window_true = torch.from_numpy(window_true_np).to(device)
        window_start, window_end = self._windows(pointers, starts, ends)
        fact_index = self.eff_fact.expand(batch, -1)

        # 1. Instantes.
        time = initial.clone()
        while True:
            offers = self._offers(self._moments(time, window_start, window_end), usable)
            updated = time.scatter_reduce(1, fact_index, offers, reduce="amin", include_self=True)
            if torch.equal(updated, time):
                break
            time = updated
        moment = self._moments(time, window_start, window_end)
        offers = self._offers(moment, usable)

        # 2. Niveles, sobre las aristas ajustadas.
        never = float(self.num_facts + 1)
        tight = (offers == time[:, self.eff_fact]) & (offers < INF)
        level = torch.where((initial == time) & (initial < INF), 0.0, never).to(DTYPE)
        pre_index = self.pre_action.expand(batch, -1)
        while True:
            action_level = torch.zeros(batch, self.num_actions, dtype=DTYPE, device=device)
            if self.pre_action.numel():
                action_level = action_level.scatter_reduce(
                    1, pre_index, level[:, self.pre_fact], reduce="amax", include_self=True
                )
            offered = torch.where(tight, action_level[:, self.eff_action] + 1, never)
            updated = level.scatter_reduce(1, fact_index, offered, reduce="amin", include_self=True)
            if torch.equal(updated, level):
                break
            level = updated

        # 3. Soportador: la acción de menor índice con la clave del hecho (num_actions hace de
        # "ninguno": columna extra que luego se descarta).
        num_actions = self.num_actions
        supporting = tight & (action_level[:, self.eff_action] + 1 == level[:, self.eff_fact])
        candidates = torch.where(supporting, self.eff_action, num_actions)
        best = torch.full((batch, self.num_facts), num_actions, dtype=torch.long, device=device)
        best = best.scatter_reduce(1, fact_index, candidates, reduce="amin")
        repeats_of = torch.ones(batch, self.num_facts, dtype=DTYPE, device=device)
        if self.eff_numeric.numel():
            chosen = supporting[:, self.eff_numeric] & (
                self.eff_action[self.eff_numeric] == best[:, self.eff_fact[self.eff_numeric]]
            )
            repeats_of = repeats_of.scatter_reduce(
                1,
                self.eff_fact[self.eff_numeric].expand(batch, -1),
                torch.where(chosen, multipliers[:, self.eff_edge], 0.0),
                reduce="amax",
                include_self=False,
            )

        # 4. Cláusula que antes se cumple (la primera si hay empate).
        clause_time = torch.zeros(batch, self.num_clauses, dtype=DTYPE, device=device)
        if self.goal_clause.numel():
            clause_time = clause_time.scatter_reduce(
                1,
                self.goal_clause.expand(batch, -1),
                time[:, self.goal_fact],
                reduce="amax",
                include_self=True,
            )
        clause_time = self._fit(
            clause_time, self.goal_window_clause, self.goal_window_node, None,
            window_start, window_end,
        )  # fmt: skip
        chosen_clause = clause_time.argmin(dim=1)
        dead = torch.isinf(clause_time.gather(1, chosen_clause.unsqueeze(1)).squeeze(1))

        # 5. Plan relajado, por capas hacia atrás.
        pending = initial == INF
        in_clause = (self.goal_clause.unsqueeze(0) == chosen_clause.unsqueeze(1)) & ~dead.unsqueeze(
            1
        )
        start = in_clause & pending[:, self.goal_fact]
        frontier = torch.zeros(batch, self.num_facts, dtype=torch.uint8, device=device)
        if self.goal_fact.numel():
            frontier = frontier.scatter_reduce(
                1, self.goal_fact.expand(batch, -1), start.to(torch.uint8), "amax"
            )
        frontier = frontier.bool()
        needed = frontier.clone()
        selected = torch.zeros(batch, num_actions + 1, dtype=torch.bool, device=device)
        pre_pending = pending[:, self.pre_fact]
        while frontier.any():
            chosen_actions = torch.zeros(batch, num_actions + 1, dtype=torch.uint8, device=device)
            chosen_actions = chosen_actions.scatter_reduce(
                1, best, frontier.to(torch.uint8), reduce="amax", include_self=True
            ).bool()
            new_actions = chosen_actions & ~selected
            selected |= new_actions
            required = new_actions[:, self.pre_action] & pre_pending
            reached = torch.zeros(batch, self.num_facts, dtype=torch.uint8, device=device)
            if self.pre_fact.numel():
                reached = reached.scatter_reduce(
                    1, self.pre_fact.expand(batch, -1), required.to(torch.uint8), "amax"
                )
            frontier = reached.bool() & ~needed
            needed |= frontier

        repeats = torch.zeros(batch, num_actions + 1, dtype=DTYPE, device=device)
        repeats = repeats.scatter_reduce(
            1, best, torch.where(needed, repeats_of, 0.0), reduce="amax"
        )[:, :num_actions]
        per_operator = torch.zeros(batch, self.num_operators, dtype=DTYPE, device=device)
        if self.num_operators and num_actions:
            per_operator = per_operator.scatter_reduce(
                1, self.operator.expand(batch, -1), repeats, reduce="amax"
            )
        values = (per_operator * self.operator_cost).sum(dim=1)

        # Acciones útiles: las del plan con sus precondiciones ciertas, también las de los
        # hechos con ventanas.
        plan = selected[:, :num_actions]
        missing = torch.zeros(batch, num_actions, dtype=DTYPE, device=device)
        if self.pre_action.numel():
            missing.index_add_(1, self.pre_action, pre_pending.to(DTYPE))
        if self.window_action.numel():
            missing.index_add_(1, self.window_action, (~window_true[:, self.window_node]).to(DTYPE))
        ready = plan & (missing == 0)
        useful = torch.zeros(batch, self.num_operators, dtype=torch.uint8, device=device)
        if self.operator.numel():
            useful = useful.scatter_reduce(
                1, self.operator.expand(batch, -1), ready.to(torch.uint8), "amax"
            )
        axiom = self.plan.axiom_operator
        per_state: list[list[int]] = [[] for _ in range(batch)]
        for row, operator in torch.nonzero(useful.bool()).tolist():
            if operator != axiom:
                per_state[row].append(operator)
        return [
            Evaluation(DEAD_END) if is_dead else Evaluation(value, tuple(actions))
            for is_dead, value, actions in zip(
                dead.tolist(), values.tolist(), per_state, strict=True
            )
        ]
