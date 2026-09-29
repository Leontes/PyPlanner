"""Implementación en PyTorch de h_max, h_add y h_FF, con evaluación por lotes en CPU o GPU.

Este módulo solo se importa si se pide un dispositivo de PyTorch (``"cpu"`` o ``"cuda"``);
el resto del planificador no depende de PyTorch.

Se usa la misma tarea relajada que la referencia
(:class:`~pyplanner.heuristics.relaxation.RelaxedTask`): acciones relajadas por efecto,
hechos complementarios, condiciones numéricas como nodos (*h^hbd*) y objetivo por
cláusulas. Todo se calcula en ``float64``.

**Aristas de efecto.** Cada acción relajada llega a sus hechos por aristas: las de sus
añadidos, con multiplicador 1, y las numéricas, con un multiplicador que depende del
estado (las repeticiones necesarias, ``inf`` si la arista no sirve). El valor que ofrece
una arista es ``multiplicador × coste(a) + acumulado(a)``, igual que en la referencia.

**Costes.** Son un **punto fijo** sobre todo el lote a la vez. En cada iteración:

1. acumulado de cada acción = suma (h_add) o máximo (h_max) de los costes de sus
   precondiciones;
2. coste de cada hecho = mínimo entre su coste actual y lo que le ofrecen sus aristas.

Las relaciones se guardan como listas de aristas (pares de índices) y se procesan con
``index_add_`` y ``scatter_reduce``, así que la memoria crece con el número de relaciones
y no con acciones × hechos. El punto fijo coincide con el Dijkstra de la referencia.

h_FF se extrae también con tensores, para todo el lote a la vez
(:meth:`TensorRelaxation.relaxed_plans`):

1. **Niveles.** El nivel de un hecho es la última iteración del punto fijo de costes en
   que bajó su coste (0 si es cierto): coincide con el nivel del desempate (coste, nivel)
   de la referencia (ver :meth:`TensorRelaxation._fact_costs`), sin un segundo punto fijo
   ni depender de un orden de extracción.
2. **Mejor soportador** de cada hecho: la acción de menor índice entre las que le dan su
   (coste, nivel). Es la regla de la referencia, así que el plan relajado es idéntico.
3. **Extracción hacia atrás por capas:** desde los objetivos pendientes se seleccionan sus
   soportadores; sus precondiciones de coste no nulo pasan a ser hechos necesarios, y así
   hasta que no aparece ninguno nuevo.
4. **Valor y acciones útiles:** cada operador suma su coste por el máximo número de
   repeticiones que necesitan sus acciones relajadas seleccionadas; las acciones útiles
   son los operadores de las seleccionadas con todas sus precondiciones ciertas.

Con costes enteros (el caso habitual) los resultados son exactamente los de la
referencia. Con costes no enteros, el orden de las sumas puede cambiar el último bit de
algún valor.
"""

from collections.abc import Iterable, Sequence

import torch

from pyplanner.errors import PDDLError
from pyplanner.grounding import GroundTask, State
from pyplanner.heuristics.base import DEAD_END, Evaluation, Heuristic
from pyplanner.heuristics.relaxation import ONCE, REPEAT_EQ, REPEAT_GE, REPEAT_GT, RelaxedTask

DTYPE = torch.float64
_COMPARATORS = {">=": 0, ">": 1, "=": 2}


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
    """La relajación de una tarea, compilada a tensores en un dispositivo."""

    def __init__(self, relaxed: RelaxedTask, device: torch.device) -> None:
        self.relaxed = relaxed
        self.device = device
        self.num_facts = relaxed.num_facts
        self.num_base_facts = relaxed.num_base
        self.num_bytes = (self.num_base_facts + 7) // 8
        self.num_operators = len(relaxed.task.actions)
        self.num_actions = relaxed.num_actions
        self.num_variables = len(relaxed.task.variables)

        def edges(pairs: list[tuple[int, int]]) -> tuple[torch.Tensor, torch.Tensor]:
            tensor = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2)
            return tensor[:, 0].to(device), tensor[:, 1].to(device)

        def longs(values: Iterable[int]) -> torch.Tensor:
            return torch.tensor(list(values), dtype=torch.long, device=device)

        def floats(values: Iterable[float]) -> torch.Tensor:
            return torch.tensor(list(values), dtype=DTYPE, device=device)

        self.pre_action, self.pre_fact = edges(
            [(a, f) for a, facts in enumerate(relaxed.pre) for f in facts]
        )
        self.goal_clause, self.goal_fact = edges(
            [(c, f) for c, facts in enumerate(relaxed.goal) for f in facts]
        )
        self.num_clauses = len(relaxed.goal)
        # Aristas de efecto: primero los añadidos y después las numéricas.
        adds = [(a, f) for a, facts in enumerate(relaxed.add) for f in facts]
        numeric_edges = [(e[0], e[1]) for e in relaxed.edges]
        self.add_action, self.add_fact = edges(adds)
        self.numeric_action, self.numeric_fact = edges(numeric_edges)
        self.edge_action, self.edge_fact = edges(adds + numeric_edges)
        self.numeric_kind = longs(e[2] for e in relaxed.edges)
        self.numeric_delta = floats(e[3] for e in relaxed.edges)
        self.numeric_node = longs(e[1] - relaxed.numeric_offset for e in relaxed.edges)
        self.cost = floats(relaxed.cost)
        self.operator = longs(relaxed.operator)
        self.operator_cost = floats(relaxed.operator_cost)
        self.negated = longs(relaxed.negated)
        self._bit_shifts = torch.arange(8, dtype=torch.uint8, device=device)

        # Condiciones numéricas lineales: términos en una matriz condiciones × términos,
        # rellena hasta el máximo; las no lineales se evalúan en Python.
        conditions = relaxed.numeric
        self.nonlinear = [j for j, c in enumerate(conditions) if c.linear is None]
        terms = [c.linear.terms if c.linear is not None else () for c in conditions]
        width = max((len(t) for t in terms), default=0)
        self.term_var = (
            torch.tensor(
                [[v for v, _ in t] + [0] * (width - len(t)) for t in terms], dtype=torch.long
            )
            .reshape(len(conditions), width)
            .to(device)
        )
        self.term_weight = (
            torch.tensor([[w for _, w in t] + [0.0] * (width - len(t)) for t in terms], dtype=DTYPE)
            .reshape(len(conditions), width)
            .to(device)
        )
        self.term_len = longs(len(t) for t in terms)
        self.constant = floats(c.linear.constant if c.linear else 0.0 for c in conditions)
        self.comparator = longs(_COMPARATORS[c.comparator] for c in conditions)

    # --- Entrada ----------------------------------------------------------------------

    def _node_values(self, states: Sequence[State]) -> torch.Tensor:
        """Matriz lote × condiciones numéricas con el valor de cada condición (NaN si es
        indefinido), sumando en el mismo orden que la referencia."""
        batch = len(states)
        conditions = self.relaxed.numeric
        values = (
            torch.tensor(
                [[float("nan") if v is None else v for v in s.values] for s in states], dtype=DTYPE
            )
            .reshape(batch, self.num_variables)
            .to(self.device)
        )
        total = self.constant.expand(batch, -1).clone()
        for k in range(self.term_var.shape[1]):
            term = self.term_weight[:, k] * values[:, self.term_var[:, k]]
            total = torch.where(k < self.term_len, total + term, total)
        for j in self.nonlinear:
            column = []
            for state in states:
                value = conditions[j].value(state.values)
                column.append(float("nan") if value is None else value)
            total[:, j] = torch.tensor(column, dtype=DTYPE, device=self.device)
        return total

    def _inputs(self, states: Sequence[State]) -> tuple[torch.Tensor, torch.Tensor]:
        """Hechos relajados ciertos (lote × hechos) y multiplicadores de las aristas
        numéricas (lote × aristas numéricas); los de los añadidos son siempre 1."""
        batch = len(states)
        device = self.device
        if self.num_bytes == 0:
            base = torch.zeros(batch, 0, dtype=torch.bool, device=device)
        else:
            raw = b"".join(s.facts.to_bytes(self.num_bytes, "little") for s in states)
            packed = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
            unpacked = (packed.view(batch, self.num_bytes).unsqueeze(-1) >> self._bit_shifts) & 1
            base = unpacked.view(batch, -1)[:, : self.num_base_facts].bool()
        parts = [base, ~base[:, self.negated]]
        multipliers = torch.ones(batch, 0, dtype=DTYPE, device=device)
        if self.relaxed.numeric:
            node = self._node_values(states)
            kind = self.comparator
            holds = torch.where(kind == 0, node >= 0, torch.where(kind == 1, node > 0, node == 0))
            parts.append(holds)
            if self.numeric_kind.numel():
                multipliers = self._repetitions(node[:, self.numeric_node])
        return torch.cat(parts, dim=1), multipliers

    def _repetitions(self, value: torch.Tensor) -> torch.Tensor:
        """Como :func:`pyplanner.heuristics.relaxation.repetitions`, para todas las aristas."""
        kind, delta = self.numeric_kind, self.numeric_delta
        inf = torch.tensor(float("inf"), dtype=DTYPE, device=self.device)
        one = torch.ones((), dtype=DTYPE, device=self.device)
        ge = torch.where(value < 0, torch.clamp(torch.ceil(-value / delta), min=1.0), one)
        gt = torch.where(value <= 0, torch.floor(-value / delta) + 1.0, one)
        towards = (value < 0) == (delta > 0)
        eq = torch.where(
            value == 0,
            one,
            torch.where(towards, torch.clamp(torch.ceil(value.abs() / delta.abs()), min=1.0), inf),
        )
        result = torch.where(kind == REPEAT_GE, ge, torch.where(kind == REPEAT_GT, gt, eq))
        result = torch.where(torch.isnan(value), inf, result)
        return torch.where(kind == ONCE, one, result)

    # --- Costes -----------------------------------------------------------------------

    def _accumulated(self, h: torch.Tensor, combine_max: bool) -> torch.Tensor:
        """Suma o máximo de los costes de las precondiciones de cada acción relajada."""
        batch = h.shape[0]
        pre_values = h[:, self.pre_fact]
        accumulated = torch.zeros(batch, self.num_actions, dtype=DTYPE, device=self.device)
        if combine_max:
            return accumulated.scatter_reduce(
                1, self.pre_action.expand(batch, -1), pre_values, reduce="amax", include_self=True
            )
        return accumulated.index_add_(1, self.pre_action, pre_values)

    def _offers(
        self, h: torch.Tensor, multipliers: torch.Tensor, combine_max: bool
    ) -> torch.Tensor:
        """Valor que ofrece cada arista de efecto (lote × aristas); ``inf`` si no sirve.

        Las de los añadidos ofrecen coste + acumulado (multiplicador 1); solo las numéricas
        necesitan el producto por las repeticiones."""
        accumulated = self._accumulated(h, combine_max)
        offered = (self.cost + accumulated)[:, self.add_action]
        if not self.numeric_action.numel():
            return offered
        action = self.numeric_action
        numeric = multipliers * self.cost[action] + accumulated[:, action]
        numeric = torch.where(torch.isinf(multipliers), float("inf"), numeric)
        return torch.cat([offered, numeric], dim=1)

    def _fact_costs(
        self, true: torch.Tensor, multipliers: torch.Tensor, combine_max: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Costes (lote × hechos) y la última iteración en que bajó cada coste.

        Esa iteración es el **nivel** de la referencia en h_add: tras la iteración k, cada
        hecho tiene el mejor coste entre las derivaciones de altura ≤ k, y con costes no
        negativos una derivación de coste óptimo lo es también en todas sus
        precondiciones. Así que la primera iteración en que un hecho alcanza su coste
        definitivo es la altura mínima de una derivación óptima: 1 + el máximo nivel de
        las precondiciones del mejor logro, como en el desempate (coste, nivel)."""
        batch = true.shape[0]
        zero = torch.zeros((), dtype=DTYPE, device=self.device)
        h = torch.where(true, zero, float("inf"))
        level = torch.where(true, zero, float("inf"))
        fact_index = self.edge_fact.expand(batch, -1)
        for iteration in range(1, self.num_facts + 2):
            offered = self._offers(h, multipliers, combine_max)
            updated = h.scatter_reduce(1, fact_index, offered, reduce="amin", include_self=True)
            improved = updated < h
            if not improved.any():
                break
            level = torch.where(improved, float(iteration), level)
            h = updated
        return h, level

    def fact_costs(self, states: Sequence[State], combine_max: bool) -> torch.Tensor:
        """Matriz lote × hechos con los costes h_max o h_add (``inf`` si inalcanzable)."""
        true, multipliers = self._inputs(states)
        return self._fact_costs(true, multipliers, combine_max)[0]

    def clause_values(self, costs: torch.Tensor, combine_max: bool) -> torch.Tensor:
        """Matriz lote × cláusulas con el valor h_max o h_add de cada cláusula."""
        batch = costs.shape[0]
        values = torch.zeros(batch, self.num_clauses, dtype=DTYPE, device=self.device)
        goal_costs = costs[:, self.goal_fact]
        if combine_max:
            return values.scatter_reduce(
                1, self.goal_clause.expand(batch, -1), goal_costs, reduce="amax", include_self=True
            )
        return values.index_add_(1, self.goal_clause, goal_costs)

    def goal_values(self, costs: torch.Tensor, combine_max: bool) -> list[float]:
        """Valor h_max o h_add del objetivo (mínimo entre cláusulas) de cada estado."""
        if self.num_clauses == 0:
            return [DEAD_END] * costs.shape[0]
        values = self.clause_values(costs, combine_max).amin(dim=1)
        return [DEAD_END if v == float("inf") else v for v in values.tolist()]

    # --- h_FF -------------------------------------------------------------------------

    def relaxed_plans(self, states: Sequence[State]) -> list[Evaluation]:
        """h_FF (valor y acciones útiles) de cada estado del lote, extraído con tensores."""
        batch = len(states)
        if self.num_clauses == 0:
            return [Evaluation(DEAD_END) for _ in states]
        num_actions = self.num_actions
        device = self.device
        inf = float("inf")
        true, multipliers = self._inputs(states)
        h, level = self._fact_costs(true, multipliers, combine_max=False)
        pending = h > 0  # hechos relajados de coste no nulo

        # Aristas que dan a su hecho el coste mínimo (con los costes definitivos).
        offered = self._offers(h, multipliers, combine_max=False)
        fact_index = self.edge_fact.expand(batch, -1)
        best_offer = torch.full((batch, self.num_facts), inf, dtype=DTYPE, device=device)
        best_offer = best_offer.scatter_reduce(1, fact_index, offered, reduce="amin")
        optimal = (offered == best_offer[:, self.edge_fact]) & ~torch.isinf(offered)
        optimal &= ~true[:, self.edge_fact]

        # Nivel que da cada arista óptima: 1 + el máximo nivel de las precondiciones.
        pre_index = self.pre_action.expand(batch, -1)
        action_level = torch.zeros(batch, num_actions, dtype=DTYPE, device=device)
        action_level = action_level.scatter_reduce(
            1, pre_index, level[:, self.pre_fact], reduce="amax", include_self=True
        )
        edge_level = torch.where(optimal, action_level[:, self.edge_action] + 1, inf)

        # Mejor soportador: la acción de menor índice con el (coste, nivel) del hecho. El
        # índice num_actions hace de "ninguno" (columna extra que luego se descarta).
        supporting = optimal & (edge_level == level[:, self.edge_fact])
        candidates = torch.where(supporting, self.edge_action, num_actions)
        best = torch.full((batch, self.num_facts), num_actions, dtype=torch.long, device=device)
        best = best.scatter_reduce(1, fact_index, candidates, reduce="amin")
        # Repeticiones con las que el soportador logra cada hecho (una arista por par).
        # Los añadidos tienen multiplicador 1; los hechos numéricos toman el de su arista.
        fact_multiplier = torch.ones(batch, self.num_facts, dtype=DTYPE, device=device)
        if self.numeric_action.numel():
            numeric_supporting = supporting[:, self.add_action.shape[0] :]
            chosen_edge = numeric_supporting & (self.numeric_action == best[:, self.numeric_fact])
            fact_multiplier = fact_multiplier.scatter_reduce(
                1,
                self.numeric_fact.expand(batch, -1),
                torch.where(chosen_edge, multipliers, 0.0),
                reduce="amax",
                include_self=False,
            )

        # Cláusula del objetivo: la de menor valor h_add (la primera si hay empate).
        clause_values = self.clause_values(h, combine_max=False)
        chosen = clause_values.argmin(dim=1)
        dead = torch.isinf(clause_values.gather(1, chosen.unsqueeze(1)).squeeze(1))
        in_clause = (self.goal_clause.unsqueeze(0) == chosen.unsqueeze(1)) & ~dead.unsqueeze(1)
        start = in_clause & pending[:, self.goal_fact]
        frontier = torch.zeros(batch, self.num_facts, dtype=torch.uint8, device=device)
        frontier = frontier.scatter_reduce(
            1, self.goal_fact.expand(batch, -1), start.to(torch.uint8), "amax"
        ).bool()
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
            reached = reached.scatter_reduce(
                1, self.pre_fact.expand(batch, -1), required.to(torch.uint8), "amax"
            ).bool()
            frontier = reached & ~needed
            needed |= frontier

        # Repeticiones de cada acción relajada: máximo entre los hechos que soporta.
        repeats = torch.zeros(batch, num_actions + 1, dtype=DTYPE, device=device)
        repeats = repeats.scatter_reduce(
            1, best, torch.where(needed, fact_multiplier, 0.0), reduce="amax"
        )[:, :num_actions]
        per_operator = torch.zeros(batch, self.num_operators, dtype=DTYPE, device=device)
        if self.num_operators and num_actions:
            per_operator = per_operator.scatter_reduce(
                1, self.operator.expand(batch, -1), repeats, reduce="amax"
            )
        values = (per_operator * self.operator_cost).sum(dim=1)
        unsatisfied = torch.zeros(batch, num_actions, dtype=DTYPE, device=device)
        unsatisfied.index_add_(1, self.pre_action, pre_pending.to(DTYPE))
        plan = selected[:, :num_actions]
        preferred = torch.nonzero(self._to_operators(plan & (unsatisfied == 0))).tolist()

        per_state: list[list[int]] = [[] for _ in range(batch)]
        for row, operator in preferred:
            per_state[row].append(operator)
        return [
            Evaluation(DEAD_END) if is_dead else Evaluation(value, tuple(actions))
            for is_dead, value, actions in zip(
                dead.tolist(), values.tolist(), per_state, strict=True
            )
        ]

    def _to_operators(self, relaxed_mask: torch.Tensor) -> torch.Tensor:
        """Proyecta una máscara lote × acciones relajadas sobre lote × operadores."""
        batch = relaxed_mask.shape[0]
        result = torch.zeros(batch, self.num_operators, dtype=torch.uint8, device=self.device)
        if self.operator.numel() == 0:
            return result.bool()
        return result.scatter_reduce(
            1, self.operator.expand(batch, -1), relaxed_mask.to(torch.uint8), "amax"
        ).bool()


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


# Las constantes de tipo de arista tienen que coincidir con las de la referencia.
assert (REPEAT_GE, REPEAT_GT, REPEAT_EQ, ONCE) == (0, 1, 2, 3)
