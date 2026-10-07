"""Heurísticas de la relajación sin borrados: h_max, h_add y h_FF (implementación de referencia).

En la relajación sin borrados se ignoran los efectos de borrado, así que un hecho, una vez
alcanzado, sigue siendo cierto. El coste de alcanzar cada hecho se define así:

* h_max: coste(a) + **máximo** de los costes de las precondiciones de la acción ``a``;
* h_add: coste(a) + **suma** de los costes de las precondiciones (Bonet y Geffner, 2001).

El coste de un hecho es el mínimo sobre las acciones que lo añaden (0 si es cierto en el
estado). Se calcula con un Dijkstra generalizado: los hechos se extraen de un montículo por
orden de coste, y una acción se "dispara" cuando se han extraído todas sus precondiciones.

h_FF (Hoffmann y Nebel, 2001) extrae un **plan relajado** a partir de los costes de h_add:
para cada objetivo pendiente elige su mejor *soportador* y repite con sus precondiciones.

**Tarea relajada** (:class:`RelaxedTask`), a partir de la tarea normalizada:

* **Hechos complementarios.** Cada hecho *p* que aparece negado tiene un hecho "no-*p*",
  cierto si *p* es falso y que hacen cierto las acciones que borran *p* (como en FF).
* **Acciones relajadas.** Cada operador da una acción relajada para sus efectos
  incondicionales y otra por cada efecto condicional (como en Fast Downward). Cada una
  recuerda su operador.
* **Condiciones numéricas** (iteración 3): cada condición numérica distinta es un nodo más
  de la relajación, con el enfoque de *h^hbd* (Scala, Haslum, Thiébaux y Ramírez, 2020).
  Una acción relajada cuyo efecto aumenta una expresión lineal en una cantidad constante Δ
  "logra" la condición repitiéndose ⌈déficit / Δ⌉ veces, así que la arista cuesta
  repeticiones × coste. El número de repeticiones depende del estado. Los efectos que no
  son incrementos constantes (``assign``, ``scale-*``, valores variables) y las condiciones
  no lineales se tratan de forma **optimista**: una sola repetición basta. Así la
  heurística nunca declara un callejón sin salida que no lo es. Los efectos que empeoran una
  condición se ignoran, como los borrados. La excepción (iteración 6c) es asignar una
  constante a la única variable de una condición: la logra si ese valor la cumple, y si no,
  solo sirve para definir la variable cuando no tiene valor (que otras acciones la aumenten
  después). Sin ella, poner en marcha el reloj de una acción durativa (``assign`` a 0)
  bastaba para la relajación para poder terminarla.
* **Objetivo disyuntivo.** El valor es el mínimo entre las cláusulas.
* **Restricciones de trayectoria** (iteración 5): las reglas de los monitores (``done`` de
  ``sometime``, quitar ``pending`` de ``sometime-after``) son acciones relajadas de coste 0
  con el mismo operador ficticio que los axiomas.
* **Predicados derivados** (iteración 4). Cada axioma es una acción relajada de coste 0
  que añade su hecho derivado. Todas comparten un **operador ficticio** (el de índice
  ``len(task.actions)``, de coste 0), que nunca es una acción útil. Las condiciones que
  **niegan** un hecho derivado se dan por ciertas, como hace Fast Downward por defecto: su
  complementario no lo "añade" ninguna acción, sino que depende de que dejen de cumplirse
  todos sus axiomas, y tratarlo como un hecho más declararía callejones sin salida que no
  lo son.

**Desempates deterministas e independientes del orden.** El Dijkstra ordena los hechos
por (coste, nivel), donde el nivel de una acción es 1 + el máximo nivel de sus
precondiciones, y un hecho cierto tiene nivel 0. El mejor soportador de un hecho es la
acción relajada de menor índice entre las que le dan su (coste, nivel) mínimo. El nivel
crece en cada paso aunque haya acciones de coste 0, así que el plan relajado no tiene
ciclos. La implementación en PyTorch, que no tiene orden de extracción, calcula lo mismo
con dos puntos fijos (costes y después niveles), así que las tres implementaciones dan
exactamente el mismo resultado.

h_FF suma, por cada operador del plan relajado, su coste por el **máximo** número de
repeticiones que necesita alguna de sus acciones relajadas elegidas. Las acciones útiles
son los operadores de las acciones relajadas elegidas que tienen todas sus precondiciones
ciertas.

Todos los costes son no negativos. Cuando se han extraído todos los hechos del objetivo,
los costes, niveles y soportadores que necesita la extracción ya son definitivos, y el
cálculo termina.
"""

import heapq
import math
from collections.abc import Sequence

from pyplanner.grounding import GroundAction, GroundTask, State, bits, mask_of
from pyplanner.grounding.numeric import Const, NumericAssignment, NumericCondition, variables_of
from pyplanner.heuristics.base import DEAD_END, Evaluation, Heuristic

#: Tipos de arista numérica: repeticiones para ``>=``, para ``>``, para ``=``, optimista y
#: la de una asignación que no cumple la condición (solo sirve para definir la variable).
REPEAT_GE, REPEAT_GT, REPEAT_EQ, ONCE, DEFINE = 0, 1, 2, 3, 4


class RelaxedTask:
    """Estructuras de la relajación, precalculadas una vez por tarea.

    Las comparten las tres implementaciones (Python, Numba y PyTorch).

    Attributes:
        num_facts: hechos de la tarea + complementarios + condiciones numéricas.
        negated: hechos con complementario; el de ``negated[i]`` es ``num_base + i``.
        numeric: condiciones numéricas; la ``j`` es el hecho ``numeric_offset + j``.
        pre, add, cost, operator: precondiciones, añadidos proposicionales, coste y
            operador de cada acción relajada.
        edges: aristas numéricas ``(acción relajada, hecho numérico, tipo, Δ)``.
        edges_of: índices de las aristas numéricas de cada acción relajada.
        goal: hechos de cada cláusula del objetivo.
    """

    def __init__(self, task: GroundTask) -> None:
        self.task = task
        self.num_base = len(task.facts)
        derived = task.derived_mask
        negated = sorted(
            f
            for f in {f for a in task.actions for f in a.neg_facts}
            | {f for a in task.actions for e in a.conditional for f in e.condition.neg_facts}
            | {f for clause in task.goal for f in clause.neg_facts}
            | {f for axiom in task.axioms for f in axiom.condition.neg_facts}
            | {f for rule in task.monitor_rules for f in rule.condition.neg_facts}
            if not derived >> f & 1
        )
        self.negated = tuple(negated)
        self.negated_mask = mask_of(negated)
        self._complement = {f: self.num_base + i for i, f in enumerate(negated)}
        self.numeric_offset = self.num_base + len(negated)
        numeric: dict[NumericCondition, int] = {}

        def node(condition: NumericCondition) -> int:
            if condition not in numeric:
                numeric[condition] = self.numeric_offset + len(numeric)
            return numeric[condition]

        def literals(
            pos: Sequence[int], neg: Sequence[int], nums: Sequence[NumericCondition]
        ) -> tuple[int, ...]:
            complement = self._complement
            negative = (complement[f] for f in neg if f in complement)  # sin los derivados
            return tuple(sorted({*pos, *negative, *map(node, nums)}))

        def adds(add: Sequence[int], deletes: Sequence[int]) -> tuple[int, ...]:
            complement = self._complement
            return tuple(sorted({*add, *(complement[f] for f in deletes if f in complement)}))

        self.pre: list[tuple[int, ...]] = []
        self.add: list[tuple[int, ...]] = []
        self.cost: list[float] = []
        self.operator: list[int] = []
        self._numeric_effects: list[tuple[NumericAssignment, ...]] = []
        for action in task.actions:
            base_pre = literals(action.pre_facts, action.neg_facts, action.numeric_pre)
            self._append(base_pre, adds(action.add_facts, action.del_facts), action, action.numeric)
            for effect in action.conditional:
                condition = literals(
                    effect.condition.pos_facts, effect.condition.neg_facts, effect.condition.numeric
                )
                pre = tuple(sorted({*base_pre, *condition}))
                self._append(pre, adds(effect.add_facts, effect.del_facts), action, effect.numeric)
        #: Operador ficticio de los axiomas (-1 si no hay), y número de operadores.
        # Axiomas y reglas de los monitores: acciones relajadas de coste 0 con un operador
        # ficticio común, que nunca es una acción útil.
        extra = bool(task.axioms or task.monitor_rules)
        self.axiom_operator = len(task.actions) if extra else -1
        self.num_operators = len(task.actions) + extra
        for axiom in task.axioms:
            body = axiom.condition
            self.pre.append(literals(body.pos_facts, body.neg_facts, body.numeric))
            self.add.append((axiom.head,))
            self.cost.append(0.0)
            self.operator.append(self.axiom_operator)
            self._numeric_effects.append(())
        for rule in task.monitor_rules:
            body = rule.condition
            self.pre.append(literals(body.pos_facts, body.neg_facts, body.numeric))
            self.add.append(adds(rule.add_facts, rule.del_facts))
            self.cost.append(0.0)
            self.operator.append(self.axiom_operator)
            self._numeric_effects.append(())
        self.goal = [literals(c.pos_facts, c.neg_facts, c.numeric) for c in task.goal]
        self.numeric = list(numeric)
        self.num_facts = self.numeric_offset + len(self.numeric)
        self.num_actions = len(self.pre)
        self.operator_cost = [a.cost for a in task.actions] + [0.0] * extra
        self.goal_facts = frozenset(f for clause in self.goal for f in clause)

        # Aristas numéricas.
        self.edges: list[tuple[int, int, int, float]] = []
        for r, effects in enumerate(self._numeric_effects):
            if not effects:
                continue
            for j, numeric_condition in enumerate(self.numeric):
                edge = _numeric_edge(numeric_condition, effects)
                if edge is not None:
                    kind, delta = edge
                    self.edges.append((r, self.numeric_offset + j, kind, delta))
        self.edges_of: list[list[int]] = [[] for _ in range(self.num_actions)]
        for e, (r, _, _, _) in enumerate(self.edges):
            self.edges_of[r].append(e)

        self.pre_of: list[list[int]] = [[] for _ in range(self.num_facts)]
        for a in range(self.num_actions):
            for f in self.pre[a]:
                self.pre_of[f].append(a)
        self.no_pre = [a for a in range(self.num_actions) if not self.pre[a]]
        self._pre_len = [len(p) for p in self.pre]

    def _append(
        self,
        pre: tuple[int, ...],
        add: tuple[int, ...],
        action: GroundAction,
        numeric: tuple[NumericAssignment, ...],
    ) -> None:
        if not add and not numeric:
            return
        self.pre.append(pre)
        self.add.append(add)
        self.cost.append(action.cost)
        self.operator.append(action.index)
        self._numeric_effects.append(numeric)

    @property
    def complement(self) -> dict[int, int]:
        """Nodo complementario ("no-*p*") de cada hecho negable."""
        return self._complement

    # --- Por estado -------------------------------------------------------------------

    def true_facts(self, state: State) -> list[int]:
        """Hechos relajados ciertos en ``state``: los del estado, los complementarios de los
        hechos negables falsos y las condiciones numéricas que se cumplen."""
        complement = self._complement
        facts = state.facts
        result = [*bits(facts), *(complement[f] for f in bits(~facts & self.negated_mask))]
        values = state.values
        offset = self.numeric_offset
        result.extend(offset + j for j, c in enumerate(self.numeric) if c.holds(values))
        return result

    def multipliers(self, state: State) -> list[float]:
        """Repeticiones de cada arista numérica en ``state`` (``inf`` si no sirve)."""
        values = state.values
        offset = self.numeric_offset
        return [
            repetitions(kind, delta, self.numeric[fact - offset].value(values))
            for _, fact, kind, delta in self.edges
        ]

    def fact_costs(self, state: State, combine_max: bool) -> list[float]:
        """Costes h_max (``combine_max``) o h_add de cada hecho relajado desde ``state``.

        Se detiene cuando se han extraído todos los hechos del objetivo; los hechos no
        extraídos pueden quedar con un coste provisional o infinito.
        """
        return self._dijkstra(state, combine_max, None)[0]

    def fact_costs_and_supporters(self, state: State) -> tuple[list[float], list[int], list[float]]:
        """Costes h_add, mejor soportador y repeticiones de cada hecho extraído (para h_FF)."""
        supporters = [-1] * self.num_facts
        cost, multiplier = self._dijkstra(state, False, supporters)
        return cost, supporters, multiplier

    def _dijkstra(
        self, state: State, combine_max: bool, supporters: list[int] | None
    ) -> tuple[list[float], list[float]]:
        # Bucle crítico: variables locales. Clave de cada hecho: (coste, nivel).
        num_facts = self.num_facts
        cost: list[float] = [DEAD_END] * num_facts
        level: list[int] = [0] * num_facts
        multiplier: list[float] = [1.0] * num_facts
        track = supporters is not None
        support: list[int] = supporters if supporters is not None else []
        heap: list[tuple[float, int, int]] = []
        for f in self.true_facts(state):
            cost[f] = 0.0
            heap.append((0.0, 0, f))
        heapq.heapify(heap)
        push, pop = heapq.heappush, heapq.heappop
        mult = self.multipliers(state) if self.edges else []
        unsatisfied = self._pre_len.copy()
        accumulated = [0.0] * self.num_actions
        action_level = [0] * self.num_actions
        action_cost, adds, pre_of = self.cost, self.add, self.pre_of
        edges, edges_of = self.edges, self.edges_of
        goal_facts = self.goal_facts

        def offer(q: int, value: float, lev: int, a: int, m: float) -> None:
            current, current_level = cost[q], level[q]
            if value < current or (value == current and lev < current_level):
                cost[q], level[q] = value, lev
                multiplier[q] = m
                push(heap, (value, lev, q))
                if track:
                    support[q] = a
            elif track and value == current and lev == current_level and a < support[q]:
                support[q] = a
                multiplier[q] = m

        def trigger(a: int) -> None:
            acc = accumulated[a]
            lev = action_level[a] + 1
            base = action_cost[a]
            for q in adds[a]:
                offer(q, base + acc, lev, a, 1.0)
            for e in edges_of[a]:
                m = mult[e]
                if m != DEAD_END:
                    offer(edges[e][1], m * base + acc, lev, a, m)

        for a in self.no_pre:
            trigger(a)
        pending_goals = sum(1 for g in goal_facts if cost[g] != 0)
        done = [False] * num_facts
        while heap and pending_goals:
            c, lev, f = pop(heap)
            if done[f] or c != cost[f] or lev != level[f]:
                continue  # entrada obsoleta
            done[f] = True
            if c != 0 and f in goal_facts:
                pending_goals -= 1
            for a in pre_of[f]:
                remaining = unsatisfied[a] - 1
                unsatisfied[a] = remaining
                if combine_max:
                    if c > accumulated[a]:
                        accumulated[a] = c
                else:
                    accumulated[a] += c
                if lev > action_level[a]:
                    action_level[a] = lev
                if remaining == 0:
                    trigger(a)
        return cost, multiplier

    def clause_values(self, cost: Sequence[float], combine_max: bool) -> list[float]:
        """Valor h_max o h_add de cada cláusula del objetivo."""
        values: list[float] = []
        for clause in self.goal:
            costs = [cost[f] for f in clause]
            values.append((max(costs) if combine_max else sum(costs)) if costs else 0.0)
        return values

    def goal_value(self, cost: Sequence[float], combine_max: bool) -> float:
        """Valor h_max o h_add del objetivo: el mínimo entre sus cláusulas."""
        return min(self.clause_values(cost, combine_max), default=DEAD_END)

    def relaxed_plan(
        self, cost: Sequence[float], supporters: Sequence[int], multiplier: Sequence[float]
    ) -> Evaluation:
        """Extrae el plan relajado de h_FF (ver la documentación del módulo)."""
        values = self.clause_values(cost, False)
        if not values or min(values) == DEAD_END:
            return Evaluation(DEAD_END)
        clause = self.goal[values.index(min(values))]
        repeats: dict[int, float] = {}  # acción relajada → repeticiones necesarias
        marked: set[int] = set()
        stack = [g for g in clause if cost[g] != 0]
        while stack:
            f = stack.pop()
            if f in marked:
                continue
            marked.add(f)
            best = supporters[f]
            if best not in repeats:
                repeats[best] = multiplier[f]
                stack.extend(p for p in self.pre[best] if cost[p] != 0)
            elif multiplier[f] > repeats[best]:
                repeats[best] = multiplier[f]
        per_operator: dict[int, float] = {}
        preferred: set[int] = set()
        for a, times in repeats.items():
            operator = self.operator[a]
            if times > per_operator.get(operator, 0.0):
                per_operator[operator] = times
            if operator != self.axiom_operator and all(cost[p] == 0 for p in self.pre[a]):
                preferred.add(operator)
        # Suma en orden de operador: el mismo orden de sumas que en Numba.
        value = sum(self.operator_cost[o] * times for o, times in sorted(per_operator.items()))
        return Evaluation(value, tuple(sorted(preferred)))


def repetitions(kind: int, delta: float, value: float | None) -> float:
    """Repeticiones de una arista numérica cuando la condición vale ``value`` (``inf`` si la
    arista no sirve). Ver la documentación del módulo."""
    if kind == ONCE:
        return 1.0
    if kind == DEFINE:
        return 1.0 if value is None else DEAD_END
    if value is None:
        return DEAD_END
    if kind == REPEAT_GE:
        return max(1.0, float(math.ceil(-value / delta))) if value < 0 else 1.0
    if kind == REPEAT_GT:
        return float(math.floor(-value / delta)) + 1.0 if value <= 0 else 1.0
    # REPEAT_EQ: hay que moverse hacia 0 en la dirección de delta.
    if value == 0:
        return 1.0
    if (value < 0) == (delta > 0):
        return max(1.0, float(math.ceil(abs(value) / abs(delta))))
    return DEAD_END


def _numeric_edge(
    condition: NumericCondition, effects: Sequence[NumericAssignment]
) -> tuple[int, float] | None:
    """Tipo y Δ de la arista de una acción relajada a una condición, o ``None`` si no la
    mejora."""
    form = condition.linear
    touched = {a.variable for a in effects}
    if form is None:
        return (ONCE, 0.0) if touched & variables_of(condition.expression) else None
    weights = dict(form.terms)
    if not touched & weights.keys():
        return None
    if len(weights) == 1:
        # Una condición sobre una sola variable y un 'assign' de una constante: la consigue
        # si, y solo si, ese valor la cumple (no hace falta ser optimista). Si no la cumple,
        # solo sirve para definir la variable, por si luego la aumentan otras acciones.
        [(variable, weight)] = weights.items()
        assigns = [e for e in effects if e.variable == variable]
        if len(assigns) == 1 and assigns[0].op == "assign" and isinstance(assigns[0].value, Const):
            value = weight * assigns[0].value.value + form.constant
            holds = {">=": value >= 0, ">": value > 0, "=": value == 0}[condition.comparator]
            return (ONCE, 0.0) if holds else (DEFINE, 0.0)
    delta = 0.0
    for effect in effects:
        weight = weights.get(effect.variable)
        if weight is None:
            continue
        if effect.op not in ("increase", "decrease") or not isinstance(effect.value, Const):
            return (ONCE, 0.0)  # no es un incremento constante: optimista
        amount = effect.value.value
        delta += weight * (amount if effect.op == "increase" else -amount)
    if condition.comparator == "=":
        return (REPEAT_EQ, delta) if delta != 0 else None
    if delta <= 0:
        return None  # empeora o no cambia la condición: se ignora, como un borrado
    return (REPEAT_GE if condition.comparator == ">=" else REPEAT_GT, delta)


class _RelaxationHeuristic(Heuristic):
    combine_max = False

    def __init__(self, task: GroundTask, relaxed: RelaxedTask | None = None) -> None:
        super().__init__(task)
        self.relaxed = relaxed or RelaxedTask(task)


class HMaxHeuristic(_RelaxationHeuristic):
    """h_max (con *h^hbd* para las condiciones numéricas)."""

    name = "hmax"
    combine_max = True

    def evaluate_batch(self, states: Sequence[State]) -> list[Evaluation]:
        r = self.relaxed
        return [Evaluation(r.goal_value(r.fact_costs(s, True), True)) for s in states]


class HAddHeuristic(_RelaxationHeuristic):
    """h_add (con *h^hbd* para las condiciones numéricas)."""

    name = "hadd"

    def evaluate_batch(self, states: Sequence[State]) -> list[Evaluation]:
        r = self.relaxed
        return [Evaluation(r.goal_value(r.fact_costs(s, False), False)) for s in states]


class FFHeuristic(_RelaxationHeuristic):
    """h_FF: coste del plan relajado, con acciones útiles."""

    name = "ff"
    provides_preferred = True

    def evaluate_batch(self, states: Sequence[State]) -> list[Evaluation]:
        r = self.relaxed
        results = []
        for state in states:
            cost, supporters, multiplier = r.fact_costs_and_supporters(state)
            results.append(r.relaxed_plan(cost, supporters, multiplier))
        return results
