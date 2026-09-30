"""Implementación compilada con Numba de h_max, h_add y h_FF (dispositivo ``"numba"``).

Es la misma relajación y el mismo algoritmo que la implementación de referencia
(:mod:`pyplanner.heuristics.relaxation`), compilados a código máquina con Numba:

* un Dijkstra generalizado con montículo propio sobre arrays, con claves (coste, nivel,
  hecho) en el mismo orden que la referencia, que se detiene al extraer todos los hechos
  del objetivo y anota el mejor soportador y las repeticiones de cada hecho;
* las condiciones numéricas como nodos, con las repeticiones de sus aristas calculadas por
  estado (*h^hbd*);
* la extracción del plan relajado de h_FF sobre la cláusula del objetivo de menor valor.

Las reglas de desempate y el orden de las sumas son los de la referencia, así que los
valores y las acciones útiles son **idénticos**; los tests lo comprueban.

El lote entero se evalúa en una sola llamada: los estados llegan como una matriz de bytes
(los bitsets) y una matriz de valores numéricos (NaN si están indefinidos), y se
decodifican dentro del código compilado. Las condiciones numéricas lineales se evalúan
dentro; las no lineales (raras) se evalúan en Python y llegan ya calculadas.

Este módulo solo se importa si Numba está instalado (``pip install -e ".[fast]"``). La
primera compilación tarda unos segundos; después se guarda en la caché de disco.
"""

from collections.abc import Sequence

import numpy as np
from numba import njit

from pyplanner.grounding import GroundTask, State
from pyplanner.heuristics.base import DEAD_END, Evaluation, Heuristic
from pyplanner.heuristics.relaxation import ONCE, REPEAT_EQ, REPEAT_GE, REPEAT_GT, RelaxedTask

INF = np.inf

#: Códigos de los comparadores canónicos de las condiciones numéricas.
_COMPARATORS = {">=": 0, ">": 1, "=": 2}


class CompiledRelaxation:
    """Una :class:`~pyplanner.heuristics.relaxation.RelaxedTask` en arrays de NumPy (CSR)."""

    def __init__(self, relaxed: RelaxedTask) -> None:
        self.relaxed = relaxed
        self.num_bytes = max(1, (relaxed.num_base + 7) // 8)
        self.num_variables = len(relaxed.task.variables)
        self.pre_ptr, self.pre_idx = _csr(relaxed.pre)
        self.pre_of_ptr, self.pre_of_idx = _csr(relaxed.pre_of)
        self.goal_ptr, self.goal_idx = _csr(relaxed.goal)
        # Efectos de cada acción relajada: sus añadidos (arista -1, multiplicador 1) y sus
        # aristas numéricas, en una sola lista para ofrecerlos en un solo bucle.
        effects = [
            [(f, -1) for f in relaxed.add[a]] + [(relaxed.edges[e][1], e) for e in edges_of]
            for a, edges_of in enumerate(relaxed.edges_of)
        ]
        self.effect_ptr, self.effect_fact = _csr([[f for f, _ in items] for items in effects])
        self.effect_edge = _csr([[e for _, e in items] for items in effects])[1]
        self.cost = np.array(relaxed.cost, dtype=np.float64)
        self.operator = np.array(relaxed.operator, dtype=np.int64)
        self.operator_cost = np.array(relaxed.operator_cost, dtype=np.float64)
        self.pre_len = np.array([len(p) for p in relaxed.pre], dtype=np.int64)
        self.no_pre = np.array(relaxed.no_pre, dtype=np.int64)
        self.negated = np.array(relaxed.negated, dtype=np.int64)
        edges = relaxed.edges
        self.edge_fact = np.array([e[1] for e in edges], dtype=np.int64)
        self.edge_kind = np.array([e[2] for e in edges], dtype=np.int64)
        self.edge_delta = np.array([e[3] for e in edges], dtype=np.float64)
        # Condiciones numéricas: forma lineal en CSR, o evaluación en Python si no es lineal.
        term_ptr = [0]
        term_var: list[int] = []
        term_weight: list[float] = []
        constant: list[float] = []
        self.nonlinear = [j for j, c in enumerate(relaxed.numeric) if c.linear is None]
        for condition in relaxed.numeric:
            form = condition.linear
            if form is not None:
                for variable, weight in form.terms:
                    term_var.append(variable)
                    term_weight.append(weight)
                constant.append(form.constant)
            else:
                constant.append(0.0)
            term_ptr.append(len(term_var))
        self.term_ptr = np.array(term_ptr, dtype=np.int64)
        self.term_var = np.array(term_var, dtype=np.int64)
        self.term_weight = np.array(term_weight, dtype=np.float64)
        self.constant = np.array(constant, dtype=np.float64)
        self.linear = np.array([c.linear is not None for c in relaxed.numeric], dtype=np.bool_)
        self.comparator = np.array(
            [_COMPARATORS[c.comparator] for c in relaxed.numeric], dtype=np.int64
        )

    def _inputs(self, states: Sequence[State]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Hechos (lote × bytes), valores (lote × variables) y valores externos (lote ×
        condiciones, solo para las no lineales)."""
        batch = len(states)
        raw = b"".join(s.facts.to_bytes(self.num_bytes, "little") for s in states)
        facts = np.frombuffer(raw, dtype=np.uint8).reshape(batch, self.num_bytes)
        if not self.num_variables:  # tarea proposicional: nada más que convertir
            empty = np.empty((batch, 0))
            return facts, empty, empty
        values = np.array(
            [[np.nan if v is None else v for v in s.values] for s in states], dtype=np.float64
        ).reshape(batch, self.num_variables)
        if not self.nonlinear:
            return facts, values, np.empty((batch, 0))
        external = np.full((batch, len(self.relaxed.numeric)), np.nan)
        for j in self.nonlinear:
            condition = self.relaxed.numeric[j]
            for row, state in enumerate(states):
                value = condition.value(state.values)
                external[row, j] = np.nan if value is None else value
        return facts, values, external

    def _arrays(self) -> tuple[object, ...]:
        r = self.relaxed
        return (
            r.num_base,
            r.num_facts,
            r.numeric_offset,
            self.negated,
            self.pre_len,
            self.pre_of_ptr,
            self.pre_of_idx,
            self.effect_ptr,
            self.effect_fact,
            self.effect_edge,
            self.cost,
            self.no_pre,
            self.goal_ptr,
            self.goal_idx,
            self.edge_fact,
            self.edge_kind,
            self.edge_delta,
            self.term_ptr,
            self.term_var,
            self.term_weight,
            self.constant,
            self.linear,
            self.comparator,
        )

    def goal_values(self, states: Sequence[State], combine_max: bool) -> list[float]:
        """Valor h_max o h_add de cada estado."""
        facts, values, external = self._inputs(states)
        result = _goal_values(facts, values, external, combine_max, *self._arrays())
        return [DEAD_END if v == INF else float(v) for v in result]

    def relaxed_plans(self, states: Sequence[State]) -> list[Evaluation]:
        """h_FF (valor y acciones útiles) de cada estado."""
        facts, values, external = self._inputs(states)
        result, preferred, pointers = _relaxed_plans(
            facts,
            values,
            external,
            *self._arrays(),
            self.pre_ptr,
            self.pre_idx,
            self.operator,
            self.operator_cost,
            len(self.relaxed.task.actions),
        )
        bounds = pointers.tolist()
        flat = preferred[: bounds[-1]].tolist()  # solo la parte usada (el array es lote × hechos)
        return [
            Evaluation(DEAD_END)
            if value == INF
            else Evaluation(value, tuple(flat[bounds[row] : bounds[row + 1]]))
            for row, value in enumerate(result.tolist())
        ]


def _csr(lists: Sequence[Sequence[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Lista de listas → formato CSR (punteros de inicio y valores concatenados)."""
    pointers = np.zeros(len(lists) + 1, dtype=np.int64)
    pointers[1:] = np.cumsum([len(items) for items in lists])
    values = np.array([x for items in lists for x in items], dtype=np.int64)
    return pointers, values


# --- Núcleos compilados -------------------------------------------------------------
# Solo tipos simples y arrays de NumPy; nada de objetos de Python. El montículo es
# indexado (una entrada por hecho, con decrease-key) y extrae en el orden (coste, nivel,
# hecho): el mismo que el montículo con entradas obsoletas de la referencia.


@njit(cache=True, inline="always")
def _less(fact_cost, level, f, g):
    """Orden de extracción: (coste, nivel, hecho), como las tuplas de la referencia."""
    if fact_cost[f] != fact_cost[g]:
        return fact_cost[f] < fact_cost[g]
    if level[f] != level[g]:
        return level[f] < level[g]
    return f < g


@njit(cache=True, inline="always")
def _sift_up(heap, position, fact_cost, level, i):
    f = heap[i]
    while i > 0:
        parent = (i - 1) >> 1
        g = heap[parent]
        if not _less(fact_cost, level, f, g):
            break
        heap[i] = g
        position[g] = i
        i = parent
    heap[i] = f
    position[f] = i


@njit(cache=True, inline="always")
def _heap_pop(heap, position, fact_cost, level, size):
    """Extrae el hecho mínimo; devuelve el hecho y el nuevo tamaño."""
    top = heap[0]
    position[top] = -1
    size -= 1
    if size > 0:
        f = heap[size]
        i = 0
        while True:
            child = 2 * i + 1
            if child >= size:
                break
            if child + 1 < size and _less(fact_cost, level, heap[child + 1], heap[child]):
                child += 1
            g = heap[child]
            if not _less(fact_cost, level, g, f):
                break
            heap[i] = g
            position[g] = i
            i = child
        heap[i] = f
        position[f] = i
    return top, size


@njit(cache=True, inline="always")
def _heap_update(heap, position, fact_cost, level, size, f):
    """Inserta ``f`` o, si ya está, lo sube tras mejorar su clave. Devuelve el tamaño."""
    i = position[f]
    if i < 0:
        i = size
        heap[i] = f
        size += 1
    _sift_up(heap, position, fact_cost, level, i)
    return size


@njit(cache=True, inline="always")
def _repetitions(kind, delta, value):
    """Igual que :func:`pyplanner.heuristics.relaxation.repetitions` (NaN = indefinido)."""
    if kind == 3:  # ONCE
        return 1.0
    if np.isnan(value):
        return INF
    if kind == 0:  # REPEAT_GE
        return max(1.0, np.ceil(-value / delta)) if value < 0 else 1.0
    if kind == 1:  # REPEAT_GT
        return np.floor(-value / delta) + 1.0 if value <= 0 else 1.0
    if value == 0:  # REPEAT_EQ
        return 1.0
    if (value < 0) == (delta > 0):
        return max(1.0, np.ceil(abs(value) / abs(delta)))
    return INF


@njit(cache=True)
def _dijkstra(
    facts_row,
    values_row,
    external_row,
    combine_max,
    track,
    num_base,
    num_facts,
    numeric_offset,
    negated,
    pre_len,
    pre_of_ptr,
    pre_of_idx,
    effect_ptr,
    effect_fact,
    effect_edge,
    cost,
    no_pre,
    goal_ptr,
    goal_idx,
    edge_fact,
    edge_kind,
    edge_delta,
    term_ptr,
    term_var,
    term_weight,
    constant,
    linear,
    comparator,
):
    """Costes, soportadores y repeticiones de los hechos relajados (ver la referencia)."""
    fact_cost = np.full(num_facts, INF)
    level = np.zeros(num_facts, dtype=np.int64)
    multiplier = np.ones(num_facts)
    supporter = np.full(num_facts, -1, dtype=np.int64)
    done = np.zeros(num_facts, dtype=np.bool_)
    is_goal = np.zeros(num_facts, dtype=np.bool_)
    for g in goal_idx:
        is_goal[g] = True
    # Montículo indexado: una entrada por hecho, con las claves en fact_cost y level.
    heap = np.empty(num_facts, dtype=np.int64)
    position = np.full(num_facts, -1, dtype=np.int64)
    size = 0
    # Hechos ciertos: del estado, complementarios y condiciones numéricas que se cumplen.
    for f in range(num_base):
        if (facts_row[f >> 3] >> (f & 7)) & 1:
            fact_cost[f] = 0.0
            size = _heap_update(heap, position, fact_cost, level, size, f)
    for i in range(negated.shape[0]):
        f = negated[i]
        if not (facts_row[f >> 3] >> (f & 7)) & 1:
            fact_cost[num_base + i] = 0.0
            size = _heap_update(heap, position, fact_cost, level, size, num_base + i)
    num_numeric = constant.shape[0]
    node_value = np.empty(num_numeric)
    for j in range(num_numeric):
        if linear[j]:
            total = constant[j]
            for t in range(term_ptr[j], term_ptr[j + 1]):
                total += term_weight[t] * values_row[term_var[t]]
        else:
            total = external_row[j]
        node_value[j] = total
        kind = comparator[j]
        holds = total >= 0.0 if kind == 0 else (total > 0.0 if kind == 1 else total == 0.0)
        if holds:
            fact_cost[numeric_offset + j] = 0.0
            size = _heap_update(heap, position, fact_cost, level, size, numeric_offset + j)
    mult = np.empty(edge_fact.shape[0])
    for e in range(edge_fact.shape[0]):
        mult[e] = _repetitions(
            edge_kind[e], edge_delta[e], node_value[edge_fact[e] - numeric_offset]
        )
    unsatisfied = pre_len.copy()
    accumulated = np.zeros(pre_len.shape[0])
    action_level = np.zeros(pre_len.shape[0], dtype=np.int64)
    # Disparo de una acción relajada: ofrece sus añadidos y sus aristas numéricas.
    pending_triggers = np.empty(pre_len.shape[0], dtype=np.int64)
    num_pending = 0
    for a in no_pre:
        pending_triggers[num_pending] = a
        num_pending += 1
    pending = 0
    first = True
    while True:
        # Procesa los disparos pendientes (en orden) antes de seguir extrayendo.
        for p in range(num_pending):
            a = pending_triggers[p]
            acc = accumulated[a]
            lev = action_level[a] + 1
            base = cost[a]
            # Oferta a cada efecto; como ``offer`` en la referencia. Va escrita aquí y no en
            # una función: pasar tantos arrays en cada llamada multiplicaba el tiempo.
            for k in range(effect_ptr[a], effect_ptr[a + 1]):
                e = effect_edge[k]
                m = 1.0 if e < 0 else mult[e]
                if m == INF:
                    continue
                q = effect_fact[k]
                value = m * base + acc
                current = fact_cost[q]
                if value < current or (value == current and lev < level[q]):
                    fact_cost[q] = value
                    level[q] = lev
                    multiplier[q] = m
                    supporter[q] = a
                    if not done[q]:
                        size = _heap_update(heap, position, fact_cost, level, size, q)
                elif track and value == current and lev == level[q] and a < supporter[q]:
                    supporter[q] = a
                    multiplier[q] = m
        num_pending = 0
        if first:
            first = False
            for f in range(num_facts):
                if is_goal[f] and fact_cost[f] != 0.0:
                    pending += 1
        if size == 0 or pending == 0:
            break
        f, size = _heap_pop(heap, position, fact_cost, level, size)
        c = fact_cost[f]
        lev = level[f]
        done[f] = True
        if c != 0.0 and is_goal[f]:
            pending -= 1
        for j in range(pre_of_ptr[f], pre_of_ptr[f + 1]):
            a = pre_of_idx[j]
            unsatisfied[a] -= 1
            if combine_max:
                if c > accumulated[a]:
                    accumulated[a] = c
            else:
                accumulated[a] += c
            if lev > action_level[a]:
                action_level[a] = lev
            if unsatisfied[a] == 0:
                pending_triggers[num_pending] = a
                num_pending += 1
    return fact_cost, supporter, multiplier


@njit(cache=True)
def _best_clause(fact_cost, combine_max, goal_ptr, goal_idx):
    """Índice y valor de la cláusula del objetivo de menor valor (la primera si hay empate)."""
    best = -1
    best_value = INF
    for clause in range(goal_ptr.shape[0] - 1):
        value = 0.0
        for j in range(goal_ptr[clause], goal_ptr[clause + 1]):
            c = fact_cost[goal_idx[j]]
            if combine_max:
                if c > value:
                    value = c
            else:
                value += c
        if best == -1 or value < best_value:
            best = clause
            best_value = value
    return best, best_value


@njit(cache=True)
def _goal_values(facts, values, external, combine_max, *arrays):
    result = np.empty(facts.shape[0])
    goal_ptr, goal_idx = arrays[12], arrays[13]
    for row in range(facts.shape[0]):
        fact_cost, _, _ = _dijkstra(
            facts[row], values[row], external[row], combine_max, False, *arrays
        )
        _, result[row] = _best_clause(fact_cost, combine_max, goal_ptr, goal_idx)
    return result


@njit(cache=True)
def _relaxed_plans(
    facts,
    values,
    external,
    num_base,
    num_facts,
    numeric_offset,
    negated,
    pre_len,
    pre_of_ptr,
    pre_of_idx,
    effect_ptr,
    effect_fact,
    effect_edge,
    cost,
    no_pre,
    goal_ptr,
    goal_idx,
    edge_fact,
    edge_kind,
    edge_delta,
    term_ptr,
    term_var,
    term_weight,
    constant,
    linear,
    comparator,
    pre_ptr,
    pre_idx,
    operator,
    operator_cost,
    num_operators,
):
    batch = facts.shape[0]
    num_actions = pre_len.shape[0]
    result = np.empty(batch)
    # Acciones útiles en formato compacto: las del estado ``row`` están en
    # preferred[preferred_ptr[row]:preferred_ptr[row + 1]], ordenadas.
    preferred = np.empty(batch * max(num_facts, 1), dtype=np.int64)
    preferred_ptr = np.zeros(batch + 1, dtype=np.int64)
    repeats = np.zeros(num_actions)  # 0 = la acción relajada no está en el plan
    marked = np.zeros(num_facts, dtype=np.bool_)
    marked_list = np.empty(num_facts, dtype=np.int64)
    plan = np.empty(num_facts, dtype=np.int64)
    stack = np.empty(num_facts, dtype=np.int64)
    per_operator = np.zeros(num_operators)
    is_preferred = np.zeros(num_operators, dtype=np.bool_)
    used = np.empty(max(num_operators, 1), dtype=np.int64)
    for row in range(batch):
        preferred_ptr[row + 1] = preferred_ptr[row]
        fact_cost, supporter, multiplier = _dijkstra(
            facts[row], values[row], external[row], False, True, num_base, num_facts,
            numeric_offset, negated, pre_len, pre_of_ptr, pre_of_idx, effect_ptr, effect_fact,
            effect_edge, cost, no_pre, goal_ptr, goal_idx, edge_fact, edge_kind, edge_delta,
            term_ptr, term_var, term_weight, constant, linear, comparator,
        )  # fmt: skip
        clause, clause_value = _best_clause(fact_cost, False, goal_ptr, goal_idx)
        if clause == -1 or clause_value == INF:
            result[row] = INF
            continue
        num_marked = 0
        num_plan = 0
        top = 0
        for j in range(goal_ptr[clause], goal_ptr[clause + 1]):
            g = goal_idx[j]
            if fact_cost[g] != 0.0 and not marked[g]:
                marked[g] = True
                marked_list[num_marked] = g
                num_marked += 1
                stack[top] = g
                top += 1
        while top > 0:
            top -= 1
            f = stack[top]
            best = supporter[f]
            if repeats[best] == 0.0:
                repeats[best] = multiplier[f]
                plan[num_plan] = best
                num_plan += 1
                for j in range(pre_ptr[best], pre_ptr[best + 1]):
                    p = pre_idx[j]
                    if fact_cost[p] != 0.0 and not marked[p]:
                        marked[p] = True
                        marked_list[num_marked] = p
                        num_marked += 1
                        stack[top] = p
                        top += 1
            elif multiplier[f] > repeats[best]:
                repeats[best] = multiplier[f]
        num_used = 0
        for i in range(num_plan):
            a = plan[i]
            o = operator[a]
            if per_operator[o] == 0.0:  # primera vez (las repeticiones son al menos 1)
                used[num_used] = o
                num_used += 1
            if repeats[a] > per_operator[o]:
                per_operator[o] = repeats[a]
            ready = True
            for j in range(pre_ptr[a], pre_ptr[a + 1]):
                if fact_cost[pre_idx[j]] != 0.0:
                    ready = False
                    break
            if ready:
                is_preferred[o] = True
        # Suma en orden de operador, como la referencia (ordenación por inserción: los
        # operadores del plan relajado son pocos); y limpieza de lo tocado.
        for i in range(1, num_used):
            o = used[i]
            j = i - 1
            while j >= 0 and used[j] > o:
                used[j + 1] = used[j]
                j -= 1
            used[j + 1] = o
        total = 0.0
        start = preferred_ptr[row]
        count = 0
        for i in range(num_used):
            o = used[i]
            total += operator_cost[o] * per_operator[o]
            if is_preferred[o]:
                preferred[start + count] = o
                count += 1
            per_operator[o] = 0.0
            is_preferred[o] = False
        preferred_ptr[row + 1] = start + count
        result[row] = total
        for i in range(num_plan):
            repeats[plan[i]] = 0.0
        for i in range(num_marked):
            marked[marked_list[i]] = False
    return result, preferred, preferred_ptr


def warm_up() -> None:
    """Compila (o carga de la caché de disco) todos los núcleos con tareas mínimas.

    El portfolio la llama en el proceso padre antes de lanzar los hijos: así la caché queda
    completa aunque un hijo se cancele a mitad de compilar, y con ``fork`` los hijos
    heredan los núcleos ya compilados en memoria. Se usan una tarea proposicional y una
    numérica, porque los tipos de algunos arrays cambian si están vacíos.
    """
    from pyplanner.grounding import Condition, GroundAction
    from pyplanner.grounding.numeric import Const, NumericAssignment, NumericCondition, Var
    from pyplanner.loader.ast import Atom, FluentTerm

    action = GroundAction(0, "a", (), (), (0,), (), neg_facts=(1,))
    task = GroundTask(
        "d", "p", (Atom("p"), Atom("q")), (action,), State(0), (Condition((0,), (1,)),)
    )
    counter = GroundAction(
        0, "inc", (), (), (), (), numeric=(NumericAssignment(0, "increase", Const(1.0)),)
    )
    numeric_task = GroundTask(
        "d",
        "p",
        (),
        (counter,),
        State(0, (0.0,)),
        (Condition(numeric=(NumericCondition(Var(0), ">="),)),),
        variables=(FluentTerm("x"),),
    )
    for t in (task, numeric_task):
        compiled = CompiledRelaxation(RelaxedTask(t))
        compiled.goal_values([t.init], True)
        compiled.goal_values([t.init], False)
        compiled.relaxed_plans([t.init])


# --- Heurísticas --------------------------------------------------------------------


class _CompiledHeuristic(Heuristic):
    combine_max = False
    device = "numba"

    def __init__(self, task: GroundTask, relaxed: RelaxedTask | None = None) -> None:
        super().__init__(task)
        self.compiled = CompiledRelaxation(relaxed or RelaxedTask(task))


class CompiledHMaxHeuristic(_CompiledHeuristic):
    """h_max compilada con Numba."""

    name = "hmax"
    combine_max = True

    def evaluate_batch(self, states: Sequence[State]) -> list[Evaluation]:
        return [Evaluation(v) for v in self.compiled.goal_values(states, True)]


class CompiledHAddHeuristic(_CompiledHeuristic):
    """h_add compilada con Numba."""

    name = "hadd"

    def evaluate_batch(self, states: Sequence[State]) -> list[Evaluation]:
        return [Evaluation(v) for v in self.compiled.goal_values(states, False)]


class CompiledFFHeuristic(_CompiledHeuristic):
    """h_FF compilada con Numba, con acciones útiles."""

    name = "ff"
    provides_preferred = True

    def evaluate_batch(self, states: Sequence[State]) -> list[Evaluation]:
        return self.compiled.relaxed_plans(states)


# Las constantes de tipo de arista tienen que coincidir con las de la referencia.
assert (REPEAT_GE, REPEAT_GT, REPEAT_EQ, ONCE) == (0, 1, 2, 3)
