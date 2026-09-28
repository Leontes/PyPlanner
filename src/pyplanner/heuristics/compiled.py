"""Implementación compilada con Numba de h_max, h_add y h_FF (dispositivo ``"numba"``).

Es la misma relajación y el mismo algoritmo que la implementación de referencia
(:mod:`pyplanner.heuristics.relaxation`), compilados a código máquina con Numba:

* un Dijkstra generalizado con montículo propio sobre arrays, que se detiene al extraer
  todos los hechos del objetivo y anota el mejor soportador de cada hecho (la acción
  relajada de menor índice con coste mínimo);
* la extracción del plan relajado de h_FF sobre la cláusula del objetivo de menor valor.

Las reglas de desempate son las de la referencia, así que los valores y las acciones útiles
son **idénticos**; los tests lo comprueban. En slitherlink p02, h_FF es unas 10 veces más
rápida y la búsqueda completa unas 6 veces (ver ``docs/motor-busqueda.md``).

El lote entero se evalúa en una sola llamada: los estados llegan como una matriz de bytes
(los bitsets) y se decodifican dentro del código compilado.

Este módulo solo se importa si Numba está instalado (``pip install -e ".[fast]"``). La
primera compilación tarda un par de segundos; después se guarda en la caché de disco.
"""

from collections.abc import Sequence

import numpy as np
from numba import njit

from pyplanner.grounding import GroundTask, State
from pyplanner.heuristics.base import DEAD_END, Evaluation, Heuristic
from pyplanner.heuristics.relaxation import RelaxedTask

INF = np.inf


class CompiledRelaxation:
    """Una :class:`~pyplanner.heuristics.relaxation.RelaxedTask` en arrays de NumPy (CSR)."""

    def __init__(self, relaxed: RelaxedTask) -> None:
        self.relaxed = relaxed
        self.num_base_facts = len(relaxed.task.facts)
        self.num_bytes = max(1, (self.num_base_facts + 7) // 8)
        self.pre_ptr, self.pre_idx = _csr(relaxed.pre)
        self.add_ptr, self.add_idx = _csr(relaxed.add)
        self.pre_of_ptr, self.pre_of_idx = _csr(relaxed.pre_of)
        self.goal_ptr, self.goal_idx = _csr(relaxed.goal)
        self.cost = np.array(relaxed.cost, dtype=np.float64)
        self.operator = np.array(relaxed.operator, dtype=np.int64)
        self.operator_cost = np.array(relaxed.operator_cost, dtype=np.float64)
        self.pre_len = np.array([len(p) for p in relaxed.pre], dtype=np.int64)
        self.no_pre = np.array(relaxed.no_pre, dtype=np.int64)
        self.negated = np.array(relaxed.negated, dtype=np.int64)

    def states_to_bytes(self, states: Sequence[State]) -> np.ndarray:
        """Matriz lote × bytes con los bitsets de los estados (orden *little endian*)."""
        raw = b"".join(s.to_bytes(self.num_bytes, "little") for s in states)
        return np.frombuffer(raw, dtype=np.uint8).reshape(len(states), self.num_bytes)

    def _arrays(self) -> tuple[object, ...]:
        return (
            self.num_base_facts,
            self.relaxed.num_facts,
            self.negated,
            self.pre_len,
            self.pre_of_ptr,
            self.pre_of_idx,
            self.add_ptr,
            self.add_idx,
            self.cost,
            self.no_pre,
            self.goal_ptr,
            self.goal_idx,
        )

    def goal_values(self, states: Sequence[State], combine_max: bool) -> list[float]:
        """Valor h_max o h_add de cada estado."""
        values = _goal_values(self.states_to_bytes(states), combine_max, *self._arrays())
        return [DEAD_END if v == INF else float(v) for v in values]

    def relaxed_plans(self, states: Sequence[State]) -> list[Evaluation]:
        """h_FF (valor y acciones útiles) de cada estado."""
        values, preferred, pointers = _relaxed_plans(
            self.states_to_bytes(states),
            *self._arrays(),
            self.pre_ptr,
            self.pre_idx,
            self.operator,
            self.operator_cost,
            len(self.relaxed.task.actions),
        )
        flat = preferred.tolist()
        bounds = pointers.tolist()
        return [
            Evaluation(DEAD_END)
            if value == INF
            else Evaluation(value, tuple(flat[bounds[row] : bounds[row + 1]]))
            for row, value in enumerate(values.tolist())
        ]


def _csr(lists: Sequence[Sequence[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Lista de listas → formato CSR (punteros de inicio y valores concatenados)."""
    pointers = np.zeros(len(lists) + 1, dtype=np.int64)
    pointers[1:] = np.cumsum([len(items) for items in lists])
    values = np.array([x for items in lists for x in items], dtype=np.int64)
    return pointers, values


# --- Núcleos compilados -------------------------------------------------------------
# Solo tipos simples y arrays de NumPy; nada de objetos de Python.


@njit(cache=True)
def _heap_push(keys, values, size, key, value):
    i = size
    keys[i] = key
    values[i] = value
    while i > 0:
        parent = (i - 1) >> 1
        if keys[parent] < keys[i] or (keys[parent] == keys[i] and values[parent] <= values[i]):
            break
        keys[parent], keys[i] = keys[i], keys[parent]
        values[parent], values[i] = values[i], values[parent]
        i = parent
    return size + 1


@njit(cache=True)
def _heap_pop(keys, values, size):
    key = keys[0]
    value = values[0]
    size -= 1
    keys[0] = keys[size]
    values[0] = values[size]
    i = 0
    while True:
        left = 2 * i + 1
        smallest = i
        for child in (left, left + 1):
            if child < size and (
                keys[child] < keys[smallest]
                or (keys[child] == keys[smallest] and values[child] < values[smallest])
            ):
                smallest = child
        if smallest == i:
            break
        keys[smallest], keys[i] = keys[i], keys[smallest]
        values[smallest], values[i] = values[i], values[smallest]
        i = smallest
    return key, value, size


@njit(cache=True)
def _dijkstra(
    state,
    combine_max,
    num_base,
    num_facts,
    negated,
    pre_len,
    pre_of_ptr,
    pre_of_idx,
    add_ptr,
    add_idx,
    cost,
    no_pre,
    goal_ptr,
    goal_idx,
):
    """Costes de los hechos relajados y mejor soportador de cada uno."""
    fact_cost = np.full(num_facts, INF)
    supporter = np.full(num_facts, -1, dtype=np.int64)
    done = np.zeros(num_facts, dtype=np.bool_)
    is_goal = np.zeros(num_facts, dtype=np.bool_)
    for g in goal_idx:
        is_goal[g] = True
    capacity = num_facts + add_idx.shape[0] + 1
    keys = np.empty(capacity)
    values = np.empty(capacity, dtype=np.int64)
    size = 0
    # Hechos ciertos: los del estado y los complementarios de los hechos negables falsos.
    for f in range(num_base):
        if (state[f >> 3] >> (f & 7)) & 1:
            fact_cost[f] = 0.0
            size = _heap_push(keys, values, size, 0.0, f)
    for i in range(negated.shape[0]):
        f = negated[i]
        if not (state[f >> 3] >> (f & 7)) & 1:
            fact_cost[num_base + i] = 0.0
            size = _heap_push(keys, values, size, 0.0, num_base + i)
    unsatisfied = pre_len.copy()
    accumulated = np.zeros(pre_len.shape[0])
    for a in no_pre:
        for j in range(add_ptr[a], add_ptr[a + 1]):
            q = add_idx[j]
            if cost[a] < fact_cost[q]:
                fact_cost[q] = cost[a]
                supporter[q] = a
                size = _heap_push(keys, values, size, cost[a], q)
            elif cost[a] == fact_cost[q] and a < supporter[q]:
                supporter[q] = a
    pending = 0
    for f in range(num_facts):
        if is_goal[f] and fact_cost[f] != 0.0:
            pending += 1
    while size > 0 and pending > 0:
        c, f, size = _heap_pop(keys, values, size)
        if done[f]:
            continue
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
            if unsatisfied[a] == 0:
                value = cost[a] + accumulated[a]
                for k in range(add_ptr[a], add_ptr[a + 1]):
                    q = add_idx[k]
                    if value < fact_cost[q]:
                        fact_cost[q] = value
                        supporter[q] = a
                        size = _heap_push(keys, values, size, value, q)
                    elif value == fact_cost[q] and a < supporter[q]:
                        supporter[q] = a
    return fact_cost, supporter


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
def _goal_values(
    states,
    combine_max,
    num_base,
    num_facts,
    negated,
    pre_len,
    pre_of_ptr,
    pre_of_idx,
    add_ptr,
    add_idx,
    cost,
    no_pre,
    goal_ptr,
    goal_idx,
):
    result = np.empty(states.shape[0])
    for row in range(states.shape[0]):
        fact_cost, _ = _dijkstra(
            states[row], combine_max, num_base, num_facts, negated, pre_len, pre_of_ptr,
            pre_of_idx, add_ptr, add_idx, cost, no_pre, goal_ptr, goal_idx,
        )  # fmt: skip
        _, result[row] = _best_clause(fact_cost, combine_max, goal_ptr, goal_idx)
    return result


@njit(cache=True)
def _relaxed_plans(
    states,
    num_base,
    num_facts,
    negated,
    pre_len,
    pre_of_ptr,
    pre_of_idx,
    add_ptr,
    add_idx,
    cost,
    no_pre,
    goal_ptr,
    goal_idx,
    pre_ptr,
    pre_idx,
    operator,
    operator_cost,
    num_operators,
):
    batch = states.shape[0]
    num_actions = pre_len.shape[0]
    values = np.empty(batch)
    # Acciones útiles en formato compacto: las del estado ``row`` están en
    # preferred[preferred_ptr[row]:preferred_ptr[row + 1]], ordenadas. Un plan relajado no
    # tiene más acciones que hechos marcados, así que num_facts por estado basta.
    preferred = np.empty(batch * max(num_facts, 1), dtype=np.int64)
    preferred_ptr = np.zeros(batch + 1, dtype=np.int64)
    # Arrays de trabajo reutilizados entre estados: se limpian solo las posiciones tocadas.
    in_plan = np.zeros(num_actions, dtype=np.bool_)
    marked = np.zeros(num_facts, dtype=np.bool_)
    marked_list = np.empty(num_facts, dtype=np.int64)
    plan = np.empty(num_facts, dtype=np.int64)
    stack = np.empty(num_facts, dtype=np.int64)
    seen_operator = np.zeros(num_operators, dtype=np.int64)  # sello: fila + 1
    row_preferred = np.empty(num_facts, dtype=np.int64)
    for row in range(batch):
        preferred_ptr[row + 1] = preferred_ptr[row]
        fact_cost, supporter = _dijkstra(
            states[row], False, num_base, num_facts, negated, pre_len, pre_of_ptr,
            pre_of_idx, add_ptr, add_idx, cost, no_pre, goal_ptr, goal_idx,
        )  # fmt: skip
        clause, clause_value = _best_clause(fact_cost, False, goal_ptr, goal_idx)
        if clause == -1 or clause_value == INF:
            values[row] = INF
            continue
        # Extracción hacia atrás desde los hechos pendientes de la cláusula elegida.
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
            best = supporter[stack[top]]
            if in_plan[best]:
                continue
            in_plan[best] = True
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
        # Cada operador cuenta una vez; es útil si alguna de sus acciones relajadas del
        # plan tiene todas las precondiciones ciertas.
        stamp = row + 1
        total = 0.0
        num_preferred = 0
        for i in range(num_plan):
            a = plan[i]
            o = operator[a]
            if seen_operator[o] != stamp and seen_operator[o] != -stamp:
                seen_operator[o] = stamp
                total += operator_cost[o]
            if seen_operator[o] == -stamp:
                continue  # ya está entre las útiles
            ready = True
            for j in range(pre_ptr[a], pre_ptr[a + 1]):
                if fact_cost[pre_idx[j]] != 0.0:
                    ready = False
                    break
            if ready:
                seen_operator[o] = -stamp
                row_preferred[num_preferred] = o
                num_preferred += 1
        values[row] = total
        ordered = np.sort(row_preferred[:num_preferred])
        start = preferred_ptr[row]
        preferred[start : start + num_preferred] = ordered
        preferred_ptr[row + 1] = start + num_preferred
        for i in range(num_plan):
            in_plan[plan[i]] = False
        for i in range(num_marked):
            marked[marked_list[i]] = False
    return values, preferred, preferred_ptr


def warm_up() -> None:
    """Compila (o carga de la caché de disco) todos los núcleos con una tarea mínima.

    El portfolio la llama en el proceso padre antes de lanzar los hijos: así la caché queda
    completa aunque un hijo se cancele a mitad de compilar, y con ``fork`` los hijos
    heredan los núcleos ya compilados en memoria.
    """
    from pyplanner.grounding import Condition, GroundAction
    from pyplanner.loader.ast import Atom

    action = GroundAction(0, "a", (), (), (0,), (), neg_facts=(1,))
    task = GroundTask("d", "p", (Atom("p"), Atom("q")), (action,), 0, (Condition((0,), (1,)),))
    compiled = CompiledRelaxation(RelaxedTask(task))
    compiled.goal_values([0], True)
    compiled.goal_values([0], False)
    compiled.relaxed_plans([0])


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
