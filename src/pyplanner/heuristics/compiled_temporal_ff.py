"""h_TFF compilada con Numba.

Es el mismo cálculo que :meth:`pyplanner.heuristics.temporal_ff.TemporalRelaxedPlan.
evaluate_reference`, sobre las mismas entradas y con el mismo resultado: un Dijkstra por
(instante, nivel) con ventanas, la cláusula del objetivo que antes se cumple y la extracción
del plan relajado. El montículo es binario con entradas obsoletas, como el de la referencia.
"""

from collections.abc import Sequence

import numpy as np
from numba import njit

from pyplanner.grounding import State
from pyplanner.heuristics.base import DEAD_END, Evaluation
from pyplanner.heuristics.compiled_deadlines import _fit
from pyplanner.heuristics.temporal_ff import TemporalRelaxedPlan

INF = np.inf


def relaxed_plans(plan: TemporalRelaxedPlan, states: Sequence[State]) -> list[Evaluation]:
    """h_TFF (valor y acciones útiles) de cada estado."""
    times, pointers, starts, ends, usable, multipliers, window_true = plan.deadlines.plan_inputs(
        states
    )
    values, preferred_ptr, preferred = _relaxed_plans(
        times, pointers, starts, ends, usable, multipliers, window_true,
        *plan.deadlines.arrays, *plan.arrays, plan.axiom_operator, plan.num_operators,
    )  # fmt: skip
    results = []
    for row, value in enumerate(values.tolist()):
        if value == DEAD_END:
            results.append(Evaluation(DEAD_END))
        else:
            actions = preferred[preferred_ptr[row] : preferred_ptr[row + 1]]
            results.append(Evaluation(value, tuple(actions.tolist())))
    return results


@njit(cache=True, inline="always")
def _before(t1, l1, f1, t2, l2, f2):
    return t1 < t2 or (t1 == t2 and (l1 < l2 or (l1 == l2 and f1 < f2)))


@njit(cache=True, inline="always")
def _push(heap_time, heap_level, heap_fact, size, t, lev, f):
    i = size
    while i > 0:
        parent = (i - 1) >> 1
        if _before(heap_time[parent], heap_level[parent], heap_fact[parent], t, lev, f):
            break
        heap_time[i], heap_level[i], heap_fact[i] = (
            heap_time[parent], heap_level[parent], heap_fact[parent],
        )  # fmt: skip
        i = parent
    heap_time[i], heap_level[i], heap_fact[i] = t, lev, f
    return size + 1


@njit(cache=True, inline="always")
def _pop(heap_time, heap_level, heap_fact, size):
    top = (heap_time[0], heap_level[0], heap_fact[0])
    size -= 1
    if size > 0:
        t, lev, f = heap_time[size], heap_level[size], heap_fact[size]
        i = 0
        while True:
            child = 2 * i + 1
            if child >= size:
                break
            right = child + 1
            if right < size and _before(
                heap_time[right], heap_level[right], heap_fact[right],
                heap_time[child], heap_level[child], heap_fact[child],
            ):  # fmt: skip
                child = right
            if _before(t, lev, f, heap_time[child], heap_level[child], heap_fact[child]):
                break
            heap_time[i], heap_level[i], heap_fact[i] = (
                heap_time[child], heap_level[child], heap_fact[child],
            )  # fmt: skip
            i = child
        heap_time[i], heap_level[i], heap_fact[i] = t, lev, f
    return top[0], top[1], top[2], size


@njit(cache=True)
def _relaxed_plans(
    times, all_pointers, starts, ends, all_usable, all_multipliers, all_window_true,
    counted_len, pre_of_ptr, pre_of_action, pre_of_weight, effect_ptr, effect_fact,
    effect_edge, release, span, window_ptr, window_idx, span_ptr, span_idx, goal_ptr,
    goal_fact, goal_window_ptr, goal_window_idx, goal_of_ptr, goal_of_clause, counted_ptr,
    counted_fact, operator, operator_cost, axiom_operator, num_operators,
):  # fmt: skip
    batch, num_facts = times.shape
    num_actions = counted_len.shape[0]
    num_clauses = goal_ptr.shape[0] - 1
    values = np.empty(batch)
    preferred_ptr = np.zeros(batch + 1, dtype=np.int64)
    preferred_rows = []
    capacity = 2 * num_facts + effect_fact.shape[0] + 1
    heap_time = np.empty(capacity)
    heap_level = np.empty(capacity, dtype=np.int64)
    heap_fact = np.empty(capacity, dtype=np.int64)
    level = np.empty(num_facts, dtype=np.int64)
    supporter = np.empty(num_facts, dtype=np.int64)
    repeats_of = np.empty(num_facts)
    unsatisfied = np.empty(num_actions, dtype=np.int64)
    accumulated = np.empty(num_actions)
    action_level = np.empty(num_actions, dtype=np.int64)
    pending_goal = np.empty(num_clauses, dtype=np.int64)
    done = np.empty(num_facts, dtype=np.bool_)
    marked = np.zeros(num_facts, dtype=np.bool_)
    repeats = np.full(num_actions, -1.0)
    per_operator = np.zeros(num_operators)
    stack = np.empty(num_facts + counted_fact.shape[0] + 1, dtype=np.int64)  # con repetidos
    selected = np.empty(num_actions, dtype=np.int64)
    never = num_facts + 1
    for row in range(batch):
        initial = times[row]
        time = initial.copy()
        pointers = all_pointers[row]
        usable = all_usable[row]
        multipliers = all_multipliers[row]
        window_true = all_window_true[row]
        for f in range(num_facts):
            level[f] = 0 if time[f] < INF else never
        supporter[:] = -1
        repeats_of[:] = 1.0
        unsatisfied[:] = counted_len
        accumulated[:] = 0.0
        action_level[:] = 0
        done[:] = False
        best_moment = INF
        best_clause = -1
        for c in range(num_clauses):
            pending_goal[c] = goal_ptr[c + 1] - goal_ptr[c]
            if pending_goal[c] == 0:
                moment = _fit(
                    0.0, goal_window_idx, goal_window_ptr[c], goal_window_ptr[c + 1], 0.0,
                    pointers, starts, ends,
                )  # fmt: skip
                if moment < INF and (
                    moment < best_moment or (moment == best_moment and c < best_clause)
                ):
                    best_moment, best_clause = moment, c
        size = 0
        for f in range(num_facts):
            if time[f] < INF:
                size = _push(heap_time, heap_level, heap_fact, size, time[f], 0, f)
        for a in range(num_actions):
            if unsatisfied[a] == 0:
                size = _trigger(
                    a, accumulated, action_level, release, span, window_ptr, window_idx,
                    span_ptr, span_idx, pointers, starts, ends, effect_ptr, effect_fact,
                    effect_edge, usable, multipliers, time, level, supporter, repeats_of,
                    heap_time, heap_level, heap_fact, size,
                )  # fmt: skip
        while size > 0 and heap_time[0] <= best_moment:
            t, lev, f, size = _pop(heap_time, heap_level, heap_fact, size)
            if done[f] or t != time[f] or lev != level[f]:
                continue
            done[f] = True
            for j in range(goal_of_ptr[f], goal_of_ptr[f + 1]):
                c = goal_of_clause[j]
                pending_goal[c] -= 1
                if pending_goal[c] == 0:
                    moment = 0.0
                    for k in range(goal_ptr[c], goal_ptr[c + 1]):
                        moment = max(moment, time[goal_fact[k]])
                    moment = _fit(
                        moment, goal_window_idx, goal_window_ptr[c], goal_window_ptr[c + 1],
                        0.0, pointers, starts, ends,
                    )  # fmt: skip
                    if moment < INF and (
                        moment < best_moment or (moment == best_moment and c < best_clause)
                    ):
                        best_moment, best_clause = moment, c
            for j in range(pre_of_ptr[f], pre_of_ptr[f + 1]):
                a = pre_of_action[j]
                value = t + pre_of_weight[j]
                if value > accumulated[a]:
                    accumulated[a] = value
                if lev > action_level[a]:
                    action_level[a] = lev
                unsatisfied[a] -= 1
                if unsatisfied[a] == 0:
                    size = _trigger(
                        a, accumulated, action_level, release, span, window_ptr, window_idx,
                        span_ptr, span_idx, pointers, starts, ends, effect_ptr, effect_fact,
                        effect_edge, usable, multipliers, time, level, supporter, repeats_of,
                        heap_time, heap_level, heap_fact, size,
                    )  # fmt: skip
        if best_clause < 0:
            values[row] = INF
            preferred_rows.append(np.empty(0, dtype=np.int64))
            continue

        # Plan relajado: desde los hechos de la cláusula que no son ciertos en el estado.
        top = 0
        for k in range(goal_ptr[best_clause], goal_ptr[best_clause + 1]):
            g = goal_fact[k]
            if initial[g] == INF:
                stack[top] = g
                top += 1
        count = 0
        while top > 0:
            top -= 1
            f = stack[top]
            if marked[f]:
                continue
            marked[f] = True
            best = supporter[f]
            if repeats[best] < 0.0:
                repeats[best] = repeats_of[f]
                selected[count] = best
                count += 1
                for j in range(counted_ptr[best], counted_ptr[best + 1]):
                    p = counted_fact[j]
                    if initial[p] == INF and not marked[p]:
                        stack[top] = p
                        top += 1
            elif repeats_of[f] > repeats[best]:
                repeats[best] = repeats_of[f]
        useful = np.zeros(num_operators, dtype=np.bool_)
        for i in range(count):
            a = selected[i]
            o = operator[a]
            if repeats[a] > per_operator[o]:
                per_operator[o] = repeats[a]
            if o != axiom_operator:
                ready = True
                for j in range(counted_ptr[a], counted_ptr[a + 1]):
                    if initial[counted_fact[j]] == INF:
                        ready = False
                        break
                if ready:
                    for j in range(window_ptr[a], window_ptr[a + 1]):
                        if not window_true[window_idx[j]]:
                            ready = False
                            break
                if ready:
                    useful[o] = True
        value = 0.0
        for o in range(num_operators):
            if per_operator[o] > 0.0:
                value += operator_cost[o] * per_operator[o]
        values[row] = value
        preferred_rows.append(np.flatnonzero(useful))
        # Se deja todo limpio para el siguiente estado.
        for i in range(count):
            a = selected[i]
            per_operator[operator[a]] = 0.0
            repeats[a] = -1.0
        marked[:] = False
    total = 0
    for row in range(batch):
        total += preferred_rows[row].shape[0]
        preferred_ptr[row + 1] = total
    preferred = np.empty(total, dtype=np.int64)
    for row in range(batch):
        preferred[preferred_ptr[row] : preferred_ptr[row + 1]] = preferred_rows[row]
    return values, preferred_ptr, preferred


@njit(cache=True, inline="always")
def _trigger(
    a, accumulated, action_level, release, span, window_ptr, window_idx, span_ptr, span_idx,
    pointers, starts, ends, effect_ptr, effect_fact, effect_edge, usable, multipliers, time,
    level, supporter, repeats_of, heap_time, heap_level, heap_fact, size,
):  # fmt: skip
    moment = max(release[a], accumulated[a])
    moment = _fit(moment, window_idx, window_ptr[a], window_ptr[a + 1], 0.0, pointers, starts, ends)
    if span_ptr[a] < span_ptr[a + 1]:
        moment = _fit(
            moment, span_idx, span_ptr[a], span_ptr[a + 1], span[a], pointers, starts, ends
        )
    if moment == INF:
        return size
    lev = action_level[a] + 1
    for j in range(effect_ptr[a], effect_ptr[a + 1]):
        q = effect_fact[j]
        edge = effect_edge[j]
        if edge >= 0 and not usable[edge]:
            continue
        m = 1.0 if edge < 0 else multipliers[edge]
        if moment < time[q] or (moment == time[q] and lev < level[q]):
            time[q] = moment
            level[q] = lev
            supporter[q] = a
            repeats_of[q] = m
            size = _push(heap_time, heap_level, heap_fact, size, moment, lev, q)
        elif moment == time[q] and lev == level[q] and a < supporter[q]:
            supporter[q] = a
            repeats_of[q] = m
    return size
