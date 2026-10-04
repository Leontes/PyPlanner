"""Dijkstra de los plazos de los TIL compilado con Numba.

Es el mismo cálculo que :meth:`pyplanner.heuristics.deadlines.DeadlineRelaxation.
reachable_reference`, sobre las mismas entradas y con el mismo resultado: instantes más
tempranos de cada hecho relajado, con ventanas, hasta que una cláusula del objetivo se
alcanza o se acaba el montículo. El montículo es binario con entradas obsoletas, como el de
la referencia.
"""

from collections.abc import Sequence

import numpy as np
from numba import njit

from pyplanner.grounding import State
from pyplanner.heuristics.deadlines import DeadlineRelaxation

INF = np.inf


def reachable(deadlines: DeadlineRelaxation, states: Sequence[State]) -> list[bool]:
    """Para cada estado, ``False`` si el objetivo no se puede alcanzar respetando los
    plazos."""
    times, pointers, starts, ends, usable = deadlines.inputs(states)
    result = _reachable(times, pointers, starts, ends, usable, *deadlines.arrays)
    return [bool(ok) for ok in result]


@njit(cache=True, inline="always")
def _fit(moment, nodes, first, last, span, pointers, starts, ends):
    """Primer instante ``≥ moment`` en el que ``[instante, instante + span]`` cabe en un
    intervalo de cada nodo ``nodes[first:last]`` (``inf`` si no hay ninguno)."""
    changed = True
    while changed and moment < INF:
        changed = False
        for j in range(first, last):
            w = nodes[j]
            fitted = INF
            for i in range(pointers[w], pointers[w + 1]):
                candidate = max(moment, starts[i])
                if candidate + span <= ends[i]:
                    fitted = candidate
                    break
            if fitted != moment:
                moment = fitted
                changed = True
    return moment


@njit(cache=True, inline="always")
def _push(heap_time, heap_fact, size, t, f):
    i = size
    while i > 0:
        parent = (i - 1) >> 1
        pt, pf = heap_time[parent], heap_fact[parent]
        if pt < t or (pt == t and pf < f):
            break
        heap_time[i], heap_fact[i] = pt, pf
        i = parent
    heap_time[i], heap_fact[i] = t, f
    return size + 1


@njit(cache=True, inline="always")
def _pop(heap_time, heap_fact, size):
    top_t, top_f = heap_time[0], heap_fact[0]
    size -= 1
    if size > 0:
        t, f = heap_time[size], heap_fact[size]
        i = 0
        while True:
            child = 2 * i + 1
            if child >= size:
                break
            right = child + 1
            if right < size and (
                heap_time[right] < heap_time[child]
                or (heap_time[right] == heap_time[child] and heap_fact[right] < heap_fact[child])
            ):
                child = right
            ct, cf = heap_time[child], heap_fact[child]
            if t < ct or (t == ct and f < cf):
                break
            heap_time[i], heap_fact[i] = ct, cf
            i = child
        heap_time[i], heap_fact[i] = t, f
    return top_t, top_f, size


@njit(cache=True, inline="always")
def _trigger(
    a, accumulated, release, span, window_ptr, window_idx, span_ptr, span_idx, pointers,
    starts, ends, effect_ptr, effect_fact, effect_edge, usable, time, heap_time, heap_fact,
    size,
):  # fmt: skip
    moment = max(release[a], accumulated[a])
    moment = _fit(moment, window_idx, window_ptr[a], window_ptr[a + 1], 0.0, pointers, starts, ends)
    if span_ptr[a] < span_ptr[a + 1]:
        moment = _fit(
            moment, span_idx, span_ptr[a], span_ptr[a + 1], span[a], pointers, starts, ends
        )
    if moment == INF:
        return size
    for j in range(effect_ptr[a], effect_ptr[a + 1]):
        q = effect_fact[j]
        edge = effect_edge[j]
        if (edge < 0 or usable[edge]) and moment < time[q]:
            time[q] = moment
            size = _push(heap_time, heap_fact, size, moment, q)
    return size


@njit(cache=True)
def _reachable(
    times, all_pointers, starts, ends, all_usable, counted_len, pre_of_ptr, pre_of_action,
    pre_of_weight, effect_ptr, effect_fact, effect_edge, release, span, window_ptr,
    window_idx, span_ptr, span_idx, goal_ptr, goal_fact, goal_window_ptr, goal_window_idx,
    goal_of_ptr, goal_of_clause,
):  # fmt: skip
    batch, num_facts = times.shape
    num_actions = counted_len.shape[0]
    num_clauses = goal_ptr.shape[0] - 1
    result = np.zeros(batch, dtype=np.bool_)
    # Montículo y, al final del array de hechos, la lista de nodos ciertos desde el 0.
    capacity = 2 * num_facts + effect_fact.shape[0] + 1
    heap_time = np.empty(capacity)
    heap_fact = np.empty(capacity, dtype=np.int64)
    unsatisfied = np.empty(num_actions, dtype=np.int64)
    accumulated = np.empty(num_actions)
    pending_goal = np.empty(num_clauses, dtype=np.int64)
    done = np.empty(num_facts, dtype=np.bool_)
    for row in range(batch):
        time = times[row].copy()
        pointers = all_pointers[row]
        usable = all_usable[row]
        unsatisfied[:] = counted_len
        accumulated[:] = 0.0
        done[:] = False
        reached = False
        for c in range(num_clauses):
            pending_goal[c] = goal_ptr[c + 1] - goal_ptr[c]
            if pending_goal[c] == 0:
                moment = _fit(
                    0.0, goal_window_idx, goal_window_ptr[c], goal_window_ptr[c + 1], 0.0,
                    pointers, starts, ends,
                )  # fmt: skip
                if moment < INF:
                    reached = True
        if reached:
            result[row] = True
            continue
        # Los nodos ciertos desde el instante 0 (casi todos) van primero, sin pasar por el
        # montículo: se extraen antes que cualquier otro, y el resultado no depende del
        # orden entre empates.
        size = 0
        zero = 0
        for f in range(num_facts):
            if time[f] == 0.0:
                heap_fact[capacity - 1 - zero] = f
                zero += 1
            elif time[f] < INF:
                size = _push(heap_time, heap_fact, size, time[f], f)
        for a in range(num_actions):
            if unsatisfied[a] == 0:
                size = _trigger(
                    a, accumulated, release, span, window_ptr, window_idx, span_ptr, span_idx,
                    pointers, starts, ends, effect_ptr, effect_fact, effect_edge, usable, time,
                    heap_time, heap_fact, size,
                )  # fmt: skip
        while (zero > 0 or size > 0) and not reached:
            if zero > 0:
                zero -= 1
                t, f = 0.0, heap_fact[capacity - 1 - zero]
            else:
                t, f, size = _pop(heap_time, heap_fact, size)
            if done[f] or t != time[f]:
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
                    if moment < INF:
                        reached = True
            if reached:
                break
            for j in range(pre_of_ptr[f], pre_of_ptr[f + 1]):
                a = pre_of_action[j]
                value = t + pre_of_weight[j]
                if value > accumulated[a]:
                    accumulated[a] = value
                unsatisfied[a] -= 1
                if unsatisfied[a] == 0:
                    size = _trigger(
                        a, accumulated, release, span, window_ptr, window_idx, span_ptr,
                        span_idx, pointers, starts, ends, effect_ptr, effect_fact,
                        effect_edge, usable, time, heap_time, heap_fact, size,
                    )  # fmt: skip
        result[row] = reached
    return result
