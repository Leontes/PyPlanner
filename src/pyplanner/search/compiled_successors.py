"""Núcleo compilado con Numba de la generación de sucesores.

Ver :mod:`pyplanner.search.successors`. Este módulo solo se importa si Numba está
instalado.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def _holds(a, state, pre, neg):
    for w in range(state.shape[0]):
        if (pre[a, w] & state[w]) != pre[a, w] or (neg[a, w] & state[w]) != 0:
            return False
    return True


@njit(cache=True)
def applicable(state, pre, neg, always, trigger_ptr, trigger_idx, out):
    """Escribe en ``out`` los índices de las acciones aplicables y devuelve cuántas son.

    Orden canónico: primero las acciones sin precondiciones positivas; después, las de cada
    hecho cierto del estado (en orden creciente) cuya primera precondición es ese hecho.
    """
    count = 0
    for a in always:
        if _holds(a, state, pre, neg):
            out[count] = a
            count += 1
    num_facts = trigger_ptr.shape[0] - 1
    for w in range(state.shape[0]):
        word = state[w]
        if word == 0:
            continue
        for b in range(64):
            if (word >> np.uint64(b)) & np.uint64(1):
                fact = 64 * w + b
                if fact >= num_facts:
                    break
                for j in range(trigger_ptr[fact], trigger_ptr[fact + 1]):
                    a = trigger_idx[j]
                    if _holds(a, state, pre, neg):
                        out[count] = a
                        count += 1
    return count


def warm_up() -> None:
    """Compila (o carga de la caché de disco) el núcleo con datos mínimos."""
    state = np.zeros(1, dtype=np.uint64)
    masks = np.zeros((1, 1), dtype=np.uint64)
    empty = np.zeros(0, dtype=np.int64)
    applicable(
        state, masks, masks, empty, np.zeros(2, dtype=np.int64), empty, np.empty(1, np.int64)
    )
