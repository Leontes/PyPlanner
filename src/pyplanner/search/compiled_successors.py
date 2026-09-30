"""Núcleo compilado con Numba de la generación de sucesores.

Ver :mod:`pyplanner.search.successors`. Este módulo solo se importa si Numba está
instalado.

Las precondiciones numéricas (lineales) se evalúan en el mismo orden que
:meth:`pyplanner.grounding.numeric.Linear.evaluate` (constante y después los términos en
orden), así que el resultado coincide con la referencia incluso en las igualdades. Un valor
indefinido llega como NaN, y cualquier comparación con NaN es falsa, igual que con
``None`` en la referencia.
"""

import numpy as np
from numba import njit

#: Códigos de los comparadores canónicos.
GE, GT, EQ = 0, 1, 2


@njit(cache=True)
def applicable(
    state,
    values,
    pre,
    neg,
    always,
    trigger_ptr,
    trigger_idx,
    cond_ptr,
    term_ptr,
    term_var,
    term_weight,
    constant,
    comparator,
    out,
):
    """Escribe en ``out`` los índices de las acciones aplicables y devuelve cuántas son.

    Orden canónico: primero las acciones sin precondiciones positivas; después, las de cada
    hecho cierto del estado (en orden creciente) cuya primera precondición es ese hecho.

    Primero se reúnen las candidatas en ese orden y después se filtran en el sitio. La
    comprobación va escrita aquí y no en una función aparte: pasar tantos arrays en cada
    llamada multiplicaba el tiempo.
    """
    count = 0
    for a in always:
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
                    out[count] = trigger_idx[j]
                    count += 1
    kept = 0
    for k in range(count):
        a = out[k]
        ok = True
        for w in range(state.shape[0]):
            if (pre[a, w] & state[w]) != pre[a, w] or (neg[a, w] & state[w]) != 0:
                ok = False
                break
        if ok:
            for c in range(cond_ptr[a], cond_ptr[a + 1]):
                total = constant[c]
                for t in range(term_ptr[c], term_ptr[c + 1]):
                    total += term_weight[t] * values[term_var[t]]
                kind = comparator[c]
                if kind == 0:
                    ok = total >= 0.0
                elif kind == 1:
                    ok = total > 0.0
                else:
                    ok = total == 0.0
                if not ok:
                    break
        if ok:
            out[kept] = a
            kept += 1
    return kept


@njit(cache=True)
def applicable_facts(state, pre, neg, always, trigger_ptr, trigger_idx, out):
    """Como :func:`applicable`, para tareas sin precondiciones numéricas: con menos
    argumentos, la llamada desde Python es más barata (Numba desempaqueta cada uno)."""
    count = 0
    for a in always:
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
                    out[count] = trigger_idx[j]
                    count += 1
    kept = 0
    for k in range(count):
        a = out[k]
        ok = True
        for w in range(state.shape[0]):
            if (pre[a, w] & state[w]) != pre[a, w] or (neg[a, w] & state[w]) != 0:
                ok = False
                break
        if ok:
            out[kept] = a
            kept += 1
    return kept


def warm_up() -> None:
    """Compila (o carga de la caché de disco) los núcleos con datos mínimos."""
    state = np.zeros(1, dtype=np.uint64)
    masks = np.zeros((1, 1), dtype=np.uint64)
    empty = np.zeros(0, dtype=np.int64)
    pointers = np.zeros(2, dtype=np.int64)
    applicable(
        state,
        np.zeros(0),
        masks,
        masks,
        empty,
        pointers,
        empty,
        pointers,
        np.zeros(1, dtype=np.int64),
        empty,
        np.zeros(0),
        np.zeros(0),
        empty,
        np.empty(1, np.int64),
    )
    applicable_facts(state, masks, masks, empty, pointers, empty, np.empty(1, np.int64))
