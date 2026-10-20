"""Búsqueda *anytime*: mejora el plan mientras quede tiempo.

Con preferencias (PDDL 3), el primer plan rara vez es el mejor, y buscar el óptimo de una
vez no es viable en tareas grandes. Como LAMA (Richter y Westphal, 2010), se encadenan
búsquedas, cada una con la cota del mejor plan encontrado hasta entonces (solo busca planes
de coste menor; ver :func:`pyplanner.search.engine.search`):

1. la configuración elegida, con **coste unitario**: busca un plan cualquiera, cuanto
   antes. Con los costes reales, una búsqueda voraz persigue las preferencias y puede
   tardar mucho en terminar (en storage, sin objetivo duro, explora sin fin estados donde
   parecen alcanzables). El coste del plan se calcula después con los costes reales;
2. la configuración elegida con los costes reales y el **incumbente** del motor (ver
   :func:`pyplanner.search.engine.search`): cada estado expandido en el que se puede
   terminar es un plan, y va dejando planes cada vez mejores. Con los paseos aleatorios
   no: no terminan si no encuentran plan, así que con la cota se quedarían con todo el
   tiempo (en storage p02 del tier 5 acababan con métrica 10, y WA* llega a 1);
3. WA* con h_FF y acciones útiles, con pesos 5, 3, 2 y 1 (y la evaluación perezosa o
   inmediata, la exploración y el desempate de la elegida);
4. A* con h_max, que es admisible: si agota la cola, el mejor plan es óptimo.

Todas salvo la primera llevan la cota y el incumbente.

Tras encontrar un plan se pasa a la siguiente; la última se repite mientras encuentre
planes. Termina al agotar el tiempo, al demostrar que el mejor plan es óptimo o al
alcanzar otro límite (memoria, parada externa). Devuelve el mejor plan.

En el portfolio (:func:`pyplanner.portfolio.run_portfolio` con ``anytime=True``), cada
miembro empieza con su configuración, y todos comparten la mejor cota (``shared_bound``).
"""

import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any, Protocol

from pyplanner.grounding import GroundTask
from pyplanner.grounding.preferences import final_state
from pyplanner.heuristics import TENSOR_HEURISTICS, Heuristic, create_heuristic
from pyplanner.search.config import SearchConfig
from pyplanner.search.engine import BOUND_TOLERANCE, search
from pyplanner.search.result import SearchResult, SearchStatistics, SearchStatus, StopSignal

#: Pesos de las búsquedas WA* que siguen a la primera.
WEIGHTS = (5.0, 3.0, 2.0, 1.0)


class SharedBound(Protocol):
    """Cota compartida entre procesos (un ``multiprocessing.Value("d")``)."""

    value: float

    def get_lock(self) -> Any: ...


def anytime_schedule(first: SearchConfig) -> list[SearchConfig]:
    """Las búsquedas que se encadenan empezando por ``first`` (ver la documentación)."""
    device = first.device
    common: dict[str, Any] = {
        "lazy": first.lazy,
        "exploration": first.exploration,
        "tie_breaking": first.tie_breaking,
        "seed": first.seed,
        "max_memory": first.max_memory,
    }
    schedule = [first]
    if first.algorithm != "mrw":  # los paseos aleatorios no terminan sin plan
        schedule.append(first.with_changes(name=f"{first.name}+coste"))
    schedule += [
        SearchConfig(
            f"{first.name}+wastar{weight:g}",
            "wastar",
            "ff",
            weight=weight,
            preferred_operators=True,
            device=device if "ff" in TENSOR_HEURISTICS else "auto",
            **common,
        )
        for weight in WEIGHTS
    ]
    schedule.append(
        SearchConfig(
            f"{first.name}+astar-hmax",
            "astar",
            "hmax",
            device=device if "hmax" in TENSOR_HEURISTICS else "auto",
            max_memory=first.max_memory,
        )
    )
    return schedule


def search_anytime(
    task: GroundTask,
    config: SearchConfig,
    time_limit: float | None = None,
    stop: StopSignal | None = None,
    on_plan: Callable[[SearchResult], None] | None = None,
    shared_bound: SharedBound | None = None,
) -> SearchResult:
    """Busca planes cada vez mejores empezando por ``config`` (ver la documentación del
    módulo).

    Args:
        time_limit: tiempo total en segundos (por defecto, el ``max_time`` de ``config``).
        stop: señal de parada externa.
        on_plan: se llama con cada plan mejor que el anterior.
        shared_bound: cota compartida con otras búsquedas: cada una busca planes mejores
            que el mejor de todas, y la actualiza con los suyos.

    Returns:
        El mejor plan (con las estadísticas sumadas de todas las búsquedas y, si se ha
        demostrado, ``lower_bound``), o el resultado de la última búsqueda si no hay plan.
    """
    if time_limit is None:
        time_limit = config.max_time
    start = time.perf_counter()
    deadline = None if time_limit is None else start + time_limit
    schedule = anytime_schedule(config)
    unit = unit_cost(task)
    heuristics: dict[tuple[bool, str, str], Heuristic] = {}
    statistics = SearchStatistics()
    best: SearchResult | None = None
    qualities: list[float] = []
    lower_bound: float | None = None
    last: SearchResult | None = None

    def improved(result: SearchResult) -> None:
        """Un plan nuevo: si mejora el mejor, pasa a serlo."""
        nonlocal best
        assert result.cost is not None and result.plan is not None
        if best is not None and best.cost is not None and result.cost >= best.cost:
            return
        best = result
        qualities.append(_quality(task, result.plan, result.cost))
        if on_plan is not None:
            on_plan(result)
        if shared_bound is not None:
            with shared_bound.get_lock():
                shared_bound.value = min(shared_bound.value, result.cost)

    step = 0
    while stop is None or not stop.is_set():
        bound = _bound(best, shared_bound)
        if bound is not None and bound <= BOUND_TOLERANCE:
            lower_bound = bound  # ningún plan cuesta menos de 0
            break
        current = schedule[step]
        if deadline is not None:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            if current.max_time is None or current.max_time > remaining:
                current = current.with_changes(max_time=remaining)
        first = step == 0  # coste unitario, sin cota ni incumbente
        searched = unit if first else task
        key = (first, current.heuristic, current.device)
        if key not in heuristics:
            heuristics[key] = create_heuristic(current.heuristic, searched, current.device)
        last = search(
            searched,
            current,
            heuristics[key],
            stop,
            None if first else bound,
            None if first else improved,
        )
        statistics.add(last.statistics)
        if last.solved:
            assert last.plan is not None
            if first:
                last = replace(last, cost=path_cost(task, last.plan))
            improved(last)
        if last.lower_bound is not None:
            lower_bound = last.lower_bound  # búsqueda exhaustiva: no hay plan mejor
            break
        if last.interrupted:
            break  # límite de tiempo o de memoria, o parada externa
        if last.solved:
            step = min(step + 1, len(schedule) - 1)
            continue
        if last.status is SearchStatus.UNSOLVABLE and bound is None:
            break  # la tarea no tiene solución
        if last.status not in (SearchStatus.UNSOLVABLE, SearchStatus.FAILED):
            break  # límite de tiempo o de memoria, o parada externa
        if step == len(schedule) - 1:
            break
        step += 1

    statistics.time = time.perf_counter() - start
    if task.metric_time_weight:
        # La duración del plan está en la métrica pero no en los costes: que no haya un plan
        # de menor coste no demuestra que este sea óptimo.
        lower_bound = None
    if best is None:
        if last is None:  # no ha dado tiempo a empezar, o la cota compartida ya es 0
            status = SearchStatus.LIMIT_REACHED if lower_bound is None else SearchStatus.CANCELLED
            return SearchResult(
                status, statistics=statistics, config_name=config.name, lower_bound=lower_bound
            )
        status, message = last.status, last.message
        if status is SearchStatus.UNSOLVABLE and bound is not None:
            # Con la cota de otra búsqueda: no hay plan mejor, pero la tarea sí tiene.
            status, message = SearchStatus.FAILED, "ningún plan mejor que la cota compartida"
        return SearchResult(
            status,
            statistics=statistics,
            config_name=last.config_name,
            message=message,
            lower_bound=lower_bound,
        )
    result = SearchResult(
        SearchStatus.SOLVED,
        best.plan,
        best.cost,
        statistics,
        best.config_name,
        lower_bound=lower_bound,
    )
    progress = " → ".join(f"{q:g}" for q in qualities)
    result.message = f"anytime: {len(qualities)} plan(es), calidad {progress}" + (
        " (óptimo)" if result.optimal else ""
    )
    return result


def unit_cost(task: GroundTask) -> GroundTask:
    """La tarea con todas las acciones a coste 1 (para el primer plan)."""
    return task.with_unit_costs()


def path_cost(task: GroundTask, plan: tuple[int, ...]) -> float:
    """El coste de un plan en la tarea: sus acciones más las penalizaciones."""
    preferences = task.preferences
    penalty = preferences.penalty if preferences is not None and preferences.violations else None
    state, total = task.init, 0.0
    for index in plan:
        action = task.actions[index]
        successor = task.apply(state, action)
        assert successor is not None, "el plan no es aplicable"
        total += action.cost + (penalty(state, successor) if penalty is not None else 0.0)
        state = successor
    return total


def _bound(best: SearchResult | None, shared: SharedBound | None) -> float | None:
    bound = float("inf") if best is None or best.cost is None else best.cost
    if shared is not None:
        bound = min(bound, shared.value)
    return None if bound == float("inf") else bound


def _quality(task: GroundTask, plan: tuple[int, ...], cost: float) -> float:
    """La métrica del plan si las preferencias la definen (como la informa el plan: con su
    signo si se maximiza y, en una tarea temporal, con la duración); si no, su coste."""
    preferences = task.preferences
    if preferences is None:
        return round(cost, 6)
    from pyplanner.plan import Plan

    metric = Plan.from_indices(task, plan).metric
    if metric is None:
        metric = preferences.metric(task, plan, final_state(task, plan))
        metric += task.metric_offset or 0.0
    return round(metric, 6)
