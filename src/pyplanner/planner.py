"""Fachada del planificador: de un dominio y un problema a un plan.

Encadena el grounding, la búsqueda (una configuración o un portfolio) y la validación
interna del plan. Es el punto de entrada para usar el planificador como librería::

    from pyplanner.frontend import read_domain, read_problem
    from pyplanner.planner import solve
    from pyplanner.search import PRESETS

    domain = read_domain("domain.pddl")
    result = solve(domain, read_problem("p01.pddl", domain), [PRESETS["gbfs-ff-pref"]])
    if result.plan is not None:
        print(result.plan.to_ipc())
"""

import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace

from pyplanner.grounding import GroundTask, ground
from pyplanner.grounding.discrete import DiscreteTask
from pyplanner.loader.ast import Domain, Problem
from pyplanner.plan import Plan, simulate
from pyplanner.portfolio import PortfolioResult, run_portfolio
from pyplanner.search import SearchConfig, SearchResult, SearchStatus, search
from pyplanner.search.anytime import search_anytime
from pyplanner.search.memory import available_memory_mb

#: Fracción de la memoria disponible que usa el planificador si no se le da un límite.
DEFAULT_MEMORY_FRACTION = 0.8

#: Límite de tiempo (s) de la búsqueda anytime de las tareas con preferencias si no se da
#: otro: siempre hay plan mejor que buscar, así que hace falta un final.
DEFAULT_ANYTIME_LIMIT = 60.0

#: Con procesos y eventos, el plan se vuelve a simular con un paso de tiempo este número de
#: veces más fino.
VALIDATION_FACTOR = 10
#: Veces que se vuelve a buscar con la mitad de paso si el plan no supera esa validación.
MAX_REFINEMENTS = 3


def available_cores() -> int:
    """Núcleos que puede usar este proceso (respeta los límites de WSL, contenedores, etc.)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - sistemas sin sched_getaffinity
        return os.cpu_count() or 1


class InvalidPlanError(RuntimeError):
    """La búsqueda ha devuelto un plan que no es válido. Indica un error del planificador."""


@dataclass
class PlanningResult:
    """Resultado completo de :func:`solve`.

    Attributes:
        status: cómo terminó la búsqueda (la del ganador, en un portfolio).
        plan: el plan, si se ha encontrado.
        search: resultado de la búsqueda que decide el estado.
        portfolio: resultados de todas las configuraciones, si se usó un portfolio.
        task: la tarea instanciada.
        grounding_time: tiempo del grounding, en segundos.
    """

    status: SearchStatus
    plan: Plan | None
    search: SearchResult
    portfolio: PortfolioResult | None
    task: GroundTask
    grounding_time: float


def default_memory_limit() -> float | None:
    """Memoria total por defecto para las búsquedas (MB): el :data:`DEFAULT_MEMORY_FRACTION`
    de la disponible, o ``None`` si no se puede medir."""
    available = available_memory_mb()
    return None if available is None else available * DEFAULT_MEMORY_FRACTION


def solve(
    domain: Domain,
    problem: Problem,
    configs: Sequence[SearchConfig],
    time_limit: float | None = None,
    memory_limit: float | None = None,
    delta: float | None = None,
) -> PlanningResult:
    """Resuelve una tarea con una configuración o, si hay varias, con un portfolio paralelo.

    Si la tarea tiene preferencias (PDDL 3), la búsqueda es *anytime*
    (:mod:`pyplanner.search.anytime`): devuelve el mejor plan que encuentre en
    ``time_limit`` segundos (por defecto, :data:`DEFAULT_ANYTIME_LIMIT`), o antes si
    demuestra que es óptimo.

    Si tiene procesos o eventos (PDDL+), el tiempo avanza en pasos de ``delta`` (1 por
    defecto; ver :mod:`pyplanner.grounding.discrete`). El plan se vuelve a simular con un
    paso :data:`VALIDATION_FACTOR` veces más fino; si falla, se busca otra vez con la mitad
    de paso, hasta :data:`MAX_REFINEMENTS` veces, dentro del límite de tiempo.

    Args:
        time_limit: límite de tiempo en segundos para cada búsqueda.
        memory_limit: memoria total (MB) que pueden usar las búsquedas; se reparte a
            partes iguales entre los miembros del portfolio. Por defecto, el 80% de la
            memoria disponible, para que una búsqueda larga nunca agote la de la máquina.
            Las configuraciones con ``max_memory`` propio lo conservan.
        delta: paso de tiempo de las tareas con procesos y eventos.

    Raises:
        InvalidPlanError: si el plan encontrado no supera la validación interna.
    """
    if not configs:
        raise ValueError("hace falta al menos una configuración de búsqueda")
    start = time.perf_counter()
    task = ground(domain, problem, delta)
    grounding_time = time.perf_counter() - start

    if memory_limit is None:
        memory_limit = default_memory_limit()
    if memory_limit is not None:
        share = max(1, int(memory_limit / len(configs)))
        configs = [
            c if c.max_memory is not None else c.with_changes(max_memory=share) for c in configs
        ]

    anytime = task.preferences is not None
    if anytime and time_limit is None:
        time_limit = DEFAULT_ANYTIME_LIMIT
    deadline = None if time_limit is None else time.perf_counter() + time_limit
    result, portfolio = _search(task, configs, time_limit, anytime)
    notes: list[str] = []
    refinements = 0
    while result.solved and isinstance(task, DiscreteTask):
        assert result.plan is not None
        fine = task.delta / VALIDATION_FACTOR
        failure = task.check(result.plan, fine)
        if failure is None:
            break
        remaining = None if deadline is None else deadline - time.perf_counter()
        if refinements == MAX_REFINEMENTS or (remaining is not None and remaining <= 0):
            result = replace(
                result,
                status=SearchStatus.FAILED,
                plan=None,
                cost=None,
                message=f"el plan no supera la validación con un paso más fino: {failure}",
            )
            break
        refinements += 1
        notes.append(
            f"el plan con δ = {task.delta:g} no supera la validación ({failure}): "
            f"se busca con δ = {task.delta / 2:g}"
        )
        task = ground(domain, problem, task.delta / 2)
        result, portfolio = _search(task, configs, remaining, anytime)
    if notes:
        task = replace(task, notes=(*task.notes, *notes))

    plan = None
    if result.solved:
        assert result.plan is not None
        error = simulate(task, result.plan)
        if error is not None:
            raise InvalidPlanError(f"{result.config_name} ha devuelto un plan inválido: {error}")
        plan = Plan.from_indices(task, result.plan)
    return PlanningResult(result.status, plan, result, portfolio, task, grounding_time)


def _search(
    task: GroundTask, configs: Sequence[SearchConfig], time_limit: float | None, anytime: bool
) -> tuple[SearchResult, PortfolioResult | None]:
    """Una configuración en este proceso, o un portfolio."""
    if len(configs) == 1:
        config = configs[0]
        if time_limit is not None and (config.max_time is None or config.max_time > time_limit):
            config = config.with_changes(max_time=max(time_limit, 1e-9))
        return (search_anytime(task, config) if anytime else search(task, config)), None
    portfolio = run_portfolio(task, configs, time_limit=time_limit, anytime=anytime)
    return portfolio.best, portfolio
