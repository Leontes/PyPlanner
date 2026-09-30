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
from dataclasses import dataclass

from pyplanner.grounding import GroundTask, ground
from pyplanner.loader.ast import Domain, Problem
from pyplanner.plan import Plan, simulate
from pyplanner.portfolio import PortfolioResult, run_portfolio
from pyplanner.search import SearchConfig, SearchResult, SearchStatus, search
from pyplanner.search.memory import available_memory_mb

#: Fracción de la memoria disponible que usa el planificador si no se le da un límite.
DEFAULT_MEMORY_FRACTION = 0.8


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


def solve(
    domain: Domain,
    problem: Problem,
    configs: Sequence[SearchConfig],
    time_limit: float | None = None,
    memory_limit: float | None = None,
) -> PlanningResult:
    """Resuelve una tarea con una configuración o, si hay varias, con un portfolio paralelo.

    Args:
        time_limit: límite de tiempo en segundos para cada búsqueda.
        memory_limit: memoria total (MB) que pueden usar las búsquedas; se reparte a
            partes iguales entre los miembros del portfolio. Por defecto, el 80% de la
            memoria disponible, para que una búsqueda larga nunca agote la de la máquina.
            Las configuraciones con ``max_memory`` propio lo conservan.

    Raises:
        InvalidPlanError: si el plan encontrado no supera la validación interna.
    """
    if not configs:
        raise ValueError("hace falta al menos una configuración de búsqueda")
    start = time.perf_counter()
    task = ground(domain, problem)
    grounding_time = time.perf_counter() - start

    if memory_limit is None:
        available = available_memory_mb()
        memory_limit = None if available is None else available * DEFAULT_MEMORY_FRACTION
    if memory_limit is not None:
        share = max(1, int(memory_limit / len(configs)))
        configs = [
            c if c.max_memory is not None else c.with_changes(max_memory=share) for c in configs
        ]

    portfolio = None
    if len(configs) == 1:
        config = configs[0]
        if time_limit is not None and config.max_time is None:
            config = config.with_changes(max_time=time_limit)
        result = search(task, config)
    else:
        portfolio = run_portfolio(task, configs, time_limit=time_limit)
        result = portfolio.best

    plan = None
    if result.solved:
        assert result.plan is not None
        error = simulate(task, result.plan)
        if error is not None:
            raise InvalidPlanError(f"{result.config_name} ha devuelto un plan inválido: {error}")
        plan = Plan.from_indices(task, result.plan)
    return PlanningResult(result.status, plan, result, portfolio, task, grounding_time)
