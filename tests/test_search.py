"""Tests del motor de búsqueda y de su configuración."""

import subprocess
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import pytest

from pyplanner.frontend import read_domain, read_problem
from pyplanner.grounding import GroundTask, State, ground
from pyplanner.heuristics import Evaluation, Heuristic
from pyplanner.plan import simulate
from pyplanner.planner import solve
from pyplanner.search import (
    DEFAULT_CORES,
    DEFAULT_PORTFOLIO,
    PORTFOLIOS,
    PRESETS,
    ConfigError,
    SearchConfig,
    SearchStatus,
    portfolio_for_cores,
    search,
)
from tests.tasks import build, chain, fork, goal_state, token, unreachable
from tests.tiers import VALIDATE, tier_dir


def tier1(domain: str, problem: str) -> GroundTask:
    base = tier_dir(f"tier1/{domain}")
    loaded = read_domain(base / "domain.pddl")
    return ground(loaded, read_problem(base / f"{problem}.pddl", loaded))


GRIPPER = tier1("gripper", "p01")
BLOCKS = tier1("blocksworld", "p02")


@pytest.mark.parametrize("name", sorted(PRESETS))
@pytest.mark.parametrize(
    "task", [chain(), fork(), GRIPPER, BLOCKS], ids=["chain", "fork", "gripper", "blocks"]
)
def test_every_preset_finds_a_valid_plan(task: GroundTask, name: str) -> None:
    result = search(task, PRESETS[name])
    if not PRESETS[name].is_complete and result.status is SearchStatus.FAILED:
        return  # IW: problemas de más anchura que su k
    assert result.status is SearchStatus.SOLVED
    assert result.plan is not None
    assert simulate(task, result.plan) is None
    assert result.cost == len(result.plan)
    assert result.config_name == name


@pytest.mark.parametrize("batch_size", [1, 4, 64])
@pytest.mark.parametrize("name", ["astar-hmax", "astar-blind"])
def test_astar_is_optimal_even_with_batches(name: str, batch_size: int) -> None:
    config = PRESETS[name].with_changes(batch_size=batch_size)
    assert search(GRIPPER, config).cost == 11
    assert search(BLOCKS, config).cost == 10


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_unsolvable_task_is_proven(name: str) -> None:
    config = PRESETS[name]
    if config.algorithm == "mrw":
        # Los paseos aleatorios no pueden demostrarlo: terminan por el límite de tiempo.
        result = search(token(), config.with_changes(max_time=0.2))
        assert result.status is SearchStatus.LIMIT_REACHED and result.plan is None
        return
    result = search(token(), config)
    assert result.status is SearchStatus.UNSOLVABLE
    assert result.plan is None


def test_dead_end_initial_state() -> None:
    result = search(unreachable(), PRESETS["gbfs-ff"])
    assert result.status is SearchStatus.UNSOLVABLE
    assert result.statistics.expanded == 0


def test_initial_state_already_goal() -> None:
    task = chain()
    solved = GroundTask(
        task.domain_name, task.problem_name, task.facts, task.actions, goal_state(task), task.goal
    )
    for name in ("gbfs-ff", "lazy-ff-pref"):
        result = search(solved, PRESETS[name])
        assert result.status is SearchStatus.SOLVED
        assert result.plan == ()


def test_expansion_limit() -> None:
    config = PRESETS["astar-blind"].with_changes(max_expansions=3)
    result = search(GRIPPER, config)
    assert result.status is SearchStatus.LIMIT_REACHED
    assert result.statistics.expanded == 3


def test_time_limit() -> None:
    config = PRESETS["astar-blind"].with_changes(max_time=1e-9)
    assert search(GRIPPER, config).status is SearchStatus.LIMIT_REACHED


def test_stop_signal_cancels() -> None:
    stop = threading.Event()
    stop.set()
    result = search(GRIPPER, PRESETS["astar-blind"], stop=stop)
    assert result.status is SearchStatus.CANCELLED


def test_batches_reduce_evaluation_calls() -> None:
    single = search(GRIPPER, PRESETS["gbfs-add"])
    batched = search(GRIPPER, PRESETS["gbfs-add"].with_changes(batch_size=8))
    assert batched.solved and single.solved
    assert batched.statistics.evaluation_batches < single.statistics.evaluation_batches


def test_random_tie_breaking_is_reproducible() -> None:
    config = PRESETS["gbfs-goalcount-rnd"]
    first, second = search(GRIPPER, config), search(GRIPPER, config)
    assert first.plan == second.plan
    assert first.statistics.expanded == second.statistics.expanded


def test_lazy_search_evaluates_fewer_states() -> None:
    eager = search(GRIPPER, PRESETS["gbfs-ff-pref"])
    lazy = search(GRIPPER, PRESETS["lazy-ff-pref"])
    assert lazy.statistics.evaluated < eager.statistics.evaluated


def test_portfolios_by_cores() -> None:
    assert PORTFOLIOS[DEFAULT_CORES] == DEFAULT_PORTFOLIO
    for size, names in PORTFOLIOS.items():
        assert 1 <= len(names) <= size
        assert len(set(names)) == len(names)
        assert set(names) <= set(PRESETS)
    # Cada portfolio contiene al anterior: con más núcleos nunca se pierde una búsqueda.
    sizes = sorted(PORTFOLIOS)
    for smaller, larger in zip(sizes, sizes[1:], strict=False):
        assert set(PORTFOLIOS[smaller]) <= set(PORTFOLIOS[larger])


@pytest.mark.parametrize(
    ("cores", "size"), [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (64, 7)]
)
def test_portfolio_for_cores(cores: int, size: int) -> None:
    assert [c.name for c in portfolio_for_cores(cores)] == list(PORTFOLIOS[size])


def test_portfolio_for_zero_cores() -> None:
    with pytest.raises(ConfigError, match="≥ 1"):
        portfolio_for_cores(0)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"algorithm": "dfs"}, "algoritmo"),
        ({"heuristic": "magic"}, "heurística desconocida"),
        ({"device": "tpu"}, "dispositivo"),
        ({"heuristic": "goalcount", "device": "cuda"}, "PyTorch"),
        ({"tie_breaking": "coin"}, "desempate"),
        ({"algorithm": "wastar", "weight": 0.5}, "peso"),
        ({"heuristic": "hadd", "preferred_operators": True}, "acciones útiles"),
        ({"algorithm": "astar", "lazy": True}, "perezosa"),
        ({"batch_size": 0}, "batch_size"),
        ({"max_time": 0}, "max_time"),
        ({"max_expansions": -1}, "max_expansions"),
        ({"algorithm": "mrw", "lazy": True}, "paseos aleatorios"),
        ({"algorithm": "ehc", "exploration": "type-based"}, "paseos aleatorios"),
    ],
)
def test_invalid_configs(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        SearchConfig(**changes)  # type: ignore[arg-type]


def test_config_helpers() -> None:
    assert PRESETS["astar-hmax"].is_optimal
    assert not PRESETS["gbfs-ff"].is_optimal
    assert PRESETS["ehc-tff-pref"].is_complete and not PRESETS["mrw-tff"].is_complete
    assert PRESETS["gbfs-ff"].to_dict()["heuristic"] == "ff"


# --- Casos límite con un grafo y una heurística a medida -------------------------------

GRAPH_DOMAIN = """
(define (domain graph)
  (:predicates (at ?x) (edge ?x ?y))
  (:action move :parameters (?x ?y)
    :precondition (and (at ?x) (edge ?x ?y))
    :effect (and (at ?y) (not (at ?x)))))
"""


def graph(edges: list[tuple[str, str]], goal: str) -> GroundTask:
    nodes = sorted({n for edge in edges for n in edge})
    problem = (
        f"(define (problem g) (:domain graph) (:objects {' '.join(nodes)})"
        f" (:init (at s) {' '.join(f'(edge {a} {b})' for a, b in edges)})"
        f" (:goal (at {goal})))"
    )
    return build(GRAPH_DOMAIN, problem)


class TableHeuristic(Heuristic):
    """Heurística dada por una tabla: nodo del grafo → valor."""

    name = "table"

    def __init__(self, task: GroundTask, table: Mapping[str, float]) -> None:
        super().__init__(task)
        self.table = table

    def evaluate_batch(self, states: Sequence[State]) -> list[Evaluation]:
        result = []
        for state in states:
            [fact] = self.task.atoms(state)
            result.append(Evaluation(self.table[str(fact.args[0])]))
        return result


def test_astar_reopens_closed_nodes() -> None:
    # Camino largo s-a-a2-c y corto s-b-c. Con h(b) = 2 (admisible, pero inconsistente
    # con h(c) = 0) y desempate LIFO, c se cierra primero con g = 3 y después hay que
    # reabrirlo con g = 2 para devolver el plan óptimo (coste 3).
    task = graph([("s", "a"), ("a", "a2"), ("a2", "c"), ("s", "b"), ("b", "c"), ("c", "t")], "t")
    table = {"s": 0, "a": 0, "a2": 0, "b": 2, "c": 0, "t": 0}
    config = SearchConfig("astar-table", "astar", "blind", tie_breaking="lifo")
    result = search(task, config, heuristic=TableHeuristic(task, table))
    assert result.cost == 3
    assert result.statistics.reopened == 1


def test_goal_that_is_not_first_in_a_batch_waits() -> None:
    # s tiene dos sucesores con f = 1: n y el objetivo t, que entran en ese orden. Con FIFO
    # y lotes de 2, n sale primero y t después; t tiene que volver a la cola en lugar de
    # aceptarse ya.
    task = graph([("s", "t"), ("s", "n")], "t")
    table = {"s": 0, "t": 0, "n": 0}
    config = SearchConfig("astar-table", "astar", "blind", batch_size=2)
    result = search(task, config, heuristic=TableHeuristic(task, table))
    assert result.cost == 1
    assert result.statistics.expanded == 2  # s y n


def test_enforced_hill_climbing_falls_back_to_gbfs() -> None:
    # Desde s, solo la trampa mejora h (1 < 2), pero no tiene salida: la búsqueda en
    # anchura desde ella se agota, y GBFS encuentra s-a-t.
    task = graph([("s", "trap"), ("s", "a"), ("a", "t")], "t")
    table = {"s": 2, "trap": 1, "a": 2, "t": 0}
    config = SearchConfig("ehc-table", "ehc", "blind")
    result = search(task, config, heuristic=TableHeuristic(task, table))
    assert result.status is SearchStatus.SOLVED and result.cost == 2
    assert "GBFS" in result.message and result.config_name == "ehc-table"


def test_random_walks_are_reproducible_and_without_cycles() -> None:
    """Con la misma semilla, el mismo plan; y sin ciclos (ningún estado se repite)."""
    first = search(BLOCKS, PRESETS["mrw-tff-pref"])
    assert first.plan is not None
    assert search(BLOCKS, PRESETS["mrw-tff-pref"]).plan == first.plan
    state, seen = BLOCKS.init, {BLOCKS.init}
    for index in first.plan:
        successor = BLOCKS.apply(state, BLOCKS.actions[index])
        assert successor is not None and successor not in seen
        state = successor
        seen.add(state)


@pytest.mark.parametrize("name", ["ehc-tff-pref", "mrw-tff-pref"])
@pytest.mark.parametrize(
    "instance",
    ["tier3/temporal/driverlog/p02", "tier5/preferences/storage/p02"],
)
def test_local_searches_in_temporal_and_preference_tasks(
    name: str, instance: str, tmp_path: Path
) -> None:
    """Planes válidos (VAL) en una tarea temporal y con preferencias (la búsqueda anytime
    encadena después WA*: con los paseos aleatorios, sin el paso con los costes reales)."""
    group, _, stem = instance.rpartition("/")
    base = tier_dir(group)
    domain_path, problem_path = base / "domain.pddl", base / f"{stem}.pddl"
    domain = read_domain(domain_path)
    problem = read_problem(problem_path, domain)
    result = solve(domain, problem, [PRESETS[name]], 10)
    assert result.plan is not None
    if VALIDATE.exists():
        plan_file = tmp_path / "plan"
        plan_file.write_text(result.plan.to_ipc())
        output = subprocess.run(
            [str(VALIDATE), str(domain_path), str(problem_path), str(plan_file)],
            capture_output=True, text=True, check=False,
        ).stdout  # fmt: skip
        assert "Plan valid" in output, output


def test_lazy_search_dead_end_and_limits() -> None:
    lazy = PRESETS["lazy-ff-pref"]
    assert search(unreachable(), lazy).status is SearchStatus.UNSOLVABLE
    limited = search(GRIPPER, lazy.with_changes(max_expansions=1))
    assert limited.status is SearchStatus.LIMIT_REACHED


# --- Límite de memoria ----------------------------------------------------------------


def test_memory_limit_stops_the_search() -> None:
    """Con un límite menor que la memoria que ya usa el proceso, para en la primera
    comprobación."""
    result = search(chain(), PRESETS["gbfs-ff"].with_changes(max_memory=1))
    assert result.status is SearchStatus.LIMIT_REACHED
    assert "límite de memoria" in result.message


def test_invalid_memory_limit() -> None:
    with pytest.raises(ConfigError, match="max_memory"):
        PRESETS["gbfs-ff"].with_changes(max_memory=0)


def test_solve_applies_a_memory_limit() -> None:
    from pyplanner.loader import load_domain, load_problem
    from pyplanner.pddl import parse_domain, parse_problem
    from pyplanner.planner import solve
    from tests.tasks import CHAIN_DOMAIN, CHAIN_PROBLEM

    domain = load_domain(parse_domain(CHAIN_DOMAIN))
    problem = load_problem(parse_problem(CHAIN_PROBLEM), domain)
    assert solve(domain, problem, [PRESETS["gbfs-ff"]], memory_limit=1).status is (
        SearchStatus.LIMIT_REACHED
    )
    assert solve(domain, problem, [PRESETS["gbfs-ff"]]).status is SearchStatus.SOLVED


@pytest.mark.parametrize("tie_breaking", ["fifo", "lifo"])
def test_grouped_successors_leave_the_queue_in_the_same_order(
    tie_breaking: Literal["fifo", "lifo"],
) -> None:
    """La búsqueda perezosa mete las acciones de un nodo con la misma prioridad en una sola
    entrada por cola; tienen que salir igual que si se metieran una a una."""
    import random

    from pyplanner.search.engine import _OpenLists

    config = SearchConfig("x", "gbfs", "ff", preferred_operators=True, tie_breaking=tie_breaking)
    grouped, single = _OpenLists(config), _OpenLists(config)
    assert grouped.groups
    rng = random.Random(7)
    popped: tuple[list[tuple[int, ...] | None], list[tuple[int, ...] | None]] = ([], [])
    for node in range(60):
        successors = [(rng.randrange(4), a, rng.random() < 0.3) for a in range(rng.randrange(9))]
        groups: dict[tuple[float, ...], tuple[list[int], list[int]]] = {}
        for h, action, preferred in successors:
            single.push((float(h),), (node, action), preferred, (float(h), 0.0))
            group = groups.setdefault((float(h),), ([], []))
            group[0].append(action)
            if preferred:
                group[1].append(action)
        for key, (actions, useful) in groups.items():
            grouped.push_successors(key, node, actions, useful)
        for _ in range(rng.randrange(6)):
            popped[0].append(grouped.pop())
            popped[1].append(single.pop())
    while single:
        popped[0].append(grouped.pop())
        popped[1].append(single.pop())
    assert not grouped
    assert popped[0] == popped[1]
