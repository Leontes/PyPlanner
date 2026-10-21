"""h_TFF: el plan relajado temporal (trabajo opcional, punto 3: parte B de la heurística
temporal)."""

import pytest

from pyplanner.frontend import read_domain, read_problem
from pyplanner.grounding import GroundTask, State, ground
from pyplanner.grounding.temporal import TemporalTask
from pyplanner.heuristics import create_heuristic, numba_available
from pyplanner.heuristics.temporal_ff import TemporalFFHeuristic
from pyplanner.plan import simulate
from pyplanner.search import PRESETS, SearchStatus, search
from tests.tasks import build
from tests.test_heuristics import random_walk_states
from tests.test_search import GRIPPER
from tests.test_temporal_engine import light
from tests.test_til_engine import deadline_heuristic, reachable_states, satellite, window
from tests.tiers import tier_dir

# La puerta a cierra antes de que quepa la acción que la cruza: h_FF propone cruzarla (tiene
# el índice menor) y h_TFF, la puerta b.
TWO_GATES = """
(define (domain two-gates) (:requirements :durative-actions :timed-initial-literals)
  (:predicates (gate-a) (gate-b) (passed))
  (:durative-action through-a :parameters () :duration (= ?duration 10)
    :condition (over all (gate-a)) :effect (at end (passed)))
  (:durative-action through-b :parameters () :duration (= ?duration 10)
    :condition (over all (gate-b)) :effect (at end (passed))))
"""
TWO_GATES_PROBLEM = """
(define (problem two-gates-1) (:domain two-gates)
  (:init (gate-a) (gate-b) (at 5 (not (gate-a))) (at 100 (not (gate-b))))
  (:goal (passed)))
"""


def temporal_tier(path: str, problem: str = "p01") -> TemporalTask:
    directory = tier_dir(path)
    domain = read_domain(directory / "domain.pddl")
    task = ground(domain, read_problem(directory / f"{problem}.pddl", domain))
    assert isinstance(task, TemporalTask)
    return task


def two_gates() -> TemporalTask:
    task = build(TWO_GATES, TWO_GATES_PROBLEM)
    assert isinstance(task, TemporalTask)
    return task


def preferred_names(task: GroundTask, name: str) -> list[str]:
    evaluation = create_heuristic(name, task, "python").evaluate(task.init)
    return [str(task.actions[i]) for i in evaluation.preferred]


def test_windows_choose_the_achiever() -> None:
    task = two_gates()
    assert preferred_names(task, "ff") == ["(through-a@start)"]
    assert preferred_names(task, "tff") == ["(through-b@start)"]
    assert create_heuristic("tff", task, "python").evaluate(task.init).value == 2


@pytest.mark.parametrize("kind", ["classical", "temporal"])
def test_tff_is_ff_without_timed_literals(kind: str) -> None:
    """Sin TIL, h_TFF solo cambiaría los soportadores (por instante en lugar de por coste),
    y eso es peor: ``tff`` es h_FF."""
    task: GroundTask = GRIPPER if kind == "classical" else light()
    heuristic = create_heuristic("tff", task, "python")
    assert heuristic.name == "ff" and not isinstance(heuristic, TemporalFFHeuristic)
    states = random_walk_states(task, 50, seed=1)
    reference = create_heuristic("ff", task, "python").evaluate_batch(states)
    assert heuristic.evaluate_batch(states) == reference


def test_tff_is_not_wrapped_by_the_deadline_check() -> None:
    assert isinstance(create_heuristic("tff", window(), "python"), TemporalFFHeuristic)


@pytest.mark.parametrize("duration", [5, 15, 9.99])
def test_dead_ends_are_those_of_the_deadline_check(duration: float) -> None:
    """Mismo Dijkstra de instantes que la parte A: los mismos callejones sin salida."""
    task = window(duration)
    states = reachable_states(task)
    tff = create_heuristic("tff", task, "python").evaluate_batch(states)
    reachable = deadline_heuristic(task).deadlines.reachable_reference(states)
    assert [not e.is_dead_end for e in tff] == reachable


def test_dead_ends_on_satellite() -> None:
    task = satellite("p01")
    states = random_walk_states(task, 600, seed=3)
    tff = create_heuristic("tff", task, "python").evaluate_batch(states)
    reachable = deadline_heuristic(task).deadlines.reachable_reference(states)
    assert [not e.is_dead_end for e in tff] == reachable and not all(reachable)


def tasks_and_states() -> list[tuple[str, TemporalTask, list[State]]]:
    """Tareas temporales con y sin TIL, con efectos continuos y con restricciones, y estados
    de cada una."""
    cases: list[tuple[str, TemporalTask, list[State]]] = []
    for duration in (5, 15, 9.99):
        task = window(duration)
        cases.append((f"window-{duration}", task, reachable_states(task)))
    gates = two_gates()
    cases.append(("two-gates", gates, reachable_states(gates)))
    cases.append(("light", light(), random_walk_states(light(), 100, seed=2)))
    for name in ("p01", "p02"):
        task = satellite(name)
        cases.append((f"satellite-{name}", task, random_walk_states(task, 800, seed=3)))
    for path in (
        "tier3/temporal/driverlog",
        "tier3/temporal/zenotravel",
        "tier3/temporal/greenhouse",
        "tier5/constraints/trucks",
    ):
        task = temporal_tier(path)
        cases.append((path.rsplit("/", 1)[1], task, random_walk_states(task, 400, seed=5)))
    return cases


# La clase directamente: también en las tareas temporales sin TIL, donde ``tff`` es h_FF.
@pytest.mark.skipif(not numba_available(), reason="Numba no está instalado")
def test_compiled_matches_the_reference() -> None:
    for name, task, states in tasks_and_states():
        reference = TemporalFFHeuristic(task, "python").evaluate_batch(states)
        assert TemporalFFHeuristic(task, "numba").evaluate_batch(states) == reference, name


def test_tensors_match_the_reference() -> None:
    pytest.importorskip("torch")
    for name, task, states in tasks_and_states():
        reference = TemporalFFHeuristic(task, "python").evaluate_batch(states)
        assert TemporalFFHeuristic(task, "cpu").evaluate_batch(states) == reference, name


@pytest.mark.parametrize("preset", ["gbfs-tff-pref", "lazy-tff-pref", "bfws-tff-pref"])
def test_tff_searches_find_valid_plans(preset: str) -> None:
    tasks: list[GroundTask] = [window(5), two_gates(), satellite("p01"), satellite("p02")]
    tasks += [temporal_tier("tier3/temporal/driverlog"), GRIPPER]
    for task in tasks:
        result = search(task, PRESETS[preset])
        assert result.status is SearchStatus.SOLVED and result.plan is not None
        assert simulate(task, result.plan) is None
