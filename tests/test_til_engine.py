"""Tests del motor con literales iniciales temporizados (TIL, PDDL 2.2)."""

import subprocess
from pathlib import Path

import pytest

from pyplanner.grounding import State
from pyplanner.grounding.temporal import TIL_SNAP, TemporalTask
from pyplanner.heuristics import create_heuristic, numba_available
from pyplanner.heuristics.deadlines import DeadlineHeuristic
from pyplanner.plan import Plan, simulate
from pyplanner.search import PRESETS, SearchStatus, search
from tests.tasks import build
from tests.test_heuristics import random_walk_states
from tests.test_temporal_engine import light
from tests.tiers import VALIDATE, tier_dir

# Una ventana: (open) solo es cierto entre los instantes 10 y 20, y (work) lo necesita
# durante toda la acción. El hecho solo lo cambian los TIL: no es estático.
WINDOW_DOMAIN = """
(define (domain window) (:requirements :durative-actions :timed-initial-literals)
  (:predicates (open) (done))
  (:durative-action work :parameters () :duration (= ?duration {duration})
    :condition (over all (open)) :effect (at end (done))))
"""
WINDOW_PROBLEM = """
(define (problem w) (:domain window) (:init (at 10 (open)) (at 20 (not (open))))
  (:goal (done)))
"""


def window(duration: float = 5) -> TemporalTask:
    task = build(WINDOW_DOMAIN.format(duration=duration), WINDOW_PROBLEM)
    assert isinstance(task, TemporalTask)
    return task


def solve(task: TemporalTask, preset: str = "gbfs-ff-pref-tb") -> Plan:
    result = search(task, PRESETS[preset])
    assert result.status is SearchStatus.SOLVED, result.message
    assert result.plan is not None and simulate(task, result.plan) is None
    return Plan.from_indices(task, result.plan)


def val(domain: str, problem: str, plan: Plan, tmp_path: Path) -> str:
    for name, text in (("domain.pddl", domain), ("problem.pddl", problem), ("plan", plan.to_ipc())):
        (tmp_path / name).write_text(text)
    return subprocess.run(
        [str(VALIDATE), *(str(tmp_path / f) for f in ("domain.pddl", "problem.pddl", "plan"))],
        capture_output=True,
        text=True,
        check=False,
    ).stdout


def test_timed_literals_become_fixed_time_steps() -> None:
    task = window()
    tils = [a for a, s in zip(task.actions, task.snaps, strict=True) if s.kind == TIL_SNAP]
    assert [str(a) for a in tils] == ["(til@0)", "(til@1)"]
    assert all(a.cost == 0 for a in tils)
    assert task.til_times == (10.0, 20.0)
    assert [s.time for s in task.snaps if s.kind == TIL_SNAP] == [10.0, 20.0]
    # (open) no es estático aunque ninguna acción del dominio lo cambie.
    assert any(f.predicate == "open" for f in task.facts)


def test_actions_fit_inside_a_window(tmp_path: Path) -> None:
    plan = solve(window())
    assert [step.name for step in plan.steps] == ["work"]  # los TIL no salen en el plan
    (step,) = plan.steps
    assert step.time is not None and step.duration is not None
    assert step.time >= 10
    assert step.time + step.duration <= 20
    if VALIDATE.exists():
        output = val(WINDOW_DOMAIN.format(duration=5), WINDOW_PROBLEM, plan, tmp_path)
        assert "Plan valid" in output


def test_actions_longer_than_the_window_are_impossible() -> None:
    result = search(window(duration=15), PRESETS["astar-blind"])
    assert result.status is SearchStatus.UNSOLVABLE


def test_steps_before_a_timed_literal_happen_before_it() -> None:
    """(ready) es cierto al principio y un TIL lo borra en el instante 5: una acción que lo
    necesita tiene que terminar antes, aunque la búsqueda no aplique el TIL (el tiempo pasa
    igual)."""
    domain = """
(define (domain early) (:requirements :durative-actions :timed-initial-literals)
  (:predicates (ready) (done))
  (:durative-action work :parameters () :duration (= ?duration {duration})
    :condition (over all (ready)) :effect (at end (done))))
"""
    problem = """
(define (problem e) (:domain early) (:init (ready) (at 5 (not (ready)))) (:goal (done)))
"""
    short = build(domain.format(duration=3), problem)
    assert isinstance(short, TemporalTask)
    (step,) = solve(short).steps
    assert step.time == 0
    long = build(domain.format(duration=10), problem)
    assert search(long, PRESETS["astar-blind"]).status is SearchStatus.UNSOLVABLE


def test_timed_literals_at_time_zero() -> None:
    task = build(
        WINDOW_DOMAIN.format(duration=5),
        WINDOW_PROBLEM.replace("(at 10 (open))", "(at 0 (open))"),
    )
    assert isinstance(task, TemporalTask)
    (step,) = solve(task).steps
    assert step.time == pytest.approx(task.epsilon)  # después del TIL, separado por ε


def test_instantaneous_actions_with_timed_literals(tmp_path: Path) -> None:
    """Un dominio temporal sin acciones durativas: los TIL bastan para que la tarea sea
    temporal."""
    domain = """
(define (domain shop) (:requirements :durative-actions :timed-initial-literals)
  (:predicates (open) (bought))
  (:action buy :parameters () :precondition (open) :effect (bought)))
"""
    problem = """
(define (problem s) (:domain shop) (:init (at 8 (open)) (at 9 (not (open))))
  (:goal (bought)))
"""
    task = build(domain, problem)
    assert isinstance(task, TemporalTask) and not task.durative
    plan = solve(task)
    (step,) = plan.steps
    assert step.name == "buy" and step.time is not None and 8 < step.time < 9
    if VALIDATE.exists():
        assert "Plan valid" in val(domain, problem, plan, tmp_path)


@pytest.mark.parametrize("preset", ["gbfs-ff-pref", "lazy-ff-pref", "bfws-ff-pref", "astar-hmax"])
def test_window_plans_are_valid_for_val(preset: str, tmp_path: Path) -> None:
    if not VALIDATE.exists():
        pytest.skip("VAL no está compilado")
    plan = solve(window(), preset)
    assert "Plan valid" in val(WINDOW_DOMAIN.format(duration=5), WINDOW_PROBLEM, plan, tmp_path)


# --- Plazos en la heurística ---------------------------------------------------------


def deadline_heuristic(task: TemporalTask, compiled: bool = False) -> DeadlineHeuristic:
    heuristic = create_heuristic("ff", task, "numba" if compiled else "python")
    assert isinstance(heuristic, DeadlineHeuristic)
    return heuristic


def test_the_deadline_check_wraps_only_tasks_with_timed_literals() -> None:
    assert isinstance(create_heuristic("ff", window()), DeadlineHeuristic)
    assert not isinstance(create_heuristic("ff", light()), DeadlineHeuristic)


def test_windows_too_short_are_dead_ends() -> None:
    """La acción dura 15 y la ventana, 10: h_FF no lo ve, pero los plazos sí."""
    task = window(duration=15)
    heuristic = deadline_heuristic(task)
    assert not heuristic.base.evaluate(task.init).is_dead_end
    assert heuristic.evaluate(task.init).is_dead_end
    assert not deadline_heuristic(window(duration=5)).evaluate(window(5).init).is_dead_end


def test_deadlines_of_facts_true_at_the_start() -> None:
    domain = """
(define (domain early) (:requirements :durative-actions :timed-initial-literals)
  (:predicates (ready) (done))
  (:durative-action work :parameters () :duration (= ?duration {duration})
    :condition (over all (ready)) :effect (at end (done))))
"""
    problem = """
(define (problem e) (:domain early) (:init (ready) (at 5 (not (ready)))) (:goal (done)))
"""
    for duration, dead in ((3, False), (10, True)):
        task = build(domain.format(duration=duration), problem)
        assert isinstance(task, TemporalTask)
        assert deadline_heuristic(task).evaluate(task.init).is_dead_end is dead


def reachable_states(task: TemporalTask) -> list[State]:
    """Todos los estados alcanzables de una tarea pequeña."""
    seen = {task.init}
    frontier: list[State] = [task.init]
    while frontier:
        state = frontier.pop()
        for action in task.applicable(state):
            successor = task.apply(state, action)
            if successor is not None and successor not in seen:
                seen.add(successor)
                frontier.append(successor)
    return list(seen)


def satellite(name: str) -> TemporalTask:
    from pyplanner.frontend import read_domain, read_problem
    from pyplanner.grounding import ground

    directory = tier_dir("tier4/timed-initial-literals/satellite")
    domain = read_domain(directory / "domain.pddl")
    task = ground(domain, read_problem(directory / f"{name}.pddl", domain))
    assert isinstance(task, TemporalTask)
    return task


@pytest.mark.parametrize("name", ["p01", "p02"])
def test_states_on_a_valid_plan_are_never_pruned(name: str) -> None:
    task = satellite(name)
    heuristic = deadline_heuristic(task, compiled=numba_available())
    result = search(task, PRESETS["gbfs-ff-pref"])
    assert result.status is SearchStatus.SOLVED and result.plan is not None
    state = task.init
    path = [state]
    for index in result.plan:
        successor = task.apply(state, task.actions[index])
        assert successor is not None
        state = successor
        path.append(state)
    assert all(heuristic.reachable(path))


@pytest.mark.skipif(not numba_available(), reason="Numba no está instalado")
@pytest.mark.parametrize("name", ["p01", "p02"])
def test_compiled_deadlines_match_the_reference(name: str) -> None:
    task = satellite(name)
    states = random_walk_states(task, 1500, seed=3)
    from pyplanner.heuristics.compiled_deadlines import reachable

    # Los dos Dijkstra directamente (el envoltorio se salta los estados sin TIL pendientes).
    deadlines = deadline_heuristic(task).deadlines
    reference = deadlines.reachable_reference(states)
    assert reachable(deadlines, states) == reference
    if name == "p01":  # los paseos aplican los TIL y cierran la ventana: estados sin salida
        assert not all(reference)


@pytest.mark.skipif(not numba_available(), reason="Numba no está instalado")
def test_compiled_deadlines_on_small_tasks() -> None:
    for task in (window(5), window(15), window(9.99)):
        states = reachable_states(task)
        reference = deadline_heuristic(task).reachable(states)
        assert deadline_heuristic(task, compiled=True).reachable(states) == reference


# --- Comprobación aplazada en las búsquedas inmediatas ------------------------------


def test_eager_search_defers_the_deadline_check() -> None:
    """La búsqueda inmediata hace la comprobación al expandir: la heurística ya no descarta
    estados al evaluar."""
    task = window(duration=15)
    heuristic = deadline_heuristic(task)
    check = heuristic.defer_dead_end_check()
    assert check is not None and heuristic.deferred
    assert not heuristic.evaluate(task.init).is_dead_end  # solo la heurística base
    assert check([task.init]) == [False]
    assert search(task, PRESETS["gbfs-ff"]).status is SearchStatus.UNSOLVABLE


@pytest.mark.parametrize("preset", ["gbfs-ff-pref", "bfws-ff"])
def test_deferred_check_keeps_the_order_of_live_nodes(
    preset: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Descartar un nodo al expandirlo no gasta el turno de su cola: la búsqueda expande
    los mismos nodos vivos que si lo hubiera descartado al generarlo (salvo con novedad,
    que mide también los nodos que luego se descartan)."""
    task = satellite("p02")
    deferred = search(task, PRESETS[preset])
    monkeypatch.setattr(DeadlineHeuristic, "defer_dead_end_check", lambda self: None)
    immediate = search(task, PRESETS[preset])
    assert deferred.status is immediate.status is SearchStatus.SOLVED
    if preset == "gbfs-ff-pref":
        assert deferred.plan == immediate.plan
        assert deferred.statistics.expanded == immediate.statistics.expanded


def test_retry_takes_the_next_entry_from_the_same_queue() -> None:
    from pyplanner.search.engine import _OpenLists

    lists = _OpenLists(PRESETS["gbfs-ff-pref"])
    for node, preferred in ((0, False), (1, True), (2, False), (3, True)):
        lists.push((float(node),), (node,), preferred, (float(node), 0.0))
    assert lists.pop() == (0,)  # cola normal
    lists.retry()
    assert lists.pop() == (1,)  # otra vez la normal
    assert lists.pop() == (1,)  # ahora la preferente (la entrada duplicada de 1)
