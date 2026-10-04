"""Tests del motor con literales iniciales temporizados (TIL, PDDL 2.2)."""

import subprocess
from pathlib import Path

import pytest

from pyplanner.grounding.temporal import TIL_SNAP, TemporalTask
from pyplanner.plan import Plan, simulate
from pyplanner.search import PRESETS, SearchStatus, search
from tests.tasks import build
from tests.tiers import DOMINIOS_DIR

VALIDATE = DOMINIOS_DIR.parent / "tools" / "bin" / "Validate"

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
