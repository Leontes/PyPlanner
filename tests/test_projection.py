"""Heurística del motor discretizado (iteración 6c): la proyección de las invariantes, la
anticipación al empezar una acción, las acciones útiles que reducen el faltante y dónde se
anula una condición a lo largo de un flujo exacto."""

from pathlib import Path

import pytest

from pyplanner.grounding import State
from pyplanner.grounding.discrete import DiscreteTask
from pyplanner.grounding.flows import Flow
from pyplanner.grounding.numeric import Const, Op, Var
from pyplanner.heuristics import create_heuristic
from pyplanner.heuristics.projection import ProjectionHeuristic
from pyplanner.plan import Plan
from pyplanner.search import PRESETS, search
from tests.tasks import build
from tests.test_discrete_engine import val
from tests.tiers import VALIDATE

# Un generador que tiene que funcionar 10 unidades de tiempo gastando 1 por unidad, y un
# repostaje de hasta 4 unidades que suma 2 por unidad sin llegar a la capacidad.
POWER = """
(define (domain power)
  (:requirements :numeric-fluents :durative-actions :duration-inequalities
                 :negative-preconditions :continuous-effects :time)
  (:predicates (ran) (refueling))
  (:functions (fuel) (capacity))
  (:durative-action generate :parameters () :duration (= ?duration 10)
    :condition (over all (> (fuel) 0))
    :effect (and (decrease (fuel) (* #t 1)) (at end (ran))))
  (:durative-action refuel :parameters () :duration (<= ?duration 4)
    :condition (and (at start (not (refueling))) (over all (< (fuel) (capacity))))
    :effect (and (at start (refueling)) (at end (not (refueling)))))
  (:process flow :parameters () :precondition (refueling)
    :effect (increase (fuel) (* #t 2))))
"""


def power(fuel: float, capacity: float) -> DiscreteTask:
    problem = f"""
(define (problem power-1) (:domain power)
  (:init (= (fuel) {fuel}) (= (capacity) {capacity}))
  (:goal (ran))
  (:metric minimize (total-time)))
"""
    task = build(POWER, problem)
    assert isinstance(task, DiscreteTask)
    return task


def step(task: DiscreteTask, state: State, name: str) -> State:
    action = next(a for a in task.actions if str(a) == name)
    successor = task.apply(state, action)
    assert successor is not None
    return successor


def test_shortfall_counts_what_waiting_would_not_cover() -> None:
    """Con 8 de combustible, el generador se para a las 8 unidades: faltan 2. Repostando
    a la vez (2 - 1 por unidad durante 4), llega."""
    task = power(8, 20)
    assert task.shortfall(task.init) == 0  # no hay nada en curso
    generating = step(task, task.init, "(generate@start)")
    assert task.shortfall(generating) == pytest.approx(2.0, abs=0.01)
    assert task.shortfall(step(task, generating, "(refuel@start)")) == 0


def test_refuelling_stops_when_its_invariant_breaks() -> None:
    """Con capacidad 8,5, el repostaje se detiene al llenarse (en 0,5 unidades): el
    proceso de llenado deja de actuar aunque el fin no se pueda aplicar, y a partir de
    ahí el combustible dura 8,5 unidades. Faltan 10 - 0,02 - 0,5 - 8,5 = 0,98."""
    task = power(8, 8.5)
    both = step(task, step(task, task.init, "(generate@start)"), "(refuel@start)")
    assert task.shortfall(both) == pytest.approx(0.98, abs=0.01)


# Un depósito que se vacía mientras arde y una bomba que, una vez encendida, llena cada vez
# más deprisa: el nivel baja, cruza el 0 y vuelve a subir.
RESERVOIR = """
(define (domain reservoir)
  (:requirements :numeric-fluents :durative-actions :negative-preconditions
                 :continuous-effects :time)
  (:predicates (lit) (pumping))
  (:functions (level) (speed))
  (:action pump :parameters () :precondition (not (pumping)) :effect (pumping))
  (:durative-action burn :parameters () :duration (= ?duration 10)
    :condition (over all (> (level) 0))
    :effect (and (decrease (level) (* #t 1)) (at end (lit))))
  (:process surge :parameters () :precondition (pumping)
    :effect (and (increase (speed) (* #t 1)) (increase (level) (* #t (speed))))))
"""
RESERVOIR_PROBLEM = """
(define (problem reservoir-1) (:domain reservoir)
  (:init (= (level) 0.3) (= (speed) 0)) (:goal (lit)))
"""


def test_invariants_that_break_and_recover_count() -> None:
    """Tras encender la bomba y empezar a arder (0,02 unidades), el nivel es
    0,2902 - 0,98·t + t²/2: es negativo entre t ≈ 0,36 y t ≈ 1,6, y al final (t ≈ 10)
    vuelve a ser positivo. La proyección ve la ruptura entre medias."""
    task = build(RESERVOIR, RESERVOIR_PROBLEM)
    assert isinstance(task, DiscreteTask)
    burning = step(task, step(task, task.init, "(pump)"), "(burn@start)")
    assert task.shortfall(burning) == pytest.approx(10 - 0.01 - 0.3636, abs=0.01)


def test_crossings_of_an_exact_flow() -> None:
    """Una pelota (x' = v, v' = -1) lanzada a 2,5 cruza la altura 3 en 2 y en 3."""
    x, v = 0, 1
    flow = Flow({x: Var(v), v: Const(-1.0)})
    height = Op("-", (Var(x), Const(3.0)))
    assert flow.crossings(height, (0.0, 2.5), 10.0) == pytest.approx([2.0, 3.0])
    assert flow.crossings(height, (0.0, 2.5), 2.5) == pytest.approx([2.0])
    assert flow.crossings(Var(v), (0.0, 2.5), 10.0) == pytest.approx([2.5])
    assert Flow({x: Var(x)}).crossings(Var(x), (1.0,), 1.0) is None  # no es exacto


def test_the_heuristic_anticipates_starting_the_generator() -> None:
    """La acción útil del estado inicial es empezar a generar: el valor ya cuenta los 2
    que faltarían, y empezar no lo sube."""
    task = power(8, 20)
    heuristic = create_heuristic("ff", task)
    assert isinstance(heuristic, ProjectionHeuristic)
    initial = heuristic.evaluate(task.init)
    assert initial.value - heuristic.base.evaluate(task.init).value == pytest.approx(2, abs=0.02)
    generating = heuristic.evaluate(step(task, task.init, "(generate@start)"))
    assert generating.value <= initial.value


def test_helpful_actions_reduce_the_shortfall() -> None:
    task = power(8, 20)
    heuristic = create_heuristic("ff", task)
    generating = step(task, task.init, "(generate@start)")
    preferred = frozenset(heuristic.evaluate(generating).preferred)
    refuel = next(a.index for a in task.actions if str(a) == "(refuel@start)")
    assert refuel not in preferred  # para la relajación basta con esperar
    assert heuristic.helpful(generating, preferred) == {refuel}


@pytest.mark.parametrize("config", ["bfws-ff-pref", "lazy-ff-pref"])
def test_greedy_searches_refuel_in_time(config: str, tmp_path: Path) -> None:
    task = power(8, 20)
    result = search(task, PRESETS[config])
    assert result.plan is not None
    plan = Plan.from_indices(task, result.plan)
    assert [s.name for s in plan.steps] == ["generate", "refuel"]
    if VALIDATE.exists():
        problem = (
            "(define (problem power-1) (:domain power) (:init (= (fuel) 8) (= (capacity) 20))"
            " (:goal (ran)) (:metric minimize (total-time)))"
        )
        (tmp_path / "domain.pddl").write_text(POWER)
        (tmp_path / "problem.pddl").write_text(problem)
        output = val(tmp_path / "domain.pddl", tmp_path / "problem.pddl", plan, tmp_path)
        assert "Plan valid" in output, output
