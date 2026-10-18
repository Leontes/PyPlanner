"""Ritmos que fijan las acciones y esperas hasta el siguiente cambio: la relajación solo
cuenta un incremento variable si es positivo y cuenta sus repeticiones con el ritmo, y
``_wait-until`` junta en un paso lo que serían muchas esperas de δ."""

import pytest

from pyplanner.grounding import State
from pyplanner.grounding.discrete import DiscreteTask
from pyplanner.heuristics import create_heuristic
from pyplanner.plan import Plan
from pyplanner.search import PRESETS, search
from tests.tasks import build
from tests.test_discrete_engine import action, tank
from tests.test_heuristics import ACCELERATED_DEVICES
from tests.test_numeric_engine import by_name, numeric_task

# --- Relajación -------------------------------------------------------------------------

# Verter suma a x lo que vale y; abrir suma 2 a y.
POUR = "(:action pour :parameters () :precondition (and) :effect (increase (x) (y)))"
OPEN = "(:action open :parameters () :precondition (and) :effect (increase (y) 2))"


@pytest.mark.parametrize("device", ["python", *ACCELERATED_DEVICES])
def test_variable_increments_need_a_positive_rate(device: str) -> None:
    """Con y = 0, verter no sirve hasta abrir; las repeticiones se cuentan con lo que
    abrir le da a y (2): 5 vertidos para llegar a 10."""
    task = numeric_task(POUR + OPEN, "(= (x) 0) (= (y) 0)", "(>= (x) 10)")
    evaluation = create_heuristic("ff", task, device).evaluate(task.init)
    assert evaluation.value == 1 + 5
    assert evaluation.preferred == (by_name(task, "(open)"),)
    stuck = numeric_task(POUR, "(= (x) 0) (= (y) 0)", "(>= (x) 10)")
    assert create_heuristic("hadd", stuck, device).evaluate(stuck.init).is_dead_end


@pytest.mark.parametrize("device", ["python", *ACCELERATED_DEVICES])
def test_repetitions_use_the_rate_of_the_state(device: str) -> None:
    """Con y = 5 bastan 2 vertidos; con y = 1, menos que lo que da abrir, cuenta 2."""
    task = numeric_task(POUR + OPEN, "(= (x) 0) (= (y) 0)", "(>= (x) 10)")
    heuristic = create_heuristic("ff", task, device)
    assert heuristic.evaluate(State(0, (0.0, 5.0))).value == 2
    assert heuristic.evaluate(State(0, (0.0, 1.0))).value == 5


# Un taller: construirlo, contratar a un carpintero (0,1 tablones por unidad de tiempo) y
# esperar a tener 50 tablones.
WORKSHOP = """
(define (domain workshop)
  (:requirements :numeric-fluents :negative-preconditions :time)
  (:predicates (built) (hired))
  (:functions (planks) (rate))
  (:action build :parameters () :precondition (not (built)) :effect (built))
  (:action hire :parameters () :precondition (and (built) (not (hired)))
    :effect (and (hired) (increase (rate) 0.1)))
  (:process work :parameters () :precondition (and)
    :effect (increase (planks) (* #t (rate)))))
"""
WORKSHOP_PROBLEM = """
(define (problem workshop-1) (:domain workshop)
  (:init (= (planks) 0) (= (rate) 0)) (:goal (>= (planks) 50)))
"""


def workshop() -> DiscreteTask:
    task = build(WORKSHOP, WORKSHOP_PROBLEM)
    assert isinstance(task, DiscreteTask)
    return task


def test_the_heuristic_estimates_the_waiting_time() -> None:
    """Dos acciones (1/64 cada una) y 500 esperas de δ = 1 al ritmo del carpintero."""
    task = workshop()
    evaluation = create_heuristic("ff", task).evaluate(task.init)
    assert evaluation.value == 500 + 2 / 64
    assert [str(task.actions[i]) for i in evaluation.preferred] == ["(build)"]


@pytest.mark.parametrize("config", ["lazy-ff-pref", "bfws-ff-pref", "astar-blind"])
def test_production_chains_are_solved_in_few_steps(config: str) -> None:
    task = workshop()
    result = search(task, PRESETS[config])
    assert result.plan is not None
    assert [str(task.actions[i]) for i in result.plan] == ["(build)", "(hire)", "(_wait-until)"]
    plan = Plan.from_indices(task, result.plan)
    assert [s.name for s in plan.steps] == ["build", "hire"]
    assert task.check(result.plan, 0.1) is None


# --- Esperar hasta el siguiente cambio ----------------------------------------------------


def test_waiting_until_an_action_can_be_applied() -> None:
    """Con el grifo a abierto (flujo 1), la espera llega justo a 5, lo que pide finish."""
    task = tank()
    level = task.variable_index[next(v for v in task.variables if str(v) == "(level)")]
    opened = task.apply(task.init, task.actions[action(task, "(turn-on a)")])
    assert opened is not None
    reached, elapsed = task.wait_until(opened)
    assert reached is not None and reached.values[level] == pytest.approx(5)
    assert elapsed == pytest.approx(4.99)
    assert task.transition(opened, task.jump) == (reached, elapsed)  # cuesta lo que avanza
    assert task.actions[action(task, "(finish)")].applicable(reached)


def test_waiting_until_an_event() -> None:
    """Pasado el 5, lo siguiente que cambia es el desbordamiento (un evento) en 10."""
    task = tank(goal="(overflowed)")
    level = task.variable_index[next(v for v in task.variables if str(v) == "(level)")]
    state = task.apply(task.init, task.actions[action(task, "(turn-on a)")])
    assert state is not None
    state, _ = task.wait_until(state)
    assert state is not None
    state, elapsed = task.wait_until(state)
    assert state is not None and task.is_goal(state)
    assert elapsed == pytest.approx(5) and state.values[level] == pytest.approx(10)


def test_waiting_until_nothing_is_not_applicable() -> None:
    task = tank()
    assert task.wait_until(task.init) == (None, 0.0)  # con los grifos cerrados no cambia nada


def test_waiting_until_is_helpful_with_the_wait() -> None:
    """Con el carpintero contratado, la relajación solo pide esperar (``_wait-until`` no
    está en ella): las útiles al expandir incluyen también la espera hasta el cambio."""
    task = workshop()
    heuristic = create_heuristic("ff", task)
    state = task.init
    for name in ("(build)", "(hire)"):
        successor = task.apply(state, task.actions[action(task, name)])
        assert successor is not None
        state = successor
    preferred = frozenset(heuristic.evaluate(state).preferred)
    assert preferred == {task.wait.index}
    assert heuristic.helpful(state, preferred) == {task.wait.index, task.jump.index}
