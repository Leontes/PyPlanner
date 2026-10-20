"""Motor numérico (iteración 3a): grounding, semántica de los efectos, h^hbd y búsqueda."""

import math

import pytest

from pyplanner.grounding import GroundTask, State
from pyplanner.heuristics import Evaluation, create_heuristic
from pyplanner.heuristics.relaxation import (
    ONCE,
    REPEAT_EQ,
    REPEAT_GE,
    REPEAT_GT,
    RelaxedTask,
    repetitions,
)
from pyplanner.loader.ast import FluentTerm
from pyplanner.plan import Plan, simulate
from pyplanner.search import PRESETS, SearchStatus, search
from tests.tasks import build
from tests.test_heuristics import ACCELERATED_DEVICES

NUMERIC = "(:requirements :numeric-fluents :disjunctive-preconditions)"


def numeric_task(actions: str, init: str, goal: str, functions: str = "(x) (y)") -> GroundTask:
    domain = f"""
    (define (domain d) {NUMERIC}
      (:predicates (p) (q))
      (:functions {functions})
      {actions})
    """
    problem = f"(define (problem t) (:domain d) (:init {init}) (:goal {goal}))"
    return build(domain, problem)


def by_name(task: GroundTask, name: str) -> int:
    return next(a.index for a in task.actions if str(a) == name)


def evaluate(
    task: GroundTask, name: str, device: str = "python", state: State | None = None
) -> Evaluation:
    return create_heuristic(name, task, device).evaluate(state or task.init)


# --- Grounding -----------------------------------------------------------------------

# Depósito: cada recarga suma 3 y el objetivo pide al menos 10 (4 recargas).
TANK = numeric_task(
    "(:action refill :parameters () :precondition (and) :effect (increase (x) 3))",
    "(= (x) 0) (= (y) 0)",
    "(>= (x) 10)",
)


def test_numeric_variables_are_the_relevant_fluents() -> None:
    # y no aparece en ninguna condición: no forma parte del estado.
    assert TANK.variables == (FluentTerm("x"),)
    assert TANK.init == State(0, (0.0,))


def test_static_functions_are_substituted() -> None:
    task = numeric_task(
        "(:action inc :parameters () :precondition (< (x) (y)) :effect (increase (x) 1))",
        "(= (x) 0) (= (y) 2)",
        "(>= (x) 2)",
    )
    assert task.variables == (FluentTerm("x"),)
    inc = task.actions[0]
    assert inc.applicable(State(0, (1.0,)))
    assert not inc.applicable(State(0, (2.0,)))


def test_fluents_only_read_by_other_effects_are_relevant() -> None:
    # y solo se lee en el efecto sobre x, que sí se usa en el objetivo.
    task = numeric_task(
        """(:action grow :parameters () :precondition (and) :effect (increase (y) 1))
           (:action copy :parameters () :precondition (and) :effect (assign (x) (y)))""",
        "(= (x) 0) (= (y) 0)",
        "(>= (x) 2)",
    )
    assert task.variables == (FluentTerm("x"), FluentTerm("y"))


def test_undefined_static_comparison_is_false() -> None:
    task = numeric_task(
        """(:action a :parameters () :precondition (> (y) 0) :effect (p))
           (:action b :parameters () :precondition (and) :effect (increase (x) 1))""",
        "(= (x) 0)",
        "(or (p) (>= (x) 1))",
    )
    # y nunca se define ni se modifica: la precondición de a es falsa y a desaparece.
    assert [str(a) for a in task.actions] == ["(b)"]


# --- Métrica como coste --------------------------------------------------------------

COSTS_DOMAIN = """
(define (domain c) (:requirements :action-costs)
  (:constants l0 l1 l2)
  (:predicates (link ?a ?b) (at ?a))
  (:functions (total-cost) - number (dist ?a ?b) - number)
  (:action go :parameters (?a ?b) :precondition (and (at ?a) (link ?a ?b))
    :effect (and (at ?b) (not (at ?a)) (increase (total-cost) (dist ?a ?b))))
  (:action jump :parameters () :precondition (at l0)
    :effect (and (at l2) (not (at l0)) (increase (total-cost) 10))))
"""


def costs_problem(metric: str) -> str:
    return f"""
    (define (problem c1) (:domain c)
      (:init (at l0) (link l0 l1) (link l1 l2) (= (dist l0 l1) 2) (= (dist l1 l2) 3)
             (= (total-cost) 4))
      (:goal (at l2)) {metric})
    """


def test_action_costs_come_from_the_metric() -> None:
    task = build(COSTS_DOMAIN, costs_problem("(:metric minimize (total-cost))"))
    assert task.metric_is_cost and task.metric_offset == 4
    assert task.variables == ()  # total-cost solo está en la métrica
    costs = {str(a): a.cost for a in task.actions}
    assert costs == {"(go l0 l1)": 2, "(go l1 l2)": 3, "(jump)": 10}
    result = search(task, PRESETS["astar-hmax"])
    assert result.plan is not None
    plan = Plan.from_indices(task, result.plan)
    assert plan.cost == 5 and plan.metric == 9 and not plan.unit_cost
    assert plan.to_ipc().endswith("; cost = 5\n; metric = 9\n")
    assert plan.to_dict()["metric"] == 9


@pytest.mark.parametrize(
    ("metric", "note"),
    [
        ("(:metric maximize (total-cost))", "no es una constante no negativa"),
        ("", None),
    ],
)
def test_without_a_usable_metric_costs_are_unit(metric: str, note: str | None) -> None:
    # Maximizar total-cost ya no es el perfil de :action-costs: necesita :numeric-fluents.
    domain = COSTS_DOMAIN.replace(":action-costs", ":action-costs :numeric-fluents")
    task = build(domain, costs_problem(metric))
    assert not task.metric_is_cost
    assert all(a.cost == 1 for a in task.actions)
    assert any(note in n for n in task.notes) if note else task.notes == ()
    plan = Plan.from_indices(task, (by_name(task, "(jump)"),))
    assert plan.unit_cost and plan.metric is None
    assert plan.to_ipc().endswith("; cost = 1 (unit cost)\n")


# --- Semántica de los efectos ---------------------------------------------------------


def test_increases_on_the_same_variable_add_up() -> None:
    task = numeric_task(
        """(:action a :parameters () :precondition (and)
             :effect (and (increase (x) 1) (increase (x) 2) (decrease (y) (x))))""",
        "(= (x) 1) (= (y) 0)",
        "(and (>= (x) 4) (>= (y) 0))",
    )
    assert task.apply(task.init, task.actions[0]) == State(0, (4.0, -1.0))


def test_conflicting_assignments_make_the_action_inapplicable() -> None:
    task = numeric_task(
        "(:action a :parameters () :precondition (and) "
        ":effect (and (assign (x) 1) (increase (x) 1)))",
        "(= (x) 0)",
        "(>= (x) 2)",
    )
    assert task.apply(task.init, task.actions[0]) is None


@pytest.mark.parametrize(
    "effect", ["(assign (x) (/ 1 (y)))", "(assign (x) (+ (y) (z)))"], ids=["div0", "undefined"]
)
def test_undefined_results_make_the_action_inapplicable(effect: str) -> None:
    task = numeric_task(
        f"""(:action a :parameters () :precondition (and) :effect {effect})
            (:action b :parameters () :precondition (and) :effect (and (increase (y) 1)
             (assign (z) 1)))""",
        "(= (x) 0) (= (y) 0)",
        "(and (>= (x) 1) (>= (y) 0) (>= (z) 0))",
        functions="(x) (y) (z)",
    )
    assert task.apply(task.init, task.actions[by_name(task, "(a)")]) is None


def test_undefined_values_make_comparisons_false() -> None:
    task = numeric_task(
        """(:action set :parameters () :precondition (and) :effect (assign (x) 5))
           (:action use :parameters () :precondition (>= (x) 0) :effect (p))""",
        "",
        "(p)",
    )
    assert task.init.values == (None,)
    assert [str(a) for a in task.applicable(task.init)] == ["(set)"]
    result = search(task, PRESETS["gbfs-ff"])
    assert result.status is SearchStatus.SOLVED
    assert [str(task.actions[i]) for i in result.plan or ()] == ["(set)", "(use)"]


# --- h^hbd ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "delta", "value", "expected"),
    [
        (REPEAT_GE, 3.0, -10.0, 4.0),
        (REPEAT_GE, 3.0, -9.0, 3.0),
        (REPEAT_GE, 3.0, 2.0, 1.0),
        (REPEAT_GT, 3.0, -9.0, 4.0),
        (REPEAT_GT, 3.0, 0.0, 1.0),
        (REPEAT_EQ, -2.0, 4.0, 2.0),
        (REPEAT_EQ, 1.0, 4.0, math.inf),
        (REPEAT_EQ, 1.0, 0.0, 1.0),
        (REPEAT_GE, 3.0, None, math.inf),
        (ONCE, 0.0, None, 1.0),
    ],
)
def test_repetitions(kind: int, delta: float, value: float | None, expected: float) -> None:
    assert repetitions(kind, delta, value) == expected


def test_hbd_counts_repetitions() -> None:
    for name in ("hmax", "hadd", "ff"):
        assert evaluate(TANK, name).value == 4
    assert evaluate(TANK, "ff", state=State(0, (7.0,))).value == 1
    assert evaluate(TANK, "ff", state=State(0, (10.0,))).value == 0


def test_hbd_equality_moves_towards_the_target() -> None:
    task = numeric_task(
        """(:action inc :parameters () :precondition (and) :effect (increase (x) 1))
           (:action dec :parameters () :precondition (and) :effect (decrease (x) 2))""",
        "(= (x) 7)",
        "(= (x) 3)",
    )
    evaluation = evaluate(task, "ff")
    assert evaluation.value == 2
    assert evaluation.preferred == (by_name(task, "(dec)"),)


def test_worsening_effects_are_ignored() -> None:
    task = numeric_task(
        "(:action dec :parameters () :precondition (and) :effect (decrease (x) 1))",
        "(= (x) 0)",
        "(>= (x) 1)",
    )
    assert evaluate(task, "hadd").is_dead_end


def test_non_constant_effects_are_optimistic() -> None:
    task = numeric_task(
        "(:action double :parameters () :precondition (and) :effect (scale-up (x) 2))",
        "(= (x) 1)",
        "(>= (x) 100)",
    )
    assert evaluate(task, "ff").value == 1


@pytest.mark.parametrize("device", ["python", *ACCELERATED_DEVICES])
def test_constant_assignments_achieve_only_what_they_satisfy(device: str) -> None:
    """Asignar una constante a la única variable de una condición la consigue si, y solo
    si, ese valor la cumple: el reloj a 0 no deja terminar una acción de duración 5. Si la
    variable no está definida, sí cuenta: la define para que otras la aumenten."""
    reset = "(:action reset :parameters () :precondition (and) :effect (assign (x) 0))"
    fill = "(:action fill :parameters () :precondition (and) :effect (assign (x) 7))"
    inc = "(:action inc :parameters () :precondition (and) :effect (increase (x) 1))"
    unreachable = numeric_task(reset, "(= (x) 3)", "(>= (x) 5)")
    assert evaluate(unreachable, "hadd", device).is_dead_end
    task = numeric_task(reset + fill, "(= (x) 3)", "(>= (x) 5)")
    assert evaluate(task, "ff", device).preferred == (by_name(task, "(fill)"),)
    undefined = numeric_task(reset + inc, "(= (y) 0)", "(>= (x) 5)")
    assert evaluate(undefined, "ff", device).preferred == (by_name(undefined, "(reset)"),)


# Desempate (coste, nivel): q se logra con coste 2 por "slow" (nivel 2, índice menor) y
# por "direct" (nivel 1). El soportador es "direct" aunque tenga un índice mayor.
TIES = build(
    """
    (define (domain ties) (:requirements :action-costs)
      (:predicates (p) (q) (r))
      (:functions (total-cost) - number)
      (:action free :parameters () :precondition (and) :effect (p))
      (:action slow :parameters () :precondition (p)
        :effect (and (q) (increase (total-cost) 2)))
      (:action direct :parameters () :precondition (and)
        :effect (and (q) (increase (total-cost) 2)))
      (:action last :parameters () :precondition (q)
        :effect (and (r) (increase (total-cost) 1))))
    """,
    """
    (define (problem ties-1) (:domain ties) (:init (= (total-cost) 0)) (:goal (r))
      (:metric minimize (total-cost)))
    """,
)


@pytest.mark.parametrize("device", ["python", *ACCELERATED_DEVICES])
def test_supporters_break_ties_by_level(device: str) -> None:
    evaluation = evaluate(TIES, "ff", device)
    assert evaluation.value == 3
    assert evaluation.preferred == (by_name(TIES, "(direct)"),)


HANDMADE = {
    "tank": TANK,
    "ties": TIES,
    "eq": numeric_task(
        """(:action inc :parameters () :precondition (and) :effect (increase (x) 1))
           (:action dec :parameters () :precondition (> (x) 0) :effect (decrease (x) 2))
           (:action flag :parameters () :precondition (= (x) 3) :effect (p))""",
        "(= (x) 8)",
        "(or (p) (>= (x) 20))",
    ),
    "undefined": numeric_task(
        """(:action set :parameters () :precondition (and) :effect (assign (x) 5))
           (:action mul :parameters () :precondition (and) :effect (scale-up (x) (y)))
           (:action grow :parameters () :precondition (>= (x) 1) :effect (increase (y) 1))
           (:action done :parameters () :precondition (>= (* (x) (y)) 30) :effect (p))""",
        "(= (y) 1)",
        "(p)",
    ),
}


def reachable(task: GroundTask, limit: int = 300) -> list[State]:
    seen = {task.init}
    frontier = [task.init]
    while frontier and len(seen) < limit:
        state = frontier.pop(0)
        for action in task.applicable(state):
            successor = task.apply(state, action)
            if successor is not None and successor not in seen:
                seen.add(successor)
                frontier.append(successor)
    return list(seen)


@pytest.mark.skipif(not ACCELERATED_DEVICES, reason="ni Numba ni PyTorch están instalados")
@pytest.mark.parametrize("device", ACCELERATED_DEVICES)
@pytest.mark.parametrize("name", ["hmax", "hadd", "ff"])
@pytest.mark.parametrize("task", HANDMADE.values(), ids=HANDMADE.keys())
def test_accelerated_numeric_matches_reference(task: GroundTask, name: str, device: str) -> None:
    states = reachable(task)
    reference = create_heuristic(name, task, "python").evaluate_batch(states)
    assert create_heuristic(name, task, device).evaluate_batch(states) == reference


def test_relaxed_task_has_one_node_per_numeric_condition() -> None:
    relaxed = RelaxedTask(HANDMADE["eq"])
    comparators = sorted(c.comparator for c in relaxed.numeric)
    assert comparators == ["=", ">", ">="]
    assert relaxed.num_facts == relaxed.numeric_offset + 3


# --- Búsqueda ------------------------------------------------------------------------


@pytest.mark.parametrize("config", ["gbfs-ff", "lazy-ff-pref", "astar-blind"])
@pytest.mark.parametrize("task", HANDMADE.values(), ids=HANDMADE.keys())
def test_search_solves_numeric_tasks(task: GroundTask, config: str) -> None:
    result = search(task, PRESETS[config])
    assert result.status is SearchStatus.SOLVED, result.message
    assert simulate(task, result.plan or ()) is None


def test_optimal_search_uses_action_costs() -> None:
    result = search(TIES, PRESETS["astar-blind"])
    assert result.plan is not None
    assert [str(TIES.actions[i]) for i in result.plan] == ["(direct)", "(last)"]
