"""Tests del motor con predicados derivados (PDDL 2.2): grounding de los axiomas,
evaluación por estratos, heurísticas y búsqueda."""

import subprocess
from pathlib import Path

import pytest

from pyplanner.grounding import GroundTask, State
from pyplanner.grounding.axioms import AxiomEvaluator
from pyplanner.grounding.discrete import DiscreteTask
from pyplanner.grounding.temporal import TemporalTask
from pyplanner.heuristics import TENSOR_HEURISTICS, create_heuristic
from pyplanner.pddl import parse_pddl, write_pddl
from pyplanner.plan import Plan
from pyplanner.search import PRESETS, SearchStatus, search
from pyplanner.search.engine import EagerSearch
from tests.catalog import VALIDATE
from tests.tasks import build
from tests.test_heuristics import ACCELERATED_DEVICES, all_states

# Un grafo dirigido cuyos arcos se pueden cortar. reach es el cierre transitivo desde el
# origen (regla recursiva, estrato 0) e isolated, su negación sobre los nodos (estrato 1).
GRAPH_DOMAIN = """
(define (domain graph)
  (:requirements :derived-predicates :negative-preconditions :existential-preconditions
                 :disjunctive-preconditions)
  (:predicates (edge ?x ?y) (start ?x) (node ?x) (reach ?x) (isolated ?x))
  (:derived (reach ?x) (or (start ?x) (exists (?y) (and (reach ?y) (edge ?y ?x)))))
  (:derived (isolated ?x) (and (node ?x) (not (reach ?x))))
  (:action cut :parameters (?x ?y)
    :precondition (and (edge ?x ?y) (not (isolated ?y)))
    :effect (not (edge ?x ?y))))
"""


def graph_problem(goal: str) -> str:
    return f"""
(define (problem g) (:domain graph)
  (:objects a b c d)
  (:init (start a) (node a) (node b) (node c) (node d)
         (edge a b) (edge b c) (edge c d) (edge a c))
  (:goal {goal}))
"""


def graph(goal: str = "(isolated c)") -> GroundTask:
    return build(GRAPH_DOMAIN, graph_problem(goal))


def true_atoms(task: GroundTask, state: State, predicate: str) -> set[str]:
    return {str(a.args[0]) for a in task.atoms(state) if a.predicate == predicate}


def action(task: GroundTask, text: str) -> int:
    return next(a.index for a in task.actions if str(a) == text)


# --- Grounding y evaluación ----------------------------------------------------------


def test_axioms_are_grounded_by_stratum() -> None:
    task = graph()
    strata = {str(task.facts[a.head].predicate): a.stratum for a in task.axioms}
    assert strata == {"reach": 0, "isolated": 1}
    assert [a.stratum for a in task.axioms] == sorted(a.stratum for a in task.axioms)
    derived = {str(task.facts[i]) for i in range(len(task.facts)) if task.derived_mask >> i & 1}
    # Un predicado derivado nunca es estático, aunque solo dependa de hechos estáticos.
    assert derived == {*(f"(reach {n})" for n in "abcd"), *(f"(isolated {n})" for n in "abcd")}


def test_initial_state_includes_derived_facts() -> None:
    task = graph()
    assert true_atoms(task, task.init, "reach") == {"a", "b", "c", "d"}
    assert true_atoms(task, task.init, "isolated") == set()


def test_derived_facts_are_recomputed_after_each_action() -> None:
    task = graph()
    state = task.apply(task.init, task.actions[action(task, "(cut c d)")])
    assert state is not None
    assert true_atoms(task, state, "reach") == {"a", "b", "c"}
    assert true_atoms(task, state, "isolated") == {"d"}
    # Cortar a→c no basta para aislar c: sigue siendo alcanzable por b.
    state = task.apply(state, task.actions[action(task, "(cut a c)")])
    assert state is not None
    assert true_atoms(task, state, "isolated") == {"d"}
    state = task.apply(state, task.actions[action(task, "(cut b c)")])
    assert state is not None
    assert true_atoms(task, state, "reach") == {"a", "b"}
    assert true_atoms(task, state, "isolated") == {"c", "d"}


def test_derived_facts_in_preconditions() -> None:
    task = graph()
    state = task.apply(task.init, task.actions[action(task, "(cut c d)")])
    assert state is not None
    # d está aislado: ya no se puede cortar a→... hacia d (no hay), pero tampoco c→d.
    applicable = {str(a) for a in task.applicable(state)}
    assert applicable == {"(cut a b)", "(cut b c)", "(cut a c)"}


def test_evaluation_ignores_stale_derived_facts() -> None:
    task = graph()
    evaluator = AxiomEvaluator(task.axioms)
    stale = State(task.init.facts | task.derived_mask, task.init.values)
    assert evaluator.evaluate(stale) == task.init


def test_static_derived_facts() -> None:
    domain = """
(define (domain s) (:requirements :derived-predicates)
  (:predicates (base ?x) (linked ?x) (done ?x))
  (:derived (linked ?x) (base ?x))
  (:action finish :parameters (?x) :precondition (linked ?x) :effect (done ?x)))
"""
    problem = """
(define (problem s1) (:domain s) (:objects a b) (:init (base a)) (:goal (done a)))
"""
    task = build(domain, problem)
    result = search(task, PRESETS["astar-blind"])
    assert result.status is SearchStatus.SOLVED and result.plan is not None
    assert [str(task.actions[i]) for i in result.plan] == ["(finish a)"]


NUMERIC_DOMAIN = """
(define (domain n) (:requirements :derived-predicates :numeric-fluents)
  (:predicates (full) (done))
  (:functions (level))
  (:derived (full) (>= (level) 3))
  (:action fill :effect (increase (level) 1))
  (:action finish :precondition (full) :effect (done)))
"""
NUMERIC_PROBLEM = """
(define (problem n1) (:domain n) (:init (= (level) 0)) (:goal (done)))
"""


def test_numeric_conditions_in_axioms() -> None:
    task = build(NUMERIC_DOMAIN, NUMERIC_PROBLEM)
    assert [str(v) for v in task.variables] == ["(level)"]
    result = search(task, PRESETS["astar-blind"])
    assert result.status is SearchStatus.SOLVED and result.plan is not None
    assert [str(task.actions[i]) for i in result.plan] == ["(fill)"] * 3 + ["(finish)"]


# --- Heurísticas ---------------------------------------------------------------------


@pytest.mark.parametrize("device", ACCELERATED_DEVICES)
@pytest.mark.parametrize("name", TENSOR_HEURISTICS)
def test_accelerated_heuristics_match_the_reference(name: str, device: str) -> None:
    for task in (graph(), graph("(and (isolated d) (reach c))")):
        states = all_states(task)
        reference = create_heuristic(name, task, "python").evaluate_batch(states)
        assert create_heuristic(name, task, device).evaluate_batch(states) == reference


@pytest.mark.parametrize("device", ["python", *ACCELERATED_DEVICES])
def test_preferred_operators_are_real_actions(device: str) -> None:
    task = graph("(reach c)")  # solo se logra con axiomas: el plan relajado es vacío
    heuristic = create_heuristic("ff", task, device)
    for evaluation in heuristic.evaluate_batch(all_states(task)):
        assert all(0 <= o < len(task.actions) for o in evaluation.preferred)


def test_relaxation_uses_axioms() -> None:
    """(reach d) se logra con axiomas encadenados (coste 0) en cuanto dos acciones crean
    los arcos b→c y c→d: h_FF = 2, las dos son útiles (aplicables y del plan relajado), y
    el operador ficticio de los axiomas no lo es."""
    domain = """
(define (domain links) (:requirements :derived-predicates :existential-preconditions
                                      :disjunctive-preconditions)
  (:predicates (edge ?x ?y) (gap ?x ?y) (start ?x) (reach ?x))
  (:derived (reach ?x) (or (start ?x) (exists (?y) (and (reach ?y) (edge ?y ?x)))))
  (:action link :parameters (?x ?y) :precondition (gap ?x ?y) :effect (edge ?x ?y)))
"""
    problem = """
(define (problem l) (:domain links) (:objects a b c d)
  (:init (start a) (edge a b) (gap b c) (gap c d) (gap d a)) (:goal (reach d)))
"""
    task = build(domain, problem)
    for device in ("python", *ACCELERATED_DEVICES):
        evaluation = create_heuristic("ff", task, device).evaluate(task.init)
        assert evaluation.value == 2
        assert [str(task.actions[o]) for o in evaluation.preferred] == ["(link b c)", "(link c d)"]


# --- Búsqueda ------------------------------------------------------------------------


def test_optimal_search_with_derived_predicates() -> None:
    task = graph()
    result = search(task, PRESETS["astar-blind"])
    assert result.status is SearchStatus.SOLVED and result.plan is not None
    # Dos cortes: a→c y, o bien b→c, o bien a→b.
    assert len(result.plan) == 2
    state = task.init
    for index in result.plan:
        successor = task.apply(state, task.actions[index])
        assert successor is not None
        state = successor
    assert "c" in true_atoms(task, state, "isolated")


def test_the_engine_does_not_skip_axioms() -> None:
    engine = EagerSearch(graph(), PRESETS["gbfs-ff"], create_heuristic("ff", graph()), None)
    assert not engine._direct_apply


@pytest.mark.parametrize("preset", ["gbfs-ff-pref", "lazy-ff-pref", "bfws-ff-pref", "astar-blind"])
def test_plans_are_valid_for_val(preset: str, tmp_path: Path) -> None:
    if not VALIDATE.exists():
        pytest.skip("VAL no está compilado")
    goal = "(and (isolated c) (reach b))"
    task = graph(goal)
    result = search(task, PRESETS[preset])
    assert result.status is SearchStatus.SOLVED and result.plan is not None
    (tmp_path / "domain.pddl").write_text(GRAPH_DOMAIN)
    (tmp_path / "problem.pddl").write_text(graph_problem(goal))
    (tmp_path / "plan").write_text(Plan.from_indices(task, result.plan).to_ipc())
    output = subprocess.run(
        [VALIDATE, tmp_path / "domain.pddl", tmp_path / "problem.pddl", tmp_path / "plan"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    assert "Plan valid" in output


# --- Dominios temporales -------------------------------------------------------------

LAUNCH_DOMAIN = """
(define (domain rocket)
  (:requirements :typing :durative-actions :derived-predicates :universal-preconditions)
  (:types engine)
  (:predicates (charged ?e - engine) (ready) (launched))
  (:derived (ready) (forall (?e - engine) (charged ?e)))
  (:durative-action charge :parameters (?e - engine) :duration (= ?duration 5)
    :condition (and) :effect (at end (charged ?e)))
  (:durative-action launch :parameters () :duration (= ?duration 2)
    :condition (at start (ready)) :effect (at end (launched))))
"""
# El :init vacío es necesario: VAL se cae (violación de segmento) si el problema no lo tiene.
LAUNCH_PROBLEM = """
(define (problem l) (:domain rocket) (:objects e1 e2 - engine) (:init) (:goal (launched)))
"""


def test_temporal_steps_are_ordered_after_what_derived_facts_depend_on(tmp_path: Path) -> None:
    """Las dos cargas son independientes y van en paralelo; el lanzamiento lee (ready), que
    depende de las dos, así que empieza después de que terminen."""
    task = build(LAUNCH_DOMAIN, LAUNCH_PROBLEM)
    assert isinstance(task, TemporalTask) and task.axioms
    result = search(task, PRESETS["gbfs-ff-pref-tb"])
    assert result.status is SearchStatus.SOLVED and result.plan is not None
    plan = Plan.from_indices(task, result.plan)
    times = {(step.name, step.args): step.time for step in plan.steps}
    assert times[("charge", ("e1",))] == times[("charge", ("e2",))] == 0
    assert times[("launch", ())] == pytest.approx(5 + task.epsilon)
    if VALIDATE.exists():
        (tmp_path / "domain.pddl").write_text(LAUNCH_DOMAIN)
        (tmp_path / "problem.pddl").write_text(LAUNCH_PROBLEM)
        (tmp_path / "plan").write_text(plan.to_ipc())
        output = subprocess.run(
            [VALIDATE, tmp_path / "domain.pddl", tmp_path / "problem.pddl", tmp_path / "plan"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        assert "Plan valid" in output


def test_derived_facts_on_continuous_variables_use_the_discrete_engine(tmp_path: Path) -> None:
    """El programa lineal del motor temporal no puede decidir un hecho derivado de una
    variable que cambia de forma continua: la tarea pasa al motor con el tiempo
    discretizado."""
    domain = """
(define (domain tank)
  (:requirements :durative-actions :derived-predicates :numeric-fluents :continuous-effects)
  (:predicates (full) (done))
  (:functions (level))
  (:derived (full) (>= (level) 10))
  (:durative-action fill :duration (= ?duration 20) :condition (and)
    :effect (increase (level) (* #t 1)))
  (:durative-action finish :duration (= ?duration 1) :condition (at start (full))
    :effect (at end (done))))
"""
    problem = "(define (problem t) (:domain tank) (:init (= (level) 0)) (:goal (done)))"
    task = build(domain, problem)
    assert isinstance(task, DiscreteTask)
    assert any("el programa lineal no puede decidir" in note for note in task.notes)
    result = search(task, PRESETS["gbfs-ff"])
    assert result.status is SearchStatus.SOLVED and result.plan is not None
    plan = Plan.from_indices(task, result.plan)
    times = {step.name: step.time for step in plan.steps}
    assert times["finish"] is not None and times["finish"] >= 10  # el depósito, lleno
    if VALIDATE.exists():
        # Escrito con el writer: VAL exige :parameters, que este dominio no lleva.
        (tmp_path / "domain.pddl").write_text(write_pddl(parse_pddl(domain, locations=False)))
        (tmp_path / "problem.pddl").write_text(problem)
        (tmp_path / "plan").write_text(plan.to_ipc())
        output = subprocess.run(
            [VALIDATE, tmp_path / "domain.pddl", tmp_path / "problem.pddl", tmp_path / "plan"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        assert "Plan valid" in output, output
