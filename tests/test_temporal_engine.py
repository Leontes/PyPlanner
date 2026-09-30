"""Motor temporal (iteración 3b): acciones de inicio y fin, órdenes, red temporal y planes."""

import pickle
import subprocess
from pathlib import Path

import pytest

from pyplanner.errors import UnsupportedFeatureError
from pyplanner.grounding import GroundTask
from pyplanner.grounding.temporal import (
    END_SNAP,
    EPSILON,
    INSTANT,
    START_SNAP,
    TemporalState,
    TemporalTask,
)
from pyplanner.heuristics import create_heuristic
from pyplanner.plan import Plan, simulate
from pyplanner.schema import validate_document
from pyplanner.search import PRESETS, SearchStatus, search
from tests.tasks import build
from tests.test_heuristics import ACCELERATED_DEVICES
from tests.test_numeric_engine import reachable
from tests.tiers import DOMINIOS_DIR

VALIDATE = DOMINIOS_DIR.parent / "tools" / "bin" / "Validate"
requires_val = pytest.mark.skipif(not VALIDATE.exists(), reason="VAL no está compilado")

# Concurrencia obligatoria: arreglar el fusible necesita luz durante toda la acción, y la
# cerilla solo da luz mientras arde.
LIGHT_DOMAIN = """
(define (domain light) (:requirements :durative-actions {extra})
  (:predicates (lit) (fixed) (ready))
  (:durative-action light-match :parameters () :duration {match}
    :condition (at start (ready))
    :effect (and (at start (lit)) (at end (not (lit)))))
  (:durative-action fix-fuse :parameters () :duration (= ?duration 4)
    :condition (over all (lit))
    :effect (at end (fixed))))
"""
LIGHT_PROBLEM = "(define (problem p) (:domain light) (:init (ready)) (:goal (fixed)))"


def light(match: str = "(= ?duration 10)", extra: str = "") -> TemporalTask:
    task = build(LIGHT_DOMAIN.format(match=match, extra=extra), LIGHT_PROBLEM)
    assert isinstance(task, TemporalTask)
    return task


def solve(task: GroundTask, config: str = "gbfs-ff-pref-tb") -> Plan:
    result = search(task, PRESETS[config])
    assert result.status is SearchStatus.SOLVED, result.message
    assert result.plan is not None
    assert simulate(task, result.plan) is None
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


def by_name(task: GroundTask, name: str) -> int:
    return next(a.index for a in task.actions if str(a) == name)


# --- Compilación ----------------------------------------------------------------------


def test_durative_actions_become_start_and_end_snaps() -> None:
    task = light()
    names = sorted(str(a) for a in task.actions)
    assert names == [
        "(fix-fuse@end)",
        "(fix-fuse@start)",
        "(light-match@end)",
        "(light-match@start)",
    ]
    kinds = {str(a): task.snaps[a.index].kind for a in task.actions}
    assert kinds["(fix-fuse@start)"] == START_SNAP and kinds["(fix-fuse@end)"] == END_SNAP
    assert [d.name for d in task.durative] == ["fix-fuse", "light-match"]
    # El objetivo exige que no quede ninguna acción en curso.
    assert set(task.goal[0].neg_facts) == {d.running for d in task.durative}
    assert isinstance(task.init, TemporalState)


def test_an_instance_does_not_overlap_itself() -> None:
    task = light()
    start = task.actions[by_name(task, "(light-match@start)")]
    state = task.apply(task.init, start)
    assert state is not None
    assert not start.applicable(state)


# --- Red temporal y órdenes -----------------------------------------------------------


@requires_val
def test_required_concurrency(tmp_path: Path) -> None:
    task = light()
    plan = solve(task)
    steps = {s.name: s for s in plan.steps}
    match, fuse = steps["light-match"], steps["fix-fuse"]
    assert match.time == 0 and match.duration == 10
    # El arreglo empieza después de encender (ε) y termina antes de que se apague.
    assert fuse.time == pytest.approx(EPSILON) and fuse.duration == 4
    assert plan.makespan == 10
    output = val(
        LIGHT_DOMAIN.format(match="(= ?duration 10)", extra=""), LIGHT_PROBLEM, plan, tmp_path
    )
    assert "Plan valid" in output, output


def test_maximum_duration_that_cannot_contain_the_other_action() -> None:
    task = light("(= ?duration 3)")
    assert search(task, PRESETS["gbfs-ff"]).status is SearchStatus.UNSOLVABLE


@requires_val
def test_duration_inequalities(tmp_path: Path) -> None:
    match = "(and (>= ?duration 1) (<= ?duration 20))"
    task = light(match, ":duration-inequalities")
    plan = solve(task)
    steps = {s.name: s for s in plan.steps}
    duration = steps["light-match"].duration
    assert duration is not None and 1 <= duration <= 20
    # La duración más corta que contiene el arreglo: 4 + 2ε.
    assert duration == pytest.approx(4 + 2 * EPSILON)
    domain = LIGHT_DOMAIN.format(match=match, extra=":duration-inequalities")
    output = val(domain, LIGHT_PROBLEM, plan, tmp_path)
    assert "Plan valid" in output, output
    too_short = light("(and (>= ?duration 1) (<= ?duration 3))", ":duration-inequalities")
    assert search(too_short, PRESETS["gbfs-ff"]).status is SearchStatus.UNSOLVABLE


PARALLEL_DOMAIN = """
(define (domain parallel) (:requirements :typing :durative-actions)
  (:types robot)
  (:predicates (done ?r - robot) (idle ?r - robot))
  (:durative-action work :parameters (?r - robot) :duration (= ?duration 5)
    :condition (at start (idle ?r))
    :effect (and (at start (not (idle ?r))) (at end (done ?r)))))
"""
PARALLEL_PROBLEM = """
(define (problem p) (:domain parallel) (:objects a b c - robot)
  (:init (idle a) (idle b) (idle c)) (:goal (and (done a) (done b) (done c))))
"""


def test_independent_actions_run_in_parallel() -> None:
    plan = solve(build(PARALLEL_DOMAIN, PARALLEL_PROBLEM))
    assert [s.time for s in plan.steps] == [0, 0, 0]
    assert plan.makespan == 5


SEQUENTIAL_DOMAIN = """
(define (domain sequential) (:requirements :typing :durative-actions)
  (:types job)
  (:predicates (done ?j - job) (free))
  (:durative-action work :parameters (?j - job) :duration (= ?duration 5)
    :condition (at start (free))
    :effect (and (at start (not (free))) (at end (free)) (at end (done ?j)))))
"""
SEQUENTIAL_PROBLEM = """
(define (problem p) (:domain sequential) (:objects j1 j2 - job)
  (:init (free)) (:goal (and (done j1) (done j2))))
"""


@requires_val
def test_dependent_actions_are_separated_by_epsilon(tmp_path: Path) -> None:
    """Un solo trabajador: el segundo trabajo empieza ε después de que acabe el primero."""
    plan = solve(build(SEQUENTIAL_DOMAIN, SEQUENTIAL_PROBLEM))
    assert [s.time for s in plan.steps] == [0, pytest.approx(5 + EPSILON)]
    assert plan.makespan == pytest.approx(10 + EPSILON)
    output = val(SEQUENTIAL_DOMAIN, SEQUENTIAL_PROBLEM, plan, tmp_path)
    assert "Plan valid" in output, output


# --- Invariantes ----------------------------------------------------------------------

BLOWN_DOMAIN = LIGHT_DOMAIN.format(match="(= ?duration 10)", extra="").replace(
    "(:durative-action fix-fuse",
    "(:action blow :parameters () :precondition (lit) :effect (not (lit)))\n"
    "  (:durative-action fix-fuse",
)


def test_steps_that_break_an_invariant_are_not_applicable() -> None:
    task = build(BLOWN_DOMAIN, LIGHT_PROBLEM)
    assert isinstance(task, TemporalTask)
    state = task.init
    for name in ("(light-match@start)", "(fix-fuse@start)"):
        state = task.apply(state, task.actions[by_name(task, name)])
        assert state is not None
    blow = task.actions[by_name(task, "(blow)")]
    assert task.snaps[blow.index].kind == INSTANT
    assert blow.applicable(state)
    assert task.apply(state, blow) is None  # apagaría la luz con el arreglo en curso


def test_instant_actions_in_temporal_plans() -> None:
    domain = LIGHT_DOMAIN.format(match="(= ?duration 10)", extra="").replace(
        "(:durative-action fix-fuse",
        "(:action strike :parameters () :precondition (and) :effect (ready))\n"
        "  (:durative-action fix-fuse",
    )
    task = build(domain, LIGHT_PROBLEM.replace("(:init (ready))", "(:init)"))
    plan = solve(task)
    strike = next(s for s in plan.steps if s.name == "strike")
    assert strike.time == 0 and strike.duration is None
    assert plan.to_ipc().splitlines()[0] == "0.000: (strike)"


# --- Numérico, ?duration y efectos condicionales --------------------------------------

TRUCK_DOMAIN = """
(define (domain truck) (:requirements :typing :durative-actions :numeric-fluents
                                       :conditional-effects :negative-preconditions)
  (:types place)
  (:predicates (at ?p - place) (clean) (washed))
  (:functions (fuel) (distance ?a ?b - place) (rate))
  (:durative-action drive :parameters (?a ?b - place)
    :duration (= ?duration (distance ?a ?b))
    :condition (and (at start (at ?a)) (at start (>= (fuel) (* (rate) (distance ?a ?b)))))
    :effect (and (at start (not (at ?a))) (at end (at ?b))
                 (at end (decrease (fuel) (* (rate) ?duration)))
                 (when (at start (clean)) (at end (washed)))
                 (at start (not (clean))))))
"""
TRUCK_PROBLEM = """
(define (problem p) (:domain truck) (:objects home shop - place)
  (:init (at home) (clean) (= (fuel) 50) (= (rate) 2)
         (= (distance home shop) 7) (= (distance shop home) 7))
  (:goal (and (at shop) (washed))) {metric})
"""


@requires_val
def test_numeric_durative_actions(tmp_path: Path) -> None:
    problem = TRUCK_PROBLEM.format(metric="")
    task = build(TRUCK_DOMAIN, problem)
    plan = solve(task)
    assert [(s.name, s.args, s.duration) for s in plan.steps] == [("drive", ("home", "shop"), 7)]
    output = val(TRUCK_DOMAIN, problem, plan, tmp_path)
    assert "Plan valid" in output, output


def test_start_conditions_of_end_effects_are_remembered() -> None:
    """``(when (at start (clean)) (at end (washed)))`` aunque la acción borra ``clean`` al
    empezar: la condición se evalúa al empezar."""
    task = build(TRUCK_DOMAIN, TRUCK_PROBLEM.format(metric=""))
    state = task.init
    for name in ("(drive@start home shop)", "(drive@end home shop)"):
        state = task.apply(state, task.actions[by_name(task, name)])
        assert state is not None
    atoms = {str(a) for a in task.atoms(state)}
    assert "(washed)" in atoms and "(clean)" not in atoms
    assert state.values == (36.0,)  # 50 - 2·7


def test_duration_depending_on_the_state() -> None:
    """Duración de una función que cambia: se evalúa al empezar."""
    domain = (
        TRUCK_DOMAIN.replace("(= ?duration (distance ?a ?b))", "(= ?duration (/ (fuel) 10))")
        .replace("(at start (>= (fuel) (* (rate) (distance ?a ?b))))", "(at start (>= (fuel) 0))")
        .replace("(at end (decrease (fuel) (* (rate) ?duration)))", "(at end (decrease (fuel) 20))")
    )
    plan = solve(build(domain, TRUCK_PROBLEM.format(metric="")))
    assert plan.steps[0].duration == 5  # 50 / 10


def test_makespan_metric() -> None:
    problem = TRUCK_PROBLEM.format(metric="(:metric minimize (total-time))")
    plan = solve(build(TRUCK_DOMAIN, problem))
    assert plan.metric == plan.makespan == 7
    other = build(TRUCK_DOMAIN, TRUCK_PROBLEM.format(metric="(:metric minimize (fuel))"))
    assert any("solo se usa la métrica" in note for note in other.notes)
    assert solve(other).metric is None


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            ("(at end (at ?b))", "(at end (at ?b)) (decrease (fuel) (* #t 1))"),
            "efectos continuos",
        ),
        (
            ("(= ?duration (distance ?a ?b))", "(<= ?duration (distance ?a ?b))"),
            "duración fija",
        ),
        (
            ("(when (at start (clean))", "(when (over all (clean))"),
            "'over all'",
        ),
    ],
)
def test_unsupported_temporal_features(change: tuple[str, str], message: str) -> None:
    domain = TRUCK_DOMAIN.replace(*change).replace(
        ":conditional-effects", ":conditional-effects :continuous-effects :duration-inequalities"
    )
    with pytest.raises(UnsupportedFeatureError, match=message):
        build(domain, TRUCK_PROBLEM.format(metric=""))


def test_duration_of_changing_functions_cannot_be_used_in_effects() -> None:
    domain = TRUCK_DOMAIN.replace("(= ?duration (distance ?a ?b))", "(= ?duration (fuel))")
    with pytest.raises(UnsupportedFeatureError, match="no depende de funciones que cambian"):
        build(domain, TRUCK_PROBLEM.format(metric=""))


# --- Salida, heurísticas y procesos ---------------------------------------------------


def test_temporal_plan_output() -> None:
    plan = solve(light())
    assert plan.to_ipc().splitlines() == [
        "0.000: (light-match) [10.000]",
        "0.010: (fix-fuse) [4.000]",
        "; makespan = 10",
    ]
    document = plan.to_dict()
    validate_document(document)
    assert document["makespan"] == 10
    assert document["steps"][1] == {"name": "fix-fuse", "args": [], "time": 0.01, "duration": 4}


@pytest.mark.skipif(not ACCELERATED_DEVICES, reason="ni Numba ni PyTorch están instalados")
@pytest.mark.parametrize("device", ACCELERATED_DEVICES)
@pytest.mark.parametrize("name", ["hmax", "hadd", "ff"])
def test_accelerated_heuristics_on_temporal_tasks(name: str, device: str) -> None:
    for task in (light(), build(TRUCK_DOMAIN, TRUCK_PROBLEM.format(metric=""))):
        states = reachable(task)
        reference = create_heuristic(name, task, "python").evaluate_batch(states)
        assert create_heuristic(name, task, device).evaluate_batch(states) == reference


def test_temporal_tasks_and_states_can_be_pickled() -> None:
    """El portfolio con ``spawn`` envía la tarea a los procesos hijos."""
    task = light()
    copy = pickle.loads(pickle.dumps(task))
    assert copy == task
    state = task.apply(task.init, task.actions[by_name(task, "(light-match@start)")])
    restored = pickle.loads(pickle.dumps(state))
    assert restored == state and restored.timeline.earliest == state.timeline.earliest
