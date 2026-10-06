"""Restricciones de trayectoria duras (PDDL 3) en el motor: monitores, heurística, planes
clásicos y temporales, validados con VAL."""

import subprocess
from pathlib import Path

import pytest

from pyplanner.grounding import GroundTask
from pyplanner.grounding.temporal import TemporalTask
from pyplanner.heuristics import create_heuristic
from pyplanner.plan import Plan, simulate
from pyplanner.search import PRESETS, SearchStatus, search
from tests.tasks import build
from tests.tiers import DOMINIOS_DIR

VALIDATE = DOMINIOS_DIR.parent / "tools" / "bin" / "Validate"

# Lámparas que se encienden y se apagan.
SWITCHES = """
(define (domain switches)
  (:requirements :typing :negative-preconditions :disjunctive-preconditions :constraints)
  (:types lamp)
  (:predicates (on ?l - lamp))
  (:action turn-on :parameters (?l - lamp) :precondition (not (on ?l)) :effect (on ?l))
  (:action turn-off :parameters (?l - lamp) :precondition (on ?l) :effect (not (on ?l))))
"""


def switches(constraints: str, goal: str = "(and (on a) (on b))", init: str = "") -> str:
    return f"""
(define (problem lamps) (:domain switches)
  (:objects a b c - lamp)
  (:init {init})
  (:goal {goal})
  (:constraints {constraints}))
"""


def solve(domain: str, problem: str, preset: str = "astar-blind") -> tuple[GroundTask, Plan]:
    task = build(domain, problem)
    result = search(task, PRESETS[preset])
    assert result.status is SearchStatus.SOLVED, result.status
    assert result.plan is not None and simulate(task, result.plan) is None
    return task, Plan.from_indices(task, result.plan)


def names(plan: Plan) -> list[str]:
    return [f"({step.name} {' '.join(step.args)})" for step in plan.steps]


def val(domain: str, problem: str, plan: Plan, tmp_path: Path) -> str:
    files = {"domain.pddl": domain, "problem.pddl": problem, "plan": plan.to_ipc()}
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    return subprocess.run(
        [str(VALIDATE), *(str(tmp_path / f) for f in files)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout


def assert_valid(domain: str, problem: str, plan: Plan, tmp_path: Path) -> None:
    if VALIDATE.exists():
        output = val(domain, problem, plan, tmp_path)
        assert "Plan valid" in output, output


# --- Restricciones sin tiempo ----------------------------------------------------------


@pytest.mark.parametrize(
    ("constraints", "expected"),
    [
        # Sin restricciones: dos pasos.
        ("(and)", ["(turn-on a)", "(turn-on b)"]),
        # c tiene que encenderse y apagarse después, y nunca a la vez que a.
        (
            "(and (always (not (and (on a) (on c)))) (sometime (on c))"
            " (sometime-after (on c) (not (on c))))",
            ["(turn-on c)", "(turn-off c)", "(turn-on a)", "(turn-on b)"],
        ),
        # b no puede encenderse antes de que lo haya estado c.
        ("(sometime-before (on b) (on c))", ["(turn-on c)", "(turn-on a)", "(turn-on b)"]),
        # at end: el objetivo crece.
        ("(at end (not (on c)))", ["(turn-on a)", "(turn-on b)"]),
    ],
)
def test_untimed_constraints(constraints: str, expected: list[str], tmp_path: Path) -> None:
    problem = switches(constraints)
    _, plan = solve(SWITCHES, problem)
    assert len(plan) == len(expected)
    assert sorted(names(plan)) == sorted(expected)
    assert_valid(SWITCHES, problem, plan, tmp_path)


def test_at_most_once() -> None:
    """a tiene que estar encendida dos veces separadas (al principio y al final): con
    at-most-once no hay plan."""
    problem = switches(
        "(and (at-most-once (on a)) (sometime-after (on b) (not (on a))))", init="(on a)"
    )
    task = build(SWITCHES, problem)
    assert search(task, PRESETS["astar-blind"]).status is SearchStatus.UNSOLVABLE


def test_violated_initial_state() -> None:
    task = build(SWITCHES, switches("(always (not (on a)))", init="(on a)"))
    assert task.goal == ()
    assert "estado inicial incumple" in " ".join(task.notes)


def test_monitor_rules_guide_the_heuristic() -> None:
    """sometime (on c) cuesta un paso más en la relajación: h_FF lo cuenta."""
    plain = build(SWITCHES, switches("(and)"))
    constrained = build(SWITCHES, switches("(sometime (on c))"))
    for device in ("python", "numba"):
        assert create_heuristic("ff", plain, device).evaluate(plain.init).value == 2
        assert create_heuristic("ff", constrained, device).evaluate(constrained.init).value == 3


@pytest.mark.parametrize("preset", ["gbfs-ff-pref", "lazy-ff-pref", "bfws-ff-pref", "iw2"])
def test_constraints_with_every_search(preset: str, tmp_path: Path) -> None:
    problem = switches(
        "(and (always (not (and (on a) (on c)))) (sometime (on c))"
        " (sometime-after (on c) (not (on c))) (sometime-before (on b) (on c)))"
    )
    task = build(SWITCHES, problem)
    result = search(task, PRESETS[preset])
    if preset == "iw2" and result.status is SearchStatus.FAILED:
        return  # IW es incompleta
    assert result.status is SearchStatus.SOLVED and result.plan is not None
    assert_valid(SWITCHES, problem, Plan.from_indices(task, result.plan), tmp_path)


# --- Restricciones con tiempo en tareas clásicas (instante = número de pasos) ------------


@pytest.mark.parametrize(
    ("constraints", "length"),
    [
        # c encendida en uno de los dos primeros estados tras el inicial.
        ("(within 1 (on c))", 3),
        # Desde el paso 2, a tiene que seguir encendida.
        ("(hold-after 1 (on a))", 2),
        # En el estado del paso 1 (instantes [1, 2)), c encendida.
        ("(hold-during 1 2 (on c))", 3),
        # Cada vez que se enciende c, a como mucho un paso después.
        ("(and (sometime (on c)) (always-within 1 (on c) (on a)))", 3),
    ],
)
def test_timed_constraints_in_classical_tasks(
    constraints: str, length: int, tmp_path: Path
) -> None:
    problem = switches(constraints)
    task, plan = solve(SWITCHES, problem)
    # always-within lleva su propio contador; los demás, el de pasos de la tarea.
    assert (task.monitor_clock is None) == ("always-within" in constraints)
    assert len(plan) == length
    if "hold-after" not in constraints:  # VAL no valida bien hold-after (falla o se cae)
        assert_valid(SWITCHES, problem, plan, tmp_path)


def test_deadlines_that_cannot_be_met() -> None:
    task = build(SWITCHES, switches("(and (within 0 (on c)))"))
    assert search(task, PRESETS["astar-blind"]).status is SearchStatus.UNSOLVABLE
    # Con plazo 0, c apagada en el mismo estado en que se enciende: imposible.
    constraints = "(and (sometime (on c)) (always-within 0 (on c) (not (on c))))"
    task = build(SWITCHES, switches(constraints))
    assert search(task, PRESETS["astar-blind"]).status is SearchStatus.UNSOLVABLE


# --- Tareas temporales ---------------------------------------------------------------

OVEN = """
(define (domain oven)
  (:requirements :durative-actions :constraints)
  (:predicates (hot) (baked) (cooled))
  (:durative-action heat :parameters () :duration (= ?duration 10)
    :condition (at start (not (hot))) :effect (at end (hot)))
  (:durative-action bake :parameters () :duration (= ?duration 5)
    :condition (over all (hot)) :effect (at end (baked)))
  (:durative-action cool :parameters () :duration (= ?duration 4)
    :condition (at start (hot)) :effect (at end (and (not (hot)) (cooled)))))
"""


def oven(constraints: str) -> str:
    return f"""
(define (problem o1) (:domain oven)
  (:init)
  (:goal (and (baked) (cooled)))
  (:constraints {constraints}))
"""


def test_temporal_within(tmp_path: Path) -> None:
    domain = OVEN.replace(":constraints)", ":constraints :negative-preconditions)")
    problem = oven("(within 20 (baked))")
    task, plan = solve(domain, problem, "gbfs-ff-pref-tb")
    assert isinstance(task, TemporalTask)
    assert task.til_times == (20.0,)  # el instante es un TIL sin literales
    assert_valid(domain, problem, plan, tmp_path)
    # Sin plan (la búsqueda temporal no lo demuestra: ver test_temporal_engine).
    impossible = build(domain, oven("(within 12 (baked))"))
    assert search(impossible, PRESETS["astar-blind"]).status is SearchStatus.FAILED


def test_temporal_always_within(tmp_path: Path) -> None:
    """Cuando el horno se calienta, hay que hornear antes de 8 unidades: la red temporal
    recibe la arista del plazo."""
    domain = OVEN.replace(":constraints)", ":constraints :negative-preconditions)")
    problem = oven("(always-within 8 (hot) (baked))")
    task, plan = solve(domain, problem, "gbfs-ff-pref-tb")
    times = {step.name: step.time for step in plan.steps}
    assert times["bake"] is not None and times["heat"] is not None
    assert times["bake"] + 5 <= times["heat"] + 10 + 8 + 1e-6
    assert_valid(domain, problem, plan, tmp_path)
    too_short = build(domain, oven("(always-within 4 (hot) (baked))"))
    assert search(too_short, PRESETS["astar-blind"]).status is SearchStatus.FAILED  # sin plan


def test_temporal_untimed_constraints(tmp_path: Path) -> None:
    domain = OVEN.replace(":constraints)", ":constraints :negative-preconditions)")
    problem = oven("(sometime-before (cooled) (baked))")
    _, plan = solve(domain, problem, "gbfs-ff-pref-tb")
    times = {step.name: step.time for step in plan.steps}
    assert times["bake"] is not None and times["cool"] is not None
    assert times["bake"] + 5 < times["cool"] + 4  # se hornea antes de que acabe de enfriarse
    assert_valid(domain, problem, plan, tmp_path)


def test_trucks() -> None:
    path = DOMINIOS_DIR / "tier5-pddl3" / "constraints" / "trucks"
    domain, problem = (path / "domain.pddl").read_text(), (path / "p01.pddl").read_text()
    task, plan = solve(domain, problem, "bfws-ff-pref")
    assert len(task.monitors) == 3
    assert plan.makespan is not None
    # La búsqueda perezosa agota la cola sin encontrarlo: no puede decir que no hay plan.
    lazy = search(task, PRESETS["lazy-ff-pref"])
    assert lazy.status is SearchStatus.FAILED
