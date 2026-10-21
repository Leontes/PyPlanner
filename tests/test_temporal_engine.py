"""Motor temporal: acciones de inicio y fin, órdenes, red temporal y planes."""

import pickle
import subprocess
from pathlib import Path

import pytest

from pyplanner.errors import UnsupportedFeatureError
from pyplanner.grounding import GroundTask, State
from pyplanner.grounding.temporal import (
    END_SNAP,
    EPSILON,
    INSTANT,
    START_SNAP,
    TemporalState,
    TemporalTask,
)
from pyplanner.heuristics import create_heuristic
from pyplanner.loader.ast import FluentTerm
from pyplanner.plan import Plan, simulate
from pyplanner.schema import validate_document
from pyplanner.search import PRESETS, SearchStatus, search
from tests.catalog import VALIDATE, collection_dir
from tests.tasks import build
from tests.test_heuristics import ACCELERATED_DEVICES
from tests.test_numeric_engine import reachable

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
    """No hay plan, pero la búsqueda temporal no lo puede demostrar: ha descartado estados
    repetidos, que con otra línea temporal quizá sí llegaban (memoización de POPF)."""
    task = light("(= ?duration 3)")
    result = search(task, PRESETS["gbfs-ff"])
    assert result.status is SearchStatus.FAILED
    assert "no se demuestra que no lo haya" in result.message


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
    assert search(too_short, PRESETS["gbfs-ff"]).status is SearchStatus.FAILED  # sin plan


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
        successor = task.apply(state, task.actions[by_name(task, name)])
        assert successor is not None
        state = successor
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
        successor = task.apply(state, task.actions[by_name(task, name)])
        assert successor is not None
        state = successor
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
            ("(at end (at ?b))", "(at end (at ?b)) (increase (fuel) (* #t ?duration))"),
            "ritmo × tiempo",
        ),
    ],
)
def test_unsupported_temporal_features(change: tuple[str, str], message: str) -> None:
    domain = (
        TRUCK_DOMAIN.replace("(= ?duration (distance ?a ?b))", "(<= ?duration 9)")
        .replace(*change)
        .replace(
            ":conditional-effects",
            ":conditional-effects :duration-inequalities :continuous-effects",
        )
    )
    with pytest.raises(UnsupportedFeatureError, match=message):
        build(domain, TRUCK_PROBLEM.format(metric=""))


def test_duration_equal_to_a_variable_that_the_action_changes() -> None:
    """``(= ?duration (fuel))`` con ``?duration`` en un efecto sobre ``fuel``: ``fuel`` pasa
    a ser una variable lineal y la duración, una incógnita igual a su valor al empezar."""
    domain = TRUCK_DOMAIN.replace("(= ?duration (distance ?a ?b))", "(= ?duration (fuel))")
    task = build(domain, TRUCK_PROBLEM.format(metric=""))
    assert isinstance(task, TemporalTask) and task.linear is not None
    plan = solve(task)
    assert plan.steps[0].duration == pytest.approx(50)


# --- Programación lineal: efectos continuos y duraciones no fijas ----------------------
# Ojo: VAL se cae (violación de segmento) si una función se llama como el dominio; por
# eso la función es battery_level y no battery.

CHARGE_DOMAIN = """
(define (domain battery) (:requirements :durative-actions :numeric-fluents :continuous-effects)
  (:predicates (used) (idle))
  (:functions (battery_level))
  (:durative-action charge :parameters () :duration (= ?duration 10)
    :condition (at start (idle))
    :effect (and (at start (not (idle))) (at end (idle)) (increase (battery_level) (* #t 2))))
  (:action use :parameters () :precondition (>= (battery_level) 5) :effect (used)))
"""
CHARGE_PROBLEM = """
(define (problem p) (:domain battery) (:init (idle) (= (battery_level) 0)) (:goal (used)))
"""


@requires_val
def test_waiting_for_a_continuous_change(tmp_path: Path) -> None:
    """``use`` necesita 5 de batería: se programa en cuanto la carga llega (2 por unidad)."""
    task = build(CHARGE_DOMAIN, CHARGE_PROBLEM)
    assert isinstance(task, TemporalTask) and task.linear is not None
    plan = solve(task)
    times = {s.name: s.time for s in plan.steps}
    assert times == {"charge": 0, "use": pytest.approx(2.5)}
    output = val(CHARGE_DOMAIN, CHARGE_PROBLEM, plan, tmp_path)
    assert "Plan valid" in output, output


GENERATOR_DOMAIN = """
(define (domain generator) (:requirements :durative-actions :numeric-fluents :continuous-effects)
  (:predicates (done) (full))
  (:functions (fuel))
  (:durative-action generate :parameters () :duration (= ?duration 10)
    :condition (over all (>= (fuel) 0))
    :effect (and (decrease (fuel) (* #t 1)) (at end (done))))
  (:durative-action refuel :parameters () :duration (= ?duration 6)
    :condition (at start (full))
    :effect (and (at start (not (full))) (increase (fuel) (* #t 1)))))
"""
GENERATOR_PROBLEM = """
(define (problem p) (:domain generator) (:init (full) (= (fuel) {fuel})) (:goal (done)))
"""


@requires_val
def test_continuous_invariant_needs_concurrent_refuelling(tmp_path: Path) -> None:
    """Con 5 de combustible, generar 10 unidades de tiempo exige repostar a la vez."""
    problem = GENERATOR_PROBLEM.format(fuel=5)
    plan = solve(build(GENERATOR_DOMAIN, problem))
    assert {s.name for s in plan.steps} == {"generate", "refuel"}
    output = val(GENERATOR_DOMAIN, problem, plan, tmp_path)
    assert "Plan valid" in output, output


def test_continuous_invariant_that_cannot_hold() -> None:
    """Con 3 de combustible no basta ni repostando (6 unidades como mucho): 3 + 6 < 10.

    El espacio de estados es infinito (el combustible toma valores distintos), así que la
    búsqueda no puede demostrar que no hay plan: se comprueba que no encuentra ninguno."""
    task = build(GENERATOR_DOMAIN, GENERATOR_PROBLEM.format(fuel=3))
    config = PRESETS["gbfs-ff"].with_changes(max_expansions=300)
    assert search(task, config).status is not SearchStatus.SOLVED


WORK_DOMAIN = """
(define (domain workshop) (:requirements :durative-actions :numeric-fluents :duration-inequalities)
  (:predicates (ready))
  (:functions (progress))
  (:durative-action work :parameters () :duration (and (>= ?duration 1) (<= ?duration 10))
    :condition (at start (ready))
    :effect (and (at start (not (ready))) (at end (ready))
                 (at end (increase (progress) ?duration)))))
"""
WORK_PROBLEM = """
(define (problem p) (:domain workshop) (:init (ready) (= (progress) 0))
  (:goal (>= (progress) {goal})))
"""


@requires_val
def test_free_duration_chosen_by_the_linear_program(tmp_path: Path) -> None:
    """El objetivo obliga a alargar la acción: la duración la elige el programa lineal."""
    problem = WORK_PROBLEM.format(goal=6)
    plan = solve(build(WORK_DOMAIN, problem))
    [step] = plan.steps
    assert step.name == "work" and step.duration == pytest.approx(6)
    output = val(WORK_DOMAIN, problem, plan, tmp_path)
    assert "Plan valid" in output, output


def test_free_duration_needs_two_actions() -> None:
    """Con un máximo de 10 por acción, 15 exige dos acciones seguidas."""
    plan = solve(build(WORK_DOMAIN, WORK_PROBLEM.format(goal=15)))
    durations = [s.duration for s in plan.steps]
    assert len(durations) == 2 and sum(durations) == pytest.approx(15)  # type: ignore[arg-type]


def test_relaxation_integrates_continuous_effects() -> None:
    """La heurística ve la carga como un efecto al terminar (2 × 10). Sin eso, ``use``
    (batería ≥ 5) sería inalcanzable en la relajación y el inicio, un callejón sin salida."""
    task = build(CHARGE_DOMAIN, CHARGE_PROBLEM)
    relaxation = task.relaxation
    assert relaxation is not task
    end = next(a for a in relaxation.actions if a.name == "charge@end")
    assert [a.op for a in end.numeric] == ["increase"]
    assert task.actions[end.index].numeric == ()  # en la búsqueda lo decide el programa lineal
    # Plan relajado: inicio y fin de la carga, y usar.
    assert create_heuristic("ff", task, "python").evaluate(task.init).value == 3


# --- Certificación: combinaciones poco habituales ------------------------------------
# No hay benchmarks públicos que las usen: el dominio ad hoc greenhouse (propios/greenhouse
# en el catálogo) las reúne todas. Aquí se prueba con más problemas y pieza a pieza.

GREENHOUSE_DOMAIN = (collection_dir("propios/greenhouse") / "domain.pddl").read_text(
    encoding="utf-8"
)
GREENHOUSE_PROBLEM = """
(define (problem gh1) (:domain greenhouse) (:objects t1 t2 - tank)
  (:init {init} (idle) (= (water t1) 0) (= (water t2) 0) (= (reserve) 0) (= (budget) 10))
  (:goal (and {goal})))
"""


def greenhouse(init: str = "(raining) (sensor-on)", goal: str = "(logged)") -> TemporalTask:
    task = build(GREENHOUSE_DOMAIN, GREENHOUSE_PROBLEM.format(init=init, goal=goal))
    assert isinstance(task, TemporalTask)
    return task


@requires_val
@pytest.mark.parametrize(
    ("init", "goal"),
    [
        ("(raining) (sensor-on)", "(logged) (>= (water t1) 6) (>= (water t2) 6) (>= (budget) 0)"),
        ("(sensor-on)", "(logged) (>= (water t1) 3) (>= (water t2) 3)"),
        ("(raining)", "(>= (water t1) 5) (>= (budget) 5)"),
    ],
    ids=["todo", "sin-lluvia", "sin-sensor"],
)
def test_certification_domain(init: str, goal: str, tmp_path: Path) -> None:
    """Efectos continuos dentro de forall y condicionales, when con over all, ?duration no
    fijo al empezar, una duración que depende de una variable continua y un efecto
    condicional sobre una variable continua: planes válidos para VAL."""
    task = greenhouse(init, goal)
    plan = solve(task)
    problem = GREENHOUSE_PROBLEM.format(init=init, goal=goal)
    output = val(GREENHOUSE_DOMAIN, problem, plan, tmp_path)
    assert "Plan valid" in output, output


def apply_all(task: TemporalTask, names: list[str]) -> TemporalState:
    state = task.init
    assert isinstance(state, TemporalState)
    for name in names:
        successor = task.apply(state, task.actions[by_name(task, name)])
        assert successor is not None, name
        state = successor
    return state


def test_over_all_conditional_effect_is_cancelled_when_the_condition_breaks() -> None:
    task = greenhouse()
    kept = apply_all(task, ["(collect-rain@start)", "(collect-rain@end)"])
    assert "(logged)" in {str(a) for a in task.atoms(kept)}
    broken = apply_all(task, ["(collect-rain@start)", "(switch-off)", "(collect-rain@end)"])
    assert "(logged)" not in {str(a) for a in task.atoms(broken)}


def test_conditional_continuous_rate_and_conditional_linear_effect() -> None:
    """Con lluvia, la reserva sube a 2 por unidad (8 en 4 unidades) y el sensor encendido
    al terminar suma 1; sin lluvia y con el sensor apagado, solo 4."""
    reserve = greenhouse().variable_index[FluentTerm("reserve")]
    rain = apply_all(greenhouse(), ["(collect-rain@start)", "(collect-rain@end)"])
    assert rain.values[reserve] == pytest.approx(9)
    dry = greenhouse(init="(sensor-on)")
    state = apply_all(dry, ["(collect-rain@start)", "(switch-off)", "(collect-rain@end)"])
    assert state.values[reserve] == pytest.approx(4)


def test_duration_bounded_by_a_continuous_variable() -> None:
    """``(<= ?duration (reserve))`` con la reserva a 0: regar es imposible sin recoger."""
    task = greenhouse()
    # La duración mínima es 1 y la máxima 0: el programa lineal lo rechaza al empezar.
    assert task.apply(task.init, task.actions[by_name(task, "(irrigate@start)")]) is None
    # Tras recoger lluvia 4 unidades (reserva 9), sí se puede.
    state = apply_all(task, ["(collect-rain@start)", "(collect-rain@end)"])
    assert task.apply(state, task.actions[by_name(task, "(irrigate@start)")]) is not None


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            ("(when (at end (sensor-on))", "(when (at end (> (reserve) 1))"),
            "programación lineal entera",
        ),
        (
            ("(over all (>= (reserve) 0))", "(over all (or (>= (reserve) 0) (sensor-on)))"),
            "invariante disyuntiva",
        ),
    ],
)
def test_limits_of_linear_programming(change: tuple[str, str], message: str) -> None:
    """Condiciones que dependen de la solución del programa lineal: harían falta variables
    enteras (disyunciones)."""
    domain = GREENHOUSE_DOMAIN.replace(*change).replace(
        ":negative-preconditions", ":negative-preconditions :disjunctive-preconditions"
    )
    with pytest.raises(UnsupportedFeatureError, match=message):
        build(domain, GREENHOUSE_PROBLEM.format(init="(raining) (sensor-on)", goal="(logged)"))


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
    assert state is not None
    restored = pickle.loads(pickle.dumps(state))
    assert restored == state and restored.timeline.earliest == state.timeline.earliest


# --- Órdenes compatibles con VAL ------------------------------------------------------

PRODUCE_DOMAIN = """
(define (domain factory) (:requirements :durative-actions)
  (:predicates (p) (q) (item ?x))
  (:durative-action produce :parameters (?x) :duration (= ?duration 5)
    :condition (at start (item ?x)) :effect (at end (p)))
  (:durative-action use :parameters () :duration (= ?duration 1)
    :condition (at start (p)) :effect (at end (q))))
"""
PRODUCE_PROBLEM = """
(define (problem factory-1) (:domain factory) (:objects a b) (:init (item a) (item b))
  (:goal (q)))
"""


def run_steps(task: TemporalTask, names: list[str]) -> tuple[list[int], TemporalState]:
    state: State = task.init
    indices = []
    for name in names:
        index = by_name(task, name)
        successor = task.apply(state, task.actions[index])
        assert successor is not None, name
        state = successor
        indices.append(index)
    assert isinstance(state, TemporalState)
    return indices, state


def test_an_action_does_not_overlap_with_itself_in_time() -> None:
    """Un segundo inicio de la misma acción va después de su fin anterior también en el
    tiempo, no solo en la secuencia (los hechos running@a cuentan para las órdenes)."""
    task = build(PRODUCE_DOMAIN, PRODUCE_PROBLEM)
    assert isinstance(task, TemporalTask)
    names = ["(produce@start a)", "(produce@end a)", "(produce@start a)"]
    _, state = run_steps(task, names)
    times = state.timeline.earliest
    assert times[3] >= times[2] + task.epsilon - 1e-9


def test_a_write_that_keeps_the_value_is_separated_from_readers(tmp_path: Path) -> None:
    """El fin de produce b vuelve a añadir (p), que ya es cierto y que lee el inicio de use:
    no pueden ser simultáneos (VAL lo rechaza como mutex)."""
    task = build(PRODUCE_DOMAIN, PRODUCE_PROBLEM)
    assert isinstance(task, TemporalTask)
    names = [
        "(produce@start a)",
        "(produce@start b)",
        "(produce@end a)",
        "(use@start)",
        "(produce@end b)",
        "(use@end)",
    ]
    indices, state = run_steps(task, names)
    times = state.timeline.earliest
    assert times[5] >= times[4] + task.epsilon - 1e-9  # fin de produce b tras el inicio de use
    if VALIDATE.exists():
        plan = Plan.from_indices(task, tuple(indices))
        assert "Plan valid" in val(PRODUCE_DOMAIN, PRODUCE_PROBLEM, plan, tmp_path)


# --- Guarda de inicio -----------------------------------------------------------------

# Hornear una bandeja exige, todo el tiempo, que esté en ese horno (estático) y que el horno
# esté cerrado; ninguna condición 'at start'.
GUARDED = """
(define (domain guarded)
  (:requirements :typing :durative-actions :preferences)
  (:types oven tray)
  (:predicates (closed ?o - oven) (baked ?t - tray) (in ?t - tray ?o - oven))
  (:action close :parameters (?o - oven) :precondition (and) :effect (closed ?o))
  (:durative-action bake :parameters (?t - tray ?o - oven) :duration (= ?duration 2)
    :condition (and (over all (in ?t ?o)) (over all (closed ?o)))
    :effect (at end (baked ?t))))
"""
GUARDED_PROBLEM = """
(define (problem guarded-1) (:domain guarded) (:objects o1 o2 - oven t1 - tray)
  (:init (in t1 o1)) (:goal (and (baked t1) (preference shut (closed o2))))
  (:metric minimize (* 2 (is-violated shut))))
"""


def test_start_guard() -> None:
    """Las condiciones 'over all' que el inicio no toca se exigen ya al empezar: no se
    instancia el inicio de la bandeja en el otro horno (nunca podría terminar), y el inicio
    exige el horno cerrado (en la búsqueda y en la heurística)."""
    task = build(GUARDED, GUARDED_PROBLEM)
    assert isinstance(task, TemporalTask)
    names = [str(f) for f in task.facts]
    assert [n for n in names if n.startswith("(running@")] == ["(running@bake t1 o1)"]
    start = next(a for a in task.actions if str(a) == "(bake@start t1 o1)")
    assert names.index("(closed o1)") in start.pre_facts
    result = search(task, PRESETS["gbfs-ff"])
    assert result.plan is not None
    steps = [str(s) for s in Plan.from_indices(task, result.plan).steps]
    assert "(close o1)" in steps and "(bake t1 o1)" in steps


def test_unit_costs_include_the_relaxation() -> None:
    """Con variables lineales, la heurística usa otras acciones (las de la relajación): la
    copia a coste unitario del primer plan de la búsqueda anytime también las cambia. Si
    no, con preferencias, guiaba con las milésimas por paso de la métrica."""
    from pyplanner.search.anytime import unit_cost

    domain = """
(define (domain heater-pref)
  (:requirements :numeric-fluents :durative-actions :continuous-effects :preferences)
  (:predicates (warm))
  (:functions (temp))
  (:durative-action heat :parameters () :duration (= ?duration 4)
    :condition (and)
    :effect (and (increase (temp) (* #t 2)) (at end (warm)))))
"""
    problem = """
(define (problem heater-pref-1) (:domain heater-pref) (:init (= (temp) 0))
  (:goal (and (warm) (preference hot (>= (temp) 100))))
  (:metric minimize (* 3 (is-violated hot))))
"""
    task = build(domain, problem)
    assert isinstance(task, TemporalTask) and task.relaxation_actions
    assert any(a.cost != 1.0 for a in task.relaxation_actions)
    assert all(a.cost == 1.0 for a in unit_cost(task).relaxation.actions)
