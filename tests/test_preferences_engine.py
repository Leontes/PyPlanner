"""Preferencias (PDDL 3) en el motor: compilación a costes, monitores blandos, poda por cota
y búsqueda anytime, con la métrica de los planes comparada con la de VAL."""

import re
import subprocess
from pathlib import Path

import pytest

from pyplanner.cli import main
from pyplanner.errors import UnsupportedFeatureError
from pyplanner.frontend import read_domain, read_problem
from pyplanner.grounding import GroundTask, State
from pyplanner.grounding.monitors import Monitor, MonitorEvaluator
from pyplanner.grounding.preferences import COLLECT, END, FORGO, split_preferences
from pyplanner.grounding.task import Condition
from pyplanner.grounding.temporal import TemporalTask
from pyplanner.loader import load_domain, load_problem
from pyplanner.loader.ast import And, Atom, Forall, Parameter, Preference, Variable
from pyplanner.pddl import parse_domain, parse_problem
from pyplanner.plan import Plan, simulate
from pyplanner.planner import solve
from pyplanner.portfolio import run_portfolio
from pyplanner.schema import validate_document
from pyplanner.search import PRESETS, SearchStatus, search
from pyplanner.search.anytime import anytime_schedule, search_anytime
from tests.tasks import build
from tests.tiers import VALIDATE, tier_dir

STORAGE = tier_dir("tier5/preferences/storage")

# Lámparas que se encienden y se apagan.
SWITCHES = """
(define (domain switches)
  (:requirements :typing :negative-preconditions :disjunctive-preconditions
                 :universal-preconditions :constraints :preferences)
  (:types lamp)
  (:predicates (on ?l - lamp))
  (:action turn-on :parameters (?l - lamp) :precondition (not (on ?l)) :effect (on ?l))
  (:action turn-off :parameters (?l - lamp) :precondition (on ?l) :effect (not (on ?l))))
"""


def switches(
    goal: str, metric: str, constraints: str = "", init: str = "", objects: str = "a b c"
) -> str:
    # Sin restricciones no hay sección: con (:constraints (and)), VAL da por incumplidas las
    # preferencias del objetivo.
    section = f"(:constraints {constraints})" if constraints else ""
    return f"""
(define (problem lamps) (:domain switches)
  (:objects {objects} - lamp)
  (:init {init})
  (:goal {goal})
  {section}
  (:metric minimize {metric}))
"""


def val_value(domain: str, problem: str, plan: Plan, tmp_path: Path) -> float | None:
    """La métrica del plan según VAL (``None`` si VAL no está o no lo da por válido)."""
    if not VALIDATE.exists():
        return None
    files = {"domain.pddl": domain, "problem.pddl": problem, "plan": plan.to_ipc()}
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    output = subprocess.run(
        [str(VALIDATE), *(str(tmp_path / f) for f in files)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    assert "Plan valid" in output, output
    match = re.search(r"Final value: (\S+)", output)
    assert match is not None, output
    return float(match.group(1))


def best_plan(task: GroundTask, preset: str = "lazy-ff-pref", limit: float = 30.0) -> Plan:
    result = search_anytime(task, PRESETS[preset], time_limit=limit)
    assert result.solved and result.plan is not None, result.status
    assert simulate(task, result.plan) is None
    assert result.optimal, result.message
    return Plan.from_indices(task, result.plan)


def check_metric(domain: str, problem: str, plan: Plan, tmp_path: Path) -> None:
    value = val_value(domain, problem, plan, tmp_path)
    if value is not None:
        assert plan.metric == pytest.approx(value), plan.to_ipc()


# --- Separar las preferencias ----------------------------------------------------------


def test_split_preferences() -> None:
    lamp = Parameter("?l", ("lamp",))
    on = Atom("on", (Variable("?l"),))
    lit = Preference("lit", on)
    hard = Atom("ready")
    formula = And((hard, Forall((lamp,), And((on, lit)))))
    rest, sites = split_preferences(formula)
    assert rest == And((hard, Forall((lamp,), And((on,)))))
    assert [(s.parameters, s.preference) for s in sites] == [((lamp,), lit)]
    assert split_preferences(hard) == (hard, [])


# --- Compilación ---------------------------------------------------------------------


def test_goal_preferences_compile_to_a_final_phase() -> None:
    problem = switches(
        "(and (on a) (preference pb (on b)) (preference pc (not (on c))))",
        "(+ (* 2 (is-violated pb)) (is-violated pc))",
    )
    task = build(SWITCHES, problem)
    preferences = task.preferences
    assert preferences is not None
    assert dict(preferences.weights) == {"pb": 2.0, "pc": 1.0}
    assert preferences.step_cost == pytest.approx(1e-3)
    auxiliary = task.actions[preferences.first_auxiliary :]
    assert [(a.name, a.args, a.cost) for a in auxiliary] == [
        (END, (), 0.0),
        (COLLECT, ("pb",), 0.0),
        (FORGO, ("pb",), 2.0),
        (COLLECT, ("pc",), 0.0),
        (FORGO, ("pc",), 1.0),
    ]
    # Las acciones del dominio cuestan su paso y no se pueden aplicar tras _end.
    ended = task.fact_index[Atom("_ended")]
    for action in task.actions[: preferences.first_auxiliary]:
        assert action.cost == pytest.approx(1e-3)
        assert ended in action.neg_facts
    assert len(task.goal) == 1 and len(task.goal[0].pos_facts) == 1


def test_plans_leave_out_the_final_phase(tmp_path: Path) -> None:
    problem = switches(
        "(and (on a) (preference pb (on b)) (preference pc (not (on c))))",
        "(+ (* 2 (is-violated pb)) (is-violated pc))",
        init="(on c)",
    )
    task = build(SWITCHES, problem)
    plan = best_plan(task, "astar-blind")
    assert sorted(map(str, plan.steps)) == ["(turn-off c)", "(turn-on a)", "(turn-on b)"]
    assert plan.metric == 0 and plan.cost == 0 and plan.violated == ()
    check_metric(SWITCHES, problem, plan, tmp_path)
    # El plan de solo encender a renuncia a las dos preferencias: 2 + 1.
    turn_on_a = next(a.index for a in task.actions if str(a) == "(turn-on a)")
    tail = [a.index for a in task.actions if a.name in (END, FORGO)]
    plan = Plan.from_indices(task, (turn_on_a, *tail))
    assert simulate(task, (turn_on_a, *tail)) is None
    assert plan.metric == 3 and plan.violated == ("pb", "pc")
    assert "; violated preferences: pb, pc" in plan.to_ipc()
    document = plan.to_dict()
    assert document["violated"] == ["pb", "pc"]
    validate_document(document)
    check_metric(SWITCHES, problem, plan, tmp_path)


@pytest.mark.parametrize(
    ("weights", "violated"),
    [((1, 3), ["x"]), ((3, 1), ["y"])],
)
def test_conflicting_preferences(
    weights: tuple[int, int], violated: list[str], tmp_path: Path
) -> None:
    """Encender c cumple x e incumple y: se renuncia a la que menos pesa."""
    problem = switches(
        "(and (on a) (preference x (on c)))",
        f"(+ (* {weights[0]} (is-violated x)) (* {weights[1]} (is-violated y)))",
        "(preference y (always (not (on c))))",
    )
    task = build(SWITCHES, problem)
    # always se incumple a mitad del plan: su peso se cobra en ese paso, sin fase final.
    assert task.preferences is not None
    assert [label for _, _, label in task.preferences.violations] == ["y"]
    assert {a.args for a in task.actions if a.name == FORGO} == {("x",)}
    plan = best_plan(task)
    assert list(plan.violated) == violated and plan.metric == 1
    check_metric(SWITCHES, problem, plan, tmp_path)


def test_penalties_are_charged_when_the_violation_happens() -> None:
    problem = switches("(on a)", "(* 3 (is-violated y))", "(preference y (always (not (on c))))")
    task = build(SWITCHES, problem)
    preferences = task.preferences
    assert preferences is not None
    turn_on = {str(a): a for a in task.actions}
    lit = task.apply(task.init, turn_on["(turn-on c)"])
    assert lit is not None
    assert preferences.penalty(task.init, lit) == 3 and preferences.committed(lit) == 3
    again = task.apply(lit, turn_on["(turn-on a)"])
    assert again is not None and preferences.penalty(lit, again) == 0  # una sola vez


# Lámparas con un predicado estático: (near ?a ?b).
NEAR = SWITCHES.replace("(on ?l - lamp))", "(on ?l - lamp) (near ?a ?b - lamp))")


@pytest.mark.parametrize(
    "constraints",
    [
        # Cierta en cualquier estado (near a c es falso), imposible de disparar, o con
        # todas las asignaciones de su forall descartadas por el join (nunca near ?l ?l).
        "(always (imply (near a c) (on c)))",
        "(preference g (sometime-before (near a c) (on b)))",
        "(forall (?l - lamp) (always (imply (near ?l ?l) (on ?l))))",
    ],
)
def test_trivial_constraints_are_dropped(constraints: str) -> None:
    goal = "(and (on a) (preference g (on b)))"
    problem = switches(goal, "(is-violated g)", constraints, init="(near a b)")
    assert build(NEAR, problem).monitors == ()


def test_constraint_foralls_are_grounded_with_a_join() -> None:
    """(always (not (and (on ?x) (on ?y) (near ?x ?y)))): solo las parejas cercanas."""
    apart = "(always (not (and (on ?x) (on ?y) (near ?x ?y))))"
    problem = switches(
        "(on a)",
        "(is-violated apart)",
        f"(forall (?x ?y - lamp) (preference apart {apart}))",
        init="(near a b) (near b c)",
    )
    task = build(NEAR, problem)
    assert sorted(m.label.split(" con ")[1] for m in task.monitors) == [
        "{'?x': 'a', '?y': 'b'}",
        "{'?x': 'b', '?y': 'c'}",
    ]


def test_quantified_preferences_count_each_instance(tmp_path: Path) -> None:
    """is-violated cuenta las instancias incumplidas: a y b no pueden estar encendidas a la
    vez, así que una de las dos (y c, si la restricción la impide) se queda sin luz."""
    problem = switches(
        "(forall (?l - lamp) (preference lit (on ?l)))",
        "(* 4 (is-violated lit))",
        "(and (always (not (and (on a) (on b)))) (always (not (on c))))",
    )
    task = build(SWITCHES, problem)
    plan = best_plan(task)
    assert plan.metric == 8 and len(plan) == 1
    assert sorted(plan.violated) in (["lit b", "lit c"], ["lit a", "lit c"])
    check_metric(SWITCHES, problem, plan, tmp_path)


@pytest.mark.parametrize(
    ("constraint", "goal", "metric", "length"),
    [
        # Encender c alguna vez cuesta un paso y ahorra 1.
        ("(preference p (sometime (on c)))", "(on a)", 0, 2),
        # b tiene que encenderse: always se incumple.
        ("(preference p (always (not (on b))))", "(on b)", 1, 1),
        # c antes que b.
        ("(preference p (sometime-before (on b) (on c)))", "(on b)", 0, 2),
        # Si se enciende c, después tiene que apagarse; con sometime (duro) se enciende.
        (
            "(and (sometime (on c)) (preference p (sometime-after (on c) (not (on c)))))",
            "(on a)",
            0,
            3,
        ),
        # c encendida en el estado 1 (o antes): c primero.
        ("(preference p (within 1 (on c)))", "(and (on a) (on b))", 0, 3),
        # a apagada en los estados 0 y 1: a en el segundo paso.
        ("(preference p (hold-during 0 2 (not (on a))))", "(and (on a) (on b))", 0, 2),
        # Tras encender c, a como mucho un paso después.
        (
            "(and (sometime (on c)) (preference p (always-within 1 (on c) (on a))))",
            "(on b)",
            0,
            3,
        ),
        # a empieza encendida, tiene que apagarse antes de encender b y acabar encendida:
        # at-most-once se incumple.
        (
            "(and (sometime-before (on b) (not (on a))) (preference p (at-most-once (on a))))",
            "(and (on a) (on b))",
            1,
            3,
        ),
        ("(preference p (at end (not (on b))))", "(on a)", 0, 1),
    ],
)
def test_constraint_preferences(
    constraint: str, goal: str, metric: int, length: int, tmp_path: Path
) -> None:
    init = "(on a)" if "at-most-once" in constraint else ""
    problem = switches(goal, "(is-violated p)", constraint, init)
    task = build(SWITCHES, problem)
    plan = best_plan(task)
    assert plan.metric == metric and len(plan) == length
    check_metric(SWITCHES, problem, plan, tmp_path)
    # El primer plan (de una búsqueda voraz) también tiene la métrica de VAL.
    result = search(task, PRESETS["gbfs-goalcount-rnd"])
    assert result.plan is not None
    check_metric(SWITCHES, problem, Plan.from_indices(task, result.plan), tmp_path)


# Encender una lámpara con c encendida molesta: cada vez cuenta.
QUIET = SWITCHES.replace(
    ":precondition (not (on ?l)) :effect (on ?l))",
    ":precondition (and (not (on ?l)) (preference quiet (not (on c)))) :effect (on ?l))",
).replace("(:types lamp)", "(:types lamp) (:constants c - lamp)")


def test_precondition_preferences_count_every_application(tmp_path: Path) -> None:
    problem = switches(
        "(and (on a) (on b) (on c))", "(* 5 (is-violated quiet))", init="(on c)", objects="a b"
    )
    task = build(QUIET, problem)
    preferences = task.preferences
    assert preferences is not None and preferences.first_auxiliary == len(task.actions)
    costs = sorted(a.cost for a in task.actions if str(a) == "(turn-on a)")
    assert costs == pytest.approx([0.005, 5.005])
    # Lo mejor: apagar c, encender a y b, y volver a encender c.
    plan = best_plan(task)
    assert plan.metric == 0 and len(plan) == 4
    check_metric(QUIET, problem, plan, tmp_path)
    # Con c encendida, encender a y b cuesta 2 veces 5.
    expensive = tuple(
        next(a.index for a in task.actions if str(a) == f"(turn-on {lamp})" and a.cost > 5)
        for lamp in ("a", "b")
    )
    assert simulate(task, expensive) is None
    plan = Plan.from_indices(task, expensive)
    assert plan.metric == 10 and plan.cost == 10
    check_metric(QUIET, problem, plan, tmp_path)


def test_preferences_outside_the_metric_are_ignored(tmp_path: Path) -> None:
    problem = switches(
        "(and (on a) (preference counted (on b)))",
        "(* 2 (is-violated counted))",
        "(preference ignored (always (not (on a))))",
    )
    task = build(SWITCHES, problem)
    assert task.monitors == ()
    plan = best_plan(task)
    assert plan.metric == 0 and len(plan) == 2
    check_metric(SWITCHES, problem, plan, tmp_path)


def test_without_metric_preferences_do_not_count() -> None:
    problem = switches("(and (on a) (preference p (on b)))", "(total-time)").replace(
        "(:metric minimize (total-time))", ""
    )
    task = build(SWITCHES, problem)
    assert task.preferences is None
    assert "no tiene métrica" in " ".join(task.notes)
    result = search(task, PRESETS["astar-blind"])
    assert result.plan is not None and len(result.plan) == 1


@pytest.mark.parametrize(
    ("metric", "message"),
    [
        ("(* (is-violated p) (is-violated p))", "expresión lineal"),
        ("(- (is-violated p))", "peso negativo"),
    ],
)
def test_metrics_that_cannot_be_costs(metric: str, message: str) -> None:
    with pytest.raises(UnsupportedFeatureError, match=message):
        build(SWITCHES, switches("(and (on a) (preference p (on b)))", metric))


def test_maximized_metrics_are_minimized_negated(tmp_path: Path) -> None:
    """Maximizar es minimizar la negación: (- 10 (* 2 (is-violated p))) es un peso 2."""
    problem = switches(
        "(and (on a) (preference p (on b)))", "(- 10 (* 2 (is-violated p)))"
    ).replace("minimize", "maximize")
    task = build(SWITCHES, problem)
    assert task.metric_sign == -1 and task.preferences is not None
    assert dict(task.preferences.weights) == {"p": 2.0}
    plan = best_plan(task)
    assert plan.metric == 10 and not plan.violated
    check_metric(SWITCHES, problem, plan, tmp_path)


def test_maximizing_violations_gives_negative_weights() -> None:
    problem = switches("(and (on a) (preference p (on b)))", "(is-violated p)").replace(
        "minimize", "maximize"
    )
    with pytest.raises(UnsupportedFeatureError, match="peso negativo"):
        build(SWITCHES, problem)


# --- Preferencias en tareas temporales -------------------------------------------------

# Calentar el horno (10) y hornear (5); limpiar (3) es aparte. Calentarlo con el horno ya
# limpio incumple la preferencia de su condición 'at start'.
OVEN = """
(define (domain oven)
  (:requirements :durative-actions :negative-preconditions :numeric-fluents :preferences
                 :constraints)
  (:predicates (hot) (baked) (cleaned))
  (:durative-action heat :parameters () :duration (= ?duration 10)
    :condition (preference early (at start (not (cleaned))))
    :effect (at end (hot)))
  (:durative-action bake :parameters () :duration (= ?duration 5)
    :condition (at start (hot)) :effect (at end (baked)))
  (:durative-action clean :parameters () :duration (= ?duration 3)
    :condition (and) :effect (at end (cleaned))))
"""


def oven(goal: str, metric: str, constraints: str = "", init: str = "") -> str:
    section = f"(:constraints {constraints})" if constraints else ""
    return f"""
(define (problem oven-1) (:domain oven) (:init {init}) (:goal {goal}) {section}
  (:metric {metric}))
"""


def temporal_plan(problem: str, preset: str = "gbfs-ff-pref") -> tuple[GroundTask, Plan]:
    domain = load_domain(parse_domain(OVEN))
    result = solve(domain, load_problem(parse_problem(problem), domain), [PRESETS[preset]], 10)
    assert result.plan is not None, result.search.status
    return result.task, result.plan


def test_temporal_goal_preferences(tmp_path: Path) -> None:
    problem = oven("(and (baked) (preference tidy (cleaned)))", "minimize (* 3 (is-violated tidy))")
    task, plan = temporal_plan(problem)
    assert task.preferences is not None and plan.is_temporal
    assert plan.metric == 0 and not plan.violated
    assert {s.name for s in plan.steps} == {"heat", "bake", "clean"}
    check_metric(OVEN, problem, plan, tmp_path)


def test_preferences_of_durative_conditions_count_at_their_snap(tmp_path: Path) -> None:
    """Con el horno limpio desde el principio, calentarlo incumple 'early' (peso 4)."""
    problem = oven("(baked)", "minimize (* 4 (is-violated early))", init="(cleaned)")
    task, plan = temporal_plan(problem)
    starts = [a for a in task.actions if a.name == "heat@start"]
    assert sorted(a.cost for a in starts) != [starts[0].cost] * len(starts)  # dos variantes
    assert plan.metric == 4
    check_metric(OVEN, problem, plan, tmp_path)


def test_soft_constraints_with_deadlines(tmp_path: Path) -> None:
    """Calentar y hornear lleva 15: 'within 12' no se puede cumplir."""
    problem = oven(
        "(baked)",
        "minimize (+ (* 5 (is-violated quick)) (* 2 (is-violated slow)))",
        "(and (preference quick (within 12 (baked))) (preference slow (within 20 (baked))))",
    )
    _, plan = temporal_plan(problem)
    assert plan.metric == 5 and plan.violated == ("quick",)
    check_metric(OVEN, problem, plan, tmp_path)


def test_maximized_temporal_metrics(tmp_path: Path) -> None:
    problem = oven(
        "(and (baked) (preference tidy (cleaned)))", "maximize (- 6 (* 6 (is-violated tidy)))"
    )
    _, plan = temporal_plan(problem)
    assert plan.metric == 6
    check_metric(OVEN, problem, plan, tmp_path)


def test_total_time_is_reported_but_not_optimized(tmp_path: Path) -> None:
    problem = oven(
        "(and (baked) (preference tidy (cleaned)))",
        "minimize (+ (* 3 (is-violated tidy)) (total-time))",
    )
    task, plan = temporal_plan(problem)
    assert any("total-time" in note for note in task.notes)
    assert plan.makespan is not None and plan.metric == pytest.approx(plan.makespan)
    check_metric(OVEN, problem, plan, tmp_path)
    domain = load_domain(parse_domain(OVEN))
    result = solve(
        domain, load_problem(parse_problem(problem), domain), [PRESETS["astar-blind"]], 10
    )
    assert result.search.lower_bound is None  # sin demostrar que sea óptimo


def test_the_final_phase_waits_for_running_actions() -> None:
    """'_end' exige además que no quede ninguna acción durativa en curso."""
    problem = oven("(hot)", "minimize (* 3 (is-violated tidy))", init="")
    problem = problem.replace("(:goal (hot))", "(:goal (and (hot) (preference tidy (cleaned))))")
    task, plan = temporal_plan(problem)
    preferences = task.preferences
    assert preferences is not None and isinstance(task, TemporalTask)
    for index in preferences.end_actions:
        assert set(task.actions[index].neg_facts) >= {d.running for d in task.durative}
    assert all(s.duration is not None for s in plan.steps)  # todas terminan


def test_over_all_preferences_are_not_supported() -> None:
    domain = OVEN.replace(
        "(preference early (at start (not (cleaned))))",
        "(preference early (over all (not (cleaned))))",
    )
    with pytest.raises(UnsupportedFeatureError, match="over all"):
        build(domain, oven("(baked)", "minimize (is-violated early)"))


# --- Monitores blandos ---------------------------------------------------------------


def test_soft_monitors_mark_their_violation() -> None:
    """Un at-most-once blando: al incumplirse borra sus hechos, marca violated y deja de
    cambiar."""
    on = Condition((0,))
    monitor = Monitor("at-most-once", (on,), facts=(1, 2), violated=3)
    evaluator = MonitorEvaluator([monitor])
    state = evaluator.initial(State(0b0001))
    assert state == State(0b0011)  # a encendida: seen
    off = evaluator.update(state, State(0b0010))
    assert off == State(0b0110)  # apagada: closed
    again = evaluator.update(off, State(0b0111))
    assert again == State(0b1001)  # otra vez encendida: violated (y nada más)
    assert evaluator.update(again, State(0b1000)) == State(0b1000)


# --- Cota y búsqueda anytime -----------------------------------------------------------


def test_bounded_search_only_finds_better_plans() -> None:
    problem = switches(
        "(and (on a) (preference pb (on b)) (preference pc (not (on c))))",
        "(+ (* 2 (is-violated pb)) (is-violated pc))",
        init="(on c)",
    )
    task = build(SWITCHES, problem)
    optimal = search(task, PRESETS["astar-blind"])
    assert optimal.cost == pytest.approx(0.003)
    for preset in ("astar-hmax", "wastar-add-3", "lazy-ff-pref"):
        bounded = search(task, PRESETS[preset], bound=optimal.cost)
        assert bounded.status is SearchStatus.UNSOLVABLE, preset
    looser = search(task, PRESETS["lazy-ff-pref"], bound=1.5)
    assert looser.cost is not None and looser.cost < 1.5


def test_the_engine_keeps_an_incumbent() -> None:
    """Sin objetivo duro, cada estado expandido es un plan: la búsqueda voraz va dejando
    planes cada vez mejores con on_plan, y devuelve el mejor."""
    domain_text = (STORAGE / "domain.pddl").read_text()
    task = build(domain_text, (STORAGE / "p02.pddl").read_text())
    found: list[float] = []
    result = search(task, PRESETS["lazy-ff-pref"], on_plan=lambda r: found.append(r.cost or 0.0))
    assert result.solved and result.cost is not None and found
    assert found == sorted(found, reverse=True) and result.cost <= found[-1]
    assert result.plan is not None and simulate(task, result.plan) is None
    # Sin on_plan no hay incumbente.
    assert search(task, PRESETS["lazy-ff-pref"]).cost != found[0] or len(found) == 1


def test_exhaustive_searches_prove_the_bound() -> None:
    problem = switches(
        "(and (on a) (preference pb (on b)) (preference pc (not (on c))))",
        "(+ (* 2 (is-violated pb)) (is-violated pc))",
        init="(on c)",
    )
    task = build(SWITCHES, problem)
    exact = search(task, PRESETS["astar-hmax"], bound=0.003)
    assert exact.status is SearchStatus.UNSOLVABLE and exact.lower_bound == pytest.approx(0.003)
    lazy = search(task, PRESETS["lazy-ff-pref"], bound=0.003)
    assert lazy.status is SearchStatus.UNSOLVABLE and lazy.lower_bound is None


def test_anytime_schedule() -> None:
    names = [c.name for c in anytime_schedule(PRESETS["lazy-ff-pref-tb"])]
    assert names == [
        "lazy-ff-pref-tb",
        "lazy-ff-pref-tb+coste",
        "lazy-ff-pref-tb+wastar5",
        "lazy-ff-pref-tb+wastar3",
        "lazy-ff-pref-tb+wastar2",
        "lazy-ff-pref-tb+wastar1",
        "lazy-ff-pref-tb+astar-hmax",
    ]
    schedule = anytime_schedule(PRESETS["lazy-ff-pref-tb"])
    assert all(c.lazy and c.exploration == "type-based" for c in schedule[1:-1])


@pytest.mark.parametrize("preset", ["lazy-ff-pref", "bfws-ff-pref"])
def test_anytime_improves_storage_p02_to_the_optimum(preset: str, tmp_path: Path) -> None:
    domain_text = (STORAGE / "domain.pddl").read_text()
    problem_text = (STORAGE / "p02.pddl").read_text()
    task = build(domain_text, problem_text)
    found: list[float] = []
    result = search_anytime(
        task,
        PRESETS[preset].with_changes(max_memory=2000),
        time_limit=60,
        on_plan=lambda r: found.append(r.cost or 0.0),
    )
    assert result.optimal and "(óptimo)" in result.message
    assert len(found) >= 2 and found == sorted(found, reverse=True)
    assert result.plan is not None
    plan = Plan.from_indices(task, result.plan)
    assert plan.metric == 1
    check_metric(domain_text, problem_text, plan, tmp_path)


def test_anytime_respects_the_time_limit() -> None:
    domain_text = (STORAGE / "domain.pddl").read_text()
    task = build(domain_text, (STORAGE / "p02.pddl").read_text())
    result = search_anytime(task, PRESETS["lazy-ff-pref"], time_limit=1e-4)
    assert result.statistics.time < 10.0
    assert not result.optimal


def test_anytime_portfolio_shares_the_bound(tmp_path: Path) -> None:
    domain_text = (STORAGE / "domain.pddl").read_text()
    problem_text = (STORAGE / "p02.pddl").read_text()
    task = build(domain_text, problem_text)
    configs = [PRESETS[n].with_changes(max_memory=1500) for n in ("bfws-ff-pref", "lazy-ff-pref")]
    portfolio = run_portfolio(task, configs, time_limit=60, anytime=True)
    best = portfolio.best
    assert best.solved and best.optimal and best.plan is not None
    plan = Plan.from_indices(task, best.plan)
    assert plan.metric == 1
    assert set(portfolio.results) == {"bfws-ff-pref", "lazy-ff-pref"}
    check_metric(domain_text, problem_text, plan, tmp_path)


@pytest.mark.parametrize(("problem", "metric"), [("p01.pddl", 0), ("p02.pddl", 1)])
def test_solve_storage(problem: str, metric: int, tmp_path: Path) -> None:
    domain = read_domain(STORAGE / "domain.pddl")
    result = solve(domain, read_problem(STORAGE / problem, domain), [PRESETS["lazy-ff-pref"]])
    assert result.plan is not None and result.search.optimal
    assert result.plan.metric == metric
    check_metric(
        (STORAGE / "domain.pddl").read_text(),
        (STORAGE / problem).read_text(),
        result.plan,
        tmp_path,
    )


def test_cli_reports_violated_preferences(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "solve",
            str(STORAGE / "domain.pddl"),
            str(STORAGE / "p02.pddl"),
            "-c",
            "lazy-ff-pref",
            "-t",
            "60",
            "-m",
            "2000",
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "; metric = 1" in captured.out
    assert "métrica 1 (óptimo demostrado)" in captured.err
    assert "Preferencias incumplidas: p1a" in captured.err
