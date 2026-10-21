"""Motor discretizado de PDDL+ (iteración 6b): integración de los procesos, eventos en su
instante exacto, acciones durativas compiladas, validación con un paso más fino y planes
del tier 6 validados con VAL."""

import math
import re
import subprocess
from pathlib import Path

import pytest

from pyplanner.cli import main
from pyplanner.errors import UnsupportedFeatureError
from pyplanner.frontend import read_domain, read_problem
from pyplanner.grounding.discrete import WAIT, DiscreteTask
from pyplanner.grounding.flows import Flow
from pyplanner.grounding.numeric import Const, Op, Var
from pyplanner.grounding.task import State
from pyplanner.loader import load_domain, load_problem
from pyplanner.loader.ast import Atom
from pyplanner.pddl import parse_domain, parse_problem
from pyplanner.plan import Plan, simulate
from pyplanner.planner import solve
from pyplanner.search import PRESETS, SearchStatus, search
from pyplanner.search.novelty import NoveltyTable
from tests.tasks import build
from tests.tiers import VALIDATE, tier_dir

# --- Integración ---------------------------------------------------------------------


def test_polynomial_dynamics_are_exact() -> None:
    """car: v' = a, d' = v. La serie de Taylor termina y la distancia es exacta."""
    d, v, a = 0, 1, 2
    flow = Flow({v: Var(a), d: Var(v)})
    assert flow.exact
    assert flow.advance((0.0, 6.0, -1.0), 6.0) == (18.0, 0.0, -1.0)


def test_cubic_dynamics_are_exact() -> None:
    """generator: el caudal crece con el cuadrado del tiempo de repostaje."""
    fuel, tank, time = 0, 1, 2
    rate = Op("*", (Const(0.001), Op("*", (Var(time), Var(time)))))
    flow = Flow({fuel: rate, tank: Op("-", (rate,)), time: Const(1.0)})
    assert flow.exact
    values = flow.advance((0.0, 40.0, 0.0), 30.0)
    assert values is not None
    assert values[fuel] == pytest.approx(0.001 * 30**3 / 3, abs=1e-9)
    assert values[tank] == pytest.approx(40 - 9, abs=1e-9)


def test_other_dynamics_use_runge_kutta() -> None:
    growth = Flow({0: Var(0)})  # x' = x: exponencial, la serie no termina
    assert not growth.exact
    values = growth.advance((1.0,), 1.0)
    assert values is not None and values[0] == pytest.approx(math.e, abs=1e-8)
    ratio = Flow({0: Op("/", (Const(1.0), Var(1))), 1: Const(1.0)})  # divide por una variable
    assert not ratio.exact


def test_undefined_rates_make_the_step_invalid() -> None:
    assert Flow({0: Var(1)}).advance((0.0, None), 1.0) is None


# --- Semántica de los pasos ------------------------------------------------------------

# Un depósito que se llena mientras el grifo está abierto. Al llegar a 10 se desborda: un
# evento lo marca y deja el nivel en 10.
TANK = """
(define (domain tank)
  (:requirements :typing :numeric-fluents :negative-preconditions :time)
  (:types tap)
  (:predicates (open ?t - tap) (overflowed) (done))
  (:functions (level) (flow ?t - tap))
  (:action turn-on :parameters (?t - tap) :precondition (not (open ?t)) :effect (open ?t))
  (:action turn-off :parameters (?t - tap) :precondition (open ?t) :effect (not (open ?t)))
  (:action finish :parameters () :precondition (>= (level) 5) :effect (done))
  (:process filling :parameters (?t - tap) :precondition (open ?t)
    :effect (increase (level) (* #t (flow ?t))))
  (:event overflow :parameters () :precondition (and (>= (level) 10) (not (overflowed)))
    :effect (overflowed)))
"""


def tank(goal: str = "(and (done) (not (overflowed)))", flow_a: float = 1.0) -> DiscreteTask:
    problem = f"""
(define (problem tank-1) (:domain tank)
  (:objects a b - tap)
  (:init (= (level) 0) (= (flow a) {flow_a}) (= (flow b) 2))
  (:goal {goal})
  (:metric minimize (total-time)))
"""
    task = build(TANK, problem)
    assert isinstance(task, DiscreteTask)
    return task


def action(task: DiscreteTask, name: str) -> int:
    return next(a.index for a in task.actions if str(a) == name)


def test_waits_integrate_the_active_processes() -> None:
    task = tank()
    assert [str(a) for a in task.actions][-1] == f"({WAIT})"
    level = task.variable_index[next(v for v in task.variables if str(v) == "(level)")]
    state = task.apply(task.init, task.actions[action(task, "(turn-on a)")])
    assert state is not None
    assert state.values[level] == pytest.approx(0.01)  # el ε tras la acción
    state = task.apply(state, task.wait)
    assert state is not None and state.values[level] == pytest.approx(1.01)


def test_events_fire_at_their_exact_instant() -> None:
    """Con flujo 3, el depósito llega a 10 a mitad de una espera: el evento se dispara
    allí, y la espera sigue hasta completar δ."""
    task = tank(flow_a=3.0)
    level = task.variable_index[next(v for v in task.variables if str(v) == "(level)")]
    state = task.apply(task.init, task.actions[action(task, "(turn-on a)")])
    for _ in range(3):
        assert state is not None
        state = task.apply(state, task.wait)
    assert state is not None and Atom("overflowed") not in task.atoms(state)  # 9,03
    state = task.apply(state, task.wait)
    assert state is not None and Atom("overflowed") in task.atoms(state)
    assert state.values[level] == pytest.approx(12.03)  # el evento no cambia el nivel
    # El evento ocurre en t = 10/3 (en mitad de la cuarta espera): VAL lo pondría ahí.
    opened = task.apply(task.init, task.actions[action(task, "(turn-on a)")])
    assert opened is not None
    middle = task.elapse(opened, 10 / 3 - 0.01 - 1e-6)
    assert middle is not None and Atom("overflowed") not in task.atoms(middle)


def test_plans_with_processes(tmp_path: Path) -> None:
    task = tank()
    result = search(task, PRESETS["astar-blind"])
    assert result.status is SearchStatus.SOLVED and result.plan is not None
    assert simulate(task, result.plan) is None
    plan = Plan.from_indices(task, result.plan)
    # Lo más rápido es abrir los dos grifos (primero b, de flujo 2: tras los dos ε hay 0,05) y
    # esperar hasta que el nivel llega a 5 (_wait-until): 4,95 / 3 más.
    assert sorted(str(step) for step in plan.steps) == ["(finish)", "(turn-on a)", "(turn-on b)"]
    assert plan.makespan == pytest.approx(0.02 + 4.95 / 3, abs=1e-6)
    assert task.check(result.plan, 0.1) is None


def test_exhausting_the_search_is_not_a_proof() -> None:
    """Sin el grifo b y con el depósito a medias no hay plan con δ = 1; la búsqueda no
    puede decir que no lo haya con otro paso."""
    text = TANK.replace("(:process filling :parameters (?t - tap) :precondition (open ?t)",
                        "(:process filling :parameters (?t - tap) "
                        ":precondition (and (open ?t) (< (level) 3))")  # fmt: skip
    problem = """
(define (problem tank-1) (:domain tank) (:objects a b - tap)
  (:init (= (level) 0) (= (flow a) 1) (= (flow b) 2)) (:goal (done)))
"""
    task = build(text, problem)
    result = search(task, PRESETS["gbfs-ff"])
    assert result.status is SearchStatus.FAILED
    assert "δ = 1" in result.message


# Una pelota que se lanza al empezar a mirarla: x' = v, v' = -1. Con v = 2,5, sube hasta
# 3,125 en t = 2,5, justo a mitad de un paso de δ = 1. La invariante exige x ≤ 3,1.
BALL = """
(define (domain ball)
  (:requirements :numeric-fluents :durative-actions :time)
  (:predicates (flying) (seen))
  (:functions (x) (v))
  (:durative-action watch :parameters () :duration (= ?duration 3)
    :condition (over all (<= (x) 3.1))
    :effect (and (at start (flying)) (at start (assign (v) 2.5)) (at end (seen))))
  (:process gravity :parameters () :precondition (flying)
    :effect (and (increase (x) (* #t (v))) (decrease (v) (* #t 1)))))
"""
BALL_PROBLEM = """
(define (problem ball-1) (:domain ball) (:init (= (x) 0) (= (v) 0)) (:goal (seen)))
"""


def test_durative_actions_end_at_their_duration() -> None:
    task = build(BALL.replace("3.1", "4"), BALL_PROBLEM)
    assert isinstance(task, DiscreteTask)
    result = search(task, PRESETS["gbfs-ff"])
    assert result.plan is not None
    [step] = Plan.from_indices(task, result.plan).steps
    assert step.name == "watch" and step.duration == pytest.approx(3.0)


def test_invariants_are_checked_with_a_finer_step() -> None:
    """Con δ = 1, la pelota pasa de 3,005 (t = 2,01) a 3 (t = 3) y la invariante se cumple en
    cada paso. Con un paso diez veces más fino se ve que llega a 3,125: el plan no vale, y
    con δ = 0,5 ya no hay plan."""
    task = build(BALL, BALL_PROBLEM)
    assert isinstance(task, DiscreteTask)
    result = search(task, PRESETS["gbfs-ff"])
    assert result.plan is not None
    assert task.check(result.plan, 0.1) is not None
    domain = load_domain(parse_domain(BALL))
    planning = solve(domain, load_problem(parse_problem(BALL_PROBLEM), domain),
                     [PRESETS["gbfs-ff"]])  # fmt: skip
    assert planning.plan is None
    assert any("se busca con δ = 0.5" in note for note in planning.task.notes)


# Un calefactor: la potencia crece con el tiempo y la temperatura, con la potencia
# (t²/2). Es PDDL 2.1 (sin procesos), pero la programación lineal de la 3b no lo admite.
HEATER = """
(define (domain heater)
  (:requirements :numeric-fluents :durative-actions :continuous-effects)
  (:predicates (warm))
  (:functions (temp) (power))
  (:durative-action heat :parameters () :duration (= ?duration 4)
    :condition (and)
    :effect (and (increase (power) (* #t 1)) (increase (temp) (* #t (power)))
                 (at end (warm)))))
"""
HEATER_PROBLEM = """
(define (problem heater-1) (:domain heater)
  (:init (= (temp) 0) (= (power) 0)) (:goal (and (warm) (>= (temp) 8))))
"""


def test_nonlinear_temporal_tasks_use_the_discrete_engine(tmp_path: Path) -> None:
    task = build(HEATER, HEATER_PROBLEM)
    assert isinstance(task, DiscreteTask)
    assert any("se usa el motor con el tiempo discretizado" in note for note in task.notes)
    result = search(task, PRESETS["gbfs-ff"])
    assert result.plan is not None
    plan = Plan.from_indices(task, result.plan)
    [step] = plan.steps
    assert step.name == "heat" and step.duration == pytest.approx(4.0)
    if VALIDATE.exists():
        (tmp_path / "domain.pddl").write_text(HEATER)
        (tmp_path / "problem.pddl").write_text(HEATER_PROBLEM)
        output = val(tmp_path / "domain.pddl", tmp_path / "problem.pddl", plan, tmp_path)
        assert "Plan valid" in output, output
    # Lo que el motor temporal sí admite sigue en él.
    linear = build(HEATER.replace("(* #t (power))", "(* #t 2)"), HEATER_PROBLEM)
    assert not isinstance(linear, DiscreteTask)


def test_events_that_never_stop_make_the_step_invalid() -> None:
    text = TANK.replace(
        "(:event overflow :parameters () :precondition (and (>= (level) 10) (not (overflowed)))\n"
        "    :effect (overflowed))",
        "(:event drip :parameters () :precondition (>= (level) 1)\n"
        "    :effect (increase (level) 1))",
    )
    task = build(text, "(define (problem p) (:domain tank) (:objects a - tap) "
                 "(:init (= (level) 0) (= (flow a) 1)) (:goal (done)))")  # fmt: skip
    assert isinstance(task, DiscreteTask)
    on = task.apply(task.init, task.actions[action(task, "(turn-on a)")])
    assert on is not None and task.apply(on, task.wait) is None


@pytest.mark.parametrize(
    ("problem", "message"),
    [("(:init) (:constraints (always-within 3 (open a) (done)))", "always-within")],
)
def test_unsupported_with_processes(problem: str, message: str) -> None:
    text = TANK.replace(":time)", ":time :durative-actions :timed-initial-literals :constraints)")
    with pytest.raises(UnsupportedFeatureError, match=message):
        build(text, f"(define (problem p) (:domain tank) (:objects a - tap) {problem} "
                    "(:goal (done)))")  # fmt: skip


def test_continuous_variables_do_not_count_for_novelty() -> None:
    task = tank()
    assert {str(task.variables[v]) for v in task.continuous_variables} == {"(level)"}
    table = NoveltyTable(1, len(task.facts), (), task.continuous_variables)
    state = task.apply(task.init, task.actions[action(task, "(turn-on b)")])
    assert state is not None
    table.novelty(state)
    waited = task.apply(state, task.wait)
    assert waited is not None and table.novelty(waited) == 2  # solo cambia (level)


# --- Literales iniciales temporizados y restricciones (trabajo opcional, punto 4) ------

# Una bomba que llena un depósito a ritmo 2, solo con luz de día (un TIL la trae a las 1 y
# otro se la lleva a las 6); al anochecer, un evento la para.
SPRINKLER = """
(define (domain sprinkler)
  (:requirements :time :negative-preconditions :timed-initial-literals :numeric-fluents
                 :constraints)
  (:predicates (pumping) (daylight))
  (:functions (water))
  (:action start-pump :parameters ()
    :precondition (and (daylight) (not (pumping))) :effect (pumping))
  (:action stop-pump :parameters () :precondition (pumping) :effect (not (pumping)))
  (:process pump :parameters () :precondition (pumping)
    :effect (increase (water) (* #t 2)))
  (:event dusk :parameters () :precondition (and (pumping) (not (daylight)))
    :effect (not (pumping))))
"""


def sprinkler(constraints: str = "") -> str:
    section = f"(:constraints {constraints})" if constraints else ""
    return f"""
(define (problem sprinkler-1) (:domain sprinkler)
  (:init (= (water) 0) (at 1 (daylight)) (at 6 (not (daylight))))
  (:goal (and (>= (water) 8) (not (pumping)))) {section})
"""


def solve_text(
    domain: str, problem: str, tmp_path: Path, time_limit: float = 20
) -> tuple[Plan | None, str]:
    """El plan del portfolio de 1 núcleo y lo que dice VAL de él (vacío sin VAL)."""
    loaded = load_domain(parse_domain(domain))
    result = solve(
        loaded, load_problem(parse_problem(problem), loaded), [PRESETS["bfws-ff-pref"]], time_limit
    )
    if result.plan is None or not VALIDATE.exists():
        return result.plan, ""
    files = {"domain.pddl": domain, "problem.pddl": problem}
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    return result.plan, val(
        tmp_path / "domain.pddl", tmp_path / "problem.pddl", result.plan, tmp_path
    )


def test_timed_literals_with_processes(tmp_path: Path) -> None:
    """Los TIL son eventos de un reloj global; una acción no coincide con un TIL (VAL no lo
    admite): va ε después."""
    plan, report = solve_text(SPRINKLER, sprinkler(), tmp_path)
    assert plan is not None
    assert [(str(s), s.time) for s in plan.steps] == [("(start-pump)", 1.01)]
    assert not report or "Plan valid" in report


def test_constraints_with_processes(tmp_path: Path) -> None:
    """Como VAL, las restricciones se comprueban en los sucesos. Con 'within 4', el agua
    tiene que llegar a 2 en un suceso antes de las 4: no basta con esperar (no hay ninguno
    hasta el anochecer), así que la bomba se para y se vuelve a poner en marcha; si solo se
    puede poner una vez, no hay plan. Con 'within 7', el anochecer (a las 6) sirve."""
    plan, report = solve_text(SPRINKLER, sprinkler("(within 4 (>= (water) 2))"), tmp_path)
    assert plan is not None and len(plan.steps) == 3
    assert not report or "Plan valid" in report
    once = "(at-most-once (pumping))"
    plan, _ = solve_text(SPRINKLER, sprinkler(f"(and (within 4 (>= (water) 2)) {once})"), tmp_path)
    assert plan is None
    plan, report = solve_text(
        SPRINKLER, sprinkler(f"(and (within 7 (>= (water) 2)) {once})"), tmp_path
    )
    assert plan is not None and [str(s) for s in plan.steps] == ["(start-pump)"]
    assert not report or "Plan valid" in report


def test_waits_until_the_next_change_update_the_monitors() -> None:
    """La espera hasta el siguiente cambio llega al TIL de las 1, que trae la luz: incumple
    'always (not (daylight))' tanto al buscar (``transition``) como al aplicarla."""
    task = build(SPRINKLER, sprinkler("(always (not (daylight)))"))
    assert isinstance(task, DiscreteTask)
    assert task.apply(task.init, task.jump) is None
    assert task.transition(task.init, task.jump)[0] is None
    assert task.transition(task.init, task.wait)[0] is None


@pytest.mark.parametrize(("early", "once", "violated"), [(1, 5, ["early"]), (5, 2, ["once"])])
def test_preferences_with_processes(
    early: int, once: int, violated: list[str], tmp_path: Path
) -> None:
    """'early' (el agua llega a 2 antes de las 3) solo se cumple parando la bomba y
    volviéndola a poner en marcha (las restricciones se comprueban en los sucesos), y eso
    incumple 'once': se renuncia a la más barata."""
    domain = SPRINKLER.replace(":constraints)", ":constraints :preferences)")
    problem = f"""
(define (problem sprinkler-1) (:domain sprinkler)
  (:init (= (water) 0) (at 1 (daylight)) (at 6 (not (daylight))))
  (:goal (and (>= (water) 8) (not (pumping))))
  (:constraints (and (preference early (within 3 (>= (water) 2)))
                     (preference once (at-most-once (pumping)))))
  (:metric minimize (+ (* {early} (is-violated early)) (* {once} (is-violated once)))))
"""
    task = build(domain, problem)
    assert isinstance(task, DiscreteTask) and task.preferences is not None
    plan, report = solve_text(domain, problem, tmp_path, time_limit=5)
    assert plan is not None
    assert plan.metric == pytest.approx(min(early, once))
    assert list(plan.violated) == violated
    assert not report or "Plan valid" in report


# --- Efectos condicionales en acciones durativas (trabajo opcional, punto 4) -----------

# Un horno que calienta a ritmo 1 mientras hornea, y a ritmo 3 si el turbo estaba puesto al
# empezar; hornea si la puerta sigue cerrada todo el tiempo. Un resorte abre la puerta al
# llegar a 2 de calor, salvo que esté cerrada con llave.
OVEN_PLUS = """
(define (domain oven-plus)
  (:requirements :time :durative-actions :conditional-effects :negative-preconditions
                 :numeric-fluents :continuous-effects)
  (:predicates (closed) (locked) (turbo) (baking) (baked))
  (:functions (heat))
  (:action close-door :parameters () :precondition (not (closed)) :effect (closed))
  (:action lock-door :parameters () :precondition (and (closed) (not (locked)))
    :effect (locked))
  (:action set-turbo :parameters () :precondition (not (turbo)) :effect (turbo))
  (:durative-action bake :parameters () :duration (= ?duration 4)
    :condition (at start (not (baking)))
    :effect (and (at start (baking)) (at end (not (baking)))
                 (increase (heat) (* #t 1))
                 (when (at start (turbo)) (increase (heat) (* #t 2)))
                 (when (over all (closed)) (at end (baked)))))
  (:event spring :parameters () :precondition (and (closed) (not (locked)) (>= (heat) 2))
    :effect (not (closed))))
"""


def oven_plus(goal: str, init: str = "") -> str:
    return f"""
(define (problem oven-plus-1) (:domain oven-plus) (:init (= (heat) 0) {init}) (:goal {goal}))
"""


def test_conditional_effects_with_over_all_conditions(tmp_path: Path) -> None:
    """Como en la 3b, un vigilante: se activa al empezar si la puerta está cerrada y se
    apaga en cuanto se abre (aquí, con el resorte, a mitad de una espera); entonces el
    horno no hornea. Con la puerta cerrada con llave antes de llegar a 2, sí."""
    task = build(OVEN_PLUS, oven_plus("(baked)"))
    assert isinstance(task, DiscreteTask)
    flag, end = Atom("cond@bake@0"), task.actions[action(task, "(bake@end)")]
    state: State | None = task.init
    for name in ("(close-door)", "(bake@start)"):
        assert state is not None
        state = task.apply(state, task.actions[action(task, name)])
    assert state is not None and flag in task.atoms(state)
    state = task.apply(state, task.wait)
    assert state is not None and flag in task.atoms(state)  # calor 1,01
    state = task.apply(state, task.wait)
    assert state is not None and {Atom("closed"), flag}.isdisjoint(task.atoms(state))
    while state is not None and not end.applicable(state):
        state = task.apply(state, task.wait)
    assert state is not None
    state = task.apply(state, end)
    assert state is not None and Atom("baked") not in task.atoms(state)
    plan, report = solve_text(OVEN_PLUS, oven_plus("(baked)"), tmp_path)
    assert plan is not None and "(lock-door)" in [str(s) for s in plan.steps]
    assert not report or "Plan valid" in report


def test_conditional_continuous_effects(tmp_path: Path) -> None:
    """Un efecto continuo condicional se decide al empezar: con el turbo, el horno calienta
    a ritmo 3 (12 en las 4 unidades que dura); sin él, a ritmo 1."""
    for init, expected in (("", 4.0), ("(turbo)", 12.0)):
        task = build(OVEN_PLUS, oven_plus("(baked)", f"(locked) {init}"))
        assert isinstance(task, DiscreteTask)
        end = task.actions[action(task, "(bake@end)")]
        state = task.apply(task.init, task.actions[action(task, "(bake@start)")])
        while state is not None and not end.applicable(state):
            state = task.apply(state, task.wait)
        assert state is not None
        heat = task.variable_index[next(v for v in task.variables if str(v) == "(heat)")]
        assert state.values[heat] == pytest.approx(expected)
    goal = "(and (>= (heat) 10) (not (baking)))"
    plan, report = solve_text(OVEN_PLUS, oven_plus(goal, "(locked)"), tmp_path)
    assert plan is not None and [str(s) for s in plan.steps] == ["(set-turbo)", "(bake)"]
    assert not report or "Plan valid" in report


# --- Tier 6 con VAL -------------------------------------------------------------------


def val(domain: Path, problem: Path, plan: Plan, tmp_path: Path) -> str:
    plan_file = tmp_path / "plan"
    plan_file.write_text(plan.to_ipc())
    return subprocess.run(
        [str(VALIDATE), str(domain), str(problem), str(plan_file)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout


@pytest.mark.parametrize("name", ["car", "generator"])
@pytest.mark.parametrize("problem", ["p01", "p02"])
def test_tier6(name: str, problem: str, tmp_path: Path) -> None:
    directory = tier_dir(f"tier6/{name}")
    domain_path, problem_path = directory / "domain.pddl", directory / f"{problem}.pddl"
    domain = read_domain(domain_path)
    result = solve(domain, read_problem(problem_path, domain), [PRESETS["bfws-ff-pref"]],
                   time_limit=60)  # fmt: skip
    assert result.plan is not None, result.search.message
    if VALIDATE.exists():
        output = val(domain_path, problem_path, result.plan, tmp_path)
        assert "Plan valid" in output, output
        value = re.search(r"Final value: (\S+)", output)
        if result.plan.metric is not None and value is not None:
            assert result.plan.metric == pytest.approx(float(value.group(1)), abs=1e-3)


def test_cli_delta(capsys: pytest.CaptureFixture[str]) -> None:
    car = tier_dir("tier6/car")
    code = main(["solve", "-c", "bfws-ff-pref", "--delta", "0.5",
                 str(car / "domain.pddl"), str(car / "p01.pddl")])  # fmt: skip
    assert code == 0
    assert "δ = 0.5" in capsys.readouterr().err
