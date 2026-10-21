"""Tests del motor con ADL: normalización, grounding, semántica, heurísticas y búsqueda."""

import subprocess
from pathlib import Path

import pytest

from pyplanner.errors import UnsupportedFeatureError
from pyplanner.grounding import GroundTask, State, mask_of
from pyplanner.grounding import normalize as normalize_module
from pyplanner.grounding.normalize import TRUE, Clause, Normalizer
from pyplanner.heuristics import TENSOR_HEURISTICS, create_heuristic
from pyplanner.loader import load_domain
from pyplanner.loader.ast import (
    And,
    Atom,
    Constant,
    Equals,
    Exists,
    Forall,
    Imply,
    Not,
    Or,
    Parameter,
    Variable,
)
from pyplanner.pddl import parse_domain, parse_pddl, write_pddl
from pyplanner.plan import Plan, simulate
from pyplanner.search import PRESETS, SearchConfig, SearchStatus, search
from pyplanner.search.engine import _TypeBuckets
from tests.adl_tasks import ALL, CHOICE, DOORS, LIGHTS, ROTATE
from tests.tasks import build
from tests.test_heuristics import ACCELERATED_DEVICES, all_states
from tests.tiers import VALIDATE


def task_of(name: str) -> GroundTask:
    domain, problem = ALL[name]
    return build(domain, problem)


def facts(task: GroundTask, indices: tuple[int, ...]) -> list[str]:
    return [str(task.facts[i]) for i in indices]


# --- Normalización ------------------------------------------------------------------


def normalizer(objects: list[str], static: dict[str, set[tuple[str, ...]]]) -> Normalizer:
    return Normalizer(
        lambda types: objects,
        frozenset(static),
        frozenset((p, args) for p, all_args in static.items() for args in all_args),
    )


P, Q = Atom("p"), Atom("q")
x = Variable("?x")


def clauses(dnf: list[Clause]) -> set[tuple[frozenset[str], frozenset[str]]]:
    def names(fs: frozenset[tuple[str, tuple[str, ...]]]) -> frozenset[str]:
        return frozenset(f"{p}{''.join(args)}" for p, args in fs)

    return {(names(c.pos), names(c.neg)) for c in dnf}


def test_dnf_of_connectives() -> None:
    n = normalizer(["a", "b"], {})
    assert clauses(n.dnf(And((P, Not(Q))), {})) == {(frozenset({"p"}), frozenset({"q"}))}
    assert clauses(n.dnf(Or((P, Q)), {})) == {
        (frozenset({"p"}), frozenset()),
        (frozenset({"q"}), frozenset()),
    }
    # not (p and q) = not p or not q
    assert clauses(n.dnf(Not(And((P, Q))), {})) == {
        (frozenset(), frozenset({"p"})),
        (frozenset(), frozenset({"q"})),
    }
    # p -> q = not p or q ;  not (p -> q) = p and not q
    assert clauses(n.dnf(Imply(P, Q), {})) == {
        (frozenset(), frozenset({"p"})),
        (frozenset({"q"}), frozenset()),
    }
    assert clauses(n.dnf(Not(Imply(P, Q)), {})) == {(frozenset({"p"}), frozenset({"q"}))}


def test_dnf_of_quantifiers_and_equality() -> None:
    n = normalizer(["a", "b"], {})
    r = Atom("r", (x,))
    assert clauses(n.dnf(Exists((Parameter("?x"),), r), {})) == {
        (frozenset({"ra"}), frozenset()),
        (frozenset({"rb"}), frozenset()),
    }
    assert clauses(n.dnf(Forall((Parameter("?x"),), r), {})) == {
        (frozenset({"ra", "rb"}), frozenset())
    }
    # forall x: x = a or r(x)   →   solo queda r(b)
    body = Or((Equals(x, Constant("a")), r))
    assert clauses(n.dnf(Forall((Parameter("?x"),), body), {})) == {
        (frozenset({"rb"}), frozenset())
    }


def test_static_facts_are_evaluated() -> None:
    n = normalizer(["a"], {"s": {("a",)}})
    assert n.dnf(Atom("s", (Constant("a"),)), {}) == TRUE
    assert n.dnf(Not(Atom("s", (Constant("a"),))), {}) == []
    assert n.dnf(And((Atom("s", (x,)), P)), {"?x": "a"}) == [Clause(pos=frozenset([("p", ())]))]


def test_contradictions_and_subsumption() -> None:
    n = normalizer([], {})
    assert n.dnf(And((P, Not(P))), {}) == []
    # (p) or (p and q)  →  (p)
    assert n.dnf(Or((P, And((P, Q)))), {}) == [Clause(pos=frozenset([("p", ())]))]


def test_dnf_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(normalize_module, "MAX_CLAUSES", 3)
    n = normalizer([], {})
    formula = And(tuple(Or((Atom(f"a{i}"), Atom(f"b{i}"))) for i in range(3)))
    with pytest.raises(UnsupportedFeatureError, match="predicados derivados"):
        n.dnf(formula, {})


# --- Grounding ----------------------------------------------------------------------


def test_conditional_effects_are_grounded() -> None:
    task = task_of("rotate")
    [rotate] = task.actions
    assert rotate.pre_facts == () and rotate.add_facts == () and rotate.del_facts == ()
    # 2 objetos × 3 posiciones, todas alcanzables.
    assert len(rotate.conditional) == 6
    effect = rotate.conditional[0]
    assert facts(task, effect.condition.pos_facts) == ["(at x1 a)"]
    assert facts(task, effect.del_facts) == ["(at x1 a)"]
    assert facts(task, effect.add_facts) == ["(at x1 b)"]


def test_negative_preconditions_and_goals() -> None:
    task = task_of("lights")
    switch_on = [a for a in task.actions if a.name == "switch-on"]
    assert [facts(task, a.neg_facts) for a in switch_on] == [["(on l1)"], ["(on l2)"]]
    [clause] = task.goal
    assert facts(task, clause.pos_facts) == ["(on l1)"]
    assert facts(task, clause.neg_facts) == ["(on l2)"]


def test_disjunctive_precondition_splits_actions() -> None:
    task = task_of("doors")
    opens = [a for a in task.actions if a.name == "open-door" and a.args == ("d1",)]
    # (exists k (has k)) or (not (locked d1)); locked es estático y cierto para d1, así
    # que solo queda la rama de la llave: una acción por llave posible (k1; fake no se
    # puede coger, así que (has fake) es inalcanzable).
    assert [facts(task, a.pre_facts) for a in opens] == [["(has k1)"]]
    opens_d2 = [a for a in task.actions if a.name == "open-door" and a.args == ("d2",)]
    assert [a.pre_facts for a in opens_d2] == [()]  # d2 no está cerrada con llave
    assert not any(a.name == "take" and a.args == ("fake",) for a in task.actions)
    [leave] = [a for a in task.actions if a.name == "leave"]
    assert facts(task, leave.pre_facts) == ["(open d1)", "(open d2)"]


def test_disjunctive_goal_has_several_clauses() -> None:
    task = task_of("choice")
    # (p) or (imply (not r) p) or (q and r)  =  p  or  r  or  (q and r)  →  p  or  r
    assert sorted(facts(task, c.pos_facts)[0] for c in task.goal) == ["(p)", "(r)"]


def test_condition_implied_by_precondition_becomes_unconditional() -> None:
    task = build(
        "(define (domain d) (:requirements :adl) (:predicates (p) (q) (r))"
        " (:action a :precondition (p)"
        "  :effect (and (when (p) (q)) (when (and (p) (not (p))) (r)) (when (not (p)) (r)))))",
        "(define (problem x) (:domain d) (:init (p)) (:goal (q)))",
    )
    [action] = task.actions
    assert facts(task, action.add_facts) == ["(q)"]
    assert action.conditional == ()


# --- Semántica ----------------------------------------------------------------------


def test_conditional_effects_read_the_previous_state() -> None:
    task = task_of("rotate")
    [rotate] = task.actions
    state = rotate.apply(task.init)
    assert state is not None
    assert sorted(str(f) for f in task.atoms(state)) == ["(at x1 b)", "(at x2 c)"]
    state = rotate.apply(state)
    assert state is not None and task.is_goal(state)


def test_negative_preconditions_block_actions() -> None:
    task = task_of("lights")
    names = {str(a) for a in task.applicable(task.init)}
    assert names == {"(switch-on l1)", "(switch-off l2)"}


def test_goal_with_negative_literals() -> None:
    task = task_of("lights")
    on_l1 = mask_of([task.fact_index[Atom("on", (Constant("l1"),))]])
    on_l2 = mask_of([task.fact_index[Atom("on", (Constant("l2"),))]])
    assert task.is_goal(State(on_l1))
    assert not task.is_goal(State(on_l1 | on_l2))


# --- Heurísticas --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        # rotate: cada objetivo necesita dos efectos condicionales encadenados (h_max = 2,
        # h_add = 2 + 2), pero los cuatro son del mismo operador: h_FF lo cuenta una vez.
        ("rotate", {"hmax": 2, "hadd": 4, "ff": 1, "goalcount": 2}),
        # lights: encender l1 y apagar l2 (el complementario de (on l2)).
        ("lights", {"hmax": 1, "hadd": 2, "ff": 2, "goalcount": 2}),
        # doors: coger k1, abrir d1 y d2, salir.
        ("doors", {"hmax": 3, "hadd": 4, "ff": 4, "goalcount": 1}),
        # choice: el mínimo entre cláusulas es make-r (1).
        ("choice", {"hmax": 1, "hadd": 1, "ff": 1, "goalcount": 1}),
    ],
)
def test_reference_values(name: str, expected: dict[str, float]) -> None:
    task = task_of(name)
    values = {h: create_heuristic(h, task).evaluate(task.init).value for h in expected}
    assert values == expected


def test_ff_counts_each_operator_once() -> None:
    # Una sola acción con dos efectos condicionales que dan los dos objetivos.
    task = build(
        "(define (domain d) (:requirements :adl) (:predicates (s) (p) (q))"
        " (:action a :effect (and (when (s) (p)) (when (s) (q)))))",
        "(define (problem x) (:domain d) (:init (s)) (:goal (and (p) (q))))",
    )
    evaluation = create_heuristic("ff", task).evaluate(task.init)
    assert evaluation.value == 1
    assert evaluation.preferred == (0,)


def test_preferred_operators_respect_negative_preconditions() -> None:
    task = task_of("lights")
    evaluation = create_heuristic("ff", task).evaluate(task.init)
    assert sorted(str(task.actions[a]) for a in evaluation.preferred) == [
        "(switch-off l2)",
        "(switch-on l1)",
    ]


@pytest.mark.skipif(not ACCELERATED_DEVICES, reason="ni Numba ni PyTorch están instalados")
@pytest.mark.parametrize("device", ACCELERATED_DEVICES)
@pytest.mark.parametrize("heuristic", TENSOR_HEURISTICS)
@pytest.mark.parametrize("name", sorted(ALL))
def test_accelerated_matches_reference_on_adl(name: str, heuristic: str, device: str) -> None:
    task = task_of(name)
    states = all_states(task)
    reference = create_heuristic(heuristic, task, "python").evaluate_batch(states)
    assert create_heuristic(heuristic, task, device).evaluate_batch(states) == reference


# --- Búsqueda -----------------------------------------------------------------------


@pytest.mark.parametrize("preset", sorted(PRESETS))
@pytest.mark.parametrize("name", sorted(ALL))
def test_every_preset_solves_adl_tasks(name: str, preset: str) -> None:
    task = task_of(name)
    result = search(task, PRESETS[preset])
    if not PRESETS[preset].is_complete and result.status is SearchStatus.FAILED:
        return  # IW: problemas de más anchura que su k
    assert result.status is SearchStatus.SOLVED
    assert result.plan is not None and simulate(task, result.plan) is None


@pytest.mark.parametrize(("name", "cost"), [("rotate", 2), ("lights", 2), ("doors", 4)])
def test_optimal_costs(name: str, cost: int) -> None:
    assert search(task_of(name), PRESETS["astar-hmax"]).cost == cost


def test_unsolvable_adl_task() -> None:
    # Hay que apagar y dejar encendida la misma luz.
    task = build(LIGHTS[0], LIGHTS[1].replace("(not (on l2))", "(not (on l1))"))
    assert task.goal == ()  # contradictorio: se detecta al normalizar
    assert search(task, PRESETS["gbfs-ff"]).status is SearchStatus.UNSOLVABLE


@pytest.mark.skipif(not VALIDATE.exists(), reason="VAL no está compilado")
@pytest.mark.parametrize("name", sorted(ALL))
def test_plans_are_valid_for_val(name: str, tmp_path: Path) -> None:
    domain, problem = ALL[name]
    task = build(domain, problem)
    result = search(task, PRESETS["gbfs-ff-pref-tb"])
    assert result.plan is not None
    # VAL recibe el PDDL regenerado desde el JSON: el camino de una tarea que llega en JSON.
    files = {
        "domain.pddl": write_pddl(parse_pddl(domain)),
        "problem.pddl": write_pddl(parse_pddl(problem)),
    }
    for file_name, text in files.items():
        (tmp_path / file_name).write_text(text)
    (tmp_path / "plan").write_text(Plan.from_indices(task, result.plan).to_ipc())
    output = subprocess.run(
        [str(VALIDATE), *(str(tmp_path / f) for f in (*files, "plan"))],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    assert "Plan valid" in output, output


# --- Exploración por tipos ----------------------------------------------------------


def test_type_buckets_pop_everything_once() -> None:
    buckets = _TypeBuckets(seed=0)
    items = [((h, g), (i,)) for i, (h, g) in enumerate((h, g) for h in range(3) for g in range(3))]
    for key, item in items:
        buckets.push(key, item)
    popped = []
    while buckets:
        popped.append(buckets.pop())
    assert sorted(popped) == sorted(item for _, item in items)


def test_type_based_exploration_is_reproducible() -> None:
    task = task_of("doors")
    config = SearchConfig("tb", "gbfs", "ff", exploration="type-based", seed=3)
    first, second = search(task, config), search(task, config)
    assert first.plan == second.plan
    assert first.statistics.expanded == second.statistics.expanded


def test_invalid_exploration() -> None:
    with pytest.raises(Exception, match="exploración desconocida"):
        SearchConfig(exploration="random-walk")  # type: ignore[arg-type]


def test_domain_with_typed_constants_only() -> None:
    domain = load_domain(parse_domain(DOORS[0]))
    assert [c.name for c in domain.constants] == ["fake"]
    assert ROTATE[0] and CHOICE[0]  # las tareas se usan en otros tests


# --- Casos límite -------------------------------------------------------------------


def test_conditional_effect_with_only_negative_condition() -> None:
    # "toggle" enciende la luz si está apagada y la apaga si está encendida.
    task = build(
        "(define (domain t) (:requirements :adl) (:predicates (on) (clicked))"
        " (:action toggle :parameters ()"
        "  :precondition (and)"
        "  :effect (and (clicked) (when (on) (not (on))) (when (not (on)) (on)))))",
        "(define (problem x) (:domain t) (:init) (:goal (and (on) (clicked))))",
    )
    [toggle] = task.actions
    assert [len(e.condition.pos_facts) for e in toggle.conditional] == [1, 0]
    once = toggle.apply(task.init)
    assert once is not None
    twice = toggle.apply(once)
    assert twice is not None
    assert task.is_goal(once) and not task.is_goal(twice)
    assert search(task, PRESETS["gbfs-ff-pref-tb"]).cost == 1


def test_actions_without_effects_are_dropped() -> None:
    task = build(
        "(define (domain t) (:requirements :adl) (:predicates (p) (q))"
        " (:action idle :parameters () :precondition (p) :effect (and))"
        " (:action unreachable-effect :parameters ()"
        "  :precondition (p) :effect (when (q) (not (p)))))",
        "(define (problem x) (:domain t) (:init (p)) (:goal (p)))",
    )
    # idle no cambia nada; el efecto de la otra exige (q), que es inalcanzable.
    assert task.actions == ()


def test_blind_heuristic_with_impossible_goal() -> None:
    task = build(LIGHTS[0], LIGHTS[1].replace("(not (on l2))", "(not (on l1))"))
    assert create_heuristic("blind", task).evaluate(task.init).is_dead_end
    assert create_heuristic("goalcount", task).evaluate(task.init).is_dead_end
