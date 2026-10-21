"""Tests del parser (fases 1 y 2) para PDDL 3: restricciones de trayectoria, preferencias e
``is-violated``."""

from typing import Any

import pytest

from pyplanner.errors import (
    PDDLSyntaxError,
    SchemaValidationError,
    SemanticError,
    UndeclaredRequirementError,
)
from pyplanner.frontend import read_domain, read_problem
from pyplanner.grounding import ground
from pyplanner.loader import load_domain, load_problem
from pyplanner.loader.ast import (
    And,
    Arithmetic,
    Atom,
    Constant,
    ConstraintAnd,
    ConstraintForall,
    Forall,
    IsViolated,
    Number,
    Parameter,
    Preference,
    TimedCondition,
    TrajectoryConstraint,
    Variable,
    preferences_in,
)
from pyplanner.pddl import parse_domain, parse_pddl, parse_problem, write_pddl
from pyplanner.schema import validate_document
from tests.tiers import tier_dir

PDDL3 = ":typing :preferences :constraints :negative-preconditions :universal-preconditions"
DOMAIN = f"""
(define (domain d) (:requirements {PDDL3})
  (:types t)
  (:predicates (p ?x - t) (q) (r))
  (:action a :parameters (?x - t)
    :precondition (and (p ?x) (preference cheap (not (q))))
    :effect (q)))
"""


def problem(body: str, requirements: str = "") -> dict[str, Any]:
    declared = f"(:requirements {requirements})" if requirements else ""
    text = f"(define (problem x) (:domain d) {declared} (:objects o1 o2 - t) (:init) {body})"
    return parse_problem(text, locations=False)


def atom(predicate: str, *args: str) -> dict[str, Any]:
    return {"op": "atom", "predicate": predicate, "args": list(args)}


def loaded(body: str) -> Any:
    domain = load_domain(parse_domain(DOMAIN))
    return load_problem(problem(body), domain)


# --- Fase 1: preferencias ------------------------------------------------------------


def test_goal_preferences() -> None:
    document = problem(
        "(:goal (and (q) (preference g1 (r))"
        " (forall (?x - t) (preference g2 (p ?x))) (preference (not (r)))))"
    )
    assert document["goal"] == {
        "op": "and",
        "args": [
            atom("q"),
            {"op": "preference", "name": "g1", "arg": atom("r")},
            {
                "op": "forall",
                "parameters": [{"name": "?x", "type": "t"}],
                "arg": {"op": "preference", "name": "g2", "arg": atom("p", "?x")},
            },
            {"op": "preference", "arg": {"op": "not", "arg": atom("r")}},
        ],
    }


def test_precondition_preferences() -> None:
    action = parse_domain(DOMAIN, locations=False)["actions"][0]
    assert action["precondition"]["args"][1] == {
        "op": "preference",
        "name": "cheap",
        "arg": {"op": "not", "arg": atom("q")},
    }


def test_durative_condition_preferences() -> None:
    text = (
        "(define (domain d) (:requirements :durative-actions :preferences)"
        " (:predicates (p) (q))"
        " (:durative-action a :duration (= ?duration 1)"
        "  :condition (and (at start (p)) (preference early (over all (q))))"
        "  :effect (at end (p))))"
    )
    document = parse_domain(text, locations=False)
    condition = document["durative_actions"][0]["condition"]
    assert condition["args"][1] == {
        "op": "preference",
        "name": "early",
        "arg": {"op": "over-all", "arg": atom("q")},
    }
    loaded_domain = load_domain(document)
    assert loaded_domain.durative_actions[0].condition.args[1] == Preference(  # type: ignore[union-attr]
        "early", TimedCondition("all", Atom("q"))
    )
    assert loaded_domain.preference_names == {"early"}


@pytest.mark.parametrize(
    ("goal", "message"),
    [
        ("(or (q) (preference g (r)))", "una preferencia solo puede ir"),
        ("(not (preference g (r)))", "una preferencia solo puede ir"),
        ("(preference g (preference h (r)))", "una preferencia solo puede ir"),
        ("(preference g (r) (q))", r"\(preference \[<nombre>\] <fórmula>\)"),
    ],
)
def test_misplaced_preferences(goal: str, message: str) -> None:
    with pytest.raises(PDDLSyntaxError, match=message):
        problem(f"(:goal {goal})")


def test_durative_preferences_wrap_a_timed_condition() -> None:
    text = (
        "(define (domain d) (:requirements :durative-actions :preferences) (:predicates (p))"
        " (:durative-action a :duration (= ?duration 1) :condition (preference g (p))"
        "  :effect (at end (p))))"
    )
    with pytest.raises(PDDLSyntaxError, match="envuelve una condición"):
        parse_domain(text)


# --- Fase 1: restricciones -----------------------------------------------------------

ALL_CONSTRAINTS = """(:constraints (and
    (at end (q)) (always (q)) (sometime (r)) (at-most-once (r))
    (sometime-after (q) (r)) (sometime-before (r) (q))
    (within 10 (q)) (hold-after 5 (r)) (always-within 3.5 (q) (r)) (hold-during 1 2 (q))
    (forall (?x - t) (preference c1 (sometime (p ?x))))))"""


def test_constraint_documents() -> None:
    document = problem(f"(:goal (q)) {ALL_CONSTRAINTS}")
    args = document["constraints"]["args"]
    assert [a["op"] for a in args] == [
        "at-end",
        "always",
        "sometime",
        "at-most-once",
        "sometime-after",
        "sometime-before",
        "within",
        "hold-after",
        "always-within",
        "hold-during",
        "forall",
    ]
    assert args[4] == {"op": "sometime-after", "args": [atom("q"), atom("r")]}
    assert args[6] == {"op": "within", "time": 10.0, "arg": atom("q")}
    assert args[8] == {"op": "always-within", "time": 3.5, "args": [atom("q"), atom("r")]}
    assert args[9] == {"op": "hold-during", "from": 1.0, "to": 2.0, "arg": atom("q")}
    assert args[10]["arg"] == {
        "op": "preference",
        "name": "c1",
        "arg": {"op": "sometime", "arg": atom("p", "?x")},
    }


def test_domain_constraints() -> None:
    text = DOMAIN.replace(
        "(:action a", "(:constraints (forall (?x - t) (at-most-once (p ?x))))\n  (:action a"
    )
    document = parse_domain(text, locations=False)
    assert document["constraints"] == {
        "op": "forall",
        "parameters": [{"name": "?x", "type": "t"}],
        "arg": {"op": "at-most-once", "arg": atom("p", "?x")},
    }
    domain = load_domain(document)
    assert domain.constraints == ConstraintForall(
        (Parameter("?x", ("t",)),),
        TrajectoryConstraint("at-most-once", (Atom("p", (Variable("?x"),)),)),
    )


@pytest.mark.parametrize(
    ("constraints", "message"),
    [
        ("(eventually (q))", "se esperaba una restricción"),
        ("(always (q) (r))", r"\(always <fórmula>\)"),
        ("(within (q))", r"\(within <número> <fórmula>\)"),
        ("(hold-during 1 (q))", r"\(hold-during <número> <número> <fórmula>\)"),
        ("(within x (q))", "'x' no es un número"),
        ("(preference p (preference q (sometime (r))))", "no puede contener otra"),
        ("(always (sometime (q)))", "los operadores modales no se anidan"),
        ("(within 5 (at end (q)))", "los operadores modales no se anidan"),
    ],
)
def test_constraint_syntax_errors(constraints: str, message: str) -> None:
    with pytest.raises(PDDLSyntaxError, match=message):
        problem(f"(:goal (q)) (:constraints {constraints})")


def test_domain_constraints_have_no_preferences() -> None:
    text = DOMAIN.replace("(:action a", "(:constraints (preference p (sometime (q))))\n(:action a")
    with pytest.raises(PDDLSyntaxError, match="las restricciones del dominio no admiten"):
        parse_domain(text)


def test_is_violated_in_the_metric() -> None:
    document = problem("(:goal (preference g (q))) (:metric minimize (* 2 (is-violated g)))")
    assert document["metric"]["expression"] == {
        "op": "*",
        "args": [{"op": "num", "value": 2.0}, {"op": "is-violated", "name": "g"}],
    }


# --- Modo estricto -------------------------------------------------------------------


def test_undeclared_pddl3_requirements() -> None:
    plain = DOMAIN.replace(":preferences :constraints ", "").replace(
        "(preference cheap (not (q)))", "(not (q))"
    )
    domain = load_domain(parse_domain(plain))
    with pytest.raises(UndeclaredRequirementError, match=":preferences"):
        load_problem(problem("(:goal (preference g (q)))"), domain)
    with pytest.raises(UndeclaredRequirementError, match=":constraints"):
        load_problem(problem("(:goal (q)) (:constraints (sometime (q)))"), domain)
    # Declarados en el problema, valen.
    load_problem(
        problem("(:goal (q)) (:constraints (sometime (q)))", ":constraints"),
        domain,
    )


def test_undeclared_requirements_in_domain_preconditions() -> None:
    text = DOMAIN.replace(":preferences ", "")
    with pytest.raises(UndeclaredRequirementError, match=":preferences"):
        parse_domain(text)


def test_preference_metrics_need_only_preferences() -> None:
    """Una métrica de números, aritmética e is-violated se acepta con :preferences (como
    en la IPC 2006); con otras funciones hace falta :numeric-fluents."""
    domain = load_domain(parse_domain(DOMAIN))
    load_problem(
        problem("(:goal (preference g (q))) (:metric minimize (+ 1 (* 2 (is-violated g))))"),
        domain,
    )
    with pytest.raises(UndeclaredRequirementError, match=":numeric-fluents"):
        load_problem(
            problem(
                "(:goal (preference g (q))) (:metric minimize (+ (total-cost) (is-violated g)))"
            ),
            domain,
        )


# --- Fase 2 --------------------------------------------------------------------------


def test_goal_and_constraint_ast() -> None:
    result = loaded(
        "(:goal (and (q) (forall (?x - t) (preference g (p ?x)))))"
        " (:constraints (and (within 10 (q)) (preference c (sometime-before (r) (q)))))"
        " (:metric minimize (+ (is-violated g) (* 3 (is-violated c)) (is-violated cheap)))"
    )
    x = Variable("?x")
    assert result.goal == And(
        (Atom("q"), Forall((Parameter("?x", ("t",)),), Preference("g", Atom("p", (x,))))),
    )
    assert result.constraints == ConstraintAnd(
        (
            TrajectoryConstraint("within", (Atom("q"),), (10.0,)),
            Preference("c", TrajectoryConstraint("sometime-before", (Atom("r"), Atom("q")))),
        )
    )
    assert result.metric.expression == Arithmetic(
        "+",
        (
            IsViolated("g"),
            Arithmetic("*", (Number(3.0), IsViolated("c"))),
            IsViolated("cheap"),  # preferencia de una precondición del dominio
        ),
    )


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            "(:goal (preference g (q))) (:metric minimize (is-violated h))",
            "la preferencia 'h' no aparece",
        ),
        (
            "(:requirements :numeric-fluents) (:goal (< (is-violated g) 1))",
            "'is-violated' solo puede aparecer en la métrica",
        ),
        ("(:goal (q)) (:constraints (hold-during 5 2 (q)))", "inicio ≤ fin"),
        ("(:goal (q)) (:constraints (sometime (p o3)))", "'o3' no declarado"),
        ("(:goal (q)) (:constraints (sometime (p ?y)))", "no está ligada"),
    ],
)
def test_pddl3_semantic_errors(body: str, message: str) -> None:
    with pytest.raises(SemanticError, match=message):
        loaded(body)


def test_schema_keeps_preferences_in_place() -> None:
    document = problem("(:goal (q))")
    document["goal"] = {"op": "or", "args": [{"op": "preference", "arg": atom("q")}]}
    with pytest.raises(SchemaValidationError):
        validate_document(document)
    domain = parse_domain(DOMAIN)
    domain["constraints"] = {"op": "preference", "arg": {"op": "sometime", "arg": atom("q")}}
    with pytest.raises(SchemaValidationError):
        validate_document(domain)
    document = problem("(:goal (q))")
    document["constraints"] = {"op": "always", "arg": {"op": "sometime", "arg": atom("q")}}
    with pytest.raises(SchemaValidationError):
        validate_document(document)


# --- Documentos completos y tier 5 ---------------------------------------------------


def test_round_trip_and_schema() -> None:
    texts = [
        DOMAIN,
        f"(define (problem x) (:domain d) (:objects o1 o2 - t) (:init (p o1))"
        f" (:goal (and (q) (preference g (r)))) {ALL_CONSTRAINTS}"
        " (:metric minimize (+ (is-violated g) (* 2 (is-violated c1)))))",
    ]
    for text in texts:
        document = parse_pddl(text, locations=False)
        validate_document(document)
        assert parse_pddl(write_pddl(document), locations=False) == document


def test_trucks_loads() -> None:
    path = tier_dir("tier5/constraints/trucks")
    domain = read_domain(path / "domain.pddl")
    result = read_problem(path / "p01.pddl", domain)
    assert result.goal == And()
    assert result.constraints == ConstraintAnd(
        (
            TrajectoryConstraint(
                "within",
                (Atom("delivered", (Constant("package1"), Constant("l1"))),),
                (919.7,),
            ),
            TrajectoryConstraint(
                "within",
                (Atom("delivered", (Constant("package2"), Constant("l2"))),),
                (919.7,),
            ),
            TrajectoryConstraint(
                "within",
                (Atom("delivered", (Constant("package3"), Constant("l2"))),),
                (1813.7,),
            ),
        )
    )


def test_storage_loads_and_grounds() -> None:
    path = tier_dir("tier5/preferences/storage")
    domain = read_domain(path / "domain.pddl")
    assert domain.supertypes("area") == ("area", "surface", "object")
    result = read_problem(path / "p02.pddl", domain)
    assert result.metric is not None
    names = {p.name for node in (result.goal, result.constraints) for p in preferences_in(node)}
    expected = {"p0a", "p1a", "p1b", "p1c", "p3a", "p3b", "p4a", "p4b", "p5a", "p6a", "p7a"}
    assert names == expected
    task = ground(domain, result)
    assert task.preferences is not None
    # p0a pesa 0 en la métrica: no cuenta.
    assert "p0a" not in dict(task.preferences.weights)
