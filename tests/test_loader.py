"""Tests de la fase 2: JSON → AST, con validación semántica."""

from typing import Any

import pytest

from pyplanner.errors import SchemaValidationError, SemanticError, UnsupportedFeatureError
from pyplanner.loader import load_domain, load_problem
from pyplanner.loader.ast import (
    Action,
    And,
    Atom,
    Constant,
    Domain,
    Not,
    ObjectDecl,
    Parameter,
    Variable,
)
from pyplanner.pddl import parse_domain, parse_problem
from tests.samples import DOMAIN, PROBLEM


@pytest.fixture
def domain() -> Domain:
    return load_domain(parse_domain(DOMAIN))


def x(name: str) -> Variable:
    return Variable(name)


def c(name: str) -> Constant:
    return Constant(name)


def test_domain_ast(domain: Domain) -> None:
    assert domain.name == "test"
    assert domain.requirements == {":strips"}
    assert domain.constants == (ObjectDecl("home"), ObjectDecl("robot"))
    assert [(p.name, p.arity) for p in domain.predicates] == [
        ("at", 2),
        ("free", 0),
        ("visited", 1),
    ]
    assert domain.action("move") == Action(
        name="move",
        parameters=(Parameter("?from"), Parameter("?to")),
        precondition=And((Atom("at", (c("robot"), x("?from"))), Atom("free"))),
        effect=And(
            (
                Atom("at", (c("robot"), x("?to"))),
                Not(Atom("at", (c("robot"), x("?from")))),
                Atom("visited", (x("?to"),)),
            )
        ),
    )


def test_missing_precondition_is_empty_conjunction(domain: Domain) -> None:
    rest = domain.action("rest")
    assert rest is not None
    assert rest.precondition == And()
    assert rest.effect == Atom("free")


def test_problem_ast(domain: Domain) -> None:
    problem = load_problem(parse_problem(PROBLEM), domain)
    assert problem.domain_name == "test"
    assert problem.objects == (ObjectDecl("kitchen"),)
    assert problem.init == {Atom("at", (c("robot"), c("home"))), Atom("free")}
    assert problem.goal == And(
        (Atom("visited", (c("kitchen"),)), Atom("at", (c("robot"), c("home"))))
    )
    assert all(isinstance(a, Atom) and a.is_ground for a in problem.init)


def test_ast_is_hashable(domain: Domain) -> None:
    assert len({domain, domain}) == 1


def test_json_names_are_normalised_to_lowercase() -> None:
    document = {
        "version": 1,
        "kind": "domain",
        "name": "D",
        "predicates": [{"name": "P", "parameters": [{"name": "?X"}]}],
        "actions": [
            {
                "name": "A",
                "parameters": [{"name": "?x"}],
                "effect": {"op": "atom", "predicate": "p", "args": ["?X"]},
            }
        ],
    }
    domain = load_domain(document)
    assert domain.name == "d"
    action = domain.action("a")
    assert action is not None and action.effect == Atom("p", (x("?x"),))


def test_objects_cannot_repeat_constants(domain: Domain) -> None:
    """Un objeto que repite una constante del dominio es un error (antes se ignoraba): cada
    nombre se declara una sola vez."""
    document = parse_problem(PROBLEM.replace("(:objects kitchen)", "(:objects kitchen home)"))
    message = "el nombre 'home' ya se ha definido previamente como constante en el dominio"
    with pytest.raises(SemanticError, match=message):
        load_problem(document, domain)


def test_schema_errors_are_reported_before_semantics() -> None:
    with pytest.raises(SchemaValidationError):
        load_domain({"kind": "domain", "name": "d"})


def test_loading_a_problem_as_domain_fails(domain: Domain) -> None:
    with pytest.raises(SemanticError, match="documento de dominio"):
        load_domain(parse_problem(PROBLEM))
    with pytest.raises(SemanticError, match="documento de problema"):
        load_problem(parse_domain(DOMAIN), domain)


def test_unsupported_requirements_in_json() -> None:
    document = {"version": 1, "kind": "domain", "name": "d", "requirements": [":time"]}
    with pytest.raises(UnsupportedFeatureError, match=":time"):
        load_domain(document)


def test_unknown_requirement_in_json() -> None:
    document = {"version": 1, "kind": "domain", "name": "d", "requirements": [":magic"]}
    with pytest.raises(SemanticError, match="desconocido") as exc:
        load_domain(document)
    assert exc.value.path == "requirements[0]"


def domain_with(body: str) -> str:
    return f"(define (domain d) (:constants k) (:predicates (p ?x) (q)) {body})"


@pytest.mark.parametrize(
    ("body", "message", "path"),
    [
        ("(:action a :effect (r))", "predicado 'r' no declarado", "actions[0].effect"),
        ("(:action a :effect (p))", "espera 1 argumentos y recibe 0", "actions[0].effect"),
        (
            "(:action a :parameters (?x) :precondition (and (q) (p ?y)))",
            "variable '\\?y' no está ligada",
            "actions[0].precondition.args[1]",
        ),
        ("(:action a :effect (not (p z)))", "'z' no declarado", "actions[0].effect.arg"),
        (
            "(:action a :parameters (?x ?x))",
            "parámetro '\\?x' repetido",
            "actions[0].parameters[1]",
        ),
        (
            "(:action a) (:action a)",
            "el nombre 'a' ya se ha definido previamente como acción",
            "actions[1]",
        ),
        (
            "(:action p)",
            "el nombre 'p' ya se ha definido previamente como predicado",
            "actions[0]",
        ),
    ],
)
def test_domain_semantic_errors(body: str, message: str, path: str) -> None:
    with pytest.raises(SemanticError, match=message) as exc:
        load_domain(parse_domain(domain_with(body), "d.pddl"))
    assert exc.value.path == path
    assert exc.value.location is not None and exc.value.location.file == "d.pddl"


def test_duplicate_declarations() -> None:
    previously = "ya se ha definido previamente como"
    with pytest.raises(SemanticError, match=f"'p' {previously} predicado"):
        load_domain(parse_domain("(define (domain d) (:predicates (p) (p ?x)))"))
    with pytest.raises(SemanticError, match=f"'k' {previously} constante"):
        load_domain(parse_domain("(define (domain d) (:constants k k))"))
    with pytest.raises(SemanticError, match=f"'d' {previously} nombre del dominio"):
        load_domain(parse_domain("(define (domain d) (:predicates (d)))"))
    with pytest.raises(SemanticError, match=f"'object' {previously} tipo predefinido"):
        load_domain(parse_domain("(define (domain d) (:predicates (object)))"))


def test_error_location_points_to_the_node() -> None:
    text = domain_with("\n(:action a\n :effect (and (q)\n   (r)))")
    with pytest.raises(SemanticError) as exc:
        load_domain(parse_domain(text, "d.pddl"))
    assert str(exc.value.location) == "d.pddl:4:4"


def problem_with(body: str, domain_name: str = "d") -> dict[str, Any]:
    return parse_problem(f"(define (problem prob) (:domain {domain_name}) {body})")


@pytest.mark.parametrize(
    ("body", "message", "path"),
    [
        ("(:goal (q))", None, None),
        ("(:init (p ?x)) (:goal (q))", "variable '\\?x' no está ligada", "init[0]"),
        ("(:init (p z)) (:goal (q))", "'z' no declarado", "init[0]"),
        ("(:objects o) (:init (p o)) (:goal (and (p k) (r)))", "'r' no declarado", "goal.args[1]"),
        (
            "(:objects o o) (:goal (q))",
            "el nombre 'o' ya se ha definido previamente como objeto",
            "objects[1]",
        ),
        (
            "(:objects q) (:goal (q))",
            "el nombre 'q' ya se ha definido previamente como predicado en el dominio",
            "objects[0]",
        ),
    ],
)
def test_problem_semantic_errors(body: str, message: str | None, path: str | None) -> None:
    domain = load_domain(parse_domain(domain_with("")))
    document = problem_with(body)
    if message is None:
        load_problem(document, domain)
        return
    with pytest.raises((SemanticError, UnsupportedFeatureError), match=message) as exc:
        load_problem(document, domain)
    if path is not None:
        assert exc.value.path == path


def test_unsupported_requirements_in_problem_json() -> None:
    domain = load_domain(parse_domain(domain_with("")))
    document = problem_with("(:goal (q))")
    document["requirements"] = [":time"]
    with pytest.raises(UnsupportedFeatureError, match=":time"):
        load_problem(document, domain)


def test_problem_for_another_domain() -> None:
    domain = load_domain(parse_domain(domain_with("")))
    with pytest.raises(SemanticError, match="dominio 'otro'"):
        load_problem(problem_with("(:goal (q))", "otro"), domain)
