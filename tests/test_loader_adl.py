"""Tests de la fase 2 para PDDL 1.2 (subconjunto ADL): tipos, cuantificadores y modo estricto."""

from typing import Any

import pytest

from pyplanner.errors import SemanticError, UndeclaredRequirementError
from pyplanner.loader import load_domain, load_problem
from pyplanner.loader.ast import (
    And,
    Atom,
    Constant,
    Domain,
    Equals,
    Exists,
    Forall,
    ForallEffect,
    Imply,
    Not,
    ObjectDecl,
    Or,
    Parameter,
    Variable,
    When,
)
from pyplanner.pddl import parse_domain, parse_problem

TYPED = """
(define (domain logistics)
  (:requirements :adl)
  (:types truck plane - vehicle package vehicle - locatable place)
  (:constants depot - place)
  (:predicates (at ?x - locatable ?p - place) (in ?p - package ?v - vehicle))
  (:action unload-all
    :parameters (?v - vehicle ?p - place)
    :precondition (and (at ?v ?p) (not (= ?p depot)))
    :effect (forall (?x - package) (when (in ?x ?v) (and (at ?x ?p) (not (in ?x ?v)))))))
"""


@pytest.fixture
def typed() -> Domain:
    return load_domain(parse_domain(TYPED))


def load(domain_body: str, requirements: str = ":adl") -> Domain:
    return load_domain(
        parse_domain(f"(define (domain d) (:requirements {requirements}) {domain_body})")
    )


# --- Tipos --------------------------------------------------------------------------


def test_type_hierarchy(typed: Domain) -> None:
    assert dict(typed.types) == {
        "truck": "vehicle",
        "plane": "vehicle",
        "package": "locatable",
        "vehicle": "locatable",
        "locatable": "object",  # declarado implícitamente como supertipo
        "place": "object",
    }
    assert typed.supertypes("truck") == ("truck", "vehicle", "locatable", "object")
    assert typed.is_subtype(("truck",), ("locatable",))
    assert not typed.is_subtype(("place",), ("locatable",))
    assert typed.is_subtype(("truck", "plane"), ("vehicle",))
    assert not typed.is_subtype(("truck", "place"), ("vehicle",))
    assert typed.constants == (ObjectDecl("depot", ("place",)),)


def test_typed_action_ast(typed: Domain) -> None:
    action = typed.action("unload-all")
    assert action is not None
    assert action.parameters == (Parameter("?v", ("vehicle",)), Parameter("?p", ("place",)))
    v, p, x = Variable("?v"), Variable("?p"), Variable("?x")
    assert action.precondition == And((Atom("at", (v, p)), Not(Equals(p, Constant("depot")))))
    assert action.effect == ForallEffect(
        (Parameter("?x", ("package",)),),
        When(Atom("in", (x, v)), And((Atom("at", (x, p)), Not(Atom("in", (x, v)))))),
    )


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("(:types a - b b - a)", "ciclo en la jerarquía"),
        ("(:types a a)", "tipo 'a' declarado dos veces"),
        ("(:constants c - nope)", "tipo 'nope' no declarado"),
        ("(:predicates (p ?x - nope))", "tipo 'nope' no declarado"),
        ("(:types t) (:constants c - (either t nope))", "tipo 'nope' no declarado"),
    ],
)
def test_type_errors(body: str, message: str) -> None:
    with pytest.raises(SemanticError, match=message):
        load(body)


def test_declaring_object_is_harmless() -> None:
    assert load("(:types object t)").types == (("t", "object"),)


def problem(body: str) -> dict[str, Any]:
    return parse_problem(f"(define (problem x) (:domain logistics) {body})")


def test_typed_problem(typed: Domain) -> None:
    loaded = load_problem(
        problem(
            "(:objects t1 - truck p1 - package home - place)"
            " (:init (at t1 home) (in p1 t1))"
            " (:goal (forall (?x - package) (at ?x depot)))"
        ),
        typed,
    )
    assert loaded.objects == (
        ObjectDecl("t1", ("truck",)),
        ObjectDecl("p1", ("package",)),
        ObjectDecl("home", ("place",)),
    )
    x = Variable("?x")
    assert loaded.goal == Forall(
        (Parameter("?x", ("package",)),), Atom("at", (x, Constant("depot")))
    )


@pytest.mark.parametrize(
    ("body", "message", "path"),
    [
        (
            "(:objects p1 - package home - place) (:init (in p1 home)) (:goal (and))",
            "'home' \\(place\\) no es del tipo vehicle",
            "init[0]",
        ),
        (
            "(:objects h - place) (:goal (at depot h))",
            "'depot' \\(place\\) no es del tipo locatable",
            "goal",
        ),
    ],
)
def test_constants_must_match_predicate_types(
    typed: Domain, body: str, message: str, path: str
) -> None:
    with pytest.raises(SemanticError, match=message) as exc:
        load_problem(problem(body), typed)
    assert exc.value.path == path


# --- Fórmulas y ámbito de las variables ---------------------------------------------


def test_adl_formula_ast() -> None:
    domain = load(
        "(:predicates (p ?x) (q))"
        " (:action a :parameters (?y)"
        "  :precondition (or (imply (q) (p ?y)) (exists (?x) (p ?x)))"
        "  :effect (and))"
    )
    action = domain.action("a")
    assert action is not None
    y, x = Variable("?y"), Variable("?x")
    assert action.precondition == Or(
        (
            Imply(Atom("q"), Atom("p", (y,))),
            Exists((Parameter("?x"),), Atom("p", (x,))),
        )
    )


@pytest.mark.parametrize(
    ("body", "message", "path"),
    [
        (
            "(:action a :parameters (?x) :precondition (exists (?x) (p ?x)))",
            "'\\?x' ya está ligada",
            "actions[0].precondition.parameters[0]",
        ),
        (
            "(:action a :precondition (and (exists (?x) (p ?x)) (p ?x)))",
            "'\\?x' no está ligada",
            "actions[0].precondition.args[1]",
        ),
        (
            "(:action a :effect (forall (?x) (when (p ?y) (p ?x))))",
            "'\\?y' no está ligada",
            "actions[0].effect.arg.condition",
        ),
        (
            "(:action a :precondition (= ?x k))",
            "'\\?x' no está ligada",
            "actions[0].precondition",
        ),
    ],
)
def test_variable_scope_errors(body: str, message: str, path: str) -> None:
    with pytest.raises(SemanticError, match=message) as exc:
        load(f"(:constants k) (:predicates (p ?x)) {body}")
    assert exc.value.path == path


def test_sibling_quantifiers_may_reuse_names() -> None:
    load(
        "(:predicates (p ?x))"
        " (:action a :precondition (and (exists (?x) (p ?x)) (forall (?x) (p ?x))))"
    )


# --- Modo estricto en la fase 2 -----------------------------------------------------


def test_json_from_another_source_is_checked() -> None:
    document = {
        "version": 2,
        "kind": "domain",
        "name": "d",
        "requirements": [":strips"],
        "types": [{"name": "t"}],
    }
    with pytest.raises(UndeclaredRequirementError, match=":typing") as exc:
        load_domain(document)
    assert exc.value.path == "types"


def test_problem_constructs_use_domain_and_problem_requirements() -> None:
    domain = load("(:predicates (p))", ":strips")
    with pytest.raises(UndeclaredRequirementError, match=":negative-preconditions"):
        load_problem(parse_problem("(define (problem x) (:domain d) (:goal (not (p))))"), domain)
    # Declarado en el propio problema: se acepta.
    load_problem(
        parse_problem(
            "(define (problem x) (:domain d) (:requirements :negative-preconditions)"
            " (:goal (not (p))))"
        ),
        domain,
    )
    # Declarado en el dominio: también.
    negative = load("(:predicates (p))", ":negative-preconditions")
    load_problem(parse_problem("(define (problem x) (:domain d) (:goal (not (p))))"), negative)


def test_typed_objects_need_typing() -> None:
    domain = load("(:predicates (p))", ":strips")
    with pytest.raises(UndeclaredRequirementError, match=":typing") as exc:
        load_problem(
            parse_problem("(define (problem x) (:domain d) (:objects o - object) (:goal (p)))"),
            domain,
        )
    assert exc.value.path == "objects[0]"
