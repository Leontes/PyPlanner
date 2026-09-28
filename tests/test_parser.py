"""Tests de la fase 1: PDDL → JSON."""

from typing import Any

import pytest

from pyplanner.errors import PDDLSyntaxError, UnsupportedFeatureError
from pyplanner.pddl import parse_domain, parse_pddl, parse_problem
from pyplanner.schema import SCHEMA_VERSION, validate_document
from tests.samples import DOMAIN, PROBLEM


def atom(predicate: str, *args: str) -> dict[str, Any]:
    return {"op": "atom", "predicate": predicate, "args": list(args)}


def test_domain_document() -> None:
    document = parse_domain(DOMAIN, locations=False)
    assert document == {
        "version": SCHEMA_VERSION,
        "kind": "domain",
        "name": "test",
        "requirements": [":strips"],
        "types": [],
        "constants": [{"name": "home"}, {"name": "robot"}],
        "predicates": [
            {"name": "at", "parameters": [{"name": "?x"}, {"name": "?y"}]},
            {"name": "free", "parameters": []},
            {"name": "visited", "parameters": [{"name": "?x"}]},
        ],
        "actions": [
            {
                "name": "move",
                "parameters": [{"name": "?from"}, {"name": "?to"}],
                "precondition": {"op": "and", "args": [atom("at", "robot", "?from"), atom("free")]},
                "effect": {
                    "op": "and",
                    "args": [
                        atom("at", "robot", "?to"),
                        {"op": "not", "arg": atom("at", "robot", "?from")},
                        atom("visited", "?to"),
                    ],
                },
            },
            {"name": "rest", "parameters": [], "effect": atom("free")},
        ],
    }


def test_problem_document() -> None:
    document = parse_problem(PROBLEM, "p.pddl", locations=False)
    assert document == {
        "version": SCHEMA_VERSION,
        "kind": "problem",
        "name": "test-1",
        "source": "p.pddl",
        "domain": "test",
        "requirements": [],
        "objects": [{"name": "kitchen"}],
        "init": [atom("at", "robot", "home"), atom("free")],
        "goal": {"op": "and", "args": [atom("visited", "kitchen"), atom("at", "robot", "home")]},
    }


def test_generated_documents_follow_the_schema() -> None:
    validate_document(parse_domain(DOMAIN, "d.pddl"))
    validate_document(parse_problem(PROBLEM, "p.pddl"))


def test_locations_point_to_the_pddl() -> None:
    document = parse_domain(DOMAIN)
    move = document["actions"][0]
    assert move["loc"] == {"line": 7, "column": 3}
    assert move["precondition"]["args"][0]["loc"] == {"line": 9, "column": 24}


def test_single_atom_precondition_and_empty_formulas() -> None:
    document = parse_domain(
        """(define (domain d) (:predicates (p))
             (:action a :parameters () :precondition (p) :effect ())
             (:action b :precondition (and) :effect (and)))""",
        locations=False,
    )
    a, b = document["actions"]
    assert a == {"name": "a", "parameters": [], "precondition": atom("p")}
    assert b == {
        "name": "b",
        "parameters": [],
        "precondition": {"op": "and", "args": []},
        "effect": {"op": "and", "args": []},
    }


def test_parse_pddl_detects_the_kind() -> None:
    assert parse_pddl(DOMAIN)["kind"] == "domain"
    assert parse_pddl(PROBLEM)["kind"] == "problem"


def test_parse_domain_rejects_a_problem() -> None:
    with pytest.raises(PDDLSyntaxError, match="se esperaba un dominio"):
        parse_domain(PROBLEM)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "vacío"),
        ("(define (domain d)) (define (domain e))", "una sola expresión"),
        ("(domain d)", r"se esperaba \(define"),
        ("(define (thing d))", "'domain' o 'problem'"),
        ("(define (domain d) (:predicates (p)) (:predicates (q)))", "repetida"),
        ("(define (domain d) (:foo))", "sección desconocida"),
        ("(define (domain d) (:requirements :magic))", "requisito desconocido"),
        ("(define (domain d) (:predicates (p x)))", "no es una variable"),
        ("(define (domain d) (:predicates (?p)))", "no es un nombre válido"),
        ("(define (domain d) (:action a :parameters))", "falta el valor"),
        ("(define (domain d) (:action a :cost 1))", "clave de acción desconocida"),
        ("(define (domain d) (:action a :effect (not (p) (q))))", "exactamente un átomo"),
        ("(define (problem p) (:goal (q)))", r"falta la sección \(:domain"),
        ("(define (problem p) (:domain d))", r"falta la sección \(:goal"),
        ("(define)", r"falta \(domain"),
        ("(define (domain))", r"se esperaba \(domain <nombre>\)"),
        ("(define (domain d) (:predicates ()))", "predicado vacía"),
        ("(define (domain d) (:action))", "falta el nombre de la acción"),
        ("(define (domain d) (:action a :effect (p) :effect (q)))", ":effect repetido"),
        ("(define (domain d) (:action a :precondition (and ())))", "fórmula vacía"),
        ("(define (domain d) (:action a :effect (and ())))", "efecto vacío"),
        ("(define (domain d) (:action a :effect ((p))))", "se esperaba un operador"),
        ("(define (domain d) (:action a :effect (p 3)))", "no es un objeto ni una variable"),
        ("(define (domain d) (:action a :effect (p (x))))", "no una lista"),
        ("(define (domain d) ())", "sección vacía"),
        ("(define (domain d) (predicates))", "se esperaba una sección"),
        ("(define (domain d) :predicates)", "se esperaba una sección"),
        ("(define (problem p) (:domain d e) (:goal (q)))", r"se esperaba \(:domain"),
        ("(define (problem p) (:domain d) (:goal (q) (r)))", r"se esperaba \(:goal"),
        ("(define (problem p) (:domain d) (:init ()) (:goal (q)))", "átomo vacío"),
    ],
)
def test_syntax_errors(text: str, message: str) -> None:
    with pytest.raises(PDDLSyntaxError, match=message):
        parse_pddl(text)


@pytest.mark.parametrize(
    ("text", "requirement"),
    [
        ("(define (domain d) (:requirements :fluents))", ":fluents"),
        ("(define (domain d) (:requirements :durative-actions))", ":durative-actions"),
        ("(define (domain d) (:functions (f)))", ":numeric-fluents"),
        ("(define (domain d) (:durative-action a))", ":durative-actions"),
        ("(define (domain d) (:derived (p) (q)))", ":derived-predicates"),
        ("(define (domain d) (:process p))", ":time"),
        ("(define (domain d) (:action a :precondition (< (f) 1)))", ":numeric-fluents"),
        ("(define (domain d) (:action a :effect (increase (f) 1)))", ":numeric-fluents"),
        ("(define (problem p) (:domain d) (:init (= (f) 1)) (:goal (q)))", ":numeric-fluents"),
        (
            "(define (problem p) (:domain d) (:init (at 10 (p))) (:goal (q)))",
            ":timed-initial-literals",
        ),
        ("(define (problem p) (:domain d) (:goal (q)) (:metric minimize (c)))", ":numeric"),
    ],
)
def test_future_features_are_unsupported(text: str, requirement: str) -> None:
    with pytest.raises(UnsupportedFeatureError, match=requirement):
        parse_pddl(text)


def test_at_is_an_ordinary_predicate_in_init() -> None:
    document = parse_problem(
        "(define (problem p) (:domain d) (:init (at ball room)) (:goal (at ball room)))",
        locations=False,
    )
    assert document["init"] == [atom("at", "ball", "room")]


def test_problem_requirements() -> None:
    document = parse_problem(
        "(define (problem p) (:domain d) (:requirements :strips) (:goal (q)))", locations=False
    )
    assert document["requirements"] == [":strips"]


def test_errors_carry_the_file_name() -> None:
    with pytest.raises(PDDLSyntaxError) as exc:
        parse_pddl("(define (domain d)\n  (:foo))", "mi.pddl")
    assert str(exc.value).startswith("mi.pddl:2:4: ")
