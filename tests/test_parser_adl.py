"""Tests de la fase 1 para PDDL 1.2 (subconjunto ADL): tipos, fórmulas y efectos."""

from typing import Any

import pytest

from pyplanner.errors import PDDLSyntaxError, UndeclaredRequirementError
from pyplanner.pddl import parse_domain, parse_pddl, parse_problem, write_pddl


def atom(predicate: str, *args: str) -> dict[str, Any]:
    return {"op": "atom", "predicate": predicate, "args": list(args)}


def domain(body: str, requirements: str = ":adl") -> dict[str, Any]:
    return parse_domain(
        f"(define (domain d) (:requirements {requirements}) {body})", locations=False
    )


def precondition(formula: str) -> dict[str, Any]:
    document = domain(f"(:predicates (p) (q) (r ?x)) (:action a :precondition {formula})")
    result: dict[str, Any] = document["actions"][0]["precondition"]
    return result


def effect(text: str) -> dict[str, Any]:
    document = domain(f"(:predicates (p) (q) (r ?x)) (:action a :effect {text})")
    result: dict[str, Any] = document["actions"][0]["effect"]
    return result


# --- Tipos --------------------------------------------------------------------------


def test_types_section() -> None:
    document = domain("(:types truck plane - vehicle vehicle place)", ":typing")
    assert document["types"] == [
        {"name": "truck", "parent": "vehicle"},
        {"name": "plane", "parent": "vehicle"},
        {"name": "vehicle"},
        {"name": "place"},
    ]


def test_typed_lists_everywhere() -> None:
    document = domain(
        "(:types t u) (:constants c1 c2 - t c3)"
        " (:predicates (at ?x - t ?y - (either t u)))"
        " (:action go :parameters (?a ?b - t ?c) :effect (at ?a ?b))",
        ":typing",
    )
    assert document["constants"] == [
        {"name": "c1", "type": "t"},
        {"name": "c2", "type": "t"},
        {"name": "c3"},
    ]
    assert document["predicates"][0]["parameters"] == [
        {"name": "?x", "type": "t"},
        {"name": "?y", "type": ["t", "u"]},
    ]
    assert document["actions"][0]["parameters"] == [
        {"name": "?a", "type": "t"},
        {"name": "?b", "type": "t"},
        {"name": "?c"},
    ]


def test_typed_objects_in_problem() -> None:
    document = parse_problem(
        "(define (problem p) (:domain d) (:objects a b - block t1 - table) (:goal (q)))",
        locations=False,
    )
    assert document["objects"] == [
        {"name": "a", "type": "block"},
        {"name": "b", "type": "block"},
        {"name": "t1", "type": "table"},
    ]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("(:constants - t)", "'-' sin elementos"),
        ("(:constants a -)", "falta el tipo"),
        ("(:types a - (either b c))", "supertipo tiene que ser un nombre"),
        ("(:constants a - (either))", r"\(either\) sin tipos"),
        ("(:constants a - (b c))", "se esperaba un tipo o"),
    ],
)
def test_typed_list_errors(body: str, message: str) -> None:
    with pytest.raises(PDDLSyntaxError, match=message):
        domain(body, ":typing")


# --- Fórmulas -----------------------------------------------------------------------


def test_formula_operators() -> None:
    assert precondition("(or (p) (not (q)))") == {
        "op": "or",
        "args": [atom("p"), {"op": "not", "arg": atom("q")}],
    }
    assert precondition("(imply (p) (q))") == {"op": "imply", "args": [atom("p"), atom("q")]}
    assert precondition("(exists (?x) (r ?x))") == {
        "op": "exists",
        "parameters": [{"name": "?x"}],
        "arg": atom("r", "?x"),
    }
    assert precondition("(forall (?x) (r ?x))")["op"] == "forall"
    assert precondition("(= ?x ?x)") == {"op": "=", "args": ["?x", "?x"]}
    assert precondition("(not (and (p) (q)))")["arg"]["op"] == "and"


@pytest.mark.parametrize(
    ("formula", "message"),
    [
        ("(not (p) (q))", r"\(not <fórmula>\)"),
        ("(imply (p))", r"\(imply <fórmula> <fórmula>\)"),
        ("(exists (?x))", r"\(exists \(<variables>\) <fórmula>\)"),
        ("(forall () (p))", "no declara variables"),
        ("(= ?x)", r"\(= <término> <término>\)"),
        ("(= ?x (p))", "no una lista"),
    ],
)
def test_formula_errors(formula: str, message: str) -> None:
    with pytest.raises(PDDLSyntaxError, match=message):
        precondition(formula)


# --- Efectos ------------------------------------------------------------------------


def test_effect_operators() -> None:
    assert effect("(forall (?x) (when (r ?x) (and (not (r ?x)) (p))))") == {
        "op": "forall",
        "parameters": [{"name": "?x"}],
        "arg": {
            "op": "when",
            "condition": atom("r", "?x"),
            "arg": {"op": "and", "args": [{"op": "not", "arg": atom("r", "?x")}, atom("p")]},
        },
    }


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("(when (p) (when (q) (p)))", "dentro de 'when' solo se admiten"),
        ("(when (p) (forall (?x) (r ?x)))", "dentro de 'when' solo se admiten"),
        ("(when (p))", r"\(when <fórmula> <efecto>\)"),
        ("(forall (?x))", r"\(forall \(<variables>\) <efecto>\)"),
        ("(not (or (p) (q)))", "solo puede negar un átomo"),
        ("(not (= ?x ?x))", "solo puede negar un átomo"),
    ],
)
def test_effect_errors(text: str, message: str) -> None:
    with pytest.raises(PDDLSyntaxError, match=message):
        effect(text)


# --- Modo estricto en la fase 1 -----------------------------------------------------


P = "(:predicates (p) (r ?x))"
XY = ":parameters (?x ?y)"


@pytest.mark.parametrize(
    ("requirements", "body", "missing"),
    [
        (":strips", "(:types t)", ":typing"),
        (":strips", "(:constants c - object)", ":typing"),
        (":strips", f"{P} (:action a :precondition (not (p)))", ":negative-preconditions"),
        (":strips", f"{P} (:action a :precondition (or (p)))", ":disjunctive-preconditions"),
        (
            ":negative-preconditions",
            f"{P} (:action a :precondition (not (and (p))))",
            ":disjunctive-preconditions",
        ),
        (
            ":strips",
            f"{P} (:action a :precondition (exists (?x) (r ?x)))",
            ":existential-preconditions",
        ),
        (
            ":existential-preconditions",
            f"{P} (:action a :precondition (forall (?x) (r ?x)))",
            ":universal-preconditions",
        ),
        (
            ":negative-preconditions",
            f"(:action a {XY} :precondition (not (= ?x ?y)))",
            ":equality",
        ),
        (
            ":equality",
            f"(:action a {XY} :precondition (not (= ?x ?y)))",
            ":negative-preconditions",
        ),
        (":strips", f"{P} (:action a :effect (when (p) (p)))", ":conditional-effects"),
        (":strips", f"{P} (:action a :effect (forall (?x) (r ?x)))", ":conditional-effects"),
    ],
)
def test_undeclared_requirements_in_domains(requirements: str, body: str, missing: str) -> None:
    with pytest.raises(UndeclaredRequirementError, match=missing) as exc:
        domain(body, requirements)
    assert exc.value.path


def test_strict_mode_accepts_declared_and_meta_requirements() -> None:
    body = (
        "(:types t) (:predicates (p ?x - t))"
        " (:action a :parameters (?x - t)"
        "  :precondition (and (not (p ?x)) (or (p ?x) (exists (?y - t) (p ?y))))"
        "  :effect (forall (?z - t) (when (p ?z) (not (p ?z)))))"
    )
    domain(body, ":adl")
    domain(
        body,
        ":typing :negative-preconditions :disjunctive-preconditions"
        " :quantified-preconditions :conditional-effects",
    )


def test_delete_effects_do_not_need_requirements() -> None:
    domain("(:predicates (p)) (:action a :effect (not (p)))", ":strips")


def test_problems_are_not_checked_in_phase_1() -> None:
    # Los requisitos de un problema dependen de su dominio: se comprueban en la fase 2.
    parse_problem("(define (problem p) (:domain d) (:goal (or (q) (not (q)))))")


def test_undeclared_error_points_to_the_construct() -> None:
    text = "(define (domain d)\n (:predicates (p))\n (:action a :precondition\n   (not (p))))"
    with pytest.raises(UndeclaredRequirementError) as exc:
        parse_pddl(text, "d.pddl")
    assert str(exc.value.location) == "d.pddl:4:4"


# --- Ida y vuelta -------------------------------------------------------------------


def test_round_trip_adl_domain() -> None:
    document = domain(
        "(:types t u - t v) (:constants k - (either t v))"
        " (:predicates (p ?x - t) (q))"
        " (:action a :parameters (?x - t ?y)"
        "  :precondition (and (not (= ?x ?y)) (imply (q) (exists (?z - u) (p ?z))))"
        "  :effect (and (forall (?z - t) (when (or (p ?z) (q)) (and (not (p ?z)) (q))))))"
    )
    assert parse_pddl(write_pddl(document), locations=False) == document  # tiene ambos


def test_untyped_item_before_typed_ones_is_written_as_object() -> None:
    document = {
        "version": 2,
        "kind": "domain",
        "name": "d",
        "requirements": [":typing"],
        "constants": [{"name": "a"}, {"name": "b", "type": "t"}],
    }
    assert "(:constants a - object b - t)" in write_pddl(document)
