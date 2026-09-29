"""Tests del parser (fases 1 y 2) para PDDL 2.1 numérico y costes de acción."""

from typing import Any

import pytest

from pyplanner.errors import (
    PDDLSyntaxError,
    SemanticError,
    UndeclaredRequirementError,
    UnsupportedFeatureError,
)
from pyplanner.loader import load_domain, load_problem
from pyplanner.loader.ast import (
    And,
    Arithmetic,
    Comparison,
    Constant,
    FluentTerm,
    Metric,
    Number,
    NumericEffect,
    TotalTime,
    Variable,
    When,
)
from pyplanner.pddl import parse_domain, parse_pddl, parse_problem, write_pddl

NUMERIC = ":typing :conditional-effects :numeric-fluents"


def fluent(name: str, *args: str) -> dict[str, Any]:
    return {"op": "fluent", "function": name, "args": list(args)}


def num(value: float) -> dict[str, Any]:
    return {"op": "num", "value": value}


def domain(body: str, requirements: str = NUMERIC) -> dict[str, Any]:
    text = f"(define (domain d) (:requirements {requirements}) {body})"
    return parse_domain(text, locations=False)


BASE = "(:types t) (:predicates (p ?x - t)) (:functions (f ?x - t) (g) - number)"


def precondition(formula: str) -> dict[str, Any]:
    document = domain(f"{BASE} (:action a :parameters (?x - t) :precondition {formula})")
    result: dict[str, Any] = document["actions"][0]["precondition"]
    return result


def effect(text: str) -> dict[str, Any]:
    document = domain(f"{BASE} (:action a :parameters (?x - t) :effect {text})")
    result: dict[str, Any] = document["actions"][0]["effect"]
    return result


# --- Fase 1 -------------------------------------------------------------------------


def test_functions_section() -> None:
    document = domain(BASE)
    assert document["functions"] == [
        {"name": "f", "parameters": [{"name": "?x", "type": "t"}]},
        {"name": "g", "parameters": []},
    ]


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("(g)", fluent("g")),
        ("(f ?x)", fluent("f", "?x")),
        ("3.5", num(3.5)),
        ("-2", num(-2.0)),
        ("(+ (g) 1 2)", {"op": "+", "args": [fluent("g"), num(1), num(2)]}),
        ("(- (g))", {"op": "-", "args": [fluent("g")]}),
        ("(- (g) 1)", {"op": "-", "args": [fluent("g"), num(1)]}),
        ("(* (g) (f ?x))", {"op": "*", "args": [fluent("g"), fluent("f", "?x")]}),
        ("(/ (g) 2)", {"op": "/", "args": [fluent("g"), num(2)]}),
    ],
)
def test_expressions(expression: str, expected: dict[str, Any]) -> None:
    assert precondition(f"(> {expression} 0)") == {
        "op": "compare",
        "comparator": ">",
        "left": expected,
        "right": num(0),
    }


@pytest.mark.parametrize("comparator", ["<", "<=", "=", ">=", ">"])
def test_comparators(comparator: str) -> None:
    node = precondition(f"({comparator} (g) (f ?x))")
    assert node["op"] == "compare" and node["comparator"] == comparator


def test_numeric_and_object_equality_are_distinguished() -> None:
    assert precondition("(= (g) 3)")["op"] == "compare"
    assert domain(
        f"{BASE} (:action a :parameters (?x ?y - t) :precondition (= ?x ?y))",
        ":adl :numeric-fluents",
    )["actions"][0]["precondition"] == {"op": "=", "args": ["?x", "?y"]}


@pytest.mark.parametrize("op", ["increase", "decrease", "assign", "scale-up", "scale-down"])
def test_numeric_effects(op: str) -> None:
    assert effect(f"({op} (f ?x) (* 2 (g)))") == {
        "op": op,
        "fluent": fluent("f", "?x"),
        "value": {"op": "*", "args": [num(2), fluent("g")]},
    }


def test_numeric_effects_inside_when() -> None:
    node = effect("(when (p ?x) (and (not (p ?x)) (increase (g) 1)))")
    assert node["arg"]["args"][1]["op"] == "increase"


def test_initial_values_and_metric() -> None:
    document = parse_problem(
        "(define (problem x) (:domain d) (:objects o - t)"
        " (:init (p o) (= (f o) 2.5) (= (g) -1)) (:goal (p o))"
        " (:metric minimize (+ (total-time) (g))))",
        locations=False,
    )
    assert document["init"] == [
        {"op": "atom", "predicate": "p", "args": ["o"]},
        {"op": "=", "fluent": fluent("f", "o"), "value": 2.5},
        {"op": "=", "fluent": fluent("g"), "value": -1.0},
    ]
    assert document["metric"] == {
        "direction": "minimize",
        "expression": {"op": "+", "args": [{"op": "total-time"}, fluent("g")]},
    }


def test_total_time_without_parentheses() -> None:
    document = parse_problem(
        "(define (problem x) (:domain d) (:goal (q)) (:metric maximize total-time))",
        locations=False,
    )
    assert document["metric"]["expression"] == {"op": "total-time"}


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (f"{BASE} (:action a :precondition (> (+ (g)) 1))", "al menos dos términos"),
        (f"{BASE} (:action a :precondition (> (/ (g)) 1))", r"\(/ <expresión> <expresión>\)"),
        (f"{BASE} (:action a :precondition (> () 1))", "expresión vacía"),
        (f"{BASE} (:action a :precondition (> (g) uno))", "'uno' no es un número"),
        (f"{BASE} (:action a :precondition (> (g)))", r"\(> <expresión> <expresión>\)"),
        (f"{BASE} (:action a :effect (increase (g)))", r"\(increase \(<función>"),
        ("(:functions - number)", "'-' mal colocado"),
        (f"{BASE} (:action a :effect (increase () 1))", "función vacía"),
        ("(:functions (g) (g))", None),
        ("(:functions ())", "declaración de función vacía"),
    ],
)
def test_syntax_errors(text: str, message: str | None) -> None:
    if message is None:  # válido en la fase 1: el error es semántico (fase 2)
        with pytest.raises(SemanticError, match="función 'g' declarada dos veces"):
            load_domain(domain(text))
        return
    with pytest.raises(PDDLSyntaxError, match=message):
        domain(text)


def test_bad_metric_direction() -> None:
    with pytest.raises(PDDLSyntaxError, match="minimize o maximize"):
        parse_problem("(define (problem x) (:domain d) (:goal (q)) (:metric reduce (g)))")


@pytest.mark.parametrize(
    ("text", "requirement"),
    [
        ("(:functions (f) - object)", ":object-fluents"),
        (f"{BASE} (:action a :precondition (> ?duration 1))", ":durative-actions"),
        (f"{BASE} (:action a :effect (increase (g) #t))", ":continuous-effects"),
        (f"{BASE} (:action a :effect (increase (g) (#t)))", ":continuous-effects"),
        (f"{BASE} (:action a :precondition (preference p (g)))", ":preferences"),
    ],
)
def test_future_numeric_features(text: str, requirement: str) -> None:
    with pytest.raises(UnsupportedFeatureError, match=requirement):
        domain(text)


def test_round_trip() -> None:
    document = domain(
        f"{BASE} (:action a :parameters (?x - t)"
        " :precondition (and (p ?x) (<= (+ (f ?x) 1) (g)))"
        " :effect (and (increase (f ?x) 1) (assign (g) (- (g) 0.5))))"
    )
    assert parse_pddl(write_pddl(document), locations=False) == document
    problem = parse_problem(
        "(define (problem x) (:domain d) (:objects o - t) (:init (= (f o) 2.5))"
        " (:goal (>= (f o) 3)) (:metric minimize (* 2 (total-time))))",
        locations=False,
    )
    assert parse_pddl(write_pddl(problem), locations=False) == problem


# --- Modo estricto y :action-costs --------------------------------------------------

COSTS_BODY = (
    "(:predicates (p)) (:functions (total-cost) (road ?x))"
    " (:action a :parameters (?x) :precondition (p)"
    "  :effect (and (not (p)) (increase (total-cost) (road ?x))))"
)


def test_action_costs_profile_is_accepted() -> None:
    domain(COSTS_BODY, ":action-costs")


@pytest.mark.parametrize(
    ("body", "construct"),
    [
        ("(:functions (g)) (:action a :effect (decrease (g) 1))", "'decrease' en un efecto"),
        ("(:functions (g)) (:action a :precondition (> (g) 1))", "una comparación numérica"),
        (
            "(:functions (total-cost)) (:action a :effect (increase (total-cost) (* 2 3)))",
            "'increase' en un efecto",
        ),
    ],
)
def test_outside_the_action_costs_profile(body: str, construct: str) -> None:
    with pytest.raises(UndeclaredRequirementError, match=f"{construct}.*:numeric-fluents"):
        domain(body, ":action-costs")


def test_numeric_constructs_need_numeric_fluents() -> None:
    with pytest.raises(UndeclaredRequirementError, match="la sección :functions"):
        domain("(:functions (g))", ":strips")


# --- Fase 2 -------------------------------------------------------------------------

DOMAIN_TEXT = (
    "(define (domain d) (:requirements :typing :conditional-effects :numeric-fluents)"
    " (:types t u) (:predicates (p ?x - t)) (:functions (f ?x - t) (g) - number)"
    " (:action a :parameters (?x - t)"
    "  :precondition (and (p ?x) (<= (+ (f ?x) 1) (g)))"
    "  :effect (and (increase (f ?x) 1) (when (p ?x) (assign (g) (* 2 (g)))))))"
)


def test_numeric_ast() -> None:
    loaded = load_domain(parse_domain(DOMAIN_TEXT))
    assert [(f.name, f.arity) for f in loaded.functions] == [("f", 1), ("g", 0)]
    action = loaded.action("a")
    assert action is not None
    fx = FluentTerm("f", (Variable("?x"),))
    g = FluentTerm("g")
    assert isinstance(action.precondition, And) and isinstance(action.effect, And)
    assert action.precondition.args[1] == Comparison("<=", Arithmetic("+", (fx, Number(1.0))), g)
    increase, conditional = action.effect.args
    assert increase == NumericEffect("increase", fx, Number(1.0))
    assert isinstance(conditional, When)
    assert conditional.arg == NumericEffect("assign", g, Arithmetic("*", (Number(2.0), g)))


def problem(body: str) -> dict[str, Any]:
    return parse_problem(f"(define (problem x) (:domain d) (:objects o - t v - u) {body})")


def test_initial_values_and_metric_ast() -> None:
    loaded = load_domain(parse_domain(DOMAIN_TEXT))
    task = load_problem(
        problem("(:init (= (f o) 2) (= (g) 5)) (:goal (p o)) (:metric minimize (total-time))"),
        loaded,
    )
    assert task.values == (
        (FluentTerm("f", (Constant("o"),)), 2.0),
        (FluentTerm("g"), 5.0),
    )
    assert task.metric == Metric("minimize", TotalTime())


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("(:init (= (h) 1)) (:goal (p o))", "función 'h' no declarada"),
        ("(:init (= (f) 1)) (:goal (p o))", "espera 1 argumentos y recibe 0"),
        ("(:init (= (f v) 1)) (:goal (p o))", "'v' \\(u\\) no es del tipo t"),
        ("(:init (= (g) 1) (= (g) 2)) (:goal (p o))", "tiene dos valores iniciales"),
        ("(:goal (> (total-time) 1))", "'total-time' solo puede aparecer en la métrica"),
        ("(:goal (p o)) (:metric minimize (f ?x))", "no está ligada"),
    ],
)
def test_semantic_errors(body: str, message: str) -> None:
    loaded = load_domain(parse_domain(DOMAIN_TEXT))
    with pytest.raises(SemanticError, match=message):
        load_problem(problem(body), loaded)


def test_numeric_problem_needs_the_requirement() -> None:
    strips = load_domain(parse_domain("(define (domain d) (:predicates (p)))"))
    with pytest.raises(UndeclaredRequirementError, match="la métrica"):
        load_problem(
            parse_problem("(define (problem x) (:domain d) (:goal (p)) (:metric minimize 1))"),
            strips,
        )
