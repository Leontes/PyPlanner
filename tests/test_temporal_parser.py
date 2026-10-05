"""Tests del parser (fases 1 y 2) para PDDL 2.1 temporal: acciones durativas."""

from typing import Any

import pytest

from pyplanner.errors import (
    PDDLSyntaxError,
    SchemaValidationError,
    SemanticError,
    UndeclaredRequirementError,
)
from pyplanner.loader import load_domain, load_problem
from pyplanner.loader.ast import (
    Arithmetic,
    Atom,
    ContinuousEffect,
    DurationConstraint,
    DurationVar,
    DurativeAnd,
    DurativeForall,
    DurativeWhen,
    FluentTerm,
    Not,
    Number,
    TimedCondition,
    TimedEffect,
    Variable,
)
from pyplanner.pddl import parse_domain, parse_pddl, parse_problem, write_pddl
from pyplanner.schema import validate_document

TEMPORAL = (
    ":typing :durative-actions :duration-inequalities :continuous-effects :numeric-fluents"
    " :conditional-effects :universal-preconditions :negative-preconditions"
)
BASE = "(:types t) (:predicates (p ?x - t) (q) (at ?x - t)) (:functions (f ?x - t) (g))"


def domain(body: str, requirements: str = TEMPORAL) -> dict[str, Any]:
    text = f"(define (domain d) (:requirements {requirements}) {body})"
    return parse_domain(text, locations=False)


def durative(
    duration: str = "(= ?duration 5)",
    condition: str = "()",
    effect: str = "()",
    requirements: str = TEMPORAL,
) -> dict[str, Any]:
    document = domain(
        f"{BASE} (:durative-action a :parameters (?x - t) :duration {duration}"
        f" :condition {condition} :effect {effect})",
        requirements,
    )
    result: dict[str, Any] = document["durative_actions"][0]
    return result


def atom(predicate: str, *args: str) -> dict[str, Any]:
    return {"op": "atom", "predicate": predicate, "args": list(args)}


# --- Fase 1 -------------------------------------------------------------------------


def test_durative_action_document() -> None:
    action = durative(
        condition="(and (at start (p ?x)) (over all (q)) (at end (not (q))))",
        effect="(and (at start (not (p ?x))) (at end (q)))",
    )
    assert action == {
        "name": "a",
        "parameters": [{"name": "?x", "type": "t"}],
        "duration": [{"comparator": "=", "value": {"op": "num", "value": 5.0}}],
        "condition": {
            "op": "and",
            "args": [
                {"op": "at-start", "arg": atom("p", "?x")},
                {"op": "over-all", "arg": atom("q")},
                {"op": "at-end", "arg": {"op": "not", "arg": atom("q")}},
            ],
        },
        "effect": {
            "op": "and",
            "args": [
                {"op": "at-start", "arg": {"op": "not", "arg": atom("p", "?x")}},
                {"op": "at-end", "arg": atom("q")},
            ],
        },
    }


def test_actions_and_durative_actions_coexist() -> None:
    document = domain(
        f"{BASE} (:action b :effect (q))"
        " (:durative-action a :duration (= ?duration 1) :effect (at end (q)))"
    )
    assert [a["name"] for a in document["actions"]] == ["b"]
    assert [a["name"] for a in document["durative_actions"]] == ["a"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("()", []),
        (
            "(<= ?duration (g))",
            [{"comparator": "<=", "value": {"op": "fluent", "function": "g", "args": []}}],
        ),
        (
            "(and (>= ?duration 1) (<= ?duration 3))",
            [
                {"comparator": ">=", "value": {"op": "num", "value": 1.0}},
                {"comparator": "<=", "value": {"op": "num", "value": 3.0}},
            ],
        ),
        (
            "(at end (<= ?duration 4))",
            [{"comparator": "<=", "value": {"op": "num", "value": 4.0}, "time": "end"}],
        ),
    ],
)
def test_duration_constraints(text: str, expected: list[dict[str, Any]]) -> None:
    assert durative(duration=text)["duration"] == expected


def test_timed_conditions_are_not_confused_with_the_at_predicate() -> None:
    action = durative(condition="(at start (at ?x))")
    assert action["condition"] == {"op": "at-start", "arg": atom("at", "?x")}


def test_forall_and_when_in_durative_actions() -> None:
    action = durative(
        condition="(forall (?y - t) (over all (p ?y)))",
        effect="(when (and (at start (q)) (at end (p ?x))) (at end (not (q))))",
    )
    assert action["condition"]["op"] == "forall"
    assert action["condition"]["arg"] == {"op": "over-all", "arg": atom("p", "?y")}
    when = action["effect"]
    assert when["op"] == "when"
    assert when["condition"]["args"][1] == {"op": "at-end", "arg": atom("p", "?x")}
    assert when["arg"] == {"op": "at-end", "arg": {"op": "not", "arg": atom("q")}}


def test_duration_and_elapsed_time_in_expressions() -> None:
    action = durative(
        condition="(at start (>= (g) ?duration))",
        effect="(and (at end (decrease (g) (* 2 ?duration))) (increase (f ?x) (* #t 3)))",
    )
    compare = action["condition"]["arg"]
    assert compare["right"] == {"op": "duration"}
    end, continuous = action["effect"]["args"]
    assert end["arg"]["value"]["args"][1] == {"op": "duration"}
    assert continuous == {
        "op": "increase",
        "fluent": {"op": "fluent", "function": "f", "args": ["?x"]},
        "value": {"op": "*", "args": [{"op": "#t"}, {"op": "num", "value": 3.0}]},
    }


@pytest.mark.parametrize(
    ("duration", "condition", "effect", "message"),
    [
        ("(< ?duration 5)", "()", "()", "comparador de duración '<'"),
        ("(= 5 ?duration)", "()", "()", r"compara \?duration"),
        ("(and)", "()", "()", r"\(and\) sin restricciones"),
        ("(at middle (= ?duration 1))", "()", "()", "start o end"),
        ("(= ?duration 1)", "(p ?x)", "()", "cada condición va dentro de"),
        ("(= ?duration 1)", "(at start (p ?x) (q))", "()", "cada condición va dentro de"),
        ("(= ?duration 1)", "()", "(q)", "cada efecto va dentro de"),
        ("(= ?duration 1)", "()", "(over all (q))", "cada efecto va dentro de"),
        ("(= ?duration 1)", "()", "(at end (forall (?y - t) (p ?y)))", r"dentro de \(at end"),
        (
            "(= ?duration 1)",
            "()",
            "(when (at start (q)) (when (at start (q)) (at end (q))))",
            "dentro de 'when'",
        ),
        ("(= ?duration 1)", "(at start (> (g) (?duration)))", "()", "va sin paréntesis"),
    ],
)
def test_syntax_errors(duration: str, condition: str, effect: str, message: str) -> None:
    with pytest.raises(PDDLSyntaxError, match=message):
        durative(duration, condition, effect)


def test_duration_is_required_and_unique() -> None:
    with pytest.raises(PDDLSyntaxError, match="falta :duration"):
        domain(f"{BASE} (:durative-action a :effect (at end (q)))")
    with pytest.raises(PDDLSyntaxError, match=":duration repetido"):
        domain(f"{BASE} (:durative-action a :duration () :duration ())")


# --- Modo estricto ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("requirements", "duration", "effect", "missing"),
    [
        (":typing", "(= ?duration 1)", "(at end (q))", ":durative-actions"),
        (":typing :durative-actions", "(<= ?duration 1)", "(at end (q))", ":duration-inequalities"),
        (
            ":typing :durative-actions",
            "(and (= ?duration 1) (= ?duration 1))",
            "(at end (q))",
            ":duration-inequalities",
        ),
        (
            ":typing :durative-actions",
            "(at end (= ?duration 1))",
            "(at end (q))",
            ":duration-inequalities",
        ),
        (
            ":typing :durative-actions :numeric-fluents",
            "(= ?duration 1)",
            "(increase (g) (* #t 1))",
            ":continuous-effects",
        ),
        (
            ":typing :durative-actions",
            "(= ?duration 1)",
            "(at end (increase (g) 1))",
            ":numeric-fluents",
        ),
        (
            ":typing :durative-actions",
            "(= ?duration 1)",
            "(forall (?y - t) (at end (p ?y)))",
            ":conditional-effects",
        ),
    ],
)
def test_undeclared_requirements(
    requirements: str, duration: str, effect: str, missing: str
) -> None:
    body = (
        "(:types t) (:predicates (p ?x - t) (q))"
        f" (:durative-action a :duration {duration} :effect {effect})"
    )
    if ":numeric-fluents" in requirements or "increase" in effect:
        body = body.replace("(q))", "(q)) (:functions (g))", 1)
    with pytest.raises(UndeclaredRequirementError, match=missing):
        domain(body, requirements)


def test_universal_conditions_need_their_requirement() -> None:
    with pytest.raises(UndeclaredRequirementError, match=":universal-preconditions"):
        domain(
            "(:types t) (:predicates (p ?x - t))"
            " (:durative-action a :duration (= ?duration 1)"
            " :condition (forall (?y - t) (at start (p ?y))))",
            ":typing :durative-actions",
        )


TEMPORAL_DOMAIN = """
(define (domain d) (:requirements :typing :durative-actions)
  (:types t) (:predicates (q))
  (:durative-action a :duration (= ?duration 1) :effect (at end (q))))
"""


def test_total_time_metric_needs_numeric_fluents() -> None:
    """Sin excepciones: la métrica necesita ``:numeric-fluents`` también en los dominios
    temporales (los benchmarks de las IPC 2002 que no lo declaraban se han corregido)."""
    loaded = load_domain(parse_domain(TEMPORAL_DOMAIN))
    problem = "(define (problem p) (:domain d) {} (:goal (q)) (:metric minimize (total-time)))"
    with pytest.raises(UndeclaredRequirementError, match=":numeric-fluents"):
        load_problem(parse_problem(problem.format("")), loaded)
    declared = parse_problem(problem.format("(:requirements :numeric-fluents)"))
    assert load_problem(declared, loaded).metric is not None


# --- Fase 2 -------------------------------------------------------------------------


def load(duration: str = "(= ?duration 5)", condition: str = "()", effect: str = "()") -> Any:
    document = domain(
        f"{BASE} (:durative-action a :parameters (?x - t) :duration {duration}"
        f" :condition {condition} :effect {effect})"
    )
    return load_domain(document).durative_actions[0]


def test_durative_action_ast() -> None:
    action = load(
        duration="(and (>= ?duration 1) (at end (<= ?duration (g))))",
        condition="(and (at start (p ?x)) (over all (q)))",
        effect="(and (at start (not (p ?x))) (at end (increase (g) ?duration)))",
    )
    x = Variable("?x")
    assert action.duration == (
        DurationConstraint(">=", Number(1.0)),
        DurationConstraint("<=", FluentTerm("g"), "end"),
    )
    assert action.condition == DurativeAnd(
        (TimedCondition("start", Atom("p", (x,))), TimedCondition("all", Atom("q")))
    )
    start, end = action.effect.args
    assert start == TimedEffect("start", Not(Atom("p", (x,))))
    assert end.time == "end" and end.arg.value == DurationVar()


def test_empty_condition_and_effect() -> None:
    action = load()
    assert action.condition == DurativeAnd() and action.effect == DurativeAnd()


@pytest.mark.parametrize(
    ("value", "rate"),
    [
        ("#t", Number(1.0)),
        ("(* #t 2)", Number(2.0)),
        ("(* (f ?x) #t)", FluentTerm("f", (Variable("?x"),))),
        ("(* 2 #t (g))", Arithmetic("*", (Number(2.0), FluentTerm("g")))),
    ],
)
def test_continuous_effects(value: str, rate: Any) -> None:
    effect = load(effect=f"(decrease (g) {value})").effect
    assert effect == ContinuousEffect("decrease", FluentTerm("g"), rate)


def test_conditional_effects_and_quantified_conditions() -> None:
    action = load(
        condition="(forall (?y - t) (over all (p ?y)))",
        effect="(when (at start (q)) (at end (not (q))))",
    )
    assert isinstance(action.condition, DurativeForall)
    assert action.effect == DurativeWhen(
        TimedCondition("start", Atom("q")), TimedEffect("end", Not(Atom("q")))
    )


@pytest.mark.parametrize(
    ("duration", "condition", "effect", "message"),
    [
        ("(= ?duration (* 2 ?duration))", "()", "()", "no puede depender de '\\?duration'"),
        ("(= ?duration 1)", "(at start (> (g) #t))", "()", "'#t' solo puede aparecer"),
        ("(= ?duration 1)", "()", "(increase (g) (+ #t 1))", "tiene la forma"),
        ("(= ?duration 1)", "()", "(increase (g) (* #t #t))", "tiene la forma"),
        ("(= ?duration 1)", "()", "(increase (g) 3)", "tiene la forma"),
        ("(= ?duration 1)", "()", "(at end (increase (g) (* #t 2)))", "'#t' solo puede aparecer"),
        (
            "(= ?duration 1)",
            "()",
            "(when (at end (q)) (at start (q)))",
            "'at start' solo puede depender",
        ),
        (
            "(= ?duration 1)",
            "()",
            "(when (over all (q)) (at start (q)))",
            "'at start' solo puede depender",
        ),
        ("(= ?duration 1)", "(at start (r))", "()", "predicado 'r' no declarado"),
        ("(= ?duration 1)", "(at start (p ?z))", "()", "no está ligada"),
    ],
)
def test_semantic_errors(duration: str, condition: str, effect: str, message: str) -> None:
    with pytest.raises(SemanticError, match=message):
        load(duration, condition, effect)


def test_duration_is_not_allowed_in_conditions() -> None:
    """Como en la gramática de PDDL 2.1 y en VAL: ?duration solo en la duración y en los
    efectos."""
    with pytest.raises(SemanticError, match="no puede aparecer en una condición"):
        load(condition="(at start (>= (g) ?duration))")
    with pytest.raises(SemanticError, match="no puede aparecer en una condición"):
        load(effect="(when (at end (> (g) ?duration)) (at end (q)))")


def test_duration_outside_durative_actions() -> None:
    document = domain(f"{BASE} (:action b :precondition (> (g) ?duration))")
    with pytest.raises(SemanticError, match="solo puede aparecer en una acción durativa"):
        load_domain(document)


def test_names_are_shared_with_actions() -> None:
    document = domain(
        f"{BASE} (:action a :effect (q)) (:durative-action a :duration () :effect (at end (q)))"
    )
    with pytest.raises(SemanticError, match="'a' ya se ha definido previamente como acción"):
        load_domain(document)


def test_schema_rejects_malformed_durative_actions() -> None:
    document = durative(effect="(at end (q))")
    document["effect"] = {
        "op": "at-start",
        "arg": {"op": "forall", "parameters": [], "arg": atom("q")},
    }
    full = domain(f"{BASE}")
    full["durative_actions"] = [document]
    with pytest.raises(SchemaValidationError):
        validate_document(full)


# --- Documentos completos -----------------------------------------------------------

FULL = f"""
(define (domain d) (:requirements {TEMPORAL})
  {BASE}
  (:durative-action a :parameters (?x - t)
    :duration (and (>= ?duration 1) (at end (<= ?duration (g))))
    :condition (and (at start (p ?x)) (over all (q)) (forall (?y - t) (at end (p ?y))))
    :effect (and (at start (not (p ?x))) (at end (assign (f ?x) ?duration))
                 (increase (g) (* #t 2))
                 (forall (?y - t) (at end (p ?y)))
                 (when (at start (q)) (at end (not (q))))))
  (:durative-action b :duration () :condition (and) :effect (and)))
"""


def test_round_trip_and_schema() -> None:
    document = parse_pddl(FULL, locations=False)
    validate_document(document)
    assert parse_pddl(write_pddl(document), locations=False) == document
    load_domain(document)


def test_missing_condition_and_effect_are_written_as_empty_conjunctions() -> None:
    """VAL no acepta ``:condition ()``, así que el writer escribe ``(and)``, que es
    equivalente."""
    document = parse_pddl(
        f"(define (domain d) (:requirements {TEMPORAL}) {BASE}"
        " (:durative-action b :duration () :condition () :effect ()))",
        locations=False,
    )
    text = write_pddl(document)
    assert ":condition (and)" in text and ":effect (and)" in text
    rewritten = load_domain(parse_pddl(text)).durative_actions[0]
    assert rewritten == load_domain(document).durative_actions[0]
