"""Tests del parser (fases 1 y 2) para PDDL 2.2: predicados derivados y literales iniciales
temporizados."""

from typing import Any

import pytest

from pyplanner.errors import (
    PDDLSyntaxError,
    SchemaValidationError,
    SemanticError,
    UndeclaredRequirementError,
)
from pyplanner.frontend import read_domain, read_problem
from pyplanner.loader import load_domain, load_problem
from pyplanner.loader.ast import (
    Atom,
    Constant,
    DerivedRule,
    Exists,
    Not,
    Parameter,
    TimedLiteral,
    Variable,
)
from pyplanner.pddl import parse_domain, parse_pddl, parse_problem, write_pddl
from pyplanner.schema import validate_document
from tests.catalog import collection_dir

DERIVED = ":typing :derived-predicates :negative-preconditions :existential-preconditions"
BASE = "(:types t) (:predicates (p ?x - t) (q ?x - t) (r ?x ?y - t) (s))"


def domain(body: str, requirements: str = DERIVED) -> dict[str, Any]:
    text = f"(define (domain d) (:requirements {requirements}) {BASE} {body})"
    return parse_domain(text, locations=False)


def atom(predicate: str, *args: str) -> dict[str, Any]:
    return {"op": "atom", "predicate": predicate, "args": list(args)}


# --- Predicados derivados: fase 1 -----------------------------------------------------


def test_derived_rule_document() -> None:
    document = domain("(:derived (q ?x - t) (and (p ?x) (not (s))))")
    assert document["derived"] == [
        {
            "predicate": "q",
            "parameters": [{"name": "?x", "type": "t"}],
            "condition": {"op": "and", "args": [atom("p", "?x"), {"op": "not", "arg": atom("s")}]},
        }
    ]


def test_a_predicate_can_have_several_rules() -> None:
    document = domain("(:derived (q ?x - t) (p ?x)) (:derived (q ?x - t) (s))")
    assert [rule["predicate"] for rule in document["derived"]] == ["q", "q"]


@pytest.mark.parametrize(
    ("rule", "message"),
    [
        ("(:derived (q ?x))", r"\(:derived \(<predicado> <variables>\) <fórmula>\)"),
        ("(:derived q (p ?x))", "la cabeza"),
        ("(:derived () (s))", "cabeza de predicado derivado vacía"),
        ("(:derived (q a) (s))", "no es una variable válida"),
    ],
)
def test_derived_syntax_errors(rule: str, message: str) -> None:
    with pytest.raises(PDDLSyntaxError, match=message):
        domain(rule)


def test_derived_rules_need_their_requirement() -> None:
    with pytest.raises(UndeclaredRequirementError, match=":derived-predicates"):
        domain("(:derived (q ?x - t) (p ?x))", ":typing")


def test_the_condition_of_a_rule_needs_its_own_requirements() -> None:
    with pytest.raises(UndeclaredRequirementError, match=":disjunctive-preconditions"):
        domain("(:derived (q ?x - t) (or (p ?x) (s)))")


# --- Predicados derivados: fase 2 -----------------------------------------------------


def test_derived_rule_ast() -> None:
    loaded = load_domain(domain("(:derived (q ?x - t) (exists (?y - t) (r ?x ?y)))"))
    x, y = Variable("?x"), Variable("?y")
    assert loaded.derived == (
        DerivedRule(
            "q",
            (Parameter("?x", ("t",)),),
            Exists((Parameter("?y", ("t",)),), Atom("r", (x, y))),
        ),
    )
    assert loaded.derived_predicates == {"q"}


def test_derived_predicates_can_be_used_in_conditions_and_goals() -> None:
    loaded = load_domain(
        domain("(:derived (s) (exists (?x - t) (p ?x))) (:action a :precondition (not (s)))")
    )
    assert loaded.actions[0].precondition == Not(Atom("s"))
    problem = parse_problem(
        "(define (problem x) (:domain d) (:objects o - t) (:init (p o)) (:goal (s)))"
    )
    assert load_problem(problem, loaded).goal == Atom("s")


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("(:derived (z ?x - t) (p ?x))", "predicado derivado 'z' no declarado"),
        ("(:derived (q) (s))", "'q' espera 1 argumentos y la regla declara 0"),
        ("(:derived (q ?x) (p ?x))", r"'\?x' \(object\) no es del tipo t"),
        ("(:derived (q ?x - t) (p ?y))", "no está ligada"),
        ("(:derived (s) (s)) (:action a :effect (s))", "'s' no puede aparecer en un efecto"),
        ("(:derived (s) (s)) (:action a :effect (not (s)))", "'s' no puede aparecer en un"),
    ],
)
def test_derived_semantic_errors(body: str, message: str) -> None:
    with pytest.raises(SemanticError, match=message):
        load_domain(domain(body))


def test_derived_predicates_are_not_in_durative_effects() -> None:
    document = domain(
        "(:derived (s) (exists (?x - t) (p ?x)))"
        " (:durative-action a :duration (= ?duration 1) :condition () :effect (at end (s)))",
        f"{DERIVED} :durative-actions",
    )
    with pytest.raises(SemanticError, match="no puede aparecer en un efecto"):
        load_domain(document)


@pytest.mark.parametrize(
    "rules",
    [
        # Recursión positiva (cierre transitivo): estratificable.
        "(:derived (q ?x - t) (or (p ?x) (exists (?y - t) (and (r ?y ?x) (q ?y)))))",
        # Negación de un predicado de un estrato inferior.
        "(:derived (q ?x - t) (p ?x)) (:derived (s) (exists (?x - t) (not (q ?x))))",
        # Doble negación: es una dependencia positiva.
        "(:derived (s) (not (not (s))))",
        # El consecuente de imply es positivo.
        "(:derived (s) (imply (exists (?x - t) (p ?x)) (s)))",
    ],
)
def test_stratifiable_rules(rules: str) -> None:
    load_domain(domain(rules, f"{DERIVED} :disjunctive-preconditions"))


@pytest.mark.parametrize(
    ("rules", "message"),
    [
        ("(:derived (s) (not (s)))", "'s' depende negativamente de 's' \\["),
        (
            "(:derived (q ?x - t) (not (s))) (:derived (s) (exists (?x - t) (q ?x)))",
            "'q' depende negativamente de 's', que a su vez depende de 'q'",
        ),
        # El antecedente de imply es negativo.
        ("(:derived (s) (imply (s) (exists (?x - t) (p ?x))))", "'s' depende negativamente"),
        # forall (?x) (not ...) también es una negación.
        ("(:derived (s) (forall (?x - t) (not (s))))", "'s' depende negativamente"),
    ],
)
def test_unstratifiable_rules(rules: str, message: str) -> None:
    requirements = f"{DERIVED} :disjunctive-preconditions :universal-preconditions"
    with pytest.raises(SemanticError, match=f"no son estratificables: {message}"):
        load_domain(domain(rules, requirements))


def test_derived_predicates_are_not_in_the_initial_state() -> None:
    loaded = load_domain(domain("(:derived (s) (exists (?x - t) (p ?x)))"))
    problem = parse_problem("(define (problem x) (:domain d) (:init (s)) (:goal (s)))")
    with pytest.raises(SemanticError, match="'s' no puede aparecer en el estado inicial"):
        load_problem(problem, loaded)


# --- Literales iniciales temporizados -------------------------------------------------

TIL_DOMAIN = (
    "(define (domain d) (:requirements :typing :durative-actions :timed-initial-literals)"
    " (:types t) (:predicates (p ?x - t) (at ?x ?y - t))"
    " (:durative-action a :parameters (?x - t) :duration (= ?duration 1)"
    "  :condition (at start (p ?x)) :effect (at end (not (p ?x)))))"
)


def problem(init: str, requirements: str = "") -> dict[str, Any]:
    declared = f"(:requirements {requirements})" if requirements else ""
    text = (
        f"(define (problem x) (:domain d) {declared} (:objects o1 o2 - t)"
        f" (:init {init}) (:goal (p o1)))"
    )
    return parse_problem(text, locations=False)


def test_timed_literal_document() -> None:
    document = problem("(p o1) (at 10 (p o2)) (at 20.5 (not (p o1)))")
    assert document["init"] == [
        atom("p", "o1"),
        {"op": "at", "time": 10.0, "literal": atom("p", "o2")},
        {"op": "at", "time": 20.5, "literal": {"op": "not", "arg": atom("p", "o1")}},
    ]


def test_the_at_predicate_is_not_a_timed_literal() -> None:
    assert problem("(at o1 o2)")["init"] == [atom("at", "o1", "o2")]


@pytest.mark.parametrize(
    ("init", "message"),
    [
        ("(at -1 (p o1))", "antes del instante 0"),
        ("(at 5 (and (p o1)))", r"un átomo o \(not <átomo>\)"),
        ("(at 5 (not (not (p o1))))", "solo puede negar un átomo"),
        ("(at 5 (not))", r"\(not <átomo>\)"),
        ("(at 5 p)", "un literal"),
    ],
)
def test_timed_literal_syntax_errors(init: str, message: str) -> None:
    with pytest.raises(PDDLSyntaxError, match=message):
        problem(init)


def test_timed_literal_ast() -> None:
    loaded = load_domain(parse_domain(TIL_DOMAIN))
    result = load_problem(problem("(at 20 (not (p o1))) (p o1) (at 10 (p o2))"), loaded)
    assert result.init == {Atom("p", (Constant("o1"),))}
    assert result.timed_literals == (
        TimedLiteral(10.0, Atom("p", (Constant("o2"),))),
        TimedLiteral(20.0, Atom("p", (Constant("o1"),)), positive=False),
    )


def test_timed_literals_need_their_requirement() -> None:
    text = TIL_DOMAIN.replace(" :timed-initial-literals", "")
    loaded = load_domain(parse_domain(text))
    with pytest.raises(UndeclaredRequirementError, match=":timed-initial-literals"):
        load_problem(problem("(at 10 (p o1))"), loaded)
    # El requisito también se puede declarar en el problema.
    load_problem(problem("(at 10 (p o1))", ":timed-initial-literals"), loaded)


@pytest.mark.parametrize(
    ("init", "message"),
    [
        ("(at 10 (p o3))", "'o3' no declarado"),
        ("(at 10 (p ?x))", "no está ligada"),
        ("(at 10 (p o1)) (at 10 (not (p o1)))", r"se añade y se borra en el mismo instante \(10\)"),
    ],
)
def test_timed_literal_semantic_errors(init: str, message: str) -> None:
    loaded = load_domain(parse_domain(TIL_DOMAIN))
    with pytest.raises(SemanticError, match=message):
        load_problem(problem(init), loaded)


def test_repeated_timed_literals_are_merged() -> None:
    loaded = load_domain(parse_domain(TIL_DOMAIN))
    result = load_problem(problem("(at 10 (p o1)) (at 10 (p o1))"), loaded)
    assert len(result.timed_literals) == 1


def test_timed_literals_need_a_temporal_domain() -> None:
    text = (
        "(define (domain d) (:requirements :timed-initial-literals) (:predicates (p))"
        " (:action a :effect (p)))"
    )
    loaded = load_domain(parse_domain(text))
    document = parse_problem("(define (problem x) (:domain d) (:init (at 1 (p))) (:goal (p)))")
    with pytest.raises(SemanticError, match="solo tienen sentido en un dominio temporal"):
        load_problem(document, loaded)


def test_timed_literals_on_derived_predicates() -> None:
    text = (
        "(define (domain d) (:requirements :durative-actions :timed-initial-literals"
        " :derived-predicates) (:predicates (p) (q)) (:derived (q) (p))"
        " (:durative-action a :duration (= ?duration 1) :condition () :effect (at end (p))))"
    )
    loaded = load_domain(parse_domain(text))
    document = parse_problem("(define (problem x) (:domain d) (:init (at 1 (q))) (:goal (q)))")
    with pytest.raises(SemanticError, match="no puede aparecer en un literal inicial"):
        load_problem(document, loaded)


def test_schema_rejects_negative_times_and_compound_literals() -> None:
    document = problem("(at 10 (p o1))")
    document["init"][0]["time"] = -1
    with pytest.raises(SchemaValidationError):
        validate_document(document)
    document["init"][0] = {"op": "at", "time": 1, "literal": {"op": "and", "args": []}}
    with pytest.raises(SchemaValidationError):
        validate_document(document)


# --- Documentos completos y colecciones de PDDL 2.2 -----------------------------------

FULL_DOMAIN = f"""
(define (domain d) (:requirements {DERIVED} :durative-actions :timed-initial-literals)
  {BASE}
  (:derived (q ?x - t) (exists (?y - t) (and (r ?x ?y) (not (p ?y)))))
  (:derived (s) (exists (?x - t) (q ?x)))
  (:durative-action a :parameters (?x - t) :duration (= ?duration 1)
    :condition (at start (q ?x)) :effect (at end (p ?x))))
"""
FULL_PROBLEM = """
(define (problem x) (:domain d) (:objects o1 o2 - t)
  (:init (r o1 o2) (at 3.5 (p o1)) (at 7 (not (r o1 o2))))
  (:goal (s)))
"""


def test_round_trip_and_schema() -> None:
    for text in (FULL_DOMAIN, FULL_PROBLEM):
        document = parse_pddl(text, locations=False)
        validate_document(document)
        assert parse_pddl(write_pddl(document), locations=False) == document
    loaded = load_domain(parse_domain(FULL_DOMAIN))
    result = load_problem(parse_problem(FULL_PROBLEM), loaded)
    assert len(loaded.derived) == 2 and len(result.timed_literals) == 2
    assert loaded.derived[1].condition == Exists(
        (Parameter("?x", ("t",)),), Atom("q", (Variable("?x"),))
    )
    assert result.init == {Atom("r", (Constant("o1"), Constant("o2")))}


@pytest.mark.parametrize(
    "directory",
    ["ipc-2004/psr-middle-derived-predicates-adl", "ipc-2004/satellite-time-time-windows-strips"],
)
@pytest.mark.parametrize("name", ["p01", "p02"])
def test_pddl22_collections_load(directory: str, name: str) -> None:
    path = collection_dir(directory)
    loaded = read_domain(path / "domain.pddl")
    result = read_problem(path / f"{name}.pddl", loaded)
    assert loaded.derived or result.timed_literals
