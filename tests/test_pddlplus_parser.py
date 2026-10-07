"""Tests del parser (fases 1 y 2) para PDDL+: procesos, eventos y literales negados en el
estado inicial."""

from typing import Any

import pytest

from pyplanner.errors import (
    PDDLSyntaxError,
    SemanticError,
    UndeclaredRequirementError,
    UnsupportedFeatureError,
)
from pyplanner.frontend import read_domain, read_problem
from pyplanner.grounding import ground
from pyplanner.loader import load_domain, load_problem
from pyplanner.loader.ast import (
    And,
    Arithmetic,
    Atom,
    ContinuousEffect,
    DurativeAnd,
    DurativeForallEffect,
    Event,
    FluentTerm,
    Number,
    NumericEffect,
    Process,
    Variable,
)
from pyplanner.pddl import parse_domain, parse_pddl, parse_problem, write_pddl
from pyplanner.schema import validate_document
from tests.tiers import DOMINIOS_DIR

TIER6 = DOMINIOS_DIR / "tier6-pddlplus"

# Un depósito que se llena mientras el grifo está abierto y se desborda si pasa de 10.
TANK = """
(define (domain tank)
  (:requirements :typing :numeric-fluents :negative-preconditions :time)
  (:types tap)
  (:predicates (open ?t - tap) (overflowed))
  (:functions (level) (flow ?t - tap))
  (:action turn-on :parameters (?t - tap) :precondition (not (open ?t)) :effect (open ?t))
  (:action turn-off :parameters (?t - tap) :precondition (open ?t) :effect (not (open ?t)))
  (:process filling
    :parameters (?t - tap)
    :precondition (open ?t)
    :effect (increase (level) (* #t (flow ?t))))
  (:event overflow
    :parameters ()
    :precondition (and (> (level) 10) (not (overflowed)))
    :effect (and (overflowed) (assign (level) 10))))
"""
TANK_PROBLEM = """
(define (problem tank-1) (:domain tank)
  (:objects a b - tap)
  (:init (= (level) 0) (= (flow a) 1) (= (flow b) 2) (not (overflowed)))
  (:goal (>= (level) 5)))
"""


def document(text: str) -> dict[str, Any]:
    return parse_pddl(text, locations=False)


# --- Fase 1 --------------------------------------------------------------------------


def test_processes_and_events_in_documents() -> None:
    domain = document(TANK)
    [process] = domain["processes"]
    assert process["name"] == "filling"
    assert process["effect"] == {
        "op": "increase",
        "fluent": {"op": "fluent", "function": "level", "args": []},
        "value": {
            "op": "*",
            "args": [{"op": "#t"}, {"op": "fluent", "function": "flow", "args": ["?t"]}],
        },
    }
    [event] = domain["events"]
    assert event["name"] == "overflow" and event["effect"]["op"] == "and"
    problem = document(TANK_PROBLEM)
    assert problem["init"][-1] == {
        "op": "not",
        "arg": {"op": "atom", "predicate": "overflowed", "args": []},
    }
    for doc in (domain, problem):
        validate_document(doc)
        assert parse_pddl(write_pddl(doc), locations=False) == doc


@pytest.mark.parametrize("domain", ["car", "generator"])
def test_tier6_documents_round_trip(domain: str) -> None:
    for name in ("domain", "p01", "p02"):
        doc = parse_pddl((TIER6 / domain / f"{name}.pddl").read_text(), locations=False)
        validate_document(doc)
        assert parse_pddl(write_pddl(doc), locations=False) == doc


@pytest.mark.parametrize(
    ("effect", "message"),
    [
        ("(assign (level) 0)", "un proceso solo tiene efectos continuos"),
        ("(open ?t)", "un proceso solo tiene efectos continuos"),
        ("(when (open ?t) (increase (level) #t))", "un proceso solo tiene efectos continuos"),
    ],
)
def test_process_effects_are_continuous(effect: str, message: str) -> None:
    text = TANK.replace("(increase (level) (* #t (flow ?t)))", effect)
    with pytest.raises(PDDLSyntaxError, match=message):
        parse_domain(text)


def test_processes_and_events_need_time() -> None:
    with pytest.raises(UndeclaredRequirementError, match=":time"):
        parse_domain(TANK.replace(" :time)", ")"))


def test_process_effects_need_numeric_fluents() -> None:
    """Los efectos continuos de un proceso no piden :continuous-effects (es de las acciones
    durativas), pero sí :numeric-fluents, como cualquier efecto numérico."""
    text = TANK.replace(":numeric-fluents", "").replace("(:functions (level) (flow ?t - tap))", "")
    text = text.replace("(* #t (flow ?t))", "#t").replace(
        "(and (> (level) 10) (not (overflowed)))", "(not (overflowed))"
    )
    text = text.replace("(and (overflowed) (assign (level) 10))", "(overflowed)")
    with pytest.raises(UndeclaredRequirementError, match=":numeric-fluents"):
        parse_domain(text)


def test_preferences_are_not_allowed_in_processes() -> None:
    text = TANK.replace(":precondition (open ?t)\n", ":precondition (preference p (open ?t))\n")
    with pytest.raises(PDDLSyntaxError):
        parse_domain(text)


# --- Fase 2 --------------------------------------------------------------------------


def test_processes_and_events_in_the_ast() -> None:
    domain = load_domain(document(TANK))
    t = Variable("?t")
    [process] = domain.processes
    assert process == Process(
        "filling",
        process.parameters,
        Atom("open", (t,)),
        ContinuousEffect("increase", FluentTerm("level"), FluentTerm("flow", (t,))),
    )
    [event] = domain.events
    assert isinstance(event, Event) and event.name == "overflow"
    assert event.effect == And(
        (Atom("overflowed"), NumericEffect("assign", FluentTerm("level"), Number(10.0)))
    )


def test_tier6_processes() -> None:
    car = read_domain(TIER6 / "car" / "domain.pddl")
    [moving] = car.processes
    assert isinstance(moving.effect, DurativeAnd) and len(moving.effect.args) == 3
    assert moving.effect.args[1] == ContinuousEffect("increase", FluentTerm("d"), FluentTerm("v"))
    assert [e.name for e in car.events] == ["engineexplode"]
    generator = read_domain(TIER6 / "generator" / "domain.pddl")
    [refuelling] = generator.processes
    assert isinstance(refuelling.effect, DurativeAnd)
    rate = refuelling.effect.args[0].rate  # type: ignore[union-attr]
    assert isinstance(rate, Arithmetic) and rate.op == "*"
    assert [e.name for e in generator.events] == ["tankempty", "generatoroverflow"]


def test_quantified_process_effects() -> None:
    text = TANK.replace(
        "(increase (level) (* #t (flow ?t)))",
        "(forall (?u - tap) (increase (flow ?u) (* #t 0.5)))",
    ).replace(":time)", ":time :conditional-effects)")
    [process] = load_domain(document(text)).processes
    assert isinstance(process.effect, DurativeForallEffect)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        # Sin #t no es un efecto continuo.
        ("(* #t (flow ?t))", "(flow ?t)", "un efecto continuo tiene la forma"),
        # Un evento es instantáneo: #t no tiene sentido.
        ("(assign (level) 10)", "(increase (level) (* #t 1))", "'#t' solo puede aparecer"),
        # ?duration es de las acciones durativas.
        ("(* #t (flow ?t))", "(* #t ?duration)", "duration"),
        # Un nombre se declara una sola vez.
        ("(:event overflow", "(:event turn-on", "ya se ha definido previamente como acción"),
        ("(:process filling", "(:process overflow", "ya se ha definido previamente"),
    ],
)
def test_semantic_errors(old: str, new: str, message: str) -> None:
    text = TANK.replace(old, new)
    with pytest.raises(SemanticError, match=message):
        load_domain(document(text))


def test_negated_initial_literals_are_redundant() -> None:
    domain = load_domain(document(TANK))
    problem = load_problem(document(TANK_PROBLEM), domain)
    assert problem.init == frozenset()  # el (not (overflowed)) no añade nada
    contradiction = TANK_PROBLEM.replace("(not (overflowed))", "(overflowed) (not (overflowed))")
    with pytest.raises(SemanticError, match="cierto y falso a la vez"):
        load_problem(document(contradiction), domain)


def test_processes_from_json() -> None:
    domain = load_domain(parse_domain(TANK, locations=False))
    assert domain.processes and domain.events
    problem = load_problem(parse_problem(TANK_PROBLEM, locations=False), domain)
    assert problem.domain_name == "tank"


def test_tier6_problems_load() -> None:
    for name in ("car", "generator"):
        domain = read_domain(TIER6 / name / "domain.pddl")
        for problem in ("p01", "p02"):
            assert read_problem(TIER6 / name / f"{problem}.pddl", domain).values


def test_the_engine_rejects_processes_for_now() -> None:
    domain = load_domain(document(TANK))
    problem = load_problem(document(TANK_PROBLEM), domain)
    with pytest.raises(UnsupportedFeatureError, match="procesos ni eventos"):
        ground(domain, problem)
