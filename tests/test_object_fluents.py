"""Funciones de objetos (``:object-fluents``, PDDL 3.1): parser, compilación a predicados y
búsqueda, con los planes validados por VAL contra el dominio compilado (VAL no las admite)."""

import subprocess
from pathlib import Path
from typing import Any

import pytest

from pyplanner.errors import SemanticError, UndeclaredRequirementError
from pyplanner.loader import load_domain, load_problem
from pyplanner.loader.ast import Atom, Constant, Exists, Parameter, Variable
from pyplanner.loader.object_fluents import compile_domain, compile_problem, object_functions
from pyplanner.pddl import parse_domain, parse_pddl, write_pddl
from pyplanner.plan import Plan, simulate
from pyplanner.schema import validate_document
from pyplanner.search import PRESETS, SearchStatus, search
from tests.tasks import build
from tests.tiers import VALIDATE

# Robots que se mueven entre sitios. (loc ?r) es una función de objetos; (holder ?b) también,
# y puede estar indefinida (nadie la lleva).
ROBOTS = """
(define (domain robots)
  (:requirements :typing :equality :negative-preconditions :object-fluents)
  (:types place robot box)
  (:predicates (road ?a ?b - place) (visited ?p - place) (box-at ?b - box ?p - place))
  (:functions (loc ?r - robot) - place (holder ?b - box) - robot)
  (:action move :parameters (?r - robot ?to - place)
    :precondition (road (loc ?r) ?to)
    :effect (and (assign (loc ?r) ?to) (visited ?to)))
  (:action pick :parameters (?r - robot ?b - box)
    :precondition (and (box-at ?b (loc ?r)) (not (= (holder ?b) ?r)))
    :effect (and (assign (holder ?b) ?r) (not (box-at ?b (loc ?r)))))
  (:action drop :parameters (?r - robot ?b - box)
    :precondition (= (holder ?b) ?r)
    :effect (and (assign (holder ?b) undefined) (box-at ?b (loc ?r)))))
"""
ROBOTS_PROBLEM = """
(define (problem robots-1) (:domain robots)
  (:objects a b c - place r1 r2 - robot x - box)
  (:init (road a b) (road b c) (road c a) (= (loc r1) a) (= (loc r2) c) (box-at x a))
  (:goal (and (box-at x c) (= (loc r1) (loc r2)))))
"""


def document(text: str) -> dict[str, Any]:
    return parse_pddl(text, locations=False)


# --- Fase 1 --------------------------------------------------------------------------


def test_function_types_follow_typed_list_groups() -> None:
    text = (
        "(define (domain d) (:requirements :typing :numeric-fluents :object-fluents)"
        " (:types t) (:functions (f) (g ?x - t) - t (h) - number (k) (m) - (either t object)))"
    )
    functions = parse_domain(text, locations=False)["functions"]
    assert [(f["name"], f.get("type")) for f in functions] == [
        ("f", "t"),
        ("g", "t"),
        ("h", None),
        ("k", ["t", "object"]),
        ("m", ["t", "object"]),
    ]


def test_terms_and_assignments_in_documents() -> None:
    domain = document(ROBOTS)
    move = domain["actions"][0]
    fluent = {"op": "fluent", "function": "loc", "args": ["?r"]}
    assert move["precondition"] == {"op": "atom", "predicate": "road", "args": [fluent, "?to"]}
    assert move["effect"]["args"][0] == {"op": "assign", "fluent": fluent, "value": "?to"}
    drop = domain["actions"][2]
    assert drop["effect"]["args"][0]["value"] == {"op": "undefined"}
    problem = document(ROBOTS_PROBLEM)
    assert problem["init"][3] == {
        "op": "=",
        "fluent": {"op": "fluent", "function": "loc", "args": ["r1"]},
        "value": "a",
    }
    # = entre dos funciones: la fase 1 no sabe si son numéricas (comparación).
    assert problem["goal"]["args"][1]["op"] == "compare"
    for doc in (domain, problem):
        validate_document(doc)
        assert parse_pddl(write_pddl(doc), locations=False) == doc


def test_object_fluents_need_their_requirement() -> None:
    with pytest.raises(UndeclaredRequirementError, match=":object-fluents"):
        parse_domain(ROBOTS.replace(" :object-fluents", ""))


def test_object_functions_do_not_need_numeric_fluents() -> None:
    """Sin funciones numéricas no hace falta :numeric-fluents: ni para declarar las de
    objetos, ni para asignarlas, ni para darles valor inicial o compararlas."""
    domain = load_domain(document(ROBOTS))
    load_problem(document(ROBOTS_PROBLEM), domain)


# --- Compilación ---------------------------------------------------------------------


def compiled(domain_text: str, problem_text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    domain, problem = document(domain_text), document(problem_text)
    functions = object_functions(domain)
    return compile_domain(domain, functions), compile_problem(problem, functions)


def test_functions_become_predicates() -> None:
    domain, problem = compiled(ROBOTS, ROBOTS_PROBLEM)
    assert domain["functions"] == []
    assert domain["predicates"][-2:] == [
        {
            "name": "loc",
            "parameters": [{"name": "?r", "type": "robot"}, {"name": "?value", "type": "place"}],
        },
        {
            "name": "holder",
            "parameters": [{"name": "?b", "type": "box"}, {"name": "?value", "type": "robot"}],
        },
    ]
    assert ":object-fluents" not in domain["requirements"]
    assert {"op": "atom", "predicate": "loc", "args": ["r1", "a"]} in problem["init"]
    validate_document(domain)
    validate_document(problem)


def test_conditions_bind_terms_with_exists() -> None:
    move = load_domain(document(ROBOTS)).action("move")
    assert move is not None
    value = Variable("?value0")
    assert isinstance(move.precondition, Exists)
    assert move.precondition.parameters == (Parameter("?value0", ("place",)),)
    assert move.precondition.arg.args == (  # type: ignore[union-attr]
        Atom("loc", (Variable("?r"), value)),
        Atom("road", (value, Variable("?to"))),
    )


def test_assignments_delete_the_old_value() -> None:
    domain, _ = compiled(ROBOTS, ROBOTS_PROBLEM)
    effect = domain["actions"][0]["effect"]
    text = write_pddl({**domain, "actions": [{**domain["actions"][0], "effect": effect}]})
    assert "(forall (?value" in text and "(not (loc ?r ?value" in text
    assert "(loc ?r ?to)" in text
    drop = write_pddl(domain).split("(:action drop")[1]
    assert "(not (holder ?b ?value" in drop and "(holder ?b ?r)" not in drop.split(":effect")[1]


def test_predicate_names_do_not_clash() -> None:
    text = ROBOTS.replace("(box-at ?b - box ?p - place))", "(box-at ?b - box ?p - place) (loc))")
    domain = document(text)
    assert [f.predicate for f in object_functions(domain)] == ["loc-value", "holder"]


@pytest.mark.parametrize(
    ("domain_text", "problem_text", "message"),
    [
        (
            ROBOTS.replace("(visited ?to)", "(increase (loc ?r) 1)"),
            ROBOTS_PROBLEM,
            "solo admite 'assign'",
        ),
        (
            ROBOTS.replace(":object-fluents)", ":object-fluents :numeric-fluents)")
            .replace("(holder ?b - box) - robot)", "(holder ?b - box) - robot (n))")
            .replace("(visited ?to)", "(visited (n))"),
            ROBOTS_PROBLEM,
            "es numérica: no puede ir como argumento",
        ),
        (
            ROBOTS,
            ROBOTS_PROBLEM.replace("(= (loc r1) a)", "(= (loc r1) 3)"),
            "toma un objeto como valor",
        ),
        (
            ROBOTS.replace(":object-fluents)", ":object-fluents :numeric-fluents)"),
            ROBOTS_PROBLEM.rstrip()[:-1] + " (:metric minimize (loc r1)))",
            "funciones de objetos no están admitidas en la métrica",
        ),
    ],
)
def test_compilation_errors(domain_text: str, problem_text: str, message: str) -> None:
    with pytest.raises(SemanticError, match=message):
        domain = load_domain(document(domain_text))
        load_problem(document(problem_text), domain)


# --- Búsqueda y validación -----------------------------------------------------------


def val(domain_text: str, problem_text: str, plan: Plan, tmp_path: Path) -> str:
    domain, problem = compiled(domain_text, problem_text)
    files = {
        "domain.pddl": write_pddl(domain),
        "problem.pddl": write_pddl(problem),
        "plan": plan.to_ipc(),
    }
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    return subprocess.run(
        [str(VALIDATE), *(str(tmp_path / f) for f in files)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout


@pytest.mark.parametrize("preset", ["astar-blind", "gbfs-ff-pref", "lazy-ff-pref"])
def test_classical_plans_with_object_fluents(preset: str, tmp_path: Path) -> None:
    task = build(ROBOTS, ROBOTS_PROBLEM)
    result = search(task, PRESETS[preset])
    assert result.status is SearchStatus.SOLVED and result.plan is not None
    assert simulate(task, result.plan) is None
    if preset == "astar-blind":
        # r1 lleva la caja de a a c (pick, move, move... o pasando por b) y acaba con r2.
        assert len(result.plan) == 4
    if VALIDATE.exists():
        plan = Plan.from_indices(task, result.plan)
        assert "Plan valid" in val(ROBOTS, ROBOTS_PROBLEM, plan, tmp_path)


TEMPORAL = """
(define (domain shuttles)
  (:requirements :typing :durative-actions :numeric-fluents :object-fluents)
  (:types place shuttle)
  (:predicates (served ?p - place))
  (:functions (at ?s - shuttle) - place (distance ?a ?b - place))
  (:durative-action fly :parameters (?s - shuttle ?to - place)
    :duration (= ?duration 3)
    :condition (at start (served (at ?s)))
    :effect (and (at end (assign (at ?s) ?to)) (at end (served ?to)))))
"""
TEMPORAL_PROBLEM = """
(define (problem s1) (:domain shuttles)
  (:objects s - shuttle base moon mars - place)
  (:init (served base) (= (at s) base))
  (:goal (and (served mars) (served moon))))
"""


def test_temporal_plans_with_object_fluents(tmp_path: Path) -> None:
    """Los efectos de una acción durativa sobre funciones de objetos llevan el momento en la
    condición nueva: (when (at end (at ?s ?v)) (at end (not (at ?s ?v))))."""
    task = build(TEMPORAL, TEMPORAL_PROBLEM)
    result = search(task, PRESETS["gbfs-ff-pref-tb"])
    assert result.status is SearchStatus.SOLVED and result.plan is not None
    plan = Plan.from_indices(task, result.plan)
    assert [s.name for s in plan.steps] == ["fly", "fly"]
    if VALIDATE.exists():
        assert "Plan valid" in val(TEMPORAL, TEMPORAL_PROBLEM, plan, tmp_path)


def test_ground_atoms_keep_their_objects() -> None:
    task = build(ROBOTS, ROBOTS_PROBLEM)
    assert Atom("loc", (Constant("r1"), Constant("a"))) in task.atoms(task.init)
