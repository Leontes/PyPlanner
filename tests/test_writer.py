"""Tests del conversor inverso JSON → PDDL (ida y vuelta)."""

from typing import Any

import pytest

from pyplanner.pddl import parse_file, parse_pddl, write_pddl
from tests.catalog import Instance, instances
from tests.samples import DOMAIN, PROBLEM

EMPTY = {"op": "and", "args": []}


def with_explicit_empty(document: dict[str, Any]) -> dict[str, Any]:
    """El conversor escribe ``(and)`` en las precondiciones y efectos ausentes."""
    for action in document.get("actions", []):
        action.setdefault("precondition", EMPTY)
        action.setdefault("effect", EMPTY)
    return document


def round_trip(document: dict[str, Any]) -> dict[str, Any]:
    return parse_pddl(write_pddl(document), locations=False)


@pytest.mark.parametrize("text", [DOMAIN, PROBLEM], ids=["domain", "problem"])
def test_round_trip_samples(text: str) -> None:
    document = parse_pddl(text, locations=False)
    assert round_trip(document) == with_explicit_empty(document)


def test_empty_conjunction_is_written_compactly() -> None:
    document = parse_pddl("(define (domain d) (:action a :precondition (and)))", locations=False)
    assert ":precondition (and)" in write_pddl(document)


STRIPS = [i for i in instances() if i.level == 1]


@pytest.mark.parametrize("instance", STRIPS, ids=lambda i: i.id)
def test_round_trip_strips(instance: Instance) -> None:
    for path in (instance.domain, instance.problem):
        document = parse_file(path, locations=False)
        del document["source"]
        assert round_trip(document) == with_explicit_empty(document)


def test_missing_precondition_and_effect_are_written_for_val() -> None:
    document = parse_pddl("(define (domain d) (:action a :parameters ()))", locations=False)
    text = write_pddl(document)
    assert ":precondition (and)" in text and ":effect (and)" in text
