"""Tests del conversor inverso JSON → PDDL (ida y vuelta)."""

import pytest

from pyplanner.pddl import parse_file, parse_pddl, write_pddl
from tests.samples import DOMAIN, PROBLEM
from tests.tiers import Instance, discover_instances


@pytest.mark.parametrize("text", [DOMAIN, PROBLEM], ids=["domain", "problem"])
def test_round_trip_samples(text: str) -> None:
    document = parse_pddl(text, locations=False)
    assert parse_pddl(write_pddl(document), locations=False) == document


def test_empty_conjunction_is_written_compactly() -> None:
    document = parse_pddl("(define (domain d) (:action a :precondition (and)))", locations=False)
    assert ":precondition (and)" in write_pddl(document)


TIER1 = [i for i in discover_instances() if i.tier == 1]


@pytest.mark.parametrize("instance", TIER1, ids=lambda i: i.id)
def test_round_trip_tier1(instance: Instance) -> None:
    for path in (instance.domain, instance.problem):
        document = parse_file(path, locations=False)
        del document["source"]
        assert parse_pddl(write_pddl(document), locations=False) == document
