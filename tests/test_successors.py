"""Tests de la generación de sucesores: referencia en Python y versión compilada."""

import pytest

from pyplanner.frontend import read_domain, read_problem
from pyplanner.grounding import GroundTask, ground
from pyplanner.heuristics import numba_available
from pyplanner.search import PRESETS, search
from pyplanner.search.successors import CompiledSuccessors, create_successor_generator
from tests.adl_tasks import ALL
from tests.tasks import build, chain
from tests.test_heuristics import all_states, random_walk_states
from tests.tiers import Instance, discover_instances

requires_numba = pytest.mark.skipif(not numba_available(), reason="Numba no está instalado")

INSTANCES = [i for i in discover_instances() if i.tier <= 2]


def load(instance: Instance) -> GroundTask:
    domain = read_domain(instance.domain)
    return ground(domain, read_problem(instance.problem, domain))


def test_canonical_order() -> None:
    task = build(
        "(define (domain d) (:requirements :negative-preconditions) (:predicates (p) (q))"
        " (:action z-no-pre :parameters () :precondition (not (q)) :effect (q))"
        " (:action b-on-q :parameters () :precondition (q) :effect (p))"
        " (:action a-on-p :parameters () :precondition (p) :effect (not (p))))",
        "(define (problem x) (:domain d) (:init (p)) (:goal (q)))",
    )
    names = [a.name for a in task.applicable(task.init)]
    # Primero las que no tienen precondiciones positivas; después, por hecho disparador.
    assert names == ["z-no-pre", "a-on-p"]


@requires_numba
@pytest.mark.parametrize("instance", INSTANCES, ids=lambda i: i.id)
def test_compiled_matches_reference_on_benchmarks(instance: Instance) -> None:
    task = load(instance)
    generator = CompiledSuccessors(task)
    for state in random_walk_states(task, 300):
        assert generator(state) == list(task.applicable(state))


@requires_numba
@pytest.mark.parametrize("name", sorted(ALL))
def test_compiled_matches_reference_on_adl(name: str) -> None:
    task = build(*ALL[name])
    generator = CompiledSuccessors(task)
    for state in all_states(task):
        assert generator(state) == list(task.applicable(state))


def test_python_device_uses_the_reference() -> None:
    generator = create_successor_generator(chain(), "python")
    assert not isinstance(generator, CompiledSuccessors)


@requires_numba
def test_search_is_identical_with_both_backends() -> None:
    instance = next(i for i in INSTANCES if i.id == "tier2/slitherlink/p01")
    task = load(instance)
    reference = search(task, PRESETS["gbfs-ff-pref-tb"].with_changes(device="python"))
    compiled = search(task, PRESETS["gbfs-ff-pref-tb"])
    assert compiled.plan == reference.plan
    assert compiled.statistics.expanded == reference.statistics.expanded
