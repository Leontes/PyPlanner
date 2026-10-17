"""Landmarks de la relajación y su conteo al estilo de LAMA (trabajo opcional, punto 2)."""

import pytest

from pyplanner.grounding import GroundTask, State
from pyplanner.heuristics import create_heuristic
from pyplanner.heuristics.landmarks import LandmarkGraph
from pyplanner.plan import simulate
from pyplanner.search import PRESETS, ConfigError, SearchConfig, SearchStatus, search
from pyplanner.search.engine import EagerSearch
from tests.tasks import build, chain
from tests.test_numeric_engine import TANK
from tests.test_search import GRIPPER


def names(task: GroundTask, graph: LandmarkGraph, facts: int) -> set[str]:
    return {str(task.facts[f]) for f in graph.facts if facts >> f & 1}


def apply(task: GroundTask, state: State, name: str) -> State:
    action = next(a for a in task.actions if str(a) == name)
    successor = action.apply(state)
    assert successor is not None
    return successor


def test_landmarks_and_orderings_of_a_chain() -> None:
    """Para llegar a p3 hay que pasar por p1 y p2: cada uno va antes que el siguiente, y
    justo antes (es precondición de la única acción que consigue el siguiente)."""
    task = chain()
    graph = LandmarkGraph(task)
    assert names(task, graph, graph.mask) == {"(at p0)", "(at p1)", "(at p2)", "(at p3)"}
    at = {str(task.facts[f]): f for f in graph.facts}
    assert names(task, graph, graph.predecessors[at["(at p2)"]]) == {"(at p0)", "(at p1)"}
    assert names(task, graph, graph.greedy[at["(at p1)"]]) == {"(at p2)"}
    assert names(task, graph, graph.goal) == {"(at p3)"}


def test_the_count_follows_the_plan() -> None:
    task = chain()
    graph = LandmarkGraph(task)
    state = task.init
    accepted = graph.accept(0, state)
    assert graph.value(accepted, state) == 3
    preferred = graph.preferred(accepted)
    assert {str(task.actions[i]) for i in preferred} == {"(step p0 p1)"}
    for expected, name in ((2, "(step p0 p1)"), (1, "(step p1 p2)"), (0, "(step p2 p3)")):
        state = apply(task, state, name)
        accepted = graph.accept(accepted, state)
        assert graph.value(accepted, state) == expected


# Para conseguir b hay que gastar a, y el objetivo pide los dos.
SPEND = """
(define (domain spend)
  (:predicates (a) (b))
  (:action make-a :parameters () :precondition (and) :effect (a))
  (:action make-b :parameters () :precondition (a) :effect (and (b) (not (a)))))
"""
SPEND_PROBLEM = "(define (problem spend-1) (:domain spend) (:init) (:goal (and (a) (b))))"


def test_lost_goal_landmarks_count_again() -> None:
    task = build(SPEND, SPEND_PROBLEM)
    graph = LandmarkGraph(task)
    state = apply(task, task.init, "(make-a)")
    accepted = graph.accept(graph.accept(0, task.init), state)
    assert graph.value(accepted, state) == 1  # falta b
    state = apply(task, state, "(make-b)")
    accepted = graph.accept(accepted, state)
    assert names(task, graph, accepted) == {"(a)", "(b)"}
    assert graph.value(accepted, state) == 1  # a está aceptado, pero el objetivo lo pide
    state = apply(task, state, "(make-a)")
    assert graph.value(graph.accept(accepted, state), state) == 0


# Para conseguir c hace falta b justo antes, y b se puede perder antes de conseguir c.
DETOUR = """
(define (domain detour)
  (:predicates (b) (c))
  (:action make-b :parameters () :precondition (and) :effect (b))
  (:action drop-b :parameters () :precondition (b) :effect (not (b)))
  (:action make-c :parameters () :precondition (b) :effect (c)))
"""
DETOUR_PROBLEM = "(define (problem detour-1) (:domain detour) (:init) (:goal (c)))"


def test_lost_greedy_predecessors_count_again() -> None:
    task = build(DETOUR, DETOUR_PROBLEM)
    graph = LandmarkGraph(task)
    state = apply(task, task.init, "(make-b)")
    accepted = graph.accept(graph.accept(0, task.init), state)
    assert graph.value(accepted, state) == 1  # falta c
    state = apply(task, state, "(drop-b)")
    accepted = graph.accept(accepted, state)
    assert graph.value(accepted, state) == 2  # c, y b otra vez (justo antes de c)


def test_numeric_conditions_are_landmarks() -> None:
    graph = LandmarkGraph(TANK)
    assert len(graph) == 1
    assert graph.value(graph.accept(0, TANK.init), TANK.init) == 1


def test_unreachable_goals_are_dead_ends() -> None:
    task = build(SPEND, "(define (problem spend-2) (:domain spend) (:init) (:goal (and (a) (b))))")
    unreachable = build(
        DETOUR.replace("(:action make-b :parameters () :precondition (and) :effect (b))", ""),
        DETOUR_PROBLEM,
    )
    assert not LandmarkGraph(task).unreachable
    graph = LandmarkGraph(unreachable)
    assert graph.unreachable and graph.value(0, unreachable.init) == float("inf")


def test_paths_to_the_same_state_keep_the_common_landmarks() -> None:
    """Como en LAMA: si se llega a un estado por dos caminos, sus landmarks aceptados son los
    aceptados por los dos."""
    config = PRESETS["gbfs-ff-lm-pref"]
    engine = EagerSearch(chain(), config, create_heuristic("ff", chain()), None)
    graph = engine.landmarks
    assert graph is not None
    root = engine.new_node(chain().init, 0, -1, -1)
    engine.accepted[root] = graph.mask  # un camino que ya lo había aceptado todo
    other = engine.new_node(apply(chain(), chain().init, "(step p0 p1)"), 1, root, 0)
    before = engine.accepted[other]
    engine.accepted[root] = graph.accept(0, chain().init)  # y otro que no
    engine.merge_landmarks(other, root, engine.states[other])
    common = graph.accept(graph.accept(0, chain().init), engine.states[other])
    assert before == graph.mask and engine.accepted[other] == common


def test_landmarks_need_greedy_search_without_novelty() -> None:
    with pytest.raises(ConfigError, match="landmarks"):
        SearchConfig("x", "astar", "hmax", landmarks=True)
    with pytest.raises(ConfigError, match="landmarks"):
        SearchConfig("x", "gbfs", "ff", novelty=2, landmarks=True)


@pytest.mark.parametrize("config", ["gbfs-ff-lm-pref", "lazy-ff-lm-pref"])
def test_landmark_searches_find_valid_plans(config: str) -> None:
    for task in (chain(), GRIPPER):
        result = search(task, PRESETS[config])
        assert result.status is SearchStatus.SOLVED and result.plan is not None
        assert simulate(task, result.plan) is None
