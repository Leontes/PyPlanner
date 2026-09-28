"""Tests del grounding y de la representación del estado."""

from pyplanner.grounding import bits, mask_of
from pyplanner.loader.ast import Atom, Constant
from tests.samples import DOMAIN, PROBLEM
from tests.tasks import build, chain, fork, unreachable


def atom(predicate: str, *args: str) -> Atom:
    return Atom(predicate, tuple(Constant(a) for a in args))


def test_bits_and_masks() -> None:
    assert list(bits(0b101001)) == [0, 3, 5]
    assert mask_of([0, 3, 5]) == 0b101001
    assert list(bits(0)) == []


def test_static_facts_are_removed() -> None:
    task = chain()
    # "next" es estático: no aparece en los hechos ni en las precondiciones.
    assert all(f.predicate == "at" for f in task.facts)
    assert [str(a) for a in task.actions] == ["(step p0 p1)", "(step p1 p2)", "(step p2 p3)"]
    step = task.actions[0]
    assert [task.facts[i] for i in step.pre_facts] == [atom("at", "p0")]


def test_only_reachable_actions_are_instantiated() -> None:
    # Hay 4 objetos, pero solo 3 pares (x, y) cumplen la precondición estática "next".
    assert len(chain().actions) == 3


def test_initial_state_and_goal() -> None:
    task = chain()
    assert task.atoms(task.init) == [atom("at", "p0")]
    [clause] = task.goal
    assert [task.facts[i] for i in clause.pos_facts] == [atom("at", "p3")]
    assert clause.neg_facts == ()
    assert not task.is_goal(task.init)


def test_applicable_and_apply() -> None:
    task = chain()
    [step] = list(task.applicable(task.init))
    assert str(step) == "(step p0 p1)"
    successor = task.apply(task.init, step)
    assert task.atoms(successor) == [atom("at", "p1")]


def test_actions_without_preconditions_are_always_applicable() -> None:
    task = build(DOMAIN, PROBLEM)
    names = {str(a) for a in task.applicable(task.init)}
    assert "(rest)" in names


def test_add_wins_over_delete() -> None:
    task = build(
        "(define (domain d) (:predicates (p))"
        " (:action a :precondition (p) :effect (and (not (p)) (p))))",
        "(define (problem x) (:domain d) (:init (p)) (:goal (p)))",
    )
    [action] = task.actions
    assert action.del_facts == ()
    assert task.apply(task.init, action) == task.init


def test_unreachable_goal_is_impossible() -> None:
    task = unreachable()
    assert task.goal == ()  # ninguna cláusula alcanzable: objetivo imposible
    assert not task.is_goal(task.init)


def test_grounding_is_deterministic() -> None:
    assert fork() == fork()
    assert [a.name for a in fork().actions] == ["both", "make-q", "make-r"]
