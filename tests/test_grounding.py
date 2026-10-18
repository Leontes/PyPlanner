"""Tests del grounding y de la representación del estado."""

from typing import Any

import pytest

from pyplanner.grounding import bits, grounder, mask_of
from pyplanner.loader.ast import Atom, Constant
from tests.samples import DOMAIN, PROBLEM
from tests.tasks import build, chain, fork, unreachable


def atom(predicate: str, *args: str) -> Atom:
    return Atom(predicate, tuple(Constant(a) for a in args))


def test_bits_and_masks() -> None:
    assert list(bits(0b101001)) == [0, 3, 5]
    assert mask_of([0, 3, 5]) == 0b101001
    assert list(bits(0)) == []
    # Máscaras grandes (otro camino): los mismos índices, de menor a mayor.
    indices = [0, 1, 63, 64, 1023, 1024, 5000, 40000]
    assert list(bits(mask_of(indices))) == indices
    assert list(bits(1 << 2000)) == [2000]


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
    assert successor is not None
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


# --- Punto fijo semi-ingenuo y poda de candidatos ----------------------------------------

# La llave se consigue en una vuelta posterior a la de la puerta: su efecto condicional
# tiene que volver a mirarse aunque no llegue ningún hecho que use su precondición.
VAULT = """
(define (domain vault)
  (:requirements :conditional-effects)
  (:predicates (has-key) (opened) (ready))
  (:action try-door :parameters () :precondition (and) :effect (when (has-key) (opened)))
  (:action prepare :parameters () :precondition (and) :effect (ready))
  (:action fetch-key :parameters () :precondition (ready) :effect (has-key)))
"""
VAULT_PROBLEM = "(define (problem vault-1) (:domain vault) (:init) (:goal (opened)))"


def test_conditional_effects_reached_in_a_later_round() -> None:
    task = build(VAULT, VAULT_PROBLEM)
    assert atom("opened") in task.facts and task.goal


# Una escalera de diez peldaños: cada uno se alcanza en una vuelta del punto fijo.
LADDER = """
(define (domain ladder)
  (:predicates (on ?r) (above ?r ?s))
  (:action climb :parameters (?r ?s) :precondition (and (on ?r) (above ?s ?r)) :effect (on ?s)))
"""
LADDER_PROBLEM = """
(define (problem ladder-1) (:domain ladder)
  (:objects r0 r1 r2 r3 r4 r5 r6 r7 r8 r9)
  (:init (on r0) (above r1 r0) (above r2 r1) (above r3 r2) (above r4 r3) (above r5 r4)
         (above r6 r5) (above r7 r6) (above r8 r7) (above r9 r8))
  (:goal (on r9)))
"""


def test_reachability_over_many_rounds() -> None:
    task = build(LADDER, LADDER_PROBLEM)
    assert [str(a) for a in task.actions] == [f"(climb r{i} r{i + 1})" for i in range(9)]


# Los destinos solo aparecen en una disyunción (drive) o en una igualdad (jump, turn).
ROADS = """
(define (domain roads)
  (:requirements :typing :disjunctive-preconditions :equality)
  (:types town side)
  (:constants west east - side)
  (:predicates (in ?t - town) (road ?a ?b - town) (facing ?s - side))
  (:action drive :parameters (?from ?to - town)
    :precondition (and (in ?from) (or (road ?from ?to) (road ?to ?from)))
    :effect (and (in ?to) (not (in ?from))))
  (:action jump :parameters (?from ?via ?to - town)
    :precondition (and (in ?from) (road ?from ?via) (= ?to ?via))
    :effect (and (in ?to) (not (in ?from))))
  (:action turn :parameters (?t - town ?s - side)
    :precondition (and (in ?t) (= ?s west))
    :effect (facing ?s)))
"""
ROADS_PROBLEM = """
(define (problem roads-1) (:domain roads)
  (:objects t0 t1 t2 t3 t4 t5 - town)
  (:init (in t0) (road t0 t1) (road t2 t1) (road t2 t3))
  (:goal (and (in t3) (facing west))))
"""


def test_disjunctions_and_equalities_prune_the_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Solo se normalizan candidatos que acaban siendo acciones: el join usa los disyuntos
    y las igualdades en lugar de probar todas las ciudades."""
    normalized = []
    original = grounder._normalize

    def counting(*args: Any) -> Any:
        normalized.append(args[2])
        return original(*args)

    monkeypatch.setattr(grounder, "_normalize", counting)
    task = build(ROADS, ROADS_PROBLEM)
    names = sorted(str(a) for a in task.actions)
    drives = ["(drive t0 t1)", "(drive t1 t0)", "(drive t1 t2)", "(drive t2 t1)"]
    drives += ["(drive t2 t3)", "(drive t3 t2)"]
    jumps = ["(jump t0 t1 t1)", "(jump t2 t1 t1)", "(jump t2 t3 t3)"]
    turns = [f"(turn t{i} west)" for i in range(4)]
    assert names == sorted(drives + jumps + turns)
    assert len(normalized) == len(names)
