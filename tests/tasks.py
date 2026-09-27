"""Tareas pequeñas hechas a mano para probar el grounding, las heurísticas y la búsqueda."""

from pyplanner.grounding import GroundTask, ground
from pyplanner.loader import load_domain, load_problem
from pyplanner.pddl import parse_domain, parse_problem


def build(domain: str, problem: str) -> GroundTask:
    """Parsea, carga e instancia una tarea escrita en PDDL."""
    loaded = load_domain(parse_domain(domain))
    return ground(loaded, load_problem(parse_problem(problem), loaded))


# Cadena p0 → p1 → p2 → p3: h_max = h_add = h_FF = 3 y la única acción útil es step p0 p1.
CHAIN_DOMAIN = """
(define (domain chain)
  (:predicates (at ?x) (next ?x ?y))
  (:action step :parameters (?x ?y)
    :precondition (and (at ?x) (next ?x ?y))
    :effect (and (at ?y) (not (at ?x)))))
"""
CHAIN_PROBLEM = """
(define (problem chain-3) (:domain chain)
  (:objects p0 p1 p2 p3)
  (:init (at p0) (next p0 p1) (next p1 p2) (next p2 p3))
  (:goal (at p3)))
"""

# Dos objetivos alcanzables con acciones separadas o con una sola acción ("both"):
# h_max = 1, h_add = 2 (cuenta dos veces), h_FF = 1 (el plan relajado usa "both").
FORK_DOMAIN = """
(define (domain fork)
  (:predicates (start) (q) (r))
  (:action both :precondition (start) :effect (and (q) (r)))
  (:action make-q :precondition (start) :effect (q))
  (:action make-r :precondition (start) :effect (r)))
"""
FORK_PROBLEM = """
(define (problem fork-1) (:domain fork) (:init (start)) (:goal (and (q) (r))))
"""

# Irresoluble pero no en la relajación: el testigo solo se puede gastar una vez y hacen
# falta x e y. La búsqueda tiene que agotar el espacio para demostrarlo.
TOKEN_DOMAIN = """
(define (domain token)
  (:predicates (token) (x) (y))
  (:action use-x :precondition (token) :effect (and (x) (not (token))))
  (:action use-y :precondition (token) :effect (and (y) (not (token)))))
"""
TOKEN_PROBLEM = """
(define (problem token-1) (:domain token) (:init (token)) (:goal (and (x) (y))))
"""

# Objetivo inalcanzable incluso en la relajación (y estático): todas las heurísticas dan
# infinito en el estado inicial.
UNREACHABLE_PROBLEM = """
(define (problem chain-bad) (:domain chain)
  (:objects p0 p1 p2)
  (:init (at p0) (next p0 p1))
  (:goal (at p2)))
"""


def chain() -> GroundTask:
    return build(CHAIN_DOMAIN, CHAIN_PROBLEM)


def fork() -> GroundTask:
    return build(FORK_DOMAIN, FORK_PROBLEM)


def token() -> GroundTask:
    return build(TOKEN_DOMAIN, TOKEN_PROBLEM)


def unreachable() -> GroundTask:
    return build(CHAIN_DOMAIN, UNREACHABLE_PROBLEM)


# Grafo con un callejón sin salida: desde s se puede ir a a (que lleva al objetivo t) o
# a b (desde donde no se llega a ninguna parte).
DEAD_END_DOMAIN = """
(define (domain oneway)
  (:predicates (at ?x) (edge ?x ?y))
  (:action move :parameters (?x ?y)
    :precondition (and (at ?x) (edge ?x ?y))
    :effect (and (at ?y) (not (at ?x)))))
"""
DEAD_END_PROBLEM = """
(define (problem oneway-1) (:domain oneway)
  (:objects s a b t)
  (:init (at s) (edge s a) (edge s b) (edge a t))
  (:goal (at t)))
"""


def dead_end() -> GroundTask:
    return build(DEAD_END_DOMAIN, DEAD_END_PROBLEM)
