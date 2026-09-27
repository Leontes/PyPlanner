"""Dominios y problemas pequeños para los tests del parser."""

DOMAIN = """
; Dominio de prueba con constantes, predicado 0-ario y acción sin precondición.
(define (domain Test)
  (:requirements :strips)
  (:constants home robot)
  (:predicates (at ?x ?y) (free) (visited ?x))
  (:action Move
    :parameters (?from ?to)
    :precondition (and (at robot ?from) (free))
    :effect (and (at robot ?to) (not (at robot ?from)) (visited ?to)))
  (:action rest
    :parameters ()
    :effect (free)))
"""

PROBLEM = """
(define (problem test-1)
  (:domain test)
  (:objects kitchen)
  (:init (at robot home) (free))
  (:goal (and (visited kitchen) (at robot home))))
"""
