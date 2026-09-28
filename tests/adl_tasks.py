"""Tareas ADL pequeñas hechas a mano: una por construcción de PDDL 1.2."""

# Rotación cíclica con efectos condicionales universales, como en rubiks-cube: todo lo que
# está en a pasa a b, lo de b a c y lo de c a a, a la vez. Las condiciones se leen del
# estado anterior; si se aplicaran en secuencia, x1 pasaría de a a b y después a c.
ROTATE = (
    """
(define (domain rotate)
  (:requirements :adl)
  (:types item slot)
  (:constants a b c - slot)
  (:predicates (at ?x - item ?s - slot))
  (:action rotate :parameters ()
    :effect (and
      (forall (?x - item) (when (at ?x a) (and (not (at ?x a)) (at ?x b))))
      (forall (?x - item) (when (at ?x b) (and (not (at ?x b)) (at ?x c))))
      (forall (?x - item) (when (at ?x c) (and (not (at ?x c)) (at ?x a)))))))
""",
    """
(define (problem rotate-1) (:domain rotate)
  (:objects x1 x2 - item)
  (:init (at x1 a) (at x2 b))
  (:goal (and (at x1 c) (at x2 a))))
""",
)

# Precondiciones y objetivos negativos: encender l1 y apagar l2.
LIGHTS = (
    """
(define (domain lights)
  (:requirements :typing :negative-preconditions)
  (:types light)
  (:predicates (on ?l - light))
  (:action switch-on :parameters (?l - light)
    :precondition (not (on ?l)) :effect (on ?l))
  (:action switch-off :parameters (?l - light)
    :precondition (on ?l) :effect (not (on ?l))))
""",
    """
(define (problem lights-1) (:domain lights)
  (:objects l1 l2 - light)
  (:init (on l2))
  (:goal (and (on l1) (not (on l2)))))
""",
)

# Precondición disyuntiva, igualdad y cuantificadores con tipos:
# - "open" exige una llave cualquiera (exists) o que la puerta no esté cerrada con llave;
# - "leave" exige que todas las puertas estén abiertas (forall);
# - "take" no puede coger la llave "fake" (not =).
DOORS = (
    """
(define (domain doors)
  (:requirements :adl)
  (:types door key)
  (:constants fake - key)
  (:predicates (open ?d - door) (locked ?d - door) (has ?k - key) (outside))
  (:action take :parameters (?k - key)
    :precondition (not (= ?k fake)) :effect (has ?k))
  (:action open :parameters (?d - door)
    :precondition (or (exists (?k - key) (has ?k)) (not (locked ?d)))
    :effect (open ?d))
  (:action leave :parameters ()
    :precondition (forall (?d - door) (open ?d))
    :effect (outside)))
""",
    """
(define (problem doors-1) (:domain doors)
  (:objects d1 d2 - door k1 - key)
  (:init (locked d1))
  (:goal (outside)))
""",
)

# Objetivo disyuntivo con implicación: basta con p, o con q y r.
CHOICE = (
    """
(define (domain choice)
  (:requirements :adl)
  (:predicates (s) (p) (q) (r))
  (:action make-p-slowly-1 :parameters () :precondition (s) :effect (and (not (s)) (q)))
  (:action make-p-slowly-2 :parameters () :precondition (q) :effect (p))
  (:action make-r :parameters () :precondition (s) :effect (r))
  (:action make-q :parameters () :precondition (r) :effect (q)))
""",
    """
(define (problem choice-1) (:domain choice)
  (:init (s))
  (:goal (or (p) (imply (not (r)) (p)) (and (q) (r)))))
""",
)

ALL = {"rotate": ROTATE, "lights": LIGHTS, "doors": DOORS, "choice": CHOICE}
