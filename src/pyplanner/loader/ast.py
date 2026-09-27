"""Representación interna (AST) de dominios y problemas PDDL.

Es el resultado de la fase 2 del parser. Todas las clases son inmutables y *hashables*, así
que se pueden usar como claves de diccionarios y en conjuntos durante el grounding.

El AST sigue siendo *levantado* (lifted): las acciones tienen parámetros y las fórmulas
pueden contener variables. La instanciación es trabajo de :mod:`pyplanner.grounding`.

Los nombres están siempre en minúsculas, porque PDDL no distingue mayúsculas.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Variable:
    """Una variable, con el ``?`` incluido en el nombre (``?x``)."""

    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Constant:
    """Un objeto del problema o una constante del dominio."""

    name: str

    def __str__(self) -> str:
        return self.name


Term = Variable | Constant


@dataclass(frozen=True)
class Atom:
    """Un átomo ``(predicado arg1 ... argN)``."""

    predicate: str
    args: tuple[Term, ...] = ()

    @property
    def is_ground(self) -> bool:
        """``True`` si ningún argumento es una variable."""
        return all(isinstance(arg, Constant) for arg in self.args)

    def __str__(self) -> str:
        return f"({' '.join([self.predicate, *map(str, self.args)])})"


@dataclass(frozen=True)
class Not:
    """Negación. En la v1 solo aparece en efectos, aplicada a un átomo (borrado)."""

    arg: Atom

    def __str__(self) -> str:
        return f"(not {self.arg})"


@dataclass(frozen=True)
class And:
    """Conjunción de fórmulas o de efectos. ``And(())`` es la fórmula vacía (verdadera)."""

    args: tuple["Formula | Effect", ...] = ()

    def __str__(self) -> str:
        return f"({' '.join(['and', *map(str, self.args)])})"


#: Fórmula lógica (precondiciones y objetivos). v1: átomos y conjunciones.
Formula = Atom | And

#: Efecto de una acción. v1: átomos añadidos, átomos borrados y conjunciones.
Effect = Atom | Not | And


@dataclass(frozen=True)
class Predicate:
    """Declaración de un predicado: nombre y parámetros."""

    name: str
    parameters: tuple[Variable, ...] = ()

    @property
    def arity(self) -> int:
        """Número de argumentos del predicado."""
        return len(self.parameters)


@dataclass(frozen=True)
class Action:
    """Esquema de acción STRIPS.

    Una precondición o un efecto ausentes se representan con ``And(())``.
    """

    name: str
    parameters: tuple[Variable, ...]
    precondition: Formula
    effect: Effect


@dataclass(frozen=True)
class Domain:
    """Un dominio PDDL.

    Attributes:
        requirements: requisitos declarados y todos los que implican (ver
            :func:`pyplanner.requirements.expand_requirements`).
    """

    name: str
    requirements: frozenset[str]
    constants: tuple[Constant, ...]
    predicates: tuple[Predicate, ...]
    actions: tuple[Action, ...]

    def predicate(self, name: str) -> Predicate | None:
        """Devuelve la declaración del predicado ``name``, o ``None`` si no existe."""
        return next((p for p in self.predicates if p.name == name), None)

    def action(self, name: str) -> Action | None:
        """Devuelve el esquema de acción ``name``, o ``None`` si no existe."""
        return next((a for a in self.actions if a.name == name), None)


@dataclass(frozen=True)
class Problem:
    """Un problema PDDL, ya comprobado contra su dominio.

    Attributes:
        objects: objetos del problema, sin incluir las constantes del dominio.
        init: hechos del estado inicial (átomos sin variables).
    """

    name: str
    domain_name: str
    requirements: frozenset[str]
    objects: tuple[Constant, ...]
    init: frozenset[Atom]
    goal: Formula
