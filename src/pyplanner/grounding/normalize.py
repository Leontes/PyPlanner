"""Normalización ADL instanciada: fórmulas → DNF de literales, efectos → efectos condicionales.

La normalización se hace **sobre instancias**: los parámetros de la acción ya tienen valor,
y los cuantificadores se expanden sobre los objetos de su tipo (``exists`` → disyunción,
``forall`` → conjunción). El resultado es siempre conjuntivo por partes:

* una fórmula se convierte en una **disyunción de cláusulas**, y cada cláusula es una
  conjunción de hechos ciertos y hechos falsos (forma normal disyuntiva, DNF). Las
  negaciones se empujan hasta los literales durante el propio recorrido (forma normal
  negativa), e ``imply`` se trata como ``(or (not a) b)``;
* un efecto se convierte en una lista de efectos simples, cada uno con su condición (vacía
  si es incondicional) y sus añadidos y borrados.

Simplificaciones que se aplican sobre la marcha, para que la DNF no crezca sin necesidad:

* ``=`` entre objetos se evalúa en el momento;
* los hechos **estáticos** se evalúan con el estado inicial;
* se descartan las cláusulas contradictorias (``p`` y ``¬p``) y las repetidas;
* se eliminan las cláusulas que contienen a otra más pequeña (*subsumción*).

Si la DNF supera :data:`MAX_CLAUSES` cláusulas se aborta con un error claro: esas fórmulas
necesitan otra técnica (axiomas, iteración 4).
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product

from pyplanner.errors import UnsupportedFeatureError
from pyplanner.loader.ast import (
    And,
    Atom,
    Constant,
    Effect,
    Equals,
    Exists,
    Forall,
    ForallEffect,
    Formula,
    Imply,
    Not,
    Or,
    Parameter,
    Term,
    When,
)

#: Un hecho instanciado: (predicado, argumentos).
Fact = tuple[str, tuple[str, ...]]

#: Asignación de variables a objetos.
Binding = Mapping[str, str]

#: Máximo de cláusulas de una DNF antes de abortar.
MAX_CLAUSES = 10_000

#: Por encima de este número de cláusulas no se buscan subsumidas (el coste es cuadrático).
SUBSUMPTION_LIMIT = 256


@dataclass(frozen=True)
class Clause:
    """Conjunción de literales: hechos ciertos (``pos``) y hechos falsos (``neg``)."""

    pos: frozenset[Fact] = frozenset()
    neg: frozenset[Fact] = frozenset()

    def conjoin(self, other: "Clause") -> "Clause | None":
        """Conjunción de dos cláusulas, o ``None`` si es contradictoria."""
        pos = self.pos | other.pos
        neg = self.neg | other.neg
        if pos & neg:
            return None
        return Clause(pos, neg)

    def subsumes(self, other: "Clause") -> bool:
        """``True`` si esta cláusula es más débil o igual (sus literales están en ``other``)."""
        return self.pos <= other.pos and self.neg <= other.neg


#: DNF: lista de cláusulas. ``TRUE`` es una cláusula vacía; ``FALSE``, ninguna.
DNF = list[Clause]
TRUE: DNF = [Clause()]
FALSE: DNF = []


@dataclass(frozen=True)
class SimpleEffect:
    """Efecto normalizado: bajo ``condition`` (DNF), añade ``adds`` y borra ``deletes``."""

    condition: tuple[Clause, ...]
    adds: frozenset[Fact]
    deletes: frozenset[Fact]


class Normalizer:
    """Instancia y normaliza fórmulas y efectos de un dominio y un problema concretos.

    Args:
        objects_of: objetos (y constantes) de unos tipos dados, ordenados.
        static_predicates: predicados que ninguna acción modifica.
        static_facts: hechos estáticos ciertos (los del estado inicial).
    """

    def __init__(
        self,
        objects_of: Callable[[tuple[str, ...]], Sequence[str]],
        static_predicates: frozenset[str],
        static_facts: frozenset[Fact],
    ) -> None:
        self.objects_of = objects_of
        self.static_predicates = static_predicates
        self.static_facts = static_facts

    # --- Fórmulas ---------------------------------------------------------------------

    def dnf(self, formula: Formula, binding: Binding, negated: bool = False) -> DNF:
        """DNF de ``formula`` (o de su negación) con las variables de ``binding``."""
        if isinstance(formula, Atom):
            fact = (formula.predicate, self._args(formula.args, binding))
            if formula.predicate in self.static_predicates:
                return TRUE if (fact in self.static_facts) != negated else FALSE
            return [Clause(neg=frozenset([fact]))] if negated else [Clause(pos=frozenset([fact]))]
        if isinstance(formula, Equals):
            left, right = self._args((formula.left, formula.right), binding)
            return TRUE if (left == right) != negated else FALSE
        if isinstance(formula, Not):
            return self.dnf(formula.arg, binding, not negated)
        if isinstance(formula, And | Or):
            parts = [self.dnf(arg, binding, negated) for arg in formula.args]  # type: ignore[arg-type]
            conjunctive = isinstance(formula, And) != negated
            return _conjunction(parts) if conjunctive else _disjunction(parts)
        if isinstance(formula, Imply):
            # a → b  ≡  ¬a ∨ b;   ¬(a → b)  ≡  a ∧ ¬b
            if negated:
                return _conjunction(
                    [self.dnf(formula.left, binding), self.dnf(formula.right, binding, True)]
                )
            return _disjunction(
                [self.dnf(formula.left, binding, True), self.dnf(formula.right, binding)]
            )
        if isinstance(formula, Exists | Forall):
            parts = [
                self.dnf(formula.arg, extended, negated)
                for extended in self.expand(formula.parameters, binding)
            ]
            conjunctive = isinstance(formula, Forall) != negated
            return _conjunction(parts) if conjunctive else _disjunction(parts)
        raise TypeError(f"fórmula desconocida: {formula!r}")

    # --- Efectos ----------------------------------------------------------------------

    def effects(self, effect: Effect, binding: Binding) -> list[SimpleEffect]:
        """Efectos simples de ``effect`` con las variables de ``binding``.

        Los efectos incondicionales se agrupan en un solo :class:`SimpleEffect` con la
        condición ``TRUE``. Los condicionales cuya condición es falsa se descartan.
        """
        adds: set[Fact] = set()
        deletes: set[Fact] = set()
        conditional: list[SimpleEffect] = []
        self._collect(effect, binding, adds, deletes, conditional)
        result = [SimpleEffect(tuple(TRUE), frozenset(adds), frozenset(deletes))]
        return result + conditional

    def _collect(
        self,
        effect: Effect,
        binding: Binding,
        adds: set[Fact],
        deletes: set[Fact],
        conditional: list[SimpleEffect],
    ) -> None:
        if isinstance(effect, And):
            for part in effect.args:
                self._collect(part, binding, adds, deletes, conditional)  # type: ignore[arg-type]
        elif isinstance(effect, Atom):
            adds.add((effect.predicate, self._args(effect.args, binding)))
        elif isinstance(effect, Not):
            assert isinstance(effect.arg, Atom)
            deletes.add((effect.arg.predicate, self._args(effect.arg.args, binding)))
        elif isinstance(effect, ForallEffect):
            for extended in self.expand(effect.parameters, binding):
                self._collect(effect.arg, extended, adds, deletes, conditional)
        elif isinstance(effect, When):
            condition = self.dnf(effect.condition, binding)
            if not condition:
                return
            inner_adds: set[Fact] = set()
            inner_deletes: set[Fact] = set()
            self._collect(effect.arg, binding, inner_adds, inner_deletes, conditional)
            if condition == TRUE:
                adds |= inner_adds
                deletes |= inner_deletes
            elif inner_adds or inner_deletes:
                conditional.append(
                    SimpleEffect(tuple(condition), frozenset(inner_adds), frozenset(inner_deletes))
                )
        else:
            raise TypeError(f"efecto desconocido: {effect!r}")

    # --- Utilidades -------------------------------------------------------------------

    def expand(self, parameters: Iterable[Parameter], binding: Binding) -> Iterable[dict[str, str]]:
        """Todas las extensiones de ``binding`` con valores para ``parameters``."""
        parameters = list(parameters)
        domains = [self.objects_of(p.types) for p in parameters]
        for values in product(*domains):
            extended = dict(binding)
            extended.update(zip((p.name for p in parameters), values, strict=True))
            yield extended

    @staticmethod
    def _args(terms: Iterable[Term], binding: Binding) -> tuple[str, ...]:
        return tuple(t.name if isinstance(t, Constant) else binding[t.name] for t in terms)


def _disjunction(parts: Iterable[DNF]) -> DNF:
    clauses: list[Clause] = []
    for part in parts:
        if part == TRUE:
            return TRUE
        clauses.extend(part)
    return _simplify(clauses)


def _conjunction(parts: Iterable[DNF]) -> DNF:
    result: DNF = TRUE
    for part in parts:
        if not part:
            return FALSE
        if part == TRUE:
            continue
        combined = []
        for left, right in product(result, part):
            clause = left.conjoin(right)
            if clause is not None:
                combined.append(clause)
        result = _simplify(combined)
        if not result:
            return FALSE
    return result


def _simplify(clauses: list[Clause]) -> DNF:
    """Quita repetidas y subsumidas. Aborta si la DNF es demasiado grande."""
    unique = list(dict.fromkeys(clauses))
    if len(unique) > MAX_CLAUSES:
        raise UnsupportedFeatureError(
            f"una fórmula da lugar a más de {MAX_CLAUSES} cláusulas al pasarla a forma normal "
            "disyuntiva; este tipo de fórmulas necesita predicados derivados (iteración 4)"
        )
    if len(unique) <= 1 or len(unique) > SUBSUMPTION_LIMIT:
        return unique
    unique.sort(key=lambda c: len(c.pos) + len(c.neg))
    kept: list[Clause] = []
    for clause in unique:
        if not any(smaller.subsumes(clause) for smaller in kept):
            kept.append(clause)
    return kept
