"""Normalización ADL instanciada: fórmulas → DNF de literales, efectos → efectos condicionales.

La normalización se hace **sobre instancias**: los parámetros de la acción ya tienen valor,
y los cuantificadores se expanden sobre los objetos de su tipo (``exists`` → disyunción,
``forall`` → conjunción). El resultado es siempre conjuntivo por partes:

* una fórmula se convierte en una **disyunción de cláusulas**, y cada cláusula es una
  conjunción de hechos ciertos, hechos falsos y condiciones numéricas (forma normal
  disyuntiva, DNF). Las negaciones se empujan hasta los literales durante el propio
  recorrido (forma normal negativa), e ``imply`` se trata como ``(or (not a) b)``;
* un efecto se convierte en una lista de efectos simples, cada uno con su condición (vacía
  si es incondicional), sus añadidos, sus borrados y sus efectos numéricos.

Simplificaciones que se aplican sobre la marcha, para que la DNF no crezca sin necesidad:

* ``=`` entre objetos se evalúa en el momento;
* los hechos **estáticos** se evalúan con el estado inicial;
* se descartan las cláusulas contradictorias (``p`` y ``¬p``) y las repetidas;
* se eliminan las cláusulas que contienen a otra más pequeña (*subsumción*).

Si la DNF supera :data:`MAX_CLAUSES` cláusulas se aborta con un error claro: esas fórmulas
hay que escribirlas con predicados derivados.

**Numérico.** Las comparaciones pasan a condiciones canónicas (ver
:mod:`pyplanner.grounding.numeric`) y se guardan en la parte numérica de la cláusula. Las
funciones estáticas se sustituyen por su valor, y las comparaciones sin variables se
evalúan en el momento. Una comparación que usa una función estática sin valor es falsa, y
un efecto que la usa hace que la instancia no sea aplicable (:class:`UndefinedEffect`).
Mientras se normaliza, las variables numéricas se identifican por su término
(``("fuel", ("plane1",))``).
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product

from pyplanner.errors import UnsupportedFeatureError
from pyplanner.grounding.numeric import (
    Const,
    NumericAssignment,
    NumericCondition,
    NumExpr,
    Var,
    comparison,
    constant_truth,
    make,
)
from pyplanner.loader.ast import (
    And,
    Arithmetic,
    Atom,
    Comparison,
    Constant,
    Effect,
    Equals,
    Exists,
    Expression,
    FluentTerm,
    Forall,
    ForallEffect,
    Formula,
    Imply,
    Not,
    Number,
    NumericEffect,
    Or,
    Parameter,
    Term,
    TotalTime,
    When,
)

#: Un hecho instanciado: (predicado, argumentos). También identifica un término de función.
Fact = tuple[str, tuple[str, ...]]

#: Asignación de variables a objetos.
Binding = Mapping[str, str]

#: Máximo de cláusulas de una DNF antes de abortar.
MAX_CLAUSES = 10_000

#: Por encima de este número de cláusulas no se buscan subsumidas (el coste es cuadrático).
SUBSUMPTION_LIMIT = 256


@dataclass(frozen=True)
class Clause:
    """Conjunción de literales: hechos ciertos (``pos``), hechos falsos (``neg``) y
    condiciones numéricas (``numeric``)."""

    pos: frozenset[Fact] = frozenset()
    neg: frozenset[Fact] = frozenset()
    numeric: frozenset[NumericCondition] = frozenset()

    def conjoin(self, other: "Clause") -> "Clause | None":
        """Conjunción de dos cláusulas, o ``None`` si es contradictoria."""
        pos = self.pos | other.pos
        neg = self.neg | other.neg
        if pos & neg:
            return None
        return Clause(pos, neg, self.numeric | other.numeric)

    def subsumes(self, other: "Clause") -> bool:
        """``True`` si esta cláusula es más débil o igual (sus literales están en ``other``)."""
        return self.pos <= other.pos and self.neg <= other.neg and self.numeric <= other.numeric

    @property
    def size(self) -> int:
        return len(self.pos) + len(self.neg) + len(self.numeric)


#: DNF: lista de cláusulas. ``TRUE`` es una cláusula vacía; ``FALSE``, ninguna.
DNF = list[Clause]
TRUE: DNF = [Clause()]
FALSE: DNF = []


@dataclass(frozen=True)
class SimpleEffect:
    """Efecto normalizado: bajo ``condition`` (DNF), añade ``adds``, borra ``deletes`` y
    aplica los efectos numéricos ``numeric``."""

    condition: tuple[Clause, ...]
    adds: frozenset[Fact]
    deletes: frozenset[Fact]
    numeric: tuple[NumericAssignment, ...] = ()


class UndefinedEffect(Exception):
    """Un efecto usa una función estática sin valor: la instancia no se puede aplicar."""


class Normalizer:
    """Instancia y normaliza fórmulas y efectos de un dominio y un problema concretos.

    Args:
        objects_of: objetos (y constantes) de unos tipos dados, ordenados.
        static_predicates: predicados que ninguna acción modifica.
        static_facts: hechos estáticos ciertos (los del estado inicial).
        static_functions: funciones que ninguna acción modifica.
        static_values: valores iniciales de los términos de las funciones estáticas.
    """

    def __init__(
        self,
        objects_of: Callable[[tuple[str, ...]], Sequence[str]],
        static_predicates: frozenset[str],
        static_facts: frozenset[Fact],
        static_functions: frozenset[str] = frozenset(),
        static_values: Mapping[Fact, float] | None = None,
    ) -> None:
        self.objects_of = objects_of
        self.static_predicates = static_predicates
        self.static_facts = static_facts
        self.static_functions = static_functions
        self.static_values: Mapping[Fact, float] = static_values or {}
        # Cada hecho (y término de función) se guarda una sola vez: los candidatos de una
        # tarea grande comparten unos pocos cientos de hechos, y con una tupla por candidato
        # eran la mitad de la memoria del grounding.
        self._facts: dict[Fact, Fact] = {}

    # --- Fórmulas ---------------------------------------------------------------------

    def dnf(self, formula: Formula, binding: Binding, negated: bool = False) -> DNF:
        """DNF de ``formula`` (o de su negación) con las variables de ``binding``."""
        if isinstance(formula, Atom):
            fact = self._fact(formula.predicate, formula.args, binding)
            if formula.predicate in self.static_predicates:
                return TRUE if (fact in self.static_facts) != negated else FALSE
            return [Clause(neg=frozenset([fact]))] if negated else [Clause(pos=frozenset([fact]))]
        if isinstance(formula, Equals):
            left, right = self._args((formula.left, formula.right), binding)
            return TRUE if (left == right) != negated else FALSE
        if isinstance(formula, Comparison):
            return self._comparison(formula, binding, negated)
        if isinstance(formula, Not):
            return self.dnf(formula.arg, binding, not negated)
        if isinstance(formula, And | Or):
            if isinstance(formula, And) != negated:
                return self._conjunction((arg, binding, negated) for arg in formula.args)  # type: ignore[misc]
            return _disjunction(self.dnf(arg, binding, negated) for arg in formula.args)  # type: ignore[arg-type]
        if isinstance(formula, Imply):
            # a → b  ≡  ¬a ∨ b;   ¬(a → b)  ≡  a ∧ ¬b
            if negated:
                return self._conjunction(
                    [(formula.left, binding, False), (formula.right, binding, True)]
                )
            return _disjunction(
                [self.dnf(formula.left, binding, True), self.dnf(formula.right, binding)]
            )
        if isinstance(formula, Exists | Forall):
            extensions = self.expand(formula.parameters, binding)
            if isinstance(formula, Forall) != negated:
                return self._conjunction((formula.arg, e, negated) for e in extensions)
            return _disjunction(self.dnf(formula.arg, e, negated) for e in extensions)
        raise TypeError(f"fórmula desconocida: {formula!r}")

    def _conjunction(self, items: Iterable[tuple[Formula, Binding, bool]]) -> DNF:
        """DNF de la conjunción de ``items`` (fórmula, asignación, negada).

        Da lo mismo que :func:`_conjunction` sobre las DNF de cada uno, pero mientras el
        resultado es una sola cláusula (lo normal: conjunciones de literales), sus
        literales se acumulan en conjuntos y la cláusula se construye al final, sin una DNF
        por literal ni una cláusula intermedia por cada conjunción. Cuando una parte tiene
        varias cláusulas, se sigue como :func:`_conjunction`. Termina en cuanto una parte es
        falsa (sin instanciar las siguientes).
        """
        pos: set[Fact] = set()
        neg: set[Fact] = set()
        numeric: set[NumericCondition] = set()
        result: DNF | None = None  # None: el resultado es la cláusula (pos, neg, numeric)
        for formula, binding, negated in items:
            if result is None and isinstance(formula, Atom):
                fact = self._fact(formula.predicate, formula.args, binding)
                if formula.predicate in self.static_predicates:
                    if (fact in self.static_facts) == negated:
                        return FALSE
                else:
                    (neg if negated else pos).add(fact)
                continue
            part = self.dnf(formula, binding, negated)
            if not part:
                return FALSE
            if result is None:
                if len(part) == 1:
                    clause = part[0]
                    pos |= clause.pos
                    neg |= clause.neg
                    numeric |= clause.numeric
                    continue
                if pos & neg:
                    return FALSE
                result = [Clause(frozenset(pos), frozenset(neg), frozenset(numeric))]
            if part == TRUE:
                continue
            result = _conjunction([result, part])
            if not result:
                return FALSE
            if len(result) == 1:  # vuelve a ser una sola cláusula
                pos, neg, numeric = set(result[0].pos), set(result[0].neg), set(result[0].numeric)
                result = None
        if result is not None:
            return result
        if pos & neg:
            return FALSE
        return [Clause(frozenset(pos), frozenset(neg), frozenset(numeric))]

    def _comparison(self, formula: Comparison, binding: Binding, negated: bool) -> DNF:
        left = self.expression(formula.left, binding)
        right = self.expression(formula.right, binding)
        if left is None or right is None:
            return FALSE  # usa una función estática sin valor
        clauses: DNF = []
        for condition in comparison(formula.comparator, left, right, negated):
            truth = constant_truth(condition)
            if truth is True:
                return TRUE
            if truth is None:
                clauses.append(Clause(numeric=frozenset([condition])))
        return _simplify(clauses)

    def expression(self, expression: Expression, binding: Binding) -> NumExpr | None:
        """Expresión instanciada, con las funciones estáticas sustituidas por su valor.

        Devuelve ``None`` si usa una función estática sin valor.
        """
        if isinstance(expression, Number):
            return Const(expression.value)
        if isinstance(expression, FluentTerm):
            key = self._fact(expression.function, expression.args, binding)
            if expression.function in self.static_functions:
                value = self.static_values.get(key)
                return None if value is None else Const(value)
            return Var(key)
        if isinstance(expression, Arithmetic):
            args = [self.expression(arg, binding) for arg in expression.args]
            if any(arg is None for arg in args):
                return None
            return make(expression.op, args)  # type: ignore[arg-type]
        if isinstance(expression, TotalTime):
            raise UnsupportedFeatureError("'total-time' solo puede aparecer en la métrica")
        raise TypeError(f"expresión desconocida: {expression!r}")

    # --- Efectos ----------------------------------------------------------------------

    def effects(self, effect: Effect, binding: Binding) -> list[SimpleEffect]:
        """Efectos simples de ``effect`` con las variables de ``binding``.

        Los efectos incondicionales se agrupan en un solo :class:`SimpleEffect` con la
        condición ``TRUE``. Los condicionales cuya condición es falsa se descartan.

        Raises:
            UndefinedEffect: si un efecto numérico usa una función estática sin valor.
        """
        adds: set[Fact] = set()
        deletes: set[Fact] = set()
        numeric: list[NumericAssignment] = []
        conditional: list[SimpleEffect] = []
        self._collect(effect, binding, adds, deletes, numeric, conditional)
        result = [SimpleEffect(tuple(TRUE), frozenset(adds), frozenset(deletes), tuple(numeric))]
        return result + conditional

    def _collect(
        self,
        effect: Effect,
        binding: Binding,
        adds: set[Fact],
        deletes: set[Fact],
        numeric: list[NumericAssignment],
        conditional: list[SimpleEffect],
    ) -> None:
        if isinstance(effect, And):
            for part in effect.args:
                self._collect(part, binding, adds, deletes, numeric, conditional)  # type: ignore[arg-type]
        elif isinstance(effect, Atom):
            adds.add(self._fact(effect.predicate, effect.args, binding))
        elif isinstance(effect, Not):
            assert isinstance(effect.arg, Atom)
            deletes.add(self._fact(effect.arg.predicate, effect.arg.args, binding))
        elif isinstance(effect, NumericEffect):
            value = self.expression(effect.value, binding)
            if value is None:
                raise UndefinedEffect(str(effect))
            key = self._fact(effect.fluent.function, effect.fluent.args, binding)
            numeric.append(NumericAssignment(key, effect.op, value))
        elif isinstance(effect, ForallEffect):
            for extended in self.expand(effect.parameters, binding):
                self._collect(effect.arg, extended, adds, deletes, numeric, conditional)
        elif isinstance(effect, When):
            condition = self.dnf(effect.condition, binding)
            if not condition:
                return
            inner_adds: set[Fact] = set()
            inner_deletes: set[Fact] = set()
            inner_numeric: list[NumericAssignment] = []
            self._collect(
                effect.arg, binding, inner_adds, inner_deletes, inner_numeric, conditional
            )
            if condition == TRUE:
                adds |= inner_adds
                deletes |= inner_deletes
                numeric.extend(inner_numeric)
            elif inner_adds or inner_deletes or inner_numeric:
                conditional.append(
                    SimpleEffect(
                        tuple(condition),
                        frozenset(inner_adds),
                        frozenset(inner_deletes),
                        tuple(inner_numeric),
                    )
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

    def _fact(self, name: str, terms: Iterable[Term], binding: Binding) -> Fact:
        """El hecho (o término de función) ``name`` con los argumentos instanciados, siempre
        el mismo objeto para el mismo hecho."""
        fact = (name, self._args(terms, binding))
        return self._facts.setdefault(fact, fact)

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
            "disyuntiva; este tipo de fórmulas hay que escribirlas con predicados derivados"
        )
    if len(unique) <= 1 or len(unique) > SUBSUMPTION_LIMIT:
        return unique
    unique.sort(key=lambda c: c.size)
    kept: list[Clause] = []
    for clause in unique:
        if not any(smaller.subsumes(clause) for smaller in kept):
            kept.append(clause)
    return kept
