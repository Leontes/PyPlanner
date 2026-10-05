"""Representación interna (AST) de dominios y problemas PDDL.

Es el resultado de la fase 2 del parser. Todas las clases son inmutables y *hashables*, así
que se pueden usar como claves de diccionarios y en conjuntos durante el grounding.

El AST sigue siendo *levantado* (lifted): las acciones tienen parámetros y las fórmulas
pueden contener variables y cuantificadores. La normalización y la instanciación son
trabajo de :mod:`pyplanner.grounding`.

Los nombres están siempre en minúsculas, porque PDDL no distingue mayúsculas.

**Tipos.** Todo objeto, constante y parámetro tiene una tupla de tipos: un solo tipo, o
varios si se declaró con ``(either ...)``. Sin ``:typing``, todo es de tipo ``object``, la
raíz de la jerarquía.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyplanner.loader.object_fluents import ObjectFunction

#: Tipo raíz de la jerarquía; existe siempre.
OBJECT = "object"


@dataclass(frozen=True)
class Variable:
    """Una variable, con el ``?`` incluido en el nombre (``?x``)."""

    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Constant:
    """Un objeto del problema o una constante del dominio, usado como término."""

    name: str

    def __str__(self) -> str:
        return self.name


Term = Variable | Constant


@dataclass(frozen=True)
class Parameter:
    """Declaración de una variable con su tipo (parámetro de acción, predicado o cuantificador)."""

    name: str
    types: tuple[str, ...] = (OBJECT,)

    @property
    def variable(self) -> Variable:
        """La variable declarada, para usarla como término."""
        return Variable(self.name)


@dataclass(frozen=True)
class ObjectDecl:
    """Declaración de un objeto del problema o de una constante del dominio, con su tipo."""

    name: str
    types: tuple[str, ...] = (OBJECT,)

    @property
    def constant(self) -> Constant:
        """El objeto declarado, para usarlo como término."""
        return Constant(self.name)


# --- Fórmulas -----------------------------------------------------------------------


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
class Equals:
    """Igualdad entre dos términos ``(= t1 t2)`` (requisito ``:equality``)."""

    left: Term
    right: Term

    def __str__(self) -> str:
        return f"(= {self.left} {self.right})"


@dataclass(frozen=True)
class Not:
    """Negación. En una fórmula niega cualquier fórmula; en un efecto, un átomo (borrado)."""

    arg: "Formula"

    def __str__(self) -> str:
        return f"(not {self.arg})"


@dataclass(frozen=True)
class And:
    """Conjunción de fórmulas o de efectos. ``And(())`` es la fórmula vacía (verdadera)."""

    args: tuple["Formula | Effect", ...] = ()

    def __str__(self) -> str:
        return f"({' '.join(['and', *map(str, self.args)])})"


@dataclass(frozen=True)
class Or:
    """Disyunción. ``Or(())`` es la fórmula falsa."""

    args: tuple["Formula", ...] = ()

    def __str__(self) -> str:
        return f"({' '.join(['or', *map(str, self.args)])})"


@dataclass(frozen=True)
class Imply:
    """Implicación ``(imply antecedente consecuente)``."""

    left: "Formula"
    right: "Formula"

    def __str__(self) -> str:
        return f"(imply {self.left} {self.right})"


@dataclass(frozen=True)
class Exists:
    """Cuantificador existencial sobre variables con tipo."""

    parameters: tuple[Parameter, ...]
    arg: "Formula"

    def __str__(self) -> str:
        return f"(exists ({' '.join(p.name for p in self.parameters)}) {self.arg})"


@dataclass(frozen=True)
class Forall:
    """Cuantificador universal sobre variables con tipo."""

    parameters: tuple[Parameter, ...]
    arg: "Formula"

    def __str__(self) -> str:
        return f"(forall ({' '.join(p.name for p in self.parameters)}) {self.arg})"


# --- Expresiones numéricas (iteración 3) ---------------------------------------------


@dataclass(frozen=True)
class Number:
    """Una constante numérica."""

    value: float

    def __str__(self) -> str:
        return f"{self.value:g}"


@dataclass(frozen=True)
class FluentTerm:
    """Un término de función ``(función arg1 ... argN)``: una variable numérica del estado."""

    function: str
    args: tuple[Term, ...] = ()

    @property
    def is_ground(self) -> bool:
        """``True`` si ningún argumento es una variable."""
        return all(isinstance(arg, Constant) for arg in self.args)

    def __str__(self) -> str:
        return f"({' '.join([self.function, *map(str, self.args)])})"


@dataclass(frozen=True)
class Arithmetic:
    """Operación aritmética: ``+`` y ``*`` con dos o más términos, ``-`` con uno (cambio de
    signo) o dos, y ``/`` con dos."""

    op: str
    args: tuple["Expression", ...]

    def __str__(self) -> str:
        return f"({' '.join([self.op, *map(str, self.args)])})"


@dataclass(frozen=True)
class TotalTime:
    """Duración del plan. Solo aparece en la métrica; en un plan secuencial, cada acción
    dura una unidad."""

    def __str__(self) -> str:
        return "(total-time)"


@dataclass(frozen=True)
class DurationVar:
    """``?duration``: la duración de la acción durativa en la que aparece (iteración 3b)."""

    def __str__(self) -> str:
        return "?duration"


@dataclass(frozen=True)
class IsViolated:
    """``(is-violated nombre)``: número de veces que se incumple la preferencia ``name``
    (PDDL 3). Solo aparece en la métrica."""

    name: str

    def __str__(self) -> str:
        return f"(is-violated {self.name})"


#: Expresión numérica.
Expression = Number | FluentTerm | Arithmetic | TotalTime | DurationVar | IsViolated


@dataclass(frozen=True)
class Comparison:
    """Comparación numérica ``(<comparador> izquierda derecha)``."""

    comparator: str
    left: Expression
    right: Expression

    def __str__(self) -> str:
        return f"({self.comparator} {self.left} {self.right})"


@dataclass(frozen=True)
class Preference:
    """Preferencia (PDDL 3): ``arg`` se desea pero no es obligatoria; cada incumplimiento
    cuenta en la métrica a través de ``(is-violated name)``.

    Puede ir en la conjunción exterior del objetivo o de una precondición (``arg`` es una
    fórmula), alrededor de una condición con momento de una acción durativa (una
    :class:`TimedCondition`) o alrededor de una restricción del problema. Sin nombre no se
    puede citar en la métrica; varias preferencias pueden compartir nombre (y una dentro de
    un ``forall`` lo comparte con todas sus instancias).
    """

    name: str | None
    arg: "Formula | TimedCondition | Constraint"

    def __str__(self) -> str:
        name = f" {self.name}" if self.name else ""
        return f"(preference{name} {self.arg})"


#: Fórmula lógica (precondiciones, objetivos y condiciones de efectos condicionales). Las
#: preferencias solo aparecen en la conjunción exterior de objetivos y precondiciones.
Formula = Atom | Equals | Not | And | Or | Imply | Exists | Forall | Comparison | Preference


# --- Efectos ------------------------------------------------------------------------


@dataclass(frozen=True)
class ForallEffect:
    """Efecto universal: ``arg`` se aplica para cada asignación de las variables."""

    parameters: tuple[Parameter, ...]
    arg: "Effect"

    def __str__(self) -> str:
        return f"(forall ({' '.join(p.name for p in self.parameters)}) {self.arg})"


@dataclass(frozen=True)
class When:
    """Efecto condicional: ``arg`` se aplica si ``condition`` es cierta *antes* de la acción."""

    condition: Formula
    arg: "Effect"

    def __str__(self) -> str:
        return f"(when {self.condition} {self.arg})"


@dataclass(frozen=True)
class NumericEffect:
    """Efecto numérico: ``increase``, ``decrease``, ``assign``, ``scale-up`` o ``scale-down``.

    ``value`` se evalúa con los valores del estado *anterior* a la acción.
    """

    op: str
    fluent: FluentTerm
    value: Expression

    def __str__(self) -> str:
        return f"({self.op} {self.fluent} {self.value})"


#: Efecto de una acción: añadidos (``Atom``), borrados (``Not(Atom)``), conjunciones; desde
#: la iteración 2, efectos universales y condicionales; desde la 3, efectos numéricos.
Effect = Atom | Not | And | ForallEffect | When | NumericEffect


# --- Acciones durativas (iteración 3b) -----------------------------------------------

#: Momentos de una acción durativa: al empezar, al terminar y durante toda la acción.
START, END, ALL = "start", "end", "all"


@dataclass(frozen=True)
class DurationConstraint:
    """Restricción de duración ``(<comparador> ?duration value)``, con ``<comparador>`` en
    ``=``, ``<=`` y ``>=``.

    ``time`` es el momento en que se evalúa ``value``: ``None`` si no se indicó (al empezar
    la acción), o ``"start"`` o ``"end"`` si se escribió ``(at start|end ...)``.
    """

    comparator: str
    value: Expression
    time: str | None = None

    def __str__(self) -> str:
        text = f"({self.comparator} ?duration {self.value})"
        return f"(at {self.time} {text})" if self.time else text


@dataclass(frozen=True)
class TimedCondition:
    """Condición de una acción durativa en un momento: ``time`` es ``"start"`` (``at
    start``), ``"end"`` (``at end``) o ``"all"`` (``over all``, toda la acción)."""

    time: str
    arg: Formula

    def __str__(self) -> str:
        return f"(over all {self.arg})" if self.time == ALL else f"(at {self.time} {self.arg})"


@dataclass(frozen=True)
class TimedEffect:
    """Efecto de una acción durativa al empezar (``"start"``) o al terminar (``"end"``)."""

    time: str
    arg: Effect

    def __str__(self) -> str:
        return f"(at {self.time} {self.arg})"


@dataclass(frozen=True)
class ContinuousEffect:
    """Efecto continuo: ``fluent`` aumenta (``increase``) o disminuye (``decrease``) a
    ``rate`` unidades por unidad de tiempo mientras dura la acción. Es la forma
    ``(increase fluent (* #t rate))``."""

    op: str
    fluent: FluentTerm
    rate: Expression

    def __str__(self) -> str:
        return f"({self.op} {self.fluent} (* #t {self.rate}))"


# Contenedores de las condiciones y los efectos de las acciones durativas. Son clases
# propias, y no And, Forall y When, para que el motor clásico nunca reciba algo con tiempo.


@dataclass(frozen=True)
class DurativeAnd:
    """Conjunción de condiciones o de efectos de una acción durativa."""

    args: tuple["DurativeCondition | DurativeEffect", ...] = ()

    def __str__(self) -> str:
        return f"({' '.join(['and', *map(str, self.args)])})"


@dataclass(frozen=True)
class DurativeForall:
    """Condición universal de una acción durativa."""

    parameters: tuple[Parameter, ...]
    arg: "DurativeCondition"

    def __str__(self) -> str:
        return f"(forall ({' '.join(p.name for p in self.parameters)}) {self.arg})"


@dataclass(frozen=True)
class DurativeForallEffect:
    """Efecto universal de una acción durativa."""

    parameters: tuple[Parameter, ...]
    arg: "DurativeEffect"

    def __str__(self) -> str:
        return f"(forall ({' '.join(p.name for p in self.parameters)}) {self.arg})"


@dataclass(frozen=True)
class DurativeWhen:
    """Efecto condicional de una acción durativa: la condición tiene momentos, y cada parte
    se comprueba en el suyo (``at start`` al empezar, ``at end`` al terminar, ``over all``
    durante toda la acción)."""

    condition: "DurativeCondition"
    arg: "DurativeEffect"

    def __str__(self) -> str:
        return f"(when {self.condition} {self.arg})"


#: Condición de una acción durativa.
DurativeCondition = TimedCondition | DurativeAnd | DurativeForall | Preference
#: Efecto de una acción durativa.
DurativeEffect = TimedEffect | ContinuousEffect | DurativeAnd | DurativeForallEffect | DurativeWhen


@dataclass(frozen=True)
class DurativeAction:
    """Esquema de acción durativa (PDDL 2.1).

    Attributes:
        duration: conjunción de restricciones de duración (vacía si no tiene ninguna).
        condition: condiciones; ``DurativeAnd(())`` si no tiene.
        effect: efectos; ``DurativeAnd(())`` si no tiene.
    """

    name: str
    parameters: tuple[Parameter, ...]
    duration: tuple[DurationConstraint, ...]
    condition: DurativeCondition
    effect: DurativeEffect


# --- Restricciones de trayectoria (PDDL 3) ------------------------------------------


@dataclass(frozen=True)
class TrajectoryConstraint:
    """Restricción sobre la secuencia de estados del plan (operador modal de PDDL 3).

    ``op`` es ``at-end``, ``always``, ``sometime``, ``at-most-once``, ``sometime-after``,
    ``sometime-before``, ``within``, ``hold-after``, ``always-within`` o ``hold-during``;
    ``formulas`` son sus fórmulas (dos en ``sometime-after``, ``sometime-before`` y
    ``always-within``: φ y ψ) y ``times``, sus instantes (dos en ``hold-during``). En un
    dominio sin acciones durativas, el instante de un estado es el número de pasos del plan.
    """

    op: str
    formulas: tuple[Formula, ...]
    times: tuple[float, ...] = ()

    def __str__(self) -> str:
        op = "at end" if self.op == "at-end" else self.op
        parts = [op, *(f"{t:g}" for t in self.times), *map(str, self.formulas)]
        return f"({' '.join(parts)})"


@dataclass(frozen=True)
class ConstraintAnd:
    """Conjunción de restricciones de trayectoria."""

    args: tuple["Constraint", ...] = ()

    def __str__(self) -> str:
        return f"({' '.join(['and', *map(str, self.args)])})"


@dataclass(frozen=True)
class ConstraintForall:
    """Restricción para cada asignación de las variables."""

    parameters: tuple[Parameter, ...]
    arg: "Constraint"

    def __str__(self) -> str:
        return f"(forall ({' '.join(p.name for p in self.parameters)}) {self.arg})"


#: Restricción de trayectoria (las preferencias solo en las del problema).
Constraint = TrajectoryConstraint | ConstraintAnd | ConstraintForall | Preference


def preferences_in(node: object) -> Iterator[Preference]:
    """Preferencias de una fórmula, condición o restricción (siempre en su capa exterior)."""
    if isinstance(node, Preference):
        yield node
    elif isinstance(node, And | DurativeAnd | ConstraintAnd):
        for arg in node.args:
            yield from preferences_in(arg)
    elif isinstance(node, Forall | DurativeForall | ConstraintForall):
        yield from preferences_in(node.arg)


def preference_names(node: object) -> Iterator[str]:
    """Nombres de las preferencias de una fórmula, condición o restricción."""
    for preference in preferences_in(node):
        if preference.name is not None:
            yield preference.name


# --- Declaraciones ------------------------------------------------------------------


@dataclass(frozen=True)
class Predicate:
    """Declaración de un predicado: nombre y parámetros con tipo."""

    name: str
    parameters: tuple[Parameter, ...] = ()

    @property
    def arity(self) -> int:
        """Número de argumentos del predicado."""
        return len(self.parameters)


def derived_uses(
    formula: "Formula", derived: frozenset[str], negated: bool = False
) -> Iterator[tuple[str, bool]]:
    """Predicados derivados que aparecen en ``formula``, y si aparecen negados (bajo un
    número impar de negaciones, contando el antecedente de ``imply``)."""
    if isinstance(formula, Atom):
        if formula.predicate in derived:
            yield formula.predicate, negated
    elif isinstance(formula, Not):
        yield from derived_uses(formula.arg, derived, not negated)
    elif isinstance(formula, Imply):
        yield from derived_uses(formula.left, derived, not negated)
        yield from derived_uses(formula.right, derived, negated)
    elif isinstance(formula, And | Or):
        for arg in formula.args:
            yield from derived_uses(arg, derived, negated)  # type: ignore[arg-type]
    elif isinstance(formula, Exists | Forall):
        yield from derived_uses(formula.arg, derived, negated)


@dataclass(frozen=True)
class DerivedRule:
    """Regla de un predicado derivado (PDDL 2.2): ``(:derived (predicate parameters...)
    condition)``.

    El átomo ``(predicate parameters...)`` es cierto en un estado si ``condition`` lo es para
    esos valores de los parámetros. Si el predicado tiene varias reglas, basta con una. Los
    predicados derivados no aparecen en efectos ni en el estado inicial: su valor se deduce
    siempre de los demás hechos.
    """

    predicate: str
    parameters: tuple[Parameter, ...]
    condition: Formula

    def __str__(self) -> str:
        head = " ".join([self.predicate, *(p.name for p in self.parameters)])
        return f"(:derived ({head}) {self.condition})"


@dataclass(frozen=True)
class Function:
    """Declaración de una función numérica: nombre y parámetros con tipo."""

    name: str
    parameters: tuple[Parameter, ...] = ()

    @property
    def arity(self) -> int:
        """Número de argumentos de la función."""
        return len(self.parameters)


@dataclass(frozen=True)
class Metric:
    """Métrica de calidad del plan: ``minimize`` o ``maximize`` una expresión."""

    direction: str
    expression: Expression


@dataclass(frozen=True)
class Action:
    """Esquema de acción.

    Una precondición o un efecto ausentes se representan con ``And(())``.
    """

    name: str
    parameters: tuple[Parameter, ...]
    precondition: Formula
    effect: Effect


@dataclass(frozen=True)
class Domain:
    """Un dominio PDDL.

    Attributes:
        requirements: requisitos declarados y todos los que implican (ver
            :func:`pyplanner.requirements.expand_requirements`).
        types: pares (tipo, supertipo), sin incluir ``object``. Vacío sin ``:typing``.
    """

    name: str
    requirements: frozenset[str]
    types: tuple[tuple[str, str], ...]
    constants: tuple[ObjectDecl, ...]
    predicates: tuple[Predicate, ...]
    actions: tuple[Action, ...]
    functions: tuple[Function, ...] = ()
    durative_actions: tuple[DurativeAction, ...] = ()
    derived: tuple[DerivedRule, ...] = ()
    constraints: Constraint | None = None
    #: Funciones de objetos, ya compiladas a predicados (ver
    #: :mod:`pyplanner.loader.object_fluents`); hacen falta para compilar los problemas.
    object_fluents: tuple["ObjectFunction", ...] = ()

    @cached_property
    def preference_names(self) -> frozenset[str]:
        """Nombres de las preferencias de las precondiciones y condiciones de las acciones."""
        names: set[str] = set()
        for action in self.actions:
            names.update(preference_names(action.precondition))
        for durative in self.durative_actions:
            names.update(preference_names(durative.condition))
        return frozenset(names)

    @cached_property
    def derived_predicates(self) -> frozenset[str]:
        """Nombres de los predicados derivados (los que tienen alguna regla)."""
        return frozenset(rule.predicate for rule in self.derived)

    @cached_property
    def derived_strata(self) -> dict[str, int]:
        """Estrato de cada predicado derivado: el menor que respeta sus dependencias.

        Un predicado está en un estrato mayor o igual que los que usa en positivo, y
        estrictamente mayor que los que usa negados. Se supone que las reglas son
        estratificables (lo comprueba la fase 2 del parser).
        """
        derived = self.derived_predicates
        strata = dict.fromkeys(sorted(derived), 0)
        changed = True
        while changed:
            changed = False
            for rule in self.derived:
                for used, negated in derived_uses(rule.condition, derived):
                    needed = strata[used] + negated
                    if strata[rule.predicate] < needed:
                        strata[rule.predicate] = needed
                        changed = True
        return strata

    @cached_property
    def _parents(self) -> dict[str, str]:
        return dict(self.types)

    def supertypes(self, type_name: str) -> tuple[str, ...]:
        """El tipo y todos sus supertipos, hasta ``object`` incluido."""
        chain = [type_name]
        while chain[-1] != OBJECT:
            chain.append(self._parents.get(chain[-1], OBJECT))
        return tuple(chain)

    def is_subtype(self, types: tuple[str, ...], of: tuple[str, ...]) -> bool:
        """``True`` si todo objeto de ``types`` es también de alguno de los tipos ``of``.

        Con varios tipos (``either``), cada uno tiene que ser subtipo de alguno de ``of``.
        """
        return all(any(t in of for t in self.supertypes(own)) for own in types)

    def predicate(self, name: str) -> Predicate | None:
        """Devuelve la declaración del predicado ``name``, o ``None`` si no existe."""
        return next((p for p in self.predicates if p.name == name), None)

    def function(self, name: str) -> Function | None:
        """Devuelve la declaración de la función ``name``, o ``None`` si no existe."""
        return next((f for f in self.functions if f.name == name), None)

    def action(self, name: str) -> Action | None:
        """Devuelve el esquema de acción ``name``, o ``None`` si no existe."""
        return next((a for a in self.actions if a.name == name), None)

    def durative_action(self, name: str) -> DurativeAction | None:
        """Devuelve el esquema de acción durativa ``name``, o ``None`` si no existe."""
        return next((a for a in self.durative_actions if a.name == name), None)


@dataclass(frozen=True)
class TimedLiteral:
    """Literal inicial temporizado (PDDL 2.2): en el instante ``time``, ``atom`` pasa a ser
    cierto (``positive``) o falso, sin que lo cause ninguna acción."""

    time: float
    atom: Atom
    positive: bool = True

    def __str__(self) -> str:
        literal = str(self.atom) if self.positive else f"(not {self.atom})"
        return f"(at {self.time:g} {literal})"


@dataclass(frozen=True)
class Problem:
    """Un problema PDDL, ya comprobado contra su dominio.

    Attributes:
        objects: objetos del problema, sin incluir las constantes del dominio.
        init: hechos del estado inicial (átomos sin variables).
        values: valores iniciales de las funciones (términos sin variables). Una función sin
            valor inicial está indefinida.
        metric: métrica de calidad del plan, si la hay.
        timed_literals: literales iniciales temporizados, ordenados por instante.
        constraints: restricciones de trayectoria del problema (PDDL 3), si las hay.
    """

    name: str
    domain_name: str
    requirements: frozenset[str]
    objects: tuple[ObjectDecl, ...]
    init: frozenset[Atom]
    goal: Formula
    values: tuple[tuple[FluentTerm, float], ...] = ()
    metric: Metric | None = None
    timed_literals: tuple[TimedLiteral, ...] = ()
    constraints: Constraint | None = None
