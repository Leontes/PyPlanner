"""Tareas temporales (PDDL 2.1, acciones durativas): grounding y semántica de los estados.

**Compilación en acciones de inicio y fin.** Cada acción durativa ``a`` se convierte, a
nivel de AST, en dos acciones clásicas (*snap actions*) y un hecho ``running@a``:

* ``a@start``: precondición = condiciones ``at start`` y ``¬running@a`` (una instancia no se
  solapa consigo misma); efectos = efectos ``at start`` y ``running@a``;
* ``a@end``: precondición = condiciones ``at end`` y ``over all``, y ``running@a``; efectos =
  efectos ``at end`` y ``¬running@a``.

Así se reutiliza todo el grounder (alcanzabilidad, normalización ADL, parte numérica), la
generación de sucesores compilada y las heurísticas (*h^hbd* sobre las acciones de inicio y
fin, en Python, Numba o PyTorch). El objetivo exige además que no quede ninguna acción en
curso. Un efecto condicional con la condición al empezar y el efecto al terminar se
compila con un hecho auxiliar ``cond@a@k`` que recuerda, al empezar, si la condición se
cumplía.

**Estado temporal** (:class:`TemporalState`): el estado clásico (hechos, incluidos los
``running@a``, y valores) más una **línea temporal** (:class:`Timeline`). Dos estados son
iguales si lo es su parte clásica (memoización al estilo de POPF): la búsqueda no distingue
dos estados con los mismos hechos y valores pero distinta planificación en el tiempo.

**Órdenes al estilo de POPF** (Coles, Coles, Fox y Long, 2010). Cada paso del plan (inicio,
fin o acción instantánea) es un punto de tiempo. Para cada hecho y cada variable numérica
se guarda el último paso que lo cambió y los que lo han leído desde que tiene su valor
actual. Un paso nuevo se ordena, con una separación ε, después de:

* el último paso que cambió lo que lee o lo que modifica;
* los que leyeron, desde su último cambio, lo que escribe. Cuenta aunque la escritura no
  cambie el valor (añadir un hecho ya cierto): VAL no admite que en un mismo instante una
  acción lea un hecho y otra lo escriba.

Los hechos ``running@a`` cuentan como cualquier otro: un nuevo inicio de la misma acción
queda después de su fin anterior, como en la secuencia.

Los pasos independientes no se ordenan entre sí, así que la red temporal los puede hacer
simultáneos: el plan sale en paralelo aunque la búsqueda los haya elegido en secuencia.

**Red temporal (STN).** Restricciones ``t_j ≥ t_i + w``: las órdenes anteriores (w = ε) y la
duración de cada acción (``t_fin ≥ t_inicio + mín`` y ``t_inicio ≥ t_fin − máx``). Se
guardan los tiempos más tempranos (el camino más largo desde el origen) y se actualizan de
forma incremental. Solo la restricción de duración máxima puede crear un ciclo positivo, y
tiene que pasar por el paso nuevo; si aparece, la secuencia es imposible y el sucesor se
descarta. Los tiempos más tempranos son la planificación del plan.

**Invariantes** (``over all``): se comprueban, en el estado resultante de cada paso, para
todas las acciones en curso (un inicio que rompe su propia invariante se descarta así).
Los hechos y variables de una invariante cuentan como leídos por el inicio y por el fin de
la acción, así que los pasos que los cambian quedan ordenados fuera de su intervalo.

**Efectos continuos y duraciones no fijas** con ``?duration``: los decide un programa lineal
por estado (ver :mod:`pyplanner.grounding.temporal_lp`). ``?duration`` no fijo se representa
con una función ``duration@a`` que el programa interpreta como ``t_fin − t_inicio``.

**Efectos condicionales con condiciones ``over all``**: un hecho auxiliar (vigilante) se
activa al empezar si se cumplen las condiciones, y se apaga en cuanto la condición
``over all`` deja de cumplirse mientras dura la acción (ver :class:`_Watch`).

**Predicados derivados** (iteración 4): los axiomas se instancian con el resto de la tarea
y se recalculan tras cada paso, antes de comprobar las invariantes. Ninguna acción escribe
un hecho derivado, así que para las órdenes un paso que lee un hecho derivado lee en
realidad todos los hechos y variables de los que depende (a través de sus axiomas, de forma
transitiva): queda ordenado después del último paso que cambió cualquiera de ellos. Es
correcto, aunque puede ordenar más de lo estrictamente necesario.

**Literales iniciales temporizados** (TIL, iteración 4): cada instante distinto *t_k* de los
TIL se compila en una acción clásica ``til@k`` de coste 0 que aplica sus literales. Se
encadenan con hechos ``til-done@k`` (``til@k`` exige ``til-done@(k-1)`` y que no haya
ocurrido ya), así que la búsqueda los aplica en orden, cuando le conviene. En la red
temporal, su paso queda fijado en *t_k*: una arista desde el origen con peso *t_k* y otra de
vuelta con peso −*t_k*. Si las órdenes lo obligan a ir más tarde, el sucesor se descarta.
Como en OPTIC, todo paso que la secuencia pone antes de un TIL tiene que ocurrir no más
tarde que él (arista de vuelta al origen): si no, el estado seguiría creyendo cierto un
hecho que el TIL ya ha borrado, porque el tiempo pasa aunque la búsqueda no aplique el TIL.
Las órdenes al estilo de POPF hacen el resto: un TIL que borra un hecho queda después de
los pasos que lo leyeron, así que esos pasos tienen que ocurrir antes de *t_k*. Compilarlos
a nivel de AST también hace que el grounder trate como no estáticos los hechos que solo
cambian los TIL. Los TIL no aparecen en el plan.

**Límites** (error claro): lo que la programación lineal no puede expresar (disyunciones
y condiciones de efectos condicionales sobre variables lineales, expresiones no lineales,
``?duration`` no fijo en el ritmo de un efecto continuo).
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cached_property
from typing import TYPE_CHECKING

from pyplanner.errors import UnsupportedFeatureError
from pyplanner.grounding import temporal_lp
from pyplanner.grounding.grounder import Binding, GroundingContext, ground_with_context
from pyplanner.grounding.numeric import (
    Const,
    NumericAssignment,
    NumericCondition,
    NumExpr,
    Op,
    Var,
    evaluate,
    linear,
    variables_of,
)
from pyplanner.grounding.task import Condition, GroundAction, GroundTask, State, bits, mask_of

if TYPE_CHECKING:
    from pyplanner.grounding.monitors import Monitor
from pyplanner.grounding.temporal_lp import LinearModel, LinearState
from pyplanner.loader.ast import (
    ALL,
    END,
    START,
    Action,
    And,
    Arithmetic,
    Atom,
    Comparison,
    Constant,
    ContinuousEffect,
    Domain,
    DurationConstraint,
    DurationVar,
    DurativeAction,
    DurativeAnd,
    DurativeCondition,
    DurativeEffect,
    DurativeForall,
    DurativeForallEffect,
    DurativeWhen,
    Effect,
    Exists,
    Expression,
    FluentTerm,
    Forall,
    ForallEffect,
    Formula,
    Function,
    Imply,
    Not,
    NumericEffect,
    Or,
    Parameter,
    Predicate,
    Preference,
    Problem,
    TimedCondition,
    TimedEffect,
    TotalTime,
    When,
)

#: Separación mínima entre dos pasos que dependen uno de otro (y duración mínima). Con 0,01
#: los planes son válidos con la tolerancia por defecto de VAL (0,01), que agrupa como
#: simultáneos los sucesos más cercanos; 0,001 exigiría validar con `Validate -t 0.001`.
EPSILON = 0.01
#: Tolerancia numérica de la red temporal.
_TOLERANCE = 1e-9

#: Tipos de paso: acción instantánea, inicio y fin de una acción durativa, y literales
#: iniciales temporizados de un instante.
INSTANT, START_SNAP, END_SNAP, TIL_SNAP = 0, 1, 2, 3
#: Prefijo de las acciones de los literales iniciales temporizados (``til@k``). Un nombre
#: PDDL no puede contener ``@``, así que no choca con ninguna acción del dominio.
_TIL = "til@"


# --- Compilación en acciones de inicio y fin ----------------------------------------


@dataclass(frozen=True)
class _Continuous:
    """Efecto continuo de una acción, con las variables de los ``forall`` que lo rodean y
    la condición ``at start`` de los ``when`` que lo rodean (``None`` si no tiene)."""

    parameters: tuple[Parameter, ...]
    condition: Formula | None
    effect: ContinuousEffect


@dataclass(frozen=True)
class _Watch:
    """Efecto condicional con condiciones ``over all``: el hecho ``flag`` se activa al
    empezar si se cumplen ``start`` (antes de los efectos de inicio) y ``over_all`` (después),
    y se apaga en cuanto ``over_all`` deja de cumplirse mientras dura la acción. El efecto
    al terminar depende de ``flag``."""

    flag: Predicate
    parameters: tuple[Parameter, ...]
    start: Formula | None
    over_all: Formula


@dataclass(frozen=True)
class _Compiled:
    """Una acción durativa compilada: sus dos esquemas clásicos y lo que necesita después."""

    action: DurativeAction
    start: Action
    end: Action
    running: Predicate
    invariant: Formula | None
    start_bounds: tuple[DurationConstraint, ...]
    end_bounds: tuple[DurationConstraint, ...]
    continuous: tuple[_Continuous, ...] = ()
    #: Función que representa ``?duration`` (duración no fija usada en los efectos).
    duration: Function | None = None
    watches: tuple[_Watch, ...] = ()


def _compile(action: DurativeAction, changing: frozenset[str]) -> tuple[_Compiled, list[Predicate]]:
    """Compila una acción durativa. ``changing``: funciones que alguna acción modifica."""
    name = action.name
    parameters = action.parameters
    running = Predicate(f"running@{name}", parameters)
    running_atom = Atom(running.name, tuple(p.variable for p in parameters))

    conditions = _split_condition(action.condition)
    flags: list[Predicate] = []
    watches: list[_Watch] = []
    effects, continuous = _split_effect(action.effect, action, (), flags, watches)
    substitute, duration = _duration_value(action, changing, effects, continuous)
    continuous = [
        replace(
            c,
            effect=ContinuousEffect(
                c.effect.op,
                c.effect.fluent,
                _replace_duration(c.effect.rate, substitute),  # type: ignore[arg-type]
            ),
        )
        for c in continuous
    ]

    def conjunction(parts: list[Formula] | list[Effect]) -> And:
        replaced = [_replace_duration(p, substitute) for p in parts]
        return And(tuple(replaced))  # type: ignore[arg-type]

    invariant = conjunction(conditions[ALL]) if conditions[ALL] else None
    start = Action(
        f"{name}@start",
        parameters,
        And((conjunction(conditions[START]), Not(running_atom))),
        And((conjunction(effects[START]), running_atom)),
    )
    end_pre: list[Formula] = [conjunction(conditions[END]), running_atom]
    if invariant is not None:
        end_pre.insert(1, invariant)
    end = Action(
        f"{name}@end",
        parameters,
        And(tuple(end_pre)),
        And((conjunction(effects[END]), Not(running_atom))),
    )
    start_bounds = tuple(c for c in action.duration if c.time in (None, START))
    end_bounds = tuple(c for c in action.duration if c.time == END)
    watches = [
        replace(
            w,
            start=None if w.start is None else _replace_duration(w.start, substitute),  # type: ignore[arg-type]
            over_all=_replace_duration(w.over_all, substitute),  # type: ignore[arg-type]
        )
        for w in watches
    ]
    compiled = _Compiled(
        action, start, end, running, invariant, start_bounds, end_bounds, tuple(continuous),
        duration, tuple(watches),
    )  # fmt: skip
    return compiled, [running, *flags]


def _duration_value(
    action: DurativeAction,
    changing: frozenset[str],
    effects: dict[str, list[Effect]],
    continuous: list[_Continuous],
) -> tuple[Expression | None, Function | None]:
    """Por qué se sustituye ``?duration`` en los efectos (``None`` si no se usa).

    * Con una duración fija (``(= ?duration e)``) cuyas funciones no cambian, por ``e``: su
      valor es el mismo al empezar y al terminar.
    * Si no, por una función nueva ``duration@a`` que el programa lineal interpreta como la
      duración de la acción, una incógnita más (ver :mod:`pyplanner.grounding.temporal_lp`).
      No se admite en el ritmo de un efecto continuo: ritmo × tiempo no sería lineal.
    """
    in_start = any(_uses_duration_var(e) for e in effects[START])
    in_rates = any(_uses_duration_var(c.effect.rate) for c in continuous)
    if not in_start and not in_rates and not any(_uses_duration_var(e) for e in effects[END]):
        return None, None
    fixed = [c for c in action.duration if c.comparator == "=" and c.time in (None, START)]
    if len(action.duration) == 1 and fixed and not set(_functions_of(fixed[0].value)) & changing:
        return fixed[0].value, None
    if in_rates:
        raise UnsupportedFeatureError(
            f"acción durativa '{action.name}': ?duration en el ritmo de un efecto continuo solo "
            "está soportado con una duración fija que no depende de funciones que cambian "
            "(ritmo × tiempo no es lineal)"
        )
    function = Function(f"duration@{action.name}", action.parameters)
    return FluentTerm(function.name, tuple(p.variable for p in action.parameters)), function


def _split_condition(condition: DurativeCondition) -> dict[str, list[Formula]]:
    """Separa una condición en sus partes ``at start``, ``over all`` y ``at end``."""
    parts: dict[str, list[Formula]] = {START: [], ALL: [], END: []}
    if isinstance(condition, TimedCondition):
        parts[condition.time].append(condition.arg)
    elif isinstance(condition, DurativeAnd):
        for arg in condition.args:
            for time, formulas in _split_condition(arg).items():  # type: ignore[arg-type]
                parts[time].extend(formulas)
    elif isinstance(condition, Preference):
        raise UnsupportedFeatureError("el motor temporal todavía no admite preferencias")
    else:  # DurativeForall: el cuantificador se reparte entre los momentos
        for time, formulas in _split_condition(condition.arg).items():
            if formulas:
                parts[time].append(Forall(condition.parameters, And(tuple(formulas))))
    return parts


def _split_effect(
    effect: DurativeEffect,
    action: DurativeAction,
    quantified: tuple[Parameter, ...],
    flags: list[Predicate],
    watches: list[_Watch],
) -> tuple[dict[str, list[Effect]], list[_Continuous]]:
    """Separa un efecto en sus partes ``at start`` y ``at end`` y sus efectos continuos.

    ``quantified``: variables de los ``forall`` que lo rodean (para los hechos auxiliares
    de los efectos condicionales). ``flags`` y ``watches`` reciben los predicados auxiliares
    y los efectos condicionales con condiciones ``over all``.
    """
    parts: dict[str, list[Effect]] = {START: [], END: []}
    continuous: list[_Continuous] = []
    if isinstance(effect, TimedEffect):
        parts[effect.time].append(effect.arg)
    elif isinstance(effect, ContinuousEffect):
        continuous.append(_Continuous((), None, effect))
    elif isinstance(effect, DurativeAnd):
        for arg in effect.args:
            inner, more = _split_effect(arg, action, quantified, flags, watches)  # type: ignore[arg-type]
            for time, effects in inner.items():
                parts[time].extend(effects)
            continuous.extend(more)
    elif isinstance(effect, DurativeForallEffect):
        inner, more = _split_effect(
            effect.arg, action, (*quantified, *effect.parameters), flags, watches
        )
        for time, effects in inner.items():
            if effects:
                parts[time].append(ForallEffect(effect.parameters, And(tuple(effects))))
        # Los efectos continuos se expanden al instanciar: se guardan sus variables.
        continuous.extend(replace(c, parameters=(*effect.parameters, *c.parameters)) for c in more)
    else:
        _split_when(effect, action, quantified, flags, watches, parts, continuous)
    return parts, continuous


def _split_when(
    effect: DurativeWhen,
    action: DurativeAction,
    quantified: tuple[Parameter, ...],
    flags: list[Predicate],
    watches: list[_Watch],
    parts: dict[str, list[Effect]],
    continuous: list[_Continuous],
) -> None:
    """Efecto condicional de una acción durativa (ver la documentación del módulo)."""
    conditions = _split_condition(effect.condition)
    inner, more = _split_effect(effect.arg, action, quantified, flags, watches)
    start_condition = And(tuple(conditions[START]))
    if more:
        # Un efecto continuo condicional se decide al empezar (su ritmo es fijo).
        if conditions[ALL] or conditions[END]:
            raise UnsupportedFeatureError(
                f"acción durativa '{action.name}': un efecto continuo condicional solo puede "
                "depender de condiciones 'at start'"
            )
        for item in more:
            condition = (
                start_condition
                if item.condition is None
                else And((start_condition, item.condition))
            )
            continuous.append(replace(item, condition=condition))
    if inner[START]:  # el loader garantiza que solo depende de condiciones 'at start'
        parts[START].append(When(start_condition, And(tuple(inner[START]))))
    if inner[END] and conditions[ALL]:
        # Vigilante: hecho auxiliar que se activa al empezar y se apaga si la condición
        # 'over all' deja de cumplirse (ver _Watch y TemporalTask.apply).
        parameters = (*action.parameters, *quantified)
        flag = Predicate(f"cond@{action.name}@{len(flags)}", parameters)
        flags.append(flag)
        atom = Atom(flag.name, tuple(p.variable for p in parameters))
        over_all = And(tuple(conditions[ALL]))
        watches.append(
            _Watch(flag, quantified, start_condition if conditions[START] else None, over_all)
        )
        # En la relajación y en el grounding, el hecho se alcanza con el efecto de inicio.
        parts[START].append(When(And((start_condition, over_all)), atom))
        parts[END].append(Not(atom))
        end_condition = And((*conditions[END], atom))
        parts[END].append(When(end_condition, And(tuple(inner[END]))))
    elif inner[END]:
        end_condition_parts: list[Formula] = list(conditions[END])
        if conditions[START]:
            # Se recuerda al empezar si la condición 'at start' se cumplía.
            parameters = (*action.parameters, *quantified)
            flag = Predicate(f"cond@{action.name}@{len(flags)}", parameters)
            flags.append(flag)
            atom = Atom(flag.name, tuple(p.variable for p in parameters))
            parts[START].append(When(start_condition, atom))
            parts[END].append(Not(atom))
            end_condition_parts.append(atom)
        parts[END].append(When(And(tuple(end_condition_parts)), And(tuple(inner[END]))))


def _uses_duration_var(node: object) -> bool:
    """``True`` si ``?duration`` aparece en una condición, un efecto o una expresión."""
    if isinstance(node, DurationVar):
        return True
    if isinstance(node, Arithmetic | And | Or | DurativeAnd):
        return any(_uses_duration_var(arg) for arg in node.args)
    if isinstance(node, Comparison):
        return _uses_duration_var(node.left) or _uses_duration_var(node.right)
    if isinstance(node, NumericEffect):
        return _uses_duration_var(node.value)
    if isinstance(node, ContinuousEffect):
        return _uses_duration_var(node.rate)
    if isinstance(node, Imply):
        return _uses_duration_var(node.left) or _uses_duration_var(node.right)
    if isinstance(node, When | DurativeWhen):
        return _uses_duration_var(node.condition) or _uses_duration_var(node.arg)
    if isinstance(
        node,
        Not
        | Exists
        | Forall
        | ForallEffect
        | TimedCondition
        | TimedEffect
        | DurativeForall
        | DurativeForallEffect,
    ):
        return _uses_duration_var(node.arg)
    return False


def _replace_duration(node: object, value: Expression | None) -> object:
    """Sustituye ``?duration`` por ``value`` en una fórmula, un efecto o una expresión."""
    if value is None:
        return node
    if isinstance(node, DurationVar):
        return value
    if isinstance(node, Arithmetic):
        return Arithmetic(node.op, tuple(_replace_duration(a, value) for a in node.args))  # type: ignore[misc]
    if isinstance(node, Comparison):
        left = _replace_duration(node.left, value)
        return Comparison(node.comparator, left, _replace_duration(node.right, value))  # type: ignore[arg-type]
    if isinstance(node, NumericEffect):
        return NumericEffect(node.op, node.fluent, _replace_duration(node.value, value))  # type: ignore[arg-type]
    if isinstance(node, And):
        return And(tuple(_replace_duration(a, value) for a in node.args))  # type: ignore[misc]
    if isinstance(node, Or):
        return Or(tuple(_replace_duration(a, value) for a in node.args))  # type: ignore[misc]
    if isinstance(node, Not):
        return Not(_replace_duration(node.arg, value))  # type: ignore[arg-type]
    if isinstance(node, Imply):
        left = _replace_duration(node.left, value)
        return Imply(left, _replace_duration(node.right, value))  # type: ignore[arg-type]
    if isinstance(node, Exists | Forall):
        return type(node)(node.parameters, _replace_duration(node.arg, value))  # type: ignore[arg-type]
    if isinstance(node, ForallEffect):
        return ForallEffect(node.parameters, _replace_duration(node.arg, value))  # type: ignore[arg-type]
    if isinstance(node, When):
        condition = _replace_duration(node.condition, value)
        return When(condition, _replace_duration(node.arg, value))  # type: ignore[arg-type]
    return node  # átomos, igualdades y números


def _functions_of(node: object) -> Iterable[str]:
    """Funciones de una expresión."""
    if hasattr(node, "function"):
        yield node.function
    elif isinstance(node, Arithmetic):
        for arg in node.args:
            yield from _functions_of(arg)


def _modified_functions(domain: Domain) -> frozenset[str]:
    """Funciones que modifica alguna acción, instantánea o durativa."""
    result: set[str] = set()

    def visit(node: object) -> None:
        if isinstance(node, NumericEffect | ContinuousEffect):
            result.add(node.fluent.function)
        elif isinstance(node, And | DurativeAnd):
            for arg in node.args:
                visit(arg)
        elif isinstance(
            node, ForallEffect | When | TimedEffect | DurativeForallEffect | DurativeWhen
        ):
            visit(node.arg)

    for action in domain.actions:
        visit(action.effect)
    for durative in domain.durative_actions:
        visit(durative.effect)
    return frozenset(result)


# --- Tarea temporal instanciada -------------------------------------------------------


@dataclass(frozen=True)
class Bound:
    """Restricción de duración instanciada: ``?duration <comparator> value``."""

    comparator: str
    value: NumExpr


@dataclass(frozen=True)
class ContinuousRate:
    """Efecto continuo instanciado: ``variable`` cambia a ``sign · rate`` por unidad de
    tiempo mientras dura la acción, si ``condition`` (DNF) se cumple al empezar (``None``:
    siempre)."""

    variable: int
    sign: float
    rate: NumExpr
    condition: tuple[Condition, ...] | None = None

    def active(self, state: State) -> bool:
        return self.condition is None or any(c.holds(state) for c in self.condition)


@dataclass(frozen=True)
class Watch:
    """Vigilante instanciado de un efecto condicional con condiciones ``over all`` (ver
    :class:`_Watch`)."""

    flag: int
    start: tuple[Condition, ...] | None
    over_all: tuple[Condition, ...]

    def starts(self, before: State, after: State) -> bool:
        start_ok = self.start is None or any(c.holds(before) for c in self.start)
        return start_ok and any(c.holds(after) for c in self.over_all)

    def holds(self, state: State) -> bool:
        return any(c.holds(state) for c in self.over_all)


@dataclass(frozen=True)
class DurativeInstance:
    """Una acción durativa instanciada.

    Attributes:
        name, args: nombre del esquema y objetos.
        running: índice del hecho ``running@a``.
        invariant: condición ``over all`` en DNF (``None`` si no tiene; una tupla vacía es
            imposible).
        start_bounds, end_bounds: restricciones de duración que se evalúan al empezar y al
            terminar.
        reads: hechos y variables (ver :class:`SnapInfo`) de la invariante y de las
            duraciones.
    """

    name: str
    args: tuple[str, ...]
    running: int
    invariant: tuple[Condition, ...] | None
    start_bounds: tuple[Bound, ...]
    end_bounds: tuple[Bound, ...]
    invariant_reads: tuple[int, ...] = ()
    continuous: tuple[ContinuousRate, ...] = ()
    watches: tuple[Watch, ...] = ()


@dataclass(frozen=True)
class SnapInfo:
    """Lo que necesita la línea temporal de cada acción de la tarea.

    Los hechos se identifican por su índice ``i ≥ 0`` y las variables numéricas por
    ``-(v + 1)``, para guardarlos en los mismos diccionarios.

    Attributes:
        kind: :data:`INSTANT`, :data:`START_SNAP`, :data:`END_SNAP` o :data:`TIL_SNAP`.
        durative: índice de la :class:`DurativeInstance` (``-1`` en las instantáneas y los
            TIL).
        reads: hechos y variables que lee (precondiciones, condiciones de efectos,
            valores de los efectos numéricos, invariantes y duraciones).
        writes: hechos y variables que puede cambiar.
        time: instante de los TIL (en los demás pasos no se usa).
    """

    kind: int
    durative: int
    reads: tuple[int, ...]
    writes: tuple[int, ...]
    time: float = 0.0


class Timeline:
    """Planificación en el tiempo de los pasos de un estado (inmutable).

    El paso 0 es el origen (tiempo 0); el paso *k* es la acción *k* del camino hasta el
    estado.

    Attributes:
        earliest: tiempo más temprano de cada paso.
        out: aristas ``(j, w)`` que salen de cada paso (``t_j ≥ t_i + w``).
        changer: último paso que cambió cada hecho o variable.
        readers: pasos que han leído cada hecho o variable desde su último cambio de valor.
        running: acciones en curso: índice → (paso de inicio, duración mínima, máxima).
        linear: el programa lineal, si la tarea tiene variables lineales.
        obligations: restricciones ``always-within`` pendientes: monitor → paso en que se
            disparó φ (el paso en que se cumpla ψ irá como mucho t después).
    """

    __slots__ = ("earliest", "out", "changer", "readers", "running", "linear", "obligations")

    def __init__(
        self,
        earliest: tuple[float, ...] = (0.0,),
        out: tuple[tuple[tuple[int, float], ...], ...] = ((),),
        changer: dict[int, int] | None = None,
        readers: dict[int, tuple[int, ...]] | None = None,
        running: dict[int, tuple[int, float, float]] | None = None,
        linear: LinearState | None = None,
        obligations: dict[int, int] | None = None,
    ) -> None:
        self.earliest = earliest
        self.out = out
        self.changer = changer or {}
        self.readers = readers or {}
        self.running = running or {}
        self.linear = linear
        self.obligations = obligations or {}

    @property
    def makespan(self) -> float:
        """Tiempo del último paso."""
        return max(self.earliest)


class TemporalState(State):
    """Estado temporal: el estado clásico y su :class:`Timeline`.

    Se compara y se usa como clave solo por los hechos y los valores (memoización al
    estilo de POPF).

    **La línea temporal se calcula cuando se necesita.** Un sucesor guarda solo de dónde
    viene (la tarea, el estado padre y la acción); su línea temporal se calcula la primera
    vez que se consulta (al expandirlo) y se guarda desde entonces. La búsqueda genera
    muchos más estados de los que expande, y cada línea temporal crece con la longitud del
    plan: guardarlas todas costaba de 10 a 14 KB por estado en las tareas temporales
    grandes, frente a menos de 1 KB así.
    """

    _timeline: "Timeline | None"
    _origin: "tuple[TemporalTask, State, int] | None"

    def __new__(
        cls,
        facts: int,
        values: tuple[float | None, ...] = (),
        timeline: "Timeline | None" = None,
        source: "tuple[TemporalTask, State, int] | None" = None,
    ) -> "TemporalState":
        self = super().__new__(cls, facts, values)
        self._timeline = timeline
        self._origin = source
        return self

    @property
    def timeline(self) -> "Timeline":
        """La línea temporal del estado (se calcula la primera vez)."""
        if self._timeline is None:
            if self._origin is None:
                self._timeline = Timeline()
            else:
                task, parent, action = self._origin
                self._timeline = task.recompute_timeline(parent, self, action)
                self._origin = None
        return self._timeline

    def peek_timeline(self) -> "Timeline":
        """La línea temporal sin guardarla si todavía no está calculada (la heurística la
        consulta en estados generados, y guardarla en todos costaría mucha memoria)."""
        if self._timeline is not None:
            return self._timeline
        if self._origin is None:
            return Timeline()
        task, parent, action = self._origin
        return task.recompute_timeline(parent, self, action)


@dataclass(frozen=True)
class TemporalStep:
    """Una acción del plan temporal: empieza en ``time`` y dura ``duration`` (``None`` en
    las acciones instantáneas)."""

    time: float
    name: str
    args: tuple[str, ...]
    duration: float | None


@dataclass(frozen=True)
class TemporalTask(GroundTask):
    """Tarea temporal: la tarea clásica de las acciones de inicio y fin, más lo necesario
    para aplicar cada paso sobre la línea temporal (ver la documentación del módulo).

    Attributes:
        durative: acciones durativas instanciadas.
        snaps: información temporal de cada acción de ``actions``.
        metric_is_makespan: ``True`` si la métrica es ``(minimize (total-time))``.
        epsilon: separación entre pasos que dependen uno de otro.
        linear: modelo del programa lineal (``None`` si no hay variables lineales).
        relaxation_actions: acciones para la heurística cuando hay variables lineales: con
            sus condiciones y efectos sobre ellas, y los efectos continuos integrados al
            terminar (ritmo × duración).
        til_times: instantes de los TIL, en orden (el de la acción ``til@k`` es el *k*).
        til_done_mask: hechos ``til-done@k`` (cuántos son ciertos dice cuántos TIL han
            ocurrido).
    """

    durative: tuple[DurativeInstance, ...] = ()
    snaps: tuple[SnapInfo, ...] = ()
    metric_is_makespan: bool = False
    epsilon: float = EPSILON
    linear: LinearModel | None = None
    relaxation_actions: tuple[GroundAction, ...] = ()
    til_times: tuple[float, ...] = ()
    til_done_mask: int = 0

    @cached_property
    def relaxation(self) -> GroundTask:
        """La tarea sobre la que se calcula la heurística."""
        if not self.relaxation_actions:
            return self
        return GroundTask(
            self.domain_name,
            self.problem_name,
            self.facts,
            self.relaxation_actions,
            self.init,
            self.goal,
            self.variables,
            axioms=self.axioms,
            monitor_rules=self.monitor_rules,
        )

    @cached_property
    def _goal_parts(self) -> tuple[tuple[Condition, tuple[NumericCondition, ...]], ...]:
        """Cada cláusula del objetivo, separada en su parte clásica y sus condiciones sobre
        variables lineales."""
        symbolic = self.linear.symbolic if self.linear is not None else frozenset()
        parts = []
        for clause in self.goal:
            linear_part = tuple(c for c in clause.numeric if variables_of(c.expression) & symbolic)
            classical = Condition(
                clause.pos_facts,
                clause.neg_facts,
                tuple(c for c in clause.numeric if c not in linear_part),
            )
            parts.append((classical, linear_part))
        return tuple(parts)

    def is_goal(self, state: State) -> bool:
        """Como en la tarea clásica, pero las condiciones sobre variables lineales se
        comprueban con el programa lineal (pueden cumplirse con otra planificación)."""
        if self.linear is None:
            return super().is_goal(state)
        return self._goal_rows(state) is not None

    def _goal_rows(self, state: State) -> list[temporal_lp.Row] | None:
        """Filas de la primera cláusula del objetivo que se puede cumplir, o ``None``."""
        assert self.linear is not None
        timeline = state.timeline if isinstance(state, TemporalState) else Timeline()
        linear = timeline.linear or LinearState()
        for classical, conditions in self._goal_parts:
            if not classical.holds(state):
                continue
            if not conditions:
                return []
            rows = temporal_lp.goal_rows(self.linear, linear, conditions, state.values)
            if rows is None:
                continue
            if temporal_lp.solve(linear, timeline.out, self.linear.initial, rows) is not None:
                return rows
        return None

    @cached_property
    def running_mask(self) -> int:
        """Hechos ``running@a`` de todas las acciones durativas."""
        return mask_of([d.running for d in self.durative])

    @cached_property
    def running_of(self) -> dict[int, int]:
        """Acción durativa de cada hecho ``running@a``."""
        return {d.running: i for i, d in enumerate(self.durative)}

    def apply(self, state: State, action: GroundAction) -> "TemporalState | None":
        """Aplica un paso: el efecto clásico, las invariantes de las acciones en curso y la
        red temporal. ``None`` si alguna falla."""
        successor = action.apply(state)
        if successor is None:
            return None
        if self.axioms:
            successor = self.axiom_evaluator.evaluate(successor)
        if self.monitors:
            monitored = self.monitor_evaluator.update(state, successor)
            if monitored is None:
                return None  # incumple una restricción de trayectoria
            successor = monitored
        running_of = self.running_of
        durative = self.durative
        if self._has_watches:
            successor = self._watch(state, successor, action.index)
        for fact in bits(successor.facts & self.running_mask):
            invariant = durative[running_of[fact]].invariant
            if invariant is not None and not any(c.holds(successor) for c in invariant):
                return None
        timeline = state.timeline if isinstance(state, TemporalState) else Timeline()
        extended = self._extend(timeline, state, successor, action.index)
        if extended is None:
            return None
        _, linear_values = extended
        values = successor.values
        if linear_values:
            updated = list(values)
            for variable, value in linear_values.items():
                updated[variable] = value
            values = tuple(updated)
        # La línea temporal ya se ha comprobado; no se guarda hasta que se expanda el estado
        # (ver TemporalState).
        return TemporalState(successor.facts, values, source=(self, state, action.index))

    def unsolvability_caveat(self, lost_paths: bool) -> str | None:
        """Dos estados iguales con distinta línea temporal son el mismo nodo (memoización
        al estilo de POPF): al descartar un repetido se pierde un camino que quizá cumplía
        unos plazos que el otro no cumple."""
        if not lost_paths:
            return None
        return (
            "sin plan, pero no se demuestra que no lo haya: la búsqueda temporal ha "
            "descartado estados repetidos, que con otra línea temporal podrían llegar "
            "al objetivo"
        )

    def recompute_timeline(self, parent: State, state: State, action: int) -> Timeline:
        """Línea temporal de ``state``, al que se llegó desde ``parent`` con ``action``."""
        timeline = parent.timeline if isinstance(parent, TemporalState) else Timeline()
        # ``state`` ya incluye los vigilantes y los valores lineales: sirve como "después".
        extended = self._extend(timeline, parent, state, action)
        assert extended is not None, "la línea temporal de un estado ya comprobado es válida"
        return extended[0]

    @cached_property
    def _always_within(self) -> tuple[tuple[int, int, float], ...]:
        """Restricciones ``always-within``: (monitor, hecho ``pending``, plazo)."""
        return tuple(
            (i, m.facts[0], m.bound)
            for i, m in enumerate(self.monitors)
            if m.kind == "always-within"
        )

    def _obligations(
        self, obligations: dict[int, int], before: State, after: State, step: int
    ) -> tuple[dict[int, int], list[tuple[int, float]]]:
        """Obligaciones ``always-within`` tras el paso: se abre una cuando φ queda pendiente
        (en este paso) y, cuando ψ la cumple, el paso queda como mucho a t del que la abrió.
        Una pendiente desde el estado inicial se abrió en el origen (paso 0)."""
        updated = obligations
        deadlines: list[tuple[int, float]] = []
        for index, pending, bound in self._always_within:
            was, now = before.facts >> pending & 1, after.facts >> pending & 1
            if not was and now:
                updated = {**updated, index: step}
            elif was and not now:
                deadlines.append((updated.get(index, 0), bound))
                updated = {k: v for k, v in updated.items() if k != index}
        return updated, deadlines

    @cached_property
    def _has_watches(self) -> bool:
        return any(d.watches for d in self.durative)

    def _watch(self, before: State, after: State, action: int) -> State:
        """Actualiza los hechos de los vigilantes (efectos condicionales con condiciones
        ``over all``): al empezar se activan si se cumplen sus condiciones, y mientras dura
        la acción se apagan en cuanto la condición ``over all`` deja de cumplirse."""
        facts = after.facts
        snap = self.snaps[action]
        if snap.kind == START_SNAP:
            for watch in self.durative[snap.durative].watches:
                if watch.starts(before, after):
                    facts |= 1 << watch.flag
                else:
                    facts &= ~(1 << watch.flag)
        for fact in bits(after.facts & self.running_mask):
            index = self.running_of[fact]
            if index == snap.durative and snap.kind == START_SNAP:
                continue
            for watch in self.durative[index].watches:
                if facts >> watch.flag & 1 and not watch.holds(after):
                    facts &= ~(1 << watch.flag)
        return after if facts == after.facts else State(facts, after.values)

    def _extend(
        self, timeline: Timeline, before: State, after: State, action: int
    ) -> tuple[Timeline, dict[int, float]] | None:
        """Añade el paso a la línea temporal. Devuelve la nueva línea y los valores de las
        variables lineales según el programa lineal, o ``None`` si es inconsistente."""
        snap = self.snaps[action]
        epsilon = self.epsilon
        step = len(timeline.earliest)
        incoming: dict[int, float] = {}

        def after_step(i: int, weight: float) -> None:
            if i != step and incoming.get(i, -math.inf) < weight:
                incoming[i] = weight

        model = self.linear
        reads = snap.reads
        writes = frozenset(-key - 1 for key in snap.writes if key < 0)
        if model is not None and timeline.linear is not None:
            # Un paso que cambia una variable de una invariante lineal en curso la comprueba
            # (antes y después): lee todas sus variables.
            extra = {
                -(v + 1)
                for d in timeline.linear.invariants
                if model.invariant_variables[d] & writes
                for v in model.invariant_variables[d]
            }
            if extra:
                reads = tuple(sorted({*reads, *extra}))

        changer = dict(timeline.changer)
        readers = dict(timeline.readers)
        for key in reads:
            previous = changer.get(key)
            if previous is not None:
                after_step(previous, epsilon)
        # Hechos que el paso escribe de verdad: los que cambian de valor y los de sus efectos
        # incondicionales y condicionales que se disparan, aunque no cambien el valor (VAL no
        # admite que en un mismo instante una acción lea un hecho y otra lo escriba).
        step_action = self.actions[action]
        written = (before.facts ^ after.facts) | step_action.add | step_action.delete
        for effect in step_action.conditional:
            if effect.condition.holds(before):
                written |= effect.add | effect.delete
        for key in snap.writes:
            previous = changer.get(key)
            if previous is not None:
                after_step(previous, epsilon)
            changes = (written >> key) & 1 if key >= 0 else True
            if changes:
                for reader in readers.get(key, ()):
                    after_step(reader, epsilon)
                readers[key] = ()
            changer[key] = step
        for key in reads:
            readers[key] = (*readers.get(key, ()), step)

        running = timeline.running
        start = -1
        upper = math.inf
        stn_bounds: tuple[float, float] | None = None
        if snap.kind == START_SNAP:
            instance = self.durative[snap.durative]
            bounds = _bounds(instance.start_bounds, before.values, epsilon)
            if bounds is None:
                return None
            stn_bounds = bounds
            running = {**running, snap.durative: (step, *bounds)}
        elif snap.kind == END_SNAP:
            instance = self.durative[snap.durative]
            start, lower, upper = running[snap.durative]
            end_bounds = _bounds(instance.end_bounds, before.values, epsilon)
            if end_bounds is None:
                return None
            lower, upper = max(lower, end_bounds[0]), min(upper, end_bounds[1])
            if lower > upper + _TOLERANCE:
                return None
            after_step(start, lower)
            running = {k: v for k, v in running.items() if k != snap.durative}
        elif snap.kind == TIL_SNAP:
            after_step(0, snap.time)

        earliest = [*timeline.earliest, 0.0]
        time = max((earliest[i] + w for i, w in incoming.items()), default=0.0)
        earliest[step] = max(time, 0.0)
        out = [*timeline.out, ()]
        for i, weight in incoming.items():
            out[i] = (*out[i], (step, weight))
        # Un TIL ocurre en su instante, y cualquier otro paso, no más tarde que el próximo
        # TIL pendiente (el que va después en la secuencia).
        deadline = snap.time if snap.kind == TIL_SNAP else self._next_til(before)
        if deadline < math.inf:
            if earliest[step] > deadline + _TOLERANCE:
                return None
            out[step] = ((0, -deadline),)
        # Aristas de vuelta: el paso va como mucho ``bound`` después de un paso anterior (la
        # duración máxima de una acción que termina, o el plazo de un always-within).
        backward: list[tuple[int, float]] = []
        if start >= 0 and upper < math.inf:
            backward.append((start, upper))
        obligations = timeline.obligations
        if self._always_within:
            obligations, deadlines = self._obligations(obligations, before, after, step)
            backward.extend(deadlines)
        propagated = False
        for origin, bound in backward:
            out[step] = (*out[step], (origin, -bound))
            if earliest[origin] < earliest[step] - bound - _TOLERANCE:
                # El paso anterior tiene que retrasarse: se propaga, y si eso retrasa este
                # paso, hay un ciclo positivo (el límite no se puede cumplir).
                earliest[origin] = earliest[step] - bound
                _propagate(earliest, out, origin, skip=step)
                propagated = True
                if earliest[0] > _TOLERANCE:
                    return None  # el retraso choca con el instante de un TIL
                again = max((earliest[i] + w for i, w in incoming.items()), default=0.0)
                if again > earliest[step] + _TOLERANCE:
                    return None

        linear = timeline.linear
        linear_values: dict[int, float] = {}
        if model is not None:
            extended = temporal_lp.extend(
                model,
                linear or LinearState(),
                action,
                step,
                snap.kind == START_SNAP,
                snap.kind == END_SNAP,
                snap.durative,
                start if snap.kind == END_SNAP else step,
                before.values,
                after.values,
                writes,
                started=tuple(
                    (r.variable, r.sign, r.rate)
                    for r in self.durative[snap.durative].continuous
                    if r.active(before)
                )
                if snap.kind == START_SNAP
                else (),
                extra_effects=tuple(
                    a
                    for condition, assignments in model.conditional[action]
                    if condition.holds(before)
                    for a in assignments
                ),
                stn_bounds=stn_bounds,
            )
            if extended is None:
                return None
            linear, changed = extended
            if changed or propagated:
                solution = temporal_lp.solve(linear, out, model.initial)
                if solution is None:
                    return None
                linear_values = temporal_lp.last_values(linear, solution, model.variables)
        new = Timeline(tuple(earliest), tuple(out), changer, readers, running, linear, obligations)
        return new, linear_values

    def _next_til(self, state: State) -> float:
        """Instante del próximo TIL pendiente en ``state`` (``inf`` si no queda ninguno).

        Los TIL se aplican en orden, así que los que han ocurrido son los primeros."""
        if not self.til_times:
            return math.inf
        k = (state.facts & self.til_done_mask).bit_count()
        return self.til_times[k] if k < len(self.til_times) else math.inf

    def schedule(self, plan: Sequence[int]) -> tuple[list[TemporalStep], float]:
        """Planificación de un plan (índices de acciones): los pasos ordenados por tiempo y
        la duración total. El plan tiene que ser válido."""
        state: State = self.init
        for index in plan:
            successor = self.apply(state, self.actions[index])
            assert successor is not None, "el plan no es aplicable"
            state = successor
        assert isinstance(state, TemporalState)
        times: Sequence[float] = state.timeline.earliest
        makespan = state.timeline.makespan
        if self.linear is not None and state.timeline.linear is not None:
            # Con variables lineales, la planificación es la solución del programa lineal.
            goal = self._goal_rows(state)
            assert goal is not None, "el plan no alcanza el objetivo"
            solution = temporal_lp.solve(
                state.timeline.linear, state.timeline.out, self.linear.initial, goal
            )
            assert solution is not None, "el programa lineal del plan no tiene solución"
            times = [solution[("t", j)] for j in range(len(state.timeline.earliest))]
            makespan = max(times)
        steps: list[tuple[float, int, TemporalStep]] = []
        open_starts: dict[int, tuple[int, float]] = {}
        for position, index in enumerate(plan, start=1):
            snap = self.snaps[index]
            action = self.actions[index]
            time = times[position]
            if snap.kind == TIL_SNAP:
                continue  # los TIL no son acciones del plan
            if snap.kind == INSTANT:
                steps.append((time, position, TemporalStep(time, action.name, action.args, None)))
            elif snap.kind == START_SNAP:
                open_starts[snap.durative] = (position, time)
            else:
                start_position, start_time = open_starts.pop(snap.durative)
                instance = self.durative[snap.durative]
                step = TemporalStep(start_time, instance.name, instance.args, time - start_time)
                steps.append((start_time, start_position, step))
        steps.sort(key=lambda item: (item[0], item[1]))
        return [step for _, _, step in steps], makespan


def _bounds(
    bounds: tuple[Bound, ...], values: tuple[float | None, ...], epsilon: float
) -> tuple[float, float] | None:
    """Duración mínima y máxima según ``bounds``; ``None`` si alguna no está definida o no
    queda ninguna duración positiva."""
    lower, upper = epsilon, math.inf
    for bound in bounds:
        value = evaluate(bound.value, values)
        if value is None:
            return None
        if bound.comparator in ("=", ">="):
            lower = max(lower, value)
        if bound.comparator in ("=", "<="):
            upper = min(upper, value)
    if lower > upper + _TOLERANCE:
        return None
    return lower, upper


def _propagate(
    earliest: list[float], out: list[tuple[tuple[int, float], ...]], source: int, skip: int
) -> None:
    """Propaga un retraso de ``source`` a los pasos que dependen de él (sin pasar por
    ``skip``). La red sin ``skip`` es consistente, así que termina."""
    pending = [source]
    while pending:
        node = pending.pop()
        for successor, weight in out[node]:
            if successor == skip:
                continue
            time = earliest[node] + weight
            if time > earliest[successor] + _TOLERANCE:
                earliest[successor] = time
                pending.append(successor)


def ground_temporal(
    domain: Domain, problem: Problem, marker_times: frozenset[float] = frozenset()
) -> TemporalTask:
    """Instancia un dominio con acciones durativas (ver la documentación del módulo)."""
    changing = _modified_functions(domain)
    compiled: list[_Compiled] = []
    predicates: list[Predicate] = []
    for durative_action in domain.durative_actions:
        item, extra = _compile(durative_action, changing)
        compiled.append(item)
        predicates.extend(extra)
    duration_functions = tuple(c.duration for c in compiled if c.duration is not None)
    til_actions, til_predicates, til_times = _timed_literal_actions(problem, marker_times)
    snap_domain = replace(
        domain,
        actions=(
            *domain.actions,
            *(a for c in compiled for a in (c.start, c.end)),
            *til_actions,
        ),
        predicates=(*domain.predicates, *predicates, *til_predicates),
        functions=(*domain.functions, *duration_functions),
        durative_actions=(),
    )
    extra_reads = {}
    for item in compiled:
        # Los ritmos dentro de 'forall' tienen variables sin ligar: solo cuentan sus
        # funciones para la relevancia (ver más abajo).
        rates = tuple(c.effect.rate for c in item.continuous if not c.parameters)
        extra_reads[item.start.name] = (*(c.value for c in item.start_bounds), *rates)
        extra_reads[item.end.name] = tuple(c.value for c in item.end_bounds)
    # Funciones que cambian sin efectos discretos: las de los efectos continuos y las que
    # representan ?duration.
    extra_targeted = frozenset(
        {c.effect.fluent.function for item in compiled for c in item.continuous}
        | {f.name for f in duration_functions}
    )
    # La métrica temporal (total-time) no es aditiva: se busca con coste unitario.
    # Los instantes de las restricciones con tiempo son TIL sin literales: su hecho
    # til-done@k marca que el instante ya ha pasado (ver pyplanner.grounding.monitors).
    markers = {time: f"til-done@{til_times.index(time)}" for time in marker_times}
    task, context = ground_with_context(
        snap_domain, replace(problem, metric=None), extra_reads, extra_targeted, markers
    )

    by_name = {item.action.name: item for item in compiled}
    groups: dict[tuple[str, tuple[str, ...]], dict[str, list[GroundAction]]] = {}
    for action in task.actions:
        base, _, kind = action.name.partition("@")
        if kind in (START, END):
            groups.setdefault((base, action.args), {START: [], END: []})[kind].append(action)

    support = _derived_support(task)
    var_index = context.var_index

    instances: list[DurativeInstance] = []
    instance_of: dict[tuple[str, tuple[str, ...]], int] = {}
    duration_vars: dict[int, int] = {}
    for (base, args), snaps in sorted(groups.items()):
        if not snaps[START] or not snaps[END]:
            continue  # nunca puede empezar o nunca puede terminar
        item = by_name[base]
        binding: Binding = {p.name: a for p, a in zip(item.action.parameters, args, strict=True)}
        invariant = None if item.invariant is None else context.conditions(item.invariant, binding)
        start_bounds = _ground_bounds(item.start_bounds, binding, context)
        end_bounds = _ground_bounds(item.end_bounds, binding, context)
        if start_bounds is None or end_bounds is None:
            continue  # la duración usa una función estática sin valor
        continuous_rates = _ground_continuous(item, binding, context)
        if continuous_rates is None:
            continue  # un ritmo usa una función estática sin valor
        watches = _ground_watches(item, args, binding, context, task.fact_index)
        running = task.fact_index[Atom(item.running.name, tuple(Constant(a) for a in args))]
        reads: set[int] = set()
        for condition in (*(invariant or ()), *(c for w in watches for c in w.over_all)):
            reads |= _condition_keys(condition)
        reads = _expand_derived(reads, support)
        instance_index = len(instances)
        instance_of[(base, args)] = instance_index
        if item.duration is not None:
            key = (item.duration.name, args)
            if key in var_index:
                duration_vars[var_index[key]] = instance_index
        instances.append(
            DurativeInstance(
                base,
                args,
                running,
                invariant,
                start_bounds,
                end_bounds,
                tuple(sorted(reads)),
                continuous_rates,
                watches,
            )
        )

    kept: list[tuple[GroundAction, int, int]] = []  # (acción, tipo, acción durativa)
    for action in task.actions:
        base, _, kind = action.name.partition("@")
        if action.name.startswith(_TIL):
            kept.append((replace(action, index=len(kept), cost=0.0), TIL_SNAP, -1))
        elif not kind:
            kept.append((replace(action, index=len(kept)), INSTANT, -1))
        elif (base, action.args) in instance_of:
            kind_code = START_SNAP if kind == START else END_SNAP
            kept.append(
                (replace(action, index=len(kept)), kind_code, instance_of[(base, action.args)])
            )

    linear_model, actions, relaxation, instances = _linear_parts(
        kept, instances, duration_vars, task.init.values
    )
    snaps_info = []
    for original, (_, kind_code, durative_index) in zip((a for a, _, _ in kept), kept, strict=True):
        instance = instances[durative_index] if durative_index >= 0 else None
        rate_vars = tuple(r.variable for r in instance.continuous) if instance else ()
        info = _snap_info(original, kind_code, durative_index, instance, rate_vars)
        if kind_code == TIL_SNAP:
            info = replace(info, time=til_times[int(original.name.removeprefix(_TIL))])
        if support:
            info = replace(info, reads=tuple(sorted(_expand_derived(set(info.reads), support))))
        if task.monitors:
            info = _monitor_orders(info, original, task.monitors, support)
        snaps_info.append(info)
    if linear_model is not None:
        for axiom in task.axioms:
            if any(
                variables_of(c.expression) & linear_model.symbolic for c in axiom.condition.numeric
            ):
                raise UnsupportedFeatureError(
                    f"el predicado derivado {task.facts[axiom.head]} depende de una variable "
                    "que cambia de forma continua o con ?duration: el programa lineal no puede "
                    "decidir su valor"
                )

    goal = tuple(
        Condition(
            clause.pos_facts,
            tuple(sorted({*clause.neg_facts, *(d.running for d in instances)})),
            clause.numeric,
        )
        for clause in task.goal
    )
    notes = list(task.notes)
    metric = problem.metric
    is_makespan = (
        metric is not None
        and metric.direction == "minimize"
        and isinstance(metric.expression, TotalTime)
    )
    if metric is not None and not is_makespan:
        notes.append(
            "en planificación temporal solo se usa la métrica (minimize (total-time)); "
            "esta se ignora"
        )
    init = TemporalState(task.init.facts, task.init.values)
    return TemporalTask(
        domain_name=task.domain_name,
        problem_name=task.problem_name,
        facts=task.facts,
        actions=tuple(actions),
        init=init,
        goal=goal,
        variables=task.variables,
        notes=tuple(notes),
        durative=tuple(instances),
        snaps=tuple(snaps_info),
        metric_is_makespan=is_makespan,
        linear=linear_model,
        relaxation_actions=tuple(relaxation),
        axioms=task.axioms,
        monitors=task.monitors,
        monitor_rules=task.monitor_rules,
        til_times=tuple(til_times),
        til_done_mask=mask_of(
            task.fact_index[Atom(f"til-done@{k}")]
            for k in range(len(til_times))
            if Atom(f"til-done@{k}") in task.fact_index
        ),
    )


def _monitor_orders(
    info: SnapInfo,
    action: GroundAction,
    monitors: tuple["Monitor", ...],
    support: Mapping[int, frozenset[int]],
) -> SnapInfo:
    """Un paso que puede cambiar algo de lo que vigila un monitor lee y escribe sus hechos (y
    lee lo vigilado), así que esos pasos quedan ordenados entre sí como en la secuencia: el
    resultado de un monitor depende del orden en que cambian sus fórmulas."""
    written = {*action.add_facts, *action.del_facts}
    for effect in action.conditional:
        written |= {*effect.add_facts, *effect.del_facts}
    variables = {v.variable for v in action.numeric} | {
        v.variable for effect in action.conditional for v in effect.numeric
    }
    reads, writes = set(info.reads), set(info.writes)
    for monitor in monitors:
        watched = _expand_derived(set(bits(monitor.mask)), support)
        if written & watched or variables & set(monitor.variables):
            reads |= watched | {-(v + 1) for v in monitor.variables} | set(monitor.facts)
            writes |= set(monitor.facts)
    return replace(info, reads=tuple(sorted(reads)), writes=tuple(sorted(writes)))


def _timed_literal_actions(
    problem: Problem, marker_times: frozenset[float] = frozenset()
) -> tuple[list[Action], list[Predicate], list[float]]:
    """Una acción ``til@k`` por cada instante distinto de los TIL y de las restricciones con
    tiempo (``marker_times``, sin literales), con sus hechos ``til-done@k`` (ver la
    documentación del módulo), y los instantes."""
    times = sorted({literal.time for literal in problem.timed_literals} | marker_times)
    actions: list[Action] = []
    predicates: list[Predicate] = []
    for k, time in enumerate(times):
        done = Atom(f"til-done@{k}")
        predicates.append(Predicate(done.predicate))
        precondition: list[Formula] = [Not(done)]
        if k:
            precondition.append(Atom(f"til-done@{k - 1}"))
        effects: list[Effect] = [done]
        for literal in problem.timed_literals:
            if literal.time == time:
                effects.append(literal.atom if literal.positive else Not(literal.atom))
        actions.append(Action(f"{_TIL}{k}", (), And(tuple(precondition)), And(tuple(effects))))
    return actions, predicates, times


def _derived_support(task: GroundTask) -> dict[int, frozenset[int]]:
    """Hechos básicos y variables (con la codificación de :class:`SnapInfo`) de los que
    depende cada hecho derivado, directamente o a través de otros hechos derivados."""
    derived = task.derived_mask
    direct: dict[int, set[int]] = {}
    for axiom in task.axioms:
        direct.setdefault(axiom.head, set()).update(_condition_keys(axiom.condition))
    support = {
        head: {k for k in keys if k < 0 or not derived >> k & 1} for head, keys in direct.items()
    }
    changed = True
    while changed:  # reglas recursivas: punto fijo
        changed = False
        for head, keys in direct.items():
            current = support[head]
            size = len(current)
            for key in keys:
                if key >= 0 and derived >> key & 1 and key != head:
                    current |= support[key]
            changed |= len(current) != size
    return {head: frozenset(keys) for head, keys in support.items()}


def _expand_derived(keys: set[int], support: Mapping[int, frozenset[int]]) -> set[int]:
    """``keys`` con cada hecho derivado sustituido por aquello de lo que depende."""
    if not support:
        return keys
    result: set[int] = set()
    for key in keys:
        result |= support.get(key, {key})
    return result


def _linear_parts(
    kept: list[tuple[GroundAction, int, int]],
    instances: list[DurativeInstance],
    duration_vars: dict[int, int],
    initial: tuple[float | None, ...],
) -> tuple[LinearModel | None, list[GroundAction], list[GroundAction], list[DurativeInstance]]:
    """Separa las condiciones y efectos sobre variables lineales (ver
    :mod:`pyplanner.grounding.temporal_lp`).

    Returns:
        El modelo lineal (``None`` si no hay variables lineales), las acciones para la
        búsqueda (sin esas condiciones y efectos), las acciones para la heurística (vacías
        si no hay variables lineales) y las acciones durativas (sin las partes lineales de
        sus invariantes y duraciones).
    """
    actions = [action for action, _, _ in kept]
    durations = frozenset(duration_vars)
    # Variables lineales: las de los efectos continuos y, por punto fijo, las que reciben
    # valores que dependen de ?duration o de otras variables lineales.
    linear_vars = {r.variable for d in instances for r in d.continuous}
    if not linear_vars and not durations:
        return None, actions, [], instances
    changed = True
    while changed:
        changed = False
        for action in actions:
            for assignment in _assignments(action):
                if assignment.variable not in linear_vars and variables_of(assignment.value) & (
                    linear_vars | durations
                ):
                    linear_vars.add(assignment.variable)
                    changed = True
    symbolic = frozenset(linear_vars) | durations

    def reads_symbolic(conditions: Iterable[Condition]) -> bool:
        return any(variables_of(c.expression) & symbolic for d in conditions for c in d.numeric)

    def unsupported(what: str, where: object, reason: str = "") -> UnsupportedFeatureError:
        return UnsupportedFeatureError(
            f"{where}: {what} sobre variables que cambian de forma continua o dependen de "
            f"?duration no está soportado{reason}"
        )

    milp = " (haría falta programación lineal entera)"
    conditions: list[tuple[NumericCondition, ...]] = []
    effects: list[tuple[NumericAssignment, ...]] = []
    conditional: list[tuple[tuple[Condition, tuple[NumericAssignment, ...]], ...]] = []
    search_actions: list[GroundAction] = []
    for action, _, _ in kept:
        lp_conditions = tuple(
            c for c in action.numeric_pre if variables_of(c.expression) & symbolic
        )
        if any(c.linear is None for c in lp_conditions):
            raise unsupported("una condición no lineal", action)
        lp_effects = tuple(a for a in action.numeric if a.variable in linear_vars)
        if any(linear(a.value) is None for a in lp_effects):
            raise unsupported("un efecto no lineal", action)
        classical_conditional = []
        lp_conditional = []
        for effect in action.conditional:
            if reads_symbolic([effect.condition]):
                raise unsupported("la condición de un efecto condicional", action, milp)
            lp_part = tuple(a for a in effect.numeric if a.variable in linear_vars)
            if any(linear(a.value) is None for a in lp_part):
                raise unsupported("un efecto no lineal", action)
            if lp_part:
                lp_conditional.append((effect.condition, lp_part))
            rest = tuple(a for a in effect.numeric if a.variable not in linear_vars)
            if effect.add_facts or effect.del_facts or rest:
                classical_conditional.append(replace(effect, numeric=rest))
        conditions.append(lp_conditions)
        effects.append(lp_effects)
        conditional.append(tuple(lp_conditional))
        search_actions.append(
            replace(
                action,
                numeric_pre=tuple(c for c in action.numeric_pre if c not in lp_conditions),
                numeric=tuple(a for a in action.numeric if a not in lp_effects),
                conditional=tuple(classical_conditional),
            )
        )

    invariants: list[tuple[NumericCondition, ...]] = []
    invariant_variables: list[frozenset[int]] = []
    bounds: list[tuple[tuple[Bound, ...], tuple[Bound, ...]]] = []
    timed: set[int] = set(duration_vars.values())
    classical_instances = []
    for index, instance in enumerate(instances):
        clauses = instance.invariant or ()
        lp_invariant = tuple(
            c for clause in clauses for c in clause.numeric if variables_of(c.expression) & symbolic
        )
        if lp_invariant and len(clauses) > 1:
            raise unsupported("una invariante disyuntiva", instance.name, milp)
        if any(c.linear is None for c in lp_invariant):
            raise unsupported("una invariante no lineal", instance.name)
        for rate in instance.continuous:
            if variables_of(rate.rate) & symbolic:
                raise unsupported("un ritmo continuo", instance.name, " (ritmo × tiempo)")
            if rate.condition is not None and reads_symbolic(rate.condition):
                raise unsupported("la condición de un efecto continuo", instance.name, milp)
        for watch in instance.watches:
            if reads_symbolic((*(watch.start or ()), *watch.over_all)):
                raise unsupported("la condición de un efecto condicional", instance.name, milp)
        lp_bounds = tuple(
            tuple(b for b in group if variables_of(b.value) & symbolic)
            for group in (instance.start_bounds, instance.end_bounds)
        )
        if any(lp_bounds):
            timed.add(index)
        invariants.append(lp_invariant)
        invariant_variables.append(
            frozenset(v for c in lp_invariant for v in variables_of(c.expression))
        )
        bounds.append(lp_bounds)  # type: ignore[arg-type]
        classical_instances.append(
            replace(
                instance,
                invariant=_classical_invariant(instance.invariant, symbolic),
                start_bounds=tuple(b for b in instance.start_bounds if b not in lp_bounds[0]),
                end_bounds=tuple(b for b in instance.end_bounds if b not in lp_bounds[1]),
            )
        )

    # Heurística: las acciones completas, con ?duration sustituido por una duración
    # representativa y los efectos continuos integrados al terminar.
    relaxation: list[GroundAction] = []
    for action, kind_code, durative_index in kept:
        numeric = list(action.numeric)
        if durative_index >= 0:
            instance = instances[durative_index]
            length = _representative_duration(instance)
            replacements = dict.fromkeys(durations, length)
            numeric = [
                NumericAssignment(a.variable, a.op, _replace_vars(a.value, replacements))
                for a in numeric
            ]
            if kind_code == END_SNAP:
                for rate in instance.continuous:
                    op = "increase" if rate.sign > 0 else "decrease"
                    numeric.append(
                        NumericAssignment(rate.variable, op, Op("*", (rate.rate, length)))
                    )
        relaxation.append(replace(action, numeric=tuple(numeric)))

    model = LinearModel(
        variables=frozenset(linear_vars),
        durations=duration_vars,
        conditions=tuple(conditions),
        effects=tuple(effects),
        conditional=tuple(conditional),
        invariants=tuple(invariants),
        invariant_variables=tuple(invariant_variables),
        bounds=tuple(bounds),
        timed=frozenset(timed),
        initial=initial,
    )
    return model, search_actions, relaxation, classical_instances


def _ground_continuous(
    item: _Compiled, binding: Binding, context: GroundingContext
) -> tuple[ContinuousRate, ...] | None:
    """Efectos continuos de una acción instanciada (los de ``forall``, expandidos). ``None``
    si un ritmo usa una función estática sin valor."""
    rates = []
    for continuous in item.continuous:
        for inner in context.normalizer.expand(continuous.parameters, binding):
            target = context.normalizer.expression(continuous.effect.fluent, inner)
            if not isinstance(target, Var) or target.key not in context.var_index:
                continue  # la función no forma parte del estado (p. ej. solo la métrica)
            try:
                rate = context.expression(continuous.effect.rate, inner)
            except KeyError as error:
                raise UnsupportedFeatureError(
                    f"acción durativa '{item.action.name}': el ritmo de un efecto continuo "
                    "dentro de 'forall' solo puede usar funciones que no cambian"
                ) from error
            if rate is None:
                return None
            condition = (
                None
                if continuous.condition is None
                else context.conditions(continuous.condition, inner)
            )
            sign = 1.0 if continuous.effect.op == "increase" else -1.0
            rates.append(ContinuousRate(context.var_index[target.key], sign, rate, condition))
    return tuple(rates)


def _ground_watches(
    item: _Compiled,
    args: tuple[str, ...],
    binding: Binding,
    context: GroundingContext,
    fact_index: Mapping[Atom, int],
) -> tuple[Watch, ...]:
    """Vigilantes de una acción instanciada (los de ``forall``, expandidos). Uno cuyo hecho
    no es alcanzable nunca se activa y se descarta."""
    watches = []
    for watch in item.watches:
        for inner in context.normalizer.expand(watch.parameters, binding):
            values = (*args, *(inner[p.name] for p in watch.parameters))
            flag = fact_index.get(Atom(watch.flag.name, tuple(Constant(v) for v in values)))
            if flag is None:
                continue
            start = None if watch.start is None else context.conditions(watch.start, inner)
            watches.append(Watch(flag, start, context.conditions(watch.over_all, inner)))
    return tuple(watches)


def _assignments(action: GroundAction) -> list[NumericAssignment]:
    return [*action.numeric, *(a for effect in action.conditional for a in effect.numeric)]


def _representative_duration(instance: DurativeInstance) -> NumExpr:
    """Duración que supone la heurística: la fija, la mínima, la máxima o 1."""
    for comparator in ("=", ">=", "<="):
        for bound in instance.start_bounds:
            if bound.comparator == comparator:
                return bound.value
    return Const(1.0)


def _replace_vars(expression: NumExpr, replacements: dict[int, NumExpr]) -> NumExpr:
    if isinstance(expression, Const):
        return expression
    if isinstance(expression, Var):
        return replacements.get(expression.key, expression)
    return Op(expression.op, tuple(_replace_vars(a, replacements) for a in expression.args))


def _classical_invariant(
    invariant: tuple[Condition, ...] | None, symbolic: frozenset[int]
) -> tuple[Condition, ...] | None:
    """La invariante sin sus condiciones sobre variables lineales (las comprueba el
    programa lineal)."""
    if invariant is None:
        return None
    return tuple(
        Condition(
            clause.pos_facts,
            clause.neg_facts,
            tuple(c for c in clause.numeric if not variables_of(c.expression) & symbolic),
        )
        for clause in invariant
    )


def _ground_bounds(
    constraints: tuple[DurationConstraint, ...], binding: Binding, context: GroundingContext
) -> tuple[Bound, ...] | None:
    result = []
    for constraint in constraints:
        value = context.expression(constraint.value, binding)
        if value is None:
            return None
        result.append(Bound(constraint.comparator, value))
    return tuple(result)


def _condition_keys(condition: Condition) -> set[int]:
    keys = {*condition.pos_facts, *condition.neg_facts}
    for numeric in condition.numeric:
        keys |= {-(v + 1) for v in variables_of(numeric.expression)}
    return keys


def _snap_info(
    action: GroundAction,
    kind: int,
    durative: int,
    instance: DurativeInstance | None,
    rate_vars: tuple[int, ...] = (),
) -> SnapInfo:
    reads = {*action.pre_facts, *action.neg_facts}
    for condition in action.numeric_pre:
        reads |= {-(v + 1) for v in variables_of(condition.expression)}
    writes = {*action.add_facts, *action.del_facts}
    assignments = list(action.numeric)
    for effect in action.conditional:
        reads |= _condition_keys(effect.condition)
        writes |= {*effect.add_facts, *effect.del_facts}
        assignments.extend(effect.numeric)
    for assignment in assignments:
        writes.add(-(assignment.variable + 1))
        reads |= {-(v + 1) for v in variables_of(assignment.value)}
    if kind in (START_SNAP, END_SNAP):
        writes |= {-(v + 1) for v in rate_vars}  # empieza o acaba un cambio continuo
    if instance is not None:
        reads |= set(instance.invariant_reads)
        bounds = instance.start_bounds if kind == START_SNAP else instance.end_bounds
        for bound in bounds:
            reads |= {-(v + 1) for v in variables_of(bound.value)}
    return SnapInfo(kind, durative, tuple(sorted(reads)), tuple(sorted(writes)))
