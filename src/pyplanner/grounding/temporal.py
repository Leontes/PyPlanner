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
* los que leyeron el valor anterior de lo que cambia de valor.

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

**Todavía no admitido** (error claro, iteración 3b): efectos continuos, ``?duration`` en
acciones con duración no fija o dependiente de funciones que cambian, y efectos
condicionales con condiciones ``over all``.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from functools import cached_property

from pyplanner.errors import UnsupportedFeatureError
from pyplanner.grounding.grounder import Binding, GroundingContext, ground_with_context
from pyplanner.grounding.numeric import NumExpr, evaluate, variables_of
from pyplanner.grounding.task import Condition, GroundAction, GroundTask, State, bits, mask_of
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
    Forall,
    ForallEffect,
    Formula,
    Imply,
    Not,
    NumericEffect,
    Or,
    Parameter,
    Predicate,
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

#: Tipos de paso: acción instantánea, inicio y fin de una acción durativa.
INSTANT, START_SNAP, END_SNAP = 0, 1, 2


# --- Compilación en acciones de inicio y fin ----------------------------------------


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


def _compile(action: DurativeAction, changing: frozenset[str]) -> tuple[_Compiled, list[Predicate]]:
    """Compila una acción durativa. ``changing``: funciones que alguna acción modifica."""
    name = action.name
    parameters = action.parameters
    running = Predicate(f"running@{name}", parameters)
    running_atom = Atom(running.name, tuple(p.variable for p in parameters))

    substitute = _duration_value(action, changing)
    conditions = _split_condition(action.condition)
    flags: list[Predicate] = []
    effects, continuous = _split_effect(action.effect, action, (), flags)
    if continuous:
        raise UnsupportedFeatureError(
            f"acción durativa '{name}': los efectos continuos todavía no están soportados"
        )

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
    compiled = _Compiled(action, start, end, running, invariant, start_bounds, end_bounds)
    return compiled, [running, *flags]


def _duration_value(action: DurativeAction, changing: frozenset[str]) -> Expression | None:
    """Expresión por la que se sustituye ``?duration`` en condiciones y efectos, o ``None``
    si la acción no lo usa.

    Solo se admite con una duración fija (``(= ?duration e)``) cuyas funciones no cambian:
    así el valor de ``e`` es el mismo al empezar y al terminar.
    """
    if not _uses_duration_var(action.condition) and not _uses_duration_var(action.effect):
        return None
    fixed = [c for c in action.duration if c.comparator == "=" and c.time in (None, START)]
    if len(action.duration) != 1 or not fixed:
        raise UnsupportedFeatureError(
            f"acción durativa '{action.name}': ?duration en condiciones o efectos solo está "
            "soportado con una duración fija (= ?duration <expresión>)"
        )
    value = fixed[0].value
    if set(_functions_of(value)) & changing:
        raise UnsupportedFeatureError(
            f"acción durativa '{action.name}': ?duration en condiciones o efectos solo está "
            "soportado si la duración no depende de funciones que cambian"
        )
    return value


def _split_condition(condition: DurativeCondition) -> dict[str, list[Formula]]:
    """Separa una condición en sus partes ``at start``, ``over all`` y ``at end``."""
    parts: dict[str, list[Formula]] = {START: [], ALL: [], END: []}
    if isinstance(condition, TimedCondition):
        parts[condition.time].append(condition.arg)
    elif isinstance(condition, DurativeAnd):
        for arg in condition.args:
            for time, formulas in _split_condition(arg).items():  # type: ignore[arg-type]
                parts[time].extend(formulas)
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
) -> tuple[dict[str, list[Effect]], list[ContinuousEffect]]:
    """Separa un efecto en sus partes ``at start`` y ``at end`` y sus efectos continuos.

    ``quantified``: variables de los ``forall`` que lo rodean (para los hechos auxiliares
    de los efectos condicionales). ``flags`` recibe los predicados auxiliares nuevos.
    """
    parts: dict[str, list[Effect]] = {START: [], END: []}
    continuous: list[ContinuousEffect] = []
    if isinstance(effect, TimedEffect):
        parts[effect.time].append(effect.arg)
    elif isinstance(effect, ContinuousEffect):
        continuous.append(effect)
    elif isinstance(effect, DurativeAnd):
        for arg in effect.args:
            inner, more = _split_effect(arg, action, quantified, flags)  # type: ignore[arg-type]
            for time, effects in inner.items():
                parts[time].extend(effects)
            continuous.extend(more)
    elif isinstance(effect, DurativeForallEffect):
        inner, more = _split_effect(effect.arg, action, (*quantified, *effect.parameters), flags)
        for time, effects in inner.items():
            if effects:
                parts[time].append(ForallEffect(effect.parameters, And(tuple(effects))))
        continuous.extend(more)
    else:
        _split_when(effect, action, quantified, flags, parts, continuous)
    return parts, continuous


def _split_when(
    effect: DurativeWhen,
    action: DurativeAction,
    quantified: tuple[Parameter, ...],
    flags: list[Predicate],
    parts: dict[str, list[Effect]],
    continuous: list[ContinuousEffect],
) -> None:
    """Efecto condicional de una acción durativa (ver la documentación del módulo)."""
    conditions = _split_condition(effect.condition)
    if conditions[ALL]:
        raise UnsupportedFeatureError(
            f"acción durativa '{action.name}': los efectos condicionales con condiciones "
            "'over all' todavía no están soportados"
        )
    inner, more = _split_effect(effect.arg, action, quantified, flags)
    if more:
        raise UnsupportedFeatureError(
            f"acción durativa '{action.name}': los efectos continuos todavía no están soportados"
        )
    start_condition = And(tuple(conditions[START]))
    if inner[START]:  # el loader garantiza que solo depende de condiciones 'at start'
        parts[START].append(When(start_condition, And(tuple(inner[START]))))
    if inner[END]:
        end_condition: list[Formula] = list(conditions[END])
        if conditions[START]:
            # Se recuerda al empezar si la condición 'at start' se cumplía.
            parameters = (*action.parameters, *quantified)
            flag = Predicate(f"cond@{action.name}@{len(flags)}", parameters)
            flags.append(flag)
            atom = Atom(flag.name, tuple(p.variable for p in parameters))
            parts[START].append(When(start_condition, atom))
            parts[END].append(Not(atom))
            end_condition.append(atom)
        parts[END].append(When(And(tuple(end_condition)), And(tuple(inner[END]))))


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


@dataclass(frozen=True)
class SnapInfo:
    """Lo que necesita la línea temporal de cada acción de la tarea.

    Los hechos se identifican por su índice ``i ≥ 0`` y las variables numéricas por
    ``-(v + 1)``, para guardarlos en los mismos diccionarios.

    Attributes:
        kind: :data:`INSTANT`, :data:`START_SNAP` o :data:`END_SNAP`.
        durative: índice de la :class:`DurativeInstance` (``-1`` en las instantáneas).
        reads: hechos y variables que lee (precondiciones, condiciones de efectos,
            valores de los efectos numéricos, invariantes y duraciones).
        writes: hechos y variables que puede cambiar.
    """

    kind: int
    durative: int
    reads: tuple[int, ...]
    writes: tuple[int, ...]


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
    """

    __slots__ = ("earliest", "out", "changer", "readers", "running")

    def __init__(
        self,
        earliest: tuple[float, ...] = (0.0,),
        out: tuple[tuple[tuple[int, float], ...], ...] = ((),),
        changer: dict[int, int] | None = None,
        readers: dict[int, tuple[int, ...]] | None = None,
        running: dict[int, tuple[int, float, float]] | None = None,
    ) -> None:
        self.earliest = earliest
        self.out = out
        self.changer = changer or {}
        self.readers = readers or {}
        self.running = running or {}

    @property
    def makespan(self) -> float:
        """Tiempo del último paso."""
        return max(self.earliest)


class TemporalState(State):
    """Estado temporal: el estado clásico y su :class:`Timeline`.

    Se compara y se usa como clave solo por los hechos y los valores (memoización al
    estilo de POPF).
    """

    timeline: Timeline

    def __new__(
        cls,
        facts: int,
        values: tuple[float | None, ...] = (),
        timeline: Timeline | None = None,
    ) -> "TemporalState":
        self = super().__new__(cls, facts, values)
        self.timeline = timeline if timeline is not None else Timeline()
        return self


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
    """

    durative: tuple[DurativeInstance, ...] = ()
    snaps: tuple[SnapInfo, ...] = ()
    metric_is_makespan: bool = False
    epsilon: float = EPSILON

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
        running_of = self.running_of
        durative = self.durative
        for fact in bits(successor.facts & self.running_mask):
            invariant = durative[running_of[fact]].invariant
            if invariant is not None and not any(c.holds(successor) for c in invariant):
                return None
        timeline = state.timeline if isinstance(state, TemporalState) else Timeline()
        extended = self._extend(timeline, state, successor, self.snaps[action.index])
        if extended is None:
            return None
        return TemporalState(successor.facts, successor.values, extended)

    def _extend(
        self, timeline: Timeline, before: State, after: State, snap: SnapInfo
    ) -> Timeline | None:
        """Añade el paso a la línea temporal; ``None`` si la red temporal es inconsistente."""
        epsilon = self.epsilon
        step = len(timeline.earliest)
        incoming: dict[int, float] = {}

        def after_step(i: int, weight: float) -> None:
            if i != step and incoming.get(i, -math.inf) < weight:
                incoming[i] = weight

        changer = dict(timeline.changer)
        readers = dict(timeline.readers)
        for key in snap.reads:
            previous = changer.get(key)
            if previous is not None:
                after_step(previous, epsilon)
        flipped = before.facts ^ after.facts
        for key in snap.writes:
            previous = changer.get(key)
            if previous is not None:
                after_step(previous, epsilon)
            changes = (flipped >> key) & 1 if key >= 0 else True
            if changes:
                for reader in readers.get(key, ()):
                    after_step(reader, epsilon)
                readers[key] = ()
            changer[key] = step
        for key in snap.reads:
            readers[key] = (*readers.get(key, ()), step)

        running = timeline.running
        start = -1
        upper = math.inf
        if snap.kind == START_SNAP:
            instance = self.durative[snap.durative]
            bounds = _bounds(instance.start_bounds, before.values, epsilon)
            if bounds is None:
                return None
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

        earliest = [*timeline.earliest, 0.0]
        time = max((earliest[i] + w for i, w in incoming.items()), default=0.0)
        earliest[step] = max(time, 0.0)
        out = [*timeline.out, ()]
        for i, weight in incoming.items():
            out[i] = (*out[i], (step, weight))
        if start >= 0 and upper < math.inf:
            out[step] = ((start, -upper),)
            if earliest[start] < earliest[step] - upper - _TOLERANCE:
                # El inicio tiene que retrasarse: se propaga, y si eso retrasa el fin, hay
                # un ciclo positivo (la duración máxima no se puede cumplir).
                earliest[start] = earliest[step] - upper
                _propagate(earliest, out, start, skip=step)
                again = max(earliest[i] + w for i, w in incoming.items())
                if again > earliest[step] + _TOLERANCE:
                    return None
        return Timeline(tuple(earliest), tuple(out), changer, readers, running)

    def schedule(self, plan: Sequence[int]) -> tuple[list[TemporalStep], float]:
        """Planificación de un plan (índices de acciones): los pasos ordenados por tiempo y
        la duración total. El plan tiene que ser válido."""
        state: State = self.init
        for index in plan:
            successor = self.apply(state, self.actions[index])
            assert successor is not None, "el plan no es aplicable"
            state = successor
        assert isinstance(state, TemporalState)
        times = state.timeline.earliest
        steps: list[tuple[float, int, TemporalStep]] = []
        open_starts: dict[int, tuple[int, float]] = {}
        for position, index in enumerate(plan, start=1):
            snap = self.snaps[index]
            action = self.actions[index]
            time = times[position]
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
        return [step for _, _, step in steps], state.timeline.makespan


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


def ground_temporal(domain: Domain, problem: Problem) -> TemporalTask:
    """Instancia un dominio con acciones durativas (ver la documentación del módulo)."""
    changing = _modified_functions(domain)
    compiled: list[_Compiled] = []
    predicates: list[Predicate] = []
    for durative_action in domain.durative_actions:
        item, extra = _compile(durative_action, changing)
        compiled.append(item)
        predicates.extend(extra)
    snap_domain = replace(
        domain,
        actions=(*domain.actions, *(a for c in compiled for a in (c.start, c.end))),
        predicates=(*domain.predicates, *predicates),
        durative_actions=(),
    )
    extra_reads = {}
    for item in compiled:
        extra_reads[item.start.name] = tuple(c.value for c in item.start_bounds)
        extra_reads[item.end.name] = tuple(c.value for c in item.end_bounds)
    # La métrica temporal (total-time) no es aditiva: se busca con coste unitario.
    task, context = ground_with_context(snap_domain, replace(problem, metric=None), extra_reads)

    by_name = {item.action.name: item for item in compiled}
    groups: dict[tuple[str, tuple[str, ...]], dict[str, list[GroundAction]]] = {}
    for action in task.actions:
        base, _, kind = action.name.partition("@")
        if kind in (START, END):
            groups.setdefault((base, action.args), {START: [], END: []})[kind].append(action)

    running_facts = {item.running.name for item in compiled}
    running_indices = frozenset(
        i for i, fact in enumerate(task.facts) if fact.predicate in running_facts
    )

    instances: list[DurativeInstance] = []
    instance_of: dict[tuple[str, tuple[str, ...]], int] = {}
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
        running = task.fact_index[Atom(item.running.name, tuple(Constant(a) for a in args))]
        reads: set[int] = set()
        for condition in invariant or ():
            reads |= _condition_keys(condition)
        instance_of[(base, args)] = len(instances)
        instances.append(
            DurativeInstance(
                base, args, running, invariant, start_bounds, end_bounds, tuple(sorted(reads))
            )
        )

    actions: list[GroundAction] = []
    snaps_info: list[SnapInfo] = []
    for action in task.actions:
        base, _, kind = action.name.partition("@")
        if not kind:
            info_kind, durative_index = INSTANT, -1
        elif (base, action.args) in instance_of:
            info_kind = START_SNAP if kind == "start" else END_SNAP
            durative_index = instance_of[(base, action.args)]
        else:
            continue
        new = replace(action, index=len(actions))
        actions.append(new)
        instance = instances[durative_index] if durative_index >= 0 else None
        snaps_info.append(_snap_info(new, info_kind, durative_index, instance, running_indices))

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
    running: frozenset[int],
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
    if instance is not None:
        reads |= set(instance.invariant_reads)
        bounds = instance.start_bounds if kind == START_SNAP else instance.end_bounds
        for bound in bounds:
            reads |= {-(v + 1) for v in variables_of(bound.value)}
    return SnapInfo(kind, durative, tuple(sorted(reads - running)), tuple(sorted(writes - running)))
