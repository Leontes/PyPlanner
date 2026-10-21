"""Tareas con procesos y eventos (PDDL+): tiempo discretizado.

El motor sigue a ENHSP (Scala et al.), DiNo y UPMurphi: el tiempo avanza a saltos. También
resuelve las tareas temporales sin procesos que el motor temporal no admite por sus efectos
continuos (ritmos que dependen de variables que cambian de forma continua, por ejemplo):
:func:`pyplanner.grounding.grounder.ground` las pasa aquí.

* **Pasos de la búsqueda.** Las acciones instantáneas del dominio, el inicio y el fin de
  cada acción durativa, y dos esperas: ``_wait`` y ``_wait-until``.
* **Esperar** (``_wait``) avanza δ unidades (``delta``, 1 por defecto), o menos si antes
  llega el final de una acción durativa (para que pueda terminar justo a su duración).
  Durante la espera, los procesos activos al empezarla cambian las funciones (ver
  :mod:`pyplanner.grounding.flows`, integración exacta si la dinámica es polinómica).
* **Esperar hasta el siguiente cambio** (``_wait-until``) avanza hasta el primer instante en
  que se cumple una condición numérica de una acción o un evento aplicables por sus hechos
  o del objetivo, una acción durativa en curso llega a una cota de su duración o cambian
  los procesos activos (:meth:`DiscreteTask.wait_until`). Las condiciones
  se buscan con las raíces exactas del flujo (:meth:`Flow.crossings
  <pyplanner.grounding.flows.Flow.crossings>`), y una invariante que se rompe por el
  camino invalida el paso. Así, juntar recursos a un ritmo lento es un paso y no miles de
  esperas de δ: el plan crece con las decisiones, no con el tiempo. Sin flujo exacto, o si
  no se va a cumplir nada, no se puede aplicar.
* **Tras cada cambio**, los **eventos** se disparan en cascada (en orden, hasta que ninguno
  se cumple), se recalculan los predicados derivados y se comprueban las **invariantes**
  (las condiciones ``over all`` de las acciones durativas en curso): si alguna no se
  cumple, el paso no es válido.
* **Eventos en su instante exacto.** Si durante una espera se cumple la precondición de un
  evento (o cambia qué procesos están activos), el instante se busca por bisección y el
  evento se dispara allí, como en VAL; la espera sigue después hasta completar su
  duración. Sin esto, un evento se detectaría al final de la espera, tarde: un depósito
  podría quedar con combustible negativo y otro recibir de más, y la búsqueda acabaría
  aprovechando ese error.
* **Una acción instantánea** va seguida de una espera de ε (0,01, la tolerancia de VAL):
  VAL no admite dos acciones en el mismo instante, y durante ese ε los procesos siguen
  actuando, igual que en VAL.

**Acciones durativas** (en dominios PDDL+): se compilan, a nivel de AST, en

* ``a@start``: condiciones ``at start`` y ``¬running@a``; efectos ``at start``,
  ``running@a``, el reloj ``clock@a := 0`` y las cotas de la duración que se evalúan al
  empezar (``bound@a@k``);
* un proceso ``a@run`` mientras ``running@a``: el reloj avanza a ritmo 1 y los efectos
  continuos de la acción actúan;
* ``a@end``: condiciones ``at end`` y ``over all``, ``running@a`` y el reloj dentro de la
  duración; efectos ``at end`` y ``¬running@a``.

``?duration`` vale, en los efectos ``at end``, lo que marca el reloj; en los ``at start`` y
en los ritmos, la duración fija (si no es fija, error claro).

**Grounding.** Los eventos (``e@event``) y los procesos (``p@process``) pasan al grounder
clásico como acciones (un proceso, con sus ritmos como efectos ``increase``): así se
reutilizan la alcanzabilidad, la normalización y el análisis de qué funciones importan.

**Heurística** (:attr:`DiscreteTask.relaxation`): en la relajación, la espera se convierte
en una acción con un efecto condicional por proceso, que aumenta cada variable en su
ritmo × δ (la *h^hbd* numérica cuenta las repeticiones si el ritmo es constante, como el
reloj de una acción durativa). Si el ritmo es una variable que fijan las acciones, el
efecto solo cuenta cuando es positivo (ver :mod:`pyplanner.heuristics.relaxation`): la
relajación tiene que ponerlo en marcha. Los efectos de los eventos sobre hechos son reglas
de coste 0, como los axiomas. ``_wait-until`` no está en la relajación: es útil cuando lo
es ``_wait``.

**Coste:** el tiempo (δ por espera; lo que avanza, por una espera hasta el siguiente
cambio, ver :meth:`DiscreteTask.transition`) y unos 2ε por acción instantánea: ε por el
tiempo que pasa y ε por ser un paso más del plan. Sin ese segundo ε, abrir y cerrar un
grifo una y otra vez costaría lo mismo por unidad de tiempo que esperar, y A* devolvía
planes con decenas de acciones inútiles. Es 1/64 y no 0,02 para que las sumas de costes
sean exactas en coma flotante: con las repeticiones de los ritmos, el ruido de redondeo
rompía los empates de la relajación de forma distinta en cada implementación. La métrica
que se informa es la duración del plan si es ``(minimize (total-time))``.

**Validación** (:meth:`DiscreteTask.check`): el plan se vuelve a simular con un δ más fino.
Si falla (un evento que debía dispararse a mitad de una espera, por ejemplo), el
planificador lo vuelve a buscar con δ/2 (ver :func:`pyplanner.planner.solve`).

**Literales iniciales temporizados**: son eventos de un reloj
global, ``til@clock``, que avanza a ritmo 1 (ver :func:`_timed_literal_events`), así que se
disparan en su instante exacto. Una espera que acaba justo en el instante de un TIL deja
pasar además ε: VAL no admite que una acción lea un hecho en el mismo instante en que un TIL
lo cambia.

**Restricciones de trayectoria**: los monitores de las tareas sin procesos, con
los instantes de ``within``, ``hold-after`` y ``hold-during`` como eventos del mismo reloj.
Como VAL, se comprueban en los sucesos: tras cada acción, y tras una espera solo si un
evento o un TIL han cambiado los hechos (el final de una espera en la que no pasa nada no
es un suceso).

**Efectos condicionales en acciones durativas**: con condiciones ``over all``, un
vigilante (como en el motor temporal) que se apaga en cuanto la condición
deja de cumplirse, tras cada paso; un efecto continuo condicional se decide al empezar: un
hecho recuerda si su condición se cumplía, y un proceso aparte lo aplica mientras la acción
sigue en curso.

**Preferencias**: como en las tareas sin procesos, compiladas en costes, con la
fase final al final de las acciones (tras las esperas), fuera del tiempo: no deja pasar ε
ni dispara eventos. ``_end`` exige además que no haya acciones durativas en curso. La
métrica ya no es la duración: cada espera cuesta lo que un paso (``step_cost``).

**Límites** (error claro): ``always-within``.
"""

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cached_property
from typing import cast

from pyplanner.errors import UnsupportedFeatureError
from pyplanner.grounding.flows import DECIMALS, Flow
from pyplanner.grounding.grounder import (
    Binding,
    GroundingContext,
    ground_with_context,
    has_preferences,
    marker_times,
)
from pyplanner.grounding.monitors import MonitorRule
from pyplanner.grounding.normalize import Fact
from pyplanner.grounding.numeric import (
    Const,
    NumericAssignment,
    NumericCondition,
    NumExpr,
    make,
    negate,
    variables_of,
)
from pyplanner.grounding.task import (
    Condition,
    ConditionalEffect,
    GroundAction,
    GroundTask,
    State,
)
from pyplanner.grounding.temporal import (
    TemporalStep,
    Watch,
    _effect_predicates,
    _final_phase,
    _functions_of,
    _replace_duration,
    _split_condition,
    _split_effect,
    _start_guard,
    _uses_duration_var,
    _Watch,
)
from pyplanner.loader.ast import (
    ALL,
    END,
    START,
    Action,
    And,
    Atom,
    Comparison,
    Constant,
    ContinuousEffect,
    Domain,
    DurationConstraint,
    DurativeAction,
    DurativeAnd,
    DurativeEffect,
    DurativeForallEffect,
    Effect,
    Expression,
    FluentTerm,
    ForallEffect,
    Formula,
    Function,
    Not,
    Number,
    NumericEffect,
    Predicate,
    Problem,
    TimedEffect,
    TotalTime,
    When,
)

#: Paso de tiempo por defecto.
DELTA = 1.0
#: Reloj global de los literales iniciales temporizados (ver _timed_literal_events).
_TIL_CLOCK = "til@clock"
#: Separación entre una acción instantánea y el paso siguiente (la tolerancia de VAL).
EPSILON = 0.01
#: Coste de una acción instantánea (ver la documentación del módulo): unos 2ε, como fracción
#: binaria (1/64) para que las sumas de costes sean exactas en coma flotante.
ACTION_COST = 1 / 64
#: Disparos de eventos seguidos que se admiten tras un cambio (más: eventos en bucle).
MAX_CASCADE = 1000
#: Precisión con la que se busca el instante de un evento (y por debajo de la cual un
#: intervalo de tiempo se considera vacío).
_TIME_TOLERANCE = 1e-9
#: Precisión de la proyección de la heurística (ver :meth:`DiscreteTask.shortfall`).
_PROJECTION_TOLERANCE = 1e-3
#: Nombres de las esperas (``_`` no puede empezar un nombre PDDL).
WAIT = "_wait"
WAIT_UNTIL = "_wait-until"
#: Hasta dónde busca ``_wait-until`` el siguiente cambio.
JUMP_HORIZON = 1e6
#: Márgenes con los que se prueba, tras la raíz, que una condición ya se cumple (con ``>``
#: y el redondeo a 9 decimales, justo en la raíz todavía no).
_CROSSING_MARGINS = (0.0, 1e-9, 1e-7, 1e-5)

#: Tipos de paso: acción instantánea, inicio, fin y espera.
INSTANT, START_SNAP, END_SNAP, WAIT_SNAP = 0, 1, 2, 3

_RUN, _EVENT, _PROCESS = "run", "event", "process"

#: Hechos positivos, hechos negativos e índices de las condiciones numéricas de una acción o
#: de una cláusula del objetivo (ver :meth:`DiscreteTask._first_crossing`).
_Group = tuple[int, int, tuple[int, ...]]


@dataclass(frozen=True)
class GroundProcess:
    """Un proceso instanciado (o el de una acción durativa en curso, ``a@run``).

    Attributes:
        conditions: su precondición en DNF (activo si se cumple alguna cláusula).
        rates: ``(variable, ritmo)``, con el signo del efecto (``decrease`` resta).
    """

    name: str
    args: tuple[str, ...]
    conditions: tuple[Condition, ...]
    rates: tuple[tuple[int, NumExpr], ...]

    def active(self, state: State) -> bool:
        return any(c.holds(state) for c in self.conditions)

    @cached_property
    def numeric(self) -> bool:
        """``True`` si su precondición tiene condiciones numéricas."""
        return any(c.numeric for c in self.conditions)


@dataclass(frozen=True)
class RunningAction:
    """Una acción durativa instanciada.

    Attributes:
        running: el hecho ``running@a``.
        clock: la variable de su reloj (tiempo desde que empezó).
        bounds: variables de las cotas de su duración fijadas al empezar (las esperas se
            acortan para llegar justo a ellas).
        limits: cotas fijas de su duración (que no dependen de nada que cambie), con el
            mismo fin.
        invariant: sus condiciones ``over all`` en DNF (``None`` si no tiene).
        minimum_bounds, minimums: las cotas inferiores de su duración (variables fijadas
            al empezar y valores fijos): hasta ellas tiene que seguir en curso.
        maximum_bounds, maximums: las cotas superiores: como mucho hasta ellas sigue.
        watches: vigilantes de sus efectos condicionales con condiciones ``over all`` (ver
            :class:`pyplanner.grounding.temporal.Watch`).
    """

    name: str
    args: tuple[str, ...]
    running: int
    clock: int
    bounds: tuple[int, ...]
    invariant: tuple[Condition, ...] | None
    limits: tuple[float, ...] = ()
    minimum_bounds: tuple[int, ...] = ()
    minimums: tuple[float, ...] = ()
    maximum_bounds: tuple[int, ...] = ()
    maximums: tuple[float, ...] = ()
    watches: tuple[Watch, ...] = ()

    def invariant_holds(self, state: State) -> bool:
        return self.invariant is None or any(c.holds(state) for c in self.invariant)

    def mandatory(self, state: State) -> float:
        """Lo que le queda como mínimo hasta poder terminar (0 si ya puede)."""
        clock = state.values[self.clock]
        if clock is None:
            return 0.0
        lowest = [state.values[b] for b in self.minimum_bounds]
        lowest += self.minimums
        return max((v - clock for v in lowest if v is not None), default=0.0)

    def latest(self, state: State) -> float | None:
        """Lo que le queda como mucho hasta tener que terminar (``None`` si no tiene
        cota superior)."""
        clock = state.values[self.clock]
        if clock is None:
            return None
        highest = [state.values[b] for b in self.maximum_bounds]
        highest += self.maximums
        remaining = [v - clock for v in highest if v is not None]
        return max(0.0, min(remaining)) if remaining else None


@dataclass(frozen=True)
class DiscreteTask(GroundTask):
    """Tarea con procesos y eventos, con el tiempo discretizado (ver la documentación del
    módulo).

    Attributes:
        delta: paso de tiempo de las esperas.
        epsilon: separación tras cada acción instantánea.
        processes: procesos instanciados (incluidos los de las acciones durativas).
        events: eventos instanciados (una acción por cláusula de su precondición).
        durative: acciones durativas instanciadas.
        kinds: tipo de cada acción de ``actions`` (:data:`INSTANT`, :data:`START_SNAP`…).
        durative_of: acción durativa de cada inicio y fin (-1 en las demás).
        metric_is_makespan: ``True`` si la métrica es ``(minimize (total-time))``.
        til_clock: variable del reloj de los literales iniciales temporizados (-1 si no hay).
        til_times: instantes de los literales iniciales temporizados.
    """

    delta: float = DELTA
    epsilon: float = EPSILON
    processes: tuple[GroundProcess, ...] = ()
    events: tuple[GroundAction, ...] = ()
    durative: tuple[RunningAction, ...] = ()
    kinds: tuple[int, ...] = ()
    durative_of: tuple[int, ...] = ()
    metric_is_makespan: bool = False
    til_clock: int = -1
    til_times: tuple[float, ...] = ()

    @cached_property
    def continuous_variables(self) -> frozenset[int]:
        """Las variables que cambian los procesos."""
        return frozenset(var for p in self.processes for var, _ in p.rates)

    @property
    def wait(self) -> GroundAction:
        """La espera de δ: la última acción, o la última antes de la fase final de las
        preferencias."""
        return self.actions[self._wait_index]

    @property
    def jump(self) -> GroundAction:
        """La espera hasta el siguiente cambio (la anterior a la espera de δ)."""
        return self.actions[self._wait_index - 1]

    @cached_property
    def _wait_index(self) -> int:
        preferences = self.preferences
        return (len(self.actions) if preferences is None else preferences.first_auxiliary) - 1

    # --- Semántica de los pasos ---------------------------------------------------------

    def apply(self, state: State, action: GroundAction) -> State | None:
        """Aplica un paso: una espera, o una acción seguida de su ε (ver la documentación
        del módulo). ``None`` si no es válido."""
        return self.step(state, action)[0]

    def transition(self, state: State, action: GroundAction) -> tuple[State | None, float]:
        """El sucesor y el coste del paso: la espera hasta el siguiente cambio cuesta lo que
        avanza (A* sigue minimizando la duración); los demás pasos, su coste fijo. Con
        preferencias, la métrica no es la duración: cada espera cuesta lo que un paso."""
        successor, elapsed = self.step(state, action)
        if action.index == self.jump.index and self.preferences is None:
            return successor, elapsed
        return successor, action.cost

    def step(self, state: State, action: GroundAction) -> tuple[State | None, float]:
        """Aplica un paso (ver :meth:`apply`) y devuelve también el tiempo que pasa."""
        if self.kinds[action.index] == WAIT_SNAP:
            if action.index == self.jump.index:
                reached, elapsed = self.wait_until(state)
            else:
                length = self.wait_length(state)
                reached, elapsed = self._past_timed_literal(self.elapse(state, length), length)
            if reached is not None and reached.facts == state.facts:
                return reached, elapsed  # sin sucesos: los monitores no lo ven (como VAL)
            return self._monitored(state, reached), elapsed
        preferences = self.preferences
        if preferences is not None and action.index >= preferences.first_auxiliary:
            return action.apply(state), 0.0  # fase final de las preferencias: fuera del tiempo
        successor = action.apply(state)
        if successor is None:
            return None, 0.0
        successor = self.settle(State(successor.facts, _round(successor.values, state)))
        if successor is None:
            return None, 0.0
        if self.kinds[action.index] == START_SNAP:
            successor = self._started(
                state, successor, self.durative[self.durative_of[action.index]]
            )
        return self._monitored(state, self.elapse(successor, self.epsilon)), self.epsilon

    @staticmethod
    def _started(before: State, after: State, running: RunningAction) -> State:
        """Los vigilantes de una acción que empieza (efectos condicionales con condiciones
        ``over all``, como en el motor temporal): se activan si se cumplen sus condiciones (las ``at
        start`` antes de empezar, las ``over all`` después). Mientras dura, :meth:`settle`
        los apaga en cuanto la condición ``over all`` deja de cumplirse."""
        facts = after.facts
        for watch in running.watches:
            if watch.starts(before, after):
                facts |= 1 << watch.flag
            else:
                facts &= ~(1 << watch.flag)
        return after if facts == after.facts else State(facts, after.values)

    def _monitored(self, before: State, after: State | None) -> State | None:
        """Los monitores de las restricciones de trayectoria tras un suceso (una acción, o una
        espera en la que un evento o un TIL cambian los hechos), en el estado al que llega el
        paso; ``None`` si incumple alguna. Como VAL, que solo las comprueba en los sucesos:
        el final de una espera en la que no pasa nada no lo es."""
        if after is None or not self.monitors:
            return after
        return self.monitor_evaluator.update(before, after)

    def wait_length(self, state: State) -> float:
        """Lo que avanza una espera: δ, o menos si antes una acción durativa en curso llega
        a una cota de su duración."""
        length = self.delta
        values = state.values
        for running in self.durative:
            if not state.facts >> running.running & 1:
                continue
            clock = values[running.clock]
            if clock is None:
                continue
            for limit in (*(values[b] for b in running.bounds), *running.limits):
                if limit is not None and clock < limit < clock + length:
                    length = limit - clock
        return round(length, DECIMALS)

    def wait_until(self, state: State) -> tuple[State | None, float]:
        """Espera hasta el siguiente cambio (ver la documentación del módulo): el estado y
        lo que ha avanzado, o ``None`` si no se va a cumplir nada o se rompe una
        invariante por el camino."""
        length = self.jump_length(state)
        if length is None:
            return None, 0.0
        watched = [
            r for r in self.durative if r.invariant is not None and state.facts >> r.running & 1
        ]
        reached, step = self._until_change(state, length, watched, scan=True)
        return self._past_timed_literal(self.settle(reached), step)

    def _past_timed_literal(
        self, state: State | None, elapsed: float
    ) -> tuple[State | None, float]:
        """Si la espera termina justo en el instante de un literal inicial temporizado, deja
        pasar además ε: VAL no admite que una acción lea un hecho en el mismo instante en que
        un TIL lo cambia (como con dos acciones, ver la documentación del módulo)."""
        if state is None or self.til_clock < 0:
            return state, elapsed
        clock = state.values[self.til_clock]
        if clock is None or not any(
            t > 0 and abs(clock - t) < _TIME_TOLERANCE for t in self.til_times
        ):
            return state, elapsed
        return self.elapse(state, self.epsilon), round(elapsed + self.epsilon, DECIMALS)

    def jump_length(self, state: State) -> float | None:
        """Lo que avanzaría ``_wait-until`` con el flujo actual (``None`` si nada): hasta la
        próxima cota de la duración de una acción en curso o la primera condición numérica
        que se cumple. Los eventos y los cambios de procesos los detiene antes
        :meth:`wait_until`."""
        values = state.values
        candidates = []
        for running in self.durative:
            if not state.facts >> running.running & 1:
                continue
            clock = values[running.clock]
            if clock is None:
                continue
            for limit in (*(values[b] for b in running.bounds), *running.limits):
                if limit is not None and limit > clock + _TIME_TOLERANCE:
                    candidates.append(limit - clock)
        horizon = min(candidates, default=JUMP_HORIZON)
        active = self._active(state)
        if active:
            crossing = self._first_crossing(state, self._flow(active), horizon)
            if crossing is not None:
                candidates.append(crossing)
        return round(min(candidates), DECIMALS) if candidates else None

    def _first_crossing(self, state: State, flow: Flow, horizon: float) -> float | None:
        """El primer instante de (0, ``horizon``) en que se cumple una condición numérica
        que ahora no se cumple, de una acción o un evento aplicables por sus hechos o del
        objetivo (``None`` si ninguna, o si el flujo no es exacto)."""
        conditions, variables, groups = self._jump_watch
        facts, values = state.facts, state.values
        changing = flow.rates.keys()
        relevant = {
            i
            for pos, neg, indices in groups
            if facts & pos == pos and not facts & neg
            for i in indices
        }
        times: list[tuple[float, int]] = []
        for i in relevant:
            condition = conditions[i]
            if variables[i].isdisjoint(changing) or condition.holds(values):
                continue
            roots = flow.crossings(condition.expression, values, horizon)
            if roots is None:
                return None
            times.extend((t, i) for t in roots)
        for t, i in sorted(times):
            for margin in _CROSSING_MARGINS:
                instant = float(math.ceil((t + margin) * 10**DECIMALS) / 10**DECIMALS)
                point = flow.advance(values, instant)
                if point is None:
                    return None
                if conditions[i].holds(point):
                    return instant
        return None

    @cached_property
    def _jump_watch(self) -> tuple[list[NumericCondition], list[frozenset[int]], list[_Group]]:
        """Las condiciones numéricas distintas de las acciones, del objetivo y de los
        eventos, sus variables, y por cada una de ellas (o cláusula del objetivo), sus
        hechos (positivos y negativos) y los índices de sus condiciones numéricas."""
        index: dict[NumericCondition, int] = {}
        groups: dict[_Group, None] = {}
        clauses = [
            (a.pre, a.neg, a.numeric_pre)
            for a in self.actions
            if self.kinds[a.index] != WAIT_SNAP and a.numeric_pre
        ]
        clauses += [(c.pos, c.neg, c.numeric) for c in self.goal if c.numeric]
        clauses += [(e.pre, e.neg, e.numeric_pre) for e in self.events if e.numeric_pre]
        for pos, neg, numeric in clauses:
            indices = tuple(sorted({index.setdefault(n, len(index)) for n in numeric}))
            groups[(pos, neg, indices)] = None
        conditions = list(index)
        variables = [frozenset(variables_of(c.expression)) for c in conditions]
        return conditions, variables, list(groups)

    def elapse(self, state: State, duration: float) -> State | None:
        """Deja pasar ``duration`` unidades de tiempo: los procesos activos cambian las
        variables, y los eventos se disparan en el instante exacto en que se cumplen.

        Si a mitad del intervalo se cumple la precondición de un evento o cambia qué
        procesos están activos, el instante del cambio se busca por bisección sobre el flujo
        (exacto si la dinámica es polinómica): allí se disparan los eventos y el tiempo
        sigue con los procesos que queden activos, hasta completar ``duration``.
        """
        current: State | None = self.settle(state)
        remaining = duration
        for _ in range(MAX_CASCADE):
            if current is None:
                return None
            if remaining <= _TIME_TOLERANCE:
                return current
            reached, step = self._until_change(current, remaining)
            if reached is None:
                return None
            current = self.settle(reached)
            remaining = round(remaining - step, DECIMALS)
        return None  # demasiados cambios en una sola espera

    def _until_change(
        self,
        state: State,
        duration: float,
        watched: Sequence[RunningAction] = (),
        tolerance: float = _TIME_TOLERANCE,
        scan: bool = False,
    ) -> tuple[State | None, float]:
        """Avanza hasta ``duration`` o hasta el primer instante en que se cumple un evento,
        cambian los procesos activos o deja de cumplirse la invariante de una de
        ``watched`` (lo que llegue antes). Devuelve el estado y lo que ha avanzado.

        El instante se busca por bisección, que solo ve los cambios que siguen al final
        del tramo: en una espera de δ, los que no duran tanto los descubre la validación
        con un paso menor. Con ``scan`` (tramos largos, como los de la proyección) se
        comprueba antes entre cada dos instantes en que se anula alguna de las condiciones
        numéricas en juego (:meth:`Flow.crossings
        <pyplanner.grounding.flows.Flow.crossings>`; si el flujo es exacto): así se ve una
        invariante que se rompe y se recupera.
        """
        active = self._active(state)
        if not active:
            return state, duration
        flow = self._flow(active)
        changes = self._changes(state, active, watched)
        end = self._advanced(state, flow, duration)
        if end is None:
            return None, duration
        low, high = 0.0, duration
        for time in self._scan(state, flow, duration, active, watched) if scan else ():
            point = self._advanced(state, flow, time)
            if point is None or changes(point):
                high = time
                break
            low = time
        if high == duration and not changes(end):
            return end, duration
        while high - low > tolerance:
            middle = (low + high) / 2
            point = self._advanced(state, flow, middle)
            if point is None or changes(point):
                high = middle
            else:
                low = middle
        # Redondeado hacia arriba: en ese instante el cambio ya se ha producido.
        step = math.ceil(high * 10**DECIMALS) / 10**DECIMALS
        return self._advanced(state, flow, step), step

    def _active(self, state: State) -> frozenset[int]:
        return frozenset(i for i, p in enumerate(self.processes) if p.active(state))

    @staticmethod
    def _advanced(state: State, flow: Flow, duration: float) -> State | None:
        values = flow.advance(state.values, duration)
        return None if values is None else State(state.facts, values)

    def _changes(
        self, before: State, active: frozenset[int], watched: Sequence[RunningAction] = ()
    ) -> Callable[[State], bool]:
        """Comprueba si en un estado posterior a ``before`` se cumple un evento que no se
        cumplía, cambian los procesos activos (``active``) o deja de cumplirse la invariante
        de una de ``watched``. Lo de ``before`` se evalúa una sola vez."""
        # Mientras avanza el tiempo los hechos no cambian: solo pueden cambiar los procesos
        # con condiciones numéricas.
        switching = [(p, i in active) for i, p in enumerate(self.processes) if p.numeric]
        guarded = [r for r in watched if r.invariant_holds(before)]
        events = [e for e in self._waiting(before) if not e.applicable(before)]

        def changes(after: State) -> bool:
            if any(p.active(after) != was for p, was in switching):
                return True
            if any(not r.invariant_holds(after) for r in guarded):
                return True
            return any(e.applicable(after) for e in events)

        return changes

    def _waiting(self, state: State) -> list[GroundAction]:
        """Los eventos cuyos hechos se cumplen: los únicos que pueden dispararse mientras
        avanza el tiempo."""
        facts = state.facts
        return [e for e in self.events if facts & e.pre == e.pre and not facts & e.neg]

    def _scan(
        self,
        state: State,
        flow: Flow,
        duration: float,
        active: frozenset[int],
        watched: Sequence[RunningAction],
    ) -> list[float]:
        """Instantes de (0, ``duration``) en los que comprobar si algo cambia: uno entre
        cada dos en que se anula alguna condición numérica que puede cambiar algo (las de
        cláusulas cuyos hechos se cumplen). Vacía si alguno no se puede calcular."""
        facts = state.facts
        clauses = [c for p in self.processes if p.numeric for c in p.conditions]
        clauses += [c for r in watched for c in r.invariant or ()]
        numeric = [n for c in clauses if facts & c.pos == c.pos and not facts & c.neg
                   for n in c.numeric]  # fmt: skip
        numeric += [n for e in self._waiting(state) for n in e.numeric_pre]
        times: set[float] = set()
        for condition in numeric:
            crossings = flow.crossings(condition.expression, state.values, duration)
            if crossings is None:
                return []
            times.update(crossings)
        bounds = [0.0, *sorted(times), duration]
        return [(a + b) / 2 for a, b in itertools.pairwise(bounds)]

    def settle(self, state: State | None) -> State | None:
        """Dispara los eventos en cascada, recalcula los predicados derivados y comprueba
        las invariantes. ``None`` si alguna no se cumple o si los eventos no se detienen."""
        state = self._cascade(state)
        if state is None:
            return None
        facts = state.facts
        for running in self.durative:
            if not facts >> running.running & 1:
                continue
            if not running.invariant_holds(state):
                return None
            # Los vigilantes se apagan en cuanto su condición 'over all' deja de cumplirse:
            # durante una espera, los hechos solo cambian aquí (con los eventos).
            for watch in running.watches:
                if facts >> watch.flag & 1 and not watch.holds(state):
                    facts &= ~(1 << watch.flag)
        return state if facts == state.facts else State(facts, state.values)

    def _cascade(self, state: State | None) -> State | None:
        """Los eventos en cascada y los predicados derivados (``None`` si los eventos no
        se detienen)."""
        if state is None:
            return None
        if self.axioms:
            state = self.axiom_evaluator.evaluate(state)
        for _ in range(MAX_CASCADE):
            fired = False
            for event in self.events:
                if not event.applicable(state):
                    continue
                successor = event.apply(state)
                if successor is None:
                    return None
                successor = State(successor.facts, _round(successor.values, state))
                if self.axioms:
                    successor = self.axiom_evaluator.evaluate(successor)
                if successor != state:
                    state, fired = successor, True
            if not fired:
                break
        else:
            return None  # los eventos se disparan sin fin
        return state

    # --- Proyección (heurística) -------------------------------------------------------

    def shortfall(self, state: State) -> float:
        """Cuánto tiempo de las acciones durativas en curso no cubriría una espera sin
        hacer nada: se simula el paso del tiempo y, si la invariante de una se rompe antes
        de su final obligatorio (su duración mínima), cuenta lo que le faltaba. En la
        simulación, cada acción en curso termina al llegar a su duración máxima (o a la
        mínima, si no tiene máxima) o al romperse su invariante: su proceso deja de actuar
        (un repostaje se detiene al llenarse el depósito).

        Una condición ``over all`` hay que **mantenerla**, y la relajación de la heurística
        no lo ve (ignora lo que empeora las condiciones, como el consumo de combustible).
        Sumada a la heurística, guía hacia las acciones que retrasan la ruptura (repostar
        antes de que sea tarde) y aleja de las que la adelantan.
        """
        watch: dict[int, float] = {}  # acción con invariante → final obligatorio
        ends: dict[int, float] = {}  # acción → cuándo termina en la simulación
        for i, running in enumerate(self.durative):
            if not state.facts >> running.running & 1:
                continue
            mandatory = running.mandatory(state)
            latest = running.latest(state)
            end = latest if latest is not None else mandatory
            if end > _PROJECTION_TOLERANCE:
                ends[i] = end
            if running.invariant is not None:
                watch[i] = mandatory
        total, elapsed = 0.0, 0.0
        current: State | None = state
        for _ in range(MAX_CASCADE):
            pending = [h for h in watch.values() if h > elapsed + _PROJECTION_TOLERANCE]
            if not pending or current is None:
                break
            stop = min(t for t in (*ends.values(), *pending) if t > elapsed) - elapsed
            watched = [self.durative[i] for i in watch]
            reached, step = self._until_change(
                current, stop, watched, _PROJECTION_TOLERANCE, scan=True
            )
            if reached is None:
                break
            elapsed += step
            current = self._cascade(reached)
            for i, end in list(ends.items()):
                if current is not None and elapsed >= end - _PROJECTION_TOLERANCE:
                    current = self._finish(current, i)
                    del ends[i]
                    watch.pop(i, None)
            for i, horizon in list(watch.items()):
                if current is None or self.durative[i].invariant_holds(current):
                    continue
                if horizon - elapsed > _PROJECTION_TOLERANCE:
                    total += horizon - elapsed
                current = self._finish(current, i)
                ends.pop(i, None)
                del watch[i]
        return round(total, DECIMALS)

    @cached_property
    def _end_actions(self) -> dict[int, tuple[GroundAction, ...]]:
        """Las acciones de fin de cada acción durativa."""
        ends: dict[int, list[GroundAction]] = {}
        for action, kind, durative in zip(self.actions, self.kinds, self.durative_of, strict=True):
            if kind == END_SNAP:
                ends.setdefault(durative, []).append(action)
        return {i: tuple(group) for i, group in ends.items()}

    def _finish(self, state: State, durative: int) -> State | None:
        """Termina una acción durativa en la proyección: con los efectos de su acción de fin
        aunque no se cumplan sus condiciones (al romperse la invariante, por ejemplo), para
        que deje de actuar también lo que dependa de ellos (un proceso que necesita
        ``refueling``). Si no se pueden aplicar, simplemente deja de estar en curso."""
        ends = self._end_actions.get(durative, ())
        action = next((a for a in ends if a.applicable(state)), ends[0] if ends else None)
        if action is not None:
            successor = action.apply(state)
            if successor is not None:
                return self._cascade(State(successor.facts, _round(successor.values, state)))
        running = self.durative[durative].running
        return self._cascade(State(state.facts & ~(1 << running), state.values))

    @cached_property
    def _flows(self) -> dict[frozenset[int], Flow]:
        return {}

    def _flow(self, active: frozenset[int]) -> Flow:
        """El flujo de un conjunto de procesos activos (se calcula una vez por conjunto)."""
        flow = self._flows.get(active)
        if flow is None:
            rates: dict[int, NumExpr] = {}
            for i in sorted(active):
                for var, rate in self.processes[i].rates:
                    rates[var] = make("+", [rates[var], rate]) if var in rates else rate
            flow = self._flows[active] = Flow(rates)
        return flow

    # --- Heurística ---------------------------------------------------------------------

    @cached_property
    def relaxation(self) -> GroundTask:
        """La espera, como una acción con un efecto condicional por cláusula de cada
        proceso (sus ritmos × δ), y los eventos como reglas relajadas."""
        effects: list[ConditionalEffect] = []
        for process in self.processes:
            numeric = tuple(
                NumericAssignment(var, "increase", make("*", [rate, Const(self.delta)]))
                for var, rate in process.rates
            )
            effects.extend(ConditionalEffect(c, (), (), numeric) for c in process.conditions)
        actions = list(self.actions)
        actions[self.wait.index] = replace(self.wait, conditional=tuple(effects))
        rules = []
        for event in self.events:
            base = Condition(event.pre_facts, event.neg_facts, event.numeric_pre)
            if event.add_facts or event.del_facts:
                rules.append(MonitorRule(base, event.add_facts, event.del_facts))
            for effect in event.conditional:
                merged = Condition(
                    tuple(sorted({*base.pos_facts, *effect.condition.pos_facts})),
                    tuple(sorted({*base.neg_facts, *effect.condition.neg_facts})),
                    (*base.numeric, *effect.condition.numeric),
                )
                rules.append(MonitorRule(merged, effect.add_facts, effect.del_facts))
        return GroundTask(
            self.domain_name,
            self.problem_name,
            self.facts,
            tuple(actions),
            self.init,
            self.goal,
            self.variables,
            axioms=self.axioms,
            monitor_rules=(*rules, *self.monitor_rules),
        )

    # --- Plan -----------------------------------------------------------------------------

    def schedule(self, plan: Sequence[int]) -> tuple[list[TemporalStep], float]:
        """El plan en el tiempo: cada acción instantánea en su instante y cada acción
        durativa con su inicio y su duración. Devuelve también la duración total (el
        instante del último paso que no es una espera)."""
        if self.preferences is not None:
            plan = self.preferences.plan_actions(plan)
        state, time, makespan = self.init, 0.0, 0.0
        steps: list[TemporalStep] = []
        started: dict[int, float] = {}
        for index in plan:
            action = self.actions[index]
            kind = self.kinds[index]
            successor, elapsed = self.step(state, action)
            assert successor is not None, "el plan no es aplicable"
            if kind == WAIT_SNAP:
                time = round(time + elapsed, DECIMALS)
            else:
                makespan = time
                durative = self.durative_of[index]
                if kind == START_SNAP:
                    started[durative] = time
                elif kind == END_SNAP:
                    instance = self.durative[durative]
                    start = started.pop(durative)
                    duration = round(time - start, DECIMALS)
                    steps.append(TemporalStep(start, instance.name, instance.args, duration))
                else:
                    steps.append(TemporalStep(time, action.name, action.args, None))
                time = round(time + self.epsilon, DECIMALS)
            state = successor
        steps.sort(key=lambda s: s.time)
        return steps, makespan

    def check(self, plan: Sequence[int], delta: float) -> str | None:
        """Simula el plan con esperas de como mucho ``delta`` (más fino que el de la
        búsqueda). Devuelve ``None`` si sigue siendo válido o la descripción del fallo.

        Cada espera se recorre a trozos de ``delta`` para comprobar los instantes de en
        medio, y termina en el estado de la búsqueda (un solo avance, exacto si el flujo lo
        es): el redondeo de miles de trozos se acumula, y un plan que agota el combustible
        justo al final de una espera larga quedaba en −10⁻⁹."""
        fine = replace(self, delta=delta)
        state: State | None = self.init
        for step, index in enumerate(plan, start=1):
            assert state is not None
            action = self.actions[index]
            if self.kinds[index] == WAIT_SNAP:
                successor, remaining = self.step(state, action)
                middle: State | None = state
                while middle is not None and successor is not None:
                    piece = fine.wait_length(middle)
                    if piece >= remaining - 1e-9:
                        break  # el final es el de la búsqueda
                    middle = fine.elapse(middle, piece)
                    remaining = round(remaining - piece, DECIMALS)
                if middle is None or successor is None:
                    return f"paso {step}: con δ = {delta:g}, la espera rompe una invariante"
                state = successor
                continue
            if not action.applicable(state):
                return f"paso {step}: con δ = {delta:g}, {action} no es aplicable"
            state = fine.apply(state, action)
            if state is None:
                return f"paso {step}: con δ = {delta:g}, {action} rompe una invariante"
        assert state is not None
        if not self.is_goal(state):
            return f"con δ = {delta:g}, el plan no alcanza el objetivo"
        return None

    def unsolvability_caveat(self, lost_paths: bool) -> str | None:
        return f"no hay plan con un paso de tiempo δ = {self.delta:g}; con uno menor podría haberlo"


def _round(
    values: tuple[float | None, ...], before: State | None = None
) -> tuple[float | None, ...]:
    """Los valores redondeados. Con ``before``, el estado del que salen, solo los que han
    cambiado (los demás son los mismos objetos, ya redondeados)."""
    if before is None:
        return tuple(None if v is None else round(v, DECIMALS) for v in values)
    return tuple(
        v if v is b or v is None else round(v, DECIMALS)
        for v, b in zip(values, before.values, strict=True)
    )


# --- Compilación y grounding ---------------------------------------------------------


@dataclass(frozen=True)
class _Durative:
    """Una acción durativa compilada: sus esquemas y lo que hace falta al instanciarla."""

    action: DurativeAction
    running: Predicate
    clock: Function
    bounds: tuple[Function, ...]
    invariant: Formula | None
    limits: tuple[Expression, ...] = ()
    #: Las cotas inferiores y superiores, entre ``bounds`` y ``limits``.
    minimum_bounds: tuple[Function, ...] = ()
    minimums: tuple[Expression, ...] = ()
    maximum_bounds: tuple[Function, ...] = ()
    maximums: tuple[Expression, ...] = ()
    watches: tuple[_Watch, ...] = ()


def _timed_literal_events(
    problem: Problem, times: Sequence[float]
) -> tuple[list[Action], list[Predicate], Function]:
    """Los literales iniciales temporizados como eventos: un reloj global (``til@clock``,
    que un proceso siempre activo hace avanzar a ritmo 1 desde el instante 0) y, por cada
    instante *t_k* de ``times``, un evento ``til@k`` que se dispara cuando el reloj llega a
    *t_k* y aplica sus literales (si los hay) una sola vez (``til-done@k``). Como los demás
    eventos, se dispara en su instante exacto, y una espera hasta el siguiente cambio se
    detiene en él."""
    clock = Function(_TIL_CLOCK, ())
    clock_term = FluentTerm(clock.name, ())
    actions = [
        Action(
            f"{_TIL_CLOCK}@{_RUN}", (), And(()), NumericEffect("increase", clock_term, Number(1.0))
        )
    ]
    predicates: list[Predicate] = []
    for k, time in enumerate(times):
        done = Atom(f"til-done@{k}")
        predicates.append(Predicate(done.predicate))
        effects: list[Effect] = [done]
        for literal in problem.timed_literals:
            if literal.time == time:
                effects.append(literal.atom if literal.positive else Not(literal.atom))
        precondition = And((Not(done), Comparison(">=", clock_term, Number(time))))
        actions.append(Action(f"til@{k}@{_EVENT}", (), precondition, And(tuple(effects))))
    return actions, predicates, clock


def _compile_durative(
    action: DurativeAction, changing: frozenset[str], domain: Domain, volatile: frozenset[str]
) -> tuple[list[Action], _Durative, list[Predicate], list[Function]]:
    """Compila una acción durativa. ``changing``: funciones que algo modifica (las cotas de
    la duración que no dependen de ellas se comparan tal cual, sin función intermedia: así
    la heurística ve cuánto falta). ``volatile``: predicados derivados y los que cambian
    los eventos (que se disparan en el mismo instante del inicio): no entran en la guarda
    del inicio (ver :func:`pyplanner.grounding.temporal._start_guard`)."""
    name, parameters = action.name, action.parameters
    variables = tuple(p.variable for p in parameters)
    running = Predicate(f"running@{name}", parameters)
    running_atom = Atom(running.name, variables)
    clock = Function(f"clock@{name}", parameters)
    clock_term = FluentTerm(clock.name, variables)

    conditions = _split_condition(action.condition)
    flags: list[Predicate] = []
    watches: list[_Watch] = []
    effects, continuous = _split_effect(action.effect, action, (), flags, watches)

    fixed = next((c.value for c in action.duration if c.comparator == "=" and c.time != END), None)
    uses_start = any(_uses_duration_var(e) for e in effects[START])
    uses_rates = any(_uses_duration_var(c.effect.rate) for c in continuous)
    if (uses_start or uses_rates) and fixed is None:
        raise UnsupportedFeatureError(
            f"acción durativa '{name}': ?duration en un efecto 'at start' o continuo necesita "
            "una duración fija"
        )

    start_effects: list[Effect] = [
        *(_effect_with(e, fixed) for e in effects[START]),
        running_atom,
        NumericEffect("assign", clock_term, Number(0.0)),
    ]
    end_effects: list[Effect] = [*(_effect_with(e, clock_term) for e in effects[END])]
    # Los efectos continuos: los incondicionales, en el proceso de la acción; cada uno
    # condicional se decide al empezar (como en el motor temporal): un hecho recuerda si su
    # condición se cumplía, y un proceso aparte lo aplica mientras la acción sigue en curso.
    rates: list[Effect] = [NumericEffect("increase", clock_term, Number(1.0))]
    gated: list[Action] = []
    for k, item in enumerate(continuous):
        rate = _replace_duration(item.effect.rate, fixed)
        numeric = NumericEffect(item.effect.op, item.effect.fluent, rate)  # type: ignore[arg-type]
        if item.condition is None:
            rates.append(ForallEffect(item.parameters, numeric) if item.parameters else numeric)
            continue
        flag_parameters = (*parameters, *item.parameters)
        flag = Predicate(f"rate@{name}@{k}", flag_parameters)
        flags.append(flag)
        atom = Atom(flag.name, tuple(p.variable for p in flag_parameters))
        set_flag: Effect = When(item.condition, atom)
        clear: Effect = Not(atom)
        if item.parameters:
            set_flag = ForallEffect(item.parameters, set_flag)
            clear = ForallEffect(item.parameters, clear)
        start_effects.append(set_flag)
        end_effects.append(clear)
        gated.append(
            Action(f"{name}#{k}@{_RUN}", flag_parameters, And((running_atom, atom)), numeric)
        )
    end_conditions: list[Formula] = [
        *(_formula_with(c, clock_term) for c in conditions[END]),
    ]
    bounds: list[Function] = []
    limits: list[Expression] = []
    minimum_bounds: list[Function] = []
    minimums: list[Expression] = []
    maximum_bounds: list[Function] = []
    maximums: list[Expression] = []
    for constraint in action.duration:
        lower = constraint.comparator in ("=", ">=")
        upper = constraint.comparator in ("=", "<=")
        if constraint.time == END:  # se evalúa al terminar
            end_conditions += _bound_conditions(constraint, clock_term, constraint.value)
            continue
        if not set(_functions_of(constraint.value)) & changing:  # fija
            end_conditions += _bound_conditions(constraint, clock_term, constraint.value)
            limits.append(constraint.value)
            if lower:
                minimums.append(constraint.value)
            if upper:
                maximums.append(constraint.value)
            continue
        bound = Function(f"bound@{name}@{len(bounds)}", parameters)
        bounds.append(bound)
        if lower:
            minimum_bounds.append(bound)
        if upper:
            maximum_bounds.append(bound)
        term = FluentTerm(bound.name, variables)
        start_effects.append(NumericEffect("assign", term, constraint.value))
        end_conditions += _bound_conditions(constraint, clock_term, term)
    invariant = And(tuple(conditions[ALL])) if conditions[ALL] else None
    if invariant is not None:
        end_conditions.append(invariant)
    end_conditions.append(running_atom)

    guard = _start_guard(action, conditions[ALL], effects[START], domain, volatile)
    start = Action(
        f"{name}@start",
        parameters,
        And((And(tuple(conditions[START])), *guard, Not(running_atom))),
        And(tuple(start_effects)),
    )
    end = Action(
        f"{name}@end",
        parameters,
        And(tuple(end_conditions)),
        And((*end_effects, Not(running_atom))),
    )
    run = Action(f"{name}@{_RUN}", parameters, running_atom, And(tuple(rates)))
    compiled = _Durative(
        action,
        running,
        clock,
        tuple(bounds),
        invariant,
        tuple(limits),
        tuple(minimum_bounds),
        tuple(minimums),
        tuple(maximum_bounds),
        tuple(maximums),
        tuple(watches),
    )
    return [start, end, run, *gated], compiled, [running, *flags], [clock, *bounds]


def _watches(
    item: _Durative,
    args: tuple[str, ...],
    binding: Binding,
    context: GroundingContext,
    fact_index: Mapping[Atom, int],
) -> tuple[Watch, ...]:
    """Los vigilantes de una acción instanciada (los de ``forall``, expandidos); uno cuyo
    hecho no es alcanzable nunca se activa y se descarta."""
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


def _variables(
    functions: Sequence[Function], args: tuple[str, ...], var_index: Mapping[Fact, int]
) -> tuple[int, ...]:
    """Las variables de unas funciones de la acción instanciada (las que existen)."""
    return tuple(var_index[(f.name, args)] for f in functions if (f.name, args) in var_index)


def _constants(
    expressions: Sequence[Expression], binding: Binding, context: GroundingContext
) -> tuple[float, ...]:
    """Los valores de unas expresiones fijas (sin las que no se pueden evaluar)."""
    values = (context.expression(e, binding) for e in expressions)
    return tuple(v.value for v in values if isinstance(v, Const))


def _changing_functions(domain: Domain) -> frozenset[str]:
    """Funciones que modifica algo: una acción, una acción durativa, un evento o un
    proceso."""
    result: set[str] = set()

    def visit(node: object) -> None:
        if isinstance(node, NumericEffect | ContinuousEffect):
            result.add(node.fluent.function)
        elif isinstance(node, And | DurativeAnd):
            for arg in node.args:
                visit(arg)
        elif isinstance(node, ForallEffect | When | TimedEffect | DurativeForallEffect):
            visit(node.arg)

    effects: list[object] = [a.effect for a in domain.actions]
    effects += [e.effect for e in domain.events]
    effects += [a.effect for a in domain.durative_actions]
    effects += [p.effect for p in domain.processes]
    for effect in effects:
        visit(effect)
    return frozenset(result)


def _effect_with(effect: Effect, duration: Expression | None) -> Effect:
    """El efecto con ``?duration`` sustituido por ``duration``."""
    return cast(Effect, _replace_duration(effect, duration))


def _formula_with(formula: Formula, duration: Expression | None) -> Formula:
    """La fórmula con ``?duration`` sustituido por ``duration``."""
    return cast(Formula, _replace_duration(formula, duration))


def _bound_conditions(
    constraint: DurationConstraint, clock: FluentTerm, value: Expression
) -> list[Formula]:
    if constraint.comparator == "=":
        return [Comparison(">=", clock, value), Comparison("<=", clock, value)]
    return [Comparison(constraint.comparator, clock, value)]


def _as_effect(effect: DurativeEffect) -> Effect:
    """El efecto de un proceso como efecto clásico: cada efecto continuo, como un
    ``increase``/``decrease`` de su ritmo (para el grounder)."""
    if isinstance(effect, ContinuousEffect):
        return NumericEffect(effect.op, effect.fluent, effect.rate)
    if isinstance(effect, DurativeAnd):
        return And(tuple(_as_effect(arg) for arg in effect.args))  # type: ignore[arg-type]
    assert isinstance(effect, DurativeForallEffect)
    return ForallEffect(effect.parameters, _as_effect(effect.arg))


def ground_discrete(domain: Domain, problem: Problem, delta: float = DELTA) -> DiscreteTask:
    """Instancia un dominio con procesos y eventos (ver la documentación del módulo)."""
    if delta <= 0:
        raise ValueError("el paso de tiempo δ tiene que ser positivo")

    compiled: dict[str, _Durative] = {}
    actions: list[Action] = list(domain.actions)
    predicates: list[Predicate] = list(domain.predicates)
    functions: list[Function] = list(domain.functions)
    changing = _changing_functions(domain)
    volatile = domain.derived_predicates | {
        name for event in domain.events for name in _effect_predicates(event.effect)
    }
    for durative_action in domain.durative_actions:
        schemas, item, extra_predicates, extra_functions = _compile_durative(
            durative_action, changing, domain, volatile
        )
        compiled[durative_action.name] = item
        actions += schemas
        predicates += extra_predicates
        functions += extra_functions
    actions += [
        Action(f"{e.name}@{_EVENT}", e.parameters, e.precondition, e.effect) for e in domain.events
    ]
    # Los TIL y los instantes de las restricciones con tiempo (TIL sin literales: su hecho
    # til-done@k marca que el instante ha pasado, ver pyplanner.grounding.monitors), como
    # eventos de un reloj global.
    til_times = tuple(sorted({literal.time for literal in problem.timed_literals}))
    markers = marker_times(domain, problem)
    clock_times = sorted({*til_times, *markers})
    preferences = has_preferences(domain, problem)
    if clock_times:
        til_actions, til_predicates, til_clock = _timed_literal_events(problem, clock_times)
        actions += til_actions
        predicates += til_predicates
        functions.append(til_clock)
        problem = replace(
            problem,
            values=(*problem.values, (FluentTerm(til_clock.name, ()), 0.0)),
            timed_literals=(),
        )
    actions += [
        Action(f"{p.name}@{_PROCESS}", p.parameters, p.precondition, _as_effect(p.effect))
        for p in domain.processes
    ]
    flat = replace(
        domain,
        actions=tuple(actions),
        predicates=tuple(predicates),
        functions=tuple(functions),
        durative_actions=(),
        processes=(),
        events=(),
    )
    # Con preferencias, la métrica se compila en los costes (con su total-time, que se
    # informa y no se optimiza: la tarea es temporal aunque no tenga marcas).
    task, context = ground_with_context(
        flat,
        problem if preferences else replace(problem, metric=None),
        time_markers={t: f"til-done@{k}" for k, t in enumerate(clock_times) if t in markers},
    )
    if any(m.kind == "always-within" for m in task.monitors):
        raise UnsupportedFeatureError("'always-within' todavía no se admite con procesos y eventos")

    processes: dict[tuple[str, tuple[str, ...]], list[GroundAction]] = {}
    events: list[GroundAction] = []
    snaps: dict[tuple[str, tuple[str, ...]], dict[str, list[GroundAction]]] = {}
    for action in task.actions:
        base, _, kind = action.name.rpartition("@")
        if kind in (_RUN, _PROCESS):
            processes.setdefault((action.name, action.args), []).append(action)
        elif kind == _EVENT:
            events.append(action)
        elif kind in (START, END):
            snaps.setdefault((base, action.args), {START: [], END: []})[kind].append(action)

    ground_processes = []
    for (name, args), clauses in sorted(processes.items()):
        if any(a.conditional or a.add_facts or a.del_facts for a in clauses):
            raise UnsupportedFeatureError(
                f"proceso '{name.rpartition('@')[0]}': solo admite efectos continuos"
            )
        rates = tuple(
            (n.variable, n.value if n.op == "increase" else negate(n.value))
            for n in clauses[0].numeric
        )
        conditions = tuple(Condition(a.pre_facts, a.neg_facts, a.numeric_pre) for a in clauses)
        ground_processes.append(GroundProcess(name, args, conditions, rates))

    var_index = context.var_index
    running_actions: list[RunningAction] = []
    instance_of: dict[tuple[str, tuple[str, ...]], int] = {}
    for (base, args), parts in sorted(snaps.items()):
        if not parts[START] or not parts[END]:
            continue  # nunca puede empezar o nunca puede terminar
        item = compiled[base]
        binding: Binding = {p.name: a for p, a in zip(item.action.parameters, args, strict=True)}
        clock = var_index.get((item.clock.name, args))
        if clock is None:
            continue
        bounds = tuple(
            var_index[(b.name, args)] for b in item.bounds if (b.name, args) in var_index
        )
        invariant = None if item.invariant is None else context.conditions(item.invariant, binding)
        running = task.fact_index[Atom(item.running.name, tuple(Constant(a) for a in args))]
        limits = _constants(item.limits, binding, context)
        instance_of[(base, args)] = len(running_actions)
        running_actions.append(
            RunningAction(
                base,
                args,
                running,
                clock,
                bounds,
                invariant,
                limits,
                _variables(item.minimum_bounds, args, var_index),
                _constants(item.minimums, binding, context),
                _variables(item.maximum_bounds, args, var_index),
                _constants(item.maximums, binding, context),
                _watches(item, args, binding, context, task.fact_index),
            )
        )

    # Las acciones: las del dominio y los inicios y fines, las dos esperas y, con
    # preferencias, la fase final (ver pyplanner.grounding.preferences). Sin preferencias,
    # cada acción cuesta ACTION_COST y cada espera, lo que avanza; con ellas, los costes del
    # grounder (un paso más lo que añaden a la métrica) y lo que un paso por espera.
    model = task.preferences
    first_auxiliary = len(task.actions) if model is None else model.first_auxiliary
    real: list[GroundAction] = []
    kinds: list[int] = []
    durative_of: list[int] = []
    renumbered: dict[int, int] = {}
    for action in task.actions[:first_auxiliary]:
        base, _, kind = action.name.rpartition("@")
        cost = ACTION_COST if model is None else action.cost
        if not base:  # acción del dominio (sin @)
            kinds.append(INSTANT)
            durative_of.append(-1)
        elif kind in (START, END) and (base, action.args) in instance_of:
            kinds.append(START_SNAP if kind == START else END_SNAP)
            durative_of.append(instance_of[(base, action.args)])
        else:
            continue
        renumbered[action.index] = len(real)
        real.append(replace(action, index=len(real), cost=cost))
    for name in (WAIT_UNTIL, WAIT):  # la espera de δ, la última antes de la fase final
        cost = delta if model is None else model.step_cost
        real.append(GroundAction(len(real), name, (), (), (), (), cost=cost))
        kinds.append(WAIT_SNAP)
        durative_of.append(-1)
    if model is not None:
        for action in task.actions[first_auxiliary:]:
            renumbered[action.index] = len(real)
            real.append(replace(action, index=len(real)))
            kinds.append(INSTANT)
            durative_of.append(-1)
        model, real = _final_phase(model, renumbered, real, (r.running for r in running_actions))

    goal = tuple(
        Condition(
            clause.pos_facts,
            tuple(sorted({*clause.neg_facts, *(r.running for r in running_actions)})),
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
    if model is None and metric is not None and not is_makespan:
        notes.append(
            "con procesos y eventos solo se usa la métrica (minimize (total-time)); esta se ignora"
        )
    discrete = DiscreteTask(
        domain_name=task.domain_name,
        problem_name=task.problem_name,
        facts=task.facts,
        actions=tuple(real),
        init=task.init,
        goal=goal,
        variables=task.variables,
        notes=tuple(notes),
        axioms=task.axioms,
        monitors=task.monitors,
        monitor_rules=task.monitor_rules,
        delta=delta,
        processes=tuple(ground_processes),
        events=tuple(events),
        durative=tuple(running_actions),
        kinds=tuple(kinds),
        durative_of=tuple(durative_of),
        metric_is_makespan=is_makespan and model is None,
        metric_is_cost=task.metric_is_cost,
        metric_offset=task.metric_offset,
        metric_sign=task.metric_sign,
        metric_time_weight=task.metric_time_weight,
        preferences=model,
        til_clock=var_index.get((_TIL_CLOCK, ()), -1),
        til_times=til_times,
    )
    # Los relojes empiezan en 0 (aunque cada inicio lo vuelve a poner): con un valor
    # indefinido, la heurística no podría contar las esperas que faltan para el final.
    values = list(task.init.values)
    for instance in running_actions:
        if values[instance.clock] is None:
            values[instance.clock] = 0.0
    # Los eventos que se cumplen al principio se disparan en el instante 0.
    init = discrete._monitored(
        task.init, discrete.settle(State(task.init.facts, _round(tuple(values))))
    )
    if init is None:
        notes.append("el estado inicial rompe una invariante o sus eventos no se detienen")
        return replace(discrete, goal=(), notes=tuple(notes))
    return replace(discrete, init=init)
