"""Tareas con procesos y eventos (PDDL+, iteración 6): tiempo discretizado.

El motor sigue a ENHSP (Scala et al.), DiNo y UPMurphi: el tiempo avanza a saltos. También
resuelve las tareas temporales sin procesos que el motor de la 3b no admite por sus efectos
continuos (ritmos que dependen de variables que cambian de forma continua, por ejemplo):
:func:`pyplanner.grounding.grounder.ground` las pasa aquí.

* **Pasos de la búsqueda.** Las acciones instantáneas del dominio, el inicio y el fin de
  cada acción durativa, y una acción ``_wait`` que deja pasar el tiempo.
* **Esperar** (``_wait``) avanza δ unidades (``delta``, 1 por defecto), o menos si antes
  llega el final de una acción durativa (para que pueda terminar justo a su duración).
  Durante la espera, los procesos activos al empezarla cambian las funciones (ver
  :mod:`pyplanner.grounding.flows`, integración exacta si la dinámica es polinómica).
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
ritmo × δ (la *h^hbd* de la 3a cuenta las repeticiones si el ritmo es constante, como el
reloj de una acción durativa). Los efectos de los eventos sobre hechos son reglas de coste
0, como los axiomas.

**Coste:** el tiempo (δ por espera) y 2ε por acción instantánea: ε por el tiempo que pasa y
ε por ser un paso más del plan. Sin ese segundo ε, abrir y cerrar un grifo una y otra vez
costaría lo mismo por unidad de tiempo que esperar, y A* devolvía planes con decenas de
acciones inútiles. La métrica que se informa es la duración del plan si es
``(minimize (total-time))``.

**Validación** (:meth:`DiscreteTask.check`): el plan se vuelve a simular con un δ más fino.
Si falla (un evento que debía dispararse a mitad de una espera, por ejemplo), el
planificador lo vuelve a buscar con δ/2 (ver :func:`pyplanner.planner.solve`).

**Límites** (error claro): literales iniciales temporizados, restricciones de trayectoria,
preferencias, efectos condicionales que dependen de condiciones ``over all`` y efectos
continuos condicionales.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import cached_property
from typing import cast

from pyplanner.errors import UnsupportedFeatureError
from pyplanner.grounding.flows import DECIMALS, Flow
from pyplanner.grounding.grounder import Binding, ground_with_context
from pyplanner.grounding.monitors import MonitorRule
from pyplanner.grounding.numeric import Const, NumericAssignment, NumExpr, make, negate
from pyplanner.grounding.task import (
    Condition,
    ConditionalEffect,
    GroundAction,
    GroundTask,
    State,
)
from pyplanner.grounding.temporal import (
    TemporalStep,
    _replace_duration,
    _split_condition,
    _split_effect,
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
    TotalTime,
)

#: Paso de tiempo por defecto.
DELTA = 1.0
#: Separación entre una acción instantánea y el paso siguiente (la tolerancia de VAL).
EPSILON = 0.01
#: Coste de una acción instantánea (ver la documentación del módulo).
ACTION_COST = 2 * EPSILON
#: Disparos de eventos seguidos que se admiten tras un cambio (más: eventos en bucle).
MAX_CASCADE = 1000
#: Precisión con la que se busca el instante de un evento (y por debajo de la cual un
#: intervalo de tiempo se considera vacío).
_TIME_TOLERANCE = 1e-9
#: Nombre de la acción de espera (``_`` no puede empezar un nombre PDDL).
WAIT = "_wait"

#: Tipos de paso: acción instantánea, inicio, fin y espera.
INSTANT, START_SNAP, END_SNAP, WAIT_SNAP = 0, 1, 2, 3

_RUN, _EVENT, _PROCESS = "run", "event", "process"


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


@dataclass(frozen=True)
class RunningAction:
    """Una acción durativa instanciada.

    Attributes:
        running: el hecho ``running@a``.
        clock: la variable de su reloj (tiempo desde que empezó).
        bounds: variables de las cotas de su duración fijadas al empezar (las esperas se
            acortan para llegar justo a ellas).
        invariant: sus condiciones ``over all`` en DNF (``None`` si no tiene).
    """

    name: str
    args: tuple[str, ...]
    running: int
    clock: int
    bounds: tuple[int, ...]
    invariant: tuple[Condition, ...] | None


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
    """

    delta: float = DELTA
    epsilon: float = EPSILON
    processes: tuple[GroundProcess, ...] = ()
    events: tuple[GroundAction, ...] = ()
    durative: tuple[RunningAction, ...] = ()
    kinds: tuple[int, ...] = ()
    durative_of: tuple[int, ...] = ()
    metric_is_makespan: bool = False

    @cached_property
    def continuous_variables(self) -> frozenset[int]:
        """Las variables que cambian los procesos."""
        return frozenset(var for p in self.processes for var, _ in p.rates)

    @property
    def wait(self) -> GroundAction:
        """La acción de espera (la última)."""
        return self.actions[-1]

    # --- Semántica de los pasos ---------------------------------------------------------

    def apply(self, state: State, action: GroundAction) -> State | None:
        """Aplica un paso: una espera, o una acción seguida de su ε (ver la documentación
        del módulo). ``None`` si no es válido."""
        if self.kinds[action.index] == WAIT_SNAP:
            return self.elapse(state, self.wait_length(state))
        successor = action.apply(state)
        if successor is None:
            return None
        successor = self.settle(State(successor.facts, _round(successor.values)))
        if successor is None:
            return None
        return self.elapse(successor, self.epsilon)

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
            for bound in running.bounds:
                limit = values[bound]
                if limit is not None and clock < limit < clock + length:
                    length = limit - clock
        return round(length, DECIMALS)

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

    def _until_change(self, state: State, duration: float) -> tuple[State | None, float]:
        """Avanza hasta ``duration`` o hasta el primer instante en que se cumple un evento o
        cambian los procesos activos (lo que llegue antes). Devuelve el estado y lo que ha
        avanzado."""
        active = self._active(state)
        if not active:
            return state, duration
        flow = self._flow(active)
        end = self._advanced(state, flow, duration)
        if end is None or not self._changes(state, end, active):
            return end, duration
        low, high = 0.0, duration
        while high - low > _TIME_TOLERANCE:
            middle = (low + high) / 2
            point = self._advanced(state, flow, middle)
            if point is None or self._changes(state, point, active):
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

    def _changes(self, before: State, after: State, active: frozenset[int]) -> bool:
        """``True`` si en ``after`` se cumple un evento que no se cumplía en ``before`` o
        cambian los procesos activos."""
        if self._active(after) != active:
            return True
        return any(e.applicable(after) and not e.applicable(before) for e in self.events)

    def settle(self, state: State | None) -> State | None:
        """Dispara los eventos en cascada, recalcula los predicados derivados y comprueba
        las invariantes. ``None`` si alguna no se cumple o si los eventos no se detienen."""
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
                successor = State(successor.facts, _round(successor.values))
                if self.axioms:
                    successor = self.axiom_evaluator.evaluate(successor)
                if successor != state:
                    state, fired = successor, True
            if not fired:
                break
        else:
            return None  # los eventos se disparan sin fin
        for running in self.durative:
            if (
                running.invariant is not None
                and state.facts >> running.running & 1
                and not any(c.holds(state) for c in running.invariant)
            ):
                return None
        return state

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
        wait = replace(self.wait, conditional=tuple(effects))
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
            (*self.actions[:-1], wait),
            self.init,
            self.goal,
            self.variables,
            axioms=self.axioms,
            monitor_rules=tuple(rules),
        )

    # --- Plan -----------------------------------------------------------------------------

    def schedule(self, plan: Sequence[int]) -> tuple[list[TemporalStep], float]:
        """El plan en el tiempo: cada acción instantánea en su instante y cada acción
        durativa con su inicio y su duración. Devuelve también la duración total (el
        instante del último paso que no es una espera)."""
        state, time, makespan = self.init, 0.0, 0.0
        steps: list[TemporalStep] = []
        started: dict[int, float] = {}
        for index in plan:
            action = self.actions[index]
            kind = self.kinds[index]
            if kind == WAIT_SNAP:
                time = round(time + self.wait_length(state), DECIMALS)
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
            successor = self.apply(state, action)
            assert successor is not None, "el plan no es aplicable"
            state = successor
        steps.sort(key=lambda s: s.time)
        return steps, makespan

    def check(self, plan: Sequence[int], delta: float) -> str | None:
        """Simula el plan con esperas de como mucho ``delta`` (más fino que el de la
        búsqueda). Devuelve ``None`` si sigue siendo válido o la descripción del fallo."""
        fine = replace(self, delta=delta)
        state: State | None = self.init
        for step, index in enumerate(plan, start=1):
            assert state is not None
            action = self.actions[index]
            if self.kinds[index] == WAIT_SNAP:
                remaining = self.wait_length(state)
                while remaining > 1e-9 and state is not None:
                    piece = min(fine.wait_length(state), remaining)
                    state = fine.elapse(state, piece)
                    remaining = round(remaining - piece, DECIMALS)
                if state is None:
                    return f"paso {step}: con δ = {delta:g}, la espera rompe una invariante"
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


def _round(values: tuple[float | None, ...]) -> tuple[float | None, ...]:
    return tuple(None if v is None else round(v, DECIMALS) for v in values)


# --- Compilación y grounding ---------------------------------------------------------


@dataclass(frozen=True)
class _Durative:
    """Una acción durativa compilada: sus esquemas y lo que hace falta al instanciarla."""

    action: DurativeAction
    running: Predicate
    clock: Function
    bounds: tuple[Function, ...]
    invariant: Formula | None


def _compile_durative(
    action: DurativeAction,
) -> tuple[list[Action], _Durative, list[Predicate], list[Function]]:
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
    if watches:
        raise UnsupportedFeatureError(
            f"acción durativa '{name}': los efectos condicionales con condiciones 'over all' "
            "todavía no se admiten en dominios con procesos y eventos"
        )
    if any(c.condition is not None for c in continuous):
        raise UnsupportedFeatureError(
            f"acción durativa '{name}': los efectos continuos condicionales todavía no se "
            "admiten en dominios con procesos y eventos"
        )

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
    end_conditions: list[Formula] = [
        *(_formula_with(c, clock_term) for c in conditions[END]),
    ]
    bounds: list[Function] = []
    for constraint in action.duration:
        if constraint.time == END:  # se evalúa al terminar
            end_conditions += _bound_conditions(constraint, clock_term, constraint.value)
            continue
        bound = Function(f"bound@{name}@{len(bounds)}", parameters)
        bounds.append(bound)
        term = FluentTerm(bound.name, variables)
        start_effects.append(NumericEffect("assign", term, constraint.value))
        end_conditions += _bound_conditions(constraint, clock_term, term)
    invariant = And(tuple(conditions[ALL])) if conditions[ALL] else None
    if invariant is not None:
        end_conditions.append(invariant)
    end_conditions.append(running_atom)

    start = Action(
        f"{name}@start",
        parameters,
        And((And(tuple(conditions[START])), Not(running_atom))),
        And(tuple(start_effects)),
    )
    end = Action(
        f"{name}@end",
        parameters,
        And(tuple(end_conditions)),
        And(
            (
                *(_effect_with(e, clock_term) for e in effects[END]),
                Not(running_atom),
            )
        ),
    )
    rates: list[Effect] = [NumericEffect("increase", clock_term, Number(1.0))]
    for item in continuous:
        rate = _replace_duration(item.effect.rate, fixed)
        numeric = NumericEffect(item.effect.op, item.effect.fluent, rate)  # type: ignore[arg-type]
        rates.append(ForallEffect(item.parameters, numeric) if item.parameters else numeric)
    run = Action(f"{name}@{_RUN}", parameters, running_atom, And(tuple(rates)))
    compiled = _Durative(action, running, clock, tuple(bounds), invariant)
    return [start, end, run], compiled, [running, *flags], [clock, *bounds]


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
    if problem.timed_literals:
        raise UnsupportedFeatureError(
            "los literales iniciales temporizados todavía no se admiten con procesos y eventos"
        )
    if domain.constraints is not None or problem.constraints is not None:
        raise UnsupportedFeatureError(
            "las restricciones de trayectoria todavía no se admiten con procesos y eventos"
        )

    compiled: dict[str, _Durative] = {}
    actions: list[Action] = list(domain.actions)
    predicates: list[Predicate] = list(domain.predicates)
    functions: list[Function] = list(domain.functions)
    for durative_action in domain.durative_actions:
        schemas, item, extra_predicates, extra_functions = _compile_durative(durative_action)
        compiled[durative_action.name] = item
        actions += schemas
        predicates += extra_predicates
        functions += extra_functions
    actions += [
        Action(f"{e.name}@{_EVENT}", e.parameters, e.precondition, e.effect) for e in domain.events
    ]
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
    task, context = ground_with_context(flat, replace(problem, metric=None))

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
        instance_of[(base, args)] = len(running_actions)
        running_actions.append(RunningAction(base, args, running, clock, bounds, invariant))

    real: list[GroundAction] = []
    kinds: list[int] = []
    durative_of: list[int] = []
    for action in task.actions:
        base, _, kind = action.name.rpartition("@")
        if not base:  # acción del dominio (sin @)
            real.append(replace(action, index=len(real), cost=ACTION_COST))
            kinds.append(INSTANT)
            durative_of.append(-1)
        elif kind in (START, END) and (base, action.args) in instance_of:
            real.append(replace(action, index=len(real), cost=ACTION_COST))
            kinds.append(START_SNAP if kind == START else END_SNAP)
            durative_of.append(instance_of[(base, action.args)])
    real.append(GroundAction(len(real), WAIT, (), (), (), (), cost=delta))
    kinds.append(WAIT_SNAP)
    durative_of.append(-1)

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
    if metric is not None and not is_makespan:
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
        delta=delta,
        processes=tuple(ground_processes),
        events=tuple(events),
        durative=tuple(running_actions),
        kinds=tuple(kinds),
        durative_of=tuple(durative_of),
        metric_is_makespan=is_makespan,
    )
    # Los eventos que se cumplen al principio se disparan en el instante 0.
    init = discrete.settle(State(task.init.facts, _round(task.init.values)))
    if init is None:
        notes.append("el estado inicial rompe una invariante o sus eventos no se detienen")
        return replace(discrete, goal=(), notes=tuple(notes))
    return replace(discrete, init=init)
