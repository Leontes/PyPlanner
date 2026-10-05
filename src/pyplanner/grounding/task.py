"""Tarea instanciada (grounded): la representación sobre la que trabaja la búsqueda.

Un **estado** (:class:`State`) tiene dos partes:

* ``facts``: un entero de Python usado como bitset; el bit *i* está a 1 si el hecho
  ``task.facts[i]`` es cierto. Las operaciones de bits se ejecutan en C, así que comprobar
  una precondición o aplicar una acción cuesta unas pocas operaciones sobre enteros;
* ``values``: los valores de las variables numéricas ``task.variables`` (``None`` si están
  indefinidas). En las tareas sin variables numéricas es una tupla vacía.

``State`` es una tupla con nombre: inmutable y *hashable*, así que sirve de clave en la
tabla de nodos de la búsqueda.

Los hechos y las funciones **estáticos** (los que ninguna acción modifica) no forman parte
del estado: el grounder los evalúa y los elimina de las condiciones.

Tras la normalización (ver :mod:`pyplanner.grounding.normalize`), todo es conjuntivo:

* una :class:`Condition` es una conjunción de hechos ciertos (``pos``), hechos falsos
  (``neg``) y condiciones numéricas;
* una acción tiene una precondición, efectos incondicionales (de hechos y numéricos) y
  efectos condicionales;
* el objetivo es una **disyunción** de condiciones (casi siempre, una sola).

**Predicados derivados** (iteración 4). Sus hechos están en ``facts`` como los demás, pero
ninguna acción los cambia: los deducen los **axiomas** (:class:`Axiom`) a partir del resto
del estado. :meth:`GroundTask.apply` los recalcula en cada sucesor (ver
:mod:`pyplanner.grounding.axioms`), y el estado inicial ya los trae calculados.
"""

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, NamedTuple

from pyplanner.grounding.numeric import NumericAssignment, NumericCondition
from pyplanner.loader.ast import Atom, FluentTerm

if TYPE_CHECKING:
    from pyplanner.grounding.axioms import AxiomEvaluator
    from pyplanner.grounding.monitors import Monitor, MonitorEvaluator, MonitorRule, StepClock


class State(NamedTuple):
    """Estado: hechos ciertos (bitset) y valores de las variables numéricas."""

    facts: int
    values: tuple[float | None, ...] = ()


def bits(mask: int) -> Iterator[int]:
    """Devuelve los índices de los bits a 1 de ``mask``, de menor a mayor."""
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def mask_of(indices: Iterable[int]) -> int:
    """Construye el bitset con los índices dados."""
    mask = 0
    for index in indices:
        mask |= 1 << index
    return mask


@dataclass(frozen=True)
class Condition:
    """Conjunción de literales: ``pos_facts`` ciertos, ``neg_facts`` falsos y condiciones
    numéricas.

    ``Condition()`` es la condición vacía (siempre cierta).
    """

    pos_facts: tuple[int, ...] = ()
    neg_facts: tuple[int, ...] = ()
    numeric: tuple[NumericCondition, ...] = ()
    pos: int = field(init=False, repr=False, compare=False)
    neg: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pos", mask_of(self.pos_facts))
        object.__setattr__(self, "neg", mask_of(self.neg_facts))

    def holds(self, state: State) -> bool:
        """``True`` si la condición es cierta en ``state``."""
        facts = state.facts
        if facts & self.pos != self.pos or facts & self.neg:
            return False
        return all(c.holds(state.values) for c in self.numeric)


@dataclass(frozen=True)
class ConditionalEffect:
    """Efecto condicional: si ``condition`` es cierta *antes* de la acción, añade, borra y
    aplica sus efectos numéricos."""

    condition: Condition
    add_facts: tuple[int, ...]
    del_facts: tuple[int, ...]
    numeric: tuple[NumericAssignment, ...] = ()
    add: int = field(init=False, repr=False, compare=False)
    delete: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "add", mask_of(self.add_facts))
        object.__setattr__(self, "delete", mask_of(self.del_facts))


@dataclass(frozen=True)
class Axiom:
    """Axioma instanciado: el hecho derivado ``head`` es cierto si ``condition`` lo es.

    Un hecho derivado es cierto si lo hace cierto alguno de sus axiomas. ``stratum`` es el
    estrato de su predicado: los axiomas se evalúan por estratos crecientes, y la
    condición solo niega hechos básicos o de estratos anteriores.
    """

    head: int
    condition: Condition
    stratum: int


@dataclass(frozen=True)
class GroundAction:
    """Una acción instanciada.

    Attributes:
        index: posición de la acción en ``GroundTask.actions``.
        name: nombre del esquema de acción.
        args: objetos con los que se ha instanciado.
        pre_facts, neg_facts: precondición proposicional (hechos ciertos y hechos falsos).
        numeric_pre: precondiciones numéricas.
        add_facts, del_facts: efectos incondicionales sobre hechos.
        numeric: efectos numéricos incondicionales.
        conditional: efectos condicionales.
        cost: coste de la acción (1, o su contribución a la métrica; ver ``GroundTask``).
        pre, neg, add, delete: los conjuntos de hechos como bitsets.

    Una acción ADL con una precondición disyuntiva da lugar a varias ``GroundAction`` con el
    mismo nombre y argumentos, una por disyunto.
    """

    index: int
    name: str
    args: tuple[str, ...]
    pre_facts: tuple[int, ...]
    add_facts: tuple[int, ...]
    del_facts: tuple[int, ...]
    cost: float = 1
    neg_facts: tuple[int, ...] = ()
    conditional: tuple[ConditionalEffect, ...] = ()
    numeric_pre: tuple[NumericCondition, ...] = ()
    numeric: tuple[NumericAssignment, ...] = ()
    pre: int = field(init=False, repr=False, compare=False)
    neg: int = field(init=False, repr=False, compare=False)
    add: int = field(init=False, repr=False, compare=False)
    delete: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pre", mask_of(self.pre_facts))
        object.__setattr__(self, "neg", mask_of(self.neg_facts))
        object.__setattr__(self, "add", mask_of(self.add_facts))
        object.__setattr__(self, "delete", mask_of(self.del_facts))

    @cached_property
    def _conditional_index(
        self,
    ) -> tuple[int, dict[int, tuple[ConditionalEffect, ...]], tuple[ConditionalEffect, ...]]:
        """Efectos condicionales agrupados por su primer hecho positivo (su "disparador").

        Al aplicar la acción solo se revisan los efectos cuyo disparador es cierto, en lugar
        de todos. Los efectos sin hechos positivos en la condición se revisan siempre.
        """
        by_trigger: dict[int, list[ConditionalEffect]] = {}
        always: list[ConditionalEffect] = []
        for effect in self.conditional:
            if effect.condition.pos_facts:
                by_trigger.setdefault(effect.condition.pos_facts[0], []).append(effect)
            else:
                always.append(effect)
        triggers = mask_of(list(by_trigger))
        return triggers, {f: tuple(e) for f, e in by_trigger.items()}, tuple(always)

    def applicable(self, state: State) -> bool:
        """``True`` si la precondición se cumple en ``state``.

        Los efectos numéricos pueden hacer que, aun así, la acción no se pueda aplicar (ver
        :meth:`apply`).
        """
        facts = state.facts
        if facts & self.pre != self.pre or facts & self.neg:
            return False
        return all(c.holds(state.values) for c in self.numeric_pre)

    def apply(self, state: State) -> State | None:
        """Aplica la acción (se supone que la precondición se cumple).

        Las condiciones de los efectos condicionales y los valores de los efectos numéricos
        se evalúan sobre el estado **anterior**. Primero se borra y después se añade, así que
        un hecho que se borra y se añade a la vez queda cierto.

        Returns:
            El estado sucesor, o ``None`` si algún efecto numérico no está definido: usa
            una variable sin valor, divide por cero o varios efectos se contradicen sobre la
            misma variable (solo ``increase`` y ``decrease`` se pueden combinar).
        """
        facts = state.facts
        if not self.conditional:
            new_facts = (facts & ~self.delete) | self.add
            if not self.numeric:
                return State(new_facts, state.values)
            values = _apply_numeric(state.values, self.numeric)
            return None if values is None else State(new_facts, values)
        adds, deletes = self.add, self.delete
        numeric = list(self.numeric)
        triggers, by_trigger, always = self._conditional_index
        for fact in bits(facts & triggers):
            for effect in by_trigger[fact]:
                if effect.condition.holds(state):
                    adds |= effect.add
                    deletes |= effect.delete
                    numeric.extend(effect.numeric)
        for effect in always:
            if effect.condition.holds(state):
                adds |= effect.add
                deletes |= effect.delete
                numeric.extend(effect.numeric)
        new_facts = (facts & ~deletes) | adds
        if not numeric:
            return State(new_facts, state.values)
        values = _apply_numeric(state.values, numeric)
        return None if values is None else State(new_facts, values)

    def __str__(self) -> str:
        return f"({' '.join([self.name, *self.args])})"


def _apply_numeric(
    values: tuple[float | None, ...], effects: Sequence[NumericAssignment]
) -> tuple[float | None, ...] | None:
    """Valores tras aplicar ``effects`` (evaluados en ``values``), o ``None`` si no están
    definidos o se contradicen."""
    if len(effects) == 1:
        effect = effects[0]
        new = effect.new_value(values)
        if new is None:
            return None
        result = list(values)
        result[effect.variable] = new
        return tuple(result)
    by_variable: dict[int, list[NumericAssignment]] = {}
    for effect in effects:
        by_variable.setdefault(effect.variable, []).append(effect)
    result = list(values)
    for variable, group in by_variable.items():
        if len(group) == 1:
            new = group[0].new_value(values)
        elif all(e.op in ("increase", "decrease") for e in group):
            new = values[variable]
            for effect in group:
                amount = effect.new_value(values)
                if new is None or amount is None:
                    return None
                new += amount - values[variable]  # type: ignore[operator]
        else:
            return None  # efectos incompatibles sobre la misma variable
        if new is None:
            return None
        result[variable] = new
    return tuple(result)


@dataclass(frozen=True)
class GroundTask:
    """Tarea de planificación instanciada.

    Attributes:
        facts: hechos (átomos sin variables) no estáticos, en un orden fijo.
        actions: acciones instanciadas; ``actions[i].index == i``.
        init: estado inicial.
        goal: disyunción de condiciones; el estado es objetivo si cumple alguna. Una tupla
            vacía significa que el objetivo es imposible.
        variables: variables numéricas del estado (funciones que alguna condición necesita).
        metric_is_cost: ``True`` si el coste de cada acción es su contribución a la métrica
            (y minimizar el coste minimiza la métrica); ``False`` si el coste es 1.
        metric_offset: valor de la métrica en el estado inicial, si ``metric_is_cost``. El
            valor de la métrica de un plan es ``metric_offset`` más su coste.
        notes: avisos del grounding (por ejemplo, por qué la métrica no se usa como coste).
        axioms: axiomas de los predicados derivados, ordenados por estrato.
        monitors: restricciones de trayectoria duras (PDDL 3), que se actualizan tras cada
            paso (ver :mod:`pyplanner.grounding.monitors`).
        monitor_rules: sus reglas relajadas, para la heurística.
        monitor_clock: contador de pasos, si una tarea clásica tiene restricciones con
            tiempo.
    """

    domain_name: str
    problem_name: str
    facts: tuple[Atom, ...]
    actions: tuple[GroundAction, ...]
    init: State
    goal: tuple[Condition, ...]
    variables: tuple[FluentTerm, ...] = ()
    metric_is_cost: bool = False
    metric_offset: float | None = None
    notes: tuple[str, ...] = ()
    axioms: tuple[Axiom, ...] = ()
    monitors: tuple["Monitor", ...] = ()
    monitor_rules: tuple["MonitorRule", ...] = ()
    monitor_clock: "StepClock | None" = None

    @cached_property
    def monitor_evaluator(self) -> "MonitorEvaluator":
        """Evaluador de las restricciones de trayectoria."""
        from pyplanner.grounding.monitors import MonitorEvaluator

        return MonitorEvaluator(self.monitors, self.monitor_clock)

    @cached_property
    def derived_mask(self) -> int:
        """Bitset de los hechos derivados (las cabezas de los axiomas)."""
        return mask_of({axiom.head for axiom in self.axioms})

    @cached_property
    def axiom_evaluator(self) -> "AxiomEvaluator":
        """Evaluador de los axiomas (ver :mod:`pyplanner.grounding.axioms`)."""
        from pyplanner.grounding.axioms import AxiomEvaluator

        return AxiomEvaluator(self.axioms)

    @cached_property
    def fact_index(self) -> dict[Atom, int]:
        """Índice de cada hecho en ``facts``."""
        return {fact: i for i, fact in enumerate(self.facts)}

    @cached_property
    def variable_index(self) -> dict[FluentTerm, int]:
        """Índice de cada variable numérica en ``variables``."""
        return {variable: i for i, variable in enumerate(self.variables)}

    @cached_property
    def _actions_by_trigger(
        self,
    ) -> tuple[tuple[GroundAction, ...], dict[int, tuple[GroundAction, ...]]]:
        """Acciones sin precondiciones positivas y acciones agrupadas por la primera.

        Para generar los sucesores de un estado basta con mirar las acciones cuya primera
        precondición positiva es cierta, en lugar de recorrer todas las acciones.
        """
        always: list[GroundAction] = []
        by_fact: dict[int, list[GroundAction]] = {}
        for action in self.actions:
            if action.pre_facts:
                by_fact.setdefault(action.pre_facts[0], []).append(action)
            else:
                always.append(action)
        return tuple(always), {f: tuple(acts) for f, acts in by_fact.items()}

    def is_goal(self, state: State) -> bool:
        """``True`` si el estado satisface alguna cláusula del objetivo."""
        return any(clause.holds(state) for clause in self.goal)

    def applicable(self, state: State) -> Iterator[GroundAction]:
        """Devuelve las acciones aplicables en ``state`` en el **orden canónico**.

        Orden canónico: primero las acciones sin precondiciones positivas; después, las de
        cada hecho disparador (su primera precondición positiva) en orden creciente de
        hecho; dentro de cada grupo, por índice. Es determinista y no necesita ordenar, y
        el generador compilado (:mod:`pyplanner.search.successors`) usa el mismo, así que la
        búsqueda se comporta igual con los dos.
        """
        always, by_fact = self._actions_by_trigger
        facts = state.facts
        values = state.values
        for action in always:
            if not facts & action.neg and all(c.holds(values) for c in action.numeric_pre):
                yield action
        for fact in bits(facts):
            for action in by_fact.get(fact, ()):
                if (
                    facts & action.pre == action.pre
                    and not facts & action.neg
                    and all(c.holds(values) for c in action.numeric_pre)
                ):
                    yield action

    @property
    def relaxation(self) -> "GroundTask":
        """La tarea sobre la que se calculan las heurísticas (la propia tarea, salvo en las
        temporales con variables que cambian de forma continua)."""
        return self

    def apply(self, state: State, action: GroundAction) -> State | None:
        """Aplica una acción; ``None`` si no se puede. Ver :meth:`GroundAction.apply`.

        Las tareas temporales (:class:`pyplanner.grounding.temporal.TemporalTask`) lo
        redefinen para comprobar además la red temporal y las invariantes.

        Si hay predicados derivados, recalcula sus hechos en el sucesor, y si hay
        restricciones de trayectoria, actualiza sus monitores (``None`` si el paso incumple
        alguna)."""
        successor = action.apply(state)
        if successor is None:
            return None
        if self.axioms:
            successor = self.axiom_evaluator.evaluate(successor)
        if self.monitors:
            return self.monitor_evaluator.update(state, successor)
        return successor

    def atoms(self, state: State) -> list[Atom]:
        """Los hechos ciertos en ``state`` (para depurar y mostrar)."""
        return [self.facts[i] for i in bits(state.facts)]
