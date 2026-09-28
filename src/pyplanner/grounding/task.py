"""Tarea instanciada (grounded): la representación sobre la que trabaja la búsqueda.

Un **estado** es un entero de Python usado como bitset: el bit *i* está a 1 si el hecho
``task.facts[i]`` es cierto. Los enteros de Python son inmutables, *hashables* y de tamaño
arbitrario, y las operaciones de bits se ejecutan en C, así que comprobar si una acción es
aplicable o aplicarla cuesta unas pocas operaciones sobre enteros.

Los hechos **estáticos** (los que ninguna acción modifica) no forman parte del estado: el
grounder los usa para descartar instancias imposibles y después los elimina de las
precondiciones y del objetivo.

Tras la normalización ADL (ver :mod:`pyplanner.grounding.normalize`), todo es conjuntivo:

* una :class:`Condition` es una conjunción de hechos que tienen que ser ciertos (``pos``) y
  hechos que tienen que ser falsos (``neg``);
* una acción tiene una precondición, efectos incondicionales y efectos condicionales;
* el objetivo es una **disyunción** de condiciones (casi siempre, una sola).
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from functools import cached_property

from pyplanner.loader.ast import Atom

#: Un estado: bitset sobre los índices de ``GroundTask.facts``.
State = int


def bits(mask: int) -> Iterator[int]:
    """Devuelve los índices de los bits a 1 de ``mask``, de menor a mayor."""
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def mask_of(indices: Iterator[int] | Sequence[int]) -> int:
    """Construye el bitset con los índices dados."""
    mask = 0
    for index in indices:
        mask |= 1 << index
    return mask


@dataclass(frozen=True)
class Condition:
    """Conjunción de literales: ``pos_facts`` ciertos y ``neg_facts`` falsos.

    ``Condition()`` es la condición vacía (siempre cierta).
    """

    pos_facts: tuple[int, ...] = ()
    neg_facts: tuple[int, ...] = ()
    pos: int = field(init=False, repr=False, compare=False)
    neg: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pos", mask_of(self.pos_facts))
        object.__setattr__(self, "neg", mask_of(self.neg_facts))

    def holds(self, state: State) -> bool:
        """``True`` si la condición es cierta en ``state``."""
        return state & self.pos == self.pos and not state & self.neg


@dataclass(frozen=True)
class ConditionalEffect:
    """Efecto condicional: si ``condition`` es cierta *antes* de la acción, añade y borra."""

    condition: Condition
    add_facts: tuple[int, ...]
    del_facts: tuple[int, ...]
    add: int = field(init=False, repr=False, compare=False)
    delete: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "add", mask_of(self.add_facts))
        object.__setattr__(self, "delete", mask_of(self.del_facts))


@dataclass(frozen=True)
class GroundAction:
    """Una acción instanciada.

    Attributes:
        index: posición de la acción en ``GroundTask.actions``.
        name: nombre del esquema de acción.
        args: objetos con los que se ha instanciado.
        pre_facts, neg_facts: precondición (hechos ciertos y hechos falsos).
        add_facts, del_facts: efectos incondicionales.
        conditional: efectos condicionales.
        cost: coste de la acción.
        pre, neg, add, delete: los mismos conjuntos como bitsets.

    Una acción ADL con una precondición disyuntiva da lugar a varias ``GroundAction`` con el
    mismo nombre y argumentos, una por disyunto.
    """

    index: int
    name: str
    args: tuple[str, ...]
    pre_facts: tuple[int, ...]
    add_facts: tuple[int, ...]
    del_facts: tuple[int, ...]
    cost: int = 1
    neg_facts: tuple[int, ...] = ()
    conditional: tuple[ConditionalEffect, ...] = ()
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
        """``True`` si la precondición se cumple en ``state``."""
        return state & self.pre == self.pre and not state & self.neg

    def apply(self, state: State) -> State:
        """Aplica la acción (se supone aplicable).

        Las condiciones de los efectos condicionales se evalúan sobre el estado **anterior**.
        Primero se borra y después se añade, así que un hecho que se borra y se añade a la
        vez queda cierto.
        """
        if not self.conditional:
            return (state & ~self.delete) | self.add
        adds, deletes = self.add, self.delete
        triggers, by_trigger, always = self._conditional_index
        for fact in bits(state & triggers):
            for effect in by_trigger[fact]:
                if effect.condition.holds(state):
                    adds |= effect.add
                    deletes |= effect.delete
        for effect in always:
            if effect.condition.holds(state):
                adds |= effect.add
                deletes |= effect.delete
        return (state & ~deletes) | adds

    def __str__(self) -> str:
        return f"({' '.join([self.name, *self.args])})"


@dataclass(frozen=True)
class GroundTask:
    """Tarea de planificación instanciada.

    Attributes:
        facts: hechos (átomos sin variables) no estáticos, en un orden fijo.
        actions: acciones instanciadas; ``actions[i].index == i``.
        init: estado inicial.
        goal: disyunción de condiciones; el estado es objetivo si cumple alguna. Una tupla
            vacía significa que el objetivo es imposible.
    """

    domain_name: str
    problem_name: str
    facts: tuple[Atom, ...]
    actions: tuple[GroundAction, ...]
    init: State
    goal: tuple[Condition, ...]

    @cached_property
    def fact_index(self) -> dict[Atom, int]:
        """Índice de cada hecho en ``facts``."""
        return {fact: i for i, fact in enumerate(self.facts)}

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
        for action in always:
            if not state & action.neg:
                yield action
        for fact in bits(state):
            for action in by_fact.get(fact, ()):
                if state & action.pre == action.pre and not state & action.neg:
                    yield action

    @staticmethod
    def apply(state: State, action: GroundAction) -> State:
        """Aplica una acción (se supone aplicable). Ver :meth:`GroundAction.apply`."""
        return action.apply(state)

    def atoms(self, state: State) -> list[Atom]:
        """Los hechos ciertos en ``state`` (para depurar y mostrar)."""
        return [self.facts[i] for i in bits(state)]
