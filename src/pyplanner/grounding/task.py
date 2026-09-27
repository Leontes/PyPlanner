"""Tarea instanciada (grounded): la representación sobre la que trabaja la búsqueda.

Un **estado** es un entero de Python usado como bitset: el bit *i* está a 1 si el hecho
``task.facts[i]`` es cierto. Los enteros de Python son inmutables, *hashables* y de tamaño
arbitrario, y las operaciones de bits se ejecutan en C, así que comprobar si una acción es
aplicable o aplicarla cuesta una o dos operaciones sobre enteros.

Los hechos **estáticos** (los que ninguna acción modifica) no forman parte del estado: el
grounder los usa para descartar instancias imposibles y después los elimina de las
precondiciones y del objetivo.
"""

from collections.abc import Iterator
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


def mask_of(indices: Iterator[int] | list[int] | tuple[int, ...]) -> int:
    """Construye el bitset con los índices dados."""
    mask = 0
    for index in indices:
        mask |= 1 << index
    return mask


@dataclass(frozen=True)
class GroundAction:
    """Una acción instanciada.

    Attributes:
        index: posición de la acción en ``GroundTask.actions``.
        name: nombre del esquema de acción.
        args: objetos con los que se ha instanciado.
        pre, add, delete: bitsets de precondiciones, añadidos y borrados.
        pre_facts, add_facts, del_facts: los mismos conjuntos como tuplas ordenadas de
            índices (las usan las heurísticas).
        cost: coste de la acción (1 en STRIPS).
    """

    index: int
    name: str
    args: tuple[str, ...]
    pre_facts: tuple[int, ...]
    add_facts: tuple[int, ...]
    del_facts: tuple[int, ...]
    cost: int = 1
    pre: int = field(init=False, repr=False, compare=False)
    add: int = field(init=False, repr=False, compare=False)
    delete: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pre", mask_of(self.pre_facts))
        object.__setattr__(self, "add", mask_of(self.add_facts))
        object.__setattr__(self, "delete", mask_of(self.del_facts))

    def __str__(self) -> str:
        return f"({' '.join([self.name, *self.args])})"


@dataclass(frozen=True)
class GroundTask:
    """Tarea de planificación instanciada.

    Attributes:
        facts: hechos (átomos sin variables) no estáticos, en un orden fijo.
        actions: acciones instanciadas; ``actions[i].index == i``.
        init: estado inicial.
        goal_facts: índices de los hechos del objetivo.
    """

    domain_name: str
    problem_name: str
    facts: tuple[Atom, ...]
    actions: tuple[GroundAction, ...]
    init: State
    goal_facts: tuple[int, ...]

    @cached_property
    def goal(self) -> int:
        """Bitset del objetivo."""
        return mask_of(self.goal_facts)

    @cached_property
    def fact_index(self) -> dict[Atom, int]:
        """Índice de cada hecho en ``facts``."""
        return {fact: i for i, fact in enumerate(self.facts)}

    @cached_property
    def _actions_by_trigger(
        self,
    ) -> tuple[tuple[GroundAction, ...], dict[int, tuple[GroundAction, ...]]]:
        """Acciones sin precondiciones y acciones agrupadas por su primera precondición.

        Para generar los sucesores de un estado basta con mirar las acciones cuya primera
        precondición es cierta, en lugar de recorrer todas las acciones de la tarea.
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
        """``True`` si el estado satisface el objetivo."""
        goal = self.goal
        return state & goal == goal

    def applicable(self, state: State) -> Iterator[GroundAction]:
        """Devuelve las acciones aplicables en ``state``, ordenadas por índice."""
        always, by_fact = self._actions_by_trigger
        candidates = list(always)
        for fact in bits(state):
            candidates.extend(by_fact.get(fact, ()))
        candidates.sort(key=lambda a: a.index)
        for action in candidates:
            if state & action.pre == action.pre:
                yield action

    @staticmethod
    def apply(state: State, action: GroundAction) -> State:
        """Aplica una acción (se supone aplicable). Los añadidos ganan a los borrados."""
        return (state & ~action.delete) | action.add

    def atoms(self, state: State) -> list[Atom]:
        """Los hechos ciertos en ``state`` (para depurar y mostrar)."""
        return [self.facts[i] for i in bits(state)]
