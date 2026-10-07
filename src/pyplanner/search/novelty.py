"""Novedad de los estados, para la búsqueda por anchura (IW) y BFWS.

Lipovetzky y Geffner (2012, 2017). La **novedad** ``w`` de un estado nuevo, respecto a los
estados vistos antes, es el tamaño de la tupla de átomos más pequeña que hace cierta por
primera vez:

* ``w = 1`` si hace cierto algún átomo que ningún estado anterior tenía;
* ``w = 2`` si no, pero hace cierto algún par de átomos que nunca se había visto junto;
* ``w = 3`` (en la práctica, "más de 2") en otro caso.

Los **átomos** son los hechos ciertos del estado; la **falsedad** de los hechos que
aparecen negados en alguna precondición, condición u objetivo (como en la relajación: si
el objetivo pide que una luz esté apagada, apagarla es algo nuevo); y, para las variables
numéricas, los pares (variable, valor).

En **BFWS** la novedad se mide por separado en cada **partición** (los estados con el mismo
valor heurístico): un estado es nuevo si lo es entre los que la heurística considera igual
de buenos. Así la búsqueda prefiere, entre ellos, los que exploran algo distinto.

**Implementación con bitsets.** Los átomos de un estado forman un entero (bitset). Para
``w = 1`` basta un bitset con los átomos vistos. Para ``w = 2``, cada átomo ``p`` guarda el
bitset de los átomos que se han visto junto a él: el estado tiene un par nuevo si, para
algún ``p`` suyo, sus átomos no están todos en ese bitset. Medir y registrar un estado
cuesta O(átomos del estado) operaciones sobre enteros grandes, que se ejecutan en C.
"""

from collections.abc import Hashable, Iterable

from pyplanner.grounding import GroundTask, State, bits, mask_of


def negated_facts(task: GroundTask) -> set[int]:
    """Hechos que aparecen negados en alguna precondición, condición u objetivo.

    En las tareas temporales se excluyen los hechos internos ``running@a``: el objetivo
    exige que ninguna acción siga en curso, y con miles de acciones durativas esos átomos
    hacían la novedad cara y poco discriminante (de 33 a 29 instancias resueltas en las
    pruebas de la iteración 3b).
    """
    result = {f for clause in task.goal for f in clause.neg_facts}
    for action in task.actions:
        result.update(action.neg_facts)
        for effect in action.conditional:
            result.update(effect.condition.neg_facts)
    running_mask: int = getattr(task, "running_mask", 0)
    return {f for f in result if not running_mask >> f & 1}


class _Partition:
    """Átomos (y pares de átomos) vistos en una partición."""

    __slots__ = ("seen", "together")

    def __init__(self) -> None:
        self.seen = 0
        self.together: dict[int, int] = {}


class NoveltyTable:
    """Mide la novedad de los estados y los registra.

    Args:
        width: tamaño máximo de las tuplas que se consideran (1 o 2). La novedad que se
            devuelve va de 1 a ``width + 1`` (``width + 1`` = no aporta nada nuevo).
        num_facts: número de hechos de la tarea. El átomo "no p" del hecho ``p`` es el bit
            ``num_facts + p``, y los numéricos van detrás.
        negated: hechos cuya falsedad cuenta como átomo.
        continuous: variables numéricas que cambian de forma continua (con procesos): no
            cuentan. Toman un valor nuevo en cada espera, así que todo estado parecería
            nuevo y la tabla crecería sin fin (en generator, 85 000 átomos y 2 GB en 20 s).
            Limitar sus valores distintos (16, 64, 256 o 1024) en lugar de ignorarlas no
            mejora: los resultados dependen del límite sin ninguna tendencia.
    """

    def __init__(
        self,
        width: int,
        num_facts: int,
        negated: Iterable[int] = (),
        continuous: Iterable[int] = (),
    ) -> None:
        if width not in (1, 2):
            raise ValueError("la anchura de la novedad tiene que ser 1 o 2")
        self.width = width
        self._num_facts = 2 * num_facts
        self._shift = num_facts
        self._negated = mask_of(sorted(set(negated)))
        self._numeric: dict[tuple[int, float | None], int] = {}
        self._continuous = frozenset(continuous)
        self._partitions: dict[Hashable, _Partition] = {}

    def _atoms(self, state: State) -> int:
        atoms = state.facts
        if self._negated:
            atoms |= (~state.facts & self._negated) << self._shift
        if state.values:
            numeric = self._numeric
            base = self._num_facts
            continuous = self._continuous
            for variable, value in enumerate(state.values):
                if variable in continuous:
                    continue
                key = (variable, value)
                index = numeric.get(key)
                if index is None:
                    index = numeric[key] = base + len(numeric)
                atoms |= 1 << index
        return atoms

    def novelty(self, state: State, partition: Hashable = None) -> int:
        """Novedad de ``state`` en ``partition``; lo registra como visto."""
        table = self._partitions.get(partition)
        if table is None:
            table = self._partitions[partition] = _Partition()
        atoms = self._atoms(state)
        result = self.width + 1
        if atoms & ~table.seen:
            result = 1
        table.seen |= atoms
        if self.width == 1:
            return result
        together = table.together
        for atom in bits(atoms):
            known = together.get(atom, 0)
            if result > 2 and atoms & ~known & ~(1 << atom):
                result = 2
            together[atom] = known | atoms
        return result
