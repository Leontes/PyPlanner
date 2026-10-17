"""Landmarks de la relajación y su conteo al estilo de LAMA (trabajo opcional, punto 2).

Un **landmark** es un hecho que se cumple en algún momento de todo plan. Aquí se calculan los
de la relajación sin borrados (que también lo son de la tarea), sobre los hechos relajados de
:class:`~pyplanner.heuristics.relaxation.RelaxedTask`: los de la tarea, los complementarios de
los que se niegan y las condiciones numéricas, así que también hay landmarks numéricos.

**Cálculo** (una vez, desde el estado inicial): para cada hecho relajado ``p``, ``LM(p)`` es el
conjunto de hechos que todo plan relajado tiene que conseguir antes que ``p`` (o ``p`` mismo).
Es el punto fijo de ``LM(p) = {p} ∪ ⋂ₐ ⋃_{q ∈ pre(a)} LM(q)``, con ``a`` las acciones relajadas
que consiguen ``p``, empezando por ``LM(p) = {p}`` en los hechos del estado inicial (Zhu y Givan;
Keyder, Richter y Helmert, 2010). Los landmarks del objetivo son la unión de los de sus hechos
(con un objetivo disyuntivo, los comunes a todas las cláusulas alcanzables).

**Ordenaciones** entre landmarks:

* naturales: ``l`` va antes que ``l'`` si ``l ∈ LM(l')``;
* codiciosas-necesarias: ``l`` es precondición de todas las acciones que pueden conseguir
  ``l'`` por primera vez, así que tiene que cumplirse justo antes.

**Conteo** (:meth:`LandmarkGraph.value`): cada camino de la búsqueda lleva los landmarks
**aceptados**; uno se acepta cuando se cumple y ya lo están sus predecesores. El valor de un
estado es lo que falta por aceptar más lo aceptado que hay que volver a conseguir: los del
objetivo que no se cumplen y los que deben cumplirse justo antes de otro aún sin aceptar.
Depende del camino, no solo del estado (como en LAMA).
"""

from pyplanner.grounding import GroundTask, State, bits
from pyplanner.heuristics.base import DEAD_END
from pyplanner.heuristics.relaxation import RelaxedTask

#: Bits: conjuntos de hechos relajados o de landmarks como enteros de Python.
Bits = int


class LandmarkGraph:
    """Landmarks del objetivo desde el estado inicial, con sus ordenaciones.

    Los conjuntos de landmarks son bits sobre la numeración de los hechos relajados: así,
    los landmarks que son hechos de la tarea que se cumplen en un estado salen de un solo
    ``AND`` con sus hechos.

    Attributes:
        facts: hechos relajados que son landmarks.
        mask: los mismos, como bits.
        predecessors: por landmark, los que van antes por una ordenación natural (bits).
        greedy: por landmark ``l``, los ``l'`` que necesitan ``l`` justo antes (bits).
        goal: los landmarks que son hechos del objetivo (bits).
        achievers: por landmark, los operadores que pueden conseguirlo.
        unreachable: ``True`` si el objetivo no es alcanzable ni en la relajación.
    """

    def __init__(self, task: GroundTask, relaxed: RelaxedTask | None = None) -> None:
        self.relaxed = relaxed if relaxed is not None else RelaxedTask(task.relaxation)
        rt = self.relaxed
        lm = _fixpoint(rt, task.init)
        goal_facts: Bits | None = None
        for clause in rt.goal:
            if any(lm[g] is None for g in clause):
                continue  # cláusula inalcanzable en la relajación
            union = 0
            for g in clause:
                union |= lm[g] or 0
            goal_facts = union if goal_facts is None else goal_facts & union
        self.unreachable = goal_facts is None
        self.mask: Bits = goal_facts or 0
        self.facts: list[int] = list(bits(self.mask))
        self.predecessors: dict[int, Bits] = {
            f: (lm[f] or 0) & self.mask & ~(1 << f) for f in self.facts
        }
        goal_clause_facts = frozenset(f for clause in rt.goal for f in clause)
        self.goal: Bits = sum(1 << f for f in self.facts if f in goal_clause_facts)

        # Quién consigue cada landmark (acciones relajadas, con sus aristas numéricas).
        achieved_by: dict[int, list[int]] = {f: [] for f in self.facts}
        for a in range(rt.num_actions):
            for f in rt.add[a]:
                if f in achieved_by:
                    achieved_by[f].append(a)
            for e in rt.edges_of[a]:
                f = rt.edges[e][1]
                if f in achieved_by:
                    achieved_by[f].append(a)
        self.achievers: dict[int, frozenset[int]] = {
            f: frozenset(
                rt.operator[a] for a in achieved_by[f] if rt.operator[a] != rt.axiom_operator
            )
            for f in self.facts
        }
        # Codiciosas-necesarias: precondiciones comunes de las acciones que pueden conseguir el
        # landmark por primera vez (las alcanzables sin él).
        self.greedy: dict[int, Bits] = dict.fromkeys(self.facts, 0)
        for f in self.facts:
            if lm[f] == 1 << f:
                continue  # cierto al principio: no hace falta conseguirlo
            first = [
                a for a in achieved_by[f] if all(not ((lm[q] or 0) >> f) & 1 for q in rt.pre[a])
            ]
            common: Bits | None = None
            for a in first:
                pre = sum(1 << q for q in rt.pre[a])
                common = pre if common is None else common & pre
            for q in bits((common or 0) & self.mask):
                self.greedy[q] |= 1 << f
        # Para ver qué landmarks se cumplen: los hechos de la tarea con un AND; los
        # complementarios y las condiciones numéricas, uno a uno.
        complement_of = {c: f for f, c in rt.complement.items()}
        self._plain = self.mask & ((1 << rt.num_base) - 1)
        self._complements = [
            (f, complement_of[f]) for f in self.facts if rt.num_base <= f < rt.numeric_offset
        ]
        self._numeric = [
            (f, rt.numeric[f - rt.numeric_offset]) for f in self.facts if f >= rt.numeric_offset
        ]

    def __len__(self) -> int:
        return len(self.facts)

    # --- Por camino -------------------------------------------------------------------

    def true_in(self, state: State) -> Bits:
        """Los landmarks que se cumplen en ``state`` (bits)."""
        facts = state.facts
        result = facts & self._plain
        for f, fact in self._complements:
            if not (facts >> fact) & 1:
                result |= 1 << f
        for f, condition in self._numeric:
            if condition.holds(state.values):
                result |= 1 << f
        return result

    def accept(self, accepted: Bits, state: State) -> Bits:
        """Aceptados tras llegar a ``state`` por un camino con ``accepted``: se añaden los que
        se cumplen y tienen ya aceptados a todos sus predecesores (en la raíz, ``accepted``
        es 0)."""
        new = self.true_in(state) & ~accepted
        for f in bits(new):
            if self.predecessors[f] & ~accepted:
                new &= ~(1 << f)
        return accepted | new

    def value(self, accepted: Bits, state: State) -> float:
        """Landmarks sin aceptar más los aceptados que hay que volver a conseguir."""
        if self.unreachable:
            return DEAD_END
        pending = self.mask & ~accepted
        lost = accepted & ~self.true_in(state)
        again = lost & self.goal
        for f in bits(lost & ~again):
            if self.greedy[f] & pending:
                again |= 1 << f
        return float(pending.bit_count() + again.bit_count())

    def next_landmarks(self, accepted: Bits) -> Bits:
        """Los landmarks sin aceptar cuyos predecesores ya lo están (los siguientes)."""
        result = 0
        for f in bits(self.mask & ~accepted):
            if not self.predecessors[f] & ~accepted:
                result |= 1 << f
        return result

    def preferred(self, accepted: Bits) -> frozenset[int]:
        """Operadores que consiguen alguno de los landmarks siguientes."""
        result: set[int] = set()
        for f in bits(self.next_landmarks(accepted)):
            result.update(self.achievers[f])
        return frozenset(result)


def _fixpoint(rt: RelaxedTask, state: State) -> list[Bits | None]:
    """``LM(p)`` de cada hecho relajado desde ``state`` (``None`` si no es alcanzable ni en
    la relajación). Ver la documentación del módulo."""
    lm: list[Bits | None] = [None] * rt.num_facts
    for f in rt.true_facts(state):
        lm[f] = 1 << f
    multipliers = rt.multipliers(state) if rt.edges else []
    effects = [
        (*rt.add[a], *(rt.edges[e][1] for e in rt.edges_of[a] if multipliers[e] != DEAD_END))
        for a in range(rt.num_actions)
    ]
    changed = True
    while changed:
        changed = False
        for a in range(rt.num_actions):
            needed = 0
            for q in rt.pre[a]:
                known = lm[q]
                if known is None:
                    break
                needed |= known
            else:
                for p in effects[a]:
                    candidate = needed | (1 << p)
                    current = lm[p]
                    updated = candidate if current is None else current & candidate
                    if updated != current:
                        lm[p] = updated
                        changed = True
    return lm
