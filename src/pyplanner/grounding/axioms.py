"""Evaluación de los axiomas de los predicados derivados (PDDL 2.2).

Los hechos derivados de un estado son el **punto fijo mínimo** de sus axiomas, calculado
**por estratos**: en el estrato *s* ya se conocen los hechos básicos y los derivados de los
estratos anteriores, que son los únicos que pueden aparecer negados en sus condiciones (la
fase 2 del parser comprueba que las reglas son estratificables). Así, la negación es la de
la semántica de mundo cerrado: un hecho derivado es falso si ningún axioma lo hace cierto.

Dentro de un estrato, cada axioma tiene dos partes en su condición:

* **externa**: hechos básicos o de estratos anteriores (positivos y negados) y condiciones
  numéricas. Ya no cambia mientras se evalúa el estrato, así que se comprueba una vez;
* **interna**: hechos derivados positivos del propio estrato (las reglas recursivas, como
  un cierre transitivo).

Con la parte externa cierta, el axioma espera a que se cumpla la interna: un contador de
hechos internos pendientes que baja cada vez que se deduce uno (como en el *AxiomEvaluator*
de Fast Downward). Para no revisar todos los axiomas en cada estado, se agrupan por su
primer hecho externo positivo (su "disparador") y solo se revisan los de disparadores
ciertos, igual que :meth:`pyplanner.grounding.task.GroundTask.applicable`.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from pyplanner.grounding.numeric import NumericCondition
from pyplanner.grounding.task import Axiom, State, bits, mask_of


@dataclass(frozen=True)
class _Stratum:
    """Axiomas de un estrato, preparados para evaluarlos.

    Attributes:
        heads: hecho derivado de cada axioma.
        outer_pos, neg: parte externa proposicional de cada axioma (bitsets).
        numeric: condiciones numéricas de cada axioma.
        inner: número de hechos internos de cada axioma.
        always: axiomas sin hechos externos positivos (se revisan siempre).
        triggers: bitset de los disparadores; ``by_trigger`` da sus axiomas.
        consumers: axiomas que tienen cada hecho del estrato en su parte interna.
    """

    heads: tuple[int, ...]
    outer_pos: tuple[int, ...]
    neg: tuple[int, ...]
    numeric: tuple[tuple[NumericCondition, ...], ...]
    inner: tuple[int, ...]
    always: tuple[int, ...]
    triggers: int
    by_trigger: dict[int, tuple[int, ...]]
    consumers: dict[int, tuple[int, ...]]


class AxiomEvaluator:
    """Calcula los hechos derivados de un estado a partir de sus hechos básicos."""

    def __init__(self, axioms: Sequence[Axiom]) -> None:
        self.derived_mask = mask_of({axiom.head for axiom in axioms})
        by_stratum: dict[int, list[Axiom]] = {}
        for axiom in axioms:
            by_stratum.setdefault(axiom.stratum, []).append(axiom)
        self.strata = tuple(_prepare(by_stratum[s]) for s in sorted(by_stratum))

    def evaluate(self, state: State) -> State:
        """``state`` con sus hechos derivados recalculados (los que traiga se ignoran)."""
        facts = self.closure(state.facts & ~self.derived_mask, state.values)
        return state if facts == state.facts else State(facts, state.values)

    def closure(self, facts: int, values: tuple[float | None, ...]) -> int:
        """Los hechos básicos ``facts`` más todos los derivados que implican."""
        for stratum in self.strata:
            heads, outer_pos, neg = stratum.heads, stratum.outer_pos, stratum.neg
            numeric, inner = stratum.numeric, stratum.inner
            pending: dict[int, int] = {}  # axioma con la parte externa cierta → internos
            queue: list[int] = []
            candidates = [*stratum.always]
            by_trigger = stratum.by_trigger
            for fact in bits(facts & stratum.triggers):
                candidates.extend(by_trigger[fact])
            for i in candidates:
                pos = outer_pos[i]
                if facts & pos != pos or facts & neg[i]:
                    continue
                if numeric[i] and not all(c.holds(values) for c in numeric[i]):
                    continue
                if inner[i]:
                    pending[i] = inner[i]
                else:
                    queue.append(heads[i])
            if not queue:
                continue
            consumers = stratum.consumers
            while queue:
                fact = queue.pop()
                bit = 1 << fact
                if facts & bit:
                    continue
                facts |= bit
                for i in consumers.get(fact, ()):
                    remaining = pending.get(i)
                    if remaining is not None:
                        if remaining == 1:
                            del pending[i]
                            queue.append(heads[i])
                        else:
                            pending[i] = remaining - 1
        return facts


def _prepare(axioms: Sequence[Axiom]) -> _Stratum:
    own = mask_of({axiom.head for axiom in axioms})
    heads: list[int] = []
    outer_pos: list[int] = []
    neg: list[int] = []
    numeric: list[tuple[NumericCondition, ...]] = []
    inner: list[int] = []
    always: list[int] = []
    by_trigger: dict[int, list[int]] = {}
    consumers: dict[int, list[int]] = {}
    for i, axiom in enumerate(axioms):
        condition = axiom.condition
        outer = [f for f in condition.pos_facts if not own >> f & 1]
        internal = [f for f in condition.pos_facts if own >> f & 1]
        assert not condition.neg & own, "los axiomas no son estratificables"
        heads.append(axiom.head)
        outer_pos.append(mask_of(outer))
        neg.append(condition.neg)
        numeric.append(condition.numeric)
        inner.append(len(internal))
        for fact in internal:
            consumers.setdefault(fact, []).append(i)
        if outer:
            by_trigger.setdefault(outer[0], []).append(i)
        else:
            always.append(i)
    return _Stratum(
        heads=tuple(heads),
        outer_pos=tuple(outer_pos),
        neg=tuple(neg),
        numeric=tuple(numeric),
        inner=tuple(inner),
        always=tuple(always),
        triggers=mask_of(list(by_trigger)),
        by_trigger={f: tuple(ids) for f, ids in by_trigger.items()},
        consumers={f: tuple(ids) for f, ids in consumers.items()},
    )
