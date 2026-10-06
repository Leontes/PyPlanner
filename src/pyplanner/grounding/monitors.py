"""Restricciones de trayectoria duras (PDDL 3) como monitores del motor.

Una restricción se cumple sobre la **secuencia de estados** del plan: el inicial y el que
deja cada paso. Cada restricción instanciada es un pequeño autómata con hechos propios en el
estado (como los derivados, pero con memoria), que se actualiza tras cada paso con el estado
nuevo. Con φ y ψ las fórmulas de la restricción y s el estado nuevo:

==================== ================= ===================================================
Operador             Hechos            Actualización
==================== ================= ===================================================
``always φ``         —                 si ¬φ(s), el paso se descarta
``sometime φ``       ``done``          ``done`` si φ(s); el objetivo exige ``done``
``at-most-once φ``   ``seen, closed``  φ(s): si ``closed``, se descarta; si no, ``seen``.
                                       ¬φ(s): si ``seen``, ``closed``
``sometime-before``  ``seen``          si φ(s) y ψ no se ha cumplido en un estado anterior
(φ ψ)                                  (``seen``), se descarta; después, ``seen`` si ψ(s)
``sometime-after``   ``pending``       si ψ(s), se quita; si no, se pone si φ(s). El
(φ ψ)                                  objetivo exige que no quede pendiente
``at end φ``         —                 φ se añade al objetivo
==================== ================= ===================================================

El estado inicial se trata igual (sin estados anteriores); si incumple una restricción, la
tarea no tiene solución.

**Restricciones con tiempo.** Usan **marcas**: hechos que dicen si ya ha pasado un instante.
En una tarea temporal, cada instante de las restricciones es un TIL sin literales y su marca
es el hecho ``til-done@k``: por la regla de OPTIC, los pasos anteriores a un TIL pendiente en
la secuencia ocurren no más tarde que él. En una tarea clásica, el instante de un estado es
el número de pasos hasta él: un **contador** de pasos (saturado en el último instante que
importa) enciende las marcas ``after@n`` (n pasos o más).

==================== ================= ===================================================
``within t φ``       ``done``          ``done`` si φ(s) y no ha pasado t; el objetivo
                                       exige ``done``
``hold-after t φ``   —                 si ha pasado t y ¬φ(s), se descarta
``hold-during``      —                 si ha pasado t1, no t2, y ¬φ(s), se descarta
(t1 t2 φ)
``always-within``    ``pending`` (y    si φ(s) sin ψ(s), queda pendiente. En una tarea
(t φ ψ)              un contador en    clásica, un contador de pasos lo descarta si pasan
                     clásico)          más de t sin ψ; en una temporal, al cumplirse ψ la
                                       red temporal recibe la arista "ψ no más tarde que
                                       t después de φ". El objetivo exige que no quede
                                       pendiente
==================== ================= ===================================================

En una tarea clásica, "ha pasado t" es "el estado está a más de t pasos del inicial".

**Heurística.** Las salidas buenas de los monitores (``done``, quitar ``pending``) son
reglas relajadas de coste 0, como los axiomas: :class:`MonitorRule`. Las que solo descartan
estados (``always``, ``at-most-once``, ``sometime-before``) no cuentan en la relajación, que
así sigue siendo optimista.

**Coste.** Un monitor solo se reevalúa si el paso cambia algún hecho o variable de sus
fórmulas: si no, su actualización no cambiaría nada. Un índice de hecho → monitores evita
recorrerlos todos en cada paso (en storage hay cientos).

**Monitores blandos** (restricciones-preferencia, ver :mod:`pyplanner.grounding.preferences`).
Donde uno duro descartaría el paso, uno blando marca su hecho ``violated``, borra los demás
y deja de actualizarse: ya no importa qué pase después.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from pyplanner.grounding.numeric import variables_of
from pyplanner.grounding.task import Condition, State, bits, mask_of

#: Operadores de las restricciones sin tiempo que vigila un monitor (``at end`` va al objetivo).
UNTIMED = ("always", "sometime", "at-most-once", "sometime-before", "sometime-after")
#: Operadores con tiempo.
TIMED = ("within", "hold-after", "hold-during", "always-within")


@dataclass(frozen=True)
class Monitor:
    """Una restricción de trayectoria instanciada.

    Attributes:
        kind: el operador (ver la documentación del módulo).
        phi, psi: sus fórmulas en DNF (``psi`` solo en ``sometime-before`` y
            ``sometime-after``).
        facts: sus hechos (``done``; ``seen`` y ``closed``; ``seen``; ``pending``; y en
            ``always-within`` clásico, tras ``pending``, el contador de pasos pendientes).
        markers: marcas de los instantes que usa (``within`` y ``hold-after``: t;
            ``hold-during``: t1 y t2).
        bound: en ``always-within``, el plazo t.
        label: la restricción, para los mensajes.
        violated: en un monitor blando, el hecho que marca que se ha incumplido (-1 en uno
            duro).
    """

    kind: str
    phi: tuple[Condition, ...]
    psi: tuple[Condition, ...] = ()
    facts: tuple[int, ...] = ()
    markers: tuple[int, ...] = ()
    bound: float = 0.0
    label: str = ""
    violated: int = -1
    mask: int = field(init=False, repr=False, compare=False)
    variables: tuple[int, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        conditions = (*self.phi, *self.psi)
        watched = {f for c in conditions for f in (*c.pos_facts, *c.neg_facts)}
        object.__setattr__(self, "mask", mask_of(watched | set(self.markers)))
        object.__setattr__(
            self,
            "variables",
            tuple(
                sorted(
                    {v for c in conditions for n in c.numeric for v in variables_of(n.expression)}
                )
            ),
        )


@dataclass(frozen=True)
class MonitorRule:
    """Regla relajada de un monitor (para la heurística): si ``condition`` se cumple, añade
    ``add_facts`` y quita ``del_facts`` (en la relajación, hace cierto su complementario)."""

    condition: Condition
    add_facts: tuple[int, ...] = ()
    del_facts: tuple[int, ...] = ()


def _holds(conditions: Sequence[Condition], state: State) -> bool:
    return any(c.holds(state) for c in conditions)


@dataclass(frozen=True)
class StepClock:
    """Contador de pasos de una tarea clásica con restricciones con tiempo.

    Attributes:
        counter: hechos ``step@0`` … ``step@N``: exactamente uno es cierto (el último, para
            N pasos o más).
        marks: marcas ``(hecho, n)``: el hecho es cierto desde el paso n.
    """

    counter: tuple[int, ...]
    marks: tuple[tuple[int, int], ...]

    def initial(self, facts: int) -> int:
        facts |= 1 << self.counter[0]
        for fact, n in self.marks:
            if n <= 0:
                facts |= 1 << fact
        return facts

    def tick(self, facts: int) -> int:
        """Un paso más (saturado en el último hecho del contador)."""
        counter = self.counter
        for i, fact in enumerate(counter[:-1]):
            if facts >> fact & 1:
                facts = (facts & ~(1 << fact)) | (1 << counter[i + 1])
                for mark, n in self.marks:
                    if n == i + 1:
                        facts |= 1 << mark
                break
        return facts


class MonitorEvaluator:
    """Actualiza los hechos de los monitores tras cada paso."""

    def __init__(self, monitors: Sequence[Monitor], clock: StepClock | None = None) -> None:
        self.monitors = tuple(monitors)
        self.clock = clock
        watchers: dict[int, list[int]] = {}
        #: Monitores que se reevalúan en cada paso (su contador avanza siempre).
        self._every_step: list[int] = []
        #: Monitores con condiciones numéricas, por sus variables.
        self._numeric: list[tuple[int, tuple[int, ...]]] = []
        for i, monitor in enumerate(self.monitors):
            if monitor.kind == "always-within-steps":
                self._every_step.append(i)
                continue
            for fact in bits(monitor.mask):
                watchers.setdefault(fact, []).append(i)
            if monitor.variables:
                self._numeric.append((i, monitor.variables))
        self._watchers = {fact: tuple(group) for fact, group in watchers.items()}
        self._watched = mask_of(watchers)

    def initial(self, state: State) -> State | None:
        """El estado inicial con los hechos de los monitores, o ``None`` si incumple alguna
        restricción."""
        facts = state.facts
        if self.clock is not None:
            facts = self.clock.initial(facts)
            state = State(facts, state.values)
        for monitor in self.monitors:
            if monitor.violated < 0:
                result = self._step(monitor, facts, state)
            else:
                result = self._soft(monitor, facts, state)
            if result is None:
                return None
            facts = result
        return State(facts, state.values)

    def update(self, before: State, after: State) -> State | None:
        """``after`` con los hechos de los monitores actualizados, o ``None`` si el paso
        incumple alguna restricción."""
        facts = after.facts
        if self.clock is not None:
            facts = self.clock.tick(facts)
            after = State(facts, after.values)
        # Solo los monitores con algún hecho o variable cambiados (los demás valen lo mismo).
        changed = (before.facts ^ after.facts) & self._watched
        selected = set(self._every_step)
        watchers = self._watchers
        for fact in bits(changed):
            selected.update(watchers[fact])
        if self._numeric and before.values != after.values:
            selected.update(
                i
                for i, variables in self._numeric
                if any(before.values[v] != after.values[v] for v in variables)
            )
        monitors = self.monitors
        for i in sorted(selected):
            monitor = monitors[i]
            if monitor.violated < 0:
                result = self._step(monitor, facts, after)
            else:
                result = self._soft(monitor, facts, after)
            if result is None:
                return None
            facts = result
        return after if facts == after.facts else State(facts, after.values)

    @staticmethod
    def _soft(monitor: Monitor, facts: int, state: State) -> int:
        """Como :meth:`_step` para un monitor blando: marca ``violated`` en lugar de
        incumplirse (y ya no se actualiza más)."""
        violated = monitor.violated
        if facts >> violated & 1:
            return facts
        result = MonitorEvaluator._step(monitor, facts, state)
        if result is not None:
            return result
        for fact in monitor.facts:
            facts &= ~(1 << fact)
        return facts | (1 << violated)

    @staticmethod
    def _step(monitor: Monitor, facts: int, state: State) -> int | None:
        """Hechos tras actualizar un monitor con el estado ``state`` (``None``: incumplido).
        Las fórmulas no dependen de los hechos de los monitores, así que se evalúan sobre
        ``state``."""
        kind = monitor.kind
        if kind == "always":
            return facts if _holds(monitor.phi, state) else None
        if kind == "sometime":
            (done,) = monitor.facts
            return facts | (1 << done) if _holds(monitor.phi, state) else facts
        if kind == "at-most-once":
            seen, closed = monitor.facts
            if _holds(monitor.phi, state):
                if facts >> closed & 1:
                    return None
                return facts | (1 << seen)
            return facts | (1 << closed) if facts >> seen & 1 else facts
        if kind == "sometime-before":
            (seen,) = monitor.facts
            if _holds(monitor.phi, state) and not facts >> seen & 1:
                return None
            return facts | (1 << seen) if _holds(monitor.psi, state) else facts
        if kind == "sometime-after":
            (pending,) = monitor.facts
            if _holds(monitor.psi, state):
                return facts & ~(1 << pending)
            return facts | (1 << pending) if _holds(monitor.phi, state) else facts
        if kind == "within":
            (done,) = monitor.facts
            (passed,) = monitor.markers
            if not facts >> passed & 1 and _holds(monitor.phi, state):
                return facts | (1 << done)
            return facts
        if kind == "hold-after":
            (passed,) = monitor.markers
            if facts >> passed & 1 and not _holds(monitor.phi, state):
                return None
            return facts
        if kind == "hold-during":
            start, end = monitor.markers
            inside = facts >> start & 1 and not facts >> end & 1
            if inside and not _holds(monitor.phi, state):
                return None
            return facts
        if kind in ("always-within", "always-within-steps"):
            return MonitorEvaluator._always_within(monitor, facts, state)
        raise ValueError(f"monitor desconocido: {kind}")

    @staticmethod
    def _always_within(monitor: Monitor, facts: int, state: State) -> int | None:
        """``pending`` mientras φ espera a ψ (el disparo más antiguo marca el plazo).

        En una tarea clásica (``always-within-steps``), ``facts[1:]`` cuenta los pasos que
        lleva pendiente: con L = ⌊t⌋, ψ tiene que cumplirse como mucho L pasos después de
        φ, así que con L = 0 tiene que cumplirse en el mismo estado. En una temporal, el
        plazo lo comprueba la red temporal al cumplirse ψ."""
        pending, *elapsed = monitor.facts
        if _holds(monitor.psi, state):
            for fact in (pending, *elapsed):
                facts &= ~(1 << fact)
            return facts
        steps = monitor.kind == "always-within-steps"
        if facts >> pending & 1:
            if not steps:
                return facts
            for i, fact in enumerate(elapsed):
                if facts >> fact & 1:
                    if i + 1 == len(elapsed):
                        return None  # se agota el plazo sin ψ
                    return (facts & ~(1 << fact)) | (1 << elapsed[i + 1])
            return facts
        if _holds(monitor.phi, state):
            if steps and not elapsed:
                return None  # plazo menor que un paso: ψ tenía que cumplirse ya
            facts |= 1 << pending
            if steps:
                facts |= 1 << elapsed[0]
        return facts
