"""Preferencias (PDDL 3) compiladas a costes de acción (Keyder y Geffner, 2009).

Una preferencia es una condición deseable: cada vez que se incumple, la métrica suma el
coeficiente de su ``(is-violated nombre)`` (su **peso**). El grounder la compila de modo
que el coste de un plan sea su métrica, y la búsqueda solo tiene que minimizar costes.

**Preferencias del objetivo y de las restricciones** (se evalúan al final del plan). Cada
instancia (una por asignación de los ``forall`` que la rodean) se resuelve en una **fase
final**:

* ``_end`` (coste 0) se aplica cuando se cumple el objetivo duro y da paso a la fase final:
  ninguna acción del dominio se puede aplicar después (todas exigen ``¬_ended``);
* cada instancia se resuelve con ``_collect`` (coste 0), si se cumple, o con ``_forgo``
  (coste su peso), si no. Se resuelven en un orden fijo (cada una exige la anterior), así
  que la fase final no multiplica los caminos;
* el objetivo es haberlas resuelto todas.

Una restricción-preferencia es un monitor **blando** (ver
:mod:`pyplanner.grounding.monitors`): en lugar de descartar el paso que la incumple, marca
su hecho ``violated``, y ese paso cuesta además su peso (la **penalización** de la
transición, que añade la búsqueda): incumplida una vez, ya no se puede evitar. Así las que
solo se incumplen a mitad del plan (``always``, ``at-most-once``, ``sometime-before``,
``hold-after``, ``hold-during``) no necesitan fase final, y en storage son decenas de miles.
Las que se deciden al final (``sometime``, ``within``, lo pendiente de ``sometime-after`` y
``always-within``) se resuelven en la fase final, según ``done`` o ``¬pending``.
``(preference p (at end φ))`` es como una del objetivo.

**Preferencias de las precondiciones** (se cuentan cada vez que se aplica la acción). Cada
acción se desdobla según qué instancias cumple: la variante que incumple unas cuesta además
la suma de sus pesos (y exige que se incumplan, para que el coste sea exacto).

**Coste de los pasos.** Si la métrica solo mide preferencias, las acciones del dominio
costarían 0 y las heurísticas no distinguirían un plan corto de uno largo. Por eso cada
acción del dominio cuesta además ``step_cost`` (una milésima del menor peso positivo): así
se prefiere cumplir una preferencia a renunciar a ella, y entre dos planes igual de buenos,
el más corto. La métrica del plan descuenta ese coste.

Las acciones de la fase final van al final de ``GroundTask.actions`` (desde
``first_auxiliary``) y no salen en el plan. Sus nombres empiezan por ``_``, que no puede
empezar un nombre PDDL.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace

from pyplanner.grounding.task import Condition, GroundAction, GroundTask, State, bits, mask_of
from pyplanner.loader.ast import And, Atom, Forall, Formula, Parameter, Preference, preferences_in

#: Fracción del menor peso positivo que cuesta cada paso del plan (ver la documentación).
STEP_COST_FRACTION = 1e-3
#: Nombres de las acciones de la fase final.
END, COLLECT, FORGO = "_end", "_collect", "_forgo"
#: El hecho de la fase final. El grounder lo pone justo tras los hechos del dominio: todas
#: las acciones lo niegan, y con un índice bajo sus máscaras de bits son pequeñas.
ENDED = Atom("_ended")


@dataclass(frozen=True)
class PreferenceSite:
    """Una preferencia dentro de una fórmula, con los parámetros de los ``forall`` que la
    rodean (cada asignación es una instancia)."""

    parameters: tuple[Parameter, ...]
    preference: Preference

    @property
    def formula(self) -> Formula:
        """Su fórmula (en un objetivo o una precondición, siempre una fórmula)."""
        return self.preference.arg  # type: ignore[return-value]


def split_preferences(formula: Formula) -> tuple[Formula, list[PreferenceSite]]:
    """Separa una fórmula (un objetivo o una precondición) en su parte dura y sus
    preferencias, que solo pueden estar en su conjunción exterior y en sus ``forall``."""
    if next(preferences_in(formula), None) is None:
        return formula, []
    sites: list[PreferenceSite] = []

    def visit(node: Formula, parameters: tuple[Parameter, ...]) -> Formula:
        if isinstance(node, Preference):
            sites.append(PreferenceSite(parameters, node))
            return And(())
        if next(preferences_in(node), None) is None:
            return node
        if isinstance(node, And):
            args = [visit(arg, parameters) for arg in node.args]  # type: ignore[arg-type]
            return And(tuple(arg for arg in args if arg != And(())))
        assert isinstance(node, Forall)
        return Forall(node.parameters, visit(node.arg, (*parameters, *node.parameters)))

    return visit(formula, ()), sites


@dataclass(frozen=True)
class SoftGoal:
    """Una instancia de una preferencia que se evalúa al final del plan.

    Attributes:
        name: nombre de la preferencia.
        args: objetos de su instancia (los de los ``forall`` que la rodean).
        weight: su peso en la métrica.
        satisfied: cuándo se cumple (DNF sobre los hechos de la tarea).
        violated: cuándo no (DNF de la negación).
    """

    name: str
    args: tuple[str, ...]
    weight: float
    satisfied: tuple[Condition, ...]
    violated: tuple[Condition, ...]


@dataclass(frozen=True)
class PreferenceModel:
    """Lo que la búsqueda y el plan necesitan saber de las preferencias compiladas.

    Attributes:
        first_auxiliary: índice de la primera acción de la fase final (las de antes son las
            del dominio).
        step_cost: coste añadido a cada acción del dominio (ver la documentación).
        violations: hecho ``violated``, peso e instancia (``nombre arg1 arg2``) de cada
            restricción-preferencia que se puede incumplir a mitad del plan.
        weights: peso de cada preferencia, por nombre.
        end_actions: las acciones ``_end`` (una por cláusula del objetivo duro).
        resolutions: por cada preferencia de la fase final, en orden: sus acciones
            ``_collect`` y ``_forgo``, su peso y cuándo se cumple.
    """

    first_auxiliary: int
    step_cost: float
    violations: tuple[tuple[int, float, str], ...] = ()
    weights: tuple[tuple[str, float], ...] = ()
    end_actions: tuple[int, ...] = ()
    resolutions: tuple[tuple[tuple[int, ...], float, tuple[Condition, ...]], ...] = ()
    violation_mask: int = field(init=False, repr=False, compare=False)
    _weight_of: dict[int, float] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "violation_mask", mask_of(f for f, _, _ in self.violations))
        object.__setattr__(self, "_weight_of", {f: w for f, w, _ in self.violations})

    def penalty(self, before: State, after: State) -> float:
        """Lo que cuesta, además de la acción, pasar de ``before`` a ``after``: los pesos
        de las restricciones-preferencia que se incumplen en ese paso."""
        new = after.facts & ~before.facts & self.violation_mask
        if not new:
            return 0.0
        weight_of = self._weight_of
        return sum(weight_of[fact] for fact in bits(new))

    def committed(self, state: State) -> float:
        """Los pesos de las restricciones-preferencia que el estado ya incumple."""
        facts = state.facts & self.violation_mask
        weight_of = self._weight_of
        return sum(weight_of[fact] for fact in bits(facts)) if facts else 0.0

    def can_end(self, task: GroundTask, state: State) -> bool:
        """``True`` si en ``state`` se puede pasar a la fase final (se cumple el objetivo
        duro y no se ha pasado ya)."""
        return any(task.actions[i].applicable(state) for i in self.end_actions)

    def end_cost(self, state: State) -> float:
        """Lo que cuesta la fase final desde ``state``: los pesos de las preferencias que
        no se cumplen en él."""
        return sum(
            weight
            for _, weight, satisfied in self.resolutions
            if not any(clause.holds(state) for clause in satisfied)
        )

    def completion(self, task: GroundTask, state: State) -> list[int]:
        """Las acciones de la fase final desde ``state`` (donde se puede aplicar ``_end``):
        ``_end`` y, por orden, ``_collect`` o ``_forgo`` de cada preferencia."""
        plan: list[int] = []
        for indices in (self.end_actions, *(r[0] for r in self.resolutions)):
            for index in indices:
                action = task.actions[index]
                if action.applicable(state):
                    successor = action.apply(state)
                    assert successor is not None
                    state = successor
                    plan.append(index)
                    break
            else:
                raise AssertionError("la fase final no se puede completar")
        return plan

    def plan_actions(self, indices: Sequence[int]) -> list[int]:
        """Las acciones del dominio de un plan (sin las de la fase final)."""
        return [i for i in indices if i < self.first_auxiliary]

    def metric(self, task: GroundTask, indices: Sequence[int], final: State) -> float:
        """La métrica de un plan que acaba en ``final``, sin ``task.metric_offset``: las
        acciones sin su ``step_cost``, las renuncias de la fase final y las
        restricciones-preferencia incumplidas."""
        steps = len(self.plan_actions(indices))
        actions = sum(task.actions[i].cost for i in indices) - self.step_cost * steps
        return actions + self.committed(final)

    def violated(self, task: GroundTask, indices: Sequence[int], final: State) -> list[str]:
        """Las instancias de preferencias del objetivo y de las restricciones que incumple
        el plan que acaba en ``final``, como ``nombre arg1 arg2``."""
        facts = final.facts
        return [
            *(label for fact, _, label in self.violations if facts >> fact & 1),
            *(
                " ".join(task.actions[i].args)
                for i in indices
                if i >= self.first_auxiliary and task.actions[i].name == FORGO
            ),
        ]


def final_state(task: GroundTask, indices: Sequence[int]) -> State:
    """El estado al que lleva un plan (válido) de la tarea."""
    state = task.init
    for index in indices:
        successor = task.apply(state, task.actions[index])
        assert successor is not None, "el plan no es aplicable"
        state = successor
    return state


def step_cost(weights: Sequence[float]) -> float:
    """Coste de cada paso: una milésima del menor peso positivo (1 si no hay ninguno)."""
    positive = [w for w in weights if w > 0]
    return min(positive) * STEP_COST_FRACTION if positive else 1.0


def compile_soft_goals(
    task: GroundTask,
    goals: Sequence[SoftGoal],
    model: PreferenceModel,
) -> GroundTask:
    """Añade a la tarea la fase final de las preferencias del objetivo y de las
    restricciones (ver la documentación del módulo)."""
    if not goals:
        return replace(task, preferences=replace(model, first_auxiliary=len(task.actions)))
    facts = list(task.facts)
    ended = task.fact_index[ENDED]
    resolved = list(range(len(facts), len(facts) + len(goals)))
    facts.extend(Atom(f"_resolved#{i}") for i in range(len(goals)))

    actions = [
        replace(action, neg_facts=tuple(sorted({*action.neg_facts, ended})))
        for action in task.actions
    ]
    first_auxiliary = len(actions)

    def add(name: str, args: tuple[str, ...], condition: Condition, fact: int, cost: float) -> None:
        actions.append(
            GroundAction(
                index=len(actions),
                name=name,
                args=args,
                pre_facts=condition.pos_facts,
                add_facts=(fact,),
                del_facts=(),
                cost=cost,
                neg_facts=tuple(sorted({*condition.neg_facts, fact})),
                numeric_pre=condition.numeric,
            )
        )

    for clause in task.goal:
        add(END, (), clause, ended, 0.0)
    end_actions = tuple(range(first_auxiliary, len(actions)))
    resolutions = []
    previous = ended
    for goal, fact in zip(goals, resolved, strict=True):
        args = (goal.name, *goal.args)
        start = len(actions)
        for name, dnf, cost in (
            (COLLECT, goal.satisfied, 0.0),
            (FORGO, goal.violated, goal.weight),
        ):
            for clause in dnf:
                add(name, args, _with(clause, previous), fact, cost)
        resolutions.append((tuple(range(start, len(actions))), goal.weight, goal.satisfied))
        previous = fact

    return replace(
        task,
        facts=tuple(facts),
        actions=tuple(actions),
        goal=(Condition((previous,)),) if task.goal else (),
        preferences=replace(
            model,
            first_auxiliary=first_auxiliary,
            end_actions=end_actions,
            resolutions=tuple(resolutions),
        ),
    )


def _with(clause: Condition, fact: int) -> Condition:
    return Condition(tuple(sorted({*clause.pos_facts, fact})), clause.neg_facts, clause.numeric)


def negations(condition: Condition) -> Iterator[Condition]:
    """La negación de una conjunción de hechos como DNF: una cláusula por literal."""
    assert not condition.numeric
    for fact in condition.pos_facts:
        yield Condition(neg_facts=(fact,))
    for fact in condition.neg_facts:
        yield Condition(pos_facts=(fact,))
