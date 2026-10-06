"""Instanciación (grounding) con análisis de alcanzabilidad relajada y normalización ADL.

En lugar de probar todas las combinaciones de objetos para cada acción, el grounder parte
de los hechos del estado inicial y va instanciando solo las acciones cuyas precondiciones
son **alcanzables** en la relajación sin borrados. Los efectos de esas acciones hacen
alcanzables hechos nuevos, y el proceso se repite hasta llegar a un punto fijo.

1. **Candidatos:** para cada esquema se buscan las asignaciones de parámetros que hacen
   alcanzables sus *precondiciones necesarias* (los átomos positivos de la conjunción
   exterior) y respetan los tipos. Los parámetros que no aparecen en ellas toman todos los
   objetos de su tipo. Es un join: cada vez se toma el átomo con más posiciones ya ligadas
   (y, a igualdad, el de menos hechos), y sus hechos se buscan en un índice por esas
   posiciones.
2. **Normalización:** cada candidato se normaliza (:mod:`pyplanner.grounding.normalize`).
   La precondición pasa a DNF y los efectos a efectos simples con condición. Se hace una
   sola vez por candidato.
3. **Punto fijo:** una cláusula de la precondición es alcanzable si lo son sus hechos
   positivos; los negativos se suponen alcanzables, como corresponde a una relajación.
   Sus efectos incondicionales, y los condicionales con alguna cláusula alcanzable, hacen
   alcanzables sus añadidos.

Después:

* se eliminan los hechos **estáticos** (los de predicados que ninguna acción modifica),
  que la normalización ya ha evaluado;
* los literales negativos sobre hechos inalcanzables se eliminan, porque siempre son
  ciertos, y las cláusulas con hechos positivos inalcanzables, porque nunca lo son;
* cada cláusula alcanzable de la precondición da una :class:`GroundAction`. En cada
  efecto condicional se quitan los literales que ya garantiza la precondición, y si no le
  queda ninguno pasa a ser incondicional;
* si ninguna cláusula del objetivo es alcanzable, el objetivo queda vacío (imposible).

**Predicados derivados** (iteración 4). Cada regla ``:derived`` se instancia como un
esquema más del punto fijo: su condición hace de precondición y su cabeza, de añadido. Cada
cláusula alcanzable de su condición da un :class:`~pyplanner.grounding.task.Axiom`, con el
estrato de su predicado. Los predicados derivados nunca son estáticos (ninguna acción los
modifica, pero cambian con los hechos de los que dependen), y el estado inicial trae ya
calculados sus hechos.

**Preferencias** (iteración 5). Se separan del objetivo y de las precondiciones antes de
instanciar, y se compilan a costes con los pesos de la métrica (ver
:mod:`pyplanner.grounding.preferences`).
"""

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import product
from typing import NamedTuple

from pyplanner.errors import UnsupportedFeatureError
from pyplanner.grounding.normalize import (
    DNF,
    TRUE,
    Clause,
    Fact,
    Normalizer,
    SimpleEffect,
    UndefinedEffect,
)
from pyplanner.grounding.numeric import (
    Const,
    NumericAssignment,
    NumericCondition,
    NumExpr,
    Var,
    linear,
    make,
    remap,
    variables_of,
)
from pyplanner.grounding.preferences import (
    ENDED,
    PreferenceModel,
    SoftGoal,
    compile_soft_goals,
    negations,
    split_preferences,
    step_cost,
)
from pyplanner.grounding.task import (
    Axiom,
    Condition,
    ConditionalEffect,
    GroundAction,
    GroundTask,
    State,
    mask_of,
)
from pyplanner.loader.ast import (
    OBJECT,
    Action,
    And,
    Arithmetic,
    Atom,
    Comparison,
    Constant,
    Constraint,
    ConstraintAnd,
    ConstraintForall,
    Domain,
    Effect,
    Exists,
    Expression,
    FluentTerm,
    Forall,
    ForallEffect,
    Formula,
    Imply,
    IsViolated,
    Not,
    NumericEffect,
    Or,
    Preference,
    Problem,
    TotalTime,
    TrajectoryConstraint,
    Variable,
    When,
    preferences_in,
)

#: Asignación de los parámetros de un esquema (nombre de la variable → objeto).
Binding = dict[str, str]


def ground(domain: Domain, problem: Problem) -> GroundTask:
    """Instancia y normaliza un dominio y un problema.

    Si el dominio tiene acciones durativas o el problema literales iniciales temporizados,
    devuelve una tarea temporal
    (:func:`pyplanner.grounding.temporal.ground_temporal`).

    Returns:
        La tarea instanciada, con hechos, variables y acciones en un orden determinista.

    Raises:
        UnsupportedFeatureError: si una fórmula es demasiado grande en forma normal
            disyuntiva, o si usa algo que el motor temporal todavía no admite.
    """
    if domain.durative_actions or problem.timed_literals:
        if _has_preferences(domain, problem):
            raise UnsupportedFeatureError(
                "el motor de búsqueda todavía no admite preferencias en tareas temporales"
            )
        from pyplanner.grounding.temporal import ground_temporal

        return ground_temporal(domain, problem, _marker_times(domain, problem))
    return ground_with_context(domain, problem)[0]


def _marker_times(domain: Domain, problem: Problem) -> frozenset[float]:
    """Instantes de las restricciones con tiempo (``within``, ``hold-after`` y
    ``hold-during``), que en una tarea temporal son TIL sin literales."""
    times: set[float] = set()

    def visit(node: object) -> None:
        if isinstance(node, ConstraintAnd):
            for arg in node.args:
                visit(arg)
        elif isinstance(node, ConstraintForall | Preference):
            visit(node.arg)
        elif isinstance(node, TrajectoryConstraint) and node.op != "always-within":
            times.update(node.times)

    visit(domain.constraints)
    visit(problem.constraints)
    return frozenset(times)


def _has_preferences(domain: Domain, problem: Problem) -> bool:
    nodes = [
        problem.goal,
        problem.constraints,
        *(a.precondition for a in domain.actions),
        *(a.condition for a in domain.durative_actions),
    ]
    return any(next(preferences_in(node), None) is not None for node in nodes)


@dataclass
class GroundingContext:
    """Lo necesario para instanciar, después del grounding, fórmulas y expresiones de un
    esquema con una asignación de sus parámetros (lo usa el motor temporal para las
    invariantes y las duraciones)."""

    normalizer: Normalizer
    fact_index: Mapping[Fact, int]
    var_index: Mapping[Fact, int]

    def conditions(self, formula: Formula, binding: Binding) -> tuple[Condition, ...]:
        """La fórmula en DNF sobre los hechos y variables de la tarea. Las cláusulas con
        hechos inalcanzables se descartan; una tupla vacía es una fórmula imposible."""
        return tuple(
            _condition(clause, self.fact_index, self.var_index)
            for clause in _sorted_clauses(self.normalizer.dnf(formula, binding))
            if all(f in self.fact_index for f in clause.pos)
        )

    def expression(self, expression: Expression, binding: Binding) -> NumExpr | None:
        """La expresión sobre las variables de la tarea, o ``None`` si usa una función
        estática sin valor."""
        result = self.normalizer.expression(expression, binding)
        return None if result is None else remap(result, self.var_index.__getitem__)


def ground_with_context(
    domain: Domain,
    problem: Problem,
    extra_reads: Mapping[str, tuple[Expression, ...]] | None = None,
    extra_targeted: frozenset[str] = frozenset(),
    time_markers: Mapping[float, str] | None = None,
) -> tuple[GroundTask, GroundingContext]:
    """Como :func:`ground`, para un dominio sin acciones durativas, pero devuelve también
    el contexto de grounding.

    Args:
        extra_reads: expresiones que lee cada esquema además de sus condiciones (por nombre
            de esquema). Sus funciones cuentan para la relevancia, y sus términos pasan a
            ser variables del estado.
        extra_targeted: funciones que se modifican fuera de los efectos de las acciones
            (con efectos continuos): no son estáticas.
        time_markers: en una tarea temporal, el hecho que marca que ha pasado cada instante
            de las restricciones con tiempo (un TIL sin literales). Sin él, la tarea es
            clásica y el instante de un estado es su número de pasos.
    """
    extra_reads = extra_reads or {}
    declarations = {o.name: o.types for o in (*domain.constants, *problem.objects)}
    universe = _TypedUniverse(domain, declarations)
    fluent_predicates = domain.derived_predicates | frozenset(
        predicate for action in domain.actions for predicate in _modified(action.effect)
    )
    static_predicates = frozenset(p.name for p in domain.predicates) - fluent_predicates
    init_facts = {(a.predicate, _args(a)) for a in problem.init}
    targeted = extra_targeted | frozenset(
        effect.fluent.function
        for action in domain.actions
        for effect in _numeric_effects(action.effect)
    )
    static_functions = frozenset(f.name for f in domain.functions) - targeted
    constraints = [c for c in (domain.constraints, problem.constraints) if c is not None]
    relevant = _relevant_functions(domain, problem, targeted, extra_reads, constraints)
    initial_values = {(t.function, _args(t)): v for t, v in problem.values}
    normalizer = Normalizer(
        universe.objects_of,
        static_predicates,
        frozenset(f for f in init_facts if f[0] in static_predicates),
        static_functions,
        {k: v for k, v in initial_values.items() if k[0] in static_functions},
    )
    metric = _MetricModel(problem, normalizer, initial_values)
    goal_formula, goal_sites = split_preferences(problem.goal)
    schemas = [_Schema(action) for action in domain.actions]
    # Cada regla de un predicado derivado, como un esquema que "añade" su cabeza. Los
    # nombres no chocan con los de las acciones porque empiezan por ':'.
    schemas += [
        _Schema(
            Action(
                f":derived{i}",
                rule.parameters,
                rule.condition,
                Atom(rule.predicate, tuple(p.variable for p in rule.parameters)),
            )
        )
        for i, rule in enumerate(domain.derived)
    ]
    weights = metric.preference_weights(_has_preferences(domain, problem))

    reachable: dict[str, set[tuple[str, ...]]] = {}
    reached: set[Fact] = set()
    for predicate, args in init_facts:
        reachable.setdefault(predicate, set()).add(args)
        reached.add((predicate, args))

    instances: dict[tuple[str, tuple[str, ...]], _Instance] = {}
    changed = True
    while changed:
        changed = False
        for schema in schemas:
            # Se materializan antes de añadir hechos: el join recorre ``reachable``.
            for binding in list(schema.bindings(reachable, universe)):
                key = (schema.name, tuple(binding[p] for p in schema.parameters))
                instance = instances.get(key)
                if instance is None:
                    instance = _normalize(normalizer, schema, binding)
                    instances[key] = instance
                for fact in list(instance.newly_reachable(reached)):
                    reached.add(fact)
                    reachable.setdefault(fact[0], set()).add(fact[1])
                    changed = True

    facts = sorted(f for f in reached if f[0] in fluent_predicates)
    if weights:
        facts.append((ENDED.predicate, ()))  # ver ENDED: justo tras los hechos del dominio
    index = {fact: i for i, fact in enumerate(facts)}
    goal_dnf = _sorted_clauses(normalizer.dnf(goal_formula, {}))
    live = {key: inst for key, inst in instances.items() if inst.reachable_clauses(reached)}
    axiom_instances = {key: inst for key, inst in live.items() if key[0].startswith(":")}
    live = {key: inst for key, inst in live.items() if key not in axiom_instances}
    schema_of = {schema.name: schema for schema in schemas}
    # Preferencias de las precondiciones de cada instancia: (se cumple, no se cumple, peso).
    choices = {
        key: _preference_choices(normalizer, schema_of[key[0]], key[1], weights)
        for key in live
        if schema_of[key[0]].preferences
    }
    trajectory = _constraint_instances(constraints, universe, reachable, weights)
    soft_goal_sites = [
        (site, binding)
        for site in goal_sites
        if weights.get(site.preference.name or "", 0.0) > 0
        for binding in normalizer.expand(site.parameters, {})
    ]

    # Variables numéricas: los términos de funciones relevantes que aparecen en las acciones
    # alcanzables, en el objetivo, en las preferencias o en las restricciones.
    keys: set[Fact] = set()
    for instance in (*live.values(), *axiom_instances.values()):
        keys |= instance.numeric_keys()
    clauses: list[Clause] = list(goal_dnf)
    for options in choices.values():
        for satisfied, violated, _ in options:
            clauses += [*satisfied, *violated]
    formulas = [
        *((site.formula, binding) for site, binding in soft_goal_sites),
        *((formula, item.binding) for item in trajectory for formula in item.node.formulas),
    ]
    for formula, binding in formulas:
        if next(_functions_in(formula), None) is not None:
            clauses += normalizer.dnf(formula, binding)
    for clause in clauses:
        for condition in clause.numeric:
            keys |= variables_of(condition.expression)
    parameters = {schema.name: schema.parameters for schema in schemas}
    for (name, args), _ in live.items():
        for expression in extra_reads.get(name, ()):
            ground_expression = normalizer.expression(
                expression, dict(zip(parameters[name], args, strict=True))
            )
            if ground_expression is not None:
                keys |= variables_of(ground_expression)
    variables = sorted(k for k in keys if k[0] in relevant)
    var_index = {key: i for i, key in enumerate(variables)}

    costs = {key: metric.cost(instance.effects) for key, instance in live.items()}
    metric_is_cost = metric.active and all(c is not None and c >= 0 for c in costs.values())
    notes = list(metric.notes)
    if metric.active and not metric_is_cost:
        notes.append(
            "la contribución de alguna acción a la métrica no es una constante no negativa: "
            "se busca con coste unitario"
        )
    model = None
    if weights:
        if not metric_is_cost:
            raise UnsupportedFeatureError(
                "con preferencias, la contribución de cada acción a la métrica tiene que ser "
                "una constante no negativa"
            )
        model = PreferenceModel(
            0, step_cost(list(weights.values())), weights=tuple(weights.items())
        )
    step = model.step_cost if model is not None else 0.0

    actions: list[GroundAction] = []
    for key, instance in sorted(live.items()):
        name, args = key
        instance_cost = costs[key]
        cost = instance_cost if metric_is_cost and instance_cost is not None else 1.0
        for clause in instance.reachable_clauses(reached):
            for variant, penalty in _variants(clause, choices.get(key, []), reached):
                action = _build_action(
                    len(actions),
                    name,
                    args,
                    variant,
                    instance.effects,
                    index,
                    var_index,
                    cost + penalty + step,
                )
                if action is not None:
                    actions.append(action)

    axioms = _build_axioms(domain, axiom_instances, reached, index, var_index)
    goal = [_condition(clause, index, var_index) for clause in goal_dnf if clause.pos <= reached]
    init_values = tuple(initial_values.get(key) for key in variables)
    init = State(mask_of([index[f] for f in init_facts if f in index]), init_values)
    if axioms:
        from pyplanner.grounding.axioms import AxiomEvaluator

        init = AxiomEvaluator(axioms).evaluate(init)
    task = GroundTask(
        domain_name=domain.name,
        problem_name=problem.name,
        facts=tuple(Atom(pred, tuple(Constant(a) for a in args)) for pred, args in facts),
        actions=tuple(actions),
        init=init,
        goal=tuple(dict.fromkeys(goal)),
        variables=tuple(FluentTerm(f, tuple(Constant(a) for a in args)) for f, args in variables),
        metric_is_cost=metric_is_cost,
        metric_offset=metric.offset if metric_is_cost else None,
        notes=tuple(notes),
        axioms=axioms,
    )
    context = GroundingContext(normalizer, index, var_index)
    soft_goals = [
        SoftGoal(
            site.preference.name or "",
            tuple(binding[p.name] for p in site.parameters),
            weights[site.preference.name or ""],
            context.conditions(site.formula, binding),
            context.conditions(Not(site.formula), binding),
        )
        for site, binding in soft_goal_sites
    ]
    soft_goals = [g for g in soft_goals if not any(c == Condition() for c in g.satisfied)]
    violations: list[tuple[int, float, str]] = []
    if trajectory:
        task, soft_constraints, violations = _attach_monitors(
            task, context, trajectory, time_markers
        )
        soft_goals += soft_constraints
    if model is not None:
        model = replace(model, violations=tuple(violations))
        task = compile_soft_goals(task, soft_goals, model)
    return task, context


class _ConstraintInstance(NamedTuple):
    """Una restricción de trayectoria instanciada; con ``preference``, es blanda y cuesta
    ``weight`` si se incumple."""

    node: TrajectoryConstraint
    binding: Binding
    preference: str | None = None
    weight: float = 0.0


def _constraint_instances(
    constraints: list[Constraint],
    universe: "_TypedUniverse",
    reachable: Mapping[str, set[tuple[str, ...]]],
    weights: Mapping[str, float],
) -> list[_ConstraintInstance]:
    """Las restricciones instanciadas (sin las preferencias que no cuentan en la métrica).

    Un ``forall`` sobre una restricción que solo se puede incumplir (``always``,
    ``hold-after``, ``hold-during``) o disparar (``at-most-once``, ``sometime-before``,
    ``sometime-after``, ``always-within``) con ciertos átomos se instancia como una acción:
    solo con las asignaciones que hacen alcanzables esos átomos. Las demás serían
    restricciones que se cumplen siempre (en storage, la inmensa mayoría de los pares de
    cajas y zonas de ``(always (imply (and (on ?c1 ?s1) (on ?c2 ?s2) (connected ?s1 ?s2))
    ...))``).
    """
    instances: list[_ConstraintInstance] = []

    def visit(node: Constraint, binding: Binding, preference: str | None) -> None:
        if isinstance(node, ConstraintAnd):
            for arg in node.args:
                visit(arg, binding, preference)
        elif isinstance(node, ConstraintForall):
            for extended in _forall_bindings(node, binding, universe, reachable):
                visit(node.arg, extended, preference)
        elif isinstance(node, Preference):
            name = node.name or ""
            if weights.get(name, 0.0) > 0:
                visit(node.arg, binding, name)  # type: ignore[arg-type]
        else:
            weight = weights[preference] if preference is not None else 0.0
            instances.append(_ConstraintInstance(node, binding, preference, weight))

    for constraint in constraints:
        visit(constraint, {}, None)
    return instances


def _forall_bindings(
    node: ConstraintForall,
    binding: Binding,
    universe: "_TypedUniverse",
    reachable: Mapping[str, set[tuple[str, ...]]],
) -> Iterator[Binding]:
    """Las asignaciones de un ``forall`` de restricciones que pueden dar una restricción no
    trivial (ver :func:`_constraint_instances`)."""
    body = node.arg.arg if isinstance(node.arg, Preference) else node.arg
    atoms: list[Atom] = []
    if isinstance(body, TrajectoryConstraint):
        if body.op in ("always", "hold-after", "hold-during"):
            atoms = list(_atoms_when_false(body.formulas[0]))
        elif body.op in ("at-most-once", "sometime-before", "sometime-after", "always-within"):
            atoms = list(_atoms_when_true(body.formulas[0]))
    if not atoms:
        names = [p.name for p in node.parameters]
        domains = [universe.objects_of(p.types) for p in node.parameters]
        for values in product(*domains):
            yield {**binding, **dict(zip(names, values, strict=True))}
        return
    # Las variables de los forall exteriores ya tienen valor.
    atoms = [
        Atom(
            atom.predicate,
            tuple(
                Constant(binding[t.name]) if isinstance(t, Variable) and t.name in binding else t
                for t in atom.args
            ),
        )
        for atom in atoms
    ]
    schema = _Schema(Action(":constraint", node.parameters, And(tuple(atoms)), And(())))
    names = [p.name for p in node.parameters]
    # En orden de objetos, como el producto: el join recorre conjuntos, cuyo orden cambia
    # de una ejecución a otra.
    for found in sorted(schema.bindings(reachable, universe), key=lambda b: [b[n] for n in names]):
        yield {**binding, **found}


def _atoms_when_true(formula: Formula) -> Iterator[Atom]:
    """Átomos que tienen que ser ciertos para que ``formula`` lo sea (una aproximación
    segura: solo los que se ven sin distribuir disyunciones)."""
    if isinstance(formula, Atom):
        yield formula
    elif isinstance(formula, And):
        for arg in formula.args:
            yield from _atoms_when_true(arg)  # type: ignore[arg-type]
    elif isinstance(formula, Not):
        yield from _atoms_when_false(formula.arg)
    elif isinstance(formula, Exists):
        yield from _atoms_when_true(formula.arg)


def _atoms_when_false(formula: Formula) -> Iterator[Atom]:
    """Átomos que tienen que ser ciertos para que ``formula`` sea falsa."""
    if isinstance(formula, Not):
        yield from _atoms_when_true(formula.arg)
    elif isinstance(formula, Or):
        for arg in formula.args:
            yield from _atoms_when_false(arg)
    elif isinstance(formula, Imply):
        yield from _atoms_when_true(formula.left)
        yield from _atoms_when_false(formula.right)
    elif isinstance(formula, Forall):
        yield from _atoms_when_false(formula.arg)


#: Máximo de variantes de una acción por sus preferencias de precondición.
MAX_PREFERENCE_VARIANTS = 256


def _preference_choices(
    normalizer: Normalizer, schema: "_Schema", args: tuple[str, ...], weights: Mapping[str, float]
) -> list[tuple[DNF, DNF, float]]:
    """Las instancias de las preferencias de la precondición de una acción: cuándo se
    cumplen, cuándo no y su peso (sin las que no cuentan en la métrica)."""
    binding = dict(zip(schema.parameters, args, strict=True))
    choices: list[tuple[DNF, DNF, float]] = []
    for site in schema.preferences:
        weight = weights.get(site.preference.name or "", 0.0)
        if weight <= 0:
            continue
        for extended in normalizer.expand(site.parameters, binding):
            satisfied = normalizer.dnf(site.formula, extended)
            violated = normalizer.dnf(site.formula, extended, negated=True)
            choices.append((satisfied, violated, weight))
    return choices


def _variants(
    clause: Clause, choices: list[tuple[DNF, DNF, float]], reached: set[Fact]
) -> list[tuple[Clause, float]]:
    """Las variantes de una cláusula de la precondición según qué preferencias cumple, con
    la penalización de las que incumple."""
    variants = [(clause, 0.0)]
    for satisfied, violated, weight in choices:
        extended: list[tuple[Clause, float]] = []
        for base, penalty in variants:
            for options, extra in ((satisfied, 0.0), (violated, weight)):
                for option in options:
                    merged = base.conjoin(option)
                    if merged is not None and merged.pos <= reached:
                        extended.append((merged, penalty + extra))
        variants = extended
        if len(variants) > MAX_PREFERENCE_VARIANTS:
            raise UnsupportedFeatureError(
                "demasiadas combinaciones de preferencias en la precondición de una acción "
                f"(más de {MAX_PREFERENCE_VARIANTS})"
            )
    return variants


def _attach_monitors(
    task: GroundTask,
    context: "GroundingContext",
    instances: list[_ConstraintInstance],
    time_markers: Mapping[float, str] | None,
) -> tuple[GroundTask, list[SoftGoal], list[tuple[int, float, str]]]:
    """La tarea con las restricciones de trayectoria como monitores (ver
    :mod:`pyplanner.grounding.monitors`): sus hechos, sus reglas relajadas y lo que añaden
    al objetivo. ``time_markers`` son las marcas de los instantes en una tarea temporal; en
    una clásica, el instante es el número de pasos y se añade un contador.

    Las restricciones-preferencia son monitores blandos: devuelve también cuándo se cumplen
    al final las que se deciden al final (para la fase final de las preferencias), y el
    hecho ``violated``, el peso y la instancia de las que se pueden incumplir a mitad del
    plan (su peso se cobra en el paso que las incumple).
    """
    from pyplanner.grounding.monitors import (
        Monitor,
        MonitorEvaluator,
        MonitorRule,
        StepClock,
    )

    facts = list(task.facts)

    def new_fact(name: str) -> int:
        facts.append(Atom(name))
        return len(facts) - 1

    # Marcas de los instantes: en una tarea temporal, los hechos til-done@k de los TIL; en
    # una clásica, after@n (n pasos o más), que enciende el contador de pasos.
    marks: dict[int, int] = {}  # clásica: n → hecho

    def passed(time: float, strict: bool) -> int:
        """Hecho cierto desde que ha pasado ``time`` (con ``strict``, desde que el instante
        es mayor que ``time``; si no, mayor o igual)."""
        if time_markers is not None:
            return task.fact_index[Atom(time_markers[time])]
        n = math.floor(time) + 1 if strict else math.ceil(time)
        if n not in marks:
            marks[n] = new_fact(f"after@{n}")
        return marks[n]

    monitors: list[Monitor] = []
    rules: list[MonitorRule] = []
    goal_pos: set[int] = set()
    goal_neg: set[int] = set()
    at_end: list[tuple[Condition, ...]] = []
    soft_goals: list[SoftGoal] = []
    violations: list[tuple[int, float, str]] = []
    for k, (node, binding, preference, weight) in enumerate(instances):
        dnfs = [context.conditions(formula, binding) for formula in node.formulas]
        if _trivial(node.op, dnfs):
            continue
        label = str(node) + (f" con {binding}" if binding else "")
        tag = f"{node.op}#{k}"
        soft = preference is not None
        if soft and node.op == "at-end":
            soft_goals.append(
                SoftGoal(
                    preference or "",
                    tuple(binding.values()),
                    weight,
                    dnfs[0],
                    context.conditions(Not(node.formulas[0]), binding),
                )
            )
            continue
        # En una tarea clásica, always-within se incumple al agotar su plazo.
        failing = node.op in _FAILING or (node.op == "always-within" and time_markers is None)
        violated = new_fact(f"{tag}-violated") if soft and failing else -1
        end_pos: set[int] = set()  # lo que exige al final (en el objetivo, si es dura)
        end_neg: set[int] = set()
        if node.op == "at-end":
            at_end.append(dnfs[0])
        elif node.op == "always":
            monitors.append(Monitor("always", dnfs[0], label=label, violated=violated))
        elif node.op == "sometime":
            done = new_fact(f"{tag}-done")
            monitors.append(Monitor("sometime", dnfs[0], facts=(done,), label=label))
            rules.extend(MonitorRule(c, add_facts=(done,)) for c in dnfs[0])
            end_pos.add(done)
        elif node.op == "at-most-once":
            seen, closed = new_fact(f"{tag}-seen"), new_fact(f"{tag}-closed")
            monitors.append(
                Monitor(
                    "at-most-once",
                    dnfs[0],
                    facts=(seen, closed),
                    label=label,
                    violated=violated,
                )
            )
        elif node.op == "sometime-before":
            seen = new_fact(f"{tag}-seen")
            monitors.append(
                Monitor(node.op, dnfs[0], dnfs[1], facts=(seen,), label=label, violated=violated)
            )
        elif node.op == "sometime-after":
            pending = new_fact(f"{tag}-pending")
            monitors.append(Monitor(node.op, dnfs[0], dnfs[1], facts=(pending,), label=label))
            rules.extend(MonitorRule(c, del_facts=(pending,)) for c in dnfs[1])
            end_neg.add(pending)
        elif node.op == "within":
            done = new_fact(f"{tag}-done")
            marker = passed(node.times[0], strict=True)
            monitors.append(
                Monitor("within", dnfs[0], facts=(done,), markers=(marker,), label=label)
            )
            rules.extend(
                MonitorRule(
                    Condition(c.pos_facts, tuple(sorted({*c.neg_facts, marker})), c.numeric),
                    add_facts=(done,),
                )
                for c in dnfs[0]
            )
            end_pos.add(done)
        elif node.op == "hold-after":
            marker = passed(node.times[0], strict=True)
            monitors.append(
                Monitor(node.op, dnfs[0], markers=(marker,), label=label, violated=violated)
            )
        elif node.op == "hold-during":
            start, end = passed(node.times[0], strict=False), passed(node.times[1], strict=False)
            monitors.append(
                Monitor(node.op, dnfs[0], markers=(start, end), label=label, violated=violated)
            )
        else:  # always-within
            pending = new_fact(f"{tag}-pending")
            bound = node.times[0]
            if time_markers is None:  # clásica: un contador de los pasos que lleva pendiente
                kind = "always-within-steps"
                elapsed = tuple(new_fact(f"{tag}-elapsed@{i}") for i in range(math.floor(bound)))
                monitor_facts = (pending, *elapsed)
            else:
                kind, monitor_facts = "always-within", (pending,)
            monitors.append(
                Monitor(
                    kind,
                    dnfs[0],
                    dnfs[1],
                    facts=monitor_facts,
                    bound=bound,
                    label=label,
                    violated=violated,
                )
            )
            rules.extend(MonitorRule(c, del_facts=(pending,)) for c in dnfs[1])
            end_neg.add(pending)
        if not soft:
            goal_pos |= end_pos
            goal_neg |= end_neg
            continue
        if violated >= 0:
            violations.append((violated, weight, " ".join((preference or "", *binding.values()))))
        if not end_pos and not end_neg:
            continue  # se decide a mitad del plan: no necesita fase final
        satisfied = Condition(tuple(sorted(end_pos)), tuple(sorted(end_neg)))
        soft_goals.append(
            SoftGoal(
                preference or "",
                tuple(binding.values()),
                weight,
                (satisfied,),
                tuple(negations(satisfied)),
            )
        )

    clock = None
    if marks:
        horizon = max(marks)
        counter = tuple(new_fact(f"step@{i}") for i in range(horizon + 1))
        clock = StepClock(counter, tuple(sorted((fact, n) for n, fact in marks.items())))

    goal = [
        Condition(
            tuple(sorted({*clause.pos_facts, *goal_pos})),
            tuple(sorted({*clause.neg_facts, *goal_neg})),
            clause.numeric,
        )
        for clause in task.goal
    ]
    for dnf in at_end:
        goal = [
            merged for clause in goal for c in dnf if (merged := _conjoin(clause, c)) is not None
        ]
    notes = list(task.notes)
    init = MonitorEvaluator(monitors, clock).initial(task.init)
    if init is None:
        init = task.init
        goal = []
        notes.append("el estado inicial incumple una restricción de trayectoria: no hay plan")
    task = replace(
        task,
        facts=tuple(facts),
        init=init,
        goal=tuple(dict.fromkeys(goal)),
        notes=tuple(notes),
        monitors=tuple(monitors),
        monitor_rules=tuple(rules),
        monitor_clock=clock,
    )
    return task, soft_goals, violations


def _trivial(op: str, dnfs: list[tuple[Condition, ...]]) -> bool:
    """``True`` si la restricción se cumple en cualquier plan porque sus fórmulas, con los
    hechos estáticos y los inalcanzables, quedan en cierto o en falso (no se buscan
    tautologías como ``(or p (not p))``)."""
    always = [any(c == _TRUE_CLAUSE for c in dnf) for dnf in dnfs]
    never = [not dnf for dnf in dnfs]
    if op in ("at-end", "always", "sometime", "within", "hold-after", "hold-during"):
        return always[0]
    if op == "at-most-once":
        return always[0] or never[0]
    if op == "sometime-before":
        return never[0]
    return never[0] or always[1]  # sometime-after, always-within


#: La cláusula vacía (siempre cierta).
_TRUE_CLAUSE = Condition()

#: Restricciones que un paso puede incumplir para siempre (las demás se deciden al final):
#: como preferencias, llevan un hecho ``violated``.
_FAILING = frozenset({"always", "at-most-once", "sometime-before", "hold-after", "hold-during"})


def _conjoin(a: Condition, b: Condition) -> Condition | None:
    """La conjunción de dos cláusulas, o ``None`` si se contradicen."""
    pos = {*a.pos_facts, *b.pos_facts}
    neg = {*a.neg_facts, *b.neg_facts}
    if pos & neg:
        return None
    numeric = tuple(dict.fromkeys((*a.numeric, *b.numeric)))
    return Condition(tuple(sorted(pos)), tuple(sorted(neg)), numeric)


def _build_axioms(
    domain: Domain,
    instances: Mapping[tuple[str, tuple[str, ...]], "_Instance"],
    reached: set[Fact],
    index: Mapping[Fact, int],
    var_index: Mapping[Fact, int],
) -> tuple[Axiom, ...]:
    """Un axioma por cada cláusula alcanzable de cada regla instanciada, ordenados por
    estrato, hecho derivado y condición."""
    strata = domain.derived_strata
    axioms: set[Axiom] = set()
    for (name, args), instance in instances.items():
        predicate = domain.derived[int(name.removeprefix(":derived"))].predicate
        head = index[(predicate, args)]
        for clause in instance.reachable_clauses(reached):
            condition = _condition(clause, index, var_index)
            if head in condition.pos_facts:
                continue  # se apoya en sí mismo: no deduce nada nuevo
            axioms.add(Axiom(head, condition, strata[predicate]))
    return tuple(
        sorted(
            axioms,
            key=lambda a: (
                a.stratum,
                a.head,
                a.condition.pos_facts,
                a.condition.neg_facts,
                tuple(map(str, a.condition.numeric)),
            ),
        )
    )


def _normalize(normalizer: Normalizer, schema: "_Schema", binding: Binding) -> "_Instance":
    clauses = normalizer.dnf(schema.action.precondition, binding)
    if not clauses:
        return _Instance([], [])
    try:
        effects = normalizer.effects(schema.action.effect, binding)
    except UndefinedEffect:
        return _Instance([], [])  # usa una función estática sin valor: nunca aplicable
    return _Instance(clauses, effects)


@dataclass
class _Instance:
    """Un candidato normalizado: precondición en DNF y efectos simples."""

    clauses: list[Clause]
    effects: list[SimpleEffect]

    def reachable_clauses(self, reached: set[Fact]) -> list[Clause]:
        return [c for c in _sorted_clauses(self.clauses) if c.pos <= reached]

    def newly_reachable(self, reached: set[Fact]) -> Iterator[Fact]:
        """Hechos que añade este candidato y todavía no son alcanzables.

        Basta con que una cláusula de la precondición sea alcanzable: los efectos no
        dependen de cuál. Un efecto condicional cuenta si alguna cláusula de su condición
        es alcanzable.
        """
        if not any(clause.pos <= reached for clause in self.clauses):
            return
        for effect in self.effects:
            new = effect.adds - reached
            if new and any(c.pos <= reached for c in effect.condition):
                yield from new

    def numeric_keys(self) -> set[Fact]:
        """Términos de función que usan sus condiciones y efectos numéricos."""
        keys: set[Fact] = set()
        for clause in self.clauses:
            for condition in clause.numeric:
                keys |= variables_of(condition.expression)
        for effect in self.effects:
            for clause in effect.condition:
                for condition in clause.numeric:
                    keys |= variables_of(condition.expression)
            for assignment in effect.numeric:
                keys.add(assignment.variable)
                keys |= variables_of(assignment.value)
        return keys


def _build_action(
    index_in_task: int,
    name: str,
    args: tuple[str, ...],
    clause: Clause,
    effects: list[SimpleEffect],
    index: Mapping[Fact, int],
    var_index: Mapping[Fact, int],
    cost: float,
) -> GroundAction | None:
    pre = {index[f] for f in clause.pos if f in index}
    neg = {index[f] for f in clause.neg if f in index}
    numeric_pre = {c.remap(var_index.__getitem__) for c in clause.numeric}
    adds: set[int] = set()
    deletes: set[int] = set()
    numeric: list[NumericAssignment] = []
    conditional: list[ConditionalEffect] = []
    for effect in effects:
        effect_adds = {index[f] for f in effect.adds if f in index}
        effect_deletes = {index[f] for f in effect.deletes if f in index}
        # Solo cuentan los efectos numéricos sobre variables del estado; los demás (p. ej.
        # sobre funciones que solo usa la métrica) ya se han tenido en cuenta en el coste.
        effect_numeric = tuple(
            a.remap(var_index.__getitem__) for a in effect.numeric if a.variable in var_index
        )
        if not effect_adds and not effect_deletes and not effect_numeric:
            continue
        for condition in effect.condition:
            if any(f not in index for f in condition.pos):
                continue  # la condición exige un hecho inalcanzable: nunca se cumple
            cond_pos = {index[f] for f in condition.pos}
            cond_neg = {index[f] for f in condition.neg if f in index}
            if cond_pos & neg or cond_neg & pre:
                continue  # contradice la precondición: nunca se cumple
            cond_pos -= pre
            cond_neg -= neg
            cond_numeric = {c.remap(var_index.__getitem__) for c in condition.numeric}
            cond_numeric -= numeric_pre
            if not cond_pos and not cond_neg and not cond_numeric:
                adds |= effect_adds
                deletes |= effect_deletes
                numeric.extend(effect_numeric)
            else:
                conditional.append(
                    ConditionalEffect(
                        Condition(
                            tuple(sorted(cond_pos)),
                            tuple(sorted(cond_neg)),
                            _sorted_numeric(cond_numeric),
                        ),
                        tuple(sorted(effect_adds)),
                        tuple(sorted(effect_deletes)),
                        effect_numeric,
                    )
                )
    if not adds and not deletes and not numeric and not conditional:
        return None  # no cambia nada del estado: no sirve en ningún plan
    return GroundAction(
        index=index_in_task,
        name=name,
        args=args,
        pre_facts=tuple(sorted(pre)),
        add_facts=tuple(sorted(adds)),
        del_facts=tuple(sorted(deletes - adds)),
        cost=cost,
        neg_facts=tuple(sorted(neg)),
        conditional=tuple(dict.fromkeys(conditional)),
        numeric_pre=_sorted_numeric(numeric_pre),
        numeric=tuple(numeric),
    )


def _condition(
    clause: Clause, index: Mapping[Fact, int], var_index: Mapping[Fact, int]
) -> Condition:
    return Condition(
        tuple(sorted(index[f] for f in clause.pos)),
        tuple(sorted(index[f] for f in clause.neg if f in index)),
        _sorted_numeric({c.remap(var_index.__getitem__) for c in clause.numeric}),
    )


def _sorted_numeric(conditions: set[NumericCondition]) -> tuple[NumericCondition, ...]:
    return tuple(sorted(conditions, key=str))


def _sorted_clauses(clauses: list[Clause]) -> list[Clause]:
    return sorted(
        clauses, key=lambda c: (sorted(c.pos), sorted(c.neg), sorted(map(str, c.numeric)))
    )


class _Schema:
    """Un esquema de acción preparado para buscar candidatos. Las preferencias de su
    precondición se separan (``preferences``): no cuentan para que sea aplicable."""

    def __init__(self, action: Action) -> None:
        precondition, self.preferences = split_preferences(action.precondition)
        if self.preferences:
            action = replace(action, precondition=precondition)
        self.action = action
        self.name = action.name
        self.parameters = [p.name for p in action.parameters]
        self.parameter_types = {p.name: p.types for p in action.parameters}
        self._necessary = [
            atom
            for atom in _necessary_atoms(action.precondition)
            if all(isinstance(t, Constant) or t.name in self.parameter_types for t in atom.args)
        ]

    def bindings(
        self, reachable: Mapping[str, set[tuple[str, ...]]], universe: "_TypedUniverse"
    ) -> Iterator[Binding]:
        """Asignaciones de los parámetros que hacen alcanzables las precondiciones
        necesarias y respetan los tipos de los parámetros."""
        plan = self._join_plan(reachable)
        indexes: list[dict[tuple[str, ...], list[tuple[str, ...]]] | None] = [None] * len(plan)
        for binding in self._join(plan, indexes, 0, {}, reachable):
            if not all(universe.has_type(v, self.parameter_types[p]) for p, v in binding.items()):
                continue
            free = [p for p in self.parameters if p not in binding]
            if not free:
                yield binding
                continue
            domains = [universe.objects_of(self.parameter_types[p]) for p in free]
            for values in product(*domains):
                yield {**binding, **dict(zip(free, values, strict=True))}

    def _join_plan(
        self, reachable: Mapping[str, set[tuple[str, ...]]]
    ) -> list[tuple[Atom, tuple[int, ...]]]:
        """Orden del join y posiciones ya ligadas de cada átomo al llegar a él.

        Cada vez se elige el átomo con más posiciones ligadas (constantes o variables de
        los anteriores) y, a igualdad, el de menos hechos alcanzables: así los primeros
        niveles generan pocas asignaciones parciales.
        """
        remaining = list(self._necessary)
        bound: set[str] = set()
        plan: list[tuple[Atom, tuple[int, ...]]] = []

        def positions(atom: Atom) -> tuple[int, ...]:
            return tuple(
                k for k, t in enumerate(atom.args) if isinstance(t, Constant) or t.name in bound
            )

        while remaining:
            atom = min(
                remaining,
                key=lambda a: (-len(positions(a)), len(reachable.get(a.predicate, ()))),
            )
            remaining.remove(atom)
            plan.append((atom, positions(atom)))
            bound.update(t.name for t in atom.args if isinstance(t, Variable))
        return plan

    def _join(
        self,
        plan: list[tuple[Atom, tuple[int, ...]]],
        indexes: list[dict[tuple[str, ...], list[tuple[str, ...]]] | None],
        i: int,
        binding: Binding,
        reachable: Mapping[str, set[tuple[str, ...]]],
    ) -> Iterator[Binding]:
        if i == len(plan):
            yield binding
            return
        atom, positions = plan[i]
        candidates: Iterable[tuple[str, ...]]
        if positions:
            # Índice de los hechos del predicado por los valores de las posiciones ligadas
            # (se construye la primera vez que se necesita).
            index = indexes[i]
            if index is None:
                index = indexes[i] = {}
                for args in reachable.get(atom.predicate, ()):
                    index.setdefault(tuple(args[k] for k in positions), []).append(args)
            key = tuple(_value(atom.args[k], binding) for k in positions)
            candidates = index.get(key, ())
        else:
            candidates = reachable.get(atom.predicate, ())
        for args in candidates:
            extended = _unify(atom, args, binding)
            if extended is not None:
                yield from self._join(plan, indexes, i + 1, extended, reachable)


class _TypedUniverse:
    """Objetos del problema (y constantes del dominio) agrupados por tipo."""

    def __init__(self, domain: Domain, declarations: Mapping[str, tuple[str, ...]]) -> None:
        self.domain = domain
        self.declarations = declarations
        self._cache: dict[tuple[str, ...], list[str]] = {}

    def has_type(self, name: str, types: tuple[str, ...]) -> bool:
        return self.domain.is_subtype(self.declarations.get(name, (OBJECT,)), types)

    def objects_of(self, types: tuple[str, ...]) -> list[str]:
        if types not in self._cache:
            self._cache[types] = sorted(n for n in self.declarations if self.has_type(n, types))
        return self._cache[types]


def _necessary_atoms(formula: Formula) -> Iterator[Atom]:
    """Átomos positivos que tiene que cumplir cualquier estado que satisfaga ``formula``.

    Solo se miran la conjunción exterior y los átomos sueltos; es una aproximación segura.
    """
    if isinstance(formula, Atom):
        yield formula
    elif isinstance(formula, And):
        for part in formula.args:
            yield from _necessary_atoms(part)  # type: ignore[arg-type]


def _modified(effect: Effect) -> Iterator[str]:
    """Predicados que añade o borra un efecto (incluidos los condicionales)."""
    if isinstance(effect, Atom):
        yield effect.predicate
    elif isinstance(effect, Not):
        assert isinstance(effect.arg, Atom)
        yield effect.arg.predicate
    elif isinstance(effect, And):
        for part in effect.args:
            yield from _modified(part)  # type: ignore[arg-type]
    elif isinstance(effect, ForallEffect | When):
        yield from _modified(effect.arg)


def _value(term: Constant | Variable, binding: Binding) -> str:
    return term.name if isinstance(term, Constant) else binding[term.name]


def _unify(atom: Atom, args: tuple[str, ...], binding: Binding) -> Binding | None:
    """Extiende ``binding`` para que ``atom`` coincida con ``args``, o devuelve ``None``."""
    extended = binding
    for term, value in zip(atom.args, args, strict=True):
        if isinstance(term, Constant):
            if term.name != value:
                return None
        else:
            bound = extended.get(term.name)
            if bound is None:
                if extended is binding:
                    extended = dict(binding)
                extended[term.name] = value
            elif bound != value:
                return None
    return extended


def _args(atom: Atom | FluentTerm) -> tuple[str, ...]:
    return tuple(t.name for t in atom.args)


# --- Funciones numéricas ---------------------------------------------------------------


def _numeric_effects(effect: Effect) -> Iterator[NumericEffect]:
    """Efectos numéricos de un efecto (incluidos los condicionales y universales)."""
    if isinstance(effect, NumericEffect):
        yield effect
    elif isinstance(effect, And):
        for part in effect.args:
            yield from _numeric_effects(part)  # type: ignore[arg-type]
    elif isinstance(effect, ForallEffect | When):
        yield from _numeric_effects(effect.arg)


def _functions_in(node: object) -> Iterator[str]:
    """Funciones que aparecen en una fórmula, una expresión o un efecto."""
    if isinstance(node, FluentTerm):
        yield node.function
    elif isinstance(node, Comparison):
        yield from _functions_in(node.left)
        yield from _functions_in(node.right)
    elif isinstance(node, Arithmetic | And | Or):
        for arg in node.args:
            yield from _functions_in(arg)
    elif isinstance(node, Not):
        yield from _functions_in(node.arg)
    elif isinstance(node, Imply):
        yield from _functions_in(node.left)
        yield from _functions_in(node.right)
    elif isinstance(node, Exists | Forall | ForallEffect | Preference | ConstraintForall):
        yield from _functions_in(node.arg)
    elif isinstance(node, When):
        yield from _functions_in(node.condition)
    elif isinstance(node, ConstraintAnd):
        for part in node.args:
            yield from _functions_in(part)
    elif isinstance(node, TrajectoryConstraint):
        for formula in node.formulas:
            yield from _functions_in(formula)


def _conditions_of(effect: Effect) -> Iterator[Formula]:
    """Condiciones de los efectos condicionales de un efecto."""
    if isinstance(effect, When):
        yield effect.condition
    elif isinstance(effect, And):
        for part in effect.args:
            yield from _conditions_of(part)  # type: ignore[arg-type]
    elif isinstance(effect, ForallEffect):
        yield from _conditions_of(effect.arg)


def _relevant_functions(
    domain: Domain,
    problem: Problem,
    targeted: frozenset[str],
    extra_reads: Mapping[str, tuple[Expression, ...]],
    constraints: Sequence[Constraint] = (),
) -> frozenset[str]:
    """Funciones que modifica alguna acción y que importan para decidir qué es aplicable
    u objetivo.

    Una función es relevante si la lee una condición (precondición, condición de un efecto
    condicional, objetivo, preferencia o restricción de trayectoria), o si la lee el valor
    de un efecto sobre una función relevante.
    Las que solo aparecen en efectos y en la métrica (``total-cost``, ``total-fuel-used``…)
    no forman parte del estado: si lo hicieran, dos caminos al mismo estado con distinto
    coste serían estados distintos y no se detectarían los repetidos.
    """
    read: set[str] = set(_functions_in(problem.goal))
    for constraint in constraints:
        read |= set(_functions_in(constraint))
    for rule in domain.derived:
        read |= set(_functions_in(rule.condition))
    for expressions in extra_reads.values():
        for expression in expressions:
            read |= set(_functions_in(expression))
    for action in domain.actions:
        read |= set(_functions_in(action.precondition))
        for condition in _conditions_of(action.effect):
            read |= set(_functions_in(condition))
    relevant = read & targeted
    changed = True
    while changed:
        changed = False
        for action in domain.actions:
            for effect in _numeric_effects(action.effect):
                if effect.fluent.function in relevant:
                    new = set(_functions_in(effect.value)) & targeted - relevant
                    if new:
                        relevant |= new
                        changed = True
    return frozenset(relevant)


#: Variable especial que representa ``total-time`` en la forma lineal de la métrica.
_TOTAL_TIME = ("total-time", ())
#: Las de ``(is-violated nombre)`` son ``(_VIOLATED, (nombre,))`` (no hay funciones con ':').
_VIOLATED = ":is-violated"


class _MetricModel:
    """La métrica como coste de acción.

    Si la métrica minimiza una expresión lineal de ``total-time`` y de funciones que las
    acciones solo cambian con ``increase``/``decrease`` de valor constante, cada acción
    contribuye con una cantidad fija: esa es su coste, y minimizar el coste del plan
    minimiza la métrica. Es el caso de ``:action-costs`` (``total-cost``) y de métricas como
    la de zenotravel. Si no, el grounder usa coste unitario y deja una nota.

    Los términos ``(is-violated p)`` dan el peso de cada preferencia (ver
    :meth:`preference_weights`).
    """

    def __init__(
        self, problem: Problem, normalizer: Normalizer, values: Mapping[Fact, float]
    ) -> None:
        self.active = False
        self.offset: float | None = None
        self.notes: list[str] = []
        self.weights: dict[Fact, float] = {}
        self.time_weight = 0.0
        self._preferences: dict[str, float] = {}
        self._metric = metric = problem.metric
        if metric is None:
            return
        if metric.direction != "minimize":
            self.notes.append("la métrica se maximiza: se busca con coste unitario")
            return
        expression = self._expression(metric.expression, normalizer)
        form = linear(expression) if expression is not None else None
        if form is None:
            self.notes.append("la métrica no es lineal: se busca con coste unitario")
            return
        self.weights = dict(form.terms)
        self.time_weight = self.weights.pop(_TOTAL_TIME, 0.0)
        self._preferences = {
            key[1][0]: self.weights.pop(key) for key in list(self.weights) if key[0] == _VIOLATED
        }
        self.active = True
        offset: float | None = form.constant
        for key, weight in self.weights.items():
            value = values.get(key)
            offset = None if offset is None or value is None else offset + weight * value
        self.offset = offset

    def preference_weights(self, preferences: bool) -> dict[str, float]:
        """Peso de cada preferencia que cuenta en la métrica (las demás no importan).

        Args:
            preferences: si la tarea tiene preferencias.

        Raises:
            UnsupportedFeatureError: si la métrica no es una expresión lineal que se
                minimiza, o si alguna preferencia tiene peso negativo.
        """
        if not preferences:
            return {}
        if self._metric is None:
            self.notes.append("el problema no tiene métrica: las preferencias no cuentan")
            return {}
        if not self.active:
            raise UnsupportedFeatureError(
                "con preferencias, la métrica tiene que ser una expresión lineal que se minimiza"
            )
        for name, weight in self._preferences.items():
            if weight < 0:
                raise UnsupportedFeatureError(
                    f"la preferencia '{name}' tiene peso negativo en la métrica: incumplirla "
                    "la mejora"
                )
        return dict(self._preferences)

    def _expression(self, expression: Expression, normalizer: Normalizer) -> NumExpr | None:
        if isinstance(expression, TotalTime):
            return Var(_TOTAL_TIME)
        if isinstance(expression, IsViolated):
            return Var((_VIOLATED, (expression.name,)))
        if isinstance(expression, Arithmetic):
            args = [self._expression(arg, normalizer) for arg in expression.args]
            if any(arg is None for arg in args):
                return None
            return make(expression.op, args)  # type: ignore[arg-type]
        return normalizer.expression(expression, {})

    def cost(self, effects: list[SimpleEffect]) -> float | None:
        """Contribución de una acción a la métrica, o ``None`` si no es constante."""
        if not self.active:
            return 1.0
        total = self.time_weight
        for effect in effects:
            for assignment in effect.numeric:
                weight = self.weights.get(assignment.variable)
                if weight is None:
                    continue
                constant = isinstance(assignment.value, Const)
                if (
                    effect.condition != tuple(TRUE)
                    or not constant
                    or assignment.op
                    not in (
                        "increase",
                        "decrease",
                    )
                ):
                    return None
                amount = assignment.value.value  # type: ignore[union-attr]
                total += weight * (amount if assignment.op == "increase" else -amount)
        return total
