"""Instanciación (grounding) con análisis de alcanzabilidad relajada y normalización ADL.

En lugar de probar todas las combinaciones de objetos para cada acción, el grounder parte
de los hechos del estado inicial y va instanciando solo las acciones cuyas precondiciones
son **alcanzables** en la relajación sin borrados. Los efectos de esas acciones hacen
alcanzables hechos nuevos, y el proceso se repite hasta llegar a un punto fijo.

1. **Candidatos:** para cada esquema se buscan las asignaciones de parámetros que hacen
   alcanzables sus *precondiciones necesarias* (los átomos positivos de la conjunción
   exterior) y respetan los tipos. Los parámetros que no aparecen en ellas toman todos los
   objetos de su tipo.
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
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import product

from pyplanner.grounding.normalize import Clause, Fact, Normalizer, SimpleEffect
from pyplanner.grounding.task import (
    Condition,
    ConditionalEffect,
    GroundAction,
    GroundTask,
    mask_of,
)
from pyplanner.loader.ast import (
    OBJECT,
    Action,
    And,
    Atom,
    Constant,
    Domain,
    Effect,
    ForallEffect,
    Formula,
    Not,
    Problem,
    Variable,
    When,
)

Binding = dict[str, str]


def ground(domain: Domain, problem: Problem) -> GroundTask:
    """Instancia y normaliza un dominio y un problema.

    Returns:
        La tarea instanciada, con hechos y acciones en un orden determinista.

    Raises:
        UnsupportedFeatureError: si una fórmula es demasiado grande en forma normal
            disyuntiva.
    """
    declarations = {o.name: o.types for o in (*domain.constants, *problem.objects)}
    universe = _TypedUniverse(domain, declarations)
    fluent_predicates = frozenset(
        predicate for action in domain.actions for predicate in _modified(action.effect)
    )
    static_predicates = frozenset(p.name for p in domain.predicates) - fluent_predicates
    init_facts = {(a.predicate, _args(a)) for a in problem.init}
    normalizer = Normalizer(
        universe.objects_of,
        static_predicates,
        frozenset(f for f in init_facts if f[0] in static_predicates),
    )
    schemas = [_Schema(action) for action in domain.actions]

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
                    instance = _Instance(
                        normalizer.dnf(schema.action.precondition, binding),
                        normalizer.effects(schema.action.effect, binding),
                    )
                    instances[key] = instance
                for fact in list(instance.newly_reachable(reached)):
                    reached.add(fact)
                    reachable.setdefault(fact[0], set()).add(fact[1])
                    changed = True

    facts = sorted(f for f in reached if f[0] in fluent_predicates)
    index = {fact: i for i, fact in enumerate(facts)}
    actions: list[GroundAction] = []
    for (name, args), instance in sorted(instances.items()):
        for clause in instance.reachable_clauses(reached):
            action = _build_action(len(actions), name, args, clause, instance.effects, index)
            if action is not None:
                actions.append(action)

    goal = [
        _condition(clause, index)
        for clause in _sorted_clauses(normalizer.dnf(problem.goal, {}))
        if clause.pos <= reached
    ]
    return GroundTask(
        domain_name=domain.name,
        problem_name=problem.name,
        facts=tuple(Atom(pred, tuple(Constant(a) for a in args)) for pred, args in facts),
        actions=tuple(actions),
        init=mask_of([index[f] for f in init_facts if f in index]),
        goal=tuple(dict.fromkeys(goal)),
    )


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


def _build_action(
    index_in_task: int,
    name: str,
    args: tuple[str, ...],
    clause: Clause,
    effects: list[SimpleEffect],
    index: Mapping[Fact, int],
) -> GroundAction | None:
    pre = {index[f] for f in clause.pos if f in index}
    neg = {index[f] for f in clause.neg if f in index}
    adds: set[int] = set()
    deletes: set[int] = set()
    conditional: list[ConditionalEffect] = []
    for effect in effects:
        effect_adds = {index[f] for f in effect.adds if f in index}
        effect_deletes = {index[f] for f in effect.deletes if f in index}
        if not effect_adds and not effect_deletes:
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
            if not cond_pos and not cond_neg:
                adds |= effect_adds
                deletes |= effect_deletes
            else:
                conditional.append(
                    ConditionalEffect(
                        Condition(tuple(sorted(cond_pos)), tuple(sorted(cond_neg))),
                        tuple(sorted(effect_adds)),
                        tuple(sorted(effect_deletes)),
                    )
                )
    if not adds and not deletes and not conditional:
        return None  # no cambia nada: no sirve en ningún plan
    return GroundAction(
        index=index_in_task,
        name=name,
        args=args,
        pre_facts=tuple(sorted(pre)),
        add_facts=tuple(sorted(adds)),
        del_facts=tuple(sorted(deletes - adds)),
        neg_facts=tuple(sorted(neg)),
        conditional=tuple(dict.fromkeys(conditional)),
    )


def _condition(clause: Clause, index: Mapping[Fact, int]) -> Condition:
    return Condition(
        tuple(sorted(index[f] for f in clause.pos)),
        tuple(sorted(index[f] for f in clause.neg if f in index)),
    )


def _sorted_clauses(clauses: list[Clause]) -> list[Clause]:
    return sorted(clauses, key=lambda c: (sorted(c.pos), sorted(c.neg)))


class _Schema:
    """Un esquema de acción preparado para buscar candidatos."""

    def __init__(self, action: Action) -> None:
        self.action = action
        self.name = action.name
        self.parameters = [p.name for p in action.parameters]
        self.parameter_types = {p.name: p.types for p in action.parameters}
        necessary = [
            atom
            for atom in _necessary_atoms(action.precondition)
            if all(isinstance(t, Constant) or t.name in self.parameter_types for t in atom.args)
        ]
        # Primero los átomos con más variables: ligan más parámetros de golpe.
        self._join_order = sorted(
            necessary, key=lambda a: -sum(isinstance(t, Variable) for t in a.args)
        )

    def bindings(
        self, reachable: Mapping[str, set[tuple[str, ...]]], universe: "_TypedUniverse"
    ) -> Iterator[Binding]:
        """Asignaciones de los parámetros que hacen alcanzables las precondiciones
        necesarias y respetan los tipos de los parámetros."""
        for binding in self._join(0, {}, reachable):
            if not all(universe.has_type(v, self.parameter_types[p]) for p, v in binding.items()):
                continue
            free = [p for p in self.parameters if p not in binding]
            if not free:
                yield binding
                continue
            domains = [universe.objects_of(self.parameter_types[p]) for p in free]
            for values in product(*domains):
                yield {**binding, **dict(zip(free, values, strict=True))}

    def _join(
        self, i: int, binding: Binding, reachable: Mapping[str, set[tuple[str, ...]]]
    ) -> Iterator[Binding]:
        if i == len(self._join_order):
            yield binding
            return
        atom = self._join_order[i]
        for args in reachable.get(atom.predicate, ()):
            extended = _unify(atom, args, binding)
            if extended is not None:
                yield from self._join(i + 1, extended, reachable)


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


def _args(atom: Atom) -> tuple[str, ...]:
    return tuple(t.name for t in atom.args)
