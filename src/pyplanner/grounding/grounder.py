"""Instanciación (grounding) de una tarea STRIPS con análisis de alcanzabilidad relajada.

En lugar de probar todas las combinaciones de objetos para cada acción, el grounder parte
de los hechos del estado inicial y va instanciando solo las acciones cuyas precondiciones
son **alcanzables** en la relajación sin borrados. Los efectos de esas acciones hacen
alcanzables hechos nuevos, y el proceso se repite hasta llegar a un punto fijo. Las
acciones y los hechos que quedan fuera no pueden aparecer en ningún plan.

Después:

* se eliminan los hechos **estáticos** (los que ninguna acción modifica): son ciertos en
  todos los estados alcanzables, así que no hace falta guardarlos ni comprobarlos;
* los hechos del objetivo que no son alcanzables se conservan, para que las heurísticas
  detecten que la tarea no tiene solución.
"""

from collections.abc import Iterator, Mapping
from itertools import product

from pyplanner.grounding.task import GroundAction, GroundTask, mask_of
from pyplanner.loader.ast import (
    Action,
    And,
    Atom,
    Constant,
    Domain,
    Effect,
    Formula,
    Not,
    Problem,
    Variable,
)

Fact = tuple[str, tuple[str, ...]]
Binding = dict[str, str]


def ground(domain: Domain, problem: Problem) -> GroundTask:
    """Instancia un dominio y un problema STRIPS.

    Returns:
        La tarea instanciada, con hechos y acciones en un orden determinista.
    """
    universe = sorted({c.name for c in domain.constants} | {o.name for o in problem.objects})
    schemas = [_Schema(action) for action in domain.actions]
    fluent_predicates = {
        atom.predicate for schema in schemas for atom in (*schema.adds, *schema.deletes)
    }

    reachable: dict[str, set[tuple[str, ...]]] = {}
    for atom in problem.init:
        reachable.setdefault(atom.predicate, set()).add(_args(atom))

    ground_actions: dict[tuple[str, tuple[str, ...]], Binding] = {}
    changed = True
    while changed:
        changed = False
        for schema in schemas:
            # Se materializan antes de añadir hechos: el join recorre ``reachable``.
            for binding in list(schema.bindings(reachable, universe)):
                key = (schema.name, tuple(binding[p] for p in schema.parameters))
                if key in ground_actions:
                    continue
                ground_actions[key] = binding
                for atom in schema.adds:
                    args = _instantiate(atom, binding)
                    known = reachable.setdefault(atom.predicate, set())
                    if args not in known:
                        known.add(args)
                        changed = True

    # Hechos del estado: los alcanzables de predicados que alguna acción modifica, más los
    # objetivos inalcanzables (para que la tarea se reconozca como irresoluble).
    static_true = {
        (pred, args)
        for pred, all_args in reachable.items()
        for args in all_args
        if pred not in fluent_predicates
    }
    fluent_facts: set[Fact] = {
        (pred, args)
        for pred, all_args in reachable.items()
        if pred in fluent_predicates
        for args in all_args
    }
    goal_facts = [_fact(atom) for atom in _conjuncts(problem.goal)]
    fluent_facts |= {f for f in goal_facts if f not in static_true}
    facts = sorted(fluent_facts)
    index = {fact: i for i, fact in enumerate(facts)}

    actions: list[GroundAction] = []
    schema_by_name = {schema.name: schema for schema in schemas}
    for (name, args), binding in sorted(ground_actions.items()):
        schema = schema_by_name[name]
        pre = sorted({index[f] for f in schema.ground(schema.preconditions, binding) if f in index})
        add = sorted({index[f] for f in schema.ground(schema.adds, binding)})
        delete = sorted(
            {index[f] for f in schema.ground(schema.deletes, binding) if f in index} - set(add)
        )
        actions.append(
            GroundAction(len(actions), name, args, tuple(pre), tuple(add), tuple(delete))
        )

    init = mask_of(index[f] for f in map(_fact, problem.init) if f in index)
    return GroundTask(
        domain_name=domain.name,
        problem_name=problem.name,
        facts=tuple(Atom(pred, tuple(Constant(a) for a in args)) for pred, args in facts),
        actions=tuple(actions),
        init=init,
        goal_facts=tuple(sorted({index[f] for f in goal_facts if f in index})),
    )


class _Schema:
    """Un esquema de acción preparado para instanciarlo."""

    def __init__(self, action: Action) -> None:
        self.name = action.name
        self.parameters = [p.name for p in action.parameters]
        self.preconditions = list(_conjuncts(action.precondition))
        self.adds: list[Atom] = []
        self.deletes: list[Atom] = []
        for effect in _effects(action.effect):
            if isinstance(effect, Not):
                self.deletes.append(effect.arg)
            else:
                self.adds.append(effect)
        # Primero las precondiciones con más variables: ligan más parámetros de golpe.
        self._join_order = sorted(
            self.preconditions, key=lambda a: -sum(isinstance(t, Variable) for t in a.args)
        )

    def bindings(
        self, reachable: Mapping[str, set[tuple[str, ...]]], universe: list[str]
    ) -> Iterator[Binding]:
        """Asignaciones de los parámetros que hacen alcanzables todas las precondiciones."""
        for binding in self._join(0, {}, reachable):
            free = [p for p in self.parameters if p not in binding]
            if not free:
                yield binding
                continue
            for values in product(universe, repeat=len(free)):
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

    @staticmethod
    def ground(atoms: list[Atom], binding: Binding) -> Iterator[Fact]:
        for atom in atoms:
            yield (atom.predicate, _instantiate(atom, binding))


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


def _instantiate(atom: Atom, binding: Binding) -> tuple[str, ...]:
    return tuple(t.name if isinstance(t, Constant) else binding[t.name] for t in atom.args)


def _args(atom: Atom) -> tuple[str, ...]:
    return tuple(t.name for t in atom.args)


def _fact(atom: Atom) -> Fact:
    return (atom.predicate, _args(atom))


def _conjuncts(formula: Formula) -> Iterator[Atom]:
    if isinstance(formula, And):
        for part in formula.args:
            yield from _conjuncts(part)  # type: ignore[arg-type]
    else:
        yield formula


def _effects(effect: Effect) -> Iterator[Atom | Not]:
    if isinstance(effect, And):
        for part in effect.args:
            yield from _effects(part)
    else:
        yield effect
