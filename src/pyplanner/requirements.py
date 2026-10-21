"""Requisitos PDDL (``:requirements``): los que existen, los que implican otros y los soportados.

Este módulo es la única fuente de verdad sobre qué requisitos acepta el planificador. Las dos
fases del parser lo consultan: la fase 1 para rechazar pronto un fichero PDDL que no se
puede procesar y la fase 2 porque su JSON de entrada puede venir de cualquier otra fuente.

Los que soporta el planificador están en :data:`SUPPORTED_REQUIREMENTS`.

**Modo estricto.** Toda construcción que se use tiene que tener su requisito declarado
(:func:`requirement_uses` y :func:`first_undeclared`). Las reglas siguen la gramática de
PDDL 2.1:

* sección ``:types`` o listas con tipos → ``:typing``;
* ``(not <átomo>)`` en una fórmula → ``:negative-preconditions``;
* ``(not <fórmula compuesta>)``, ``or`` e ``imply`` → ``:disjunctive-preconditions``;
* ``exists`` → ``:existential-preconditions``; ``forall`` en una fórmula →
  ``:universal-preconditions``;
* ``=`` → ``:equality`` (y ``(not (= ...))`` además ``:negative-preconditions``);
* ``forall`` y ``when`` en un efecto → ``:conditional-effects``;
* funciones, expresiones, comparaciones, efectos numéricos, valores iniciales y métrica →
  ``:numeric-fluents``. Lo que encaja en el perfil de ``:action-costs`` (PDDL 3.1) también
  se acepta con ese requisito: declarar funciones y darles valor inicial, ``(increase
  (total-cost) <número o función>)`` y ``(:metric minimize (total-cost))``;
* acciones durativas → ``:durative-actions``;
* restricciones de duración con ``<=`` o ``>=``, varias restricciones o una evaluada en
  ``(at start|end ...)`` → ``:duration-inequalities``;
* efectos continuos (``increase``/``decrease`` con ``#t``, fuera de ``at start/end``) →
  ``:continuous-effects``;
* reglas ``:derived`` → ``:derived-predicates`` (su condición, además, los de cualquier
  fórmula);
* literales ``(at <t> <literal>)`` en ``:init`` → ``:timed-initial-literals``;
* funciones de objetos (con tipo de objeto), términos de función como argumento,
  ``assign`` de un objeto o ``undefined`` y valores iniciales de objeto →
  ``:object-fluents``;
* restricciones de trayectoria (``:constraints`` y sus operadores modales) →
  ``:constraints``, y ``forall`` en ellas → ``:universal-preconditions``;
* ``preference`` (en objetivos, precondiciones, condiciones y restricciones) e
  ``is-violated`` → ``:preferences``. Una métrica que solo usa números, aritmética e
  ``is-violated`` también se acepta con ``:preferences`` (es la de las preferencias de la
  IPC 2006);
* procesos y eventos (PDDL+) → ``:time``. Los efectos continuos de un proceso no necesitan
  además ``:continuous-effects`` (ese es el de las acciones durativas), pero sí
  ``:numeric-fluents``, como cualquier efecto numérico.

Un literal negado en ``:init`` (PDDL 3.1) no necesita nada: es redundante.

``(not <átomo>)`` en un efecto es un borrado STRIPS y no necesita nada.
"""

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Any

#: Todos los requisitos que existen en las versiones de PDDL, soportados o no.
#: Un requisito que no está aquí es un error; uno que está pero no se soporta, también,
#: pero con un mensaje distinto.
KNOWN_REQUIREMENTS: frozenset[str] = frozenset(
    {
        # PDDL 1.2
        ":strips",
        ":typing",
        ":negative-preconditions",
        ":disjunctive-preconditions",
        ":equality",
        ":existential-preconditions",
        ":universal-preconditions",
        ":quantified-preconditions",
        ":conditional-effects",
        ":action-expansions",
        ":foreach-expansions",
        ":dag-expansions",
        ":domain-axioms",
        ":subgoals-through-axioms",
        ":safety-constraints",
        ":expression-evaluation",
        ":fluents",
        ":open-world",
        ":true-negation",
        ":adl",
        ":ucpop",
        # PDDL 2.1
        ":numeric-fluents",
        ":durative-actions",
        ":duration-inequalities",
        ":continuous-effects",
        # PDDL 2.2
        ":derived-predicates",
        ":timed-initial-literals",
        # PDDL+
        ":time",
        # PDDL 3.x
        ":constraints",
        ":preferences",
        ":action-costs",
        ":object-fluents",
    }
)

#: Requisitos que equivalen a un conjunto de otros requisitos.
REQUIREMENT_EXPANSIONS: dict[str, frozenset[str]] = {
    ":quantified-preconditions": frozenset(
        {":existential-preconditions", ":universal-preconditions"}
    ),
    ":adl": frozenset(
        {
            ":strips",
            ":typing",
            ":negative-preconditions",
            ":disjunctive-preconditions",
            ":equality",
            ":existential-preconditions",
            ":universal-preconditions",
            ":quantified-preconditions",
            ":conditional-effects",
        }
    ),
    ":fluents": frozenset({":numeric-fluents"}),
}

#: Requisitos que soporta el planificador.
SUPPORTED_REQUIREMENTS: frozenset[str] = frozenset(
    {
        # STRIPS
        ":strips",
        # PDDL 1.2 (subconjunto ADL)
        ":typing",
        ":negative-preconditions",
        ":disjunctive-preconditions",
        ":equality",
        ":existential-preconditions",
        ":universal-preconditions",
        ":conditional-effects",
        # PDDL 2.1 numérico y costes de acción de PDDL 3.1
        ":numeric-fluents",
        ":action-costs",
        # PDDL 2.1 temporal
        ":durative-actions",
        ":duration-inequalities",
        ":continuous-effects",
        # PDDL 2.2
        ":derived-predicates",
        ":timed-initial-literals",
        # PDDL 3
        ":constraints",
        ":preferences",
        ":object-fluents",
        # PDDL+
        ":time",
    }
)

#: Requisito que se supone cuando un dominio no declara ``:requirements``.
DEFAULT_REQUIREMENT = ":strips"


def unknown_requirement_message(requirement: str) -> str:
    """Mensaje de error para un requisito que no existe, con el más parecido si lo hay
    (``:durative-inequalities`` → ``:duration-inequalities``)."""
    message = f"requisito desconocido '{requirement}'"
    close = get_close_matches(requirement, sorted(KNOWN_REQUIREMENTS), n=1, cutoff=0.8)
    return f"{message}; ¿quería decir '{close[0]}'?" if close else message


def expand_requirements(requirements: Iterable[str]) -> frozenset[str]:
    """Devuelve los requisitos junto con todos los que implican, de forma transitiva.

    Si no hay ningún requisito se devuelve :data:`DEFAULT_REQUIREMENT`.
    """
    pending = list(requirements) or [DEFAULT_REQUIREMENT]
    result: set[str] = set()
    while pending:
        requirement = pending.pop()
        if requirement not in result:
            result.add(requirement)
            pending.extend(REQUIREMENT_EXPANSIONS.get(requirement, ()))
    return frozenset(result)


def unsupported_requirements(requirements: Iterable[str]) -> list[str]:
    """Devuelve, ordenados, los requisitos (ya expandidos) que no están soportados.

    Los metarrequisitos como ``:adl`` se consideran soportados si lo están todos los
    requisitos a los que equivalen.
    """
    expanded = expand_requirements(requirements)
    return sorted(
        r for r in expanded if r not in SUPPORTED_REQUIREMENTS and r not in REQUIREMENT_EXPANSIONS
    )


# --- Modo estricto --------------------------------------------------------------------


@dataclass(frozen=True)
class RequirementUse:
    """Una construcción del documento que necesita un requisito.

    Attributes:
        requirement: requisito necesario (``":typing"``…).
        construct: descripción de la construcción, para los mensajes de error.
        path: ruta dentro del documento JSON.
        loc: posición en el PDDL original (``{"line", "column"}``), si el documento la trae.
        alternatives: otros requisitos que también la permiten (``:action-costs`` para las
            construcciones numéricas de su perfil).
    """

    requirement: str
    construct: str
    path: str
    loc: Mapping[str, int] | None = None
    alternatives: tuple[str, ...] = ()


def requirement_uses(
    document: Mapping[str, Any], objects: frozenset[str] | None = None
) -> Iterator[RequirementUse]:
    """Recorre un documento JSON (dominio o problema) y devuelve cada construcción que
    necesita un requisito, en orden de aparición.

    ``objects`` son las funciones de objetos (``:object-fluents``): en un dominio se sacan
    de sus declaraciones; un problema necesita las de su dominio. Se supone que el
    documento cumple el esquema.
    """
    if objects is None:
        objects = frozenset(f["name"] for f in document.get("functions", []) if "type" in f)
    if document.get("types"):
        yield RequirementUse(":typing", "la sección :types", "types", _loc(document["types"][0]))
    for key in ("constants", "objects"):
        yield from _typed_items(document.get(key, []), key)
    for i, function in enumerate(document.get("functions", [])):
        path = f"functions[{i}]"
        if "type" in function:
            yield RequirementUse(":object-fluents", "una función de objetos", path, _loc(function))
        else:
            yield RequirementUse(
                ":numeric-fluents", "la sección :functions", path, _loc(function), _COSTS
            )
        yield from _typed_items(function.get("parameters", []), f"{path}.parameters")
    for i, predicate in enumerate(document.get("predicates", [])):
        yield from _typed_items(predicate.get("parameters", []), f"predicates[{i}].parameters")
    for i, action in enumerate(document.get("actions", [])):
        path = f"actions[{i}]"
        yield from _typed_items(action.get("parameters", []), f"{path}.parameters")
        if "precondition" in action:
            yield from _formula_uses(action["precondition"], f"{path}.precondition", objects)
        if "effect" in action:
            yield from _effect_uses(action["effect"], f"{path}.effect", objects)
    for i, action in enumerate(document.get("durative_actions", [])):
        path = f"durative_actions[{i}]"
        yield RequirementUse(":durative-actions", "una acción durativa", path, _loc(action))
        yield from _typed_items(action.get("parameters", []), f"{path}.parameters")
        yield from _duration_uses(action["duration"], f"{path}.duration")
        if "condition" in action:
            yield from _timed_condition_uses(action["condition"], f"{path}.condition", objects)
        if "effect" in action:
            yield from _timed_effect_uses(action["effect"], f"{path}.effect", objects)
    for i, rule in enumerate(document.get("derived", [])):
        path = f"derived[{i}]"
        yield RequirementUse(":derived-predicates", "un predicado derivado", path, _loc(rule))
        yield from _typed_items(rule.get("parameters", []), f"{path}.parameters")
        yield from _formula_uses(rule["condition"], f"{path}.condition", objects)
    for key, kind in (("processes", "un proceso"), ("events", "un evento")):
        for i, happening in enumerate(document.get(key, [])):
            path = f"{key}[{i}]"
            yield RequirementUse(":time", kind, path, _loc(happening))
            yield from _typed_items(happening.get("parameters", []), f"{path}.parameters")
            if "precondition" in happening:
                yield from _formula_uses(happening["precondition"], f"{path}.precondition", objects)
            if "effect" in happening:
                yield from _effect_uses(happening["effect"], f"{path}.effect", objects)
    for i, element in enumerate(document.get("init", [])):
        is_object = element["op"] == "=" and (
            isinstance(element["value"], str) or element["fluent"]["function"] in objects
        )
        if is_object:
            yield RequirementUse(
                ":object-fluents", "un valor inicial de objeto", f"init[{i}]", _loc(element)
            )
        elif element["op"] == "=":
            yield RequirementUse(
                ":numeric-fluents", "un valor inicial numérico", f"init[{i}]", _loc(element), _COSTS
            )
        elif element["op"] == "at":
            yield RequirementUse(
                ":timed-initial-literals",
                "un literal inicial temporizado",
                f"init[{i}]",
                _loc(element),
            )
    if "goal" in document:
        yield from _formula_uses(document["goal"], "goal", objects)
    if "constraints" in document:
        yield from _constraint_uses(document["constraints"], "constraints", objects)
    if "metric" in document:
        metric = document["metric"]
        expression = metric["expression"]
        alternatives: tuple[str, ...] = ()
        if metric["direction"] == "minimize" and _is_total_cost(expression):
            alternatives = _COSTS
        elif _only_violations(expression):
            alternatives = (":preferences",)
        yield RequirementUse(":numeric-fluents", "la métrica", "metric", _loc(metric), alternatives)
        yield from _violation_uses(expression, "metric.expression")


def first_undeclared(
    document: Mapping[str, Any],
    declared: Iterable[str],
    objects: frozenset[str] | None = None,
) -> RequirementUse | None:
    """La primera construcción de ``document`` cuyo requisito no está en ``declared``.

    ``declared`` se expande antes de comparar (``:adl`` cubre ``:typing``, etc.), y
    ``objects`` son las funciones de objetos (ver :func:`requirement_uses`).
    """
    expanded = expand_requirements(declared)
    for use in requirement_uses(document, objects):
        if use.requirement not in expanded and not any(a in expanded for a in use.alternatives):
            return use
    return None


#: Las construcciones numéricas de este perfil también se aceptan con ``:action-costs``.
_COSTS = (":action-costs",)


def _is_total_cost(node: Mapping[str, Any]) -> bool:
    return node["op"] == "fluent" and node["function"] == "total-cost" and not node["args"]


def _only_violations(node: Mapping[str, Any]) -> bool:
    """``True`` si la expresión solo tiene números, aritmética e ``is-violated``: una
    métrica de preferencias, que se acepta con ``:preferences`` (como en la IPC 2006)."""
    op = node["op"]
    if op == "is-violated" or op == "num":
        return True
    if op in ("+", "-", "*", "/"):
        return all(_only_violations(arg) for arg in node["args"])
    return False


def _term_uses(
    args: list[Any], path: str, loc: Mapping[str, int] | None
) -> Iterator[RequirementUse]:
    """Un término de función como argumento (``(at (loc ?r))``) necesita
    ``:object-fluents``."""
    if any(isinstance(arg, Mapping) for arg in args):
        yield RequirementUse(":object-fluents", "una función de objetos como argumento", path, loc)


def _object_equality(node: Mapping[str, Any], objects: frozenset[str]) -> bool:
    """``True`` si una comparación ``=`` compara dos funciones de objetos (la fase 1 no
    puede distinguirla de una numérica sin las declaraciones)."""
    return node["comparator"] == "=" and all(
        side["op"] == "fluent" and side["function"] in objects
        for side in (node["left"], node["right"])
    )


def _violation_uses(node: Mapping[str, Any], path: str) -> Iterator[RequirementUse]:
    if node["op"] == "is-violated":
        yield RequirementUse(":preferences", "'is-violated' en la métrica", path, _loc(node))
    for i, arg in enumerate(node.get("args", ())):
        if isinstance(arg, Mapping):
            yield from _violation_uses(arg, f"{path}.args[{i}]")


def _loc(node: Any) -> Mapping[str, int] | None:
    return node.get("loc") if isinstance(node, Mapping) else None


def _typed_items(items: list[Mapping[str, Any]], path: str) -> Iterator[RequirementUse]:
    for i, item in enumerate(items):
        if "type" in item:
            yield RequirementUse(":typing", "una declaración con tipo", f"{path}[{i}]", _loc(item))


def _formula_uses(
    node: Mapping[str, Any], path: str, objects: frozenset[str]
) -> Iterator[RequirementUse]:
    op = node["op"]
    loc = _loc(node)
    if op == "preference":
        yield RequirementUse(":preferences", "una preferencia", path, loc)
        yield from _formula_uses(node["arg"], f"{path}.arg", objects)
    elif op == "atom":
        yield from _term_uses(node["args"], path, loc)
    elif op == "=":
        yield RequirementUse(":equality", "'=' en una fórmula", path, loc)
        yield from _term_uses(node["args"], path, loc)
    elif op == "compare" and _object_equality(node, objects):
        yield RequirementUse(":equality", "'=' en una fórmula", path, loc)
        yield RequirementUse(":object-fluents", "una función de objetos", path, loc)
    elif op == "compare":
        yield RequirementUse(":numeric-fluents", "una comparación numérica", path, loc)
    elif op == "not":
        if node["arg"]["op"] in ("atom", "=", "compare"):
            yield RequirementUse(":negative-preconditions", "'not' en una fórmula", path, loc)
        else:
            yield RequirementUse(
                ":disjunctive-preconditions", "'not' sobre una fórmula compuesta", path, loc
            )
        yield from _formula_uses(node["arg"], f"{path}.arg", objects)
    elif op in ("or", "imply"):
        yield RequirementUse(":disjunctive-preconditions", f"'{op}' en una fórmula", path, loc)
    elif op == "exists":
        yield RequirementUse(":existential-preconditions", "'exists' en una fórmula", path, loc)
    elif op == "forall":
        yield RequirementUse(":universal-preconditions", "'forall' en una fórmula", path, loc)
    if op in ("exists", "forall"):
        yield from _typed_items(node["parameters"], f"{path}.parameters")
        yield from _formula_uses(node["arg"], f"{path}.arg", objects)
    elif op in ("and", "or", "imply"):
        for i, arg in enumerate(node["args"]):
            yield from _formula_uses(arg, f"{path}.args[{i}]", objects)


def _effect_uses(
    node: Mapping[str, Any], path: str, objects: frozenset[str]
) -> Iterator[RequirementUse]:
    op = node["op"]
    if op == "and":
        for i, arg in enumerate(node["args"]):
            yield from _effect_uses(arg, f"{path}.args[{i}]", objects)
    elif op == "forall":
        yield RequirementUse(":conditional-effects", "'forall' en un efecto", path, _loc(node))
        yield from _typed_items(node["parameters"], f"{path}.parameters")
        yield from _effect_uses(node["arg"], f"{path}.arg", objects)
    elif op == "when":
        yield RequirementUse(":conditional-effects", "'when' en un efecto", path, _loc(node))
        yield from _formula_uses(node["condition"], f"{path}.condition", objects)
        yield from _effect_uses(node["arg"], f"{path}.arg", objects)
    elif op == "atom":
        yield from _term_uses(node["args"], path, _loc(node))
    elif op == "not":
        yield from _term_uses(node["arg"]["args"], path, _loc(node))
    elif op in NUMERIC_EFFECTS and node["fluent"]["function"] in objects:
        yield RequirementUse(
            ":object-fluents", f"'{op}' de una función de objetos", path, _loc(node)
        )
    elif op in NUMERIC_EFFECTS:
        value = node["value"]
        cost_only = (
            op == "increase"
            and _is_total_cost(node["fluent"])
            and (value["op"] == "num" or value["op"] == "fluent")
        )
        yield RequirementUse(
            ":numeric-fluents",
            f"'{op}' en un efecto",
            path,
            _loc(node),
            _COSTS if cost_only else (),
        )


#: Operadores de los efectos numéricos.
NUMERIC_EFFECTS = ("increase", "decrease", "assign", "scale-up", "scale-down")


def _duration_uses(constraints: list[Mapping[str, Any]], path: str) -> Iterator[RequirementUse]:
    for i, constraint in enumerate(constraints):
        where = f"{path}[{i}]"
        if i == 1:
            yield RequirementUse(
                ":duration-inequalities",
                "varias restricciones de duración",
                where,
                _loc(constraint),
            )
        if constraint["comparator"] != "=":
            yield RequirementUse(
                ":duration-inequalities",
                f"'{constraint['comparator']}' en la duración",
                where,
                _loc(constraint),
            )
        if "time" in constraint:
            yield RequirementUse(
                ":duration-inequalities",
                f"una duración evaluada en (at {constraint['time']} ...)",
                where,
                _loc(constraint),
            )


def _timed_condition_uses(
    node: Mapping[str, Any], path: str, objects: frozenset[str]
) -> Iterator[RequirementUse]:
    op = node["op"]
    if op == "and":
        for i, arg in enumerate(node["args"]):
            yield from _timed_condition_uses(arg, f"{path}.args[{i}]", objects)
    elif op == "forall":
        yield RequirementUse(
            ":universal-preconditions", "'forall' en una condición", path, _loc(node)
        )
        yield from _typed_items(node["parameters"], f"{path}.parameters")
        yield from _timed_condition_uses(node["arg"], f"{path}.arg", objects)
    elif op == "preference":
        yield RequirementUse(":preferences", "una preferencia", path, _loc(node))
        yield from _timed_condition_uses(node["arg"], f"{path}.arg", objects)
    else:  # at-start, at-end, over-all
        yield from _formula_uses(node["arg"], f"{path}.arg", objects)


def _constraint_uses(
    node: Mapping[str, Any], path: str, objects: frozenset[str]
) -> Iterator[RequirementUse]:
    op = node["op"]
    loc = _loc(node)
    if op == "and":
        for i, arg in enumerate(node["args"]):
            yield from _constraint_uses(arg, f"{path}.args[{i}]", objects)
    elif op == "forall":
        yield RequirementUse(":universal-preconditions", "'forall' en una restricción", path, loc)
        yield from _typed_items(node["parameters"], f"{path}.parameters")
        yield from _constraint_uses(node["arg"], f"{path}.arg", objects)
    elif op == "preference":
        yield RequirementUse(":preferences", "una preferencia", path, loc)
        yield from _constraint_uses(node["arg"], f"{path}.arg", objects)
    else:
        yield RequirementUse(":constraints", f"la restricción '{op}'", path, loc)
        if "arg" in node:
            yield from _formula_uses(node["arg"], f"{path}.arg", objects)
        for i, arg in enumerate(node.get("args", ())):
            yield from _formula_uses(arg, f"{path}.args[{i}]", objects)


def _timed_effect_uses(
    node: Mapping[str, Any], path: str, objects: frozenset[str]
) -> Iterator[RequirementUse]:
    op = node["op"]
    if op == "and":
        for i, arg in enumerate(node["args"]):
            yield from _timed_effect_uses(arg, f"{path}.args[{i}]", objects)
    elif op == "forall":
        yield RequirementUse(":conditional-effects", "'forall' en un efecto", path, _loc(node))
        yield from _typed_items(node["parameters"], f"{path}.parameters")
        yield from _timed_effect_uses(node["arg"], f"{path}.arg", objects)
    elif op == "when":
        yield RequirementUse(":conditional-effects", "'when' en un efecto", path, _loc(node))
        yield from _timed_condition_uses(node["condition"], f"{path}.condition", objects)
        yield from _timed_effect_uses(node["arg"], f"{path}.arg", objects)
    elif op in ("at-start", "at-end"):
        yield from _effect_uses(node["arg"], f"{path}.arg", objects)
    else:  # increase o decrease continuo
        yield RequirementUse(":continuous-effects", "un efecto continuo", path, _loc(node))
