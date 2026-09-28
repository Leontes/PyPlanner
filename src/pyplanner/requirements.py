"""Requisitos PDDL (``:requirements``): los que existen, los que implican otros y los soportados.

Este módulo es la única fuente de verdad sobre qué requisitos acepta el planificador. Las dos
fases del parser lo consultan: la fase 1 para rechazar pronto un fichero PDDL que no se
puede procesar y la fase 2 porque su JSON de entrada puede venir de cualquier otra fuente.

Cada iteración del roadmap añade sus requisitos a :data:`SUPPORTED_REQUIREMENTS`.

**Modo estricto.** Toda construcción que se use tiene que tener su requisito declarado
(:func:`requirement_uses` y :func:`first_undeclared`). Las reglas siguen la gramática de
PDDL 2.1:

* sección ``:types`` o listas con tipos → ``:typing``;
* ``(not <átomo>)`` en una fórmula → ``:negative-preconditions``;
* ``(not <fórmula compuesta>)``, ``or`` e ``imply`` → ``:disjunctive-preconditions``;
* ``exists`` → ``:existential-preconditions``; ``forall`` en una fórmula →
  ``:universal-preconditions``;
* ``=`` → ``:equality`` (y ``(not (= ...))`` además ``:negative-preconditions``);
* ``forall`` y ``when`` en un efecto → ``:conditional-effects``.

``(not <átomo>)`` en un efecto es un borrado STRIPS y no necesita nada.
"""

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
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
        # PDDL 3.x (fuera del alcance del proyecto)
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

#: Requisitos que el planificador soporta en la iteración actual.
SUPPORTED_REQUIREMENTS: frozenset[str] = frozenset(
    {
        # Iteración 1
        ":strips",
        # Iteración 2 (PDDL 1.2, subconjunto ADL)
        ":typing",
        ":negative-preconditions",
        ":disjunctive-preconditions",
        ":equality",
        ":existential-preconditions",
        ":universal-preconditions",
        ":conditional-effects",
    }
)

#: Requisito que se supone cuando un dominio no declara ``:requirements``.
DEFAULT_REQUIREMENT = ":strips"


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
    """

    requirement: str
    construct: str
    path: str
    loc: Mapping[str, int] | None = None


def requirement_uses(document: Mapping[str, Any]) -> Iterator[RequirementUse]:
    """Recorre un documento JSON (dominio o problema) y devuelve cada construcción que
    necesita un requisito, en orden de aparición.

    Se supone que el documento cumple el esquema.
    """
    if document.get("types"):
        yield RequirementUse(":typing", "la sección :types", "types", _loc(document["types"][0]))
    for key in ("constants", "objects"):
        yield from _typed_items(document.get(key, []), key)
    for i, predicate in enumerate(document.get("predicates", [])):
        yield from _typed_items(predicate.get("parameters", []), f"predicates[{i}].parameters")
    for i, action in enumerate(document.get("actions", [])):
        path = f"actions[{i}]"
        yield from _typed_items(action.get("parameters", []), f"{path}.parameters")
        if "precondition" in action:
            yield from _formula_uses(action["precondition"], f"{path}.precondition")
        if "effect" in action:
            yield from _effect_uses(action["effect"], f"{path}.effect")
    if "goal" in document:
        yield from _formula_uses(document["goal"], "goal")


def first_undeclared(document: Mapping[str, Any], declared: Iterable[str]) -> RequirementUse | None:
    """La primera construcción de ``document`` cuyo requisito no está en ``declared``.

    ``declared`` se expande antes de comparar (``:adl`` cubre ``:typing``, etc.).
    """
    expanded = expand_requirements(declared)
    for use in requirement_uses(document):
        if use.requirement not in expanded:
            return use
    return None


def _loc(node: Any) -> Mapping[str, int] | None:
    return node.get("loc") if isinstance(node, Mapping) else None


def _typed_items(items: list[Mapping[str, Any]], path: str) -> Iterator[RequirementUse]:
    for i, item in enumerate(items):
        if "type" in item:
            yield RequirementUse(":typing", "una declaración con tipo", f"{path}[{i}]", _loc(item))


def _formula_uses(node: Mapping[str, Any], path: str) -> Iterator[RequirementUse]:
    op = node["op"]
    loc = _loc(node)
    if op == "=":
        yield RequirementUse(":equality", "'=' en una fórmula", path, loc)
    elif op == "not":
        if node["arg"]["op"] in ("atom", "="):
            yield RequirementUse(":negative-preconditions", "'not' en una fórmula", path, loc)
        else:
            yield RequirementUse(
                ":disjunctive-preconditions", "'not' sobre una fórmula compuesta", path, loc
            )
        yield from _formula_uses(node["arg"], f"{path}.arg")
    elif op in ("or", "imply"):
        yield RequirementUse(":disjunctive-preconditions", f"'{op}' en una fórmula", path, loc)
    elif op == "exists":
        yield RequirementUse(":existential-preconditions", "'exists' en una fórmula", path, loc)
    elif op == "forall":
        yield RequirementUse(":universal-preconditions", "'forall' en una fórmula", path, loc)
    if op in ("exists", "forall"):
        yield from _typed_items(node["parameters"], f"{path}.parameters")
        yield from _formula_uses(node["arg"], f"{path}.arg")
    elif op in ("and", "or", "imply"):
        for i, arg in enumerate(node["args"]):
            yield from _formula_uses(arg, f"{path}.args[{i}]")


def _effect_uses(node: Mapping[str, Any], path: str) -> Iterator[RequirementUse]:
    op = node["op"]
    if op == "and":
        for i, arg in enumerate(node["args"]):
            yield from _effect_uses(arg, f"{path}.args[{i}]")
    elif op == "forall":
        yield RequirementUse(":conditional-effects", "'forall' en un efecto", path, _loc(node))
        yield from _typed_items(node["parameters"], f"{path}.parameters")
        yield from _effect_uses(node["arg"], f"{path}.arg")
    elif op == "when":
        yield RequirementUse(":conditional-effects", "'when' en un efecto", path, _loc(node))
        yield from _formula_uses(node["condition"], f"{path}.condition")
        yield from _effect_uses(node["arg"], f"{path}.arg")
