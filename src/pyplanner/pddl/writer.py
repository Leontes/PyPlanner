"""Conversor inverso: documento JSON → texto PDDL.

Sirve para dos cosas:

* comprobar en los tests que la fase 1 no pierde información (PDDL → JSON → PDDL → JSON
  tiene que dar el mismo documento);
* pasarle a herramientas externas como VAL una tarea que llegó directamente en JSON.

El documento se supone válido según el esquema; no se vuelve a comprobar aquí.
"""

from collections.abc import Mapping
from typing import Any

_INDENT = "  "


def write_pddl(document: Mapping[str, Any]) -> str:
    """Genera el texto PDDL de un documento de dominio o de problema."""
    if document["kind"] == "domain":
        return _write_domain(document)
    return _write_problem(document)


def _write_domain(document: Mapping[str, Any]) -> str:
    lines = [f"(define (domain {document['name']})"]
    lines += _requirements(document)
    types = document.get("types", [])
    if types:
        lines.append(f"{_INDENT}(:types {_typed_list(types, key='parent')})")
    constants = document.get("constants", [])
    if constants:
        lines.append(f"{_INDENT}(:constants {_typed_list(constants)})")
    predicates = document.get("predicates", [])
    if predicates:
        lines.append(f"{_INDENT}(:predicates")
        for predicate in predicates:
            lines.append(f"{_INDENT * 2}{_signature(predicate)}")
        lines.append(f"{_INDENT})")
    for action in document.get("actions", []):
        lines.append(f"{_INDENT}(:action {action['name']}")
        lines.append(f"{_INDENT * 2}:parameters ({_typed_list(action.get('parameters', []))})")
        if "precondition" in action:
            lines.append(f"{_INDENT * 2}:precondition {_formula(action['precondition'])}")
        if "effect" in action:
            lines.append(f"{_INDENT * 2}:effect {_effect(action['effect'])}")
        lines.append(f"{_INDENT})")
    lines.append(")")
    return "\n".join(lines) + "\n"


def _write_problem(document: Mapping[str, Any]) -> str:
    lines = [f"(define (problem {document['name']})", f"{_INDENT}(:domain {document['domain']})"]
    lines += _requirements(document)
    objects = document.get("objects", [])
    if objects:
        lines.append(f"{_INDENT}(:objects {_typed_list(objects)})")
    lines.append(f"{_INDENT}(:init")
    for atom in document.get("init", []):
        lines.append(f"{_INDENT * 2}{_formula(atom)}")
    lines.append(f"{_INDENT})")
    lines.append(f"{_INDENT}(:goal {_formula(document['goal'])})")
    lines.append(")")
    return "\n".join(lines) + "\n"


def _requirements(document: Mapping[str, Any]) -> list[str]:
    requirements = document.get("requirements", [])
    return [f"{_INDENT}(:requirements {' '.join(requirements)})"] if requirements else []


def _typed_list(items: list[Mapping[str, Any]], key: str = "type") -> str:
    """Lista PDDL con tipos: agrupa los elementos consecutivos del mismo tipo.

    Un elemento sin tipo seguido de elementos con tipo se escribe como ``- object``, porque
    en PDDL tomaría el tipo del grupo siguiente.
    """
    parts: list[str] = []
    group: list[str] = []
    group_type: Any = None
    for i, item in enumerate(items):
        item_type = item.get(key)
        if group and item_type != group_type:
            parts.append(_group(group, group_type, typed_after=True))
            group = []
        group.append(item["name"])
        group_type = item_type
        if i == len(items) - 1:
            parts.append(_group(group, group_type, typed_after=False))
    return " ".join(parts)


def _group(names: list[str], type_spec: Any, typed_after: bool) -> str:
    if type_spec is None:
        return " ".join(names) + (" - object" if typed_after else "")
    spec = type_spec if isinstance(type_spec, str) else f"(either {' '.join(type_spec)})"
    return f"{' '.join(names)} - {spec}"


def _signature(predicate: Mapping[str, Any]) -> str:
    parameters = _typed_list(predicate.get("parameters", []))
    return f"({predicate['name']}{' ' + parameters if parameters else ''})"


def _formula(node: Mapping[str, Any]) -> str:
    """Escribe una fórmula."""
    op = node["op"]
    if op == "atom":
        return f"({' '.join([node['predicate'], *node['args']])})"
    if op == "=":
        return f"(= {' '.join(node['args'])})"
    if op == "not":
        return f"(not {_formula(node['arg'])})"
    if op in ("and", "or", "imply"):
        return f"({' '.join([op, *(_formula(arg) for arg in node['args'])])})"
    if op in ("exists", "forall"):
        return f"({op} ({_typed_list(node['parameters'])}) {_formula(node['arg'])})"
    raise ValueError(f"operador de fórmula desconocido: {op}")


def _effect(node: Mapping[str, Any]) -> str:
    """Escribe un efecto."""
    op = node["op"]
    if op == "atom":
        return _formula(node)
    if op == "not":
        return f"(not {_formula(node['arg'])})"
    if op == "and":
        return f"({' '.join(['and', *(_effect(arg) for arg in node['args'])])})"
    if op == "forall":
        return f"(forall ({_typed_list(node['parameters'])}) {_effect(node['arg'])})"
    if op == "when":
        return f"(when {_formula(node['condition'])} {_effect(node['arg'])})"
    raise ValueError(f"operador de efecto desconocido: {op}")
