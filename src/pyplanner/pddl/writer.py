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
    constants = document.get("constants", [])
    if constants:
        lines.append(f"{_INDENT}(:constants {_names(constants)})")
    predicates = document.get("predicates", [])
    if predicates:
        lines.append(f"{_INDENT}(:predicates")
        for predicate in predicates:
            lines.append(f"{_INDENT * 2}{_signature(predicate)}")
        lines.append(f"{_INDENT})")
    for action in document.get("actions", []):
        lines.append(f"{_INDENT}(:action {action['name']}")
        lines.append(f"{_INDENT * 2}:parameters ({_names(action.get('parameters', []))})")
        if "precondition" in action:
            lines.append(f"{_INDENT * 2}:precondition {_formula(action['precondition'])}")
        if "effect" in action:
            lines.append(f"{_INDENT * 2}:effect {_formula(action['effect'])}")
        lines.append(f"{_INDENT})")
    lines.append(")")
    return "\n".join(lines) + "\n"


def _write_problem(document: Mapping[str, Any]) -> str:
    lines = [f"(define (problem {document['name']})", f"{_INDENT}(:domain {document['domain']})"]
    lines += _requirements(document)
    objects = document.get("objects", [])
    if objects:
        lines.append(f"{_INDENT}(:objects {_names(objects)})")
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


def _names(items: list[Mapping[str, Any]]) -> str:
    return " ".join(item["name"] for item in items)


def _signature(predicate: Mapping[str, Any]) -> str:
    parameters = predicate.get("parameters", [])
    return f"({' '.join([predicate['name'], *(p['name'] for p in parameters)])})"


def _formula(node: Mapping[str, Any]) -> str:
    """Escribe una fórmula o un efecto (comparten representación en la v1)."""
    op = node["op"]
    if op == "atom":
        return f"({' '.join([node['predicate'], *node['args']])})"
    if op == "not":
        return f"(not {_formula(node['arg'])})"
    if op == "and":
        return f"({' '.join(['and', *(_formula(arg) for arg in node['args'])])})"
    raise ValueError(f"operador desconocido: {op}")
