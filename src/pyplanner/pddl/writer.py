"""Conversor inverso: documento JSON → texto PDDL.

Sirve para dos cosas:

* comprobar en los tests que la fase 1 no pierde información (PDDL → JSON → PDDL → JSON
  tiene que dar el mismo documento);
* pasarle a herramientas externas como VAL una tarea que llegó directamente en JSON.

El documento se supone válido según el esquema; no se vuelve a comprobar aquí. Una
precondición, una condición o un efecto ausentes se escriben como ``(and)``, porque VAL no
acepta acciones sin ``:precondition`` ni ``:condition ()`` en las acciones durativas. Es
equivalente: una precondición ausente es la vacía.
"""

from collections.abc import Callable, Mapping
from typing import Any

_INDENT = "  "
_EMPTY = {"op": "and", "args": []}


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
    functions = document.get("functions", [])
    if functions:
        lines.append(f"{_INDENT}(:functions")
        for function in functions:
            lines.append(f"{_INDENT * 2}{_signature(function)} - number")
        lines.append(f"{_INDENT})")
    for action in document.get("actions", []):
        lines.append(f"{_INDENT}(:action {action['name']}")
        lines.append(f"{_INDENT * 2}:parameters ({_typed_list(action.get('parameters', []))})")
        # VAL exige :precondition y :effect aunque estén vacíos, así que se escriben siempre.
        precondition = action.get("precondition", _EMPTY)
        lines.append(f"{_INDENT * 2}:precondition {_formula(precondition)}")
        lines.append(f"{_INDENT * 2}:effect {_effect(action.get('effect', _EMPTY))}")
        lines.append(f"{_INDENT})")
    for action in document.get("durative_actions", []):
        lines.append(f"{_INDENT}(:durative-action {action['name']}")
        lines.append(f"{_INDENT * 2}:parameters ({_typed_list(action.get('parameters', []))})")
        lines.append(f"{_INDENT * 2}:duration {_duration(action['duration'])}")
        condition = action.get("condition", _EMPTY)
        lines.append(f"{_INDENT * 2}:condition {_timed(condition, _formula)}")
        lines.append(f"{_INDENT * 2}:effect {_timed(action.get('effect', _EMPTY), _effect)}")
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
    for element in document.get("init", []):
        if element["op"] == "=":
            value = _number(element["value"])
            lines.append(f"{_INDENT * 2}(= {_expression(element['fluent'])} {value})")
        else:
            lines.append(f"{_INDENT * 2}{_formula(element)}")
    lines.append(f"{_INDENT})")
    lines.append(f"{_INDENT}(:goal {_formula(document['goal'])})")
    if "metric" in document:
        metric = document["metric"]
        expression = _expression(metric["expression"])
        lines.append(f"{_INDENT}(:metric {metric['direction']} {expression})")
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
    if op == "compare":
        left, right = _expression(node["left"]), _expression(node["right"])
        return f"({node['comparator']} {left} {right})"
    raise ValueError(f"operador de fórmula desconocido: {op}")


def _expression(node: Mapping[str, Any]) -> str:
    """Escribe una expresión numérica."""
    op = node["op"]
    if op == "num":
        return _number(node["value"])
    if op == "fluent":
        return f"({' '.join([node['function'], *node['args']])})"
    if op == "total-time":
        return "(total-time)"
    if op == "duration":
        return "?duration"
    if op == "#t":
        return "#t"
    return f"({' '.join([op, *(_expression(arg) for arg in node['args'])])})"


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(float(value))


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
    if op in ("increase", "decrease", "assign", "scale-up", "scale-down"):
        return f"({op} {_expression(node['fluent'])} {_expression(node['value'])})"
    raise ValueError(f"operador de efecto desconocido: {op}")


def _duration(constraints: list[Mapping[str, Any]]) -> str:
    """Escribe la duración: ``()``, una restricción o ``(and ...)`` de varias."""
    parts = []
    for constraint in constraints:
        text = f"({constraint['comparator']} ?duration {_expression(constraint['value'])})"
        parts.append(f"(at {constraint['time']} {text})" if "time" in constraint else text)
    if not parts:
        return "()"
    return parts[0] if len(parts) == 1 else f"(and {' '.join(parts)})"


_TIMED = {"at-start": "at start", "at-end": "at end", "over-all": "over all"}


def _timed(node: Mapping[str, Any], inner: Callable[[Mapping[str, Any]], str]) -> str:
    """Escribe una condición o un efecto de una acción durativa; ``inner`` escribe lo que va
    dentro de ``at start``, ``at end`` y ``over all``."""
    op = node["op"]
    if op in _TIMED:
        return f"({_TIMED[op]} {inner(node['arg'])})"
    if op == "and":
        return f"({' '.join(['and', *(_timed(arg, inner) for arg in node['args'])])})"
    if op == "forall":
        return f"(forall ({_typed_list(node['parameters'])}) {_timed(node['arg'], inner)})"
    if op == "when":
        return f"(when {_timed(node['condition'], _formula)} {_timed(node['arg'], inner)})"
    return _effect(node)  # efecto continuo
