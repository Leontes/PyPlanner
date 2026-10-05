"""Funciones de objetos (``:object-fluents``, PDDL 3.1) compiladas a predicados.

La fase 2 del parser las elimina del documento JSON antes de construir el AST, así que ni
el AST ni el motor las ven, y todo lo demás (ADL, tiempo, preferencias…) funciona con ellas
sin cambios. Es la compilación de Fast Downward:

* cada función de objetos ``(f ?x1 … ?xn) - t`` pasa a ser un predicado ``(f ?x1 … ?xn
  ?value - t)``, cierto para su valor actual (ninguno si está indefinida);
* un término ``(f a)`` dentro de un átomo o de una igualdad se sustituye por una variable
  nueva ligada con ``exists``: ``(at (loc ?r))`` → ``(exists (?v - place) (and (loc ?r ?v)
  (at ?v)))``. Si la función está indefinida, el átomo es falso;
* en los efectos, el término se evalúa en el estado anterior, como cualquier condición:
  ``(at (loc ?r) ?p)`` → ``(forall (?v) (when (loc ?r ?v) (at ?v ?p)))``. En las acciones
  durativas, la condición lleva el momento del efecto (``(at end …)``);
* ``(assign (f a) v)`` borra el valor anterior y añade el nuevo (si ``v`` es ``undefined``,
  solo borra). Si coinciden, queda cierto: el añadido gana;
* un valor inicial ``(= (f a) b)`` pasa a ser el hecho ``(f a b)``.

El plan no cambia (no se añaden parámetros a las acciones), así que se puede validar con VAL
contra el dominio compilado: VAL no admite funciones de objetos. Lo que no tiene una
traducción sencilla da un error claro: funciones de objetos dentro de la métrica, de las
duraciones o de un efecto continuo.
"""

import copy
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, NoReturn

from pyplanner.errors import SemanticError, SourceLocation

JSONDict = dict[str, Any]
#: Una variable nueva, su tipo y el átomo que la liga al valor de un término.
Binding = tuple[str, Any, JSONDict]

#: Requisitos que necesita el documento compilado.
COMPILED_REQUIREMENTS = (":existential-preconditions", ":conditional-effects")


@dataclass(frozen=True, eq=False)
class ObjectFunction:
    """Una función de objetos y el predicado que la representa tras compilarla.

    Attributes:
        name: nombre de la función (en minúsculas).
        parameters: sus parámetros (nodos JSON con ``name`` y, si lo tiene, ``type``).
        type: tipo de su valor (un tipo o una lista de tipos, como en el JSON).
        predicate: nombre del predicado compilado.
    """

    name: str
    parameters: tuple[JSONDict, ...]
    type: Any
    predicate: str


def object_functions(domain: Mapping[str, Any]) -> tuple[ObjectFunction, ...]:
    """Funciones de objetos declaradas en un documento de dominio. El predicado se llama
    como la función, salvo que ya exista un predicado con ese nombre."""
    taken = {p["name"].lower() for p in domain.get("predicates", [])}
    result = []
    for function in domain.get("functions", []):
        if "type" not in function:
            continue
        name = function["name"].lower()
        predicate = name
        while predicate in taken:
            predicate = f"{predicate}-value"
        taken.add(predicate)
        parameters = tuple(function.get("parameters", []))
        result.append(ObjectFunction(name, parameters, function["type"], predicate))
    return tuple(result)


def compile_domain(document: Mapping[str, Any], functions: tuple[ObjectFunction, ...]) -> JSONDict:
    """El documento de dominio sin funciones de objetos (ver la documentación del módulo)."""
    compiler = _Compiler(document, functions)
    result = copy.deepcopy(dict(document))
    names = {f.name for f in functions}
    result["functions"] = [f for f in result.get("functions", []) if f["name"].lower() not in names]
    result["predicates"] = [
        *result.get("predicates", []),
        *(
            {
                "name": f.predicate,
                "parameters": [*f.parameters, {"name": "?value", "type": f.type}],
            }
            for f in functions
        ),
    ]
    result["requirements"] = _with_requirements(result.get("requirements", []))
    for action in result.get("actions", []):
        if "precondition" in action:
            action["precondition"] = compiler.formula(action["precondition"])
        if "effect" in action:
            action["effect"] = compiler.effect(action["effect"])
    for action in result.get("durative_actions", []):
        for constraint in action["duration"]:
            compiler.no_terms_in_expression(constraint["value"], "una duración")
        if "condition" in action:
            action["condition"] = compiler.timed_condition(action["condition"])
        if "effect" in action:
            action["effect"] = compiler.timed_effect(action["effect"])
    for rule in result.get("derived", []):
        rule["condition"] = compiler.formula(rule["condition"])
    if "constraints" in result:
        result["constraints"] = compiler.constraint(result["constraints"])
    return result


def compile_problem(document: Mapping[str, Any], functions: tuple[ObjectFunction, ...]) -> JSONDict:
    """El documento de problema sin funciones de objetos: sus valores iniciales pasan a ser
    hechos y sus fórmulas se compilan como las del dominio."""
    compiler = _Compiler(document, functions)
    result = copy.deepcopy(dict(document))
    init = []
    for element in result.get("init", []):
        if element["op"] == "=":
            fluent = element["fluent"]
            function = compiler.functions.get(fluent["function"].lower())
            if any(not isinstance(arg, str) for arg in fluent["args"]):
                compiler.fail("en :init los argumentos de una función son objetos", element)
            if function is not None:
                if not isinstance(element["value"], str):
                    compiler.fail(
                        f"la función de objetos '{function.name}' toma un objeto como valor",
                        element,
                    )
                atom = {
                    "op": "atom",
                    "predicate": function.predicate,
                    "args": [*fluent["args"], element["value"]],
                }
                if "loc" in element:
                    atom["loc"] = element["loc"]
                init.append(atom)
                continue
        elif element["op"] == "at" and compiler.has_terms(element["literal"]):
            compiler.fail("un literal inicial temporizado no admite funciones de objetos", element)
        init.append(element)
    result["init"] = init
    result["goal"] = compiler.formula(result["goal"])
    if "constraints" in result:
        result["constraints"] = compiler.constraint(result["constraints"])
    if "metric" in result:
        compiler.no_terms_in_expression(result["metric"]["expression"], "la métrica")
    if result.get("requirements"):
        result["requirements"] = _with_requirements(result["requirements"])
    return result


def _with_requirements(requirements: list[str]) -> list[str]:
    """Los requisitos del documento compilado: ya no usa funciones de objetos (VAL no admite
    ``:object-fluents``) y sí cuantificadores y efectos condicionales."""
    kept = [r for r in requirements if r.lower() != ":object-fluents"]
    return [*kept, *(r for r in COMPILED_REQUIREMENTS if r not in kept)]


class _Compiler:
    """Sustituye los términos de funciones de objetos de un documento."""

    def __init__(self, document: Mapping[str, Any], functions: tuple[ObjectFunction, ...]):
        self.source = document.get("source")
        self.functions = {f.name: f for f in functions}
        self.used = set(_variables(document))
        self.counter = 0

    def fail(self, message: str, node: Mapping[str, Any]) -> NoReturn:
        loc = node.get("loc")
        location = SourceLocation(loc["line"], loc["column"], self.source) if loc else None
        raise SemanticError(message, location)

    def fresh(self) -> str:
        while True:
            name = f"?value{self.counter}"
            self.counter += 1
            if name not in self.used:
                self.used.add(name)
                return name

    # --- Términos ----------------------------------------------------------------------

    def term(self, term: Any) -> tuple[str, list[Binding]]:
        """Un término simple (objeto o variable) y las ligaduras que necesita."""
        if isinstance(term, str):
            return term, []
        function = self.functions.get(term["function"].lower())
        if function is None:
            self.fail(
                f"la función '{term['function']}' es numérica: no puede ir como argumento",
                term,
            )
        args, bindings = self.terms(term["args"])
        value = self.fresh()
        atom = {"op": "atom", "predicate": function.predicate, "args": [*args, value]}
        return value, [*bindings, (value, function.type, atom)]

    def terms(self, terms: list[Any]) -> tuple[list[str], list[Binding]]:
        simple: list[str] = []
        bindings: list[Binding] = []
        for term in terms:
            name, more = self.term(term)
            simple.append(name)
            bindings.extend(more)
        return simple, bindings

    def has_terms(self, node: Any) -> bool:
        """``True`` si ``node`` contiene algún término de función de objetos."""
        if isinstance(node, Mapping):
            if node.get("op") == "fluent" and node["function"].lower() in self.functions:
                return True
            return any(self.has_terms(value) for value in node.values())
        if isinstance(node, list):
            return any(self.has_terms(item) for item in node)
        return False

    def expression(self, node: JSONDict) -> tuple[JSONDict, list[Binding]]:
        """Una expresión numérica cuyas funciones pueden llevar términos como argumento."""
        op = node["op"]
        if op == "fluent":
            if node["function"].lower() in self.functions:
                self.fail(
                    f"la función de objetos '{node['function']}' no puede usarse como número",
                    node,
                )
            args, bindings = self.terms(node["args"])
            return {**node, "args": args}, bindings
        if "args" in node:
            parts = [self.expression(arg) for arg in node["args"]]
            return {**node, "args": [p for p, _ in parts]}, [b for _, bs in parts for b in bs]
        return node, []

    def no_terms_in_expression(self, node: Mapping[str, Any], where: str) -> None:
        if self.has_terms(node):
            self.fail(f"las funciones de objetos no están admitidas en {where}", node)

    # --- Fórmulas ----------------------------------------------------------------------

    def formula(self, node: JSONDict) -> JSONDict:
        op = node["op"]
        if op == "atom":
            args, bindings = self.terms(node["args"])
            return _exists({**node, "args": args}, bindings)
        if op == "=":
            args, bindings = self.terms(node["args"])
            return _exists({**node, "args": args}, bindings)
        if op == "compare":
            left, right = node["left"], node["right"]
            if node["comparator"] == "=" and all(
                side["op"] == "fluent" and side["function"].lower() in self.functions
                for side in (left, right)
            ):
                # Igualdad de dos funciones de objetos (la fase 1 no puede distinguirla).
                args, bindings = self.terms([left, right])
                equality = {"op": "=", "args": args}
                if "loc" in node:
                    equality["loc"] = node["loc"]
                return _exists(equality, bindings)
            new_left, left_bindings = self.expression(left)
            new_right, right_bindings = self.expression(right)
            compiled = {**node, "left": new_left, "right": new_right}
            return _exists(compiled, left_bindings + right_bindings)
        if op in ("not", "exists", "forall", "preference"):
            return {**node, "arg": self.formula(node["arg"])}
        if op in ("and", "or", "imply"):
            return {**node, "args": [self.formula(arg) for arg in node["args"]]}
        return node

    def timed_condition(self, node: JSONDict) -> JSONDict:
        op = node["op"]
        if op == "and":
            return {**node, "args": [self.timed_condition(arg) for arg in node["args"]]}
        if op in ("forall", "preference"):
            return {**node, "arg": self.timed_condition(node["arg"])}
        return {**node, "arg": self.formula(node["arg"])}  # at-start, at-end, over-all

    def constraint(self, node: JSONDict) -> JSONDict:
        op = node["op"]
        if op == "and":
            return {**node, "args": [self.constraint(arg) for arg in node["args"]]}
        if op in ("forall", "preference"):
            return {**node, "arg": self.constraint(node["arg"])}
        if "arg" in node:
            return {**node, "arg": self.formula(node["arg"])}
        return {**node, "args": [self.formula(arg) for arg in node["args"]]}

    # --- Efectos -----------------------------------------------------------------------

    def effect(self, node: JSONDict) -> JSONDict:
        parts = list(self._simple_effects(node, [], []))
        if len(parts) == 1 and not parts[0][0] and not parts[0][1]:
            return parts[0][2]
        return {"op": "and", "args": [_rebuild(*part) for part in parts]}

    def _simple_effects(
        self, node: JSONDict, variables: list[JSONDict], conditions: list[JSONDict]
    ) -> Iterator[tuple[list[JSONDict], list[JSONDict], JSONDict]]:
        """Efectos simples (literal o numérico) con las variables y condiciones que los
        envuelven, ya sin términos de funciones de objetos."""
        op = node["op"]
        if op == "and":
            for arg in node["args"]:
                yield from self._simple_effects(arg, variables, conditions)
        elif op == "forall":
            yield from self._simple_effects(
                node["arg"], [*variables, *node["parameters"]], conditions
            )
        elif op == "when":
            condition = self.formula(node["condition"])
            yield from self._simple_effects(node["arg"], variables, [*conditions, condition])
        elif op in ("atom", "not"):
            atom = node if op == "atom" else node["arg"]
            args, bindings = self.terms(atom["args"])
            literal = {**atom, "args": args}
            if op == "not":
                literal = {**node, "arg": literal}
            yield self._bound(variables, conditions, bindings, literal)
        else:  # efecto numérico o assign
            yield from self._assignment(node, variables, conditions)

    def _assignment(
        self, node: JSONDict, variables: list[JSONDict], conditions: list[JSONDict]
    ) -> Iterator[tuple[list[JSONDict], list[JSONDict], JSONDict]]:
        fluent = node["fluent"]
        function = self.functions.get(fluent["function"].lower())
        if function is None:  # efecto numérico: sus funciones pueden llevar términos
            target, target_bindings = self.expression(fluent)
            raw = node["value"]
            value, value_bindings = (
                self.expression(raw)
                if isinstance(raw, dict) and raw["op"] != "undefined"
                else (raw, [])
            )
            compiled = {**node, "fluent": target, "value": value}
            yield self._bound(variables, conditions, target_bindings + value_bindings, compiled)
            return
        if node["op"] != "assign":
            self.fail(f"una función de objetos solo admite 'assign', no '{node['op']}'", node)
        args, bindings = self.terms(fluent["args"])
        old = self.fresh()
        current = {"op": "atom", "predicate": function.predicate, "args": [*args, old]}
        delete = {"op": "not", "arg": current}
        yield self._bound(variables, conditions, [*bindings, (old, function.type, current)], delete)
        value = node["value"]
        if isinstance(value, Mapping) and value["op"] == "undefined":
            return
        new, value_bindings = self.term(value)
        add = {"op": "atom", "predicate": function.predicate, "args": [*args, new]}
        yield self._bound(variables, conditions, bindings + value_bindings, add)

    def _bound(
        self,
        variables: list[JSONDict],
        conditions: list[JSONDict],
        bindings: list[Binding],
        effect: JSONDict,
    ) -> tuple[list[JSONDict], list[JSONDict], JSONDict]:
        names = {v["name"] for v in variables}
        extra = [{"name": v, "type": t} for v, t, _ in bindings if v not in names]
        return [*variables, *extra], [*conditions, *(atom for _, _, atom in bindings)], effect

    def timed_effect(self, node: JSONDict) -> JSONDict:
        """Efecto de una acción durativa: las condiciones nuevas llevan el momento de cada
        efecto (``(at end (loc ?r ?v))``)."""
        op = node["op"]
        if op == "and":
            return {**node, "args": [self.timed_effect(arg) for arg in node["args"]]}
        if op == "forall":
            return {**node, "arg": self.timed_effect(node["arg"])}
        if op == "when":
            condition = self.timed_condition(node["condition"])
            return _merge_when(condition, self.timed_effect(node["arg"]))
        if op in ("at-start", "at-end"):
            parts = list(self._simple_effects(node["arg"], [], []))
            rebuilt = [_rebuild_timed(op, *part) for part in parts]
            return rebuilt[0] if len(rebuilt) == 1 else {"op": "and", "args": rebuilt}
        self.no_terms_in_expression(node, "un efecto continuo")
        return node


def _exists(formula: JSONDict, bindings: list[Binding]) -> JSONDict:
    if not bindings:
        return formula
    return {
        "op": "exists",
        "parameters": [{"name": v, "type": t} for v, t, _ in bindings],
        "arg": {"op": "and", "args": [*(atom for _, _, atom in bindings), formula]},
    }


def _rebuild(variables: list[JSONDict], conditions: list[JSONDict], effect: JSONDict) -> JSONDict:
    result = effect
    if conditions:
        condition = conditions[0] if len(conditions) == 1 else {"op": "and", "args": conditions}
        result = {"op": "when", "condition": condition, "arg": result}
    if variables:
        result = {"op": "forall", "parameters": variables, "arg": result}
    return result


def _rebuild_timed(
    time: str, variables: list[JSONDict], conditions: list[JSONDict], effect: JSONDict
) -> JSONDict:
    result: JSONDict = {"op": time, "arg": effect}
    if conditions:
        condition = conditions[0] if len(conditions) == 1 else {"op": "and", "args": conditions}
        result = {"op": "when", "condition": {"op": time, "arg": condition}, "arg": result}
    if variables:
        result = {"op": "forall", "parameters": variables, "arg": result}
    return result


def _merge_when(condition: JSONDict, arg: JSONDict) -> JSONDict:
    """``(when C E)`` cuyo efecto ``E`` ya es un ``forall``/``when`` compilado: se juntan las
    condiciones en un solo ``when`` (no se admite uno dentro de otro)."""
    if arg["op"] == "and":
        return {"op": "and", "args": [_merge_when(condition, a) for a in arg["args"]]}
    if arg["op"] == "forall":
        return {**arg, "arg": _merge_when(condition, arg["arg"])}
    if arg["op"] == "when":
        both = {"op": "and", "args": [condition, arg["condition"]]}
        return {**arg, "condition": both}
    return {"op": "when", "condition": condition, "arg": arg}


def _variables(node: Any) -> Iterator[str]:
    """Todas las variables que aparecen en un documento (para no repetir nombres)."""
    if isinstance(node, str):
        if node.startswith("?"):
            yield node
    elif isinstance(node, Mapping):
        for value in node.values():
            yield from _variables(value)
    elif isinstance(node, list):
        for item in node:
            yield from _variables(item)
