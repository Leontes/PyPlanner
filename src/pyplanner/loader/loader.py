"""Fase 2 del parser: documento JSON → AST.

Es el punto de entrada del planificador cuando se integra en otro sistema, así que **no se
fía de su entrada**: el JSON puede venir de la fase 1 o de cualquier otro componente. Por
eso hace tres comprobaciones, en este orden:

1. **Estructura**: el documento cumple el esquema (:func:`pyplanner.schema.validate_document`).
2. **Requisitos**: todos los requisitos declarados existen y están soportados.
3. **Semántica**: nombres sin repetir, predicados declarados y con la aridad correcta,
   variables ligadas a un parámetro, objetos declarados y problema asociado a su dominio.

Los errores indican la ruta dentro del JSON y, si el documento trae posiciones (``loc``),
la línea y la columna del PDDL original.
"""

from collections.abc import Iterable, Mapping
from typing import Any, NoReturn

from pyplanner.errors import SemanticError, SourceLocation, UnsupportedFeatureError
from pyplanner.loader.ast import (
    Action,
    And,
    Atom,
    Constant,
    Domain,
    Effect,
    Formula,
    Not,
    Predicate,
    Problem,
    Term,
    Variable,
)
from pyplanner.requirements import (
    KNOWN_REQUIREMENTS,
    expand_requirements,
    unsupported_requirements,
)
from pyplanner.schema import validate_document

JSON = Mapping[str, Any]


def load_domain(document: JSON) -> Domain:
    """Convierte un documento JSON de dominio en un :class:`~pyplanner.loader.ast.Domain`.

    Raises:
        SchemaValidationError: si el documento no cumple el esquema.
        UnsupportedFeatureError: si declara requisitos no soportados.
        SemanticError: si el dominio no es coherente.
    """
    validate_document(document)
    ctx = _Context(document)
    if document["kind"] != "domain":
        ctx.fail("se esperaba un documento de dominio", document, "kind")
    requirements = ctx.requirements(document)

    constants = ctx.unique_names(document.get("constants", []), "constants", "constante")
    predicates: dict[str, Predicate] = {}
    for i, node in enumerate(document.get("predicates", [])):
        path = f"predicates[{i}]"
        name = node["name"].lower()
        if name in predicates:
            ctx.fail(f"predicado '{name}' declarado dos veces", node, path)
        parameters = ctx.parameters(node, path)
        predicates[name] = Predicate(name, parameters)

    scope = _Scope(predicates, frozenset(c.name for c in constants))
    actions: dict[str, Action] = {}
    for i, node in enumerate(document.get("actions", [])):
        path = f"actions[{i}]"
        name = node["name"].lower()
        if name in actions:
            ctx.fail(f"acción '{name}' declarada dos veces", node, path)
        parameters = ctx.parameters(node, path)
        action_scope = scope.with_variables(p.name for p in parameters)
        precondition: Formula = And()
        effect: Effect = And()
        if "precondition" in node:
            precondition = ctx.formula(node["precondition"], f"{path}.precondition", action_scope)
        if "effect" in node:
            effect = ctx.effect(node["effect"], f"{path}.effect", action_scope)
        actions[name] = Action(name, parameters, precondition, effect)

    return Domain(
        name=document["name"].lower(),
        requirements=requirements,
        constants=constants,
        predicates=tuple(predicates.values()),
        actions=tuple(actions.values()),
    )


def load_problem(document: JSON, domain: Domain) -> Problem:
    """Convierte un documento JSON de problema en un :class:`~pyplanner.loader.ast.Problem`.

    El problema se comprueba contra ``domain``: tiene que referirse a él y usar solo sus
    predicados.

    Raises:
        SchemaValidationError: si el documento no cumple el esquema.
        UnsupportedFeatureError: si declara requisitos no soportados.
        SemanticError: si el problema no es coherente con el dominio.
    """
    validate_document(document)
    ctx = _Context(document)
    if document["kind"] != "problem":
        ctx.fail("se esperaba un documento de problema", document, "kind")
    domain_name = document["domain"].lower()
    if domain_name != domain.name:
        ctx.fail(
            f"el problema es del dominio '{domain_name}', pero se ha cargado '{domain.name}'",
            document,
            "domain",
        )
    requirements = ctx.requirements(document) | domain.requirements

    constant_names = {c.name for c in domain.constants}
    objects = tuple(
        o
        for o in ctx.unique_names(document.get("objects", []), "objects", "objeto")
        if o.name not in constant_names
    )
    predicates = {p.name: p for p in domain.predicates}
    scope = _Scope(predicates, frozenset(constant_names | {o.name for o in objects}))

    init: set[Atom] = set()
    for i, node in enumerate(document.get("init", [])):
        init.add(ctx.atom(node, f"init[{i}]", scope))
    goal = ctx.formula(document["goal"], "goal", scope)

    return Problem(
        name=document["name"].lower(),
        domain_name=domain_name,
        requirements=requirements,
        objects=objects,
        init=frozenset(init),
        goal=goal,
    )


class _Scope:
    """Lo que puede aparecer en una fórmula: predicados, objetos y variables ligadas."""

    def __init__(
        self,
        predicates: Mapping[str, Predicate],
        objects: frozenset[str],
        variables: frozenset[str] = frozenset(),
    ) -> None:
        self.predicates = predicates
        self.objects = objects
        self.variables = variables

    def with_variables(self, names: Iterable[str]) -> "_Scope":
        return _Scope(self.predicates, self.objects, self.variables | frozenset(names))


class _Context:
    """Convierte los nodos de un documento y genera errores con ruta y posición."""

    def __init__(self, document: JSON) -> None:
        self.source: str | None = document.get("source")

    def fail(
        self,
        message: str,
        node: JSON,
        path: str,
        error: type[SemanticError | UnsupportedFeatureError] = SemanticError,
    ) -> NoReturn:
        location = None
        loc = node.get("loc")
        if loc is not None:
            location = SourceLocation(loc["line"], loc["column"], self.source)
        raise error(message, location, path)

    def requirements(self, document: JSON) -> frozenset[str]:
        declared = [r.lower() for r in document.get("requirements", [])]
        for i, requirement in enumerate(declared):
            if requirement not in KNOWN_REQUIREMENTS:
                self.fail(f"requisito desconocido '{requirement}'", document, f"requirements[{i}]")
        unsupported = unsupported_requirements(declared)
        if unsupported:
            self.fail(
                f"requisitos no soportados: {', '.join(unsupported)}",
                document,
                "requirements",
                UnsupportedFeatureError,
            )
        return expand_requirements(declared)

    def unique_names(self, nodes: list[JSON], path: str, what: str) -> tuple[Constant, ...]:
        seen: dict[str, Constant] = {}
        for i, node in enumerate(nodes):
            name = node["name"].lower()
            if name in seen:
                self.fail(f"{what} '{name}' declarado dos veces", node, f"{path}[{i}]")
            seen[name] = Constant(name)
        return tuple(seen.values())

    def parameters(self, node: JSON, path: str) -> tuple[Variable, ...]:
        seen: dict[str, Variable] = {}
        for i, parameter in enumerate(node.get("parameters", [])):
            name = parameter["name"].lower()
            if name in seen:
                self.fail(f"parámetro '{name}' repetido", parameter, f"{path}.parameters[{i}]")
            seen[name] = Variable(name)
        return tuple(seen.values())

    def formula(self, node: JSON, path: str, scope: _Scope) -> Formula:
        if node["op"] == "and":
            return And(
                tuple(
                    self.formula(arg, f"{path}.args[{i}]", scope)
                    for i, arg in enumerate(node["args"])
                )
            )
        return self.atom(node, path, scope)

    def effect(self, node: JSON, path: str, scope: _Scope) -> Effect:
        if node["op"] == "and":
            return And(
                tuple(
                    self.effect(arg, f"{path}.args[{i}]", scope)
                    for i, arg in enumerate(node["args"])
                )
            )
        if node["op"] == "not":
            return Not(self.atom(node["arg"], f"{path}.arg", scope))
        return self.atom(node, path, scope)

    def atom(self, node: JSON, path: str, scope: _Scope) -> Atom:
        name = node["predicate"].lower()
        predicate = scope.predicates.get(name)
        if predicate is None:
            self.fail(f"predicado '{name}' no declarado", node, path)
        args = node["args"]
        if len(args) != predicate.arity:
            self.fail(
                f"'{name}' espera {predicate.arity} argumentos y recibe {len(args)}", node, path
            )
        return Atom(name, tuple(self.term(arg.lower(), node, path, scope) for arg in args))

    def term(self, name: str, node: JSON, path: str, scope: _Scope) -> Term:
        if name.startswith("?"):
            if name not in scope.variables:
                where = (
                    "no es un parámetro de la acción" if scope.variables else "no se permite aquí"
                )
                self.fail(f"la variable '{name}' {where}", node, path)
            return Variable(name)
        if name not in scope.objects:
            self.fail(f"objeto o constante '{name}' no declarado", node, path)
        return Constant(name)
