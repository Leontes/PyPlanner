"""Fase 2 del parser: documento JSON → AST.

Es el punto de entrada del planificador cuando se integra en otro sistema, así que **no se
fía de su entrada**: el JSON puede venir de la fase 1 o de cualquier otro componente. Por
eso hace cuatro comprobaciones, en este orden:

1. **Estructura**: el documento cumple el esquema (:func:`pyplanner.schema.validate_document`).
2. **Requisitos**: todos los requisitos declarados existen y están soportados.
3. **Modo estricto**: toda construcción usada tiene su requisito declarado (en el problema
   cuentan los del dominio y los del problema).
4. **Semántica**:
   * nombres sin repetir;
   * tipos declarados y jerarquía sin ciclos;
   * predicados declarados y con la aridad correcta;
   * variables ligadas a un parámetro o a un cuantificador, sin ocultar otra variable;
   * funciones declaradas y con la aridad correcta; valores iniciales sin variables y sin
     repetir; ``total-time`` solo en la métrica;
   * acciones durativas: nombres distintos de los de las acciones, ``?duration`` solo dentro
     de ellas (y no en su propia duración), ``#t`` solo en efectos continuos con la forma
     ``(increase|decrease f (* #t e))``, y efectos ``at start`` que no dependen de
     condiciones posteriores (``at end`` u ``over all``);
   * objetos declarados y con un tipo compatible con el del predicado;
   * problema asociado a su dominio.

Los errores indican la ruta dentro del JSON y, si el documento trae posiciones (``loc``),
la línea y la columna del PDDL original.
"""

from collections.abc import Iterable, Mapping
from typing import Any, NoReturn

from pyplanner.errors import (
    PDDLError,
    SemanticError,
    SourceLocation,
    UndeclaredRequirementError,
    UnsupportedFeatureError,
)
from pyplanner.loader.ast import (
    ALL,
    END,
    OBJECT,
    START,
    Action,
    And,
    Arithmetic,
    Atom,
    Comparison,
    Constant,
    ContinuousEffect,
    Domain,
    DurationConstraint,
    DurationVar,
    DurativeAction,
    DurativeAnd,
    DurativeCondition,
    DurativeEffect,
    DurativeForall,
    DurativeForallEffect,
    DurativeWhen,
    Effect,
    Equals,
    Exists,
    Expression,
    FluentTerm,
    Forall,
    ForallEffect,
    Formula,
    Function,
    Imply,
    Metric,
    Not,
    Number,
    NumericEffect,
    ObjectDecl,
    Or,
    Parameter,
    Predicate,
    Problem,
    Term,
    TimedCondition,
    TimedEffect,
    TotalTime,
    Variable,
    When,
)
from pyplanner.requirements import (
    KNOWN_REQUIREMENTS,
    NUMERIC_EFFECTS,
    expand_requirements,
    first_undeclared,
    unknown_requirement_message,
    unsupported_requirements,
)
from pyplanner.schema import validate_document

JSON = Mapping[str, Any]


def load_domain(document: JSON) -> Domain:
    """Convierte un documento JSON de dominio en un :class:`~pyplanner.loader.ast.Domain`.

    Raises:
        SchemaValidationError: si el documento no cumple el esquema.
        UnsupportedFeatureError: si declara requisitos no soportados.
        UndeclaredRequirementError: si usa una construcción sin declarar su requisito.
        SemanticError: si el dominio no es coherente.
    """
    validate_document(document)
    ctx = _Context(document)
    if document["kind"] != "domain":
        ctx.fail("se esperaba un documento de dominio", document, "kind")
    requirements = ctx.requirements(document)
    ctx.check_declared(document, document.get("requirements", []))

    types = ctx.types(document.get("types", []))
    ctx.known_types = {OBJECT, *(name for name, _ in types)}
    constants = ctx.objects(document.get("constants", []), "constants", "constante")
    predicates: dict[str, Predicate] = {}
    for i, node in enumerate(document.get("predicates", [])):
        path = f"predicates[{i}]"
        name = node["name"].lower()
        if name in predicates:
            ctx.fail(f"predicado '{name}' declarado dos veces", node, path)
        predicates[name] = Predicate(name, ctx.parameters(node.get("parameters", []), path))
    functions: dict[str, Function] = {}
    for i, node in enumerate(document.get("functions", [])):
        path = f"functions[{i}]"
        name = node["name"].lower()
        if name in functions:
            ctx.fail(f"función '{name}' declarada dos veces", node, path)
        functions[name] = Function(name, ctx.parameters(node.get("parameters", []), path))

    # Dominio provisional (sin acciones) para consultar la jerarquía de tipos al comprobar.
    ctx.domain = Domain(document["name"].lower(), requirements, types, constants, (), ())
    scope = _Scope(predicates, {c.name: c.types for c in constants}, functions=functions)
    actions: dict[str, Action] = {}
    for i, node in enumerate(document.get("actions", [])):
        path = f"actions[{i}]"
        name = node["name"].lower()
        if name in actions:
            ctx.fail(f"acción '{name}' declarada dos veces", node, path)
        parameters = ctx.parameters(node.get("parameters", []), path)
        action_scope = scope.bind(parameters)
        precondition: Formula = And()
        effect: Effect = And()
        if "precondition" in node:
            precondition = ctx.formula(node["precondition"], f"{path}.precondition", action_scope)
        if "effect" in node:
            effect = ctx.effect(node["effect"], f"{path}.effect", action_scope)
        actions[name] = Action(name, parameters, precondition, effect)
    durative_actions: dict[str, DurativeAction] = {}
    for i, node in enumerate(document.get("durative_actions", [])):
        path = f"durative_actions[{i}]"
        name = node["name"].lower()
        if name in actions or name in durative_actions:
            ctx.fail(f"acción '{name}' declarada dos veces", node, path)
        durative_actions[name] = ctx.durative_action(node, path, scope)

    return Domain(
        name=document["name"].lower(),
        requirements=requirements,
        types=types,
        constants=constants,
        predicates=tuple(predicates.values()),
        actions=tuple(actions.values()),
        functions=tuple(functions.values()),
        durative_actions=tuple(durative_actions.values()),
    )


def load_problem(document: JSON, domain: Domain) -> Problem:
    """Convierte un documento JSON de problema en un :class:`~pyplanner.loader.ast.Problem`.

    El problema se comprueba contra ``domain``: tiene que referirse a él y usar solo sus
    predicados y tipos.

    Raises:
        SchemaValidationError: si el documento no cumple el esquema.
        UnsupportedFeatureError: si declara requisitos no soportados.
        UndeclaredRequirementError: si usa una construcción sin declarar su requisito.
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
    ctx.check_declared(document, requirements)

    ctx.domain = domain
    ctx.known_types = {OBJECT, *(name for name, _ in domain.types)}
    constant_types = {c.name: c.types for c in domain.constants}
    objects = tuple(
        o
        for o in ctx.objects(document.get("objects", []), "objects", "objeto")
        if o.name not in constant_types
    )
    predicates = {p.name: p for p in domain.predicates}
    functions = {f.name: f for f in domain.functions}
    scope = _Scope(
        predicates, {**constant_types, **{o.name: o.types for o in objects}}, functions=functions
    )

    init: set[Atom] = set()
    values: dict[FluentTerm, float] = {}
    for i, node in enumerate(document.get("init", [])):
        path = f"init[{i}]"
        if node["op"] == "=":
            fluent = ctx.fluent(node["fluent"], f"{path}.fluent", scope)
            if fluent in values:
                ctx.fail(f"la función {fluent} tiene dos valores iniciales", node, path)
            values[fluent] = float(node["value"])
        else:
            init.add(ctx.atom(node, path, scope))
    goal = ctx.formula(document["goal"], "goal", scope)
    metric = None
    if "metric" in document:
        node = document["metric"]
        expression = ctx.expression(node["expression"], "metric.expression", scope, metric=True)
        metric = Metric(node["direction"], expression)

    return Problem(
        name=document["name"].lower(),
        domain_name=domain_name,
        requirements=requirements,
        objects=objects,
        init=frozenset(init),
        goal=goal,
        values=tuple(sorted(values.items(), key=lambda item: str(item[0]))),
        metric=metric,
    )


class _Scope:
    """Lo que puede aparecer en una fórmula: predicados, objetos y variables ligadas."""

    def __init__(
        self,
        predicates: Mapping[str, Predicate],
        objects: Mapping[str, tuple[str, ...]],
        variables: Mapping[str, tuple[str, ...]] | None = None,
        functions: Mapping[str, Function] | None = None,
        durative: bool = False,
    ) -> None:
        self.predicates = predicates
        self.objects = objects
        self.variables: Mapping[str, tuple[str, ...]] = variables or {}
        self.functions: Mapping[str, Function] = functions or {}
        #: ``True`` dentro de una acción durativa, donde se puede usar ``?duration``.
        self.durative = durative

    def bind(self, parameters: Iterable[Parameter], *, durative: bool | None = None) -> "_Scope":
        variables = dict(self.variables)
        variables.update((p.name, p.types) for p in parameters)
        inside = self.durative if durative is None else durative
        return _Scope(self.predicates, self.objects, variables, self.functions, inside)


class _Context:
    """Convierte los nodos de un documento y genera errores con ruta y posición."""

    def __init__(self, document: JSON) -> None:
        self.source: str | None = document.get("source")
        self.known_types: set[str] = {OBJECT}
        self.domain: Domain | None = None

    def fail(
        self,
        message: str,
        node: JSON | None,
        path: str,
        error: type[PDDLError] = SemanticError,
    ) -> NoReturn:
        location = None
        loc = node.get("loc") if node is not None else None
        if loc is not None:
            location = SourceLocation(loc["line"], loc["column"], self.source)
        raise error(message, location, path)

    # --- Requisitos -----------------------------------------------------------------

    def requirements(self, document: JSON) -> frozenset[str]:
        declared = [r.lower() for r in document.get("requirements", [])]
        for i, requirement in enumerate(declared):
            if requirement not in KNOWN_REQUIREMENTS:
                self.fail(unknown_requirement_message(requirement), document, f"requirements[{i}]")
        unsupported = unsupported_requirements(declared)
        if unsupported:
            self.fail(
                f"requisitos no soportados: {', '.join(unsupported)}",
                document,
                "requirements",
                UnsupportedFeatureError,
            )
        return expand_requirements(declared)

    def check_declared(self, document: JSON, declared: Iterable[str]) -> None:
        use = first_undeclared(document, declared)
        if use is not None:
            self.fail(
                f"{use.construct} necesita declarar el requisito {use.requirement}",
                {"loc": use.loc} if use.loc is not None else None,
                use.path,
                UndeclaredRequirementError,
            )

    # --- Declaraciones --------------------------------------------------------------

    def types(self, nodes: list[JSON]) -> tuple[tuple[str, str], ...]:
        """Jerarquía de tipos. Un supertipo no declarado se declara implícitamente."""
        parents: dict[str, str] = {}
        for i, node in enumerate(nodes):
            name = node["name"].lower()
            if name == OBJECT:
                continue
            if name in parents:
                self.fail(f"tipo '{name}' declarado dos veces", node, f"types[{i}]")
            parents[name] = node.get("parent", OBJECT).lower()
        for parent in list(parents.values()):
            if parent != OBJECT and parent not in parents:
                parents[parent] = OBJECT
        for i, node in enumerate(nodes):
            seen = {node["name"].lower()}
            current = parents.get(node["name"].lower(), OBJECT)
            while current != OBJECT:
                if current in seen:
                    self.fail(
                        f"ciclo en la jerarquía de tipos: {' - '.join(seen)}", node, f"types[{i}]"
                    )
                seen.add(current)
                current = parents[current]
        return tuple(sorted(parents.items()))

    def type_spec(self, node: JSON, path: str) -> tuple[str, ...]:
        spec = node.get("type", OBJECT)
        names = [spec] if isinstance(spec, str) else list(spec)
        types = tuple(dict.fromkeys(t.lower() for t in names))
        for type_name in types:
            if type_name not in self.known_types:
                self.fail(f"tipo '{type_name}' no declarado", node, path)
        return types

    def objects(self, nodes: list[JSON], path: str, what: str) -> tuple[ObjectDecl, ...]:
        seen: dict[str, ObjectDecl] = {}
        for i, node in enumerate(nodes):
            name = node["name"].lower()
            if name in seen:
                self.fail(f"{what} '{name}' declarado dos veces", node, f"{path}[{i}]")
            seen[name] = ObjectDecl(name, self.type_spec(node, f"{path}[{i}]"))
        return tuple(seen.values())

    def parameters(self, nodes: list[JSON], path: str) -> tuple[Parameter, ...]:
        seen: dict[str, Parameter] = {}
        for i, parameter in enumerate(nodes):
            name = parameter["name"].lower()
            if name in seen:
                self.fail(f"parámetro '{name}' repetido", parameter, f"{path}.parameters[{i}]")
            seen[name] = Parameter(name, self.type_spec(parameter, f"{path}.parameters[{i}]"))
        return tuple(seen.values())

    # --- Fórmulas y efectos ---------------------------------------------------------

    def formula(self, node: JSON, path: str, scope: _Scope) -> Formula:
        op = node["op"]
        if op in ("and", "or"):
            args = tuple(
                self.formula(arg, f"{path}.args[{i}]", scope) for i, arg in enumerate(node["args"])
            )
            return And(args) if op == "and" else Or(args)
        if op == "not":
            return Not(self.formula(node["arg"], f"{path}.arg", scope))
        if op == "imply":
            left, right = node["args"]
            return Imply(
                self.formula(left, f"{path}.args[0]", scope),
                self.formula(right, f"{path}.args[1]", scope),
            )
        if op in ("exists", "forall"):
            parameters, inner = self.quantified(node, path, scope)
            body = self.formula(node["arg"], f"{path}.arg", inner)
            return Exists(parameters, body) if op == "exists" else Forall(parameters, body)
        if op == "=":
            left, right = (self.term(t.lower(), node, path, scope) for t in node["args"])
            return Equals(left, right)
        if op == "compare":
            return Comparison(
                node["comparator"],
                self.expression(node["left"], f"{path}.left", scope),
                self.expression(node["right"], f"{path}.right", scope),
            )
        return self.atom(node, path, scope)

    def effect(self, node: JSON, path: str, scope: _Scope) -> Effect:
        op = node["op"]
        if op == "and":
            return And(
                tuple(
                    self.effect(arg, f"{path}.args[{i}]", scope)
                    for i, arg in enumerate(node["args"])
                )
            )
        if op == "not":
            return Not(self.atom(node["arg"], f"{path}.arg", scope))
        if op == "forall":
            parameters, inner = self.quantified(node, path, scope)
            return ForallEffect(parameters, self.effect(node["arg"], f"{path}.arg", inner))
        if op == "when":
            condition = self.formula(node["condition"], f"{path}.condition", scope)
            return When(condition, self.effect(node["arg"], f"{path}.arg", scope))
        if op in NUMERIC_EFFECTS:
            return NumericEffect(
                op,
                self.fluent(node["fluent"], f"{path}.fluent", scope),
                self.expression(node["value"], f"{path}.value", scope),
            )
        return self.atom(node, path, scope)

    def expression(
        self, node: JSON, path: str, scope: _Scope, *, metric: bool = False
    ) -> Expression:
        """Expresión numérica. ``total-time`` solo se admite en la métrica (``metric``)."""
        op = node["op"]
        if op == "num":
            return Number(float(node["value"]))
        if op == "fluent":
            return self.fluent(node, path, scope)
        if op == "total-time":
            if not metric:
                self.fail("'total-time' solo puede aparecer en la métrica", node, path)
            return TotalTime()
        if op == "duration":
            if not scope.durative:
                self.fail("'?duration' solo puede aparecer en una acción durativa", node, path)
            return DurationVar()
        if op == "#t":
            self.fail(
                "'#t' solo puede aparecer en un efecto continuo: "
                "(increase|decrease <función> (* #t <expresión>))",
                node,
                path,
            )
        args = tuple(
            self.expression(arg, f"{path}.args[{i}]", scope, metric=metric)
            for i, arg in enumerate(node["args"])
        )
        return Arithmetic(op, args)

    def fluent(self, node: JSON, path: str, scope: _Scope) -> FluentTerm:
        name = node["function"].lower()
        function = scope.functions.get(name)
        if function is None:
            self.fail(f"función '{name}' no declarada", node, path)
        args = node["args"]
        if len(args) != function.arity:
            self.fail(
                f"la función '{name}' espera {function.arity} argumentos y recibe {len(args)}",
                node,
                path,
            )
        terms = tuple(self.term(arg.lower(), node, path, scope) for arg in args)
        self.check_argument_types(name, terms, function.parameters, scope, node, path)
        return FluentTerm(name, terms)

    # --- Acciones durativas (iteración 3b) -------------------------------------------

    def durative_action(self, node: JSON, path: str, scope: _Scope) -> DurativeAction:
        parameters = self.parameters(node.get("parameters", []), path)
        inner = scope.bind(parameters, durative=True)
        duration = tuple(
            self.duration_constraint(c, f"{path}.duration[{i}]", inner)
            for i, c in enumerate(node["duration"])
        )
        condition: DurativeCondition = DurativeAnd()
        effect: DurativeEffect = DurativeAnd()
        if "condition" in node:
            condition = self.timed_condition(node["condition"], f"{path}.condition", inner)
        if "effect" in node:
            effect = self.timed_effect(node["effect"], f"{path}.effect", inner)
        return DurativeAction(node["name"].lower(), parameters, duration, condition, effect)

    def duration_constraint(self, node: JSON, path: str, scope: _Scope) -> DurationConstraint:
        value = self.expression(node["value"], f"{path}.value", scope)
        if _uses_duration(value):
            self.fail("la duración no puede depender de '?duration'", node, f"{path}.value")
        return DurationConstraint(node["comparator"], value, node.get("time"))

    def timed_condition(self, node: JSON, path: str, scope: _Scope) -> DurativeCondition:
        op = node["op"]
        if op == "and":
            return DurativeAnd(
                tuple(
                    self.timed_condition(arg, f"{path}.args[{i}]", scope)
                    for i, arg in enumerate(node["args"])
                )
            )
        if op == "forall":
            parameters, inner = self.quantified(node, path, scope)
            return DurativeForall(
                parameters, self.timed_condition(node["arg"], f"{path}.arg", inner)
            )
        time = _TIMES[op]
        return TimedCondition(time, self.formula(node["arg"], f"{path}.arg", scope))

    def timed_effect(self, node: JSON, path: str, scope: _Scope) -> DurativeEffect:
        op = node["op"]
        if op == "and":
            return DurativeAnd(
                tuple(
                    self.timed_effect(arg, f"{path}.args[{i}]", scope)
                    for i, arg in enumerate(node["args"])
                )
            )
        if op == "forall":
            parameters, inner = self.quantified(node, path, scope)
            return DurativeForallEffect(
                parameters, self.timed_effect(node["arg"], f"{path}.arg", inner)
            )
        if op == "when":
            condition = self.timed_condition(node["condition"], f"{path}.condition", scope)
            arg = self.timed_effect(node["arg"], f"{path}.arg", scope)
            if START in _effect_times(arg) and _condition_times(condition) - {START}:
                self.fail(
                    "un efecto 'at start' solo puede depender de condiciones 'at start'",
                    node,
                    path,
                )
            return DurativeWhen(condition, arg)
        if op in ("at-start", "at-end"):
            return TimedEffect(_TIMES[op], self.effect(node["arg"], f"{path}.arg", scope))
        return self.continuous_effect(node, path, scope)

    def continuous_effect(self, node: JSON, path: str, scope: _Scope) -> ContinuousEffect:
        """``(increase|decrease f #t)`` o ``(increase|decrease f (* ... #t ...))``: el ritmo es
        1 o el producto del resto de factores."""
        value = node["value"]
        form = "(increase|decrease <función> (* #t <expresión>))"
        if value["op"] == "#t":
            rate: Expression = Number(1.0)
        elif value["op"] == "*" and sum(arg["op"] == "#t" for arg in value["args"]) == 1:
            factors = tuple(
                self.expression(arg, f"{path}.value.args[{i}]", scope)
                for i, arg in enumerate(value["args"])
                if arg["op"] != "#t"
            )
            rate = factors[0] if len(factors) == 1 else Arithmetic("*", factors)
        else:
            self.fail(f"un efecto continuo tiene la forma {form}", node, f"{path}.value")
        fluent = self.fluent(node["fluent"], f"{path}.fluent", scope)
        return ContinuousEffect(node["op"], fluent, rate)

    def quantified(
        self, node: JSON, path: str, scope: _Scope
    ) -> tuple[tuple[Parameter, ...], _Scope]:
        parameters = self.parameters(node["parameters"], path)
        for i, parameter in enumerate(parameters):
            if parameter.name in scope.variables:
                self.fail(
                    f"la variable '{parameter.name}' ya está ligada fuera del cuantificador",
                    node["parameters"][i],
                    f"{path}.parameters[{i}]",
                )
        return parameters, scope.bind(parameters)

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
        terms = tuple(self.term(arg.lower(), node, path, scope) for arg in args)
        self.check_argument_types(name, terms, predicate.parameters, scope, node, path)
        return Atom(name, terms)

    def check_argument_types(
        self,
        name: str,
        terms: tuple[Term, ...],
        parameters: tuple[Parameter, ...],
        scope: _Scope,
        node: JSON,
        path: str,
    ) -> None:
        """Las constantes de ``terms`` tienen que ser del tipo de los ``parameters``."""
        assert self.domain is not None
        for position, (term, parameter) in enumerate(zip(terms, parameters, strict=True)):
            if isinstance(term, Constant):
                own = scope.objects[term.name]
                if not self.domain.is_subtype(own, parameter.types):
                    self.fail(
                        f"'{term.name}' ({' | '.join(own)}) no es del tipo "
                        f"{' | '.join(parameter.types)} que espera el argumento "
                        f"{position + 1} de '{name}'",
                        node,
                        path,
                    )

    def term(self, name: str, node: JSON, path: str, scope: _Scope) -> Term:
        if name.startswith("?"):
            if name not in scope.variables:
                self.fail(
                    f"la variable '{name}' no está ligada (no es un parámetro ni una variable "
                    "cuantificada)",
                    node,
                    path,
                )
            return Variable(name)
        if name not in scope.objects:
            self.fail(f"objeto o constante '{name}' no declarado", node, path)
        return Constant(name)


#: Momento de cada operador temporal del JSON.
_TIMES = {"at-start": START, "at-end": END, "over-all": ALL}


def _uses_duration(expression: Expression) -> bool:
    if isinstance(expression, DurationVar):
        return True
    if isinstance(expression, Arithmetic):
        return any(_uses_duration(arg) for arg in expression.args)
    return False


def _condition_times(condition: DurativeCondition) -> set[str]:
    """Momentos en que se comprueba alguna parte de la condición."""
    if isinstance(condition, TimedCondition):
        return {condition.time}
    if isinstance(condition, DurativeForall):
        return _condition_times(condition.arg)
    return set().union(*(_condition_times(arg) for arg in condition.args))  # type: ignore[arg-type]


def _effect_times(effect: DurativeEffect) -> set[str]:
    """Momentos en que actúa alguna parte del efecto (``all`` para los continuos)."""
    if isinstance(effect, TimedEffect):
        return {effect.time}
    if isinstance(effect, ContinuousEffect):
        return {ALL}
    if isinstance(effect, DurativeForallEffect | DurativeWhen):
        return _effect_times(effect.arg)
    return set().union(*(_effect_times(arg) for arg in effect.args))  # type: ignore[arg-type]
