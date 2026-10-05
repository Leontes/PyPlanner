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
   * acciones durativas: nombres distintos de los de las acciones, ``?duration`` solo en
     los efectos numéricos de ellas (ni en sus condiciones ni en su propia duración, como
     en la gramática de PDDL 2.1 y en VAL), ``#t`` solo en efectos continuos con la forma
     ``(increase|decrease f (* #t e))``, y efectos ``at start`` que no dependen de
     condiciones posteriores (``at end`` u ``over all``);
   * objetos declarados y con un tipo compatible con el del predicado;
   * predicados derivados (iteración 4): declarados en ``:predicates``, con la aridad y
     tipos compatibles en la cabeza de cada regla, ausentes de los efectos y del estado
     inicial, y **estratificables**: ninguno depende negativamente (bajo un número impar
     de negaciones, contando el antecedente de ``imply``) de sí mismo, directa o
     indirectamente;
   * literales iniciales temporizados (iteración 4): solo en dominios temporales, sobre
     predicados no derivados y sin dar a un hecho dos valores en el mismo instante;
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
    Constraint,
    ConstraintAnd,
    ConstraintForall,
    ContinuousEffect,
    DerivedRule,
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
    IsViolated,
    Metric,
    Not,
    Number,
    NumericEffect,
    ObjectDecl,
    Or,
    Parameter,
    Predicate,
    Preference,
    Problem,
    Term,
    TimedCondition,
    TimedEffect,
    TimedLiteral,
    TotalTime,
    TrajectoryConstraint,
    Variable,
    When,
    derived_uses,
    preference_names,
)
from pyplanner.loader.object_fluents import (
    COMPILED_REQUIREMENTS,
    compile_domain,
    compile_problem,
    object_functions,
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
    _Names(ctx).domain(document)
    # Las funciones de objetos se compilan a predicados antes de construir el AST.
    fluents = object_functions(document)
    if fluents:
        document = compile_domain(document, fluents)
        requirements = requirements | expand_requirements(COMPILED_REQUIREMENTS)

    types = ctx.types(document.get("types", []))
    ctx.known_types = {OBJECT, *(name for name, _ in types)}
    constants = ctx.objects(document.get("constants", []), "constants", "constante")
    predicates: dict[str, Predicate] = {}
    for i, node in enumerate(document.get("predicates", [])):
        path = f"predicates[{i}]"
        name = node["name"].lower()
        predicates[name] = Predicate(name, ctx.parameters(node.get("parameters", []), path))
    functions: dict[str, Function] = {}
    for i, node in enumerate(document.get("functions", [])):
        path = f"functions[{i}]"
        name = node["name"].lower()
        functions[name] = Function(name, ctx.parameters(node.get("parameters", []), path))

    # Dominio provisional (sin acciones) para consultar la jerarquía de tipos al comprobar.
    ctx.domain = Domain(document["name"].lower(), requirements, types, constants, (), ())
    scope = _Scope(predicates, {c.name: c.types for c in constants}, functions=functions)
    derived = tuple(
        ctx.derived_rule(node, f"derived[{i}]", scope)
        for i, node in enumerate(document.get("derived", []))
    )
    ctx.derived = frozenset(rule.predicate for rule in derived)
    ctx.check_stratified(derived, document.get("derived", []))
    actions: dict[str, Action] = {}
    for i, node in enumerate(document.get("actions", [])):
        path = f"actions[{i}]"
        name = node["name"].lower()
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
        durative_actions[name] = ctx.durative_action(node, path, scope)
    constraints = None
    if "constraints" in document:
        constraints = ctx.constraint(document["constraints"], "constraints", scope)

    return Domain(
        name=document["name"].lower(),
        requirements=requirements,
        types=types,
        constants=constants,
        predicates=tuple(predicates.values()),
        actions=tuple(actions.values()),
        functions=tuple(functions.values()),
        durative_actions=tuple(durative_actions.values()),
        derived=derived,
        constraints=constraints,
        object_fluents=fluents,
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
    ctx.check_declared(document, requirements, frozenset(f.name for f in domain.object_fluents))
    _Names(ctx).problem(document, domain)
    if domain.object_fluents:
        document = compile_problem(document, domain.object_fluents)

    ctx.domain = domain
    ctx.derived = domain.derived_predicates
    ctx.known_types = {OBJECT, *(name for name, _ in domain.types)}
    constant_types = {c.name: c.types for c in domain.constants}
    objects = ctx.objects(document.get("objects", []), "objects", "objeto")
    predicates = {p.name: p for p in domain.predicates}
    functions = {f.name: f for f in domain.functions}
    scope = _Scope(
        predicates, {**constant_types, **{o.name: o.types for o in objects}}, functions=functions
    )

    init: set[Atom] = set()
    values: dict[FluentTerm, float] = {}
    timed: dict[tuple[float, Atom], TimedLiteral] = {}
    for i, node in enumerate(document.get("init", [])):
        path = f"init[{i}]"
        if node["op"] == "=":
            fluent = ctx.fluent(node["fluent"], f"{path}.fluent", scope)
            if fluent in values:
                ctx.fail(f"la función {fluent} tiene dos valores iniciales", node, path)
            values[fluent] = float(node["value"])
        elif node["op"] == "at":
            literal = ctx.timed_literal(node, path, scope)
            previous = timed.setdefault((literal.time, literal.atom), literal)
            if previous.positive != literal.positive:
                ctx.fail(
                    f"el hecho {literal.atom} se añade y se borra en el mismo instante "
                    f"({literal.time:g})",
                    node,
                    path,
                )
        else:
            init.add(ctx.basic_atom(node, path, scope, "el estado inicial"))
    goal = ctx.formula(document["goal"], "goal", scope)
    constraints = None
    if "constraints" in document:
        constraints = ctx.constraint(document["constraints"], "constraints", scope)
    metric = None
    if "metric" in document:
        node = document["metric"]
        ctx.preferences = (
            domain.preference_names
            | frozenset(preference_names(goal))
            | frozenset(preference_names(constraints))
        )
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
        timed_literals=tuple(sorted(timed.values(), key=lambda lit: (lit.time, str(lit.atom)))),
        constraints=constraints,
    )


class _Names:
    """Registro de todos los nombres declarados: un nombre solo se puede declarar una vez,
    sea donde sea (dominio, problema, tipo, constante, objeto, predicado, función o acción).

    Un mismo nombre en dos sitios confunde a otras herramientas (VAL se cae si una función
    se llama como el dominio) y casi siempre es un error. La única excepción son las
    preferencias: varias pueden compartir nombre (``is-violated`` las cuenta juntas), pero
    tampoco pueden llamarse como otra cosa. Las variables (``?x``) tienen su propio ámbito.
    """

    def __init__(self, ctx: "_Context") -> None:
        self.ctx = ctx
        #: nombre → (qué es, dónde se declaró, para el mensaje)
        self.seen: dict[str, tuple[str, str]] = {OBJECT: ("tipo predefinido", "")}

    def add(self, name: str, kind: str, node: JSON, path: str, *, shared: bool = False) -> None:
        name = name.lower()
        previous = self.seen.get(name)
        if previous is not None:
            if shared and previous[0] == kind:
                return
            self.ctx.fail(
                f"el nombre '{name}' ya se ha definido previamente como {previous[0]}{previous[1]}",
                node,
                path,
            )
        loc = node.get("loc")
        self.seen[name] = (kind, f" (línea {loc['line']})" if loc else "")

    def preferences(self, node: Any, path: str) -> None:
        """Los nombres de las preferencias de un objetivo, condición o restricción."""
        if isinstance(node, Mapping):
            if node.get("op") == "preference" and "name" in node:
                self.add(node["name"], "preferencia", node, path, shared=True)
            for key, value in node.items():
                self.preferences(value, f"{path}.{key}")
        elif isinstance(node, list):
            for i, item in enumerate(node):
                self.preferences(item, f"{path}[{i}]")

    def domain(self, document: JSON) -> None:
        self.add(document["name"], "nombre del dominio", document, "name")
        declared = {node["name"].lower() for node in document.get("types", [])}
        for i, node in enumerate(document.get("types", [])):
            if node["name"].lower() != OBJECT:
                self.add(node["name"], "tipo", node, f"types[{i}]")
        for i, node in enumerate(document.get("types", [])):
            parent = node.get("parent", OBJECT).lower()
            if parent not in declared and parent not in self.seen:
                self.add(parent, "tipo", node, f"types[{i}].parent")  # implícito
        for key, kind in (
            ("constants", "constante"),
            ("predicates", "predicado"),
            ("functions", "función"),
            ("actions", "acción"),
            ("durative_actions", "acción durativa"),
        ):
            for i, node in enumerate(document.get(key, [])):
                self.add(node["name"], kind, node, f"{key}[{i}]")
        for key in ("actions", "durative_actions"):
            for i, node in enumerate(document.get(key, [])):
                self.preferences(node, f"{key}[{i}]")

    def problem(self, document: JSON, domain: Domain) -> None:
        compiled = {f.predicate for f in domain.object_fluents}
        here = " en el dominio"
        self.seen[domain.name] = ("nombre del dominio", "")
        for name, _ in domain.types:
            self.seen[name] = ("tipo", here)
        for kind, names in (
            ("constante", [c.name for c in domain.constants]),
            ("predicado", [p.name for p in domain.predicates if p.name not in compiled]),
            (
                "función",
                [*(f.name for f in domain.functions), *(f.name for f in domain.object_fluents)],
            ),
            ("acción", [a.name for a in domain.actions]),
            ("acción durativa", [a.name for a in domain.durative_actions]),
            ("preferencia", sorted(domain.preference_names)),
        ):
            for name in names:
                self.seen[name] = (kind, here)
        self.add(document["name"], "nombre del problema", document, "name")
        for i, node in enumerate(document.get("objects", [])):
            self.add(node["name"], "objeto", node, f"objects[{i}]")
        self.preferences(document.get("goal"), "goal")
        self.preferences(document.get("constraints"), "constraints")


class _Scope:
    """Lo que puede aparecer en una fórmula: predicados, objetos y variables ligadas."""

    def __init__(
        self,
        predicates: Mapping[str, Predicate],
        objects: Mapping[str, tuple[str, ...]],
        variables: Mapping[str, tuple[str, ...]] | None = None,
        functions: Mapping[str, Function] | None = None,
        durative: bool = False,
        condition: bool = False,
    ) -> None:
        self.predicates = predicates
        self.objects = objects
        self.variables: Mapping[str, tuple[str, ...]] = variables or {}
        self.functions: Mapping[str, Function] = functions or {}
        #: ``True`` dentro de una acción durativa, donde se puede usar ``?duration``...
        self.durative = durative
        #: ...salvo en sus condiciones (``True`` dentro de ellas).
        self.condition = condition

    def bind(self, parameters: Iterable[Parameter], *, durative: bool | None = None) -> "_Scope":
        variables = dict(self.variables)
        variables.update((p.name, p.types) for p in parameters)
        inside = self.durative if durative is None else durative
        return _Scope(
            self.predicates, self.objects, variables, self.functions, inside, self.condition
        )

    def in_condition(self) -> "_Scope":
        """El mismo ámbito, dentro de una condición de una acción durativa."""
        return _Scope(
            self.predicates, self.objects, self.variables, self.functions, self.durative, True
        )


class _Context:
    """Convierte los nodos de un documento y genera errores con ruta y posición."""

    def __init__(self, document: JSON) -> None:
        self.source: str | None = document.get("source")
        self.known_types: set[str] = {OBJECT}
        self.domain: Domain | None = None
        #: Predicados derivados del dominio: no pueden aparecer en efectos ni en el estado
        #: inicial.
        self.derived: frozenset[str] = frozenset()
        #: Nombres de las preferencias, para comprobar los ``is-violated`` de la métrica.
        self.preferences: frozenset[str] = frozenset()

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

    def check_declared(
        self, document: JSON, declared: Iterable[str], objects: frozenset[str] | None = None
    ) -> None:
        use = first_undeclared(document, declared, objects)
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
        for node in nodes:
            name = node["name"].lower()
            if name == OBJECT:
                continue
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
        if op == "preference":  # el esquema solo la admite en la capa exterior
            return Preference(
                _preference_name(node), self.formula(node["arg"], f"{path}.arg", scope)
            )
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
            return Not(self.basic_atom(node["arg"], f"{path}.arg", scope, "un efecto"))
        if op == "forall":
            parameters, inner = self.quantified(node, path, scope)
            return ForallEffect(parameters, self.effect(node["arg"], f"{path}.arg", inner))
        if op == "when":
            condition = self.formula(node["condition"], f"{path}.condition", scope)
            return When(condition, self.effect(node["arg"], f"{path}.arg", scope))
        if op in NUMERIC_EFFECTS:
            value = node["value"]
            if isinstance(value, str) or value["op"] == "undefined":
                # Un objeto o undefined: solo vale para una función de objetos, y esas ya se
                # han compilado a predicados.
                self.fail(
                    f"la función '{node['fluent']['function']}' es numérica: su valor es un número",
                    node,
                    path,
                )
            return NumericEffect(
                op,
                self.fluent(node["fluent"], f"{path}.fluent", scope),
                self.expression(value, f"{path}.value", scope),
            )
        return self.basic_atom(node, path, scope, "un efecto")

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
        if op == "is-violated":
            name = node["name"].lower()
            if not metric:
                self.fail("'is-violated' solo puede aparecer en la métrica", node, path)
            if name not in self.preferences:
                self.fail(
                    f"la preferencia '{name}' no aparece en el dominio ni en el problema",
                    node,
                    path,
                )
            return IsViolated(name)
        if op == "duration":
            if not scope.durative:
                self.fail("'?duration' solo puede aparecer en una acción durativa", node, path)
            if scope.condition:
                self.fail(
                    "'?duration' no puede aparecer en una condición: PDDL 2.1 solo lo admite "
                    "en la duración y en los efectos numéricos",
                    node,
                    path,
                )
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

    # --- PDDL 2.2 (iteración 4) -----------------------------------------------------

    def derived_rule(self, node: JSON, path: str, scope: _Scope) -> DerivedRule:
        """Una regla ``:derived``: la cabeza es un predicado declarado, con la misma aridad
        y parámetros de tipos compatibles con los suyos."""
        assert self.domain is not None
        name = node["predicate"].lower()
        predicate = scope.predicates.get(name)
        if predicate is None:
            self.fail(
                f"predicado derivado '{name}' no declarado en :predicates",
                node,
                f"{path}.predicate",
            )
        parameters = self.parameters(node.get("parameters", []), path)
        if len(parameters) != predicate.arity:
            self.fail(
                f"'{name}' espera {predicate.arity} argumentos y la regla declara "
                f"{len(parameters)}",
                node,
                f"{path}.parameters",
            )
        for i, (own, declared) in enumerate(zip(parameters, predicate.parameters, strict=True)):
            if not self.domain.is_subtype(own.types, declared.types):
                self.fail(
                    f"el parámetro '{own.name}' ({' | '.join(own.types)}) no es del tipo "
                    f"{' | '.join(declared.types)} del argumento {i + 1} de '{name}'",
                    node,
                    f"{path}.parameters[{i}]",
                )
        condition = self.formula(node["condition"], f"{path}.condition", scope.bind(parameters))
        return DerivedRule(name, parameters, condition)

    def check_stratified(self, rules: tuple[DerivedRule, ...], nodes: list[JSON]) -> None:
        """Las reglas tienen que ser estratificables: ningún predicado derivado puede
        depender negativamente de sí mismo, ni directa ni indirectamente.

        Sin esa condición, el valor de los predicados derivados no está bien definido (por
        ejemplo, ``p`` si no ``p``).
        """
        edges: dict[str, set[str]] = {}
        negative: list[tuple[int, str, str]] = []
        for i, rule in enumerate(rules):
            for used, negated in derived_uses(rule.condition, self.derived):
                edges.setdefault(rule.predicate, set()).add(used)
                if negated:
                    negative.append((i, rule.predicate, used))
        for i, head, used in negative:
            if _reaches(edges, used, head):
                cycle = "" if used == head else f", que a su vez depende de '{head}'"
                self.fail(
                    "los predicados derivados no son estratificables: "
                    f"'{head}' depende negativamente de '{used}'{cycle}",
                    nodes[i],
                    f"derived[{i}].condition",
                )

    def basic_atom(self, node: JSON, path: str, scope: _Scope, where: str) -> Atom:
        """Un átomo de un predicado no derivado (en un efecto o en el estado inicial)."""
        atom = self.atom(node, path, scope)
        if atom.predicate in self.derived:
            self.fail(
                f"el predicado derivado '{atom.predicate}' no puede aparecer en {where}: su "
                "valor se deduce de sus reglas",
                node,
                path,
            )
        return atom

    def timed_literal(self, node: JSON, path: str, scope: _Scope) -> TimedLiteral:
        assert self.domain is not None
        if ":durative-actions" not in self.domain.requirements:
            self.fail(
                "los literales iniciales temporizados solo tienen sentido en un dominio "
                "temporal (:durative-actions)",
                node,
                path,
            )
        literal = node["literal"]
        where = "un literal inicial temporizado"
        if literal["op"] == "not":
            atom = self.basic_atom(literal["arg"], f"{path}.literal.arg", scope, where)
            return TimedLiteral(float(node["time"]), atom, positive=False)
        return TimedLiteral(
            float(node["time"]), self.basic_atom(literal, f"{path}.literal", scope, where)
        )

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
        if op == "preference":  # el esquema garantiza que envuelve una condición con momento
            arg = self.timed_condition(node["arg"], f"{path}.arg", scope)
            assert isinstance(arg, TimedCondition)
            return Preference(_preference_name(node), arg)
        time = _TIMES[op]
        return TimedCondition(time, self.formula(node["arg"], f"{path}.arg", scope.in_condition()))

    # --- Restricciones de trayectoria (PDDL 3) ---------------------------------------

    def constraint(self, node: JSON, path: str, scope: _Scope) -> Constraint:
        """Una restricción: el esquema ya garantiza la forma (sin anidar operadores modales
        ni preferencias, y sin preferencias en las del dominio)."""
        op = node["op"]
        if op == "and":
            return ConstraintAnd(
                tuple(
                    self.constraint(arg, f"{path}.args[{i}]", scope)
                    for i, arg in enumerate(node["args"])
                )
            )
        if op == "forall":
            parameters, inner = self.quantified(node, path, scope)
            return ConstraintForall(parameters, self.constraint(node["arg"], f"{path}.arg", inner))
        if op == "preference":
            arg = self.constraint(node["arg"], f"{path}.arg", scope)
            return Preference(_preference_name(node), arg)
        formulas: tuple[Formula, ...]
        if "arg" in node:
            formulas = (self.formula(node["arg"], f"{path}.arg", scope),)
        else:
            formulas = tuple(
                self.formula(arg, f"{path}.args[{i}]", scope) for i, arg in enumerate(node["args"])
            )
        if op == "hold-during":
            if node["from"] > node["to"]:
                self.fail("'hold-during' necesita un intervalo con inicio ≤ fin", node, path)
            times: tuple[float, ...] = (float(node["from"]), float(node["to"]))
        elif "time" in node:
            times = (float(node["time"]),)
        else:
            times = ()
        return TrajectoryConstraint(op, formulas, times)

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


def _reaches(edges: Mapping[str, set[str]], start: str, goal: str) -> bool:
    """``True`` si hay un camino (quizá vacío) de ``start`` a ``goal``."""
    seen, pending = {start}, [start]
    while pending:
        current = pending.pop()
        if current == goal:
            return True
        for following in edges.get(current, ()):
            if following not in seen:
                seen.add(following)
                pending.append(following)
    return False


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
    if isinstance(condition, Preference):
        assert isinstance(condition.arg, TimedCondition)
        return {condition.arg.time}
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


def _preference_name(node: JSON) -> str | None:
    """Nombre de una preferencia, en minúsculas como el resto de nombres (``None`` si no
    tiene)."""
    name = node.get("name")
    return None if name is None else str(name).lower()
