"""Fase 1 del parser: PDDL → documento JSON.

Esta fase solo comprueba la **sintaxis**. Recorre el árbol de s-expresiones y genera un
documento JSON que sigue el esquema de :mod:`pyplanner.schema`, conservando la estructura del
PDDL original (no simplifica conjunciones ni normaliza fórmulas). Las comprobaciones de
significado (predicados declarados, aridades, variables libres…) son cosa de la fase 2
(:mod:`pyplanner.loader`).

Hay dos clases de error:

* :class:`~pyplanner.errors.PDDLSyntaxError` si el texto no es PDDL bien formado.
* :class:`~pyplanner.errors.UnsupportedFeatureError` si es PDDL válido pero usa algo que el
  planificador todavía no soporta (por ejemplo, ``:functions`` o acciones durativas).
* :class:`~pyplanner.errors.UndeclaredRequirementError` si un **dominio** usa una
  construcción sin declarar su requisito (modo estricto, ver :mod:`pyplanner.requirements`).
  Los problemas se comprueban en la fase 2, porque sus requisitos dependen del dominio.
"""

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn

from pyplanner.errors import (
    PDDLSyntaxError,
    SourceLocation,
    UndeclaredRequirementError,
    UnsupportedFeatureError,
)
from pyplanner.pddl.sexpr import SExpr, SList, Symbol, parse_sexprs
from pyplanner.requirements import (
    KNOWN_REQUIREMENTS,
    first_undeclared,
    unsupported_requirements,
)
from pyplanner.schema import SCHEMA_VERSION

JSONDict = dict[str, Any]

_NAME = re.compile(r"^[a-z][a-z0-9_-]*$")
_VARIABLE = re.compile(r"^\?[a-z][a-z0-9_-]*$")
_NUMBER = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)$")

# Secciones y operadores de versiones posteriores de PDDL, con el requisito que los introduce.
# Sirven para dar un error de "no soportado" en lugar de uno de sintaxis.
_FUTURE_DOMAIN_SECTIONS = {
    ":durative-action": ":durative-actions",
    ":derived": ":derived-predicates",
    ":process": ":time",
    ":event": ":time",
    ":constraints": ":constraints",
}
_FUTURE_PROBLEM_SECTIONS = {
    ":constraints": ":constraints",
}
#: Operadores lógicos: no pueden usarse como nombre de predicado dentro de un borrado.
_OPERATORS = frozenset({"and", "or", "not", "imply", "exists", "forall", "=", "when"})

_FUTURE_FORMULA_OPS = {
    "preference": ":preferences",
}

#: Comparadores numéricos.
_COMPARATORS = ("<", "<=", "=", ">=", ">")
#: Operadores aritméticos.
_ARITHMETIC = ("+", "-", "*", "/")
#: Efectos numéricos.
_NUMERIC_EFFECTS = ("increase", "decrease", "assign", "scale-up", "scale-down")
#: Símbolos de expresiones temporales (iteración 3b).
_FUTURE_EXPRESSION_SYMBOLS = {
    "#t": ":continuous-effects",
    "?duration": ":durative-actions",
}
#: Duración total del plan: solo puede aparecer en la métrica. En un plan secuencial, cada
#: acción dura una unidad, así que es el número de acciones.
_TOTAL_TIME = "total-time"


def parse_pddl(text: str, filename: str | None = None, *, locations: bool = True) -> JSONDict:
    """Convierte un dominio o un problema PDDL en su documento JSON.

    Args:
        text: contenido PDDL.
        filename: nombre del fichero; se guarda en ``source`` y aparece en los errores.
        locations: si es ``True``, cada nodo lleva su posición (``loc``) en el PDDL.

    Returns:
        Un documento con ``"kind": "domain"`` o ``"kind": "problem"``.

    Raises:
        PDDLSyntaxError: si el texto no es PDDL bien formado.
        UnsupportedFeatureError: si usa funcionalidad todavía no soportada.
        UndeclaredRequirementError: si un dominio usa una construcción sin declarar su
            requisito.
    """
    document = _Parser(filename, locations).parse(text)
    if document["kind"] == "domain":
        _check_declared(document, filename)
    return document


def _check_declared(document: JSONDict, filename: str | None) -> None:
    use = first_undeclared(document, document["requirements"])
    if use is not None:
        location = None
        if use.loc is not None:
            location = SourceLocation(use.loc["line"], use.loc["column"], filename)
        raise UndeclaredRequirementError(
            f"{use.construct} necesita declarar el requisito {use.requirement}",
            location,
            use.path,
        )


def parse_domain(text: str, filename: str | None = None, *, locations: bool = True) -> JSONDict:
    """Igual que :func:`parse_pddl`, pero exige que el texto sea un dominio."""
    return _expect_kind(parse_pddl(text, filename, locations=locations), "domain", filename)


def parse_problem(text: str, filename: str | None = None, *, locations: bool = True) -> JSONDict:
    """Igual que :func:`parse_pddl`, pero exige que el texto sea un problema."""
    return _expect_kind(parse_pddl(text, filename, locations=locations), "problem", filename)


def parse_file(path: str | Path, *, locations: bool = True) -> JSONDict:
    """Lee un fichero PDDL (dominio o problema) y devuelve su documento JSON."""
    path = Path(path)
    return parse_pddl(path.read_text(encoding="utf-8"), str(path), locations=locations)


def _expect_kind(document: JSONDict, kind: str, filename: str | None) -> JSONDict:
    if document["kind"] != kind:
        where = f"{filename}: " if filename else ""
        raise PDDLSyntaxError(f"{where}se esperaba un {_KIND_NAMES[kind]}")
    return document


_KIND_NAMES = {"domain": "dominio", "problem": "problema"}


class _Parser:
    """Recorre el árbol de s-expresiones de un fichero y construye el documento JSON."""

    def __init__(self, filename: str | None, locations: bool) -> None:
        self.filename = filename
        self.locations = locations

    # --- Entrada -----------------------------------------------------------------------

    def parse(self, text: str) -> JSONDict:
        exprs = parse_sexprs(text, self.filename)
        if not exprs:
            raise PDDLSyntaxError(f"{self.filename or 'la entrada'} está vacío")
        if len(exprs) > 1:
            raise PDDLSyntaxError(
                "se esperaba una sola expresión (define ...) por fichero", exprs[1].location
            )
        define = self._list(exprs[0], "(define ...)")
        if not define.items or not self._is_symbol(define.items[0], "define"):
            raise PDDLSyntaxError("se esperaba (define ...)", define.location)
        if len(define.items) < 2:
            raise PDDLSyntaxError("falta (domain <nombre>) o (problem <nombre>)", define.location)
        header = self._list(define.items[1], "(domain <nombre>) o (problem <nombre>)")
        if len(header.items) != 2:
            raise PDDLSyntaxError(
                "se esperaba (domain <nombre>) o (problem <nombre>)", header.location
            )
        kind = self._symbol(header.items[0], "'domain' o 'problem'")
        name = self._name(header.items[1], "el nombre")
        sections = [self._section(item) for item in define.items[2:]]
        if kind.value == "domain":
            return self._domain(name, sections)
        if kind.value == "problem":
            return self._problem(name, sections)
        raise PDDLSyntaxError(f"se esperaba 'domain' o 'problem', no '{kind.value}'", kind.location)

    def _header(self, kind: str, name: str) -> JSONDict:
        document: JSONDict = {"version": SCHEMA_VERSION, "kind": kind, "name": name}
        if self.filename is not None:
            document["source"] = self.filename
        return document

    # --- Dominio -----------------------------------------------------------------------

    def _domain(self, name: str, sections: list[tuple[Symbol, SList]]) -> JSONDict:
        document = self._header("domain", name)
        document.update(
            requirements=[], types=[], constants=[], predicates=[], functions=[], actions=[]
        )
        seen: set[str] = set()
        for keyword, section in sections:
            key = keyword.value
            if key != ":action":
                self._check_unique_section(key, keyword, seen)
            body = section.items[1:]
            if key == ":requirements":
                document["requirements"] = self._requirements(body)
            elif key == ":types":
                document["types"] = self._types(body)
            elif key == ":constants":
                document["constants"] = self._objects(body)
            elif key == ":predicates":
                document["predicates"] = [self._predicate(item) for item in body]
            elif key == ":functions":
                document["functions"] = self._functions(body)
            elif key == ":action":
                document["actions"].append(self._action(section))
            else:
                self._unknown_section(keyword, _FUTURE_DOMAIN_SECTIONS, "dominio")
        return document

    def _predicate(self, expr: SExpr) -> JSONDict:
        predicate = self._list(expr, "una declaración de predicado (nombre ?x ...)")
        if not predicate.items:
            raise PDDLSyntaxError("declaración de predicado vacía", predicate.location)
        node: JSONDict = {
            "name": self._name(predicate.items[0], "el nombre del predicado"),
            "parameters": self._parameters(predicate.items[1:]),
        }
        return self._with_loc(node, predicate)

    def _action(self, section: SList) -> JSONDict:
        if len(section.items) < 2:
            raise PDDLSyntaxError("falta el nombre de la acción", section.location)
        node: JSONDict = {
            "name": self._name(section.items[1], "el nombre de la acción"),
            "parameters": [],
        }
        fields = section.items[2:]
        seen: set[str] = set()
        for i in range(0, len(fields), 2):
            keyword = self._symbol(fields[i], "una clave de acción (:parameters, ...)")
            if i + 1 >= len(fields):
                raise PDDLSyntaxError(f"falta el valor de {keyword.value}", keyword.location)
            if keyword.value in seen:
                raise PDDLSyntaxError(f"{keyword.value} repetido", keyword.location)
            seen.add(keyword.value)
            value = fields[i + 1]
            if keyword.value == ":parameters":
                node["parameters"] = self._parameters(
                    self._list(value, "la lista de parámetros").items
                )
            elif keyword.value == ":precondition":
                if not self._is_empty_list(value):
                    node["precondition"] = self._formula(value)
            elif keyword.value == ":effect":
                if not self._is_empty_list(value):
                    node["effect"] = self._effect(value)
            else:
                raise PDDLSyntaxError(
                    f"clave de acción desconocida '{keyword.value}'", keyword.location
                )
        return self._with_loc(node, section)

    # --- Problema ----------------------------------------------------------------------

    def _problem(self, name: str, sections: list[tuple[Symbol, SList]]) -> JSONDict:
        document = self._header("problem", name)
        document.update(domain=None, requirements=[], objects=[], init=[], goal=None)
        seen: set[str] = set()
        for keyword, section in sections:
            key = keyword.value
            self._check_unique_section(key, keyword, seen)
            body = section.items[1:]
            if key == ":domain":
                if len(body) != 1:
                    raise PDDLSyntaxError("se esperaba (:domain <nombre>)", section.location)
                document["domain"] = self._name(body[0], "el nombre del dominio")
            elif key == ":requirements":
                document["requirements"] = self._requirements(body)
            elif key == ":objects":
                document["objects"] = self._objects(body)
            elif key == ":init":
                document["init"] = [self._init_element(item) for item in body]
            elif key == ":goal":
                if len(body) != 1:
                    raise PDDLSyntaxError("se esperaba (:goal <fórmula>)", section.location)
                document["goal"] = self._formula(body[0])
            elif key == ":metric":
                document["metric"] = self._metric(section)
            else:
                self._unknown_section(keyword, _FUTURE_PROBLEM_SECTIONS, "problema")
        if document["domain"] is None:
            raise PDDLSyntaxError("falta la sección (:domain <nombre>)", self._start(sections))
        if document["goal"] is None:
            raise PDDLSyntaxError("falta la sección (:goal ...)", self._start(sections))
        return document

    def _init_element(self, expr: SExpr) -> JSONDict:
        element = self._list(expr, "un átomo")
        head = element.items[0] if element.items else None
        if isinstance(head, Symbol):
            if head.value == "=":
                rest = element.items[1:]
                self._arity(element, rest, 2, "(= (<función> <objetos>) <número>)")
                node = {
                    "op": "=",
                    "fluent": self._fluent(self._list(rest[0], "una función")),
                    "value": self._number(rest[1]),
                }
                return self._with_loc(node, element)
            if (
                head.value == "at"
                and len(element.items) == 3
                and isinstance(element.items[1], Symbol)
                and _NUMBER.match(element.items[1].value)
            ):
                self._unsupported(
                    "literales iniciales temporizados", ":timed-initial-literals", head
                )
        return self._atom(element)

    # --- Fórmulas y efectos ------------------------------------------------------------

    def _formula(self, expr: SExpr) -> JSONDict:
        formula = self._list(expr, "una fórmula")
        if not formula.items:
            raise PDDLSyntaxError("fórmula vacía", formula.location)
        head = self._symbol(formula.items[0], "un operador o un predicado")
        op, rest = head.value, formula.items[1:]
        if op in ("and", "or"):
            node: JSONDict = {"op": op, "args": [self._formula(item) for item in rest]}
        elif op == "not":
            self._arity(formula, rest, 1, "(not <fórmula>)")
            node = {"op": "not", "arg": self._formula(rest[0])}
        elif op == "imply":
            self._arity(formula, rest, 2, "(imply <fórmula> <fórmula>)")
            node = {"op": "imply", "args": [self._formula(rest[0]), self._formula(rest[1])]}
        elif op in ("exists", "forall"):
            self._arity(formula, rest, 2, f"({op} (<variables>) <fórmula>)")
            node = {
                "op": op,
                "parameters": self._quantified(rest[0]),
                "arg": self._formula(rest[1]),
            }
        elif op == "=" and len(rest) == 2 and all(self._is_term(item) for item in rest):
            node = {"op": "=", "args": [self._term(rest[0]), self._term(rest[1])]}
        elif op in _COMPARATORS:
            self._arity(formula, rest, 2, f"({op} <expresión> <expresión>)")
            node = {
                "op": "compare",
                "comparator": op,
                "left": self._expression(rest[0]),
                "right": self._expression(rest[1]),
            }
        elif op in _FUTURE_FORMULA_OPS:
            self._unsupported(f"'{op}' en fórmulas", _FUTURE_FORMULA_OPS[op], head)
        else:
            return self._atom(formula)
        return self._with_loc(node, formula)

    def _effect(self, expr: SExpr, *, conditional: bool = False) -> JSONDict:
        """Un efecto. Dentro de ``when`` (``conditional``) solo se admiten literales y ``and``."""
        effect = self._list(expr, "un efecto")
        if not effect.items:
            raise PDDLSyntaxError("efecto vacío", effect.location)
        head = self._symbol(effect.items[0], "un operador o un predicado")
        op, rest = head.value, effect.items[1:]
        if op == "and":
            args = [self._effect(item, conditional=conditional) for item in rest]
            return self._with_loc({"op": "and", "args": args}, effect)
        if op == "not":
            if len(rest) != 1:
                raise PDDLSyntaxError("(not ...) lleva exactamente un átomo", effect.location)
            negated = self._list(rest[0], "un átomo")
            if negated.items and self._is_symbol_in(negated.items[0], _OPERATORS):
                raise PDDLSyntaxError(
                    "en un efecto, (not ...) solo puede negar un átomo", negated.location
                )
            atom = self._atom(negated)
            return self._with_loc({"op": "not", "arg": atom}, effect)
        if op in ("forall", "when") and conditional:
            raise PDDLSyntaxError(
                f"dentro de 'when' solo se admiten literales y 'and', no '{op}'", head.location
            )
        if op == "forall":
            self._arity(effect, rest, 2, "(forall (<variables>) <efecto>)")
            node = {
                "op": "forall",
                "parameters": self._quantified(rest[0]),
                "arg": self._effect(rest[1]),
            }
            return self._with_loc(node, effect)
        if op == "when":
            self._arity(effect, rest, 2, "(when <fórmula> <efecto>)")
            node = {
                "op": "when",
                "condition": self._formula(rest[0]),
                "arg": self._effect(rest[1], conditional=True),
            }
            return self._with_loc(node, effect)
        if op in _NUMERIC_EFFECTS:
            self._arity(effect, rest, 2, f"({op} (<función> <términos>) <expresión>)")
            node = {
                "op": op,
                "fluent": self._fluent(self._list(rest[0], "una función")),
                "value": self._expression(rest[1]),
            }
            return self._with_loc(node, effect)
        return self._atom(effect)

    # --- Expresiones numéricas ---------------------------------------------------------

    def _expression(self, expr: SExpr) -> JSONDict:
        """Expresión numérica: número, función o ``(+ - * /)`` de expresiones.

        ``total-time`` se acepta aquí y el loader comprueba que solo aparece en la métrica.
        """
        if self._is_symbol(expr, _TOTAL_TIME) or (
            isinstance(expr, SList)
            and len(expr.items) == 1
            and self._is_symbol(expr.items[0], _TOTAL_TIME)
        ):
            return self._with_loc({"op": "total-time"}, expr)
        if isinstance(expr, Symbol):
            if expr.value in _FUTURE_EXPRESSION_SYMBOLS:
                requirement = _FUTURE_EXPRESSION_SYMBOLS[expr.value]
                self._unsupported(f"'{expr.value}' en una expresión", requirement, expr)
            return self._with_loc({"op": "num", "value": self._number(expr)}, expr)
        if not expr.items:
            raise PDDLSyntaxError("expresión vacía", expr.location)
        head = self._symbol(expr.items[0], "un operador aritmético o una función")
        op, rest = head.value, expr.items[1:]
        if op in _ARITHMETIC:
            if op == "-" and len(rest) == 1:
                node: JSONDict = {"op": "-", "args": [self._expression(rest[0])]}
            elif op in ("-", "/"):
                self._arity(expr, rest, 2, f"({op} <expresión> <expresión>)")
                node = {"op": op, "args": [self._expression(item) for item in rest]}
            else:
                if len(rest) < 2:
                    raise PDDLSyntaxError(
                        f"({op} ...) necesita al menos dos términos", expr.location
                    )
                node = {"op": op, "args": [self._expression(item) for item in rest]}
            return self._with_loc(node, expr)
        if op in _FUTURE_EXPRESSION_SYMBOLS:
            self._unsupported(f"'{op}' en una expresión", _FUTURE_EXPRESSION_SYMBOLS[op], head)
        return self._fluent(expr)

    def _fluent(self, expr: SList) -> JSONDict:
        """Término de función: ``(nombre arg1 ... argN)``."""
        if not expr.items:
            raise PDDLSyntaxError("función vacía", expr.location)
        name = self._name(expr.items[0], "el nombre de la función")
        args = [self._term(item) for item in expr.items[1:]]
        return self._with_loc({"op": "fluent", "function": name, "args": args}, expr)

    def _number(self, expr: SExpr) -> float:
        symbol = self._symbol(expr, "un número")
        if not _NUMBER.match(symbol.value):
            raise PDDLSyntaxError(f"'{symbol.value}' no es un número", symbol.location)
        return float(symbol.value)

    def _is_term(self, expr: SExpr) -> bool:
        """``True`` si es un objeto o una variable (para distinguir la igualdad de objetos)."""
        return (
            isinstance(expr, Symbol)
            and bool(_NAME.match(expr.value) or _VARIABLE.match(expr.value))
            and expr.value not in _FUTURE_EXPRESSION_SYMBOLS
            and expr.value != _TOTAL_TIME
        )

    def _metric(self, section: SList) -> JSONDict:
        body = section.items[1:]
        self._arity(section, body, 2, "(:metric minimize|maximize <expresión>)")
        direction = self._symbol(body[0], "minimize o maximize")
        if direction.value not in ("minimize", "maximize"):
            raise PDDLSyntaxError(
                f"se esperaba minimize o maximize, no '{direction.value}'", direction.location
            )
        node = {"direction": direction.value, "expression": self._expression(body[1])}
        return self._with_loc(node, section)

    def _functions(self, items: tuple[SExpr, ...]) -> list[JSONDict]:
        """``(:functions (f ?x - t) (g) - number ...)``: el tipo, si aparece, es ``number``."""
        functions: list[JSONDict] = []
        i = 0
        while i < len(items):
            item = items[i]
            if self._is_symbol(item, "-"):
                if i + 1 >= len(items) or not functions:
                    raise PDDLSyntaxError("'-' mal colocado en :functions", item.location)
                result_type = self._symbol(items[i + 1], "el tipo del resultado")
                if result_type.value != "number":
                    self._unsupported(
                        f"funciones de tipo '{result_type.value}'", ":object-fluents", result_type
                    )
                i += 2
                continue
            declaration = self._list(item, "una declaración de función (nombre ?x ...)")
            if not declaration.items:
                raise PDDLSyntaxError("declaración de función vacía", declaration.location)
            node: JSONDict = {
                "name": self._name(declaration.items[0], "el nombre de la función"),
                "parameters": self._parameters(declaration.items[1:]),
            }
            functions.append(self._with_loc(node, declaration))
            i += 1
        return functions

    def _arity(self, expr: SList, rest: tuple[SExpr, ...], count: int, form: str) -> None:
        if len(rest) != count:
            raise PDDLSyntaxError(f"se esperaba {form}", expr.location)

    def _quantified(self, expr: SExpr) -> list[JSONDict]:
        variables = self._list(expr, "la lista de variables del cuantificador")
        if not variables.items:
            raise PDDLSyntaxError("el cuantificador no declara variables", variables.location)
        return self._parameters(variables.items)

    def _atom(self, atom: SList) -> JSONDict:
        if not atom.items:
            raise PDDLSyntaxError("átomo vacío", atom.location)
        predicate = self._name(atom.items[0], "el nombre del predicado")
        args = [self._term(item) for item in atom.items[1:]]
        return self._with_loc({"op": "atom", "predicate": predicate, "args": args}, atom)

    def _term(self, expr: SExpr) -> str:
        symbol = self._symbol(expr, "un objeto o una variable")
        if _NAME.match(symbol.value) or _VARIABLE.match(symbol.value):
            return symbol.value
        raise PDDLSyntaxError(f"'{symbol.value}' no es un objeto ni una variable", symbol.location)

    # --- Listas de declaraciones -------------------------------------------------------

    def _requirements(self, items: tuple[SExpr, ...]) -> list[str]:
        requirements = []
        for item in items:
            symbol = self._symbol(item, "un requisito (:strips, ...)")
            if symbol.value not in KNOWN_REQUIREMENTS:
                raise PDDLSyntaxError(f"requisito desconocido '{symbol.value}'", symbol.location)
            unsupported = unsupported_requirements([symbol.value])
            if unsupported:
                raise UnsupportedFeatureError(
                    f"requisito no soportado '{symbol.value}'"
                    + (
                        f" (implica {', '.join(unsupported)})"
                        if unsupported != [symbol.value]
                        else ""
                    ),
                    symbol.location,
                )
            requirements.append(symbol.value)
        return requirements

    def _objects(self, items: tuple[SExpr, ...]) -> list[JSONDict]:
        return self._typed_list(items, lambda item: self._name(item, "un nombre de objeto"))

    def _parameters(self, items: tuple[SExpr, ...]) -> list[JSONDict]:
        return self._typed_list(items, self._variable)

    def _types(self, items: tuple[SExpr, ...]) -> list[JSONDict]:
        """``(:types a b - c d)``: cada tipo con su supertipo (``parent``), si lo tiene."""
        types = self._typed_list(
            items, lambda item: self._name(item, "un nombre de tipo"), allow_either=False
        )
        for node in types:
            if "type" in node:
                node["parent"] = node.pop("type")
        return types

    def _typed_list(
        self,
        items: tuple[SExpr, ...],
        element: Callable[[SExpr], str],
        *,
        allow_either: bool = True,
    ) -> list[JSONDict]:
        """Lista PDDL con tipos opcionales: ``a b - t c - (either u v) d``.

        Los elementos sin tipo (los del final) no llevan el campo ``type``.
        """
        result: list[JSONDict] = []
        pending: list[JSONDict] = []
        i = 0
        while i < len(items):
            item = items[i]
            if self._is_symbol(item, "-"):
                if not pending:
                    raise PDDLSyntaxError("'-' sin elementos antes", item.location)
                if i + 1 >= len(items):
                    raise PDDLSyntaxError("falta el tipo después de '-'", item.location)
                type_spec = self._type_spec(items[i + 1], allow_either)
                for node in pending:
                    node["type"] = type_spec
                result.extend(pending)
                pending = []
                i += 2
            else:
                pending.append(self._with_loc({"name": element(item)}, item))
                i += 1
        result.extend(pending)
        return result

    def _type_spec(self, expr: SExpr, allow_either: bool) -> str | list[str]:
        if isinstance(expr, Symbol):
            return self._name(expr, "un tipo")
        if not allow_either:
            raise PDDLSyntaxError("un supertipo tiene que ser un nombre de tipo", expr.location)
        if not expr.items or not self._is_symbol(expr.items[0], "either"):
            raise PDDLSyntaxError("se esperaba un tipo o (either <tipos>)", expr.location)
        types = [self._name(item, "un tipo") for item in expr.items[1:]]
        if not types:
            raise PDDLSyntaxError("(either) sin tipos", expr.location)
        return types

    # --- Utilidades --------------------------------------------------------------------

    def _section(self, expr: SExpr) -> tuple[Symbol, SList]:
        section = self._list(expr, "una sección (:nombre ...)")
        if not section.items:
            raise PDDLSyntaxError("sección vacía", section.location)
        keyword = self._symbol(section.items[0], "el nombre de la sección")
        if not keyword.value.startswith(":"):
            raise PDDLSyntaxError(
                f"se esperaba una sección (:nombre ...), no '{keyword.value}'", keyword.location
            )
        return keyword, section

    def _check_unique_section(self, key: str, keyword: Symbol, seen: set[str]) -> None:
        if key in seen:
            raise PDDLSyntaxError(f"sección {key} repetida", keyword.location)
        seen.add(key)

    def _unknown_section(self, keyword: Symbol, future: dict[str, str], where: str) -> NoReturn:
        if keyword.value in future:
            self._unsupported(f"la sección {keyword.value}", future[keyword.value], keyword)
        raise PDDLSyntaxError(
            f"sección desconocida en un {where}: '{keyword.value}'", keyword.location
        )

    def _unsupported(self, what: str, requirement: str, where: SExpr) -> NoReturn:
        raise UnsupportedFeatureError(
            f"{what} necesita el requisito {requirement}, que todavía no está soportado",
            where.location,
        )

    def _start(self, sections: list[tuple[Symbol, SList]]) -> SourceLocation | None:
        return sections[0][1].location if sections else None

    def _list(self, expr: SExpr, expected: str) -> SList:
        if isinstance(expr, SList):
            return expr
        raise PDDLSyntaxError(f"se esperaba {expected}, no '{expr.value}'", expr.location)

    def _symbol(self, expr: SExpr, expected: str) -> Symbol:
        if isinstance(expr, Symbol):
            return expr
        raise PDDLSyntaxError(f"se esperaba {expected}, no una lista", expr.location)

    def _name(self, expr: SExpr, expected: str) -> str:
        symbol = self._symbol(expr, expected)
        if not _NAME.match(symbol.value):
            raise PDDLSyntaxError(f"'{symbol.value}' no es un nombre válido", symbol.location)
        return symbol.value

    def _variable(self, expr: SExpr) -> str:
        symbol = self._symbol(expr, "una variable (?x)")
        if not _VARIABLE.match(symbol.value):
            raise PDDLSyntaxError(f"'{symbol.value}' no es una variable válida", symbol.location)
        return symbol.value

    def _is_symbol(self, expr: SExpr, value: str) -> bool:
        return isinstance(expr, Symbol) and expr.value == value

    def _is_symbol_in(self, expr: SExpr, values: frozenset[str]) -> bool:
        return isinstance(expr, Symbol) and expr.value in values

    def _is_empty_list(self, expr: SExpr) -> bool:
        return isinstance(expr, SList) and not expr.items

    def _with_loc(self, node: JSONDict, expr: SExpr) -> JSONDict:
        if self.locations:
            node["loc"] = {"line": expr.location.line, "column": expr.location.column}
        return node
