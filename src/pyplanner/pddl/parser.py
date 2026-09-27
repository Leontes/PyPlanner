"""Fase 1 del parser: PDDL → documento JSON.

Esta fase solo comprueba la **sintaxis**. Recorre el árbol de s-expresiones y genera un
documento JSON que sigue el esquema de :mod:`pyplanner.schema`, conservando la estructura del
PDDL original (no simplifica conjunciones ni normaliza fórmulas). Las comprobaciones de
significado (predicados declarados, aridades, variables libres…) son cosa de la fase 2
(:mod:`pyplanner.loader`).

Hay dos clases de error:

* :class:`~pyplanner.errors.PDDLSyntaxError` si el texto no es PDDL bien formado.
* :class:`~pyplanner.errors.UnsupportedFeatureError` si es PDDL válido pero usa algo que el
  planificador todavía no soporta (por ejemplo, ``:types`` o una precondición negativa).
"""

import re
from pathlib import Path
from typing import Any, NoReturn

from pyplanner.errors import PDDLSyntaxError, SourceLocation, UnsupportedFeatureError
from pyplanner.pddl.sexpr import SExpr, SList, Symbol, parse_sexprs
from pyplanner.requirements import KNOWN_REQUIREMENTS, unsupported_requirements
from pyplanner.schema import SCHEMA_VERSION

JSONDict = dict[str, Any]

_NAME = re.compile(r"^[a-z][a-z0-9_-]*$")
_VARIABLE = re.compile(r"^\?[a-z][a-z0-9_-]*$")
_NUMBER = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)$")

# Secciones y operadores de versiones posteriores de PDDL, con el requisito que los introduce.
# Sirven para dar un error de "no soportado" en lugar de uno de sintaxis.
_FUTURE_DOMAIN_SECTIONS = {
    ":types": ":typing",
    ":functions": ":numeric-fluents",
    ":durative-action": ":durative-actions",
    ":derived": ":derived-predicates",
    ":process": ":time",
    ":event": ":time",
    ":constraints": ":constraints",
}
_FUTURE_PROBLEM_SECTIONS = {
    ":metric": ":numeric-fluents",
    ":constraints": ":constraints",
}
_FUTURE_FORMULA_OPS = {
    "not": ":negative-preconditions",
    "or": ":disjunctive-preconditions",
    "imply": ":disjunctive-preconditions",
    "exists": ":existential-preconditions",
    "forall": ":universal-preconditions",
    "=": ":equality",
    "<": ":numeric-fluents",
    ">": ":numeric-fluents",
    "<=": ":numeric-fluents",
    ">=": ":numeric-fluents",
    "preference": ":preferences",
}
_FUTURE_EFFECT_OPS = {
    "when": ":conditional-effects",
    "forall": ":conditional-effects",
    "assign": ":numeric-fluents",
    "increase": ":numeric-fluents",
    "decrease": ":numeric-fluents",
    "scale-up": ":numeric-fluents",
    "scale-down": ":numeric-fluents",
}


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
    """
    return _Parser(filename, locations).parse(text)


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
        document.update(requirements=[], constants=[], predicates=[], actions=[])
        seen: set[str] = set()
        for keyword, section in sections:
            key = keyword.value
            if key != ":action":
                self._check_unique_section(key, keyword, seen)
            body = section.items[1:]
            if key == ":requirements":
                document["requirements"] = self._requirements(body)
            elif key == ":constants":
                document["constants"] = self._objects(body)
            elif key == ":predicates":
                document["predicates"] = [self._predicate(item) for item in body]
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
                self._unsupported("valores iniciales de funciones", ":numeric-fluents", head)
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
        if head.value == "and":
            args = [self._formula(item) for item in formula.items[1:]]
            return self._with_loc({"op": "and", "args": args}, formula)
        if head.value in _FUTURE_FORMULA_OPS:
            self._unsupported(f"'{head.value}' en fórmulas", _FUTURE_FORMULA_OPS[head.value], head)
        return self._atom(formula)

    def _effect(self, expr: SExpr) -> JSONDict:
        effect = self._list(expr, "un efecto")
        if not effect.items:
            raise PDDLSyntaxError("efecto vacío", effect.location)
        head = self._symbol(effect.items[0], "un operador o un predicado")
        if head.value == "and":
            args = [self._effect(item) for item in effect.items[1:]]
            return self._with_loc({"op": "and", "args": args}, effect)
        if head.value == "not":
            if len(effect.items) != 2:
                raise PDDLSyntaxError("(not ...) lleva exactamente un átomo", effect.location)
            atom = self._atom(self._list(effect.items[1], "un átomo"))
            return self._with_loc({"op": "not", "arg": atom}, effect)
        if head.value in _FUTURE_EFFECT_OPS:
            self._unsupported(f"'{head.value}' en efectos", _FUTURE_EFFECT_OPS[head.value], head)
        return self._atom(effect)

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
        self._reject_types(items)
        return [
            self._with_loc({"name": self._name(item, "un nombre de objeto")}, item)
            for item in items
        ]

    def _parameters(self, items: tuple[SExpr, ...]) -> list[JSONDict]:
        self._reject_types(items)
        return [self._with_loc({"name": self._variable(item)}, item) for item in items]

    def _reject_types(self, items: tuple[SExpr, ...]) -> None:
        for item in items:
            if isinstance(item, Symbol) and item.value == "-":
                self._unsupported("listas con tipos", ":typing", item)

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

    def _is_empty_list(self, expr: SExpr) -> bool:
        return isinstance(expr, SList) and not expr.items

    def _with_loc(self, node: JSONDict, expr: SExpr) -> JSONDict:
        if self.locations:
            node["loc"] = {"line": expr.location.line, "column": expr.location.column}
        return node
