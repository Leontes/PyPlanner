"""Construcción del árbol de s-expresiones a partir de los tokens.

El árbol resultante no tiene ningún significado PDDL todavía: solo listas y símbolos con
su posición. El parser (:mod:`pyplanner.pddl.parser`) lo recorre para generar el JSON.
"""

from dataclasses import dataclass

from pyplanner.errors import PDDLSyntaxError, SourceLocation
from pyplanner.pddl.lexer import TokenKind, tokenize


@dataclass(frozen=True)
class Symbol:
    """Un símbolo atómico (nombre, variable, palabra clave…)."""

    value: str
    location: SourceLocation


@dataclass(frozen=True)
class SList:
    """Una lista entre paréntesis."""

    items: tuple["SExpr", ...]
    location: SourceLocation


SExpr = Symbol | SList


def parse_sexprs(text: str, filename: str | None = None) -> list[SExpr]:
    """Convierte un texto en la lista de s-expresiones de nivel superior.

    Raises:
        PDDLSyntaxError: si los paréntesis no están equilibrados.
    """
    stack: list[tuple[SourceLocation, list[SExpr]]] = []
    top: list[SExpr] = []
    for token in tokenize(text, filename):
        if token.kind is TokenKind.LPAREN:
            stack.append((token.location, []))
        elif token.kind is TokenKind.RPAREN:
            if not stack:
                raise PDDLSyntaxError("paréntesis de cierre sin abrir", token.location)
            location, items = stack.pop()
            node = SList(tuple(items), location)
            (stack[-1][1] if stack else top).append(node)
        else:
            (stack[-1][1] if stack else top).append(Symbol(token.text, token.location))
    if stack:
        raise PDDLSyntaxError("paréntesis sin cerrar", stack[-1][0])
    return top
