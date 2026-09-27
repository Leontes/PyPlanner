"""Análisis léxico de PDDL.

PDDL es un lenguaje de s-expresiones, así que solo hay tres clases de token: paréntesis de
apertura, paréntesis de cierre y símbolos. Un símbolo es cualquier secuencia de caracteres
sin espacios, paréntesis ni ``;``. El lexer no decide si un símbolo es un nombre, una
variable (``?x``) o una palabra clave (``:action``); eso lo hace el parser.

PDDL no distingue mayúsculas de minúsculas, así que todos los símbolos se pasan a
minúsculas. Los comentarios van de ``;`` hasta el final de la línea.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum, auto

from pyplanner.errors import SourceLocation


class TokenKind(Enum):
    """Clase de un token."""

    LPAREN = auto()
    RPAREN = auto()
    SYMBOL = auto()


@dataclass(frozen=True)
class Token:
    """Un token con su texto (ya en minúsculas) y su posición."""

    kind: TokenKind
    text: str
    location: SourceLocation


_DELIMITERS = frozenset("();")


def tokenize(text: str, filename: str | None = None) -> Iterator[Token]:
    """Divide un texto PDDL en tokens.

    Args:
        text: contenido del fichero PDDL.
        filename: nombre del fichero, solo para las posiciones de los errores.

    Yields:
        Los tokens en orden de aparición.
    """
    line = 1
    line_start = 0
    i = 0
    n = len(text)
    while i < n:
        char = text[i]
        if char == "\n":
            line += 1
            i += 1
            line_start = i
        elif char.isspace():
            i += 1
        elif char == ";":
            while i < n and text[i] != "\n":
                i += 1
        elif char == "(" or char == ")":
            kind = TokenKind.LPAREN if char == "(" else TokenKind.RPAREN
            yield Token(kind, char, SourceLocation(line, i - line_start + 1, filename))
            i += 1
        else:
            start = i
            while i < n and not text[i].isspace() and text[i] not in _DELIMITERS:
                i += 1
            location = SourceLocation(line, start - line_start + 1, filename)
            yield Token(TokenKind.SYMBOL, text[start:i].lower(), location)
