"""Errores del planificador y posiciones en el código fuente.

Todos los errores que produce la lectura de una tarea heredan de :class:`PDDLError`, de modo
que quien integre el planificador puede capturarlos con un único ``except``. Cada error
puede llevar la posición del fichero PDDL original en la que se detectó.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceLocation:
    """Posición dentro de un fichero fuente (línea y columna empiezan en 1)."""

    line: int
    column: int
    file: str | None = None

    def __str__(self) -> str:
        prefix = f"{self.file}:" if self.file else ""
        return f"{prefix}{self.line}:{self.column}"


class PDDLError(Exception):
    """Error base de la lectura de dominios y problemas.

    Args:
        message: descripción del error.
        location: posición en el fichero PDDL original, si se conoce.
        path: ruta dentro del documento JSON (p. ej. ``actions[0].effect``), si se conoce.
    """

    def __init__(
        self,
        message: str,
        location: SourceLocation | None = None,
        path: str | None = None,
    ) -> None:
        self.message = message
        self.location = location
        self.path = path
        super().__init__(str(self))

    def __str__(self) -> str:
        parts = []
        if self.location is not None:
            parts.append(f"{self.location}: ")
        parts.append(self.message)
        if self.path:
            parts.append(f" [en {self.path}]")
        return "".join(parts)


class PDDLSyntaxError(PDDLError):
    """El texto PDDL no está bien formado (fase 1)."""


class UnsupportedFeatureError(PDDLError):
    """La entrada usa un requisito o una construcción que esta versión no soporta."""


class SchemaValidationError(PDDLError):
    """El documento JSON no cumple el esquema (fase 2, validación estructural)."""


class SemanticError(PDDLError):
    """El documento está bien formado pero no tiene sentido (fase 2, validación semántica).

    Por ejemplo: un predicado sin declarar, una aridad incorrecta o una variable libre.
    """
