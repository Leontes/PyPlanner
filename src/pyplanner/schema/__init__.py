"""Esquema JSON del formato de entrada del planificador y su validación estructural.

El esquema (``pddl.schema.json``) es el contrato entre la fase 1 del parser (PDDL → JSON) y
la fase 2 (JSON → objetos internos). También es el formato que usará cualquier otro
componente que quiera pasarle una tarea al planificador sin escribir PDDL.
"""

import json
from functools import cache
from importlib.resources import files
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError, best_match

from pyplanner.errors import SchemaValidationError

#: Versión más reciente del formato JSON que genera y acepta el planificador.
SCHEMA_VERSION = 3


@cache
def load_schema() -> dict[str, Any]:
    """Devuelve el esquema JSON como diccionario."""
    text = files(__package__).joinpath("pddl.schema.json").read_text(encoding="utf-8")
    schema: dict[str, Any] = json.loads(text)
    return schema


@cache
def _validator() -> Draft202012Validator:
    schema = load_schema()
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_document(document: Any) -> None:
    """Comprueba que un documento cumple el esquema.

    Raises:
        SchemaValidationError: con el error más relevante y su ruta dentro del documento.
    """
    error = best_match(_validator().iter_errors(document))
    if error is not None:
        raise SchemaValidationError(
            f"el documento no cumple el esquema: {error.message}", path=_format_path(error)
        )


def _format_path(error: ValidationError) -> str:
    """Convierte la ruta de un error de ``jsonschema`` en algo como ``actions[0].effect``."""
    path = ""
    for part in error.absolute_path:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    return path.lstrip(".") or "<raíz>"
