"""Lectura de tareas desde fichero, en PDDL o en JSON.

Un fichero ``.json`` se considera ya convertido y va directo a la fase 2. Cualquier otro
fichero se trata como PDDL y pasa por las dos fases. Así la CLI y los tests usan el mismo
camino que usará el planificador integrado en otro sistema.
"""

import json
from pathlib import Path
from typing import Any

from pyplanner.errors import SchemaValidationError
from pyplanner.loader import load_domain, load_problem
from pyplanner.loader.ast import Domain, Problem
from pyplanner.pddl import parse_file


def read_document(path: str | Path, *, locations: bool = True) -> dict[str, Any]:
    """Devuelve el documento JSON de un fichero ``.json`` o ``.pddl``.

    Raises:
        PDDLError: si el fichero PDDL no es válido o el JSON no se puede leer.
    """
    path = Path(path)
    if path.suffix.lower() == ".json":
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SchemaValidationError(f"{path}: JSON mal formado: {exc}") from exc
        if isinstance(document, dict):
            document.setdefault("source", str(path))
        return document  # type: ignore[no-any-return]
    return parse_file(path, locations=locations)


def read_domain(path: str | Path) -> Domain:
    """Lee un dominio en PDDL o JSON y lo convierte en AST."""
    return load_domain(read_document(path))


def read_problem(path: str | Path, domain: Domain) -> Problem:
    """Lee un problema en PDDL o JSON y lo convierte en AST, comprobándolo contra ``domain``."""
    return load_problem(read_document(path), domain)
