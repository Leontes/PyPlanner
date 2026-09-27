"""Interfaz de línea de comandos de PyPlanner.

Subcomandos:

* ``parse FICHERO``: fase 1. Convierte un dominio o problema PDDL en JSON.
* ``check DOMINIO PROBLEMA``: fases 1 y 2. Comprueba que la tarea es válida y la resume.
  Acepta ficheros ``.pddl`` y ``.json``.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pyplanner import __version__
from pyplanner.errors import PDDLError
from pyplanner.frontend import read_domain, read_problem
from pyplanner.pddl import parse_file

#: Código de salida cuando la entrada no es válida.
EXIT_INVALID_INPUT = 2


def build_parser() -> argparse.ArgumentParser:
    """Construye el analizador de argumentos de la CLI."""
    parser = argparse.ArgumentParser(prog="pyplanner", description="Planificador PDDL en Python")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    parse = commands.add_parser("parse", help="convierte un fichero PDDL en JSON (fase 1)")
    parse.add_argument("file", type=Path, help="dominio o problema PDDL")
    parse.add_argument("-o", "--output", type=Path, help="fichero de salida (por defecto, stdout)")
    parse.add_argument(
        "--no-loc", action="store_true", help="no incluir las posiciones del PDDL en el JSON"
    )
    parse.add_argument("--compact", action="store_true", help="JSON en una sola línea")

    check = commands.add_parser("check", help="comprueba un dominio y un problema (fases 1 y 2)")
    check.add_argument("domain", type=Path, help="dominio (.pddl o .json)")
    check.add_argument("problem", type=Path, help="problema (.pddl o .json)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada de la CLI. Devuelve el código de salida."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "parse":
            return _parse(args)
        return _check(args)
    except PDDLError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID_INPUT
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID_INPUT


def _parse(args: argparse.Namespace) -> int:
    document = parse_file(args.file, locations=not args.no_loc)
    text = json.dumps(document, ensure_ascii=False, indent=None if args.compact else 2)
    if args.output is None:
        print(text)
    else:
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


def _check(args: argparse.Namespace) -> int:
    domain = read_domain(args.domain)
    problem = read_problem(args.problem, domain)
    print(
        f"Dominio '{domain.name}': {len(domain.predicates)} predicados, "
        f"{len(domain.actions)} acciones, {len(domain.constants)} constantes"
    )
    print(
        f"Problema '{problem.name}': {len(problem.objects)} objetos, "
        f"{len(problem.init)} hechos iniciales"
    )
    print(f"Requisitos: {' '.join(sorted(domain.requirements | problem.requirements))}")
    return 0
