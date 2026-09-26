"""Interfaz de línea de comandos de PyPlanner."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from pyplanner import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyplanner", description="Planificador PDDL en Python")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("domain", type=Path, help="fichero de dominio PDDL")
    parser.add_argument("problem", type=Path, help="fichero de problema PDDL")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"Dominio: {args.domain}\nProblema: {args.problem}")
    print("El planificador aún no está implementado.")
    return 1
