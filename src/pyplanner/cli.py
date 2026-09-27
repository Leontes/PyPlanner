"""Interfaz de línea de comandos de PyPlanner.

Subcomandos:

* ``parse FICHERO``: fase 1. Convierte un dominio o problema PDDL en JSON.
* ``check DOMINIO PROBLEMA``: fases 1 y 2. Comprueba que la tarea es válida y la resume.
  Acepta ficheros ``.pddl`` y ``.json``.
* ``solve DOMINIO PROBLEMA``: busca un plan. Por defecto lanza el portfolio que corresponde
  a los núcleos indicados con ``-j`` (5, o menos si la máquina no los tiene); con ``-c`` se
  eligen las configuraciones a mano. El plan sale por la salida estándar (o a un fichero)
  y las estadísticas por la de error.
* ``configs``: lista las configuraciones predefinidas y los portfolios por núcleos.

Códigos de salida: 0 plan encontrado (o comando correcto), 2 entrada no válida, 3 se ha
demostrado que no hay plan, 4 no se ha encontrado plan dentro de los límites.
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
from pyplanner.planner import PlanningResult, available_cores, solve
from pyplanner.search import (
    DEFAULT_CORES,
    PORTFOLIOS,
    PRESETS,
    ConfigError,
    SearchStatus,
    portfolio_for_cores,
)

#: Código de salida cuando la entrada no es válida.
EXIT_INVALID_INPUT = 2
#: Código de salida cuando se ha demostrado que la tarea no tiene solución.
EXIT_UNSOLVABLE = 3
#: Código de salida cuando no se ha encontrado plan dentro de los límites.
EXIT_NO_PLAN = 4


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

    solve_cmd = commands.add_parser("solve", help="busca un plan")
    solve_cmd.add_argument("domain", type=Path, help="dominio (.pddl o .json)")
    solve_cmd.add_argument("problem", type=Path, help="problema (.pddl o .json)")
    solve_cmd.add_argument(
        "-c",
        "--config",
        action="append",
        choices=sorted(PRESETS),
        metavar="NOMBRE",
        help="configuración de búsqueda (repetible: varias forman un portfolio); por "
        "defecto se usa el portfolio de los núcleos disponibles (ver 'pyplanner configs')",
    )
    solve_cmd.add_argument(
        "-j",
        "--cores",
        type=int,
        help=f"núcleos a usar (por defecto {DEFAULT_CORES}, o los que tenga la máquina si son "
        "menos); determina qué búsquedas se lanzan",
    )
    solve_cmd.add_argument("-t", "--time-limit", type=float, help="límite de tiempo en segundos")
    solve_cmd.add_argument("--batch-size", type=int, help="tamaño de lote de evaluación")
    solve_cmd.add_argument(
        "--device", choices=("python", "cpu", "cuda"), help="dónde se evalúa la heurística"
    )
    solve_cmd.add_argument(
        "-o", "--output", type=Path, help="fichero del plan (por defecto, stdout)"
    )
    solve_cmd.add_argument(
        "-f", "--format", choices=("ipc", "json"), default="ipc", help="formato del plan"
    )
    solve_cmd.add_argument("-q", "--quiet", action="store_true", help="sin estadísticas")

    commands.add_parser("configs", help="lista las configuraciones de búsqueda predefinidas")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada de la CLI. Devuelve el código de salida."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "parse":
            return _parse(args)
        if args.command == "check":
            return _check(args)
        if args.command == "solve":
            return _solve(args)
        return _configs()
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


def _solve(args: argparse.Namespace) -> int:
    machine = available_cores()
    cores = args.cores if args.cores is not None else min(DEFAULT_CORES, machine)
    if args.config:
        names = list(dict.fromkeys(args.config))  # sin repetidos, en orden
        if cores < 1:
            raise ConfigError(f"el número de núcleos tiene que ser ≥ 1 (se ha pedido {cores})")
        if len(names) > cores:
            raise ConfigError(
                f"se han pedido {len(names)} configuraciones pero solo {cores} núcleos; "
                "sube --cores o quita configuraciones"
            )
        configs = [PRESETS[n] for n in names]
    else:
        configs = list(portfolio_for_cores(cores))
    if not args.quiet:
        if cores > machine:
            print(
                f"aviso: se han pedido {cores} núcleos, pero la máquina tiene {machine}",
                file=sys.stderr,
            )
        searches = ", ".join(c.name for c in configs)
        print(f"Núcleos: {cores} → {len(configs)} búsqueda(s): {searches}", file=sys.stderr)
    overrides = {
        key: value
        for key, value in (("batch_size", args.batch_size), ("device", args.device))
        if value is not None
    }
    if overrides:
        configs = [c.with_changes(**overrides) for c in configs]

    domain = read_domain(args.domain)
    problem = read_problem(args.problem, domain)
    result = solve(domain, problem, configs, time_limit=args.time_limit)
    if not args.quiet:
        _report(result)

    if result.plan is not None:
        text = result.plan.to_json() + "\n" if args.format == "json" else result.plan.to_ipc()
        if args.output is None:
            sys.stdout.write(text)
        else:
            args.output.write_text(text, encoding="utf-8")
        return 0
    if result.status is SearchStatus.UNSOLVABLE:
        return EXIT_UNSOLVABLE
    return EXIT_NO_PLAN


def _report(result: PlanningResult) -> None:
    task, search = result.task, result.search
    stats = search.statistics

    def err(line: str) -> None:
        print(line, file=sys.stderr)

    err(
        f"Tarea: {len(task.facts)} hechos, {len(task.actions)} acciones "
        f"(grounding {result.grounding_time:.3f} s)"
    )
    if result.portfolio is not None:
        for name, member in result.portfolio.results.items():
            mark = "*" if name == search.config_name else " "
            err(
                f" {mark} {name:22s} {member.status.value:14s} "
                f"exp={member.statistics.expanded} t={member.statistics.time:.3f} s"
            )
    err(f"Resultado: {search.status.value} ({search.config_name})")
    err(
        f"Expandidos {stats.expanded}, generados {stats.generated}, "
        f"evaluados {stats.evaluated} en {stats.evaluation_batches} lotes, "
        f"callejones {stats.dead_ends}, tiempo {stats.time:.3f} s"
    )
    if result.plan is not None:
        err(f"Plan: {len(result.plan)} acciones, coste {result.plan.cost:g}")
    if search.message:
        err(f"Detalle: {search.message}")


def _configs() -> int:
    membership: dict[str, list[int]] = {}
    for size, names in PORTFOLIOS.items():
        for name in names:
            membership.setdefault(name, []).append(size)
    print("Configuraciones:")
    for name, config in PRESETS.items():
        extras = []
        if config.preferred_operators:
            extras.append("acciones útiles")
        if config.lazy:
            extras.append("perezosa")
        if config.algorithm == "wastar":
            extras.append(f"peso {config.weight:g}")
        if config.tie_breaking != "fifo":
            extras.append(f"desempate {config.tie_breaking}")
        tags = []
        if name in membership:
            tags.append("núcleos " + ",".join(map(str, membership[name])))
        if config.is_optimal:
            tags.append("óptima")
        detail = f"{config.algorithm} + {config.heuristic}"
        if extras:
            detail += f" ({', '.join(extras)})"
        print(f"  {name:20s} {detail}" + (f"  [{', '.join(tags)}]" if tags else ""))
    print(f"\nPortfolios por núcleos (-j N; por defecto {DEFAULT_CORES}):")
    for size, names in PORTFOLIOS.items():
        print(f"  {size}: {', '.join(names)}")
    print("  Con otro número de núcleos se usa el portfolio más grande que quepa.")
    return 0
