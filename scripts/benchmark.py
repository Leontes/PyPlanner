"""Mide configuraciones de búsqueda sobre una colección de problemas y valida los planes.

Uso::

    python scripts/benchmark.py benchmarks/timed-initial-literals/satellite \\
        -c gbfs-ff-pref bfws-ff-pref -t 60 -m 2500 -w 4

El directorio tiene ``domain.pddl`` y problemas ``pNN.pddl``. Cada par (problema,
configuración) se resuelve en su propio proceso con un límite de tiempo y de memoria; los
planes se validan con VAL (``tools/bin/Validate``) si está compilado. La salida es una tabla
por problema y un resumen por configuración (resueltos y tiempo total, contando el límite
para los no resueltos), y opcionalmente un JSON con todas las filas.

Con preferencias, la búsqueda es *anytime* (el mejor plan en el límite de tiempo): la tabla
da además la métrica del plan (con la de VAL si no coincide) y si es óptimo, y el resumen,
la suma de las métricas.

**Memoria:** como mucho 5 procesos a la vez y siempre con límite de memoria (dos medidas
agotaron la memoria de WSL y obligaron a reiniciar). Cada proceso tiene dos límites:

* el de la búsqueda (``max_memory``), que la termina con ``limit_reached``;
* uno duro del sistema (``RLIMIT_AS``, memoria virtual) de ``--memory`` más 512 MB, que
  cubre también el grounding (el de la búsqueda solo se mira mientras busca): al pasarlo,
  el proceso recibe ``MemoryError`` en lugar de agotar la máquina. No se pone con las
  configuraciones que usan PyTorch (``cpu``, ``cuda``, ``hybrid``), cuya reserva de memoria
  virtual (CUDA) lo rompería.

Antes de empezar se comprueba que procesos × (``--memory`` + 512 MB) quepa en el 80 % de la
memoria disponible.
"""

import argparse
import json
import multiprocessing
import re
import resource
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
VALIDATE = ROOT / "tools" / "bin" / "Validate"
MAX_WORKERS = 5
#: Memoria virtual de más sobre ``--memory`` en el límite duro (Python, NumPy y Numba
#: reservan unos 400 MB).
VIRTUAL_MARGIN_MB = 512


def run_one(domain: str, problem: str, config: str, limit: float, memory: int) -> dict[str, Any]:
    """Resuelve un problema con una configuración (en un proceso aparte)."""
    from pyplanner.search import PRESETS as presets

    if presets[config].device not in ("cpu", "cuda", "hybrid"):
        limit_bytes = (memory + VIRTUAL_MARGIN_MB) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

    from pyplanner.frontend import read_domain, read_problem
    from pyplanner.grounding import ground
    from pyplanner.plan import Plan
    from pyplanner.search import PRESETS, SearchStatus, search
    from pyplanner.search.anytime import search_anytime

    start = time.perf_counter()
    loaded = read_domain(domain)
    task = ground(loaded, read_problem(problem, loaded))
    grounding = time.perf_counter() - start
    settings = PRESETS[config].with_changes(max_time=limit, max_memory=memory)
    anytime = task.preferences is not None
    result = search_anytime(task, settings) if anytime else search(task, settings)
    row: dict[str, Any] = {
        "problem": Path(problem).stem,
        "config": config,
        "status": result.status.value,
        "time": round(grounding + result.statistics.time, 3),
        "expanded": result.statistics.expanded,
        "dead_ends": result.statistics.dead_ends,
        "valid": None,
    }
    if result.status is SearchStatus.SOLVED and result.plan is not None:
        plan = Plan.from_indices(task, result.plan)
        row["length"] = len(plan)
        row["makespan"] = plan.makespan
        row["metric"] = plan.metric
        if anytime:
            row["optimal"] = result.optimal
        if VALIDATE.exists():
            with tempfile.NamedTemporaryFile("w", suffix=".plan", delete=False) as handle:
                handle.write(plan.to_ipc())
            output = subprocess.run(
                [str(VALIDATE), domain, problem, handle.name],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            ).stdout
            Path(handle.name).unlink()
            row["valid"] = "Plan valid" in output
            value = re.search(r"Final value: (\S+)", output)
            row["val_metric"] = float(value.group(1)) if value else None
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directory", type=Path, help="directorio con domain.pddl y pNN.pddl")
    parser.add_argument("-c", "--configs", nargs="+", default=["gbfs-ff-pref"])
    parser.add_argument("-p", "--problems", nargs="*", help="problemas (p01 p02...); todos")
    parser.add_argument("-t", "--time", type=float, default=60.0, help="segundos por búsqueda")
    parser.add_argument("-m", "--memory", type=int, default=2500, help="MB por búsqueda")
    parser.add_argument("-w", "--workers", type=int, default=4, help=f"máximo {MAX_WORKERS}")
    parser.add_argument("-o", "--output", type=Path, help="JSON con todas las filas")
    args = parser.parse_args()
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers tiene que estar entre 1 y {MAX_WORKERS}")
    from pyplanner.search.memory import available_memory_mb

    available = available_memory_mb()
    needed = args.workers * (args.memory + VIRTUAL_MARGIN_MB)
    if available is None or needed > 0.8 * available:
        parser.error(
            f"{args.workers} procesos × ({args.memory} + {VIRTUAL_MARGIN_MB}) MB = {needed} MB no "
            f"caben en el 80% de la memoria disponible ({available} MB): baja -w o -m"
        )

    domain = args.directory / "domain.pddl"
    problems = sorted(args.directory.glob("p[0-9]*.pddl"))
    if args.problems:
        problems = [p for p in problems if p.stem in set(args.problems)]
    jobs = [
        (str(domain), str(problem), config, args.time, args.memory)
        for problem in problems
        for config in args.configs
    ]
    context = multiprocessing.get_context("spawn")
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        futures = [pool.submit(run_one, *job) for job in jobs]
        for job, future in zip(jobs, futures, strict=True):
            try:
                row = future.result()
            except Exception as error:  # noqa: BLE001 - se informa y se sigue
                row = {"problem": Path(job[1]).stem, "config": job[2], "status": "error",
                       "time": None, "error": str(error)}  # fmt: skip
            rows.append(row)
            valid = {True: "válido", False: "INVÁLIDO", None: ""}[row.get("valid")]
            quality = ""
            if row.get("metric") is not None:
                quality = f" métrica={row['metric']:g}"
                val_metric = row.get("val_metric")
                if val_metric is not None and abs(val_metric - row["metric"]) > 1e-3:
                    quality += f" (VAL {val_metric:g})"
                if row.get("optimal"):
                    quality += " óptimo"
            if row.get("error"):
                quality = f" {row['error'] or 'MemoryError'}"
            print(
                f"{row['problem']:5} {row['config']:22} {row['status']:14} "
                f"t={row['time'] if row['time'] is not None else '-':>8} "
                f"exp={row.get('expanded', '-'):>8} "
                f"callejones={row.get('dead_ends', '-'):>7} {valid}{quality}",
                flush=True,
            )

    print("\nResumen (los no resueltos cuentan el límite de tiempo):")
    for config in args.configs:
        mine = [r for r in rows if r["config"] == config]
        solved = [r for r in mine if r["status"] == "solved" and r.get("valid") is not False]
        total = sum(r["time"] if r in solved else args.time for r in mine)
        invalid = sum(r.get("valid") is False for r in mine)
        note = f", {invalid} planes inválidos" if invalid else ""
        if any("optimal" in r for r in solved):
            metric = sum(r["metric"] for r in solved if r.get("metric") is not None)
            optimal = sum(bool(r.get("optimal")) for r in solved)
            note += f", métrica total {metric:g}, {optimal} óptimos"
        print(f"  {config:22} {len(solved)}/{len(mine)} resueltos, {total:.1f} s{note}")
    if args.output:
        args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
