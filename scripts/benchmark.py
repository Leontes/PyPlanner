"""Mide configuraciones de búsqueda sobre una colección de problemas y valida los planes.

Uso::

    python scripts/benchmark.py benchmarks/timed-initial-literals/satellite \\
        -c gbfs-ff-pref bfws-ff-pref -t 60 -m 2500 -w 4

El directorio tiene ``domain.pddl`` y problemas ``pNN.pddl`` (o un dominio por problema,
``domain-pNN.pddl``, como en algunas colecciones de las IPC). Cada par (problema,
configuración) se resuelve en su propio proceso con un límite de tiempo y de memoria; los
planes se validan con VAL (``tools/bin/Validate``) si está compilado. La salida es una tabla
por problema y un resumen por configuración (resueltos y tiempo total, contando el límite
para los no resueltos), y opcionalmente un JSON con todas las filas.

Con preferencias, la búsqueda es *anytime* (el mejor plan en el límite de tiempo): la tabla
da además la métrica del plan (con la de VAL si no coincide) y si es óptimo, y el resumen,
la suma de las métricas.

Cada trabajo va en un proceso nuevo: uno reutilizado conservaría la memoria de la búsqueda
anterior y llegaría antes a su límite.

**Memoria:** como mucho 5 procesos a la vez y siempre con límite de memoria (dos medidas
agotaron la memoria de WSL y obligaron a reiniciar). Cada proceso tiene dos límites:

* el de la búsqueda (``max_memory``), que la termina con ``limit_reached``;
* uno duro del sistema de ``--memory`` más 512 MB, que cubre también el grounding (el de
  la búsqueda solo se mira mientras busca): al pasarlo, el proceso recibe ``MemoryError``
  en lugar de agotar la máquina. Es ``RLIMIT_AS`` (memoria virtual), salvo con las
  configuraciones que usan PyTorch (``cpu``, ``cuda``, ``hybrid``): CUDA reserva decenas de
  GB de memoria virtual que no usa, así que en ellas es ``RLIMIT_DATA`` (la memoria de
  datos), con 1024 MB más para PyTorch y CUDA. Antes no tenían ninguno, y WSL se cayó
  midiendo el miembro con GPU de los portfolios.

Antes de empezar se comprueba que procesos × (``--memory`` + los márgenes) quepa en el 80 %
de la memoria disponible.

VAL se ejecuta al final, cuando ya no hay búsquedas, uno a uno y con su propio límite duro
(el 80 % de la memoria disponible). Antes heredaba el de la búsqueda y se quedaba sin
memoria en los planes grandes, que contaban como inválidos. Si VAL termina sin veredicto,
el plan queda sin validar (``val_error``), no inválido.
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
#: Memoria de datos de más con PyTorch y CUDA (importar PyTorch son unos 630 MB).
TORCH_MARGIN_MB = 1024
#: Dispositivos que cargan PyTorch.
TORCH_DEVICES = ("cpu", "cuda", "hybrid")


def run_one(domain: str, problem: str, config: str, limit: float, memory: int) -> dict[str, Any]:
    """Resuelve un problema con una configuración (en un proceso aparte)."""
    from pyplanner.search import PRESETS as presets

    if presets[config].device in TORCH_DEVICES:
        limit_bytes = (memory + VIRTUAL_MARGIN_MB + TORCH_MARGIN_MB) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_DATA, (limit_bytes, limit_bytes))
    else:
        limit_bytes = (memory + VIRTUAL_MARGIN_MB) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

    from pyplanner.frontend import read_domain, read_problem
    from pyplanner.grounding.discrete import DiscreteTask
    from pyplanner.planner import solve
    from pyplanner.search import PRESETS, SearchStatus

    start = time.perf_counter()
    loaded = read_domain(domain)
    settings = PRESETS[config].with_changes(max_memory=memory)
    # solve: con preferencias, búsqueda anytime; con procesos y eventos, la validación con
    # un paso más fino (y los reintentos con δ/2), todo dentro del límite de tiempo.
    solved = solve(
        loaded, read_problem(problem, loaded), [settings], time_limit=limit, memory_limit=memory
    )
    elapsed = time.perf_counter() - start
    task, result = solved.task, solved.search
    anytime = task.preferences is not None
    row: dict[str, Any] = {
        "problem": Path(problem).stem,
        "config": config,
        "status": result.status.value,
        "time": round(elapsed, 3),
        "expanded": result.statistics.expanded,
        "dead_ends": result.statistics.dead_ends,
        "valid": None,
    }
    if isinstance(task, DiscreteTask):
        row["delta"] = task.delta
    if result.status is SearchStatus.SOLVED and solved.plan is not None:
        plan = solved.plan
        row["length"] = len(plan)
        row["makespan"] = plan.makespan
        row["metric"] = plan.metric
        if anytime:
            row["optimal"] = result.optimal
        row["plan"] = plan.to_ipc()  # se valida al final (ver validate)
    return row


def validate(row: dict[str, Any], domain: str, problem: str, memory: int) -> None:
    """Valida con VAL el plan de ``row`` (y lo quita de la fila).

    Se llama desde el proceso principal cuando ya no hay búsquedas, con un límite duro de
    ``memory`` MB: si VAL lo heredara de la búsqueda, como antes, podría quedarse sin
    memoria (storage p20 necesita 2,3 GB) y su plan contaría como inválido. Si VAL termina
    sin veredicto (sin memoria o con un fallo de segmentación), el plan queda sin validar
    (``valid`` es ``None``), no inválido.
    """
    plan = row.pop("plan", None)
    if plan is None or not VALIDATE.exists():
        return
    limit = memory * 1024 * 1024
    with tempfile.NamedTemporaryFile("w", suffix=".plan", delete=False) as handle:
        handle.write(plan)
    try:
        process = subprocess.run(
            [str(VALIDATE), domain, problem, handle.name],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
            preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_AS, (limit, limit)),
        )
    except subprocess.TimeoutExpired:
        row["val_error"] = "VAL no terminó en 300 s"
        return
    finally:
        Path(handle.name).unlink()
    output = process.stdout
    if "Plan valid" in output:
        row["valid"] = True
    elif process.returncode in (0, 1):  # 1: plan inválido; 0: "Bad plan file!"
        row["valid"] = False
    else:
        row["val_error"] = f"VAL terminó con código {process.returncode}"
    value = re.search(r"Final value: (\S+)", output)
    row["val_metric"] = float(value.group(1)) if value else None


def _domain_of(problem: Path) -> Path:
    """El dominio de un problema: el suyo propio (``domain-pNN.pddl``) o el común."""
    own = problem.with_name(f"domain-{problem.name}")
    return own if own.exists() else problem.with_name("domain.pddl")


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
    from pyplanner.search import PRESETS
    from pyplanner.search.memory import available_memory_mb

    available = available_memory_mb()
    margin = VIRTUAL_MARGIN_MB
    if any(PRESETS[c].device in TORCH_DEVICES for c in args.configs):
        margin += TORCH_MARGIN_MB
    needed = args.workers * (args.memory + margin)
    if available is None or needed > 0.8 * available:
        parser.error(
            f"{args.workers} procesos × ({args.memory} + {margin}) MB = {needed} MB no "
            f"caben en el 80% de la memoria disponible ({available} MB): baja -w o -m"
        )

    problems = sorted(args.directory.glob("p[0-9]*.pddl"))
    if args.problems:
        problems = [p for p in problems if p.stem in set(args.problems)]
    jobs = [
        (str(_domain_of(problem)), str(problem), config, args.time, args.memory)
        for problem in problems
        for config in args.configs
    ]
    context = multiprocessing.get_context("spawn")
    rows: list[dict[str, Any]] = []
    # Un proceso nuevo por trabajo: uno reutilizado conserva la memoria de la búsqueda
    # anterior (llegaría antes a su límite) y no puede subir su límite duro.
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=context, max_tasks_per_child=1
    ) as pool:
        futures = [pool.submit(run_one, *job) for job in jobs]
        for job, future in zip(jobs, futures, strict=True):
            try:
                row = future.result()
            except Exception as error:  # noqa: BLE001 - se informa y se sigue
                row = {"problem": Path(job[1]).stem, "config": job[2], "status": "error",
                       "time": None, "error": str(error)}  # fmt: skip
            rows.append(row)
            quality = ""
            if row.get("metric") is not None:
                quality = f" métrica={row['metric']:g}"
                if row.get("optimal"):
                    quality += " óptimo"
            if row.get("delta") not in (None, 1.0):
                quality += f" δ={row['delta']:g}"
            if row.get("error"):
                quality = f" {row['error'] or 'MemoryError'}"
            print(
                f"{row['problem']:5} {row['config']:22} {row['status']:14} "
                f"t={row['time'] if row['time'] is not None else '-':>8} "
                f"exp={row.get('expanded', '-'):>8} "
                f"callejones={row.get('dead_ends', '-'):>7}{quality}",
                flush=True,
            )

    # VAL, cuando ya no hay búsquedas: así no compite con ellas por la memoria.
    if VALIDATE.exists():
        print("\nValidación con VAL (solo los planes no válidos o con otra métrica):")
        for job, row in zip(jobs, rows, strict=True):
            validate(row, job[0], job[1], int(0.8 * available))
            name = f"  {row['problem']:5} {row['config']:22}"
            metric, val_metric = row.get("metric"), row.get("val_metric")
            if row.get("val_error"):
                print(f"{name} sin validar: {row['val_error']}")
            elif row.get("valid") is False:
                print(f"{name} INVÁLIDO")
            elif metric is not None and val_metric is not None and abs(val_metric - metric) > 1e-3:
                print(f"{name} métrica {metric:g}, VAL {val_metric:g}")
    for row in rows:
        row.pop("plan", None)

    print("\nResumen (los no resueltos cuentan el límite de tiempo):")
    for config in args.configs:
        mine = [r for r in rows if r["config"] == config]
        solved = [r for r in mine if r["status"] == "solved" and r.get("valid") is not False]
        total = sum(r["time"] if r in solved else args.time for r in mine)
        invalid = sum(r.get("valid") is False for r in mine)
        unchecked = sum(bool(r.get("val_error")) for r in mine)
        note = f", {invalid} planes inválidos" if invalid else ""
        note += f", {unchecked} sin validar" if unchecked else ""
        if any("optimal" in r for r in solved):
            metric = sum(r["metric"] for r in solved if r.get("metric") is not None)
            optimal = sum(bool(r.get("optimal")) for r in solved)
            note += f", métrica total {metric:g}, {optimal} óptimos"
        print(f"  {config:22} {len(solved)}/{len(mine)} resueltos, {total:.1f} s{note}")
    if args.output:
        args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
