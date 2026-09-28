"""Portfolio paralelo: varias configuraciones de búsqueda compitiendo en procesos separados.

Cada configuración se ejecuta en su propio proceso, porque el GIL de Python impide que los
hilos ejecuten búsquedas en paralelo. Gana la **primera** que encuentra un plan: el
portfolio activa una señal de parada compartida y los demás procesos terminan en cuanto la
ven (o se les fuerza a terminar si tardan).

Arranque de los procesos:

* ``fork`` (por defecto en Linux): los hijos heredan la tarea instanciada sin copiarla.
* ``spawn``: los hijos arrancan un intérprete nuevo y reciben la tarea serializada. Es más
  lento de arrancar, pero seguro. Se usa si:

  * algún miembro evalúa en ``"cuda"``, porque CUDA no se puede usar en un proceso creado
    con ``fork`` (el proceso padre nunca inicializa CUDA);
  * el proceso padre tiene varios hilos o ya ha cargado PyTorch (que crea los suyos),
    porque hacer ``fork`` de un proceso con varios hilos puede bloquear al hijo.

Resultado: el plan del ganador, o, si ninguno lo encuentra, el estado más informativo.
Por orden de prioridad: ``UNSOLVABLE`` si alguno lo ha demostrado; si no, ``LIMIT_REACHED``;
si no, ``ERROR``.
"""

import multiprocessing as mp
import os
import queue
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from multiprocessing.context import ForkContext, ForkServerContext, SpawnContext
from typing import Any

from pyplanner.grounding import GroundTask
from pyplanner.heuristics import numba_available
from pyplanner.search import SearchConfig, SearchResult, SearchStatus, search

#: Segundos que se espera a que un proceso cancelado termine antes de forzarlo.
JOIN_TIMEOUT = 2.0

#: Cada cuántos segundos se consulta la cola de resultados.
POLL_INTERVAL = 0.05


@dataclass
class PortfolioResult:
    """Resultado del portfolio.

    Attributes:
        best: resultado del ganador (o el más informativo si nadie encontró plan).
        results: resultado de cada configuración que llegó a informar, por nombre.
        time: tiempo total.
    """

    best: SearchResult
    results: dict[str, SearchResult] = field(default_factory=dict)
    time: float = 0.0


def run_portfolio(
    task: GroundTask,
    configs: Sequence[SearchConfig],
    time_limit: float | None = None,
    start_method: str | None = None,
) -> PortfolioResult:
    """Ejecuta las configuraciones en paralelo y devuelve la primera solución.

    Args:
        task: tarea instanciada.
        configs: configuraciones del portfolio (nombres distintos).
        time_limit: límite de tiempo global en segundos.
        start_method: ``"fork"`` o ``"spawn"``; por defecto se elige según los dispositivos.
    """
    names = [c.name for c in configs]
    if not configs:
        raise ValueError("el portfolio necesita al menos una configuración")
    if len(set(names)) != len(names):
        raise ValueError(f"nombres de configuración repetidos en el portfolio: {names}")
    if time_limit is not None:
        configs = [
            c if c.max_time is not None else c.with_changes(max_time=time_limit) for c in configs
        ]

    _warm_up_compiled(configs)
    context = _context(configs, start_method)
    stop = context.Event()
    results_queue = context.Queue()
    processes = {
        c.name: context.Process(
            target=_worker,
            args=(task, c, stop, results_queue),
            name=f"pyplanner-{c.name}",
            daemon=True,
        )
        for c in configs
    }
    start = time.perf_counter()
    for process in processes.values():
        process.start()

    results: dict[str, SearchResult] = {}
    winner: SearchResult | None = None
    deadline = None if time_limit is None else start + time_limit + JOIN_TIMEOUT
    try:
        while len(results) < len(processes) and winner is None:
            if deadline is not None and time.perf_counter() > deadline:
                break
            try:
                result: SearchResult = results_queue.get(timeout=POLL_INTERVAL)
            except queue.Empty:
                if not any(p.is_alive() for p in processes.values()):
                    break  # todos han terminado; lo que quede en la cola se recoge abajo
                continue
            results[result.config_name] = result
            if result.solved:
                winner = result
    finally:
        stop.set()
        _shutdown(processes.values())
    while winner is None:  # resultados enviados justo antes de terminar
        try:
            result = results_queue.get(timeout=POLL_INTERVAL)
        except queue.Empty:
            break
        results[result.config_name] = result
        if result.solved:
            winner = result

    elapsed = time.perf_counter() - start
    for name, process in processes.items():
        if name not in results:
            crashed = winner is None and process.exitcode not in (0, -15)
            status = SearchStatus.ERROR if crashed else SearchStatus.CANCELLED
            message = (
                f"el proceso terminó sin informar (código {process.exitcode})" if crashed else ""
            )
            results[name] = SearchResult(status, config_name=name, message=message)
    return PortfolioResult(winner or _most_informative(results), results, elapsed)


def _worker(task: GroundTask, config: SearchConfig, stop: Any, results_queue: Any) -> None:
    """Punto de entrada de cada proceso: busca e informa del resultado."""
    try:
        result = search(task, config, stop=stop)
    except Exception as exc:  # noqa: BLE001 - se informa al padre en lugar de perderse
        result = SearchResult(
            SearchStatus.ERROR, config_name=config.name, message=f"{type(exc).__name__}: {exc}"
        )
    results_queue.put(result)


def _context(
    configs: Sequence[SearchConfig], start_method: str | None
) -> ForkContext | SpawnContext | ForkServerContext:
    if start_method is None:
        start_method = "fork" if _fork_is_safe(configs) else "spawn"
    if start_method == "fork":
        return mp.get_context("fork")
    if start_method == "spawn":
        return mp.get_context("spawn")
    if start_method == "forkserver":
        return mp.get_context("forkserver")
    raise ValueError(f"método de arranque desconocido: '{start_method}'")


def _warm_up_compiled(configs: Sequence[SearchConfig]) -> None:
    """Precompila las heurísticas de Numba en el padre si algún miembro las va a usar.

    Si no, un hijo cancelado a mitad de compilar dejaría la caché incompleta, y todas las
    ejecuciones pagarían la compilación y la espera del cierre.
    """
    if all(c.device == "python" for c in configs):
        return  # nadie usa código compilado (ni heurística ni sucesores)
    if numba_available():
        from pyplanner.heuristics import compiled
        from pyplanner.search import compiled_successors

        compiled.warm_up()
        compiled_successors.warm_up()


def _fork_is_safe(configs: Sequence[SearchConfig]) -> bool:
    """``True`` si se puede usar ``fork`` sin riesgo de bloqueos (ver la documentación)."""
    return (
        os.name == "posix"
        and not any(c.device == "cuda" for c in configs)
        and "torch" not in sys.modules
        and threading.active_count() == 1
    )


def _shutdown(processes: Any) -> None:
    deadline = time.perf_counter() + JOIN_TIMEOUT
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.perf_counter()))
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join()


def _most_informative(results: dict[str, SearchResult]) -> SearchResult:
    for status in (SearchStatus.UNSOLVABLE, SearchStatus.LIMIT_REACHED, SearchStatus.ERROR):
        for result in results.values():
            if result.status is status:
                return result
    return next(iter(results.values()))


__all__ = ["PortfolioResult", "run_portfolio"]
