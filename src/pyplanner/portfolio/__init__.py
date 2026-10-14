"""Portfolio paralelo: varias configuraciones de búsqueda compitiendo en procesos separados.

Cada configuración se ejecuta en su propio proceso, porque el GIL de Python impide que los
hilos ejecuten búsquedas en paralelo. Gana la **primera** que encuentra un plan: el
portfolio activa una señal de parada compartida y los demás procesos terminan en cuanto la
ven (o se les fuerza a terminar si tardan).

Arranque de los procesos:

* ``fork`` (por defecto en Linux): los hijos heredan la tarea instanciada sin copiarla.
* ``spawn``: los hijos arrancan un intérprete nuevo y reciben la tarea serializada. Es más
  lento de arrancar (unos segundos con PyTorch), pero seguro. Se usa:

  * para los miembros que evalúan en ``"cuda"`` o ``"hybrid"``, porque CUDA no se puede
    usar en un proceso creado con ``fork`` (el proceso padre nunca inicializa CUDA). Los
    demás miembros siguen arrancando con ``fork``: así un miembro con GPU no retrasa a los
    de CPU, que resuelven las tareas pequeñas en décimas de segundo;
  * para todos, si el proceso padre tiene varios hilos (de Python o del sistema, como los
    de HiGHS u OpenBLAS tras resolver un programa lineal) o ya ha cargado PyTorch, porque
    hacer ``fork`` de un proceso con varios hilos puede bloquear al hijo.

  La señal de parada y la cola de resultados se crean con ``spawn`` si algún miembro lo
  usa: sus semáforos tienen nombre y sirven también a los hijos de ``fork`` (al revés no).

  Con ``spawn``, cada hijo vuelve a importar el script principal del padre: si ese script
  llama al planificador fuera de ``if __name__ == "__main__":``, el hijo se encontraría la
  llamada a mitad de su propio arranque. El portfolio lo detecta y lanza un error (como
  hace ``multiprocessing`` al arrancar procesos): el hijo termina sin buscar y el padre
  informa de la causa.
  Con Python embebido en otro programa, además, ``spawn`` lanza ``sys.executable``, que
  tiene que ser un intérprete de Python (``multiprocessing.set_executable``).

**Miembros que fallan:** si un proceso no se puede arrancar o termina sin informar (un
hijo de ``spawn`` que no llega a buscar, uno que muere por falta de memoria…), su resultado
es ``ERROR`` con la causa en cuanto se detecta, y el portfolio sigue con los demás. Ningún
fallo queda solo en la salida de error de un hilo o de un hijo.

Resultado: el plan del ganador, o, si ninguno lo encuentra, el estado más informativo.
Por orden de prioridad: ``UNSOLVABLE`` si alguno lo ha demostrado; si no, ``LIMIT_REACHED``;
si no, ``FAILED`` (una búsqueda incompleta que se rindió); si no, ``ERROR``.

**Modo anytime** (``anytime=True``, para las tareas con preferencias): cada miembro hace una
búsqueda *anytime* (:mod:`pyplanner.search.anytime`) que empieza por su configuración, e
informa de cada plan mejor que encuentra. Todos comparten la mejor cota, así que cada uno
solo busca planes mejores que el mejor de todos. Termina cuando se agota el tiempo, cuando
todos han terminado o cuando uno demuestra que el mejor plan es óptimo; el resultado es el
mejor plan.
"""

import multiprocessing as mp
import os
import queue
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from multiprocessing.context import ForkContext, ForkServerContext, SpawnContext
from typing import Any

from pyplanner.grounding import GroundTask
from pyplanner.heuristics import numba_available
from pyplanner.search import SearchConfig, SearchResult, SearchStatus, search
from pyplanner.search.anytime import SharedBound, search_anytime

#: Segundos que se espera a que un proceso cancelado termine antes de forzarlo.
JOIN_TIMEOUT = 2.0
#: Lo mismo cuando ya hay un plan: basta con que los demás paren pronto.
WINNER_JOIN_TIMEOUT = 0.05

#: Cada cuántos segundos se consulta la cola de resultados.
POLL_INTERVAL = 0.05

#: Causa probable cuando un hijo de ``spawn`` termina con código 1 sin informar: ``_worker``
#: informa de cualquier excepción de la búsqueda, así que falló al arrancar.
SPAWN_HINT = (
    ". Con el arranque 'spawn', cada proceso vuelve a importar el script principal: la "
    "llamada al planificador tiene que ir dentro de 'if __name__ == \"__main__\":' (y, con "
    "Python embebido, multiprocessing.set_executable tiene que apuntar a un intérprete de "
    "Python)"
)


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
    anytime: bool = False,
) -> PortfolioResult:
    """Ejecuta las configuraciones en paralelo y devuelve la primera solución (o, con
    ``anytime``, la mejor que encuentren en el tiempo dado).

    Args:
        task: tarea instanciada.
        configs: configuraciones del portfolio (nombres distintos).
        time_limit: límite de tiempo global en segundos.
        start_method: ``"fork"`` o ``"spawn"``; por defecto se elige según los dispositivos.
        anytime: modo anytime (ver la documentación del módulo).
    """
    if getattr(mp.current_process(), "_inheriting", False):
        # Un hijo de spawn que importa el script principal del padre y se encuentra la
        # llamada al planificador: como hace el propio multiprocessing, se para aquí (si
        # no, cada hijo repetiría el trabajo del script antes de buscar).
        raise RuntimeError(
            "el planificador se ha llamado al importar el script principal en un proceso "
            "hijo de 'spawn': la llamada tiene que ir dentro de 'if __name__ == \"__main__\":'"
        )
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
    contexts = {c.name: _context(c, configs, start_method) for c in configs}
    shared = (
        mp.get_context("spawn")
        if any(ctx.get_start_method() == "spawn" for ctx in contexts.values())
        else next(iter(contexts.values()))
    )
    stop = shared.Event()
    results_queue = shared.Queue()
    bound = shared.Value("d", float("inf")) if anytime else None
    processes = {
        c.name: contexts[c.name].Process(
            target=_anytime_worker if anytime else _worker,
            args=(task, c, stop, results_queue, *((bound,) if anytime else ())),
            name=f"pyplanner-{c.name}",
            daemon=True,
        )
        for c in configs
    }
    start = time.perf_counter()
    spawned = {name for name, ctx in contexts.items() if ctx.get_start_method() != "fork"}
    members = _Members(processes, spawned)
    members.start()
    if anytime:
        return _collect_anytime(members, stop, results_queue, start, time_limit)

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
                results.update(members.failed(results))
                if not members.running():
                    break  # todos han terminado; lo que quede en la cola se recoge abajo
                continue
            results[result.config_name] = result
            if result.solved:
                winner = result
    finally:
        stop.set()
        members.starter.join()
        # Con un ganador, los demás solo pueden aportar resultados descartables: se les da
        # un margen corto (un miembro con GPU puede estar aún importando PyTorch).
        _shutdown(processes.values(), WINNER_JOIN_TIMEOUT if winner else JOIN_TIMEOUT)
    while winner is None:  # resultados enviados justo antes de terminar
        try:
            result = results_queue.get(timeout=POLL_INTERVAL)
        except queue.Empty:
            break
        results[result.config_name] = result
        if result.solved:
            winner = result

    elapsed = time.perf_counter() - start
    for name in processes:
        if name not in results:
            results[name] = members.final(name, cancelled=winner is not None)
    return PortfolioResult(winner or _most_informative(results), results, elapsed)


class _Members:
    """Los procesos del portfolio: los arranca y sabe cuáles ya no van a informar.

    Los de ``fork`` arrancan al instante. Los de ``spawn`` (GPU) se arrancan en un hilo: hay
    que serializarles la tarea, y mientras tanto el padre ya puede recoger los resultados de
    los demás (en las tareas pequeñas, un miembro de CPU gana antes de que acaben). Si uno
    no se puede arrancar, el error se guarda y se informa como su resultado.
    """

    def __init__(self, processes: dict[str, Any], spawned: set[str]) -> None:
        self.processes = processes
        self.spawned = spawned
        self.start_errors: dict[str, str] = {}
        self.starter = threading.Thread(target=self._start_spawned, daemon=True)

    def start(self) -> None:
        for name, process in self.processes.items():
            if name not in self.spawned:
                process.start()
        self.starter.start()

    def _start_spawned(self) -> None:
        for name, process in self.processes.items():
            if name in self.spawned:
                try:
                    process.start()
                except Exception as exc:  # noqa: BLE001 - se informa como resultado del miembro
                    self.start_errors[name] = f"{type(exc).__name__}: {exc}"

    def running(self) -> bool:
        """``True`` si alguno puede informar todavía (o está arrancando)."""
        return self.starter.is_alive() or any(p.is_alive() for p in self.processes.values())

    def failed(self, reported: dict[str, SearchResult]) -> dict[str, SearchResult]:
        """Resultado ``ERROR`` de los que no han informado y ya no lo harán: no se pudieron
        arrancar o han terminado. Si informaron justo antes de terminar, su resultado real
        sustituye a este al recogerlo."""
        return {
            name: self._error(name)
            for name, process in self.processes.items()
            if name not in reported and (name in self.start_errors or process.exitcode is not None)
        }

    def final(self, name: str, cancelled: bool) -> SearchResult:
        """Resultado de un miembro que no ha informado cuando el portfolio termina:
        cancelado (con un ganador, o parado por el portfolio) o ``ERROR``."""
        exitcode = self.processes[name].exitcode
        if name not in self.start_errors and (cancelled or exitcode in (0, -15)):
            return SearchResult(SearchStatus.CANCELLED, config_name=name)
        return self._error(name)

    def _error(self, name: str) -> SearchResult:
        if name in self.start_errors:
            message = f"no se pudo arrancar el proceso ({self.start_errors[name]})"
        else:
            exitcode = self.processes[name].exitcode
            message = f"el proceso terminó sin informar (código {exitcode})"
            if name in self.spawned and exitcode == 1:
                message += SPAWN_HINT
        return SearchResult(SearchStatus.ERROR, config_name=name, message=message)


def _collect_anytime(
    members: _Members,
    stop: Any,
    results_queue: Any,
    start: float,
    time_limit: float | None,
) -> PortfolioResult:
    """Recoge los planes de los miembros en modo anytime y se queda con el mejor."""
    processes = members.processes
    results: dict[str, SearchResult] = {}  # resultado final de cada miembro
    best: SearchResult | None = None
    lower: float | None = None  # cota inferior demostrada del coste de los planes
    deadline = None if time_limit is None else start + time_limit + JOIN_TIMEOUT

    def receive(timeout: float) -> bool:
        """Procesa un mensaje; ``False`` si no había ninguno."""
        nonlocal best, lower
        try:
            kind, member, result = results_queue.get(timeout=timeout)
        except queue.Empty:
            return False
        if kind == "done":
            results[member] = result
        if result.solved and (best is None or result.cost < best.cost):
            best = replace(result, lower_bound=lower)
        if result.lower_bound is not None and (lower is None or result.lower_bound > lower):
            lower = result.lower_bound
            if best is not None:
                best = replace(best, lower_bound=lower)
        return True

    try:
        while len(results) < len(processes):
            if deadline is not None and time.perf_counter() > deadline:
                break
            if best is not None and best.optimal:
                break
            if not receive(POLL_INTERVAL):
                results.update(members.failed(results))
                if not members.running():
                    break
    finally:
        stop.set()
        members.starter.join()
        _shutdown(processes.values(), JOIN_TIMEOUT)
    while receive(POLL_INTERVAL):  # mensajes enviados justo antes de terminar
        pass

    elapsed = time.perf_counter() - start
    for name in processes:
        if name not in results:
            results[name] = members.final(name, cancelled=False)
    return PortfolioResult(best or _most_informative(results), results, elapsed)


def _anytime_worker(
    task: GroundTask, config: SearchConfig, stop: Any, results_queue: Any, bound: SharedBound
) -> None:
    """Punto de entrada de cada proceso en modo anytime: informa de cada plan mejor y, al
    terminar, del resultado final."""
    try:
        result = search_anytime(
            task,
            config,
            stop=stop,
            on_plan=lambda plan: results_queue.put(("plan", config.name, plan)),
            shared_bound=bound,
        )
    except Exception as exc:  # noqa: BLE001 - se informa al padre en lugar de perderse
        result = SearchResult(
            SearchStatus.ERROR, config_name=config.name, message=f"{type(exc).__name__}: {exc}"
        )
    results_queue.put(("done", config.name, result))


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
    config: SearchConfig, configs: Sequence[SearchConfig], start_method: str | None
) -> ForkContext | SpawnContext | ForkServerContext:
    """Modo de arranque de un miembro: el indicado, o ``fork`` si es seguro para él."""
    if start_method is None:
        start_method = "fork" if _fork_is_safe(config) else "spawn"
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


def _fork_is_safe(config: SearchConfig) -> bool:
    """``True`` si se puede arrancar el miembro con ``fork`` sin riesgo (ver la
    documentación del módulo)."""
    return (
        os.name == "posix"
        and config.device not in ("cuda", "hybrid")
        and "torch" not in sys.modules
        and threading.active_count() == 1
        and _os_threads() <= 1
    )


def _os_threads() -> int:
    """Hilos del proceso según el sistema operativo (no solo los de Python): bibliotecas
    como HiGHS (programación lineal) u OpenBLAS crean los suyos, y un ``fork`` con ellos
    vivos puede bloquear al hijo. Pasa si el proceso ya ha resuelto una tarea con
    programación lineal (al usar el planificador como librería). Sin ``/proc``, 1."""
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("Threads:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return 1


def _shutdown(processes: Any, timeout: float) -> None:
    """Espera a que terminen los procesos y, pasado ``timeout``, los fuerza (los que no se
    llegaron a arrancar no cuentan)."""
    processes = [p for p in processes if p.pid is not None]
    deadline = time.perf_counter() + timeout
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.perf_counter()))
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join()


def _most_informative(results: dict[str, SearchResult]) -> SearchResult:
    for status in (
        SearchStatus.UNSOLVABLE,
        SearchStatus.LIMIT_REACHED,
        SearchStatus.FAILED,
        SearchStatus.ERROR,
    ):
        for result in results.values():
            if result.status is status:
                return result
    return next(iter(results.values()))


__all__ = ["PortfolioResult", "run_portfolio"]
