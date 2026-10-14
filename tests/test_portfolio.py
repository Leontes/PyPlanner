"""Tests del portfolio paralelo."""

import os
import subprocess
import sys
import textwrap
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from pyplanner import portfolio
from pyplanner.grounding import GroundTask
from pyplanner.heuristics import numba_available
from pyplanner.plan import simulate
from pyplanner.portfolio import run_portfolio
from pyplanner.search import (
    DEFAULT_PORTFOLIO,
    PRESETS,
    SearchConfig,
    SearchResult,
    SearchStatus,
)
from tests.tasks import chain, token
from tests.test_search import GRIPPER

ROOT = Path(__file__).resolve().parent.parent


def test_first_solution_wins_and_the_rest_are_cancelled() -> None:
    configs = [PRESETS[n] for n in DEFAULT_PORTFOLIO]
    result = run_portfolio(GRIPPER, configs)
    assert result.best.solved
    assert result.best.plan is not None and simulate(GRIPPER, result.best.plan) is None
    assert set(result.results) == set(DEFAULT_PORTFOLIO)
    solved = [r for r in result.results.values() if r.solved]
    assert result.best in solved


def test_slow_member_is_cancelled() -> None:
    slow = SearchConfig("slow", "astar", "blind", max_expansions=None)
    fast = PRESETS["gbfs-ff-pref"]
    # La tarea es pequeña: el miembro lento puede terminar a tiempo, pero no ganar
    # siempre. Lo que se comprueba es que nunca queda un proceso colgado.
    result = run_portfolio(GRIPPER, [slow, fast])
    assert result.best.solved
    assert {r.status for r in result.results.values()} <= {
        SearchStatus.SOLVED,
        SearchStatus.CANCELLED,
    }


def test_unsolvable_task() -> None:
    result = run_portfolio(token(), [PRESETS["gbfs-ff"], PRESETS["astar-blind"]])
    assert result.best.status is SearchStatus.UNSOLVABLE
    assert all(r.status is SearchStatus.UNSOLVABLE for r in result.results.values())


def test_time_limit_is_applied_to_every_member() -> None:
    configs = [
        PRESETS["astar-blind"].with_changes(max_expansions=0),
        PRESETS["gbfs-ff"].with_changes(max_expansions=0),
    ]
    result = run_portfolio(GRIPPER, configs, time_limit=5)
    assert result.best.status is SearchStatus.LIMIT_REACHED


def test_errors_in_members_are_reported() -> None:
    # Se fuerza un error en el hijo con un dispositivo que no existe.
    broken = SearchConfig("broken", "gbfs", "ff")
    object.__setattr__(broken, "device", "tpu")  # salta la validación a propósito
    result = run_portfolio(chain(), [broken])
    assert result.best.status is SearchStatus.ERROR
    assert "HeuristicError" in result.best.message


# Aquí se fuerza "fork" aunque el proceso de tests tenga hilos (PyTorch): es a propósito.
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
@pytest.mark.parametrize("start_method", ["fork", "spawn"])
def test_explicit_start_methods(start_method: str) -> None:
    result = run_portfolio(chain(), [PRESETS["gbfs-ff"]], start_method=start_method)
    assert result.best.solved


def test_invalid_portfolios() -> None:
    with pytest.raises(ValueError, match="al menos una"):
        run_portfolio(chain(), [])
    with pytest.raises(ValueError, match="repetidos"):
        run_portfolio(chain(), [PRESETS["gbfs-ff"], PRESETS["gbfs-ff"]])
    with pytest.raises(ValueError, match="método de arranque"):
        run_portfolio(chain(), [PRESETS["gbfs-ff"]], start_method="thread")


def test_fork_is_only_used_when_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    cpu = PRESETS["gbfs-ff"]
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(threading, "active_count", lambda: 1)
    monkeypatch.setattr(portfolio, "_os_threads", lambda: 1)
    assert portfolio._fork_is_safe(cpu)
    assert not portfolio._fork_is_safe(cpu.with_changes(device="cuda"))
    assert not portfolio._fork_is_safe(cpu.with_changes(device="hybrid"))
    monkeypatch.setattr(threading, "active_count", lambda: 3)
    assert not portfolio._fork_is_safe(cpu)
    monkeypatch.setattr(threading, "active_count", lambda: 1)
    monkeypatch.setattr(portfolio, "_os_threads", lambda: 12)  # hilos de HiGHS u OpenBLAS
    assert not portfolio._fork_is_safe(cpu)


@pytest.mark.skipif(not numba_available(), reason="Numba no está instalado")
def test_gpu_members_do_not_slow_down_cpu_members() -> None:
    """Un miembro híbrido arranca con spawn y los demás con fork: comparten la señal de
    parada y la cola de resultados, y el de CPU gana en una tarea trivial."""
    pytest.importorskip("torch")
    configs = [PRESETS["gbfs-ff"], PRESETS["gbfs-ff-pref-hybrid"]]
    result = run_portfolio(chain(), configs, time_limit=60)
    assert result.best.solved
    assert set(result.results) == {"gbfs-ff", "gbfs-ff-pref-hybrid"}


@pytest.mark.skipif(not numba_available(), reason="Numba no está instalado")
def test_numba_kernels_are_warmed_up_in_the_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    from pyplanner.heuristics import compiled

    calls: list[int] = []
    original = compiled.warm_up

    def counting_warm_up() -> None:
        calls.append(1)
        original()

    monkeypatch.setattr(compiled, "warm_up", counting_warm_up)
    run_portfolio(chain(), [PRESETS["gbfs-ff"]])
    assert calls == [1]
    run_portfolio(chain(), [PRESETS["gbfs-ff"].with_changes(device="python")])  # sin Numba
    assert calls == [1]


def test_members_started_in_the_background_are_waited_for(monkeypatch: pytest.MonkeyPatch) -> None:
    """Si ningún miembro puede usar fork (p. ej. PyTorch ya cargado), todos arrancan desde
    un hilo: el portfolio no puede darlos por terminados mientras arrancan."""
    monkeypatch.setattr(portfolio, "_fork_is_safe", lambda config: False)
    result = run_portfolio(chain(), [PRESETS["gbfs-ff"], PRESETS["lazy-ff-pref"]], time_limit=60)
    assert result.best.solved


# --- Miembros que fallan ------------------------------------------------------------------


def test_members_that_cannot_start_are_reported() -> None:
    """Un miembro de spawn cuya tarea no se puede serializar no llega a arrancar: el error
    (que antes se quedaba en el hilo que lo arranca) es su resultado."""
    task = replace(chain(), notes=(lambda: None,))  # type: ignore[arg-type]
    result = run_portfolio(task, [PRESETS["gbfs-ff"]], start_method="spawn")
    assert result.best.status is SearchStatus.ERROR
    assert "no se pudo arrancar el proceso" in result.best.message


def _crashing_worker(task: GroundTask, config: SearchConfig, stop: Any, results_queue: Any) -> None:
    """Sustituye a ``_worker``: "crash" muere sin informar; el otro informa al poco."""
    if config.name == "crash":
        os._exit(3)
    stop.wait(timeout=1)
    results_queue.put(SearchResult(SearchStatus.LIMIT_REACHED, config_name=config.name))


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_members_that_die_without_reporting_are_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(portfolio, "_worker", _crashing_worker)
    configs = [PRESETS["gbfs-ff"].with_changes(name=name) for name in ("crash", "other")]
    result = run_portfolio(chain(), configs, start_method="fork")
    assert result.results["crash"].status is SearchStatus.ERROR
    assert "terminó sin informar (código 3)" in result.results["crash"].message
    assert result.results["other"].status is SearchStatus.LIMIT_REACHED


def test_spawn_without_main_guard_explains_the_cause(tmp_path: Path) -> None:
    """Un script que llama al portfolio sin ``if __name__ == "__main__":``: con spawn, el
    hijo vuelve a importarlo y se encuentra la llamada a mitad de su arranque. Termina con
    código 1 sin buscar (sin repetir el trabajo del script) y el resultado lo explica."""
    script = tmp_path / "unguarded.py"
    script.write_text(
        textwrap.dedent(
            """
            from pyplanner.portfolio import run_portfolio
            from pyplanner.search import PRESETS
            from tests.tasks import chain

            result = run_portfolio(chain(), [PRESETS["gbfs-ff"]], start_method="spawn")
            print("RESULT", result.best.status.value, result.best.message)
            """
        )
    )
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    completed = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=tmp_path,
        check=False,
    )
    [line] = [x for x in completed.stdout.splitlines() if x.startswith("RESULT")]  # el padre
    assert line.startswith("RESULT error")
    assert "código 1" in line and 'if __name__ == "__main__":' in line
    assert "se ha llamado al importar el script principal" in completed.stderr  # el hijo
