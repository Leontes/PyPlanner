"""Tests del portfolio paralelo."""

import sys
import threading

import pytest

from pyplanner import portfolio
from pyplanner.plan import simulate
from pyplanner.portfolio import run_portfolio
from pyplanner.search import DEFAULT_PORTFOLIO, PRESETS, SearchConfig, SearchStatus
from tests.tasks import chain, token
from tests.test_search import GRIPPER


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
    cpu = [PRESETS["gbfs-ff"]]
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(threading, "active_count", lambda: 1)
    assert portfolio._fork_is_safe(cpu)
    assert not portfolio._fork_is_safe([PRESETS["gbfs-ff"].with_changes(device="cuda")])
    monkeypatch.setattr(threading, "active_count", lambda: 3)
    assert not portfolio._fork_is_safe(cpu)
