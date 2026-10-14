"""Límite duro de memoria (``RLIMIT_DATA``): el cálculo, su efecto en un proceso, la
búsqueda que agota la memoria, los miembros del portfolio y la CLI."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from pyplanner import cli, portfolio
from pyplanner.cli import EXIT_NO_PLAN, main
from pyplanner.portfolio import run_portfolio
from pyplanner.search import PRESETS, SearchStatus, memory, search
from pyplanner.search import engine as engine_module
from pyplanner.search.memory import hard_limit_mb
from tests.samples import DOMAIN, PROBLEM
from tests.tasks import chain

ROOT = Path(__file__).resolve().parent.parent
linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="RLIMIT_DATA de Linux")


def test_hard_limit_is_above_the_soft_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory, "data_memory_mb", lambda: 100.0)
    assert hard_limit_mb(1000) == 1000 + 512  # margen mínimo
    assert hard_limit_mb(4000) == 4000 + 1000  # el 25 %
    assert hard_limit_mb(1000, cuda=True) == 1000 + 512 + 512
    monkeypatch.setattr(memory, "data_memory_mb", lambda: 900.0)
    assert hard_limit_mb(200) == 900 + 512  # lo que ya ocupa el proceso, si es más


def run_python(code: str) -> str:
    """Ejecuta ``code`` en otro proceso (para no limitar el de los tests)."""
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        check=True,
    )
    return completed.stdout


@linux_only
def test_limit_process_makes_large_allocations_fail() -> None:
    output = run_python(
        """
        import resource
        from pyplanner.search.memory import limit_process
        assert limit_process(300)
        try:
            data = bytearray(1024**3)
        except MemoryError:
            print("MemoryError")
        print(resource.getrlimit(resource.RLIMIT_DATA)[0] // 2**20)
        """
    )
    assert output.split() == ["MemoryError", "300"]


@linux_only
def test_limit_process_never_raises_a_lower_limit() -> None:
    output = run_python(
        """
        import resource
        from pyplanner.search.memory import limit_process
        limit_process(300)
        limit_process(1000)
        print(resource.getrlimit(resource.RLIMIT_DATA)[0] // 2**20)
        """
    )
    assert output.split() == ["300"]


def _out_of_memory(self: Any) -> Any:
    raise MemoryError


def test_running_out_of_memory_is_a_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_module.EagerSearch, "run", _out_of_memory)
    result = search(chain(), PRESETS["gbfs-ff"])
    assert result.status is SearchStatus.LIMIT_REACHED
    assert "límite de memoria" in result.message and "límite duro" in result.message


def _allocate(self: Any) -> Any:
    """Sustituye a la búsqueda: reserva 2 GB (posible sin límite en la máquina de los tests)."""
    data = bytearray(2 * 1024**3)
    raise AssertionError(f"no hay límite duro: se han reservado {len(data)} bytes")


@linux_only
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_portfolio_members_have_a_hard_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Con ``max_memory`` = 200 MB, el límite duro del miembro queda muy por debajo de 2 GB:
    la reserva falla y la búsqueda termina con ``LIMIT_REACHED``."""
    monkeypatch.setattr(engine_module.EagerSearch, "run", _allocate)
    config = PRESETS["gbfs-ff"].with_changes(max_memory=200)
    result = run_portfolio(chain(), [config], start_method="fork")
    assert result.best.status is SearchStatus.LIMIT_REACHED, result.best.message
    assert "límite duro" in result.best.message


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_running_out_of_memory_before_searching(monkeypatch: pytest.MonkeyPatch) -> None:
    """Si se agota al preparar la búsqueda (al cargar PyTorch o la tarea), también es un
    límite, no un error."""

    def fail(config: Any) -> None:
        raise MemoryError

    monkeypatch.setattr(portfolio, "_limit_memory", fail)
    result = run_portfolio(chain(), [PRESETS["gbfs-ff"]], start_method="fork")
    assert result.best.status is SearchStatus.LIMIT_REACHED
    assert "al preparar la búsqueda" in result.best.message


@pytest.fixture
def task_files(tmp_path: Path) -> tuple[Path, Path]:
    domain, problem = tmp_path / "domain.pddl", tmp_path / "problem.pddl"
    domain.write_text(DOMAIN)
    problem.write_text(PROBLEM)
    return domain, problem


def test_the_cli_limits_itself_before_grounding(
    task_files: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Con un total grande, para que el límite blando no dependa de lo que ya ocupe el
    proceso de los tests."""
    limits: list[float] = []
    monkeypatch.setattr(cli, "limit_process", limits.append)
    domain, problem = task_files
    assert main(["solve", "-q", "-c", "gbfs-ff", "-m", "100000", str(domain), str(problem)]) == 0
    [limit] = limits
    assert limit == pytest.approx(100000 * 1.25)


def test_the_cli_reports_running_out_of_memory(
    task_files: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def solve(*args: Any, **kwargs: Any) -> Any:
        raise MemoryError

    monkeypatch.setattr(cli, "limit_process", lambda megabytes: True)
    monkeypatch.setattr(cli, "solve", solve)
    domain, problem = task_files
    code = main(["solve", "-q", "-c", "gbfs-ff", "-m", "300", str(domain), str(problem)])
    assert code == EXIT_NO_PLAN
    assert "se ha agotado la memoria permitida (300 MB)" in capsys.readouterr().err
