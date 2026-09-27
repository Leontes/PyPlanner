import json
from pathlib import Path

import pytest

from pyplanner import __version__, cli
from pyplanner.cli import EXIT_INVALID_INPUT, EXIT_NO_PLAN, EXIT_UNSOLVABLE, main
from tests.samples import DOMAIN, PROBLEM


@pytest.fixture
def task(tmp_path: Path) -> tuple[Path, Path]:
    domain = tmp_path / "domain.pddl"
    problem = tmp_path / "problem.pddl"
    domain.write_text(DOMAIN)
    problem.write_text(PROBLEM)
    return domain, problem


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_requires_a_command() -> None:
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_parse_prints_json(task: tuple[Path, Path], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["parse", "--no-loc", str(task[0])]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["kind"] == "domain"
    assert "loc" not in document["actions"][0]


def test_parse_to_file_and_check_json(
    task: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    domain_json = tmp_path / "domain.json"
    problem_json = tmp_path / "problem.json"
    assert main(["parse", str(task[0]), "-o", str(domain_json)]) == 0
    assert main(["parse", str(task[1]), "-o", str(problem_json), "--compact"]) == 0
    assert main(["check", str(domain_json), str(problem_json)]) == 0
    out = capsys.readouterr().out
    assert "Dominio 'test': 3 predicados, 2 acciones, 2 constantes" in out
    assert "Problema 'test-1': 1 objetos, 2 hechos iniciales" in out


def test_invalid_input_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.pddl"
    bad.write_text("(define (domain d) (:requirements :typing))")
    assert main(["parse", str(bad)]) == EXIT_INVALID_INPUT
    assert "no soportado ':typing'" in capsys.readouterr().err


def test_malformed_json_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{")
    assert main(["check", str(bad), str(bad)]) == EXIT_INVALID_INPUT
    assert "JSON mal formado" in capsys.readouterr().err


def test_missing_file_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["parse", str(tmp_path / "nada.pddl")]) == EXIT_INVALID_INPUT
    assert "error:" in capsys.readouterr().err


def test_solve_prints_ipc_plan(task: tuple[Path, Path], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["solve", "-j", "1", str(task[0]), str(task[1])]) == 0
    captured = capsys.readouterr()
    assert captured.out.endswith("; cost = 2 (unit cost)\n")
    assert "(move home kitchen)" in captured.out
    assert "Núcleos: 1 → 1 búsqueda(s): lazy-ff-pref" in captured.err
    assert "Resultado: solved (lazy-ff-pref)" in captured.err


@pytest.mark.parametrize(("cores", "expected"), [(1, 1), (2, 2), (3, 2), (4, 4), (5, 5), (9, 5)])
def test_solve_selects_portfolio_by_cores(
    task: tuple[Path, Path], capsys: pytest.CaptureFixture[str], cores: int, expected: int
) -> None:
    assert main(["solve", "-j", str(cores), *map(str, task)]) == 0
    assert f"Núcleos: {cores} → {expected} búsqueda(s)" in capsys.readouterr().err


def test_default_cores_are_limited_by_the_machine(
    task: tuple[Path, Path], capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "available_cores", lambda: 2)
    assert main(["solve", *map(str, task)]) == 0
    assert "Núcleos: 2 → 2 búsqueda(s)" in capsys.readouterr().err


def test_asking_for_more_cores_than_available_warns(
    task: tuple[Path, Path], capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "available_cores", lambda: 1)
    assert main(["solve", "-j", "2", *map(str, task)]) == 0
    assert "aviso: se han pedido 2 núcleos, pero la máquina tiene 1" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["-j", "0"], "≥ 1"),
        (["-j", "0", "-c", "gbfs-add"], "≥ 1"),
        (["-j", "1", "-c", "gbfs-add", "-c", "astar-hmax"], "2 configuraciones pero solo 1"),
    ],
)
def test_invalid_core_requests(
    task: tuple[Path, Path], capsys: pytest.CaptureFixture[str], args: list[str], message: str
) -> None:
    assert main(["solve", *args, *map(str, task)]) == EXIT_INVALID_INPUT
    assert message in capsys.readouterr().err


def test_solve_json_to_file_with_manual_portfolio(
    task: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "plan.json"
    args = ["-c", "astar-hmax", "-c", "gbfs-add", "-f", "json", "-o", str(out), "-q"]
    assert main(["solve", *args, *map(str, task)]) == 0
    assert capsys.readouterr().err == ""
    document = json.loads(out.read_text())
    assert document["kind"] == "plan" and document["cost"] == 2


def test_solve_overrides(task: tuple[Path, Path], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["solve", "-c", "astar-hmax", "--batch-size", "4", *map(str, task)]) == 0
    assert "lotes" in capsys.readouterr().err


def test_solve_unsolvable(tmp_path: Path) -> None:
    from tests.tasks import TOKEN_DOMAIN, TOKEN_PROBLEM

    domain, problem = tmp_path / "d.pddl", tmp_path / "p.pddl"
    domain.write_text(TOKEN_DOMAIN)
    problem.write_text(TOKEN_PROBLEM)
    assert main(["solve", "-q", str(domain), str(problem)]) == EXIT_UNSOLVABLE


def test_solve_limit(task: tuple[Path, Path]) -> None:
    code = main(["solve", "-q", "-c", "astar-blind", "-t", "0.000000001", *map(str, task)])
    assert code == EXIT_NO_PLAN


def test_configs_lists_presets(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["configs"]) == 0
    out = capsys.readouterr().out
    assert "lazy-ff-pref" in out and "[núcleos 1,2,4,5]" in out
    assert "astar-hmax" in out and "óptima" in out
    assert "1: lazy-ff-pref" in out
