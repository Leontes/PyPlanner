import json
from pathlib import Path

import pytest

from pyplanner import __version__
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
    assert main(["solve", str(task[0]), str(task[1])]) == 0
    captured = capsys.readouterr()
    assert captured.out.endswith("; cost = 2 (unit cost)\n")
    assert "(move home kitchen)" in captured.out
    assert "Resultado: solved (gbfs-ff-pref)" in captured.err


def test_solve_json_to_file_with_portfolio(
    task: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "plan.json"
    code = main(
        ["solve", "-p", "-c", "astar-hmax", "-f", "json", "-o", str(out), "-q", *map(str, task)]
    )
    assert code == 0
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
    assert "gbfs-ff-pref" in out and "[portfolio, por defecto]" in out
    assert "astar-hmax" in out and "óptima" in out
