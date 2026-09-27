import json
from pathlib import Path

import pytest

from pyplanner import __version__
from pyplanner.cli import EXIT_INVALID_INPUT, main
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
