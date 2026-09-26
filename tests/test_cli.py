import pytest

from pyplanner import __version__
from pyplanner.cli import main


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_requires_domain_and_problem() -> None:
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
