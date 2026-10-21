"""Evaluación por tiers del planificador completo: plan encontrado y validado con VAL.

Cada instancia de los tiers soportados se resuelve con el portfolio de cada número de
núcleos (de 1 a 7), y el plan se valida con VAL (``tools/bin/Validate``, ver
``scripts/build_val.sh``). Si el plan informa de la métrica, tiene que coincidir con la de
VAL. Si VAL no está compilado, estos tests se saltan.
"""

import re
import subprocess
from pathlib import Path

import pytest

from pyplanner.frontend import read_domain, read_problem
from pyplanner.planner import solve
from pyplanner.search import PORTFOLIOS, SearchStatus, portfolio_for_cores
from tests.tiers import VALIDATE, Instance, discover_instances, planner_supports

requires_val = pytest.mark.skipif(
    not VALIDATE.exists(), reason="VAL no está compilado (scripts/build_val.sh)"
)

INSTANCES = [i for i in discover_instances() if planner_supports(i)]

#: Límite de tiempo por instancia en los tests.
TIME_LIMIT = 120.0


def validate_with_val(instance: Instance, plan_file: Path) -> str:
    """Ejecuta VAL y devuelve su salida.

    VAL puede salir con código 0 aunque falle (por ejemplo, "Bad plan file!"), así que el
    resultado se decide por el texto "Plan valid" de su salida.
    """
    completed = subprocess.run(
        [str(VALIDATE), str(instance.domain), str(instance.problem), str(plan_file)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return completed.stdout + completed.stderr


@requires_val
@pytest.mark.parametrize("cores", sorted(PORTFOLIOS), ids=lambda n: f"{n}-cores")
@pytest.mark.parametrize("instance", INSTANCES, ids=lambda i: i.id)
def test_supported_tiers_are_solved_and_validated(
    instance: Instance, cores: int, tmp_path: Path
) -> None:
    domain = read_domain(instance.domain)
    problem = read_problem(instance.problem, domain)
    result = solve(domain, problem, portfolio_for_cores(cores), time_limit=TIME_LIMIT)
    assert result.status is SearchStatus.SOLVED, result.search.message
    assert result.plan is not None
    plan_file = tmp_path / "plan.ipc"
    plan_file.write_text(result.plan.to_ipc())
    output = validate_with_val(instance, plan_file)
    assert "Plan valid" in output, output
    value = re.search(r"Final value: (\S+)", output)
    if result.plan.metric is not None and value is not None:
        assert result.plan.metric == pytest.approx(float(value.group(1)), rel=1e-5, abs=1e-3)


def test_val_rejects_a_bad_plan(tmp_path: Path) -> None:
    """Comprueba que la validación con VAL detecta planes incorrectos."""
    if not VALIDATE.exists():
        pytest.skip("VAL no está compilado")
    instance = next(i for i in discover_instances() if i.tier == 1)
    plan_file = tmp_path / "bad.ipc"
    plan_file.write_text("(this-action-does-not-exist)\n")
    assert "Plan valid" not in validate_with_val(instance, plan_file)
