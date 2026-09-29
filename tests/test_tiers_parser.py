"""Evaluación por tiers del parser: fases 1 y 2 sobre los benchmarks de ``dominios/``."""

import pytest

from pyplanner.errors import PDDLError
from pyplanner.frontend import read_domain, read_problem
from tests.tiers import Instance, discover_instances, parser_supports

INSTANCES = discover_instances()
SUPPORTED = [i for i in INSTANCES if parser_supports(i)]
UNSUPPORTED = [i for i in INSTANCES if not parser_supports(i)]


@pytest.mark.parametrize("instance", SUPPORTED, ids=lambda i: i.id)
def test_supported_tiers_load(instance: Instance) -> None:
    domain = read_domain(instance.domain)
    problem = read_problem(instance.problem, domain)
    assert problem.domain_name == domain.name
    assert problem.init or problem.values


@pytest.mark.parametrize("instance", UNSUPPORTED, ids=lambda i: i.id)
def test_unsupported_tiers_fail_cleanly(instance: Instance) -> None:
    """Un fichero de un tier no soportado da un error PDDL legible, nunca otra excepción."""
    with pytest.raises(PDDLError):
        domain = read_domain(instance.domain)
        read_problem(instance.problem, domain)
