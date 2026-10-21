"""Las dos fases del parser sobre los problemas de prueba (``tests/dataset.toml``, del
catálogo PDDLDomains)."""

import pytest

from pyplanner.frontend import read_domain, read_problem
from tests.catalog import Instance, instances


@pytest.mark.parametrize("instance", instances(), ids=lambda i: i.id)
def test_dataset_loads(instance: Instance) -> None:
    domain = read_domain(instance.domain)
    problem = read_problem(instance.problem, domain)
    assert problem.domain_name == domain.name
    assert problem.init or problem.values
