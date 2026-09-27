"""Comprobaciones de integridad del conjunto de benchmarks de ``dominios/``."""

from tests.tiers import PARSER_TIER, SUPPORTED_TIER, discover_instances

EXPECTED_TIERS = {1, 2, 3, 4, 5}


def test_every_tier_has_instances() -> None:
    tiers = {instance.tier for instance in discover_instances()}
    assert tiers == EXPECTED_TIERS


def test_every_domain_has_at_least_two_problems() -> None:
    per_domain: dict[str, int] = {}
    for instance in discover_instances():
        key = str(instance.domain)
        per_domain[key] = per_domain.get(key, 0) + 1
    assert per_domain
    assert all(count >= 2 for count in per_domain.values()), per_domain


def test_supported_tiers_are_in_range() -> None:
    assert 0 <= SUPPORTED_TIER <= PARSER_TIER <= max(EXPECTED_TIERS)
