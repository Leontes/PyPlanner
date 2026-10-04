"""Descubrimiento de los benchmarks de ``dominios/`` para la evaluación por tiers.

Cada problema ``pNN.pddl`` que esté en un directorio con ``domain.pddl`` dentro de
``dominios/tierN-*/`` es una instancia del tier ``N``.
"""

import re
from dataclasses import dataclass
from pathlib import Path

DOMINIOS_DIR = Path(__file__).resolve().parent.parent / "dominios"

#: Tier máximo que el planificador tiene que resolver (con plan válido según VAL).
#: Se sube al cerrar cada iteración.
SUPPORTED_TIER = 4

#: Tier máximo que el parser (fases 1 y 2) tiene que aceptar. Durante una iteración va por
#: delante de :data:`SUPPORTED_TIER`; al cerrarla, los dos valen lo mismo.
PARSER_TIER = 4

#: Partes (*subtiers*) de tiers superiores que ya están soportadas, para las iteraciones que
#: se dividen en partes (la 3 en numérica y temporal). Al cerrar la iteración se sube el
#: tier y se vacía su entrada.
PARSER_PARTIAL: dict[int, frozenset[str]] = {}
SUPPORTED_PARTIAL: dict[int, frozenset[str]] = {}

_TIER_DIR = re.compile(r"^tier(\d+)-")


@dataclass(frozen=True)
class Instance:
    """Un problema de benchmark junto con su dominio y su tier."""

    tier: int
    domain: Path
    problem: Path

    @property
    def id(self) -> str:
        """Identificador legible para pytest, p. ej. ``tier1/gripper/p01``."""
        rel = self.problem.parent.relative_to(DOMINIOS_DIR)
        return f"tier{self.tier}/{'/'.join(rel.parts[1:])}/{self.problem.stem}"

    @property
    def subtier(self) -> str | None:
        """Parte del tier (``numeric``, ``temporal``…) si el dominio está en un subdirectorio."""
        parts = self.problem.parent.relative_to(DOMINIOS_DIR).parts
        return parts[1] if len(parts) > 2 else None


def parser_supports(instance: Instance) -> bool:
    """``True`` si el parser tiene que aceptar la instancia."""
    return instance.tier <= PARSER_TIER or instance.subtier in PARSER_PARTIAL.get(
        instance.tier, frozenset()
    )


def planner_supports(instance: Instance) -> bool:
    """``True`` si el planificador tiene que resolver la instancia (y VAL validar el plan)."""
    return instance.tier <= SUPPORTED_TIER or instance.subtier in SUPPORTED_PARTIAL.get(
        instance.tier, frozenset()
    )


def discover_instances(root: Path = DOMINIOS_DIR) -> list[Instance]:
    """Devuelve todas las instancias de ``root``, ordenadas por tier y ruta."""
    instances = []
    for tier_dir in sorted(root.iterdir()):
        match = _TIER_DIR.match(tier_dir.name)
        if not tier_dir.is_dir() or match is None:
            continue
        tier = int(match.group(1))
        for domain in sorted(tier_dir.rglob("domain.pddl")):
            for problem in sorted(domain.parent.glob("p*.pddl")):
                instances.append(Instance(tier, domain, problem))
    return instances
