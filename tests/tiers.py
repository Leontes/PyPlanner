"""Las instancias de la evaluación por tiers, sacadas del catálogo PDDLDomains.

Los tiers son un dataset del catálogo, ``tests/tiers.toml``, en el formato de los datasets
de PDDLDomains. Cada grupo ``tier<N>/[<parte>/]<dominio>`` es un dominio del tier ``N`` con
sus problemas (ver ``docs/tiers.md``).

El catálogo es siempre el del submódulo ``vendor/PDDLDomains``, fijado en un commit. Así los
tests no dependen de ``$PDDL_DOMINIOS`` ni de otra copia instalada del paquete, y un cambio
del catálogo solo llega a PyPlanner al actualizar el submódulo. Los grupos se resuelven
directamente en el catálogo, sin el análisis de los datasets: no se escribe nada en el
submódulo.
"""

import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pddldomains import Catalog

REPO = Path(__file__).resolve().parent.parent
#: El submódulo con el catálogo.
CATALOG_REPO = REPO / "vendor" / "PDDLDomains"
#: El dataset de los tiers.
TIERS_FILE = Path(__file__).resolve().with_name("tiers.toml")
#: VAL, compilado con ``scripts/build_val.sh``.
VALIDATE = REPO / "tools" / "bin" / "Validate"

#: Tier máximo que el planificador tiene que resolver (con plan válido según VAL).
#: Se sube al cerrar cada iteración.
SUPPORTED_TIER = 6

#: Tier máximo que el parser (fases 1 y 2) tiene que aceptar. Durante una iteración va por
#: delante de :data:`SUPPORTED_TIER`; al cerrarla, los dos valen lo mismo.
PARSER_TIER = 6

#: Partes (*subtiers*) de tiers superiores que ya están soportadas, para las iteraciones que
#: se dividen en partes (la 3 en numérica y temporal). Al cerrar la iteración se sube el
#: tier y se vacía su entrada.
PARSER_PARTIAL: dict[int, frozenset[str]] = {}
SUPPORTED_PARTIAL: dict[int, frozenset[str]] = {}


@dataclass(frozen=True)
class Instance:
    """Un problema de benchmark junto con su dominio y su tier."""

    tier: int
    group: str
    domain: Path
    problem: Path

    @property
    def id(self) -> str:
        """Identificador legible para pytest, p. ej. ``tier1/gripper/p01``."""
        return f"{self.group}/{self.problem.stem}"

    @property
    def subtier(self) -> str | None:
        """Parte del tier (``numeric``, ``temporal``…) si el grupo la tiene."""
        parts = self.group.split("/")
        return parts[1] if len(parts) > 2 else None


@cache
def catalog() -> "Catalog":
    """El catálogo del submódulo, con el paquete ``pddldomains`` del propio submódulo."""
    source = CATALOG_REPO / "src"
    if not (source / "pddldomains" / "__init__.py").exists():
        raise RuntimeError(
            f"falta el submódulo del catálogo en {CATALOG_REPO}: "
            "git submodule update --init vendor/PDDLDomains"
        )
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    import pddldomains

    loaded = Path(pddldomains.__file__).resolve().parent
    if loaded != (source / "pddldomains").resolve():
        raise RuntimeError(f"pddldomains se ha importado de {loaded}, no del submódulo")
    return pddldomains.Catalog(CATALOG_REPO / "dominios")


@cache
def _groups() -> dict[str, tuple[str, list[str]]]:
    """Los grupos del dataset: nombre → (colección, problemas)."""
    catalog()  # importa pddldomains del submódulo
    from pddldomains.datasets import load_definition

    definition = load_definition(TIERS_FILE)
    if "select" in definition:
        raise RuntimeError(f"{TIERS_FILE}: los tiers solo admiten [[group]] explícitos")
    return {g["name"]: (g["collection"], g["problems"]) for g in definition["group"]}


def tier_dir(group: str) -> Path:
    """El directorio de la colección de un grupo (``tier1/blocksworld``)."""
    collection, _problems = _groups()[group]
    return catalog().find(collection).path


def discover_instances() -> list[Instance]:
    """Devuelve todas las instancias de los tiers, ordenadas por tier y grupo."""
    instances = []
    for group, (collection_id, problems) in sorted(_groups().items()):
        tier = int(group.split("/", 1)[0].removeprefix("tier"))
        collection = catalog().find(collection_id)
        for name in problems:
            task = collection.task(name)
            instances.append(Instance(tier, group, task.domain, task.problem))
    return instances


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
