"""Los problemas de prueba, sacados del catálogo PDDLDomains.

Los problemas que se resuelven en los tests son un dataset del catálogo,
``tests/dataset.toml``, en el formato de los datasets de PDDLDomains. Es una selección de
colecciones agrupada por nivel de versión de PDDL, de STRIPS (nivel 1) a PDDL+ (nivel 6),
como el dataset ``by-level`` del catálogo: cada colección es un grupo
``level<N>/<colección>``, con el nivel que calcula el analizador del catálogo (ver
``docs/es/pruebas.md``).

El catálogo es siempre el del submódulo ``vendor/PDDLDomains``, fijado en un commit. Así los
tests no dependen de ``$PDDL_DOMINIOS`` ni de otra copia instalada del paquete, y un cambio
del catálogo solo llega a PyPlanner al actualizar el submódulo. El análisis se hace en
memoria, sin la caché del catálogo: no se escribe nada en el submódulo.
"""

import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pddldomains import Catalog

REPO = Path(__file__).resolve().parent.parent
#: El submódulo con el catálogo.
CATALOG_REPO = REPO / "vendor" / "PDDLDomains"
#: El dataset de los problemas de prueba.
DATASET_FILE = Path(__file__).resolve().with_name("dataset.toml")
#: VAL, compilado con ``scripts/build_val.sh``.
VALIDATE = REPO / "tools" / "bin" / "Validate"


@dataclass(frozen=True)
class Instance:
    """Un problema de prueba con su dominio y su nivel de versión de PDDL."""

    level: int
    group: str
    domain: Path
    problem: Path

    @property
    def id(self) -> str:
        """Identificador para pytest: ``level1/ipc-1998/gripper-round-1-strips/p01``."""
        return f"{self.group}/{self.problem.stem}"


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


def definition() -> dict[str, Any]:
    """La definición del dataset, comprobada con las reglas del catálogo."""
    catalog()  # importa pddldomains del submódulo
    from pddldomains.datasets import load_definition

    return load_definition(DATASET_FILE)


@cache
def instances() -> list[Instance]:
    """Los problemas de prueba, ordenados por nivel y grupo."""
    domains = catalog()  # antes de importar pddldomains: lo importa del submódulo
    from pddldomains.analysis import Analyzer
    from pddldomains.datasets import select_entries

    selections = definition()
    if "group" in selections:
        raise RuntimeError(f"{DATASET_FILE}: solo admite [[select]], con el nivel del catálogo")
    analyzer = Analyzer(domains, cache_file=None)  # sin caché: no escribe en el submódulo
    entries = [
        entry
        for selection in selections["select"]
        for entry in select_entries(domains, analyzer, selection)
    ]
    found = [
        Instance(entry.analysis.level, entry.group, entry.task.domain, entry.task.problem)
        for entry in entries
    ]
    return sorted(found, key=lambda i: (i.level, i.group, i.problem.stem))


def collection_dir(collection: str) -> Path:
    """El directorio de una colección del catálogo (``ipc-2000/blocks-strips-untyped``)."""
    return catalog().find(collection).path
