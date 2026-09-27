"""Fase 2 del parser: documento JSON → objetos internos (AST).

Ver :mod:`pyplanner.loader.loader` para las comprobaciones que se hacen y
:mod:`pyplanner.loader.ast` para las clases resultantes.
"""

from pyplanner.loader.loader import load_domain, load_problem

__all__ = ["load_domain", "load_problem"]
