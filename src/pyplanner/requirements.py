"""Requisitos PDDL (``:requirements``): los que existen, los que implican otros y los soportados.

Este módulo es la única fuente de verdad sobre qué requisitos acepta el planificador. Las dos
fases del parser lo consultan: la fase 1 para rechazar pronto un fichero PDDL que no se
puede procesar y la fase 2 porque su JSON de entrada puede venir de cualquier otra fuente.

Cada iteración del roadmap añade sus requisitos a :data:`SUPPORTED_REQUIREMENTS`.
"""

from collections.abc import Iterable

#: Todos los requisitos que existen en las versiones de PDDL, soportados o no.
#: Un requisito que no está aquí es un error; uno que está pero no se soporta, también,
#: pero con un mensaje distinto.
KNOWN_REQUIREMENTS: frozenset[str] = frozenset(
    {
        # PDDL 1.2
        ":strips",
        ":typing",
        ":negative-preconditions",
        ":disjunctive-preconditions",
        ":equality",
        ":existential-preconditions",
        ":universal-preconditions",
        ":quantified-preconditions",
        ":conditional-effects",
        ":action-expansions",
        ":foreach-expansions",
        ":dag-expansions",
        ":domain-axioms",
        ":subgoals-through-axioms",
        ":safety-constraints",
        ":expression-evaluation",
        ":fluents",
        ":open-world",
        ":true-negation",
        ":adl",
        ":ucpop",
        # PDDL 2.1
        ":numeric-fluents",
        ":durative-actions",
        ":duration-inequalities",
        ":continuous-effects",
        # PDDL 2.2
        ":derived-predicates",
        ":timed-initial-literals",
        # PDDL+
        ":time",
        # PDDL 3.x (fuera del alcance del proyecto)
        ":constraints",
        ":preferences",
        ":action-costs",
        ":object-fluents",
    }
)

#: Requisitos que equivalen a un conjunto de otros requisitos.
REQUIREMENT_EXPANSIONS: dict[str, frozenset[str]] = {
    ":quantified-preconditions": frozenset(
        {":existential-preconditions", ":universal-preconditions"}
    ),
    ":adl": frozenset(
        {
            ":strips",
            ":typing",
            ":negative-preconditions",
            ":disjunctive-preconditions",
            ":equality",
            ":existential-preconditions",
            ":universal-preconditions",
            ":quantified-preconditions",
            ":conditional-effects",
        }
    ),
    ":fluents": frozenset({":numeric-fluents"}),
}

#: Requisitos que el planificador soporta en la iteración actual.
SUPPORTED_REQUIREMENTS: frozenset[str] = frozenset({":strips"})

#: Requisito que se supone cuando un dominio no declara ``:requirements``.
DEFAULT_REQUIREMENT = ":strips"


def expand_requirements(requirements: Iterable[str]) -> frozenset[str]:
    """Devuelve los requisitos junto con todos los que implican, de forma transitiva.

    Si no hay ningún requisito se devuelve :data:`DEFAULT_REQUIREMENT`.
    """
    pending = list(requirements) or [DEFAULT_REQUIREMENT]
    result: set[str] = set()
    while pending:
        requirement = pending.pop()
        if requirement not in result:
            result.add(requirement)
            pending.extend(REQUIREMENT_EXPANSIONS.get(requirement, ()))
    return frozenset(result)


def unsupported_requirements(requirements: Iterable[str]) -> list[str]:
    """Devuelve, ordenados, los requisitos (ya expandidos) que no están soportados.

    Los metarrequisitos como ``:adl`` se consideran soportados si lo están todos los
    requisitos a los que equivalen.
    """
    expanded = expand_requirements(requirements)
    return sorted(
        r for r in expanded if r not in SUPPORTED_REQUIREMENTS and r not in REQUIREMENT_EXPANSIONS
    )
