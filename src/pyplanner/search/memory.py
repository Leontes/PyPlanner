"""Medida de la memoria del proceso y de la disponible, y límite duro de memoria.

Se leen ``/proc/self/statm``, ``/proc/self/status`` y ``/proc/meminfo`` (Linux, incluido
WSL). En otros sistemas las funciones devuelven ``None`` y no se aplica ningún límite
automático.

**Dos límites.** El blando (``max_memory`` de cada búsqueda) lo comprueba el motor cada
pocas iteraciones con la memoria residente y para la búsqueda limpiamente, con sus
estadísticas. Pero no ve lo que crece de golpe entre dos comprobaciones, ni el grounding.
El duro (:func:`limit_process`) es ``RLIMIT_DATA``: al pasarlo, cualquier reserva de memoria
falla con ``MemoryError``, y la búsqueda termina con ``LIMIT_REACHED``
(:func:`pyplanner.search.search`). Se usa ``RLIMIT_DATA`` y no ``RLIMIT_AS`` porque CUDA
reserva decenas de GB de memoria virtual que no usa; ``RLIMIT_DATA`` solo cuenta la memoria
de datos (montículo y reservas anónimas), que sigue de cerca a la residente. Lo fijan cada
miembro del portfolio (en su proceso) y la CLI (antes del grounding); como librería no se
limita al proceso que llama.
"""

import os
import sys

#: Margen del límite duro sobre el blando (el mayor de los dos): el blando para la búsqueda
#: con sus estadísticas, y el duro solo salta si algo crece de golpe.
HARD_MARGIN_MB = 512
HARD_MARGIN_FRACTION = 0.25
#: Margen adicional para CUDA (bibliotecas, contexto y memoria fijada que reserva al usar
#: la GPU, después de inicializarla).
CUDA_MARGIN_MB = 512


def resident_memory_mb() -> float | None:
    """Memoria residente del proceso, en MB (``None`` si no se puede medir)."""
    try:
        with open("/proc/self/statm", encoding="ascii") as statm:
            pages = int(statm.read().split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return pages * os.sysconf("SC_PAGE_SIZE") / 2**20


def available_memory_mb() -> float | None:
    """Memoria disponible en el sistema, en MB (``None`` si no se puede medir)."""
    try:
        with open("/proc/meminfo", encoding="ascii") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def data_memory_mb() -> float | None:
    """Memoria de datos del proceso (``VmData``, lo que cuenta ``RLIMIT_DATA``), en MB."""
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("VmData:"):
                    return int(line.split()[1]) / 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def hard_limit_mb(soft: float, cuda: bool = False) -> float:
    """El límite duro para un proceso con límite blando ``soft`` (MB): el mayor entre
    ``soft`` y lo que ya ocupa, más un margen (y el de CUDA si la usa)."""
    current = data_memory_mb() or 0.0
    margin = max(HARD_MARGIN_MB, soft * HARD_MARGIN_FRACTION) + (CUDA_MARGIN_MB if cuda else 0)
    return max(soft, current) + margin


def limit_process(megabytes: float) -> bool:
    """Fija el límite duro de memoria de datos (``RLIMIT_DATA``) de este proceso y de los
    que cree después. Solo en Linux (en macOS no se aplica a todas las reservas). Si ya
    había uno menor, se queda ese: un proceso no puede subir su límite. Devuelve ``True``
    si se ha aplicado."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        import resource
    except ImportError:
        return False
    limit = int(megabytes * 2**20)
    _, hard = resource.getrlimit(resource.RLIMIT_DATA)
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_DATA, (limit, limit))
    return True
