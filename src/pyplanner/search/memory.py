"""Medida de la memoria del proceso y de la disponible, para el límite de memoria.

Se leen ``/proc/self/statm`` y ``/proc/meminfo`` (Linux, incluido WSL). En otros sistemas
las funciones devuelven ``None`` y no se aplica ningún límite automático.
"""

import os


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
