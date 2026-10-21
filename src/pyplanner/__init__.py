"""PyPlanner: planificador PDDL en Python."""

import os

# OpenBLAS (el álgebra lineal de NumPy) crea al importarse un hilo por núcleo. El
# planificador no hace álgebra lineal pesada, y esos hilos impedirían arrancar el
# portfolio con ``fork`` sin riesgo de bloqueos (ver :mod:`pyplanner.portfolio`). Se respeta
# la configuración del usuario, y no tiene efecto si NumPy ya estaba importado; en ese
# caso el portfolio lo detecta y arranca los procesos con ``spawn``.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

__version__ = "1.0.0"
