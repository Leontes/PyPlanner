# Benchmarks de medición

Colecciones más grandes que las de [`dominios/`](../dominios/README.md), para **medir**
el rendimiento del planificador. No forman parte de los tests: la batería de tiers solo
recorre `dominios/`, y aquí hay instancias que el planificador puede no resolver.

Se miden con [`scripts/benchmark.py`](../scripts/benchmark.py), que resuelve cada
problema con las configuraciones pedidas, con límite de tiempo y de memoria por búsqueda,
y valida los planes con VAL:

```
python scripts/benchmark.py benchmarks/timed-initial-literals/satellite \
    -c gbfs-ff-pref bfws-ff-pref -t 60 -m 2500 -w 4
```

| Colección | Instancias | Origen | Para qué |
|-----------|------------|--------|----------|
| `timed-initial-literals/satellite` | p01–p20 | IPC 2004 (satellite-time-time-windows-strips), instancias 1–20 | Heurística temporal con plazos (iteración 4); resultados en [motor-busqueda.md](../docs/motor-busqueda.md#iteración-4-literales-iniciales-temporizados) |

Mismas convenciones que en `dominios/`: los ficheros se copian sin modificar, salvo las
erratas de `:requirements`, que se marcan con (*). Las de satellite no tienen ninguna.
p01 y p02 son las mismas que las del tier 4. Las instancias 21 a 36 de la colección
original (de 90 KB a 3 MB) no se han copiado.
