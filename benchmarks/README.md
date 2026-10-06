# Benchmarks de medición

Colecciones más grandes que las de [`dominios/`](../dominios/README.md), para **medir**
el rendimiento del planificador. No forman parte de los tests: la batería de tiers solo
recorre `dominios/`, y aquí hay instancias que el planificador puede no resolver.

Se miden con [`scripts/benchmark.py`](../scripts/benchmark.py), que resuelve cada
problema con las configuraciones pedidas, con límite de tiempo y de memoria por búsqueda,
y valida los planes con VAL:

```
python scripts/benchmark.py benchmarks/timed-initial-literals/satellite \
    -c gbfs-ff-pref bfws-ff-pref -t 60 -m 2500 -w 3
```

Cada proceso tiene además un límite duro de memoria virtual (`--memory` + 512 MB) que
cubre también el grounding, y el script no empieza si los procesos no caben en el 80 % de
la memoria disponible: dos medidas sin esas garantías agotaron la memoria de WSL.

| Colección | Instancias | Origen | Para qué |
|-----------|------------|--------|----------|
| `timed-initial-literals/satellite` | p01–p20 | IPC 2004 (satellite-time-time-windows-strips), instancias 1–20 | Heurística temporal con plazos (iteración 4); resultados en [motor-busqueda.md](../docs/motor-busqueda.md#iteración-4-literales-iniciales-temporizados) |
| `preferences/storage` | p01–p20 | IPC 2006 (storage-preferences-qualitative), instancias 1–20 | Preferencias y búsqueda *anytime* (iteración 5c): calidad de los planes en el límite de tiempo |

Mismas convenciones que en `dominios/`: los ficheros se copian sin modificar, salvo las
erratas de `:requirements`, que se marcan con (*), y las discordancias de otro tipo que se
decide corregir, con (‡). Satellite tiene una (‡): el dominio se llamaba igual que su tipo
`satellite` y se ha renombrado a `satellite-time-windows`, como en `dominios/`.
p01 y p02 son las mismas que las del tier 4. Las instancias 21 a 36 de la colección
original (de 90 KB a 3 MB) no se han copiado.

Storage tiene otra (‡), la misma que en el tier 5: el dominio declaraba dos veces el tipo
`area` y se ha quitado la primera declaración. p01 y p02 son las del tier 5. Con
preferencias, `scripts/benchmark.py` usa la búsqueda *anytime*: cada problema agota el
límite de tiempo salvo que se demuestre el óptimo antes.
