# Benchmarks de medición

Para **medir** el rendimiento del planificador se usan colecciones más grandes que las de
los [tiers](tiers.md), todas del catálogo
[PDDLDomains](https://github.com/Leontes/PDDLDomains) del submódulo `vendor/PDDLDomains`.
No forman parte de los tests: hay instancias que el planificador puede no resolver.

Se miden con [`scripts/benchmark.py`](../scripts/benchmark.py), que resuelve cada
problema con las configuraciones pedidas, con límite de tiempo y de memoria por búsqueda,
y valida los planes con VAL. Acepta una colección del catálogo (su identificador o, si no
es ambigua, su parte final), un dataset del catálogo (`-d`, por nombre o un fichero
`.toml`) o un directorio:

```
python scripts/benchmark.py ipc-2004/satellite-time-time-windows-strips \
    -c gbfs-ff-pref bfws-ff-pref -t 60 -m 2500 -w 3
python scripts/benchmark.py -d tests/tiers.toml -c bfws-ff-pref
```

Cada proceso tiene además un límite duro de memoria (`--memory` + 512 MB) que cubre
también el grounding, y el script no empieza si los procesos no caben en el 80 % de la
memoria disponible: dos medidas sin esas garantías agotaron la memoria de WSL. Con el
miembro de los portfolios que usa la GPU, `-w 3`.

Las erratas corregidas de cada colección están en su ficha del catálogo (`info.toml`), con
las mismas marcas que en los tiers: (*) para las de `:requirements` y (‡) para las demás.

## Colecciones

| Colección del catálogo | Instancias | Para qué |
|------------------------|------------|----------|
| `ipc-2004/satellite-time-time-windows-strips` | p01–p20 | Heurística temporal con plazos (iteración 4); resultados en [motor-busqueda.md](motor-busqueda.md#iteración-4-literales-iniciales-temporizados) |
| `ipc-2006/storage-preferences-qualitative` | p01–p20 | Preferencias y búsqueda *anytime* (iteración 5c): calidad de los planes en el límite de tiempo |
| `smtplan/car-nodrag` | p01–p10 | PDDL+ (iteración 6): un proceso con dinámica cuadrática y un evento |
| `smtplan/generator-events` | p01–p08 | PDDL+ (iteración 6): acción durativa, un proceso no lineal y dos eventos |
| `smtplan/generator-linear` | p01–p08 | PDDL 2.1 con efectos continuos lineales (motor temporal de la 3b) |
| `smtplan/generator-nonlinear` | p01–p08 | PDDL 2.1 con efectos continuos no lineales: fuera de la programación lineal de la 3b, van al motor discretizado (iteración 6b) |
| `smtplan/generator-toricelli` | p01–p09 | PDDL 2.1 con efectos continuos no lineales y duraciones que dependen del estado (motor discretizado) |
| `ipc-2002/driverlog-time-simple-automatic`, `ipc-2002/zenotravel-time-simple-automatic` | p01–p17 | Temporales sin TIL: elección de los portfolios (trabajo opcional, puntos 2 y 3) |
| Las 14 `ipc-2011/*-sequential-satisficing` y `ipc-2023-classical/{recharging-robots,ricochet-robots,slitherlink}` | p01–p20 | Clásicas: elección de los portfolios con los landmarks (trabajo opcional, punto 2) |
| `ipc-2006/{storage,trucks,tpp,pathways,pipesworld}-preferences-complex` | p01–p20 (pipesworld, 18) | Preferencias en tareas temporales (trabajo opcional, punto 4): 98 problemas |

Las 14 clásicas de la IPC 2011 son barman, elevator, floor-tile, no-mystery, openstacks,
parc-printer, parking, peg-solitaire, scanalyzer-3d, sokoban, tidybot, transport,
visit-all y woodworking. Openstacks y parc-printer tienen un dominio por problema
(`domain-pNN.pddl`).

`scripts/benchmark.py` resuelve con `solve`, como el planificador: con preferencias, la
búsqueda es *anytime* (cada problema agota el límite de tiempo salvo que se demuestre el
óptimo antes); con procesos y eventos, el plan se valida con un paso de tiempo más fino y,
si falla, se busca otra vez con δ/2 (la tabla da el δ final cuando no es 1).

## Rutas anteriores

Hasta la versión 0.8, los tiers estaban copiados en `dominios/` y las colecciones de
medida en `benchmarks/` (unas en git y otras solo en local). Las medidas de
[motor-busqueda.md](motor-busqueda.md) y del [roadmap](ROADMAP.md) usan esas rutas. Son los
mismos ficheros, byte a byte, que estas colecciones del catálogo (greenhouse solo cambia
en el comentario de cabecera):

| Ruta anterior | Colección del catálogo |
|---------------|------------------------|
| `dominios/tier1-strips/blocksworld` | `ipc-2000/blocks-strips-untyped` (p01, p02) |
| `dominios/tier1-strips/gripper` | `ipc-1998/gripper-round-1-strips` (p01, p02) |
| `dominios/tier2-adl/rubiks-cube` | `ipc-2023-classical/rubiks-cube` (p01, p02) |
| `dominios/tier2-adl/slitherlink` | `ipc-2023-classical/slitherlink` (p01, p02) |
| `dominios/tier3-pddl21/numeric/counters` | `ipc-2023-numeric/counters` (p01, p02) |
| `dominios/tier3-pddl21/numeric/zenotravel` | `ipc-2023-numeric/zenotravel` (p01, p02) |
| `dominios/tier3-pddl21/temporal/driverlog` | `ipc-2002/driverlog-time-simple-automatic` (p01, p02) |
| `dominios/tier3-pddl21/temporal/zenotravel` | `ipc-2002/zenotravel-time-simple-automatic` (p01, p02) |
| `dominios/tier3-pddl21/temporal/greenhouse` | `propios/greenhouse` (p01, p02) |
| `dominios/tier4-pddl22/derived-predicates/psr-middle` | `ipc-2004/psr-middle-derived-predicates-adl` (p01, p02) |
| `dominios/tier4-pddl22/timed-initial-literals/satellite` | `ipc-2004/satellite-time-time-windows-strips` (p01, p02) |
| `dominios/tier5-pddl3/constraints/trucks` | `ipc-2006/trucks-time-constraints` (p01, p02) |
| `dominios/tier5-pddl3/preferences/storage` | `ipc-2006/storage-preferences-qualitative` (p01, p02) |
| `dominios/tier6-pddlplus/car` | `smtplan/car-nodrag` (p01, p02) |
| `dominios/tier6-pddlplus/generator` | `smtplan/generator-events` (p01, p02) |
| `benchmarks/timed-initial-literals/satellite` | `ipc-2004/satellite-time-time-windows-strips` |
| `benchmarks/preferences/storage` | `ipc-2006/storage-preferences-qualitative` |
| `benchmarks/pddlplus/car-nodrag` | `smtplan/car-nodrag` |
| `benchmarks/pddlplus/generator-events` | `smtplan/generator-events` |
| `benchmarks/continuous-effects/generator-{linear,nonlinear,toricelli}` | `smtplan/generator-{linear,nonlinear,toricelli}` |
| `benchmarks/temporal/{driverlog,zenotravel}` | `ipc-2002/{driverlog,zenotravel}-time-simple-automatic` |
| `benchmarks/classical/<dominio>` (las 14 de la IPC 2011) | `ipc-2011/<dominio>-sequential-satisficing` |
| `benchmarks/classical/{recharging-robots,ricochet-robots,slitherlink}` | `ipc-2023-classical/<dominio>` |
| `benchmarks/temporal-preferences/<dominio>` | `ipc-2006/<dominio>-preferences-complex` |

Las colecciones de `benchmarks/` tenían los mismos problemas que las del catálogo. Las del
catálogo vienen del mismo origen y llevan las mismas erratas, explicadas en sus fichas.
