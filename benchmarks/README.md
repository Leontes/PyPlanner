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
| `pddlplus/car-nodrag` | p01–p10 | SMTPlan (car_nodrag) | PDDL+ (iteración 6): un proceso con dinámica cuadrática y un evento |
| `pddlplus/generator-events` | p01–p08 | SMTPlan (generator_events) | PDDL+ (iteración 6): acción durativa, un proceso no lineal y dos eventos |
| `continuous-effects/generator-linear` | p01–p08 | SMTPlan (generator_linear) | PDDL 2.1 con efectos continuos lineales (motor temporal de la 3b) |
| `continuous-effects/generator-nonlinear` | p01–p08 | SMTPlan (generator_nonlinear) | PDDL 2.1 con efectos continuos no lineales: fuera de la programación lineal de la 3b, van al motor discretizado (iteración 6b) |
| `continuous-effects/generator-toricelli` | p01–p09 | SMTPlan (generator_toricelli) | PDDL 2.1 con efectos continuos no lineales y duraciones que dependen del estado (motor discretizado) |

Mismas convenciones que en `dominios/`: los ficheros se copian sin modificar, salvo las
erratas de `:requirements`, que se marcan con (*), y las discordancias de otro tipo que se
decide corregir, con (‡). Satellite tiene una (‡): el dominio se llamaba igual que su tipo
`satellite` y se ha renombrado a `satellite-time-windows`, como en `dominios/`.
p01 y p02 son las mismas que las del tier 4. Las instancias 21 a 36 de la colección
original (de 90 KB a 3 MB) no se han copiado.

Las de SMTPlan:

- `car-nodrag` y `generator-events` son las colecciones de donde salen car y generator del
  tier 6 (p01 y p02 son las mismas), con sus erratas: (‡) funciones sin paréntesis en los
  problemas de car (`(= d 0)`), (‡) un guion pegado al tipo en generator (`?t -tank`),
  (‡) sin valor inicial para `(ptime ?t)` en los problemas de generator (sin él no hay
  ningún plan válido, tampoco para VAL) y (*) `:continuous-effects` sin declarar en
  generator.
- Las tres de `continuous-effects` no son PDDL+ (no tienen procesos ni eventos), sino
  PDDL 2.1 con efectos continuos. (*) Ninguna declaraba `:continuous-effects`.
  (‡) Los problemas de `generator-nonlinear` y `generator-toricelli` eran de un dominio
  `generator`, pero los dominios se llamaban `generator2`; como `generator` es además un
  tipo, dominio y problemas se han renombrado con el nombre de la colección.
  (‡) En `generator-toricelli`, `? g` y `? duration` tenían un espacio tras la
  interrogación.

Storage tiene otra (‡), la misma que en el tier 5: el dominio declaraba dos veces el tipo
`area` y se ha quitado la primera declaración. p01 y p02 son las del tier 5.

`scripts/benchmark.py` resuelve con `solve`, como el planificador: con preferencias, la
búsqueda es *anytime* (cada problema agota el límite de tiempo salvo que se demuestre el
óptimo antes); con procesos y eventos, el plan se valida con un paso de tiempo más fino y,
si falla, se busca otra vez con δ/2 (la tabla da el δ final cuando no es 1).

## Colecciones locales (fuera de git)

Para elegir los portfolios con los landmarks (trabajo opcional, punto 2) se midieron dos
colecciones más, y para las preferencias en tareas temporales (punto 4), una tercera. No
están en el repositorio (están en `.git/info/exclude`). Se rehacen así:

- **`classical/`**: 17 colecciones de 20 instancias, con el mismo formato (`domain.pddl`
  o `domain-pNN.pddl` por problema, como openstacks y parc-printer, y `pNN.pddl`):
  - las 14 de `ipc-2011/domains/*-sequential-satisficing` de
    [potassco/pddl-instances](https://github.com/potassco/pddl-instances) (`cf19edf`),
    sin el sufijo en el nombre;
  - recharging-robots, ricochet-robots y slitherlink de `sat/` de
    [ipc2023-classical/ipc2023-dataset](https://github.com/ipc2023-classical/ipc2023-dataset)
    (`1e29a0e`).

  Erratas: (*) floor-tile usa `total-cost` sin declarar `:action-costs`; (*) tidybot usa
  `not` sin declarar `:negative-preconditions`; (‡) las acciones `up`, `down`, `left` y
  `right` de floor-tile se llamaban igual que sus predicados y pasan a `move-up`, etc.;
  (‡) el objeto `cart` de tidybot se llamaba igual que su tipo y pasa a `cart0`; (‡) los
  problemas de parking se llamaban igual que el dominio y pasan a `parking-pNN`; (‡) el
  p10 de woodworking tenía una línea `- board` sin objetos, que se ha quitado.
- **`temporal/`**: driverlog y zenotravel de la IPC 2002 (*time-simple-automatic*) de
  potassco, instancias 1 a 17, con el dominio del tier 3 (que lleva su errata). p01 y p02
  son las del tier.
- **`temporal-preferences/`**: storage, trucks, tpp, pathways y pipesworld,
  `ipc-2006/domains/*-preferences-complex` de potassco (`cf19edf`), instancias 1 a 20
  (pipesworld solo tiene 18): 98 problemas con acciones durativas, preferencias (en el
  objetivo, en las condiciones de las acciones y en las restricciones de trayectoria) y
  restricciones duras. Erratas: (‡) storage declaraba dos veces el tipo `area` (como en el
  tier 5) y se ha quitado la primera declaración; (‡) en tpp, `?g -goods` tenía el guion
  pegado al tipo; (*) pathways usa `not` en las precondiciones sin declarar
  `:negative-preconditions`; (‡) nueve problemas de pathways (p08, p09, p12, p14 y p16 a
  p20) daban dos valores iniciales a la misma función: se queda el último, el que usa VAL.
