# Problemas de prueba

Los tests resuelven un conjunto de problemas que cubre todo el lenguaje que acepta el
planificador, de STRIPS a PDDL+, y validan cada plan con
[VAL](https://github.com/KCL-Planning/VAL). Los problemas salen del catálogo
[PDDLDomains](https://github.com/Leontes/PDDLDomains), que está en el submódulo
`vendor/PDDLDomains`, fijado en un commit.

## El dataset

[`tests/dataset.toml`](../tests/dataset.toml) es un dataset del catálogo: los problemas
`p01` y `p02` de quince colecciones, al menos dos por **nivel de versión de PDDL**. Como en
el dataset `by-level` del catálogo, cada colección es un grupo `level<N>/<colección>`. El
nivel lo calcula el catálogo a partir de los requisitos que usa cada problema, en una escala
acumulativa de STRIPS (1) a PDDL+ (6). Un planificador que llega al nivel *N* tiene que
resolver todos los problemas de los niveles 1 a *N* (ver "Versión de PDDL (niveles)" en
`docs/es/catalogo.md` del catálogo).

| Nivel | Versión | Colección | Requisitos declarados |
|------:|---------|-----------|-----------------------|
| 1 | STRIPS | `ipc-2000/blocks-strips-untyped` | `:strips` |
| 1 | STRIPS | `ipc-1998/gripper-round-1-strips` | ninguno (STRIPS implícito) |
| 2 | PDDL 1.2 (ADL) | `ipc-2023-classical/rubiks-cube` | `:adl` |
| 2 | PDDL 1.2 (ADL) | `ipc-2023-classical/slitherlink` | `:typing :adl` |
| 3 | PDDL 2.1, numérico | `ipc-2023-numeric/counters` | `:strips :typing :equality :adl :numeric-fluents` (*) |
| 3 | PDDL 2.1, numérico | `ipc-2023-numeric/zenotravel` | `:typing :fluents` (*) |
| 3 | PDDL 2.1, temporal | `ipc-2002/driverlog-time-simple-automatic` | `:typing :durative-actions :numeric-fluents` (*) |
| 3 | PDDL 2.1, temporal | `ipc-2002/zenotravel-time-simple-automatic` | `:durative-actions :typing :numeric-fluents` (*) |
| 3 | PDDL 2.1, temporal | `propios/greenhouse` (†) | `:typing :durative-actions :numeric-fluents :continuous-effects :duration-inequalities :conditional-effects :negative-preconditions` |
| 4 | PDDL 2.2 | `ipc-2004/psr-middle-derived-predicates-adl` | `:adl :derived-predicates` |
| 4 | PDDL 2.2 | `ipc-2004/satellite-time-time-windows-strips` (‡) | `:strips :equality :typing :fluents :durative-actions :timed-initial-literals` |
| 5 | PDDL 3 | `ipc-2006/trucks-time-constraints` | `:typing :adl :durative-actions :fluents :constraints` |
| 5 | PDDL 3 | `ipc-2006/storage-preferences-qualitative` (‡) | `:typing :adl :preferences :constraints` |
| 6 | PDDL+ | `smtplan/car-nodrag` (‡) | `:typing :durative-actions :fluents :time :negative-preconditions :timed-initial-literals` |
| 6 | PDDL+ | `smtplan/generator-events` (*) (‡) | `:fluents :durative-actions :duration-inequalities :adl :typing :time :continuous-effects` |

[`tests/catalog.py`](../tests/catalog.py) lee el dataset y da los problemas a los tests.
Los identificadores de pytest son `<grupo>/<problema>`, por ejemplo
`level3/ipc-2002/driverlog-time-simple-automatic/p01`. Los usan:

- `test_dataset_parser.py`: las dos fases del parser (PDDL → JSON → AST);
- `test_dataset_solve.py`: el portfolio de cada número de núcleos (de 1 a 7) resuelve
  cada problema, y VAL valida el plan y su métrica;
- `test_heuristics.py`: las heurísticas con Numba y PyTorch dan los mismos valores que la
  referencia en Python;
- `test_dataset.py`: la integridad del dataset. Cada nivel tiene al menos dos colecciones,
  cada colección de la lista da sus problemas (una colección que falte o cambie de nombre
  en el catálogo no desaparece en silencio) y cada problema es del nivel de su grupo.

Para añadir problemas basta con ampliar `problems` o `collections` en
`tests/dataset.toml`.

## El catálogo en el submódulo

- **Inicializarlo:** `git clone --recursive`, o en un clon ya hecho,
  `git submodule update --init vendor/PDDLDomains`. Sin él, los tests que usan el catálogo
  fallan con un error que lo dice.
- **Siempre el del submódulo:** `tests/catalog.py` importa el paquete `pddldomains` del
  propio submódulo (no uno instalado) y usa su directorio `dominios/` (no
  `$PDDL_DOMINIOS`). Si se importara otro, da un error.
- **Sin escribir en él:** los tests analizan las colecciones en memoria, sin la caché del
  catálogo, así que no escriben nada en el submódulo. Los datasets que mide
  `scripts/benchmark.py` sí usan la caché, en `.cache/` del submódulo, que el catálogo
  ignora en git.
- **Actualizarlo** es un cambio explícito de PyPlanner: `git -C vendor/PDDLDomains fetch`,
  `git -C vendor/PDDLDomains checkout <commit>`, pasar los tests y hacer commit del
  submódulo. Un cambio del catálogo nunca llega a PyPlanner sin ese paso.

## Convenciones

Son las del catálogo (ver su `docs/es/catalogo.md`):

- Los ficheros se copian **sin modificar**. Hay dos excepciones, que se corrigen en el
  propio fichero: las erratas en `:requirements`, marcadas aquí con (*), y las
  discordancias de otro tipo cuando se decide corregirlas, marcadas con (‡). Cada
  colección explica las suyas en su ficha, `info.toml`.
- **Toda discordancia de requisitos se trata como una errata del fichero:** se añade a
  mano el requisito que falta, y el modo estricto no hace excepciones.
- **Modo estricto:** el planificador exige que toda construcción que se use tenga su
  requisito declarado. Si no se declara `:requirements`, se supone `:strips`.
- **Nombres únicos:** el parser no admite un nombre declarado dos veces en ningún sitio
  (tipos, dominio, acciones…). Varias erratas (‡) del catálogo son renombrados por eso.

## Notas

- Las IPC 2023 no tienen track temporal, de PDDL 3 ni de PDDL+. Por eso los niveles 1, 3
  (temporal), 4, 5 y 6 salen del archivo histórico de las IPC y de los benchmarks de
  SMTPlan. Los de PDDL 3 son de la IPC 2006: storage combina preferencias con
  restricciones sin tiempo (`sometime`, `at-most-once`) y una métrica con `is-violated`;
  trucks, restricciones con tiempo (`within`) sobre acciones durativas.
- Los benchmarks PDDL+ de ENHSP se han descartado porque usan `:constraint`, una
  extensión propia de ese planificador que no forma parte del estándar.
- (†) `greenhouse` **no es un benchmark oficial**. Se creó para PyPlanner para probar
  casos límite de expresividad de PDDL 2.1 temporal que no aparecen en ningún benchmark
  público encontrado (ni en las IPC ni en los de COLIN/OPTIC o VAL): efectos continuos
  dentro de `forall` y de `when`, efectos condicionales con condiciones `over all`,
  `?duration` no fijo en efectos `at start`, una duración cuyo máximo depende de una
  variable que cambia de forma continua y un efecto condicional sobre una variable
  continua. Sus planes se validan con VAL como los del resto.
  `tests/test_temporal_engine.py` lo usa además con otros problemas y pieza a pieza.
- Varios dominios usan funcionalidad de niveles anteriores: por ejemplo, `car` usa
  `:timed-initial-literals` y trucks necesita el motor temporal. Es lo esperado, porque
  la escala es acumulativa.


## Antes del catálogo

Hasta la versión 0.8, los problemas de prueba eran copias en `dominios/tierN-<nombre>/…`,
organizadas en *tiers*, uno por iteración del [roadmap](ROADMAP.md). Los tiers 1 a 6 son
los niveles 1 a 6 del catálogo, con los mismos ficheros (byte a byte, salvo el comentario
de cabecera de greenhouse). La tabla de equivalencias de las rutas antiguas está en
[benchmarks.md](benchmarks.md#rutas-anteriores).
