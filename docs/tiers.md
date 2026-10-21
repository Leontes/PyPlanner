# Tiers de evaluación

Los tests evalúan el planificador por **tiers**. Cada tier corresponde a una iteración del
[roadmap](ROADMAP.md). Al terminar la iteración *N* tienen que resolverse **todos** los
problemas de los tiers 1…*N*. Así, la funcionalidad nueva nunca puede romper la anterior.

Los problemas salen del catálogo [PDDLDomains](https://github.com/Leontes/PDDLDomains), que
está en el submódulo `vendor/PDDLDomains`, fijado en un commit. Los tiers son un dataset de
ese catálogo: [`tests/tiers.toml`](../tests/tiers.toml). Cada grupo es
`tier<N>/[<parte>/]<dominio>`, con los dos primeros problemas de una colección, que suelen
ser los más sencillos. [`tests/tiers.py`](../tests/tiers.py) lo lee y da las instancias a
los tests (`test_tiers_parser.py`, `test_tiers_solve.py`…).

| Tier | Iteración | Grupo | Colección del catálogo | Requisitos declarados |
|------|-----------|-------|------------------------|-----------------------|
| 1 | STRIPS | `tier1/blocksworld` | `ipc-2000/blocks-strips-untyped` | `:strips` |
| 1 | STRIPS | `tier1/gripper` | `ipc-1998/gripper-round-1-strips` | ninguno (STRIPS implícito) |
| 2 | PDDL 1.2 (ADL) | `tier2/rubiks-cube` | `ipc-2023-classical/rubiks-cube` | `:adl` |
| 2 | PDDL 1.2 (ADL) | `tier2/slitherlink` | `ipc-2023-classical/slitherlink` | `:typing :adl` |
| 3 | PDDL 2.1 numérico | `tier3/numeric/counters` | `ipc-2023-numeric/counters` | `:strips :typing :equality :adl :numeric-fluents` (*) |
| 3 | PDDL 2.1 numérico | `tier3/numeric/zenotravel` | `ipc-2023-numeric/zenotravel` | `:typing :fluents` (*) |
| 3 | PDDL 2.1 temporal | `tier3/temporal/driverlog` | `ipc-2002/driverlog-time-simple-automatic` | `:typing :durative-actions :numeric-fluents` (*) |
| 3 | PDDL 2.1 temporal | `tier3/temporal/zenotravel` | `ipc-2002/zenotravel-time-simple-automatic` | `:durative-actions :typing :numeric-fluents` (*) |
| 3 | PDDL 2.1 temporal | `tier3/temporal/greenhouse` | `propios/greenhouse` (†) | `:typing :durative-actions :numeric-fluents :continuous-effects :duration-inequalities :conditional-effects :negative-preconditions` |
| 4 | PDDL 2.2 | `tier4/derived-predicates/psr-middle` | `ipc-2004/psr-middle-derived-predicates-adl` | `:adl :derived-predicates` |
| 4 | PDDL 2.2 | `tier4/timed-initial-literals/satellite` | `ipc-2004/satellite-time-time-windows-strips` (‡) | `:strips :equality :typing :fluents :durative-actions :timed-initial-literals` |
| 5 | PDDL 3 | `tier5/constraints/trucks` | `ipc-2006/trucks-time-constraints` | `:typing :adl :durative-actions :fluents :constraints` |
| 5 | PDDL 3 | `tier5/preferences/storage` | `ipc-2006/storage-preferences-qualitative` (‡) | `:typing :adl :preferences :constraints` |
| 6 | PDDL+ | `tier6/car` | `smtplan/car-nodrag` (‡) | `:typing :durative-actions :fluents :time :negative-preconditions :timed-initial-literals` |
| 6 | PDDL+ | `tier6/generator` | `smtplan/generator-events` (*) (‡) | `:fluents :durative-actions :duration-inequalities :adl :typing :time :continuous-effects` |

Los identificadores de pytest son `<grupo>/<problema>` (`tier3/temporal/driverlog/p01`).
`SUPPORTED_TIER` y `PARSER_TIER` (en `tests/tiers.py`) dicen hasta qué tier tienen que
llegar el planificador y el parser.

## El catálogo en el submódulo

- **Inicializarlo:** `git clone --recursive`, o en un clon ya hecho,
  `git submodule update --init vendor/PDDLDomains`. Sin él, los tests que usan los tiers
  fallan con un error que lo dice.
- **Siempre el del submódulo:** `tests/tiers.py` importa el paquete `pddldomains` del
  propio submódulo (no uno instalado) y usa su directorio `dominios/` (no
  `$PDDL_DOMINIOS`). Si se importara otro, da un error.
- **Sin escribir en él:** los grupos de `tests/tiers.toml` se resuelven directamente en el
  catálogo, sin su analizador, así que los tests no escriben nada en el submódulo. Los
  datasets que mide `scripts/benchmark.py` sí lo usan, y guardan su caché en `.cache/` del
  submódulo, que el catálogo ignora en git.
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

Para añadir problemas a un tier basta con añadirlos a `problems` de su grupo en
`tests/tiers.toml`; para añadir un dominio, un `[[group]]` nuevo.

## Notas

- Las IPC 2023 no tienen track temporal, de PDDL 3 ni de PDDL+. Por eso los tiers 1, 3
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
- Varios dominios usan funcionalidad de tiers anteriores: por ejemplo, `car` usa
  `:timed-initial-literals` y trucks necesita el motor temporal. Es lo esperado, porque
  cada tier incluye a los anteriores.

## Antes del catálogo

Hasta la versión 0.8, los tiers eran copias en `dominios/tierN-<nombre>/…` y las
colecciones de medida, en `benchmarks/`. Son los mismos ficheros que las colecciones del
catálogo (con el mismo contenido byte a byte, salvo el comentario de cabecera de
greenhouse). La tabla de equivalencias para reproducir las medidas antiguas está en
[benchmarks.md](benchmarks.md#rutas-anteriores).
