# Roadmap de PyPlanner

Objetivo: un planificador PDDL escrito en Python que soporte **todas** las versiones de
PDDL hasta PDDL+, incluido PDDL 3. El motor es una **búsqueda heurística hacia delante**. El
proyecto avanza en 6 iteraciones y cada una amplía el lenguaje que se acepta:

| Iteración | Lenguaje | Versión | Estado |
|-----------|----------|---------|--------|
| 1 | STRIPS | v0.1 | ✅ |
| 2 | PDDL 1.2 (subconjunto ADL) | v0.2 | ✅ |
| 3 | PDDL 2.1 (3a numérico con `:action-costs`, 3b temporal) | v0.3 | ✅ |
| 4 | PDDL 2.2 (predicados derivados y TIL) | v0.4 | ✅ |
| 5 | PDDL 3 (restricciones de trayectoria y preferencias; `:object-fluents`) | v0.5 | ✅ |
| 6 | PDDL+ (procesos y eventos) | v0.6 | ✅ |

**Planificación original cerrada** (2026-10-07): las 6 iteraciones están completas y los
tiers 1 a 6 se resuelven y validan con VAL con todos los portfolios. Lo que sigue es el
trabajo opcional: técnicas del repertorio y un estudio de la integración de HTN (ver
"Trabajo opcional").

Referencia principal del lenguaje: [planning.wiki](https://planning.wiki/ref/pddl/requirements).
PDDL 3 se incluye para no dejar ninguna versión a medias, no por una necesidad concreta.

## Reglas comunes a todas las iteraciones

En cada iteración hay que:

1. **Parser**: ampliar las dos fases (PDDL → JSON y JSON → AST), el esquema JSON y el
   AST para aceptar los nuevos requisitos. Si un fichero declara un requisito que todavía
   no está soportado, se rechaza con un error claro.
   **Modo estricto:** toda construcción tiene que tener su requisito declarado en
   `:requirements` (en el dominio o en el problema). Si un fichero usa, por ejemplo,
   `:types` sin declarar `:typing`, se rechaza aunque `:typing` esté soportado. Si no se
   declara ningún requisito, se supone `:strips`. La comprobación se hace en las dos fases,
   porque el JSON puede venir de otra fuente, con las mismas reglas construcción →
   requisito, que están en un solo sitio (`requirements.py`).
2. **Motor**: ampliar el grounding, el modelo de estado, la aplicación de acciones y la
   heurística para la nueva semántica.
3. **Documentación**: docstrings en todo el código público y notas de diseño en `docs/`.
4. **Tests**:
   - unitarios del parser y del motor;
   - de integración: evaluación por tiers con los dominios de `dominios/` (ver más abajo);
   - de validación de planes con VAL. Todo plan que se genere tiene que validarse.
   - La suite de las iteraciones anteriores tiene que seguir pasando: no se admiten regresiones.
5. **Cierre**: `pytest`, `ruff` y `mypy --strict` sin errores. Después, commit y tag `v0.N`.

## Arquitectura (se fija en la iteración 1 y se amplía después)

```
src/pyplanner/
  pddl/            fase 1: lexer (s-expresiones) → parser → documento JSON; conversor inverso
  schema/          esquema JSON versionado (JSON Schema): el contrato entre las dos fases
  loader/          fase 2: JSON → validación semántica → AST tipado (dataclasses inmutables)
  requirements.py  requisitos soportados y reglas construcción → requisito (modo estricto)
  grounding/       instanciación con alcanzabilidad, normalización ADL y numérica
                   (normalize, numeric), tarea instanciada (task: estado, acciones,
                   objetivos), axiomas de los predicados derivados (axioms), motor
                   temporal (temporal, temporal_lp), restricciones de trayectoria
                   (monitors), preferencias compiladas a costes (preferences) y motor
                   con el tiempo discretizado de PDDL+ (discrete, con la integración
                   de los procesos en flows)
  heuristics/      blind, goalcount, h_max, h_add y h_FF con sus extensiones numéricas y
                   temporales (plazos de los TIL); proyección de las invariantes con
                   procesos (projection); interfaz por lotes; referencia en Python,
                   Numba (compiled) y PyTorch (tensor), y evaluación híbrida
  search/          GBFS, A*, WA* e IW sobre una interfaz común: configuraciones, novedad,
                   sucesores compilados, límite de memoria y búsqueda anytime (anytime)
  portfolio/       ejecución paralela de configuraciones (procesos, cancelación)
  plan/            representación del plan; salida en formato IPC (la que lee VAL) o en JSON
  planner.py       fachada: grounding, búsqueda o portfolio, y comprobación del plan
  frontend.py      lectura de dominios y problemas en PDDL o en JSON
  cli.py
```

Fuera de `src/`: `dominios/` (benchmarks de los tiers, que recorren los tests),
`benchmarks/` (colecciones más grandes para medir el rendimiento, fuera de los tests,
con `scripts/benchmark.py`), `tests/` y `tools/` (VAL compilado en local).

Decisiones (tomadas en la iteración 1):
- **Parser propio sin dependencias.** PDDL es s-expresiones, así que basta un lexer
  sencillo y un descenso recursivo. Así controlamos los mensajes de error y cada
  construcción del AST corresponde a un requisito.
- **Parser en dos fases unidas por JSON.** El planificador se integrará en un proyecto
  más grande en el que las partes se comunican con JSON. Por eso:
  - **Fase 1 (PDDL → JSON)**: solo sintaxis. Convierte los ficheros PDDL en un documento
    JSON que sigue un esquema publicado.
  - **Fase 2 (JSON → objetos internos)**: el punto de entrada del planificador. No da por
    bueno su JSON de entrada, porque en el proyecto final puede venir de otra fuente.
    Valida la estructura con el esquema y la semántica (tipos, aridades, variables libres,
    requisitos declarados).
  - La validación estructural se hace con `jsonschema`.
  - El esquema lleva un campo `version` y cada iteración solo le añade cosas nuevas.
  - Cada nodo puede llevar una posición opcional (`loc`: fichero, línea, columna) para
    que los errores de la fase 2 apunten al PDDL original.
  - Hay un conversor inverso JSON → PDDL. Con él los tests comprueban el ida y vuelta
    PDDL → JSON → PDDL.
  - La CLI permite ejecutar cada fase por separado (`pyplanner parse`, `pyplanner solve`
    sobre JSON o sobre PDDL).
- **Separar el AST (lo que se lee) de la tarea instanciada (lo que se busca).** El
  parser no sabe nada de la búsqueda.
- **Identificadores en inglés; docstrings y documentación en español.**

## Motor de búsqueda: diseño y repertorio

No se trata de revolucionar el estado del arte, sino de tener un planificador al día que
podamos controlar y modificar. Para eso hay que conocer desde el principio el repertorio de
técnicas disponible y diseñar el motor para poder incorporarlas sin rehacerlo.

**Configuración de búsqueda.** Una búsqueda queda definida por una configuración:
algoritmo, heurística, desempate, uso de acciones útiles, evaluación perezosa o
inmediata, **tamaño de lote**, **dispositivo de evaluación** (Python, CPU o GPU con
PyTorch), semilla y límites de tiempo, memoria y expansiones. Toda búsqueda se puede
**cancelar desde fuera**, comprobando periódicamente una señal de parada.

**Paralelismo: portfolio.** Por el GIL de Python, el paralelismo real exige procesos y no
hilos. El planificador usa un **portfolio paralelo**:
- N procesos, cada uno con una configuración distinta; gana la primera solución y el
  resto se cancelan;
- la tarea se instancia una vez y los hijos la heredan con `fork` (copia bajo demanda);
- las configuraciones tienen que ser **diversas** (heurística, algoritmo, desempate,
  semilla…), porque N búsquedas deterministas idénticas encuentran lo mismo al mismo tiempo;
- coste: la memoria se multiplica por N (cada proceso tiene su propia frontera).

**Evaluación de estados por lotes (CPU o GPU con PyTorch).** Se incorpora desde la
iteración 1 porque condiciona la interfaz de la heurística y el ciclo de la búsqueda, que
son lo más caro de cambiar después. No necesita Python sin GIL: el paralelismo está dentro
de cada llamada a PyTorch, no en hilos de Python.
- **Interfaz:** toda heurística expone `evaluate_batch(states) -> valores`. La
  evaluación de un solo estado es un lote de tamaño 1.
- **Backends:** cada heurística tiene una **implementación de referencia en Python puro**
  (siempre disponible, la que usan los tests) y, si se puede expresar con tensores, una
  implementación en PyTorch que funciona en CPU o en GPU. Las dos tienen que dar los mismos
  valores; los tests lo comprueban en todas las instancias soportadas.
- **Tarea tensorial:** tras el grounding, la tarea se compila una vez a tensores
  (matriz de precondiciones, matriz de efectos, costes). Los estados en bitset se
  convierten en lotes de vectores binarios.
- **Qué se puede pasar a tensores:**
  - h_max y h_add: sí, como un punto fijo de operaciones min/suma sobre matrices.
  - h_FF: los costes de h_add se calculan en tensores; la extracción del plan relajado va
    en tensores o en CPU según rinda.
  - Heurísticas numéricas y temporales: se estudian en su iteración. Si una no se puede
    pasar a tensores, se queda solo con la implementación de referencia.
- **Búsqueda por lotes:** con `batch_size = 1` la búsqueda es la clásica. Con `k > 1`,
  se sacan hasta *k* nodos de la cola, se expanden y sus sucesores se evalúan juntos.
  El orden de expansión cambia ligeramente, y eso es aceptable.
- **GPU y portfolio:**
  - CUDA no es compatible con `fork` una vez inicializado. Los miembros del portfolio
    que usen la GPU se lanzan con `spawn`, y el proceso padre nunca inicializa CUDA.
  - Cada proceso con CUDA reserva su propio contexto en la GPU (cientos de MB). Con la
    GPU actual, una GTX 970 de 4 GB, habrá como mucho 1 o 2 miembros en GPU; el resto va
    en CPU.
- **Dependencia opcional:** PyTorch no es obligatorio (`pip install -e ".[gpu]"`). Sin
  PyTorch, el planificador funciona con los backends de Python y los tests de PyTorch se
  saltan.
- **Expectativa realista:** la GPU compensa con tareas grandes y lotes grandes. En tareas
  pequeñas manda el coste de pasar los datos entre CPU y GPU, y gana la CPU. El portfolio
  permite tener las dos opciones a la vez.

**Descartado: búsqueda distribuida (HDA\*, reparto de la frontera).** Repartir los estados
entre procesos obliga a serializarlos y enviarlos. En Python ese coste supera a la ganancia.
Solo se reconsideraría si el núcleo del motor pasara a un lenguaje compilado.

**Prioridades de eficiencia.** Antes que el paralelismo, que mejora como mucho en un factor
igual al número de núcleos, pesan más:
1. una representación compacta del estado;
2. buenas heurísticas;
3. acciones útiles;
4. evaluación perezosa.

El portfolio se monta encima de todo eso.

**Repertorio de referencia** (para tenerlo presente, no para implementarlo todo):

| Técnica | Qué aporta | Estado |
|---------|------------|--------|
| GBFS, A*, WA* | búsquedas base (satisfacible / óptima / acotada) | iteración 1 |
| h_max, h_add, h_FF | heurísticas de relajación | iteración 1 |
| Acciones útiles, evaluación perezosa | menos evaluaciones de la heurística | iteración 1 |
| Portfolio paralelo | robustez y aprovechamiento de varios núcleos | final de la iteración 1 |
| Exploración por tipos (Xie et al., 2014) | salir de mesetas y zonas con callejones sin salida | iteración 2 |
| Heurísticas compiladas con Numba | h_max, h_add y h_FF 12–19 veces más rápidas | iteración 2 |
| Generación de sucesores compilada con Numba | mismo orden que la referencia; del 46% al 5% del tiempo de búsqueda | iteración 2 |
| Enforced hill-climbing (FF) | búsqueda local rápida | opcional (5.º) |
| Paseos aleatorios (Arvand) | diversidad en el portfolio | opcional (5.º) |
| Colas alternadas / multiheurística | combinar heurísticas en una sola búsqueda | opcional (2.º, con landmarks) |
| Landmarks (h_LM) | heurística informada más potente | opcional (2.º) |
| Búsqueda por novedad (IW, BFWS) | salir de mesetas; diversidad en el portfolio | final de la iteración 3b ✅ |
| HDA\*, frontera distribuida | escalar una sola búsqueda | descartado (Python) |
| Evaluación de la heurística por lotes (PyTorch, CPU o GPU) | muchos estados evaluados a la vez | iteración 1 |
| Búsqueda completa en GPU | expansión masiva en paralelo | descartado (ver abajo) |
| Lotes adaptativos y evaluación híbrida (Numba en CPU / PyTorch en GPU) | lotes grandes solo cuando compensan, en el dispositivo que más rinde | final de la iteración 3b ✅ |

**Sobre la búsqueda completa en GPU.** Hay trabajos que hacen toda la búsqueda en GPU
(expansión, detección de estados repetidos y cola de prioridad). Pero la búsqueda es
irregular: cada estado tiene un número distinto de sucesores, hay muchas bifurcaciones
en el código y la cola de prioridad es secuencial. Llevarla a GPU exige rediseñar el
motor entero. Solo se lleva a GPU la evaluación de la heurística, que sí es regular.

## Evaluación por tiers

Los benchmarks están en `dominios/tierN-*/` (detalles y procedencia en
[dominios/README.md](../dominios/README.md)). Cada tier corresponde a una iteración:

| Tier | Contenido | Obligatorio desde |
|------|-----------|-------------------|
| 1 | STRIPS | Iteración 1 |
| 2 | PDDL 1.2 (ADL) | Iteración 2 |
| 3 | PDDL 2.1 (numérico y temporal) | Iteración 3 |
| 4 | PDDL 2.2 (predicados derivados y TIL) | Iteración 4 |
| 5 | PDDL 3 (preferencias y restricciones de trayectoria) | Iteración 5 |
| 6 | PDDL+ | Iteración 6 |

Al terminar la iteración *N*, todos los problemas de los tiers 1…*N* tienen que resolverse
con un plan válido según VAL. Los tiers superiores se saltan. En los tests, el tier
máximo soportado es una constante que se sube al cerrar cada iteración.

---

## Iteración 1: STRIPS

**Requisitos:** `:strips`.

- **Esquema JSON v1**: dominio y problema STRIPS, con validación y documentación.
- **Parser:** `define`, `domain`, `problem`, `:requirements`, `:predicates`, `:constants`,
  `:objects`, `:action` (`:parameters`, `:precondition` y `:effect` como conjunciones de
  átomos positivos, más `not` en los efectos), `:init` y `:goal`. Errores con número de línea.
- **Parser:** ✅ hecho (fases 1 y 2, esquema v1, conversor inverso, CLI `parse`/`check`).
- **Motor:** ✅ hecho (ver [motor-busqueda.md](motor-busqueda.md)):
  - grounding con poda por alcanzabilidad relajada y eliminación de hechos estáticos;
  - estado compacto: bitset sobre enteros de Python;
  - búsquedas GBFS, A* y WA*, inmediatas o perezosas, **configurables y cancelables**;
  - heurísticas blind, goalcount, h_max, h_add y h_FF, con acciones útiles;
  - interfaz de evaluación por lotes. h_max, h_add y h_FF tienen implementación de
    referencia (Python) y en PyTorch (CPU y GPU), con valores idénticos. La búsqueda
    admite `batch_size`;
  - salida del plan en formato IPC (para VAL) y en JSON;
  - **portfolio paralelo** con varias configuraciones, con portfolios predefinidos para
    1, 2, 4, 5, 6 y 7 núcleos (`-j N`; por defecto 6, o los que tenga la máquina; los de 6
    y 7 desde la iteración 3b).
- **Tests:** ✅ tier 1 (blocksworld y gripper) resuelto y validado con VAL, con el
  portfolio de cada número de núcleos. También hay problemas sin solución hechos
  a mano, tests del portfolio y tests de equivalencia entre la referencia y PyTorch.

## Iteración 2: PDDL 1.2 (subconjunto ADL)

**Requisitos:** `:typing`, `:negative-preconditions`, `:disjunctive-preconditions`,
`:equality`, `:existential-preconditions`, `:universal-preconditions`,
`:quantified-preconditions`, `:conditional-effects` y `:adl`.

- **Parser:** ✅ hecho (esquema JSON v2, modo estricto en las dos fases, tipos y ADL;
  ver [formato-json.md](formato-json.md)):
  - jerarquía de tipos (`either` incluido);
  - fórmulas arbitrarias con `not`, `or`, `imply`, `exists`, `forall` y `=`;
  - efectos `forall` y `when`;
  - expansión de los metarrequisitos (`:adl` y `:quantified-preconditions`).
- **Motor:** ✅ hecho (ver [motor-busqueda.md](motor-busqueda.md)):
  - normalización instanciada: cuantificadores expandidos sobre los objetos de su tipo,
    fórmulas a DNF con simplificación (igualdad, hechos estáticos, contradicciones,
    subsunción) y una acción por disyunto;
  - precondiciones y objetivos negativos exactos en el estado; en las heurísticas, hechos
    complementarios solo dentro de la relajación;
  - efectos condicionales evaluados sobre el estado anterior e indexados por disparador;
    en la relajación, una acción relajada por efecto, y h_FF cuenta cada operador una vez;
  - objetivos disyuntivos (mínimo entre cláusulas);
  - implementación en PyTorch adaptada, con equivalencia exacta con la referencia;
  - **exploración por tipos** en la búsqueda y portfolios rehechos a partir de ella;
  - **heurísticas compiladas con Numba** (dispositivo `numba`, por defecto con `auto`),
    idénticas a la referencia, y generación de sucesores compilada con el mismo orden
    canónico que la referencia: slitherlink p02 pasa de 22 s a 2,4 s.
- **Tests:** ✅ tier 2 (rubiks-cube y slitherlink) resuelto y validado con VAL con los 4
  portfolios; tareas ADL hechas a mano para cada construcción, con sus planes validados
  con VAL.

**Fuera de alcance (propuesta):** los requisitos obsoletos o casi sin uso de PDDL 1.2:
`:action-expansions`, `:foreach-expansions`, `:dag-expansions`, `:domain-axioms`,
`:subgoals-through-axioms`, `:safety-constraints`, `:expression-evaluation`,
`:open-world`, `:true-negation`, `:ucpop` y `:fluents` con la semántica de 1.2.

## Iteración 3: PDDL 2.1

Es la iteración más grande, así que se divide en dos.

### 3a: fluentes numéricos
**Requisitos:** `:numeric-fluents` (y `:fluents`), y `:action-costs` (PDDL 3.1).

- **Parser:** ✅ hecho (esquema JSON v3; ver [formato-json.md](formato-json.md)).
- **Motor:** ✅ hecho (ver [motor-busqueda.md](motor-busqueda.md#motor-numérico-iteración-3a)):
  estado `State(facts, values)`, funciones estáticas y relevancia, métrica como coste de
  acción, *h^hbd* en Python, Numba y PyTorch con resultados idénticos, y desempate
  (coste, nivel) para los costes 0. Resuelve el tier 3 numérico, validado con VAL.
- **Tests:** ✅ tier 3 numérico (counters y zenotravel de la IPC 2023) resuelto y validado
  con VAL con todos los portfolios; parser numérico y de costes de acción, semántica de
  los efectos numéricos, *h^hbd* y equivalencia exacta entre Python, Numba y PyTorch.

Solo lo necesario para PDDL 2.1, más `:action-costs`. Es una forma restringida de
fluentes numéricos (`total-cost`, costes fijos por acción), pero la usa casi todo el
track clásico de las IPC 2023. El resto de PDDL 3 va a la iteración 5.
- **Parser:**
  - `:functions`;
  - expresiones aritméticas;
  - comparaciones;
  - efectos `assign`, `increase`, `decrease`, `scale-up` y `scale-down`;
  - `:metric` en el problema.
- **Motor:**
  - estado lógico más vector numérico;
  - heurística numérica relajada al estilo Metric-FF o ENHSP (h_add/h_FF sobre intervalos);
  - A* sobre la métrica, cuando la haya;
  - `:action-costs`: costes por acción en el motor y las heurísticas. Admite costes 0, así
    que hay que revisar el corte temprano del Dijkstra, que ahora supone costes positivos.

### 3b: acciones durativas
**Requisitos:** `:durative-actions`, `:duration-inequalities` y `:continuous-effects`.

- **Parser:** ✅ hecho (esquema JSON v4; ver [formato-json.md](formato-json.md#acciones-durativas-v4)).
  `?duration` solo en la duración y en los efectos numéricos, como en la gramática y en VAL.
- **Motor:** ✅ hecho (ver
  [motor-busqueda.md](motor-busqueda.md#motor-temporal-iteración-3b)): acciones de inicio y
  fin, órdenes al estilo POPF, red temporal con duraciones fijas o con desigualdades,
  invariantes, efectos numéricos, efectos condicionales de inicio a fin, planes temporales
  y programación lineal al estilo COLIN (HiGHS, con SciPy como dependencia) para los
  efectos continuos y `?duration` con duraciones no fijas. Resuelve el tier 3 temporal,
  validado con VAL, y un dominio de certificación hecho a mano con las combinaciones poco
  habituales (efectos continuos en `forall` y `when`, `when` con `over all`, `?duration` no
  fijo al empezar, duraciones que dependen de variables continuas). Quedan fuera, con un
  error claro, los límites de la programación lineal (disyunciones sobre variables
  continuas, que exigirían programación lineal entera, y expresiones no lineales).
- **Tests:** ✅ tier 3 temporal (driverlog y zenotravel de la IPC 2002, y greenhouse)
  resuelto y validado con VAL con todos los portfolios; parser temporal, órdenes, red
  temporal, invariantes, programación lineal y greenhouse pieza a pieza; búsqueda por
  novedad, lotes adaptativos y portfolio de arranque mixto.
- **Parser:**
  - `:durative-action` y `:duration`;
  - condiciones `at start`, `at end` y `over all`;
  - efectos temporales;
  - `#t` y `?duration`.
- **Motor:**
  - acciones divididas en *snap actions* (inicio y fin);
  - búsqueda hacia delante sobre estados temporales (acciones en ejecución y comprobación
    de invariantes `over all`);
  - planificación temporal con **STN al estilo POPF** (decidido; ver "Decisiones
    tomadas"): la consistencia de cada estado se comprueba con la red temporal, y la
    programación lineal (al estilo COLIN) solo entra con efectos continuos o duraciones
    que dependen del estado;
  - efectos continuos lineales;
  - salida de planes temporales (`t: (acción) [d]`).
- **Al final de la 3b (opcional):** ✅ búsqueda por **novedad** (IW y BFWS; Lipovetzky y
  Geffner, 2012 y 2017). BFWS con h_FF y acciones útiles es la mejor configuración
  individual en las pruebas y entra en todos los portfolios (ver
  [motor-busqueda.md](motor-busqueda.md#búsqueda-por-novedad)).
- **Al final de la 3b (opcional):** ✅ **lotes adaptativos y evaluación híbrida** (Numba en
  CPU / PyTorch en GPU), con umbral calibrado por tarea y portfolio de arranque mixto
  (ver [motor-busqueda.md](motor-busqueda.md#lotes-adaptativos-y-evaluación-híbrida)). La
  GPU recorta un 30–55% en las tareas grandes; en los portfolios aporta poco en las
  pruebas, porque la búsqueda perezosa ya resuelve rápido esas tareas.

## Iteración 4: PDDL 2.2

**Requisitos:** `:derived-predicates` y `:timed-initial-literals`.
- **Parser:** ✅ hecho (esquema JSON v5; ver [formato-json.md](formato-json.md)):
  - reglas `:derived`, varias por predicado y recursivas, con comprobación de
    estratificación y prohibidas en efectos y en el estado inicial;
  - literales `(at t literal)` en `:init`, solo en dominios temporales.
- **Motor:** ✅ hecho (ver [motor-busqueda.md](motor-busqueda.md#predicados-derivados-iteración-4)):
  - ✅ evaluación de los axiomas por estratos, con punto fijo en cada estado (también en
    el motor temporal, donde leer un hecho derivado ordena tras lo que lo determina);
  - ✅ soporte de los axiomas en la heurística: acciones relajadas de coste 0, y las
    negaciones de hechos derivados se dan por ciertas (como Fast Downward);
  - ✅ los TIL se tratan como eventos obligatorios en el eje temporal del motor de la
    iteración 3b: acciones `til@k` encadenadas, con instante fijo en la red temporal y
    plazos al estilo de OPTIC para los pasos anteriores;
  - ✅ plazos en la heurística (parte A de la heurística temporal): instantes más
    tempranos y ventanas de los TIL en la relajación, para descartar estados sin salida.
    La parte B (plan relajado temporal) queda como trabajo opcional.
- **Tests:** ✅ tier 4 (psr-middle con predicados derivados y satellite con ventanas de
  visibilidad) resuelto y validado con VAL con todos los portfolios. Medidas aparte en
  `benchmarks/` (satellite p01–p20).
- **Correcciones del motor temporal de la 3b** que destaparon los TIL (ver
  [motor-busqueda.md](motor-busqueda.md#motor-temporal-iteración-3b)): la búsqueda solo
  cambia el padre de un nodo temporal sin expandir, y las órdenes separan de los lectores
  toda escritura real (también las que no cambian el valor) y cuentan los hechos
  `running@a`.

## Iteración 5: PDDL 3

**Requisitos:** `:constraints`, `:preferences` y `:object-fluents` (PDDL 3.1; `:action-costs`
ya está desde la 3a).

Se divide en cuatro partes (decidido el 2026-10-05): 5a parser; 5b restricciones duras como
monitores en el motor (las de tiempo, sobre la maquinaria de los TIL; en dominios sin
acciones durativas, el instante es el número de pasos); 5c preferencias compiladas a costes
con búsqueda *anytime* que devuelve el mejor plan dentro del límite de tiempo; 5d funciones
de objetos compiladas a predicados (VAL no las admite: los planes se validan contra el
dominio compilado).

- **Parser (5a):** ✅ restricciones de trayectoria, preferencias, `is-violated` y funciones
  de objetos (esquema JSON v6; ver [formato-json.md](formato-json.md#pddl-3-v6)), con la
  gramática de Gerevini y Long (2005); el motor rechaza las restricciones y las
  preferencias hasta la 5b y la 5c.
- **Restricciones duras (5b):** ✅ monitores en el motor para los diez operadores, en tareas
  clásicas (el instante es el número de pasos) y temporales (marcas de instante como TIL
  sin literales; `always-within` añade aristas a la red temporal). Sus salidas buenas son
  reglas relajadas de la heurística. Resuelve trucks (plazos `within`), validado con VAL
  (ver [motor-busqueda.md](motor-busqueda.md#restricciones-de-trayectoria-iteración-5b)).
- **Preferencias (5c):** ✅ compiladas a costes (Keyder y Geffner) con los pesos de la
  métrica lineal: una fase final (`_end`, `_collect` y `_forgo`) para las del objetivo y
  las que se deciden al final del plan; las restricciones-preferencia que se incumplen a
  mitad del plan (monitores blandos) cobran su peso en el paso que las incumple; variantes
  de las acciones para las de las precondiciones. Las restricciones triviales se descartan
  y los `forall` de restricciones se instancian con un join. Búsqueda *anytime* al estilo
  LAMA (primer plan con coste unitario, la configuración elegida con incumbente, WA* con
  pesos 5, 3, 2 y 1, y A* con h_max) con poda por cota, también en el portfolio con la cota
  compartida; 60 s por defecto. Resuelve storage p01 y p02 con el óptimo demostrado
  (métricas 0 y 1), validado con VAL (ver
  [motor-busqueda.md](motor-busqueda.md#preferencias-y-búsqueda-anytime-iteración-5c)).
  Las preferencias en tareas temporales quedan como trabajo opcional.
- **Funciones de objetos (5d):** ✅ adelantada a la 5a: la fase 2 las compila a predicados
  sobre el documento JSON, antes del AST, así que el motor no las ve y funcionan con todo
  lo demás. Planes clásicos y temporales validados con VAL contra el dominio compilado.
- **Parser (plan inicial):**
  - sección `:constraints` (en dominio y problema) con los operadores modales `always`,
    `sometime`, `at-most-once`, `sometime-before`, `sometime-after`, `at end`, `within`,
    `always-within`, `hold-during` y `hold-after`;
  - `preference` en objetivos, precondiciones y restricciones;
  - `is-violated` en la métrica;
  - funciones de objetos (`:object-fluents`).
- **Motor:**
  - **restricciones sin tiempo** compiladas a monitores: hechos nuevos actualizados con
    efectos condicionales en cada acción (reutiliza la normalización de la iteración 2);
  - **restricciones con tiempo** sobre el reloj del motor temporal de la iteración 3b;
  - **preferencias**: los objetivos blandos se compilan a objetivos duros con acciones de
    "renuncia" con coste (Keyder y Geffner, 2009), y una **búsqueda *anytime*** sigue
    mejorando el plan hasta agotar el tiempo. Las preferencias en precondiciones y
    restricciones cuentan sus incumplimientos;
  - funciones de objetos compiladas a predicados (como en Fast Downward).
- **Tests:** ✅ tier 5 (storage con preferencias cualitativas y trucks con restricciones
  con tiempo) resuelto y validado con VAL con todos los portfolios, y la métrica de los
  planes comparada con la de VAL en todos los tiers. VAL admite las restricciones y las
  preferencias, con dos fallos: `hold-after` (esos planes se comprueban con la simulación
  propia) y una sección `(:constraints (and))` vacía, que le hace dar por incumplidas las
  preferencias del objetivo. Medidas aparte en `benchmarks/` (storage p01–p20).
- **Correcciones del motor** que destapó la iteración: una búsqueda que agota la cola sin
  poder demostrar que no hay plan (IW, o una tarea temporal que ha descartado estados
  repetidos con otra línea temporal) termina con `failed`, no con `unsolvable`; y el join
  del grounding usa un índice por posiciones ligadas (mismo resultado, más rápido).
- **Carga estimada:** la mayor de todas. Las preferencias cambian el objetivo de la
  búsqueda: ya no basta el primer plan, importa su calidad.

## Iteración 6: PDDL+

**Requisitos:** `:time`.

Se divide en tres partes (decidido el 2026-10-07):

- **6a, parser** (esquema JSON v7): ✅ `:process` y `:event` (parámetros, precondición y
  efecto; en los procesos, efectos continuos con `#t`), y literales negados en `:init`
  (válidos en PDDL 3.1; redundantes con el mundo cerrado, se ignoran). Ver
  [formato-json.md](formato-json.md#pddl-v7). El motor rechaza procesos y eventos hasta
  la 6b.
- **6b, motor discretizado** (al estilo de ENHSP, DiNo o UPMurphi): ✅ tier 6 resuelto y
  validado con VAL; car-nodrag 10 de 10 y generator-events 8 de 8 con `bfws-ff-pref` (ver
  [motor-busqueda.md](motor-busqueda.md#procesos-y-eventos-iteración-6b)). Además de lo
  previsto, los eventos se disparan en su instante exacto (bisección sobre el flujo, como
  VAL), y las variables continuas no cuentan para la novedad.
  - la búsqueda tiene las acciones instantáneas y una acción "esperar δ", que integra los
    procesos activos durante δ, dispara los eventos en cascada y comprueba las
    invariantes;
  - **δ = 1 por defecto**, configurable. El plan se valida además con un δ más fino; si
    falla (por ejemplo, un evento que debía dispararse a mitad de un paso), se reintenta
    con δ/2 un número limitado de veces;
  - **integración exacta** de los procesos con series de Taylor, que terminan en las
    dinámicas polinómicas (car, generator) y dan los mismos valores que VAL; si la serie no
    termina (un crecimiento exponencial, por ejemplo), Runge-Kutta 4 con subpasos;
  - **acciones durativas** de los dominios PDDL+ compiladas en el motor discretizado: un
    inicio, un proceso que hace de reloj y un fin cuando el reloj llega a la duración; las
    condiciones `over all` son invariantes. Las tareas temporales sin `:time` siguen en el
    motor de la 3b, que es más preciso;
  - plan con marcas de tiempo y las acciones simultáneas separadas en ε (VAL no admite dos
    acciones en el mismo instante). VAL simula hasta el último *happening*: no puede
    validar un plan cuyo objetivo solo se cumple esperando tras la última acción, que se
    comprueba con la simulación propia.
- **6c, heurística y rendimiento:** ✅ (ver
  [motor-busqueda.md](motor-busqueda.md#heurística-con-procesos-y-eventos-iteración-6c)).
  Los procesos (su ritmo por δ) y los eventos ya estaban en la relajación desde la 6b; lo
  que faltaba eran las invariantes, que la relajación no ve:
  - duraciones fijas comparadas con el reloj tal cual, relojes a 0 desde el inicio y una
    asignación constante que solo logra una condición si su valor la cumple;
  - **proyección**: la heurística suma lo que una espera sin hacer nada no cubriría de la
    duración obligatoria de las acciones en curso, con anticipación al empezar una con
    invariante, y las acciones útiles son las que reducen ese faltante;
  - la proyección ve las rupturas que no duran hasta el final de un tramo (las raíces de
    las condiciones a lo largo del flujo exacto);
  - rendimiento: menos redondeos y comprobaciones durante las esperas.
  
  Las esperas más largas cuando no cambia nada no hicieron falta: con la heurística, las
  1000 esperas de generator se recorren sin desviarse (unas 1000 expansiones).
  Colecciones de `benchmarks/` con `bfws-ff-pref` y `lazy-ff-pref`: de 66 de 86 búsquedas
  resueltas a 86 de 86, todas válidas para VAL; generator-toricelli, de 5 y 1 de 9 a 9 y
  9. Los tiers 1 a 5 hacen las mismas búsquedas con el mismo coste.

**Tests:** ✅ tier 6 (car y generator, de SMTPlan) resuelto y validado con VAL con todos
los portfolios, más medidas en `benchmarks/` con colecciones de PDDL+ (car-nodrag,
generator-events y las de efectos continuos: 86 de 86 búsquedas con `bfws-ff-pref` y
`lazy-ff-pref`). Erratas de los ficheros del tier 6 (‡): en car,
funciones sin paréntesis (`(= d 0)`); en generator, un guion pegado al tipo (`?t -tank`) y
sin valor inicial para `(ptime ?t)`, que el proceso de repostaje usa (ni VAL ni el
planificador admitían ningún plan que repostara).

---

## Pendiente de decidir

Nada por ahora.

## Decisiones tomadas

- **Motor temporal: STN al estilo POPF** (2026-09-29), frente a *decision epochs* al estilo
  TFD. Los *decision epochs* son más sencillos y rápidos, pero incompletos con
  concurrencia obligatoria (`generator`, en el tier 6, y ventanas de TIL en la iteración
  4). Además obligan a fijar la duración al empezar una acción y apenas admiten efectos
  continuos, que forman parte de la 3b. Con STN, las desigualdades de duración con
  límites constantes son intervalos de la red, y la programación lineal solo hace falta
  con efectos continuos o duraciones que dependen del estado. El estado
  `State(facts, values)` y *h^hbd* se reutilizan: la heurística trabaja sobre las acciones
  de inicio y fin.

## Trabajo opcional (al final del proyecto, fuera de las iteraciones)

**Orden acordado** (2026-10-07), para retomarlo así. El proyecto principal ya funciona con
la v0.6, así que nada de esto es crítico. El orden pesa si arregla un problema real, cuánto
mejora el planificador y cuánto cuesta; lo cambiaría que el proyecto principal necesitara
algo concreto. La eficiencia va primero: lo que promete mejoras de rendimiento empieza
midiendo. Cada tarea va en su rama, y las ramas se fusionan juntas cuando todo lo opcional
funcione.

**Hecho en rama, pendiente de fusionar:**

- **Ritmos que fijan las acciones** (rama `pddlplus/settlers`, 2026-10-14), a petición del
  proyecto principal, que no encontraba plan en su dominio de Settlers (una economía con
  un proceso de producción cuyo ritmo fijan las acciones). Son mecanismos generales, no
  específicos de ese dominio:
  - la relajación solo cuenta un incremento variable si es positivo, y cuenta las esperas
    con el ritmo del estado;
  - `_wait-until`, una espera hasta el siguiente cambio que cuesta lo que avanza;
  - el coste de las acciones instantáneas pasa a 1/64, para que las sumas sean exactas;
  - la validación con δ/10 termina cada espera en el estado de la búsqueda.

  Resultados:
  - los tiers 1 a 5 hacen las mismas búsquedas;
  - el motor discretizado es más rápido, pero los planes de car-nodrag son más largos;
  - Settlers p01 sigue sin plan: el dominio tiene una trampa que se arregla en el proyecto
    principal.

  Quedan por decidir dos mejoras que pide Settlers: podar los objetos intercambiables
  (simetrías) y una heurística que cuente el consumo de recursos.

**Pendiente, en este orden:**

1. **Límite de memoria duro en las búsquedas** (de la iteración 5c): el único que ya ha
   causado problemas reales. Hoy el límite del planificador (`max_memory`, `-m`) se mira
   durante la búsqueda cada 32 iteraciones y no cubre el grounding; una medida de storage
   agotó la memoria de WSL por eso (el script de benchmarks ya pone un límite duro por
   proceso). Medido el 2026-10-06:
   - `RLIMIT_AS` no sirve con CUDA (reserva 34 GB de memoria virtual). `RLIMIT_DATA` sí:
     CUDA y Numba arrancan, al pasarlo Python y NumPy reciben `MemoryError`, y en
     búsquedas sin GPU sigue de cerca a la memoria residente (142 MB frente a 170; 405
     frente a 430 con storage p20). Con CUDA el proceso parte de 1,2 GB.
   - El límite no cuesta nada en la búsqueda; lanzarla en un proceso hijo con `fork`
     cuesta de 10 a 14 ms (con `spawn`, de 1 a 3 s).
   - **Opción A** (barata, sin cambiar la API): cada miembro del portfolio fija
     `RLIMIT_DATA` a su parte (con el margen de CUDA en el miembro con GPU) y un
     `MemoryError` es `limit_reached`; la CLI se lo pone a sí misma antes del grounding.
     Como librería, el grounding y las búsquedas de una configuración siguen con el límite
     blando (no se puede limitar el proceso anfitrión).
   - **Opción B** (completa): `solve()` hace el grounding y la búsqueda en un proceso hijo
     con el límite duro. Cambia la API: `PlanningResult.task` no se puede devolver barato
     desde el hijo.
2. **Landmarks (h_LM) con colas alternadas**, al estilo de LAMA: la mayor mejora esperable
   en las tareas clásicas y numéricas (suele ganar cobertura y calidad de los planes frente
   a h_FF sola). Las colas alternadas sirven también para combinar otras heurísticas.
   Empezar midiendo dónde se queda corta h_FF.
3. **Grafo de planificación temporal completo** (opción B de la heurística temporal,
   decidida en la iteración 4). La iteración 4 añade la parte A: instantes más tempranos y
   ventanas de los TIL en la relajación, para descartar estados sin salida. La parte B es
   extraer además un plan relajado que respete los tiempos (como POPF y OPTIC) y usarlo como
   heurística de las tareas temporales en lugar de h_FF, en Python, Numba y PyTorch con
   valores idénticos. Incluye evaluarla frente a h_FF con la parte A en los benchmarks
   temporales: en POPF no es mejor en todos los dominios.
   Empezar midiendo dónde se queda corta h_FF: con la parte A, `lazy-ff-pref` no resuelve
   satellite p02 (tier 4, con TIL) en 60 s, aunque `bfws-ff-pref` sí.
4. **Huecos de cobertura.** Hoy dan un error claro y ningún tier los usa:
   - **preferencias en tareas temporales** (de la iteración 5c): la fase final como pasos
     instantáneos al final del plan, y los monitores blandos con las marcas de instante de
     la 5b. El tier 5 no tiene ningún dominio que las use;
   - **límites del motor discretizado** (iteración 6): literales iniciales temporizados,
     restricciones de trayectoria y preferencias con procesos y eventos; efectos
     condicionales que dependen de condiciones `over all` y efectos continuos condicionales
     en las acciones durativas.
5. **Más diversidad en el portfolio**: *enforced hill-climbing* (FF) y paseos aleatorios
   (Arvand). El portfolio ya es diverso (BFWS, exploración por tipos, goalcount aleatorio):
   solo si aparece un dominio que lo pida.
6. **Más allá de la programación lineal**. Ha perdido valor con la 6b: las invariantes
   disyuntivas con efectos continuos ya pasan al motor discretizado. El motor temporal
   resuelve los efectos continuos lineales con un programa lineal por estado. Dos líneas:
   - ✅ **expresiones no lineales por discretización del tiempo** (hecho en la 6b, a
     petición del usuario): las tareas temporales que el motor de la 3b no admite por sus
     efectos continuos pasan al motor discretizado de PDDL+ (paso δ, integración exacta,
     validación con un δ más fino). generator-nonlinear: 8 de 8; generator-toricelli: 5 de 9,
     y 9 de 9 con la heurística de la 6c (ver `benchmarks/continuous-effects`);
   - **disyunciones sobre variables continuas** (condiciones de efectos condicionales,
     invariantes disyuntivas) con **programación lineal entera** (HiGHS, ya disponible con
     `scipy.optimize.milp`). El coste en eficiencia es importante (la programación entera
     puede ser exponencial), así que se activaría solo en los dominios que lo necesiten,
     como la programación lineal ahora.
7. **Estudio de la integración de HTN** (planificación jerárquica): viabilidad de aceptar
   dominios HTN (HDDL, el lenguaje del track HTN de las IPC) y de combinar la
   descomposición jerárquica con nuestro motor de búsqueda. Sería una mejora bienvenida
   para organizar mejor la información en el proyecto principal, pero no es crítica.

---

## Material de referencia (propuesta)

- Especificaciones:
  - McDermott et al., 1998: PDDL 1.2.
  - Fox & Long, 2003: PDDL2.1 (JAIR 20).
  - Edelkamp & Hoffmann, 2004: PDDL2.2.
  - Fox & Long, 2006: PDDL+ (JAIR 27).
  - Gerevini & Long, 2005: PDDL3 (restricciones de trayectoria y preferencias).
  - Höller et al., 2020: HDDL, el lenguaje de la planificación HTN en las IPC.
- Grounding y normalización: Helmert, 2009, *Concise finite-domain representations for
  PDDL planning tasks* (el traductor de Fast Downward).
- Heurísticas:
  - Bonet & Geffner, 2001: h_add y h_max;
  - Hoffmann & Nebel, 2001: FF;
  - Hoffmann, 2003: Metric-FF;
  - Scala et al., 2016–2020: ENHSP y las heurísticas numéricas.
- Búsqueda paralela y portfolios:
  - Helmert, Röger & Karpas, 2011: Fast Downward Stone Soup (portfolio secuencial);
  - Nakhost, Müller et al., 2011: Arvand y ArvandHerd (paseos aleatorios, portfolio paralelo);
  - Kishimoto, Fukunaga & Botea, 2009: HDA\* (búsqueda distribuida por hash);
  - Kuroiwa & Fukunaga, 2019–2020: anomalías de la GBFS paralela.
- Búsqueda:
  - Xie, Müller, Holte & Imai, 2014: exploración por tipos;
  - Lipovetzky & Geffner, 2012 y 2017: novedad, IW y BFWS.
- PDDL 3:
  - Keyder & Geffner, 2009: compilación de objetivos blandos a costes;
  - Baier & McIlraith, 2006: compilación de restricciones de trayectoria a autómatas.
- Planificación temporal: Coles et al., 2010 (POPF) y Eyerich et al., 2009 (TFD).
- PDDL+:
  - Della Penna et al., 2009: UPMurphi;
  - Piotrowski et al., 2016: DiNo;
  - Scala et al., 2016: ENHSP.
- Libro de texto: Ghallab, Nau & Traverso, *Automated Planning* (2004) y
  *Automated Planning and Acting* (2016).
- Herramientas:
  - [VAL](https://github.com/KCL-Planning/VAL) para validar los planes;
  - dominios de las IPC ([potassco/pddl-instances](https://github.com/potassco/pddl-instances))
    para los benchmarks.
