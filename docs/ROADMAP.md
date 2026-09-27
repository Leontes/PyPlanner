# Roadmap de PyPlanner

Objetivo: un planificador PDDL escrito en Python que llegue hasta PDDL+. El motor es una
**búsqueda heurística hacia delante**. El proyecto avanza en 5 iteraciones y cada una amplía
el lenguaje que se acepta.

Referencia principal del lenguaje: [planning.wiki](https://planning.wiki/ref/pddl/requirements).
PDDL 3.0 queda fuera del alcance (`:constraints`, `:preferences`).

## Reglas comunes a todas las iteraciones

En cada iteración hay que:

1. **Parser**: ampliar las dos fases (PDDL → JSON y JSON → AST), el esquema JSON y el
   AST para aceptar los nuevos requisitos. Si un fichero declara un requisito que todavía
   no está soportado, se rechaza con un error claro.
   **Modo estricto:** toda construcción tiene que tener su requisito declarado en
   `:requirements` (en el dominio o en el problema). Si un fichero usa, por ejemplo,
   `:types` sin declarar `:typing`, se rechaza aunque `:typing` esté soportado. Si no se
   declara ningún requisito, se supone `:strips`. La comprobación se hace en las dos fases,
   porque el JSON puede venir de otra fuente. La tabla construcción → requisito ya existe
   en la fase 1 (`_FUTURE_*` en `pddl/parser.py`) y hay que llevarla a un sitio común.
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
  pddl/        fase 1: lexer (s-expresiones) → parser → documento JSON
  schema/      esquema JSON versionado (JSON Schema): el contrato entre las dos fases
  loader/      fase 2: JSON → validación semántica → AST tipado (dataclasses inmutables)
  grounding/   instanciación de acciones y análisis de alcanzabilidad
  model/       tarea instanciada: estado, acciones, objetivos
  search/      algoritmos de búsqueda (GBFS, A*, WA*) sobre una interfaz común
  heuristics/  h_add, h_max, h_FF y sus extensiones numéricas y temporales;
               interfaz por lotes; backends de referencia y PyTorch (tensor/)
  portfolio/   ejecución paralela de configuraciones (procesos, cancelación)
  plan/        representación del plan; salida en formato IPC (la que lee VAL) o en JSON
  cli.py
```

Decisiones propuestas:
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
| Enforced hill-climbing (FF) | búsqueda local rápida | opcional |
| Paseos aleatorios (Arvand) | diversidad en el portfolio | opcional |
| Colas alternadas / multiheurística | combinar heurísticas en una sola búsqueda | opcional |
| Landmarks (h_LM) | heurística informada más potente | opcional |
| HDA\*, frontera distribuida | escalar una sola búsqueda | descartado (Python) |
| Evaluación de la heurística por lotes (PyTorch, CPU o GPU) | muchos estados evaluados a la vez | iteración 1 |
| Búsqueda completa en GPU | expansión masiva en paralelo | descartado (ver abajo) |

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
| 5 | PDDL+ | Iteración 5 |

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
    1, 2, 4 y 5 núcleos (`-j N`; por defecto 5, o los que tenga la máquina).
- **Tests:** ✅ tier 1 (blocksworld y gripper) resuelto y validado con VAL, con el
  portfolio de cada número de núcleos. También hay problemas sin solución hechos
  a mano, tests del portfolio y tests de equivalencia entre la referencia y PyTorch.

## Iteración 2: PDDL 1.2 (subconjunto ADL)

**Requisitos:** `:typing`, `:negative-preconditions`, `:disjunctive-preconditions`,
`:equality`, `:existential-preconditions`, `:universal-preconditions`,
`:quantified-preconditions`, `:conditional-effects` y `:adl`.

- **Parser:**
  - jerarquía de tipos (`either` incluido);
  - fórmulas arbitrarias con `not`, `or`, `imply`, `exists`, `forall` y `=`;
  - efectos `forall` y `when`;
  - expansión de los metarrequisitos (`:adl` y `:quantified-preconditions`).
- **Motor:**
  - normalización: forma negada normal, eliminación de cuantificadores sobre universos
    finitos y división de las disyunciones en varias acciones (al estilo del traductor de
    Fast Downward);
  - precondiciones negativas;
  - efectos condicionales en la aplicación de acciones y en la heurística relajada.
- **Tests:** tier 2 (rubiks-cube y slitherlink) y casos límite de tipos y
  cuantificadores.

**Fuera de alcance (propuesta):** los requisitos obsoletos o casi sin uso de PDDL 1.2:
`:action-expansions`, `:foreach-expansions`, `:dag-expansions`, `:domain-axioms`,
`:subgoals-through-axioms`, `:safety-constraints`, `:expression-evaluation`,
`:open-world`, `:true-negation`, `:ucpop` y `:fluents` con la semántica de 1.2.

## Iteración 3: PDDL 2.1

Es la iteración más grande, así que se divide en dos.

### 3a: fluentes numéricos
**Requisitos:** `:numeric-fluents` (y `:fluents`).
- **Parser:**
  - `:functions`;
  - expresiones aritméticas;
  - comparaciones;
  - efectos `assign`, `increase`, `decrease`, `scale-up` y `scale-down`;
  - `:metric` en el problema.
- **Motor:**
  - estado lógico más vector numérico;
  - heurística numérica relajada al estilo Metric-FF o ENHSP (h_add/h_FF sobre intervalos);
  - A* sobre la métrica, cuando la haya.

### 3b: acciones durativas
**Requisitos:** `:durative-actions`, `:duration-inequalities` y `:continuous-effects`.
- **Parser:**
  - `:durative-action` y `:duration`;
  - condiciones `at start`, `at end` y `over all`;
  - efectos temporales;
  - `#t` y `?duration`.
- **Motor:**
  - acciones divididas en *snap actions* (inicio y fin);
  - búsqueda hacia delante sobre estados temporales (acciones en ejecución y comprobación
    de invariantes `over all`);
  - planificación temporal con STN al estilo POPF, o por *decision epochs* al estilo TFD
    (hay que elegir una de las dos);
  - efectos continuos lineales;
  - salida de planes temporales (`t: (acción) [d]`).

## Iteración 4: PDDL 2.2

**Requisitos:** `:derived-predicates` y `:timed-initial-literals`.
- **Parser:** `:derived` en el dominio y literales `(at t hecho)` en `:init`.
- **Motor:**
  - evaluación de los axiomas por estratos, con punto fijo en cada estado;
  - soporte de los axiomas en la heurística;
  - los TIL se tratan como eventos obligatorios en el eje temporal del motor de la
    iteración 3b.

## Iteración 5: PDDL+

**Requisitos:** `:time`.
- **Parser:** `:process`, `:event` y `#t` en los efectos de los procesos.
- **Motor:**
  - semántica de PDDL+ por discretización del tiempo, al estilo UPMurphi, DiNo o ENHSP.
    Tiene tres partes:
    - paso δ configurable;
    - disparo en cascada de los eventos;
    - avance de los procesos activos;
  - validación del plan con una δ más fina (*plan validation*);
  - heurística que tiene en cuenta los procesos (extensión de la heurística numérica);
  - salida con `@end` y horizonte temporal.

---

## Pendiente de decidir

- `:action-costs` (PDDL 3.1) aparece en casi todo el track clásico de las IPC 2023. No
  forma parte de las versiones que cubrimos, pero es una forma restringida de fluentes
  numéricos (`total-cost`). Se podría soportar gratis en la iteración 3a.

---

## Material de referencia (propuesta)

- Especificaciones:
  - McDermott et al., 1998: PDDL 1.2.
  - Fox & Long, 2003: PDDL2.1 (JAIR 20).
  - Edelkamp & Hoffmann, 2004: PDDL2.2.
  - Fox & Long, 2006: PDDL+ (JAIR 27).
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
