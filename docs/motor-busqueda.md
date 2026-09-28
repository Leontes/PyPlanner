# Motor de búsqueda (iteraciones 1 y 2)

Cómo está construido el motor y cómo se usa. Las decisiones de fondo (portfolio, lotes,
GPU, técnicas descartadas) están en el [roadmap](ROADMAP.md#motor-de-búsqueda-diseño-y-repertorio).

## Flujo

```
Domain + Problem (AST) ──ground()──► GroundTask ──search() / run_portfolio()──► SearchResult ──► Plan
```

| Paso | Módulo | Qué hace |
|------|--------|----------|
| Grounding | `pyplanner.grounding` | Instancia las acciones alcanzables en la relajación y elimina los hechos estáticos |
| Heurísticas | `pyplanner.heuristics` | Evalúan lotes de estados; implementación de referencia y en PyTorch |
| Búsqueda | `pyplanner.search` | GBFS / A* / WA*, inmediata o perezosa, con acciones útiles y lotes |
| Portfolio | `pyplanner.portfolio` | Varias configuraciones en procesos paralelos; gana la primera |
| Plan | `pyplanner.plan` | Salida IPC o JSON y validación interna (simulación) |
| Fachada | `pyplanner.planner.solve` | Encadena todo lo anterior |

## Normalización ADL (iteración 2)

El grounder (`grounding/grounder.py`) normaliza cada instancia de acción con
`grounding/normalize.py`, de modo que la búsqueda solo ve estructuras conjuntivas:

- **Precondiciones y objetivos → DNF de literales.** Los cuantificadores se expanden sobre
  los objetos de su tipo (`exists` → disyunción, `forall` → conjunción), las negaciones se
  empujan hasta los átomos e `imply` se trata como `(or (not a) b)`. `=` y los hechos
  estáticos se evalúan en el momento; las cláusulas contradictorias y las subsumidas se
  eliminan. Si la DNF supera 10 000 cláusulas se aborta con un error claro (esas fórmulas
  necesitarán axiomas, iteración 4).
- **Precondición disyuntiva:** una `GroundAction` por cláusula, con el mismo nombre y
  argumentos, así que en el plan se ven igual.
- **Efectos → efectos incondicionales + efectos condicionales** (`forall` se expande;
  cada cláusula de la condición de un `when` da un efecto condicional). Si la precondición
  ya garantiza la condición, el efecto pasa a ser incondicional; si la contradice, se
  descarta.
- **Objetivo:** disyunción de cláusulas. Si ninguna es alcanzable, el objetivo queda
  vacío y la tarea se reconoce como irresoluble sin buscar.
- **Alcanzabilidad relajada:** los literales negativos se suponen alcanzables, y un
  efecto condicional cuenta si alguna cláusula de su condición es alcanzable.

## Tarea instanciada y estados

- Un **estado** es un `int` usado como bitset sobre `GroundTask.facts`. Una acción es
  aplicable si `state & pre == pre` y `state & neg == 0` (precondiciones negativas).
- **Efectos condicionales:** las condiciones se evalúan sobre el estado **anterior**;
  después se borra y se añade (un hecho borrado y añadido a la vez queda cierto). Los
  efectos condicionales de cada acción se indexan por su primer hecho de condición, así
  que al aplicarla solo se revisan los que pueden dispararse. En rubiks-cube cada giro
  tiene cientos de efectos condicionales y solo unos pocos se disparan.
- Los hechos **estáticos** no están en el estado.
- **Generación de sucesores** (`search/successors.py`), con un **orden canónico**:
  1. primero las acciones sin precondiciones positivas;
  2. después, las de cada hecho cierto en orden creciente cuya primera precondición es
     ese hecho;
  3. dentro de cada grupo, por índice.

  No hace falta ordenar, y hay dos implementaciones con el mismo orden: la de referencia
  (`GroundTask.applicable`) y una compilada con Numba, que comprueba las precondiciones
  con palabras de 64 bits. La búsqueda usa la compilada salvo con `--device python`; como
  el orden es el mismo, expande los mismos nodos con las dos.
- Para generar sucesores, las acciones se agrupan por su primera precondición: solo se
  miran las acciones cuya primera precondición es cierta en el estado.
- Hechos y acciones tienen un orden determinista, así que dos ejecuciones iguales dan el
  mismo plan.

## Heurísticas

| Nombre | Admisible | Acciones útiles | Numba | PyTorch |
|--------|-----------|-----------------|-------|---------|
| `blind` | sí | no | no | no |
| `goalcount` | no | no | no | no |
| `hmax` | sí | no | sí | sí |
| `hadd` | no | no | sí | sí |
| `ff` | no | sí | sí | sí (costes y plan relajado en tensores) |

**Dispositivos** (`SearchConfig.device`, `--device`):

| Dispositivo | Qué es | Cuándo usarlo |
|-------------|--------|---------------|
| `auto` (por defecto) | `numba` si está instalado; si no, `python` | siempre, salvo que se quiera forzar otro |
| `python` | implementación de referencia en Python puro | tests y máquinas sin Numba |
| `numba` | la referencia compilada con Numba (`pip install -e ".[fast]"`); 12–19 veces más rápida | es la opción rápida para búsquedas normales |
| `cpu`, `cuda` | PyTorch por lotes (`pip install -e ".[gpu]"`) | solo con lotes grandes; en GPU, 4 veces más rápido que Python con lotes de 256 |

Los tres dan **exactamente los mismos valores y acciones útiles** que la referencia; los
tests lo comprueban en los tiers 1 y 2 y en las tareas hechas a mano.

- **Interfaz:** `evaluate_batch(states) -> list[Evaluation]`. Cada `Evaluation` lleva el
  valor (`inf` si es un callejón sin salida) y las acciones útiles.
- **Tarea relajada** (`RelaxedTask`, compartida por las dos implementaciones):
  - **Hechos complementarios:** cada hecho que aparece negado tiene un hecho "no-p",
    cierto si p es falso y que hacen cierto las acciones que borran p. Así las negaciones
    pasan a ser positivas **solo dentro de la relajación**, como en FF.
  - **Acciones relajadas:** una por los efectos incondicionales de cada operador y otra
    por cada efecto condicional (con la condición añadida a la precondición), como en
    Fast Downward.
  - **Objetivo:** el valor es el mínimo entre cláusulas.
- **Referencia en Python:** Dijkstra generalizado sobre la relajación sin borrados. Se
  detiene al extraer todos los hechos del objetivo, y anota el mejor soportador durante el
  propio recorrido, así que h_FF no tiene que buscarlo después.
- **Numba** (`heuristics/compiled.py`): el mismo Dijkstra y la misma extracción,
  compilados, sobre la tarea relajada en arrays CSR. El lote entero se evalúa en una
  llamada y los estados se decodifican dentro del código compilado. La primera ejecución
  tras instalar compila los núcleos (unos segundos) y los guarda en la caché de disco.
- **PyTorch:** punto fijo sobre todo el lote con `index_add_` y `scatter_reduce`, usando
  listas de aristas acción-hecho y no matrices densas.
- **h_FF:** el mejor soportador es la acción relajada de menor índice con coste mínimo, y
  se usa la cláusula del objetivo de menor valor (la primera si hay empate). Son reglas
  deterministas y compartidas, así que las dos implementaciones dan exactamente el mismo
  valor y las mismas acciones útiles; los tests lo comprueban. El valor cuenta cada
  **operador** una sola vez, aunque el plan relajado use varios de sus efectos.
- **h_FF en tensores:** la extracción del plan relajado también se hace para todo el lote
  a la vez, por capas hacia atrás desde los objetivos (ver `heuristics/tensor.py`).

## Configuración de búsqueda

`SearchConfig` define por completo una búsqueda y se valida al crearla. Campos:

- `algorithm`: `gbfs`, `astar` o `wastar` (con `weight`);
- `heuristic`;
- `preferred_operators`: dos colas alternas, la normal y la de acciones útiles;
- `lazy`: evaluación perezosa; no se puede usar con A*;
- `tie_breaking`: `fifo`, `lifo` o `random`, con `seed`;
- `exploration`: `"type-based"` añade una cola de **exploración por tipos** (Xie et al.,
  2014) que se alterna con las demás: saca un tipo (h, g) al azar y un nodo al azar de
  ese tipo. Evita que la búsqueda voraz se atasque en mesetas y zonas con muchos
  callejones sin salida;
- `batch_size`;
- `device`: `auto` (por defecto), `python`, `numba`, `cpu` o `cuda` (ver arriba);
- `max_time` y `max_expansions`.

Garantías:

- **Óptimo:** A* con `blind` o `hmax` devuelve planes óptimos, también con lotes. En la
  búsqueda inmediata, un objetivo solo se acepta si es el primero que sale de la cola en
  su lote.
- **Completo:** todas las configuraciones lo son. Los callejones sin salida que se
  descartan son estados desde los que el objetivo es inalcanzable incluso en la
  relajación.
- **Cancelable:** `search(..., stop=evento)` consulta el evento periódicamente.

Configuraciones predefinidas (`pyplanner configs`):

| Nombre | Configuración | Notas |
|--------|---------------|-------|
| `gbfs-ff-pref-tb` | GBFS + h_FF + acciones útiles + exploración por tipos | la de 1 núcleo: la más robusta |
| `gbfs-ff-pref` | GBFS + h_FF + acciones útiles | |
| `lazy-ff-pref` | GBFS perezosa + h_FF + acciones útiles | evalúa muchos menos estados |
| `lazy-ff-pref-tb` | la anterior + exploración por tipos (semilla 1) | |
| `gbfs-ff` | GBFS + h_FF | |
| `gbfs-add` | GBFS + h_add | |
| `wastar-add-3` | WA* (w = 3) + h_add | |
| `gbfs-goalcount-rnd` | GBFS + objetivos pendientes, desempate aleatorio | muy barata; aporta diversidad |
| `astar-hmax` | A* + h_max | óptima |
| `astar-blind` | A* ciega | óptima; coste uniforme |

## Portfolio

**Qué búsquedas se lanzan.** Depende del número de núcleos (`-j N`). Hay un portfolio
predefinido para 1, 2, 4 y 5 núcleos (`PORTFOLIOS` en `search/config.py`), y cada uno
contiene al anterior:

| Núcleos | Búsquedas | Qué se añade |
|---------|-----------|--------------|
| 1 | `gbfs-ff-pref-tb` | la configuración individual más robusta; se ejecuta en el propio proceso |
| 2 | + `lazy-ff-pref` | la más rápida en tareas grandes y fáciles |
| 4 | + `gbfs-goalcount-rnd`, `lazy-ff-pref-tb` | la más barata y aleatoria, y la perezosa con exploración |
| 5 | + `wastar-add-3` | WA*, que tiende a planes más cortos |

- Con otro número de núcleos se usa el portfolio más grande que quepa: con 3, el de 2;
  con más de 5, el de 5. Nunca hay más búsquedas que núcleos.
- **Por defecto se usan 5 núcleos**, para la máxima robustez, pero limitados a los que
  tenga la máquina (`os.sched_getaffinity`). En un sistema mononúcleo se usa el
  portfolio de 1 sin necesidad de indicarlo.
- Con `-c` se eligen las configuraciones a mano. Si se piden más configuraciones que
  núcleos, es un error.

**Cómo se ejecutan.**

- Un proceso por configuración; gana el primer plan y el resto se cancelan.
- Se usa `fork` cuando es seguro: sin CUDA, sin PyTorch cargado en el padre y con un
  solo hilo. En cualquier otro caso se usa `spawn`.
- Si un miembro falla, informa con estado `ERROR` y un mensaje, pero no tumba el
  portfolio.
- **Precompilación de Numba:** antes de lanzar los hijos, el padre compila (o carga de la
  caché) los núcleos de Numba si algún miembro los va a usar. Sin esto, un hijo cancelado
  a mitad de compilar dejaba la caché incompleta, y cada ejecución pagaba la compilación
  y la espera del cierre. Con `fork`, los hijos heredan los núcleos ya compilados.

## Uso desde la terminal

```bash
pyplanner configs                                   # configuraciones y portfolios
pyplanner solve dominio.pddl problema.pddl          # portfolio de 5 núcleos (o menos)
pyplanner solve -j 1 dominio.pddl problema.pddl     # un solo núcleo
pyplanner solve -j 2 -t 60 dominio.pddl p.pddl      # 2 núcleos, 60 s como máximo
pyplanner solve -c astar-hmax dominio.pddl p.pddl   # plan óptimo
pyplanner solve -c gbfs-add --device cuda --batch-size 64 dominio.pddl p.pddl
pyplanner solve -f json -o plan.json dominio.pddl p.pddl
```

- El plan sale por la salida estándar y las estadísticas por la de error; `-q` las
  suprime.
- Códigos de salida:
  - 0: plan encontrado;
  - 2: entrada no válida;
  - 3: se ha demostrado que no hay plan;
  - 4: no hay plan dentro de los límites.

## Uso como librería

```python
from pyplanner.frontend import read_domain, read_problem
from pyplanner.planner import solve
from pyplanner.search import portfolio_for_cores

domain = read_domain("domain.pddl")
problem = read_problem("p01.pddl", domain)
result = solve(domain, problem, portfolio_for_cores(2), time_limit=60)
if result.plan is not None:
    print(result.plan.to_ipc())  # o result.plan.to_dict() para el JSON
```

## Rendimiento medido

Medido en la máquina de desarrollo (GTX 970, 12 núcleos) con gripper de las IPC 1998,
`instance-20`: 42 bolas, 172 hechos y 340 acciones instanciadas.

**Coste de evaluar un estado** (1024 estados de paseos aleatorios):

| Heurística | Python | PyTorch CPU (lote 1024) | PyTorch GPU (lote 64) | PyTorch GPU (lote 1024) |
|------------|--------|-------------------------|-----------------------|-------------------------|
| h_add | 164 µs | 15 µs | 21 µs | **5.7 µs** |
| h_FF | 269 µs | 28 µs | 51 µs | **9.7 µs** |

**Búsqueda completa** (tiempo total del proceso):

| Configuración | Plan | Tiempo |
|---------------|------|--------|
| `gbfs-ff-pref` | 165 | 8.0 s |
| `gbfs-ff-pref`, CPU, lote 64 | 125 | 10.9 s |
| `gbfs-ff-pref`, GPU, lote 64 | 125 | 7.6 s |
| `lazy-ff-pref` | 165 | 0.47 s |
| `gbfs-add` | 167 | 1.1 s |
| `gbfs-add`, CPU, lote 64 | 163 | 5.0 s |
| `gbfs-add`, GPU, lote 64 | 163 | 4.1 s |
| portfolio de 5 núcleos | 129 (ganó `gbfs-goalcount-rnd`) | 0.52 s |

Conclusiones:

- **Evaluar por lotes con tensores sale muy a cuenta:** con lotes grandes, h_add es 11
  veces más rápida en CPU y 29 en GPU, y h_FF (con la extracción del plan relajado también
  en tensores) 10 y 28 veces.
- **Los lotes no aceleran por sí solos la búsqueda completa en tareas de este tamaño:**
  - importar PyTorch e inicializar CUDA cuesta unos 2 s;
  - con lotes, la GBFS expande *k* nodos antes de volver a mirar la heurística, así que
    se guía peor y expande más.

  Compensan cuando la evaluación domina el tiempo total, es decir, con tareas grandes o
  heurísticas caras. Por eso los lotes son una opción y no el valor por defecto.
- **El portfolio aporta robustez de verdad:** una configuración muy barata ganó a todas
  las demás, y además con un plan mejor. Es la diversidad de la que hablábamos.

### Iteración 2: slitherlink

slitherlink p02 (tablero de 4×6, 306 hechos, 1920 acciones con 2400 precondiciones
negativas) es el caso más duro de los tiers 1 y 2. Lecciones:

- **La heurística domina el tiempo:** más del 90%, y más de la mitad de los estados
  evaluados son callejones sin salida. Dos cambios en el Dijkstra de referencia duplicaron
  el ritmo de expansión:
  - precalcular el número de precondiciones de cada acción;
  - anotar el mejor soportador durante el recorrido, en lugar de buscarlo después.
- **La búsqueda voraz pura se atasca.** `gbfs-ff-pref` necesitaba unas 49 000 expansiones
  (unos 80 s al ritmo medido). Con lotes (*k-best-first search*) el número de expansiones
  variaba entre 1 400 y 82 000 según el tamaño de lote y la instancia: suerte con el
  orden, no una mejora fiable.
- **La exploración por tipos sí es robusta:** con 4 semillas resuelve p02 siempre, en
  17–24 s y con 9 000–14 000 expansiones. Por eso es la base de los portfolios desde la
  iteración 2.

### Iteración 2: heurísticas compiladas con Numba

Medido sobre 3000 estados reales de una búsqueda en slitherlink p02 (45% de callejones sin
salida; los hermanos de cada expansión dan lotes de unos 2,8 estados):

| Opción | µs/estado (h_FF) | Frente a Python |
|--------|------------------|-----------------|
| Referencia en Python | ~600 | 1× |
| Comprobación de alcanzabilidad previa + h_FF solo en estados vivos | 613 | 1,0× |
| PyTorch CPU / GPU, lotes de 3 | 892 / 1524 | 0,7× / 0,4× |
| PyTorch GPU, lotes de 256 | 142 | 4,3× |
| **Numba** | **~42** | **~19×** |

- **La comprobación previa no gana nada.** Los callejones sin salida son *más baratos* que
  los estados vivos (unos 490 µs frente a 760 µs), así que lo que ahorra en unos lo
  pierde en otros.
- **PyTorch no sirve con los lotes pequeños de la búsqueda:** domina el coste fijo de
  cada llamada.
- **Numba es 12–19 veces más rápido** según el dominio (gripper, rubiks-cube,
  slitherlink), con resultados idénticos a la referencia.

Búsqueda completa en slitherlink p02 (CLI, tiempo total del proceso):

| | Python | Numba (`auto`) |
|--|--------|----------------|
| `-j 1` | 22,3 s | **3,1 s** |
| `-j 5` | 23,6 s | **3,6 s** |

El test de tiers completo (tiers 1 y 2, 4 portfolios, validación con VAL) pasó de 97 s a
18 s. La búsqueda expande exactamente los mismos nodos: la ganancia es pura velocidad.
Ahora la heurística ya no domina el tiempo total; el siguiente cuello de botella es la
propia búsqueda en Python (generar sucesores y gestionar las colas).

### Iteración 2: NumPy y generación de sucesores

Se estudió sustituir las listas y los enteros de Python por arrays de NumPy. Medido en el
código del planificador:

| Operación | Python | NumPy |
|-----------|--------|-------|
| Acceso elemento a elemento en un bucle (patrón del Dijkstra de referencia) | 28 ns | 104 ns (3,6× más lento) |
| Aplicar una acción y calcular el hash del estado | 240 ns (entero bitset) | 1098 ns (array; 5× más lento) |
| Memoria por estado | 68 B | 418 B |
| Acciones aplicables (1920 acciones, vectorizado) | 70 µs | 80 µs |

**Conclusión: ahora mismo no nos beneficia usarlo de forma general.** NumPy gana cuando
un bucle entero corre en C, y eso ya lo aprovechamos dentro de los núcleos de Numba y de
PyTorch. En el código que va elemento a elemento es más lento. Además, el estado como
entero de Python es mejor que cualquier array: las operaciones de bits y el hash son una
sola operación en C sobre un solo objeto.

No es una decisión cerrada. NumPy es ya una dependencia y está disponible. Si en alguna
iteración aparece una parte concreta que se beneficie de operaciones vectoriales, se usará
ahí, de forma localizada; por ejemplo, el cálculo numérico de la iteración 3.

El perfil sí mostró que, con la heurística ya compilada, la **generación de sucesores**
se llevaba el 46% del tiempo: ordenar los candidatos de cada expansión era el 18%.
Resultados en slitherlink p02 (tiempo de búsqueda, sin arranque del proceso):

| Paso | Tiempo | µs/expansión |
|------|--------|--------------|
| Punto de partida (heurística con Numba) | 2,31 s | 210 |
| Orden canónico sin ordenar | 1,99 s | 177 |
| Generador de sucesores compilado con Numba | 1,53 s | 137 |
| Extracción de h_FF recorriendo solo el plan relajado y acciones útiles compactas | **1,44 s** | **129** |

Con la CLI (tiempo total del proceso), slitherlink p02 queda en **2,4 s con 1 núcleo y
2,5 s con 5**, frente a los 22–24 s de la versión con la heurística en Python. Ahora la
generación de sucesores es el 5% del tiempo y la heurística compilada unos 35 µs por
estado (unas tres cuartas partes del total). Bajar más exigiría cambios de algoritmo
(heurísticas incrementales, menos evaluaciones), no de implementación.
