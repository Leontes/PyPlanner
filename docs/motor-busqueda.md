# Motor de búsqueda (iteración 1)

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

## Tarea instanciada y estados

- Un **estado** es un `int` usado como bitset sobre `GroundTask.facts`. Comprobar si una
  acción es aplicable es `state & pre == pre`, y aplicarla es `(state & ~del) | add`.
- Los hechos **estáticos** no están en el estado. Los objetivos inalcanzables se
  conservan, para que las heurísticas den infinito y la tarea se reconozca como
  irresoluble sin buscar.
- Para generar sucesores, las acciones se agrupan por su primera precondición: solo se
  miran las acciones cuya primera precondición es cierta en el estado.
- Hechos y acciones tienen un orden determinista, así que dos ejecuciones iguales dan el
  mismo plan.

## Heurísticas

| Nombre | Admisible | Acciones útiles | PyTorch |
|--------|-----------|-----------------|---------|
| `blind` | sí | no | no |
| `goalcount` | no | no | no |
| `hmax` | sí | no | sí |
| `hadd` | no | no | sí |
| `ff` | no | sí | sí (costes y plan relajado en tensores) |

- **Interfaz:** `evaluate_batch(states) -> list[Evaluation]`. Cada `Evaluation` lleva el
  valor (`inf` si es un callejón sin salida) y las acciones útiles.
- **Referencia en Python:** Dijkstra generalizado sobre la relajación sin borrados. Se
  detiene al extraer todos los objetivos.
- **PyTorch:** punto fijo sobre todo el lote con `index_add_` y `scatter_reduce`, usando
  listas de aristas acción-hecho y no matrices densas.
- **h_FF:** el mejor soportador es la acción de menor índice con coste mínimo. Es una
  regla determinista y compartida, así que las dos implementaciones dan exactamente el
  mismo valor y las mismas acciones útiles. Los tests lo comprueban.
- **h_FF en tensores:** la extracción del plan relajado también se hace para todo el lote
  a la vez, por capas hacia atrás desde los objetivos (ver `heuristics/tensor.py`).

## Configuración de búsqueda

`SearchConfig` define por completo una búsqueda y se valida al crearla. Campos:

- `algorithm`: `gbfs`, `astar` o `wastar` (con `weight`);
- `heuristic`;
- `preferred_operators`: dos colas alternas, la normal y la de acciones útiles;
- `lazy`: evaluación perezosa; no se puede usar con A*;
- `tie_breaking`: `fifo`, `lifo` o `random`, con `seed`;
- `batch_size`;
- `device`: `python`, `cpu` o `cuda`;
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
| `gbfs-ff-pref` | GBFS + h_FF + acciones útiles | |
| `lazy-ff-pref` | GBFS perezosa + h_FF + acciones útiles | evalúa muchos menos estados; la de 1 núcleo |
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
| 1 | `lazy-ff-pref` | la configuración individual más robusta; se ejecuta en el propio proceso |
| 2 | + `gbfs-goalcount-rnd` | la más distinta: muy barata y con desempate aleatorio |
| 4 | + `gbfs-ff-pref`, `gbfs-add` | h_FF con evaluación inmediata y h_add |
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
