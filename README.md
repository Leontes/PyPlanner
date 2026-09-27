# PyPlanner

Planificador PDDL en Python.

## Estructura

```
src/pyplanner/       Código del planificador
tests/               Tests (pytest)
dominios/            Benchmarks PDDL por tiers (ver dominios/README.md)
docs/                Documentación y notas
```

## Puesta en marcha

Requiere Python 3.12 o superior.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Validador de planes (VAL)

Los tests validan los planes con [VAL](https://github.com/KCL-Planning/VAL). Se compila en
local (necesita `git`, `cmake` y `g++`):

```bash
scripts/build_val.sh          # deja el ejecutable en tools/bin/Validate
```

## Uso

```bash
# Buscar un plan (formato IPC por la salida estándar; estadísticas por la de error)
pyplanner solve dominio.pddl problema.pddl
pyplanner solve -p -t 60 dominio.pddl problema.pddl       # portfolio paralelo, 60 s
pyplanner solve -c astar-hmax -f json -o plan.json dominio.pddl problema.pddl
pyplanner configs                                          # configuraciones disponibles

# Fase 1: PDDL → JSON (el formato de entrada del planificador, ver docs/formato-json.md)
pyplanner parse dominio.pddl -o dominio.json

# Fases 1 y 2: comprueba que un dominio y un problema son válidos (.pddl o .json)
pyplanner check dominio.pddl problema.pddl
```

Más detalles del motor en [docs/motor-busqueda.md](docs/motor-busqueda.md).

### Evaluación en GPU (opcional)

Las heurísticas `hmax`, `hadd` y `ff` pueden evaluar lotes de estados con PyTorch
(`--device cpu|cuda --batch-size N`). PyTorch no es obligatorio:

```bash
pip install -e ".[gpu]"
# GPUs antiguas (Maxwell, p. ej. GTX 970): solo las builds para CUDA 12.6 las soportan
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

## Desarrollo

```bash
pytest               # tests
ruff check .         # linter
ruff format .        # formateo
mypy                 # comprobación de tipos
```
