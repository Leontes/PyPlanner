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
# Fase 1: PDDL → JSON (el formato de entrada del planificador, ver docs/formato-json.md)
pyplanner parse dominio.pddl -o dominio.json

# Fases 1 y 2: comprueba que un dominio y un problema son válidos (.pddl o .json)
pyplanner check dominio.pddl problema.pddl
```

La búsqueda (`pyplanner solve`) llega en la iteración 1; ver [docs/ROADMAP.md](docs/ROADMAP.md).

## Desarrollo

```bash
pytest               # tests
ruff check .         # linter
ruff format .        # formateo
mypy                 # comprobación de tipos
```
