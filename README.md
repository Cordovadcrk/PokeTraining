[README.md](https://github.com/user-attachments/files/30205985/README.md)
# PokeTraining

[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/Cordovadcrk/PokeTraining/actions/workflows/ci.yml/badge.svg)](https://github.com/Cordovadcrk/PokeTraining/actions/workflows/ci.yml)

PokeTraining es un proyecto educativo de aprendizaje por refuerzo profundo que
entrena un agente **Deep Q-Network (DQN)** para interactuar con Pokémon Red a
través del emulador programable [PyBoy](https://github.com/Baekalfen/PyBoy). La
implementación usa TensorFlow/Keras, observaciones visuales y señales de la WRAM
del juego para estudiar exploración, interacción y progresión.

> **Estado del proyecto:** el notebook académico original se conserva sin
> alteraciones en `notebooks/archive/`. La implementación modular corrige errores
> comprobados en las transiciones del replay buffer y en varias direcciones de
> memoria. Los resultados históricos se documentan por trazabilidad, pero todavía
> no se ha generado un nuevo baseline con el código corregido porque la ROM no
> forma parte del proyecto.

## Objetivo

Construir un pipeline reproducible que permita entrenar, inspeccionar y extender
un agente DQN para Pokémon Red sin distribuir material propietario y sin ocultar
las limitaciones experimentales del prototipo original.

## Funcionamiento general

1. PyBoy ejecuta una ROM aportada localmente por el usuario en modo *headless*.
2. Cada pantalla se convierte a escala de grises, se redimensiona a `84 × 84` y
   se apilan cuatro frames para representar el estado.
3. La red convolucional estima un valor Q para cada acción disponible.
4. Una política epsilon-greedy elige acciones y reduce la exploración a lo largo
   del entrenamiento.
5. El replay buffer guarda cada transición completa —incluido el estado
   siguiente y la distinción entre terminación y truncamiento— y la red objetivo
   se sincroniza de forma periódica.
6. Las recompensas se calculan mediante estrategias configurables. La estrategia
   principal premia cambios visuales, bloques/mapas nuevos y progreso real de la
   Pokédex; la variante avanzada conserva el objetivo experimental de explorar
   mapas, entrada a combates y progreso heurístico de colección sin duplicar
   todo el entrenador. Un cambio de Pokédex o del equipo no demuestra por sí
   solo que ocurrió una captura.
7. Pesos, métricas y capturas se escriben bajo `results/runs/`, que está excluido
   de Git por defecto.

Los detalles, direcciones WRAM y decisiones de corrección se explican en
[`docs/methodology.md`](docs/methodology.md).

## Estructura del repositorio

```text
PokeTraining/
├── configs/                 # configuraciones reproducibles de ejemplo
├── data/                    # instrucciones; la ROM no se versiona
│   └── roms/
├── docs/                    # metodología, resultados y procedencia
├── notebooks/
│   ├── PokeTraining.ipynb   # recorrido seguro y modular
│   └── archive/             # notebook original, preservado byte a byte
├── results/
│   ├── figures/             # figuras históricas pequeñas y trazables
│   └── legacy_run.json      # métricas extraídas del notebook original
├── src/poketraining/        # paquete Python reutilizable
├── tests/                   # pruebas unitarias y smoke tests
├── pyproject.toml           # metadatos, dependencias y herramientas
├── requirements.txt         # dependencias directas verificadas
└── requirements-dev.txt     # pruebas, estilo, notebooks y seguridad
```

## Requisitos

- Python `3.10`, `3.11`, `3.12` o `3.13` de 64 bits.
- Una copia obtenida legalmente de Pokémon Red. La ROM debe ser compatible con
  las direcciones WRAM documentadas; no se incluye ni se descarga desde este
  repositorio.
- Memoria suficiente para TensorFlow y el replay buffer. Con capacidad `20 000`
  y estados completos de `84 × 84 × 4`, almacenar estado y estado siguiente
  puede requerir alrededor de 1,1 GB solo para observaciones.
- GPU opcional. TensorFlow funciona en CPU, aunque un entrenamiento largo puede
  tardar muchas horas o días.

## Instalación

```bash
git clone https://github.com/Cordovadcrk/PokeTraining.git
cd PokeTraining
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Para desarrollar, ejecutar pruebas o abrir el notebook:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
```

Las versiones directas están fijadas y fueron comprobadas conjuntamente. PyBoy
se mantiene en `2.7.0` porque [la versión `2.7.1` fue retirada](https://pypi.org/project/pyboy/2.7.1/)
por un problema que afectaba Pokémon Red/Blue.

## Preparación de los datos

Coloca la ROM legal en:

```text
data/roms/POKEMON_RED.gb
```

No renombres una ROM distinta para eludir la validación. Los archivos `*.gb`,
partidas, estados, RAM persistente, pesos y ejecuciones generadas están cubiertos
por `.gitignore`. Consulta [`data/README.md`](data/README.md) para conocer el
tratamiento de datos y artefactos.

## Ejecución

Entrenamiento con la estrategia principal y la llamada visible del notebook
histórico (200 episodios; el método base y los perfiles JSON usan 500):

```bash
poketraining train \
  --rom data/roms/POKEMON_RED.gb \
  --episodes 200 \
  --max-steps 5000 \
  --reward-mode exploration
```

Para usar el perfil avanzado, `--config` se coloca antes del subcomando:

```bash
poketraining --config configs/advanced.json train \
  --rom data/roms/POKEMON_RED.gb
```

Evaluación greedy de pesos locales:

```bash
poketraining play \
  --rom data/roms/POKEMON_RED.gb \
  --weights results/runs/<run-id>/checkpoints/dqn_final.weights.h5 \
  --episodes 5 \
  --max-steps 500
```

Debe usarse la misma configuración y el mismo espacio de acciones con que se
crearon los pesos. Para pesos del perfil avanzado, antepón
`--config configs/advanced.json` a `play`.

Una comprobación corta —no constituye un resultado científico— puede hacerse
con:

```bash
poketraining train \
  --rom data/roms/POKEMON_RED.gb \
  --episodes 1 \
  --max-steps 100 \
  --buffer-size 256 \
  --screenshot-interval 0
```

Para inspeccionar todas las opciones:

```bash
poketraining --help
poketraining train --help
```

El notebook modular se abre con:

```bash
jupyter lab notebooks/PokeTraining.ipynb
```

El entrenamiento no se inicia automáticamente al ejecutar todas sus celdas.

## Ejemplo desde Python

```python
from pathlib import Path

from poketraining.config import TrainingConfig
from poketraining.trainer import PokemonAITrainer

config = TrainingConfig(
    episodes=3,
    max_steps=500,
    reward_mode="exploration",
    seed=42,
)
trainer = PokemonAITrainer.from_rom(
    Path("data/roms/POKEMON_RED.gb"),
    config,
    Path("results/runs/example"),
)
metrics = trainer.train()
```

La semilla mejora la repetibilidad del muestreo y de TensorFlow, pero no garantiza
determinismo completo entre plataformas o versiones del emulador.

## Datos utilizados

No existe un dataset estático versionado. Las observaciones son frames generados
durante la emulación y los indicadores auxiliares se leen de la WRAM. La ROM es
propiedad de sus titulares y debe ser aportada por cada usuario. Las capturas,
checkpoints y métricas nuevas se consideran resultados generados, no datos fuente.

## Resultados históricos

El notebook original conserva una única salida textual de **tres episodios**,
aunque la celda visible solicita 200. Todos llegaron al límite de 5 000 pasos:

| Episodio | Pasos | Recompensa | Epsilon final |
|---------:|------:|-----------:|--------------:|
| 0 | 5 000 | 3 779,74 | 0,050 |
| 1 | 5 000 | 4 457,80 | 0,050 |
| 2 | 5 000 | 4 348,40 | 0,050 |

La recompensa media fue `4 195,31`. Estas cifras se extrajeron del notebook, no
se reprodujeron durante la preparación del repositorio. Debido a los errores
corregidos en terminación, mapa, Pokédex, replay buffer, red objetivo, epsilon e
inicialización del emulador, no son comparables directamente con futuras
ejecuciones.

![Recompensa histórica de tres episodios](results/figures/legacy_rewards.png)

No hay evidencia suficiente para afirmar convergencia, victorias, capturas o un
nivel de juego estable. Consulta [`docs/results.md`](docs/results.md) y
[`results/legacy_run.json`](results/legacy_run.json) para la trazabilidad completa.

## Pruebas y calidad

```bash
ruff check .
ruff format --check .
pytest
git ls-files -z | xargs -0 detect-secrets-hook \
  --baseline .secrets.baseline --no-verify
```

El entrenamiento real requiere una ROM y no forma parte de la suite automatizada.
Las pruebas usan datos sintéticos o dobles controlados para validar el buffer, las
recompensas, el preprocesamiento y la inicialización de las redes.

## Limitaciones

- El nuevo pipeline no se ha reentrenado en este repositorio; hace falta establecer
  un baseline corregido antes de comparar estrategias.
- Las direcciones WRAM dependen de la edición del juego y deben validarse para ROMs
  distintas de la versión soportada.
- El punto jugable inicial se alcanza mediante la secuencia de botones heredada
  del notebook; requiere una prueba de integración con la ROM legal antes de
  considerarlo estable para todas las plataformas.
- El reward shaping sigue siendo experimental y puede inducir conductas no deseadas.
- DQN con imágenes consume mucha memoria y cómputo; no se realizó una búsqueda de
  hiperparámetros ni una evaluación estadística con varias semillas.
- Los resultados históricos proceden de tres episodios y de una ejecución que no
  fue limpia de arriba abajo.
- El proyecto no está afiliado con Nintendo, Game Freak, The Pokémon Company ni los
  autores de PyBoy.

## Autoría y agradecimientos

**Mantenedor:** Ignacio Cordova
([@Cordovadcrk](https://github.com/Cordovadcrk)).

El proyecto académico original acredita también como coautores a **Joaquín López**
y **Vicente Silva**. El profesor **Miguel Cárcamo** se conserva en la procedencia
como contexto docente, no como autor del software. Véase
[`docs/provenance.md`](docs/provenance.md).

## Licencia

El código y la documentación propia se distribuyen bajo la [licencia MIT](LICENSE),
con los tres coautores originales en el aviso de copyright. La licencia no cubre
Pokémon Red, ROMs, marcas, gráficos del juego ni dependencias de terceros; cada
dependencia conserva su propia licencia.
