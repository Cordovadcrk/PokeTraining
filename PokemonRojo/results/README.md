# Resultados de ejecuciones

Este directorio está reservado para artefactos generados por entrenamiento y evaluación. El repositorio no contiene una ejecución reproducible completa del notebook histórico; sus tres recompensas conservadas se documentan en `docs/results.md`.

## Organización recomendada

```text
results/
├── README.md
├── legacy_run.json
├── figures/
│   ├── legacy_actions.png
│   ├── legacy_loss.png
│   └── legacy_rewards.png
├── runs/
│   └── <run-id>/
│       ├── run_config.json
│       ├── metrics.jsonl
│       ├── play_metrics.jsonl
│       ├── checkpoints/
│       ├── figures/
│       └── screenshots/
└── figures/generated/
```

Cada `run-id` debe ser único y el programa rechaza reutilizar un directorio existente. `run_config.json` registra fecha UTC, revisión Git cuando está disponible, Python, plataforma, dependencias, huellas de la ROM, configuración, semillas y acciones. En evaluación añade la huella SHA-256 de los pesos. Nunca incluye la ROM, los pesos ni sus rutas.

## Artefactos ignorados por defecto

- `runs/`: logs y métricas detalladas de ejecuciones locales.
- `runs/<run-id>/checkpoints/`: pesos intermedios y finales, normalmente grandes.
- `figures/generated/`: gráficos y capturas recreables.
- capturas masivas, videos, archivos de TensorBoard y perfiles.

Estos artefactos pueden crecer rápidamente y no deben añadirse accidentalmente al historial Git.

## Artefactos que pueden versionarse

Se pueden versionar de forma deliberada:

- tablas de resumen pequeñas y verificadas;
- configuraciones necesarias para interpretar un resultado publicado;
- una selección reducida de figuras directamente bajo `figures/`;
- metadatos sin información privada;
- scripts que regeneren las figuras.

Cada figura curada debe indicar el `run-id` de origen y el script o procedimiento que la produjo. Una imagen sin sus métricas o configuración de origen no debe presentarse como evidencia reproducible.

## Pesos y artefactos grandes

Los checkpoints valiosos deben publicarse mediante Git LFS o como artefactos de una Release, acompañados de tamaño, SHA-256, configuración y licencia. No deben añadirse al Git normal.

Antes de versionar cualquier resultado se debe comprobar que no contiene ROM, estados del emulador, recursos propietarios, credenciales, nombres de usuario o rutas absolutas.
