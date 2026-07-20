# Contribuir a PokeTraining

Gracias por mejorar el proyecto. Las contribuciones deben conservar su objetivo
educativo, separar claramente resultados históricos de ejecuciones nuevas y no
incluir material propietario.

## Entorno local

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
ruff check .
ruff format --check .
pytest
```

Use una rama descriptiva y commits pequeños. Una propuesta que cambie
recompensas, direcciones WRAM, terminación, preprocesamiento o arquitectura debe
explicar el motivo, aportar pruebas y distinguir el cambio metodológico de una
corrección comprobable.

## Datos y seguridad

No añada ROMs, partidas, estados, capturas masivas, pesos grandes, credenciales,
rutas privadas ni recursos extraídos del juego. Ejecute `git ls-files -z |
xargs -0 detect-secrets-hook --baseline .secrets.baseline --no-verify` antes de
abrir un pull request. Revise manualmente cualquier hallazgo nuevo. Consulte
`data/README.md` y `SECURITY.md`.

## Resultados

No presente una ejecución como reproducible sin configuración, revisión del
código, huellas de la ROM, semillas y métricas. Las figuras pequeñas que se
versionen deben identificar su procedencia y no contener recursos propietarios.
