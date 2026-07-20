# Datos locales

Este proyecto no distribuye la ROM de Pokémon Red. Para ejecutar el emulador, cada usuario debe aportar una copia obtenida legalmente y colocarla en:

```text
data/roms/POKEMON_RED.gb
```

No se proporciona ni se debe añadir un enlace de descarga.

## Preparación

1. Cree el directorio local `data/roms/` si no existe.
2. Copie allí su ROM legal y renómbrela exactamente como `POKEMON_RED.gb` para usar los ejemplos.
3. Calcule su huella SHA-1 sin modificar el archivo:

   ```bash
   shasum -a 1 data/roms/POKEMON_RED.gb
   ```

4. Compruebe que la revisión es compatible con las direcciones WRAM documentadas antes de entrenar.

La revisión internacional de Pokémon Red usada como referencia por el disassembly [pret/pokered](https://github.com/pret/pokered) tiene SHA-1 `ea9bcae617fdf159b045185467ae58b2e4a48b9a`. Si la huella local no coincide, no se debe asumir que las direcciones de memoria sean equivalentes.

## Política de versionado

`data/roms/` debe permanecer ignorado por Git. También deben ignorarse:

- ROM de Game Boy y Game Boy Advance;
- partidas guardadas y archivos RTC;
- estados del emulador;
- volcados de memoria;
- recursos extraídos del juego;
- datos privados o con una licencia incompatible.

Solo este archivo informativo y, si hace falta para conservar la carpeta, un `.gitkeep` pueden versionarse dentro de `data/`.

## Rutas portables

El CLI exige `--rom`. La ruta relativa `data/roms/POKEMON_RED.gb` funciona al ejecutar desde la raíz del repositorio; desde otro directorio debe proporcionarse la ruta correspondiente. El código la convierte internamente en una ruta absoluta, pero no la registra en los resultados.

PyBoy puede cargar automáticamente los archivos laterales `<ROM>.ram` y `<ROM>.rtc`. PokeTraining proporciona almacenamiento vacío en memoria y no consume esos sidecars, evitando que una partida previa o privada cambie una ejecución. Tampoco escribe RAM al cerrar el emulador.

## Datos derivados

Métricas, capturas, pesos y logs no son datos de entrada y deben escribirse bajo `results/`. Cada ejecución debe registrar únicamente la huella de la ROM; nunca debe copiar la ROM al directorio de resultados.
