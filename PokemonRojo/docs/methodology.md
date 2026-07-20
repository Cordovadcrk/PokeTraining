# Metodología

Este documento describe la metodología que debe seguir la versión modular del proyecto. Distingue el diseño científico conservado del notebook histórico de las correcciones necesarias para que futuras ejecuciones sean coherentes y reproducibles. Los resultados históricos no se recalcularon con estas correcciones.

## Objetivo

Entrenar un agente de aprendizaje por refuerzo profundo para interactuar con Pokémon Red mediante PyBoy. El agente observa imágenes del emulador, selecciona una acción discreta y actualiza una Deep Q-Network (DQN) a partir de transiciones almacenadas en una memoria de repetición.

## Entrada y compatibilidad de la ROM

La ROM no forma parte del repositorio. Cada persona debe aportar una copia obtenida legalmente en `data/roms/POKEMON_RED.gb`, según las instrucciones de `data/README.md`.

Las direcciones WRAM documentadas aquí corresponden al mapa internacional de Pokémon Red/Blue. Antes de entrenar se debe verificar la revisión de la ROM y detener la ejecución si no coincide con una revisión admitida. No se deben reutilizar direcciones de memoria de otra edición, idioma o revisión sin validarlas.

PyBoy busca automáticamente archivos `.ram` y `.rtc` junto a la ROM. El adaptador los sustituye por almacenamiento vacío en memoria para que una partida local o privada no cambie silenciosamente el estado inicial. La secuencia de botones heredada del notebook se ejecuta sobre ese estado limpio y solo después se guarda el punto reutilizable para `reset_game()`.

## Representación del estado

1. PyBoy ejecuta la ROM sin ventana para el entrenamiento.
2. Cada cuadro RGB del emulador se convierte a escala de grises.
3. El cuadro se redimensiona a `84 × 84` píxeles.
4. Se apilan cuatro cuadros consecutivos sobre el último eje, produciendo un estado de forma `(84, 84, 4)` y tipo `uint8`.
5. La red convierte los valores a `float32` y los normaliza al intervalo `[0, 1]`.

El apilado aporta información temporal sin alterar la observación visual original del proyecto.

## Espacio de acciones

La variante principal utiliza seis acciones discretas: arriba, abajo, izquierda, derecha, A y B. Cada acción debe incluir su evento de pulsación, varios ciclos del emulador y el evento de liberación correspondiente.

La variante experimental que añade START debe mantenerse como una configuración separada. Así se evita que ejecutar una celda redefina clases o cambie silenciosamente el tamaño de la salida de la red.

## Modelo DQN

La arquitectura conservada es:

1. `Conv2D(32, kernel_size=8, strides=4, activation="relu")`.
2. `Conv2D(64, kernel_size=4, strides=2, activation="relu")`.
3. `Conv2D(64, kernel_size=3, strides=1, activation="relu")`.
4. `Flatten`.
5. `Dense(512, activation="relu")`.
6. `Dense(num_actions)` para estimar un valor Q por acción.

Se utilizan una red en línea y una red objetivo. Ambas deben construirse y sincronizarse antes del primer paso de optimización. La red objetivo se actualiza después a intervalos configurables.

La acción del siguiente estado se elige con la red en línea y se evalúa con la red objetivo, manteniendo el esquema Double DQN presente en la intención del notebook.

## Memoria de repetición

Cada transición debe almacenarse explícitamente como:

```text
(state, action, reward, next_state, terminated, truncated)
```

No se debe reconstruir `next_state` mediante `observations[index + 1]`. Esa asociación deja de ser válida cuando el buffer circular sobrescribe posiciones y puede mezclar episodios. El muestreo debe comprobar que existen suficientes transiciones para formar el lote solicitado.

## Política de exploración

La política es epsilon-greedy: con probabilidad `epsilon` se toma una acción aleatoria y, en caso contrario, la acción con mayor valor Q.

El decaimiento de `epsilon` debe continuar entre episodios. Reiniciarlo a `1.0` al comienzo de cada episodio elimina el progreso acumulado de exploración. Las semillas de Python, NumPy, TensorFlow y del entorno deben registrarse junto con cada ejecución; aun así, el emulador y el hardware pueden introducir variación adicional.

## Recompensas y direcciones WRAM

Las recompensas deben calcularse a partir de cambios observables y transiciones verificadas, no de nombres inferidos para bytes de memoria.

| Dirección | Significado verificado | Uso permitido |
| --- | --- | --- |
| `0xD35E` | Número del mapa actual | Identificar regiones o mapas nuevos |
| `0xD363` | Coordenada Y del bloque actual | Identificar posiciones exploradas |
| `0xD364` | Coordenada X del bloque actual | Identificar posiciones exploradas |
| `0xD367` | Tileset del mapa | Información visual; no sustituye al identificador de mapa |
| `0xD057` | Estado/tipo de combate | Detectar si existe un combate, después de validar la revisión |
| `0xD163` | Número de Pokémon en el equipo | Señal auxiliar de colección; deja de aumentar cuando el equipo está lleno |
| `0xD2F7`–`0xD309` | Bitfield completo de Pokédex “obtenidos” (19 bytes) | Detectar bits nuevos; el bit de relleno posterior a la especie 151 se ignora |
| `0xD300`–`0xD301` | Bytes incluidos en el rango anterior para especies 73–88 | Corrigen el uso escalar del notebook histórico |
| `0xCFF3` | Nivel del Pokémon enemigo actual | Ponderar una entrada a combate; no prueba victoria |
| `0xD1F6` | Campo catch rate/held item del cuarto Pokémon | No usar como indicador de fin de episodio |

La clave de exploración espacial debe incluir `(map_id, block_x, block_y)`, usando `0xD35E`, `0xD364` y `0xD363`. `0xD367` identifica el tileset; distintos mapas pueden compartirlo.

Para progreso de colección se compara el estado anterior y posterior del bitfield completo y se recompensan únicamente bits que pasan de cero a uno. Un aumento del equipo se usa como señal auxiliar. Ninguna de esas señales distingue por sí sola entre captura, regalo, intercambio o evolución; las direcciones `0xD300` y `0xD301` tampoco son indicadores escalares de hitos genéricos.

No se ha verificado que `0xD3E8`–`0xD3EA` representen diálogos. Esas señales deben quedar deshabilitadas hasta documentar su significado para la revisión exacta de la ROM.

Pokémon Red no dispone de un “game over” simple en `0xD1F6`. Hasta implementar y probar una condición terminal válida, cada episodio debe terminar por el límite de pasos configurado o por una condición de blackout respaldada por pruebas específicas.

La recompensa de combate no debe sumarse en cada cuadro solo por permanecer en batalla, ya que eso podría favorecer combates prolongados. Debe basarse en eventos o diferencias verificables, como comenzar/terminar un combate o cambios de progreso definidos y probados.

## Entrenamiento y registro

Cada ejecución debe registrar, como mínimo:

- identificador y fecha UTC de la ejecución;
- versión de Python, plataforma y dependencias directas;
- huellas SHA-1 y SHA-256 de la ROM, sin distribuirla ni registrar su ruta;
- configuración e hiperparámetros;
- semillas;
- recompensa total y número de pasos por episodio;
- pérdidas de optimización;
- frecuencia de acciones;
- revisión Git cuando la ejecución proviene de un working tree.

La implementación escribe esos datos, junto con la configuración y las acciones, en `run_config.json`; una evaluación añade también la huella SHA-256 de los pesos cargados, sin su ruta. Las métricas de episodio se añaden a `metrics.jsonl` o `play_metrics.jsonl` y los pesos de entrenamiento quedan bajo `checkpoints/` dentro del directorio único de la ejecución.

Los pesos, capturas masivas y logs se guardan bajo `results/` y se ignoran por defecto. Solo se versionan resúmenes y figuras pequeñas que indiquen la ejecución de origen.

## Evaluación

La evaluación debe ejecutarse sin exploración aleatoria y en episodios reiniciados de forma independiente. No se deben usar los mismos episodios de entrenamiento como única evidencia de rendimiento. Cualquier comparación entre configuraciones requiere la misma ROM, presupuesto de pasos, criterio de terminación y conjunto de semillas.

## Fuentes técnicas

- [Mapa RAM de Pokémon Red/Blue](https://datacrystal.tcrf.net/wiki/Pokemon_Red%3ARAM_map): significado de `D1F6`, `D300`, `D301`, `D35E`, `D363`, `D364` y `D367`.
- [Disassembly pret/pokered](https://github.com/pret/pokered): referencia para validar símbolos y comportamiento por revisión.
- [Direcciones usadas por PokemonRedExperiments](https://raw.githubusercontent.com/PWhiddy/PokemonRedExperiments/master/baselines/memory_addresses.py): corroboración independiente de `0xD35E` como identificador de mapa.
- [Documentación de PyBoy](https://docs.pyboy.dk/): API del emulador y control programático.

Estas fuentes sustentan la corrección de las direcciones, pero no sustituyen pruebas de integración contra la ROM concreta aportada localmente.
