# Resultados históricos

El notebook original conserva la salida de una ejecución de tres episodios. Estos son los únicos valores numéricos de recompensa disponibles:

| Episodio | Recompensa total |
| ---: | ---: |
| 0 | 3779.74 |
| 1 | 4457.80 |
| 2 | 4348.40 |
| **Media** | **4195.31** |

## Alcance de estos resultados

- Los valores se transcribieron de la salida incrustada en el notebook original.
- La ejecución no se reprodujo durante la preparación del repositorio.
- No se conservan semillas, versiones exactas de todas las dependencias, huella de la ROM, pesos ni logs estructurados que permitan repetirla de forma exacta.
- La celda fuente indica 200 episodios, pero la salida histórica solo contiene tres. No se infieren resultados para los 197 episodios restantes.
- Las imágenes de pérdida, recompensa y distribución de acciones están incrustadas, pero no se conservan sus series numéricas completas. Las tres figuras se extrajeron sin modificación a `results/figures/`; por ello no se reportan métricas adicionales derivadas de los gráficos.
- Estos valores no demuestran por sí solos que el agente haya aprendido una política útil o generalizable.

## Comparabilidad después de las correcciones

Los resultados anteriores se obtuvieron con la lógica histórica. No son directamente comparables con futuras ejecuciones que corrijan la condición terminal, el identificador de mapa, los bitfields de Pokédex, la memoria de repetición, la sincronización de la red objetivo o el reinicio de epsilon.

Las primeras ejecuciones corregidas deben considerarse una nueva línea base. Toda comparación futura deberá usar la misma ROM verificada, configuración, presupuesto de pasos, criterio de terminación y conjunto de semillas.
