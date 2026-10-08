# Guía rápida docente — sesión 3 h

## Antes de la clase

- Subir a Colab o pedir clonar `ml_clasificacion/Ejemplo_Clase`.
- Verificar que corre la celda **Configuración** con `MODO_CLASE_3H = True`.
- Tener proyectada la **agenda** (tabla al inicio del notebook).

## Durante la sesión

Proyecte la celda **Objetivo / En palabras simples / tabla / Confirma por escrito** antes de cada código. Obligue a rellenar los “Anota: _____” en B3, C1 y E. Use los **checklists** al fin del bloque B y la **matriz traducida** en D.

| Min | Acción |
|-----|--------|
| 0–25 | Bloque A: tablas §1.1–1.2 + carga + glosario columnas + EDA + Actividad A1 |
| 25–75 | Bloque B: limpieza → pasos etiqueta SMA → tabla indicadores → correlación + Actividad B1 |
| 75–115 | Bloque C: diagrama X→ŷ → split → CV paso a paso → tabla algoritmos → benchmark + Actividad C1 |
| 115–155 | Bloque D: grid paso a paso → validación/confusión → importancia → guardar |
| 155–180 | Bloque E: backtest 7 pasos + conclusión + sección 9 (tarea) |

## Si se atrasa

1. Omitir heatmaps del grid (solo imprimir `best_params_`).
2. Omitir Actividad C1 en vivo (dejarla como tarea).
3. Reducir `N_FILAS` a `20_000` en config.

## Si sobra tiempo

- Ejecutar sección 9 con `MODO_CLASE_3H = False` (solo docente o avanzados).
- Discutir limitaciones: split aleatorio vs temporal, comisiones.
