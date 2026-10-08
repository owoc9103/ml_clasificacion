# Guía rápida docente — sesión 3 h

## Antes de la clase

- Subir a Colab o pedir clonar `ml_clasificacion/Ejemplo_Clase`.
- Verificar que corre la celda **Configuración** con `MODO_CLASE_3H = True`.
- Tener proyectada la **agenda** (tabla al inicio del notebook).

## Durante la sesión

| Min | Acción |
|-----|--------|
| 0–25 | Bloque A: problema + carga + EDA compacto + Actividad A1 |
| 25–75 | Bloque B: limpieza → SMA → indicadores (celda larga: dejar correr) → correlación + Actividad B1 |
| 75–115 | Bloque C: split → métricas → gráficos hiperparámetros → benchmark 5 modelos + Actividad C1 (recall) |
| 115–155 | Bloque D: grid RF → validación → importancia → guardar modelo |
| 155–180 | Bloque E: backtest + conclusión + mencionar sección 9 (tarea) |

## Si se atrasa

1. Omitir heatmaps del grid (solo imprimir `best_params_`).
2. Omitir Actividad C1 en vivo (dejarla como tarea).
3. Reducir `N_FILAS` a `20_000` en config.

## Si sobra tiempo

- Ejecutar sección 9 con `MODO_CLASE_3H = False` (solo docente o avanzados).
- Discutir limitaciones: split aleatorio vs temporal, comisiones.
