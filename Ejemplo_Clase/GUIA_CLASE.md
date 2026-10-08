# Guía docente — Notebook Bitcoin + clasificación

## Antes de la sesión

1. Pedir a los estudiantes subir el CSV a **Google Drive** y anotar la ruta.
2. Verificar que en Colab el **Paso A1** monta Drive y el **Paso A2** tiene `RUTA_CSV` correcta.
3. Probar una corrida con `MODO_LIGERO = True`.

## Durante la sesión

- Proyecte cada celda **Objetivo / En palabras simples / tabla / Confirma por escrito** antes del código.
- Parte **A:** Drive + EDA — no avanzar si `¿Existe el archivo?` es False.
- Parte **B:** checklist al final (signal, SMA fuera de X, filas tras dropna).
- Parte **C–D:** train vs validation, CV, Grid RF, matriz de confusión traducida.
- Parte **E:** backtest y comparación ML vs SMA.

## Errores frecuentes

| Síntoma | Causa | Solución |
|---------|--------|----------|
| `FileNotFoundError` | `RUTA_CSV` mal escrita | Copiar ruta desde panel de archivos de Drive |
| Accuracy ~90% trivial | Clase desbalanceada | Ver `value_counts` de signal; discutir precision |
| Backtest flat | No ejecutó predicciones antes | Orden: modelo final → validation → backtest |
