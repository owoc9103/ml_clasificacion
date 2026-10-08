# Clase 11 — Clasificación y estrategia Bitcoin (Colab)

## Contenido

| Archivo | Descripción |
|---------|-------------|
| `Bitcoin_Estrategia_Clasificacion_Colab.ipynb` | Notebook Colab (~240 celdas), explícito y atomizado |
| `_build_notebook.py` | Generador del `.ipynb` (editar y ejecutar para regenerar) |
| `data/BitstampData_sample.csv` | Muestra local opcional (en clase se usa **Google Drive**) |

## Colab + Google Drive

1. Sube tu CSV a Drive (p. ej. `BitstampData_sample.csv`).
2. Abre el notebook en Colab.
3. En **Paso A2**, edita `RUTA_CSV` con la ruta real, por ejemplo:
   `/content/drive/MyDrive/ML_Fin_Econ/BitstampData_sample.csv`
4. Ejecuta **celda a celda** desde el montaje de Drive.
5. Docente: ver `GUIA_CLASE.md`.

## Modos

- `MODO_LIGERO = True` — menos filas y modelos (Colab más rápido).
- `MODO_LIGERO = False` — más filas y benchmark completo.
