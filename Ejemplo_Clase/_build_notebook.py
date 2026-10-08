# -*- coding: utf-8 -*-
"""Genera el notebook Colab: clasificación + estrategia Bitcoin (español)."""
import json
from pathlib import Path


def md(text: str):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": text.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


def step_md(
    step: str,
    title: str,
    hace: str,
    mirar: str,
    trading: str = "",
    ml: str = "",
    objetivo: str = "",
    en_palabras: str = "",
    anota: str = "",
):
    """Markdown explícito antes de cada celda de código."""
    obj = objetivo or f"Completar «{title}» y verificar la salida antes de continuar."
    simple = en_palabras or (
        f"{hace} En trading: {trading}" if trading else hace
    )
    nota = anota or f"La salida coincide con: {mirar}"
    lines = [
        f"### {step} — {title}\n\n",
        f"**Objetivo:** {obj}\n\n",
        f"**En palabras simples:** {simple}\n\n",
        "**Detalle (lee antes de ejecutar):**\n\n",
        "| Pregunta | Respuesta |\n|----------|----------|\n",
        f"| ¿Qué hace el código? | {hace} |\n",
        f"| ¿Qué debo ver al ejecutar? | {mirar} |\n",
    ]
    if trading:
        lines.append(f"| ¿Qué implica para Bitcoin? | {trading} |\n")
    if ml:
        lines.append(f"| ¿Qué implica para el modelo? | {ml} |\n")
    lines.append(f"\n> **Confirma por escrito (cuaderno o chat):** {nota}\n")
    return md("".join(lines))


def post_md(mirar: str, trading: str = "", ml: str = "", concluye: str = ""):
    """Interpretación explícita después de ejecutar."""
    lines = ["**✓ Celda ejecutada — interpreta así:**\n\n", f"1. {mirar}\n"]
    if trading:
        lines.append(f"2. **Trading:** {trading}\n")
    if ml:
        lines.append(f"3. **ML:** {ml}\n")
    if concluye:
        lines.append(f"\n**Conclusión explícita:** {concluye}\n")
    return md("".join(lines))


def block_intro(parte: str, historia: str, al_terminar: str, pregunta: str):
    return md(f"""---
## Parte {parte}

**Qué hacemos aquí (sin jerga):** {historia}

**Al terminar la parte {parte}, debes poder decir en voz alta:** {al_terminar}

**Pregunta guía:** {pregunta}
""")


def feature_step(step: str, col: str, calculo: str, trading_tip: str, ml_tip: str = ""):
    ml_tip = ml_tip or (
        f"La columna `{col}` entra en **X**; el Random Forest la usará junto con las demás "
        "para predecir si `signal` es 0 o 1."
    )
    cells.append(step_md(
        step,
        f"Indicador `{col}`",
        calculo,
        f"Se imprime el último valor de `{col}` (número finito, no NaN).",
        trading=trading_tip,
        ml=ml_tip,
        objetivo=f"Dejar creada la columna `{col}` en `dataset`.",
        en_palabras=f"Convertimos el historial de precios en el número `{col}`; "
        f"ese número resume una idea de mercado (tendencia, momentum, etc.).",
        anota=f"Último `{col}` = _____ (rellena). ¿Tiene sentido frente al Close actual?",
    ))


cells = []

cells.append(md("""# Estrategia Bitcoin con clasificación (Google Colab)

**Datos:** archivo CSV de Bitstamp en **tu Google Drive** (tú defines la ruta en el notebook).

**Pedagogía:** **una idea por celda** — lee **Objetivo → En palabras simples → tabla → Confirma por escrito** antes de ejecutar código.

**Partes del cuaderno:** A datos · B preparación · C modelos · D Random Forest final · E backtest.
"""))

cells.append(md("""## Cómo leer este notebook (obligatorio)

1. **Aplicación Bitcoin:** en cada minuto tenemos precio → construimos indicadores → definimos si “conviene estar largo” (`signal`) → entrenamos un modelo que **imita/mejora** esa regla → simulamos ganancias con **backtest**.
2. **Modelos ML:** EDA → separar Y/X → train/validation → comparar algoritmos → Grid Search → evaluar → guardar → backtest.
3. **No es lo mismo:** alta **accuracy** ≠ ganar dinero; **signal=1** ≈ “comprado” en esta demo; el **backtest** es donde ves si la señal ML paga (sin comisiones).

**Convención:** `Y` = columna `signal` (0 no largo, 1 largo). `X` = todo lo demás numérico. **Validation** = datos que el modelo no usa para entrenar en el paso final.
"""))

cells.append(md("""## Contenido del cuaderno

* [Configuración de la sesión](#cfg)
* [1. Definición del problema](#0)
* [2. Librerías y datos](#1)
* [3. EDA (breve)](#2)
* [4. Preparación de datos](#3)
* [5. Evaluación y modelos](#4)
* [6. Grid Search](#5)
* [7. Modelo final](#6)
* [8. Backtesting](#7)
* [9. Modo con más filas (opcional)](#8)
* [Google Drive — ruta del CSV](#drive)
"""))

cells.append(md("""<a id='0'></a>
# 1. Definición del problema

## 1.1 Enfoque de la aplicación (Bitcoin / trading)

| Paso | Qué hacemos | Para qué sirve en trading |
|------|-------------|---------------------------|
| 1 | Cargar OHLCV minuto a minuto (Bitstamp) | Tener la serie de precios y liquidez |
| 2 | Limpiar y ordenar en el tiempo | Evitar señales con huecos falsos |
| 3 | Definir **etiqueta** con SMA 10 vs SMA 60 | Traducir “¿estoy alcista de corto plazo?” en 0/1 |
| 4 | Construir **indicadores** (RSI, estocástico, etc.) | Dar al ML pistas de tendencia y momentum |
| 5 | Entrenar clasificadores | Aproximar (y a veces mejorar) la regla SMA |
| 6 | Elegir modelo + hiperparámetros | Balance entre acierto y estabilidad |
| 7 | Evaluar en **validación** (matriz de confusión) | Ver falsas compras vs subidas perdidas |
| 8 | **Backtest** simple | Simular PnL antes de arriesgar capital |

**Señal de referencia (regla SMA):**

- **1 = compra / largo:** SMA₁₀(Close) > SMA₆₀(Close).
- **0 = no largo:** en caso contrario.

El ML **no sustituye** esa idea: aprende una función $f(X_t) \\approx \\text{signal}_t$ usando muchos indicadores a la vez.

### Historia completa en una frase (léela en voz alta)

> Cargamos minutos de Bitcoin → limpiamos → marcamos cada minuto con **0/1** según medias móviles → calculamos RSI, EMA, etc. → entrenamos un clasificador para predecir ese 0/1 → medimos si acierta y si **ganaría dinero** retrasando la señal un minuto.

### Dos hilos que SIEMPRE van juntos

| | **Hilo trading** | **Hilo ML** |
|---|------------------|-------------|
| **Entrada** | Precio/volumen minuto a minuto | Filas = observaciones, columnas = features |
| **Decisión** | ¿Largo (1) o no (0)? | Clasificación binaria |
| **Éxito en clase** | Entiendes falsas compras y backtest | Entiendes train/val, CV y Grid Search |

## 1.2 Enfoque de modelos (clasificación supervisada)

| Paso ML | Sección del notebook | Idea |
|---------|----------------------|------|
| 1 | EDA | Entender datos antes de modelar |
| 2 | Preparación | $Y$ = signal, $X$ = indicadores |
| 3 | Train / validation split | Medir generalización |
| 4 | Validación cruzada | Comparar algoritmos sin mirar el test |
| 5 | Grid Search | Ajustar hiperparámetros del ganador |
| 6 | Modelo final + métricas | Reportar en hold-out |
| 7 | Persistencia + backtest | Cerrar el ciclo “investigación → decisión” |

**Datos:** sube tu CSV a Drive (p. ej. `BitstampData_sample.csv` o el archivo completo) y apunta `RUTA_CSV` a esa ubicación.
"""))

cells.append(md("""<a id='1'></a>
# 2. Inicio — librerías y datos
<a id='1.1'></a>
## 2.1 Cargar librerías

Usamos `pandas`, `sklearn`, gráficos y (opcional) `MLPClassifier` como red neuronal simple en Colab.
"""))

cells.append(step_md(
    "Paso 2.1a", "Detectar Colab",
    "Comprueba si el notebook corre en Google Colab.",
    "Debe imprimir `Colab: True` o `False`.",
    ml="En local no hace falta `pip`; en Colab instalamos paquetes si faltan.",
))

cells.append(code("""try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
print('Colab:', IN_COLAB)
"""))

cells.append(step_md(
    "Paso 2.1b", "Instalar dependencias (solo Colab)",
    "Instala `seaborn` y `scikit-learn` si estamos en Colab.",
    "Si no estás en Colab, esta celda no instala nada (está bien).",
))

cells.append(code("""if IN_COLAB:
    !pip -q install seaborn scikit-learn
else:
    print('Entorno local: usa tu venv / conda.')
"""))

cells.append(step_md(
    "Paso 2.1c", "NumPy, Pandas y gráficos",
    "Importa herramientas para tablas y visualización.",
    "Sin errores de importación.",
    trading="Aquí manipularemos precios minuto a minuto (OHLCV).",
))

cells.append(code("""import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
import seaborn as sns
"""))

cells.append(step_md(
    "Paso 2.1d", "Scikit-learn — preparación y validación",
    "Importa partición de datos, CV y escalado.",
    "Nombres como `train_test_split`, `GridSearchCV`.",
    ml="Estas funciones preparan partición de datos, CV y escalado.",
))

cells.append(code("""from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, KFold, cross_val_score, GridSearchCV, validation_curve,
)
"""))

cells.append(step_md(
    "Paso 2.1e", "Scikit-learn — modelos",
    "Importa los clasificadores del benchmark.",
    "Lista de clases sin error.",
    ml="Cada algoritmo es un candidato para aproximar la etiqueta SMA.",
))

cells.append(code("""from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
"""))

cells.append(step_md(
    "Paso 2.1f", "Métricas y guardado",
    "Importa métricas de clasificación y `pickle` para persistir el modelo.",
    "Funciones `accuracy_score`, `confusion_matrix`, etc.",
    trading="Precision/recall traducen errores en falsas compras o subidas perdidas.",
))

cells.append(code("""from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pickle import dump, load
print('Librerías listas.')
"""))

cells.append(md("""<a id='cfg'></a>
## Configuración de la sesión (ejecutar primero)

- **`MODO_LIGERO = True`:** 30 000 filas, 3 folds CV, 5 modelos en benchmark, grid RF pequeño (Colab más rápido).
- **`MODO_LIGERO = False`:** 100 000 filas, 10 folds, 9 modelos, grid más grande (más pesado).
"""))

cells.append(step_md(
    "Paso 2.2a", "Modo de cómputo",
    "Activa modo ligero o completo según tamaño de CSV y potencia de Colab.",
    "`MODO_LIGERO` True o False impreso.",
    ml="Menos filas/folds = entrenamiento más rápido, menos exhaustivo.",
))

cells.append(code("""MODO_LIGERO = True
print('Modo ligero:', MODO_LIGERO)
"""))

cells.append(step_md(
    "Paso 2.2b", "Tamaño de muestra y CV",
    "Define cuántas filas finales y cuántos folds usar.",
    "Imprime `N_FILAS` y `N_FOLDS`.",
    trading="Modelar solo el tramo reciente enfatiza el régimen de mercado actual (con trade-offs).",
    ml="Más folds = estimación más estable pero más lento.",
))

cells.append(code("""N_FILAS = 30_000 if MODO_LIGERO else 100_000
N_FOLDS = 3 if MODO_LIGERO else 10
CV_SEED = 7
print('Filas para modelado:', N_FILAS, '| Folds CV:', N_FOLDS, '| Seed:', CV_SEED)
"""))

cells.append(block_intro(
    "A",
    "Vamos a **conectar Google Drive**, **cargar** el CSV de Bitstamp y **mirar** si los precios tienen sentido antes de calcular nada.",
    "Dónde está tu archivo en Drive, cuántas filas hay y qué significa cada columna OHLCV.",
    "¿Por qué cada fila es un minuto y por qué eso importa para la estrategia?",
))
cells.append(md("""<a id='drive'></a>
<a id='1.2'></a>
### 2.3 Datos desde Google Drive

**Antes de la clase:** sube tu CSV a Drive (ejemplo de nombre: `BitstampData_sample.csv`).

**Ruta típica en Colab:** `/content/drive/MyDrive/TU_CARPETA/BitstampData_sample.csv`

**Importante:** en la celda **Paso A2** debes **editar** `RUTA_CSV` con la ruta real de tu archivo (copiar desde el explorador de archivos de Drive).
"""))

cells.append(step_md(
    "Paso A1", "Montar Google Drive",
    "En Colab, monta Drive para leer archivos con ruta `/content/drive/...`.",
    "Mensaje de autorización (Colab) o aviso de entorno local.",
    objetivo="Tener acceso de lectura a tu carpeta de Drive.",
    en_palabras="Conectamos Colab con tu Google Drive como si fuera un disco.",
    anota="¿Apareció el enlace de permisos de Google y pudiste montar Drive? (sí/no)",
))

cells.append(code("""if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
else:
    print('No estás en Colab: salta el montaje; usa una ruta local en RUTA_CSV.')
"""))

cells.append(step_md(
    "Paso A2", "Definir RUTA_CSV (EDITAR AQUÍ)",
    "Variable de texto con la ruta **completa** al CSV en Drive (o local).",
    "Imprime la ruta que configuraste.",
    trading="Este archivo contiene todos los minutos de precio que usará la estrategia.",
    objetivo="Apuntar al archivo correcto antes de leer.",
    en_palabras="Le dices al notebook dónde está tu tabla de precios dentro de Drive.",
    anota="Mi RUTA_CSV es: _________________________________ (copia la misma ruta del código).",
))

cells.append(code("""from pathlib import Path

# --- EDITA SOLO ESTA LÍNEA (ruta en Drive o local) ---
RUTA_CSV = '/content/drive/MyDrive/ML_Fin_Econ/BitstampData_sample.csv'

print('RUTA_CSV configurada:')
print(RUTA_CSV)
"""))

cells.append(step_md(
    "Paso A3", "Comprobar que el archivo existe",
    "Verifica que `Path(RUTA_CSV).exists()` sea True.",
    "Debe imprimir `¿Existe el archivo? True`.",
    objetivo="Detectar rutas mal escritas antes de un error críptico.",
    en_palabras="Comprobamos que el CSV está donde crees que está.",
    anota="Si sale False: corrige mayúsculas, carpetas y extensión `.csv`.",
))

cells.append(code("""archivo_datos = Path(RUTA_CSV)
print('¿Existe el archivo?', archivo_datos.exists())
if not archivo_datos.exists():
    raise FileNotFoundError(
        'No se encontró el CSV. Sube el archivo a Drive y corrige RUTA_CSV en el Paso A2.'
    )
"""))

cells.append(step_md(
    "Paso A4", "Leer el CSV en pandas",
    "`read_csv(RUTA_CSV)` carga todo en `dataset`.",
    "Imprime número de filas y columnas.",
    trading="Cada fila = un minuto de mercado Bitstamp.",
))

cells.append(code("""dataset = read_csv(RUTA_CSV)
print('Archivo leído:', archivo_datos)
print('Filas cargadas:', len(dataset))
print('Columnas:', list(dataset.columns))
"""))

cells.append(md("""### Paso a paso — qué es cada columna (Bitcoin)

| Columna | Significado en mercado |
|---------|-------------------------|
| `Open`, `High`, `Low`, `Close` | Precio en ese minuto (vela de 1 min) |
| `Volume_(BTC)` | Cantidad negociada en BTC (liquidez relativa) |
| `Volume_(Currency)` | Volumen en moneda fiat (se elimina más adelante en el repo) |
| `Weighted_Price` | Precio promedio ponderado del minuto |

Cada **fila = un minuto**. La estrategia decide en cada minuto si el régimen es “alcista de corto plazo” (etiqueta) usando medias e indicadores **calculados solo con información hasta ese minuto** (cuidado con *look-ahead* en proyectos propios).
"""))

cells.append(step_md(
    "Paso A4", "Herramientas de visualización en notebook",
    "Configura `display` y precisión de pandas.",
    "`Librerías listas` implícito al ejecutar sin error.",
))

cells.append(code("""try:
    from IPython.display import display
except ImportError:
    display = print
set_option('display.width', 100)
set_option('precision', 3)
"""))

cells.append(step_md(
    "Paso A5", "Dimensiones del dataset",
    "Muestra filas × columnas.",
    "Tupla `(filas, columnas)` — anota el número de filas de tu CSV.",
))

cells.append(code("""print('Shape (filas, columnas):', dataset.shape)
"""))

cells.append(post_md(
    "¿Cuántas filas tienes? En clase usaremos solo las últimas `N_FILAS` minutos.",
    ml="Menos filas = entrenamiento más rápido pero menos historia.",
))

cells.append(step_md(
    "Paso A6", "Cola de la serie",
    "Muestra los últimos 5 minutos cargados.",
    "Precios y volúmenes recientes; orden temporal (más nuevo abajo).",
    trading="Verifica que `Close` tenga sentido (no ceros ni saltos absurdos).",
))

cells.append(code("""display(dataset.tail(5))
"""))

cells.append(step_md(
    "Paso A7", "Estadísticas descriptivas",
    "Resumen numérico por columna (`describe`).",
    "Columna `count`: si es menor que filas, hay NaNs.",
    trading="Rango de `Close` te sitúa en qué época/precio está la muestra.",
))

cells.append(code("""display(dataset.describe())
"""))

cells.append(post_md(
    "Compara `count` con el número de filas. Anota min/max de `Close`.",
    ml="NaNs o outliers extremos pueden distorsionar indicadores y modelos.",
    concluye="Si `count` < filas, hay missing data; la limpieza de la parte B lo tratará.",
))

cells.append(md("""> **Actividad A1 (3 min):** *¿Por qué no entrenar con toda la serie desde 2012 en Colab free?* (memoria, tiempo, cambio de régimen).
"""))

cells.append(block_intro(
    "B",
    "Preparamos la tabla que el ML consumirá: limpiar → **crear la etiqueta signal (SMA)** → calcular indicadores → quitar columnas que harían trampa.",
    "Qué es `signal`, qué entra en **X**, y por qué quitamos las SMA de la etiqueta antes de entrenar.",
    "Si el modelo viera `short_mavg` en X, ¿estaría aprendiendo o copiando la regla?",
))
cells.append(md("""<a id='3'></a>
# 4. Preparación de datos
<a id='3.1'></a>
## 4.1 Limpieza de datos
"""))

cells.append(step_md(
    "Paso B1.1", "Detectar valores faltantes",
    "Pregunta si existe algún NaN en la tabla.",
    "`Null Values = False` es lo ideal antes de indicadores.",
    trading="Huecos en precios generan señales falsas si no se tratan.",
))

cells.append(code("""print('¿Hay NaNs?', dataset.isnull().values.any())
print('NaNs por columna:\\n', dataset.isnull().sum())
"""))

cells.append(step_md(
    "Paso B1.2", "Forward fill",
    "Rellena NaNs con el último valor observado (forward fill).",
    "Repite el chequeo mental: menos NaNs tras ejecutar.",
    trading="Asume que el último precio sigue vigente en el hueco — razonable en minutos, dudoso en días.",
))

cells.append(code("""dataset[dataset.columns.values] = dataset[dataset.columns.values].ffill()
print('¿NaNs tras ffill?', dataset.isnull().values.any())
"""))

cells.append(step_md(
    "Paso B1.3", "Quitar marca de tiempo",
    "Elimina `Timestamp`; el orden de filas ya indexa el tiempo.",
    "Columna `Timestamp` ya no está en `dataset.columns`.",
    ml="El modelo no usa la fecha cruda; solo features numéricas.",
))

cells.append(code("""dataset = dataset.drop(columns=['Timestamp'])
print('Columnas:', list(dataset.columns))
"""))

cells.append(md("""<a id='3.2'></a>
## 4.2 Datos categóricos (plantilla maestra)

En otros problemas de ML se codifican variables categóricas (one-hot). **En Bitcoin todas las entradas son numéricas** (OHLCV). Si añadieras calendario (día de la semana), ahí codificarías esa variable.
"""))

cells.append(md("""<a id='3.3'></a>
## 4.3 Preparar datos para clasificación

### Paso a paso — construcción de la etiqueta (aplicación)

1. **Elegir ventanas:** 10 min (corto) y 60 min (largo).
2. **Calcular SMA** sobre `Close`.
3. **Comparar:** corto > largo → etiqueta **1** (régimen alcista de corto plazo).
4. **Operativa:** 1 ≈ “estar largo”; 0 ≈ “no largo” (sin short explícito en el caso base).

$$\\text{signal}_t = \\mathbb{1}\\{\\text{SMA}_{10}(P)_t > \\text{SMA}_{60}(P)_t\\}$$

Las SMA usadas para etiquetar se **eliminan** de $X$ antes de entrenar para evitar copiar la regla trivialmente.
"""))

cells.append(step_md(
    "Paso B3.1", "SMA corta (10 min)",
    "Calcula media móvil simple de `Close` con ventana 10.",
    "`short_mavg` cerca de `Close` pero más suave.",
    trading="Captura tendencia muy reciente (≈10 minutos).",
))

cells.append(code("""dataset['short_mavg'] = dataset['Close'].rolling(window=10, min_periods=1, center=False).mean()
dataset[['Close', 'short_mavg']].tail(3)
"""))

cells.append(step_md(
    "Paso B3.2", "SMA larga (60 min)",
    "Media móvil de 60 minutos sobre `Close`.",
    "`long_mavg` cambia más lento que `short_mavg`.",
    trading="Proxy de tendencia de la última hora.",
))

cells.append(code("""dataset['long_mavg'] = dataset['Close'].rolling(window=60, min_periods=1, center=False).mean()
dataset[['Close', 'short_mavg', 'long_mavg']].tail(3)
"""))

cells.append(step_md(
    "Paso B3.3", "Crear etiqueta `signal`",
    "Crea la columna `signal`: 1.0 si `short_mavg > long_mavg`, si no 0.0.",
    "Columna `signal` solo con 0.0 y 1.0; en `tail` ves 0/1 junto a las medias.",
    trading="**1 = en este minuto la regla SMA dice ‘entorno alcista de corto plazo’** (en la demo = estar largo). **0 = no.**",
    ml="**Esta columna es Y (la verdad que el modelo intentará predecir).** No confundir con la predicción ŷ (viene mucho más adelante).",
    objetivo="Tener la variable objetivo Y definida con la regla SMA 10 vs 60.",
    en_palabras="Traducimos el cruce de medias en un sí/no numérico para que sklearn pueda entrenar.",
    anota="En tus palabras: signal=1 significa _______________; signal=0 significa _______________.",
))

cells.append(code("""dataset['signal'] = np.where(dataset['short_mavg'] > dataset['long_mavg'], 1.0, 0.0)
dataset[['Close', 'short_mavg', 'long_mavg', 'signal']].tail(5)
"""))

cells.append(step_md(
    "Paso B3.4", "Balance de clases",
    "Cuenta proporción de 0 y 1 en `signal`.",
    "Porcentajes en `value_counts(normalize=True)`.",
    ml="Clase muy mayoritaria infla accuracy de un modelo trivial.",
    trading="Muchos 1 → mercado alcista en la ventana; pocos 1 → lateral/bajista.",
))

cells.append(code("""dataset['signal'].value_counts(normalize=True)
"""))

cells.append(post_md(
    "Si ~90% es clase 1, un modelo que siempre predice 1 ya tiene ~90% accuracy.",
    trading="Por eso en long-only importan precision y falsos positivos.",
))

cells.append(md("""> **Actividad B1 (5 min):** si la clase 1 tiene > 55 %, ¿qué accuracy obtiene un modelo que *siempre* predice 1? (Respuesta: ~proporción de unos.) Por eso miramos precision/recall en trading.

"""))

cells.append(md("""<a id='3.4'></a>
## 4.4 Ingeniería de características — indicadores técnicos

### Paso a paso — para qué sirve cada familia (Bitcoin)

| Familia | Variables (ej.) | Lectura en trading |
|---------|-----------------|---------------------|
| **Tendencia** | EMA10/30/200, MA21/63/252 | Precio vs media: ¿vamos arriba o abajo del consenso? |
| **Momentum** | MOM10/30, ROC10/30 | Velocidad del cambio de precio (% o diferencia) |
| **Sobrecompra/sobreventa** | RSI10/30/200 | RSI alto → muchas subidas recientes; bajo → muchas caídas |
| **Estocástico** | %K, %D | Posición del cierre dentro del rango High–Low reciente |
| **Liquidez / nivel** | `Close`, `Volume_(BTC)`, `Weighted_Price` | Contexto de precio y actividad |

A continuación: **un cálculo por celda**. Lee **Objetivo / En palabras simples** antes de ejecutar.
"""))

# --- Indicadores técnicos (generados celda a celda) ---
cells.append(step_md(
    "Paso B4.0a", "Definir función EMA",
    "Crea la función auxiliar `EMA(df, n)`.",
    "Solo define la función; aún no modifica `dataset`.",
    trading="EMA pondera más los cierres recientes que una SMA.",
))

cells.append(code("""def EMA(df, n):
    EMA = pd.Series(df['Close'].ewm(span=n, min_periods=n).mean(), name='EMA_' + str(n))
    return EMA
"""))

for n, tip in [
    (10, "Si Close > EMA10, el precio va por encima de la tendencia muy reciente."),
    (30, "Compara el precio con el consenso de ~30 minutos."),
    (200, "Tendencia lenta: útil para ver si el minuto actual va ‘con’ o ‘contra’ el tramo largo."),
]:
    feature_step(
        f"Paso B4.1.{n}",
        f"EMA{n}",
        f"`dataset['EMA{n}'] = EMA(dataset, {n})` usando la función del paso anterior.",
        tip,
    )
    cells.append(code(f"""dataset['EMA{n}'] = EMA(dataset, {n})
print('EMA{n} (último):', round(float(dataset['EMA{n}'].iloc[-1]), 2))
"""))

cells.append(step_md(
    "Paso B4.2a", "Definir función ROC",
    "Rate of change en porcentaje respecto a hace n minutos.",
    "Función `ROC` definida.",
    trading="ROC > 0 → precio subió vs hace n minutos.",
))

cells.append(code("""def ROC(df, n):
    M = df.diff(n - 1)
    N = df.shift(n - 1)
    ROC = pd.Series(((M / N) * 100), name='ROC_' + str(n))
    return ROC
"""))

for n in (10, 30):
    feature_step(
        f"Paso B4.2.{n}",
        f"ROC{n}",
        f"Porcentaje de cambio del Close respecto a hace {n} minutos.",
        f"ROC{n} > 0 ⇒ el precio subió en ese horizonte; ROC{n} < 0 ⇒ bajó.",
    )
    cells.append(code(f"""dataset['ROC{n}'] = ROC(dataset['Close'], {n})
print('ROC{n} (último):', round(float(dataset['ROC{n}'].iloc[-1]), 3))
"""))

cells.append(step_md(
    "Paso B4.3a", "Definir función MOM",
    "Momentum = diferencia de precio en n minutos.",
    "Función `MOM` definida.",
    trading="Similar a ROC pero en unidades de precio, no %.",
))

cells.append(code("""def MOM(df, n):
    MOM = pd.Series(df.diff(n), name='Momentum_' + str(n))
    return MOM
"""))

for n in (10, 30):
    feature_step(
        f"Paso B4.3.{n}",
        f"MOM{n}",
        f"Diferencia de precio Close(t) − Close(t−{n}).",
        f"MOM{n} positivo ⇒ precio subió {n} minutos; negativo ⇒ bajó.",
    )
    cells.append(code(f"""dataset['MOM{n}'] = MOM(dataset['Close'], {n})
print('MOM{n} (último):', round(float(dataset['MOM{n}'].iloc[-1]), 2))
"""))

cells.append(step_md(
    "Paso B4.4a", "Definir función RSI",
    "Implementación RSI del repo (0–100).",
    "Función `RSI` definida.",
    trading="RSI alto → dominan subidas recientes; bajo → dominan caídas.",
))

cells.append(code("""def RSI(series, period):
    delta = series.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[period - 1]] = np.mean(u[:period])
    u = u.drop(u.index[:(period - 1)])
    d[d.index[period - 1]] = np.mean(d[:period])
    d = d.drop(d.index[:(period - 1)])
    rs = u.ewm(com=period - 1, adjust=False).mean() / d.ewm(com=period - 1, adjust=False).mean()
    return 100 - 100 / (1 + rs)
"""))

for n in (10, 30, 200):
    feature_step(
        f"Paso B4.4.{n}",
        f"RSI{n}",
        f"Índice de fuerza relativa con ventana {n} (fórmula del repo).",
        f"RSI{n} cerca de 70+ ⇒ muchas subidas recientes; cerca de 30− ⇒ muchas caídas (lectura clásica).",
    )
    cells.append(code(f"""dataset['RSI{n}'] = RSI(dataset['Close'], {n})
print('RSI{n} (último):', round(float(dataset['RSI{n}'].iloc[-1]), 2))
"""))

cells.append(step_md(
    "Paso B4.5a", "Funciones estocástico %K y %D",
    "Posición del cierre en el rango High–Low reciente.",
    "Funciones `STOK` y `STOD` definidas.",
    trading="%K reacciona rápido; %D es media móvil de %K (más suave).",
))

cells.append(code("""def STOK(close, low, high, n):
    STOK = ((close - low.rolling(n).min()) / (high.rolling(n).max() - low.rolling(n).min())) * 100
    return STOK

def STOD(close, low, high, n):
    STOK = ((close - low.rolling(n).min()) / (high.rolling(n).max() - low.rolling(n).min())) * 100
    STOD = STOK.rolling(3).mean()
    return STOD
"""))

for n in (10, 30, 200):
    cells.append(step_md(
        f"Paso B4.5.{n}K", f"%K{n}",
        f"Estocástico %K con ventana {n}.",
        f"Último %K{n} (0–100).",
    ))
    cells.append(code(f"""dataset['%K{n}'] = STOK(dataset['Close'], dataset['Low'], dataset['High'], {n})
print('%K{n} (último):', round(float(dataset['%K{n}'].iloc[-1]), 2))
"""))
    cells.append(step_md(
        f"Paso B4.5.{n}D", f"%D{n}",
        f"Suavizado %D (media de %K, ventana 3).",
        f"Último %D{n}.",
    ))
    cells.append(code(f"""dataset['%D{n}'] = STOD(dataset['Close'], dataset['Low'], dataset['High'], {n})
print('%D{n} (último):', round(float(dataset['%D{n}'].iloc[-1]), 2))
"""))

cells.append(step_md(
    "Paso B4.6a", "Definir función MA",
    "Media móvil simple con `min_periods=n`.",
    "Función `MA` definida.",
))

cells.append(code("""def MA(df, n):
    MA = pd.Series(df['Close'].rolling(n, min_periods=n).mean(), name='MA_' + str(n))
    return MA
"""))

for win, col, tip in [
    (10, "MA21", "Nombre MA21 en repo; ventana 10 min."),
    (30, "MA63", "Nombre MA63; ventana 30 min."),
    (200, "MA252", "Nombre MA252; ventana 200 min."),
]:
    cells.append(step_md(
        f"Paso B4.6.{col}", f"Columna {col}",
        f"MA simple con ventana {win} minutos.",
        f"Último valor de {col}.",
        trading=tip,
    ))
    cells.append(code(f"""dataset['{col}'] = MA(dataset, {win})
print('{col} (último):', round(float(dataset['{col}'].iloc[-1]), 2))
"""))

cells.append(step_md(
    "Paso B4.7", "Revisión del dataset enriquecido",
    "Muestra las últimas filas con precio, señal e indicadores.",
    "Muchas columnas; verifica que no haya NaN al final.",
    ml="Tras `dropna` perderemos filas iniciales con ventanas incompletas.",
))

cells.append(code("""print('Número de columnas:', len(dataset.columns))
dataset.tail(3)
"""))

cells.append(md("""<a id='3.5'></a>
## 4.5 Visualización de datos

Exploramos relaciones lineales entre indicadores (como en el caso Bitcoin del repo).
"""))

cells.append(step_md(
    "Paso B5.1", "Eliminar columnas no predictivas",
    "Quita OHLC crudo, volumen fiat y SMA de etiquetado.",
    "Lista de columnas restantes (incluye `signal`).",
    ml="Evita que el modelo copie trivialmente la regla SMA vía `short_mavg`.",
    trading="Mantenemos `Close`, volumen BTC y features derivados.",
))

cells.append(code("""dataset = dataset.drop(['High', 'Low', 'Open', 'Volume_(Currency)', 'short_mavg', 'long_mavg'], axis=1)
print('Columnas tras drop:', list(dataset.columns))
"""))

cells.append(step_md(
    "Paso B5.2", "Eliminar filas con NaN",
    "Borra filas donde algún indicador aún no está definido.",
    "Cuántas filas se eliminaron vs antes.",
    ml="Primeras filas tras ventanas largas (p. ej. 200 min) suelen ser NaN.",
))

cells.append(code("""filas_antes = len(dataset)
dataset = dataset.dropna(axis=0)
print('Filas eliminadas por NaN:', filas_antes - len(dataset))
print('Filas listas para modelado:', len(dataset))
"""))

cells.append(step_md(
    "Paso B5.3a", "Matriz de correlación (números)",
    "Calcula correlación lineal entre todas las columnas numéricas.",
    "Objeto `correlation` (usa la celda siguiente para graficar).",
    ml="Correlaciones |r|>0.9 sugieren redundancia entre features.",
))

cells.append(code("""correlation = dataset.corr()
print('Tamaño matriz:', correlation.shape)
"""))

cells.append(step_md(
    "Paso B5.3b", "Heatmap de correlación",
    "Visualiza la matriz (como el caso Bitcoin del repo).",
    "Bloques rojos/azules = variables que se mueven juntas.",
    trading="EMA/MA/EMA200 a menudo correlacionan fuerte con Close.",
    ml="LDA/LR sufren con colinealidad; RF/GBM la toleran mejor.",
))

cells.append(code("""figsize = (10, 10) if MODO_LIGERO else (15, 15)
plt.figure(figsize=figsize)
plt.title('Matriz de correlación')
sns.heatmap(correlation, vmax=1, square=True, annot=False, cmap='cubehelix')
plt.show()
"""))

cells.append(md("""<a id='3.6'></a>
## 4.6 Selección de variables (plantilla maestra)

Aquí **mantenemos** precio, volumen e indicadores calculados, y excluimos columnas crudas ya reemplazadas por features. La variable objetivo es `signal`.
"""))

cells.append(md("""<a id='3.7'></a>
## 4.7 Transformación — estandarización

**StandardScaler** (media 0, varianza 1) ayuda a modelos sensibles a escala (KNN, redes). Aquí se usa en el **Grid Search** del Random Forest; el **modelo final** se entrena sobre `X_train` **sin escalar**.
"""))

cells.append(md("""**✓ Fin del bloque B — checklist explícito**

- [ ] Existe columna **`signal`** (0/1) y sé qué significa en trading.
- [ ] Calculé indicadores; **`short_mavg` / `long_mavg` ya NO están** en X (las eliminamos).
- [ ] Sé cuántas filas quedan tras `dropna`.

Si algún ítem no lo tienes claro, **vuelve al paso B3 o B5** antes de la parte C.
"""))

cells.append(block_intro(
    "C",
    "Separamos **Y** y **X**, partimos train/validation y **comparamos algoritmos** con validación cruzada (sin mirar validation todavía para elegir).",
    "Qué es train vs validation, qué mide CV, y cuál modelo candidato llevar al Grid Search (casi siempre RF).",
    "¿Por qué no entrenamos y evaluamos en las mismas filas si queremos saber si funciona en el futuro?",
))
cells.append(md("""<a id='4'></a>
# 5. Evaluar algoritmos y modelos

### Vista unificada: del precio Bitcoin a la predicción

```
Precio minuto a minuto → indicadores X_t → modelo → ŷ_t (0/1)
                                              ↓
                         comparar con signal_t (SMA) en validación
                                              ↓
                         backtest: retorno × ŷ_{t-1}
```

<a id='4.1'></a>
## 5.1 Partición entrenamiento / validación

### Paso a paso (ML)

1. Tomar las **últimas `N_FILAS`** (mercado reciente).
2. Separar **Y** = `signal`, **X** = resto de columnas numéricas.
3. **80 % train** → ajustar modelos; **20 % validation** → simular “futuro” no visto en el ajuste.
4. `stratify=Y` mantiene proporción de 0/1 en ambos conjuntos.

Usamos `random_state=1` para reproducibilidad.
"""))

cells.append(step_md(
    "Paso C1.1", "Recortar ventana temporal",
    "Toma las últimas `N_FILAS` filas del dataset limpio.",
    "Valor de `n_use` impreso.",
    trading="Enfoca el modelo en el tramo más reciente de la muestra.",
))

cells.append(code("""n_use = min(N_FILAS, len(dataset))
subset_dataset = dataset.iloc[-n_use:]
print('Filas para modelado:', n_use)
"""))

cells.append(step_md(
    "Paso C1.2", "Variable objetivo Y",
    "Asigna `Y = subset_dataset['signal']` — solo la columna etiqueta.",
    "Imprime shape de Y y **P(Y=1)** (proporción de minutos con signal=1).",
    trading="P(Y=1) alto ⇒ en esta ventana la regla SMA estuvo ‘larga’ la mayor parte del tiempo.",
    ml="**Y es lo que el modelo debe predecir.** No incluyas Y dentro de X.",
    objetivo="Separar claramente etiqueta vs predictores.",
    en_palabras="Y es la respuesta correcta del examen; X son las pistas.",
    anota="P(Y=1) en mi muestra = _____ (escribe el número impreso).",
))

cells.append(code("""Y = subset_dataset['signal']
print('Y shape:', Y.shape, '| P(Y=1):', round(float(Y.mean()), 3))
"""))

cells.append(step_md(
    "Paso C1.3", "Matriz de features X",
    "Todas las columnas excepto `signal`.",
    "Número de features y filas en `X`.",
    ml="Cada fila es un minuto; cada columna un indicador o precio/volumen.",
))

cells.append(code("""X = subset_dataset.loc[:, subset_dataset.columns != 'signal']
print('X shape:', X.shape)
print('Features:', list(X.columns))
"""))

cells.append(step_md(
    "Paso C1.4", "Partición train / validation",
    "`train_test_split(X, Y, test_size=0.2, random_state=1)` — **80% train, 20% validation**.",
    "Imprime shapes; train debe tener ~80% de las filas de X.",
    trading="Train = minutos donde ‘estudiamos’; validation = minutos donde ‘probamos’ sin haber estudiado esas filas al entrenar.",
    ml="**Nunca ajustes el modelo final mirando solo train**; validation es el primer hold-out honesto (aunque el split no es temporal estricto).",
    objetivo="Tener cuatro objetos: X_train, X_validation, Y_train, Y_validation.",
    en_palabras="Separamos el examen en parte para practicar (train) y parte sorpresa (validation).",
    anota="Train tiene _____ filas; validation tiene _____ filas (escribe los números).",
))

cells.append(code("""validation_size = 0.2
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, Y, test_size=validation_size, random_state=1
)
print('Train:', X_train.shape, 'Validation:', X_validation.shape)
"""))

cells.append(post_md(
    "Comprueba que la proporción de Y=1 sea similar en train y validation.",
    ml="No es split temporal estricto; es un hold-out aleatorio 80/20.",
))

cells.append(md("""<a id='4.2'></a>
## 5.2 Opciones de prueba y métricas de evaluación
"""))

cells.append(step_md(
    "Paso C2.1", "Número de folds",
    "Copia `N_FOLDS` de la configuración de sesión.",
    "`num_folds` (3 en clase, 10 en modo completo).",
))

cells.append(code("""num_folds = N_FOLDS
print('Folds CV:', num_folds)
"""))

cells.append(step_md(
    "Paso C2.2", "Semilla de CV",
    "Fija aleatoriedad del KFold para resultados reproducibles.",
    "`seed` = CV_SEED.",
))

cells.append(code("""seed = CV_SEED
print('Seed CV:', seed)
"""))

cells.append(step_md(
    "Paso C2.3", "Métrica de scoring",
    "Define qué optimiza CV y Grid Search.",
    "Variable `scoring` (por defecto `accuracy`).",
    trading="Prueba `'precision'` o `'recall'` en Actividad C1 (long-only).",
))

cells.append(code("""scoring = 'accuracy'
# En clase: descomenta UNA línea y comenta accuracy
# scoring = 'precision'
# scoring = 'recall'
print('Scoring activo:', scoring)
"""))

cells.append(md("""**Métricas de clasificación:**

- **Accuracy:** $\\frac{TP+TN}{N}$ — útil si clases balanceadas.
- **Precision (clase 1):** $\\frac{TP}{TP+FP}$ — penaliza falsas compras.
- **Recall (clase 1):** $\\frac{TP}{TP+FN}$ — penaliza perder subidas.
- **AUC-ROC:** calidad del ranking de probabilidades.

Para estrategias **long-only**, a menudo se discute precision vs recall (ver conclusión del notebook original).

### Paso a paso — validación cruzada (qué hace el código después)

1. Partir **train** en `N_FOLDS` bloques temporales aleatorios (KFold).
2. En cada fold: entrenar en k−1 bloques, medir accuracy en el bloque restante.
3. Promediar → estimación de error **sin usar validation**.
4. Repetir para cada algoritmo → **boxplot** al final de la parte C.
"""))

cells.append(md("""<a id='4.3'></a>
## 5.3 Guía de modelos e hiperparámetros

### Paso a paso — algoritmos en **modo clase** (qué aprende cada uno)

| Modelo | Entrada | Salida | En Bitcoin, intuitivamente… |
|--------|---------|--------|-----------------------------|
| **LR** | Vector $X_t$ | $P(\\text{compra})$ | Combinación lineal de RSI, ROC… → probabilidad de régimen alcista |
| **LDA** | $X_t$ | Clase 0/1 | Frontera lineal; baseline rápido |
| **CART** | $X_t$ | 0/1 | Reglas del tipo “si RSI < 30 y ROC > 0 → compra” |
| **GBM** | $X_t$ | 0/1 | Corrige errores de árboles pequeños secuencialmente |
| **RF** | $X_t$ | 0/1 | Muchos árboles en submuestras → robustez al ruido minuto a minuto |

Con **`MODO_LIGERO = False`** se añaden KNN, NB, NN y AdaBoost al benchmark.

| Código | Modelo | Idea | Hiperparámetro clave | Si lo subes demasiado… |
|--------|--------|-------------------|----------------------|-------------------------|
| LR | Regresión logística | $P(y=1|x)=\\sigma(w^\\top x+b)$ | `C` (regularización) | Sobreajuste, coeficientes explosivos |
| LDA | Discriminante lineal | Frontera lineal, covarianza compartida | — | Modelo rígido |
| KNN | k-vecinos | Voto local | `n_neighbors` | Frontera ruidosa si k es pequeño |
| CART | Árbol | Particiones por impureza (Gini/entropía) | `max_depth` | Memoriza ruido minuto a minuto |
| NB | Naive Bayes | Independencia condicional | — | Sesgo si features correlacionadas |
| NN | MLP | Capas ocultas no lineales | `hidden_layer_sizes`, `alpha` | Más capacidad → más riesgo de sobreajuste |
| AB / GBM | Boosting | Suma de clasificadores débiles | `n_estimators`, `learning_rate` | Mejora CV hasta plateau |
| RF | Random Forest | Bagging de árboles | `n_estimators`, `max_depth`, `criterion` | Más árboles → menor varianza |

**Demostración visual** (dos features para poder graficar): efecto de `max_depth` (árbol) y `C` (logística).
"""))

cells.append(md("""> **Pausa docente (2 min):** enseñar un gráfico train vs CV antes de seguir al benchmark.
"""))

cells.append(step_md(
    "Paso C3.0", "Submuestra 2D para gráficos",
    "Solo RSI10 y ROC10 + signal (visualización de hiperparámetros).",
    "Tamaño de `Xd_tr`.",
    ml="Reducir a 2 features permite interpretar curvas de validación.",
))

cells.append(code("""from sklearn.pipeline import Pipeline

demo_cols = ['RSI10', 'ROC10']
demo = subset_dataset[demo_cols + ['signal']].dropna()
Xd = demo[demo_cols]
yd = demo['signal']
Xd_tr, _, yd_tr, _ = train_test_split(Xd, yd, test_size=0.3, random_state=1, stratify=yd)
print('Demo 2D — filas train:', len(Xd_tr))
"""))

cells.append(step_md(
    "Paso C3.1", "Curva de validación — árbol (max_depth)",
    "Entrena árboles con distintas profundidades; compara train vs CV.",
    "Gráfico: si train sube y CV baja → sobreajuste.",
    trading="Profundidad alta = reglas muy específicas al ruido del minuto.",
))

cells.append(code("""depths = [2, 4, 6, 8, 10, 15] if MODO_LIGERO else [2, 3, 4, 5, 6, 8, 10, 15, 20]
pipe = Pipeline([('sc', StandardScaler()), ('clf', DecisionTreeClassifier(random_state=1))])
tr, va = validation_curve(pipe, Xd_tr, yd_tr, param_name='clf__max_depth', param_range=depths, cv=3, scoring='accuracy')
plt.figure(figsize=(8, 4))
plt.plot(depths, tr.mean(1), 'o-', label='Train')
plt.plot(depths, va.mean(1), 'o-', label='CV')
plt.xlabel('max_depth'); plt.ylabel('Accuracy'); plt.title('Árbol: profundidad vs desempeño')
plt.legend(); plt.show()
"""))

cells.append(step_md(
    "Paso C3.2", "Curva de validación — logística (C)",
    "Varía regularización `C` en escala logarítmica.",
    "Gráfico semilogx: busca C donde CV sea alta y train no dispare.",
    ml="C grande → modelo más flexible; C pequeño → más regularización.",
))

cells.append(code("""Cs = np.logspace(-3, 2, 8)
pipe_lr = Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression(max_iter=3000, random_state=1))])
tr, va = validation_curve(pipe_lr, Xd_tr, yd_tr, param_name='clf__C', param_range=Cs, cv=3, scoring='accuracy')
plt.figure(figsize=(8, 4))
plt.semilogx(Cs, tr.mean(1), 'o-', label='Train')
plt.semilogx(Cs, va.mean(1), 'o-', label='CV')
plt.xlabel('C'); plt.title('Logística: regularización'); plt.legend(); plt.show()
"""))

cells.append(md("""**Cómo leer los gráficos:** si la curva de *train* sube y la de *CV* cae, estás **sobreajustando** (modelo memoriza). Elige hiperparámetros donde CV esté alta y la brecha train–CV sea moderada.
"""))

cells.append(md("""<a id='4.4'></a>
## 5.4 Comparar modelos

### Paso a paso — lectura del benchmark

1. Ejecutar la celda que define la **lista `models`**.
2. Ejecutar el **bucle CV**: imprime `media (desv. estándar)` por modelo.
3. Ejecutar el **boxplot**: cada caja = distribución de accuracy en folds.
4. **Elegir candidato** para Grid Search (suele ser Random Forest).

- **Modo clase:** 5 modelos (~8–12 min Colab).
- **Modo completo:** 9 modelos.
"""))

cells.append(step_md(
    "Paso C4.1", "Inicializar benchmark CV",
    "Crea listas vacías y un KFold compartido para todos los modelos.",
    "Mensaje con número de folds.",
))

cells.append(code("""results = []
names = []
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
print('Benchmark CV — folds:', num_folds, '| scoring:', scoring)
"""))

MODELS_LIGERO = [
    ("LR", "LogisticRegression(max_iter=3000)", "Frontera lineal; baseline.", "P(compra) ~ w·indicadores."),
    ("LDA", "LinearDiscriminantAnalysis()", "Frontera lineal LDA.", "Baseline rápido en alta dimensión."),
    ("CART", "DecisionTreeClassifier(max_depth=12, random_state=1)", "Árbol prof. limitada 12.", "Reglas locales minuto a minuto."),
    ("GBM", "GradientBoostingClassifier(n_estimators=50, random_state=1)", "50 árboles débiles secuenciales.", "Corrige errores de modelos previos."),
    ("RF", "RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=1)", "50 árboles en paralelo.", "Robusto al ruido; suele ganar el benchmark."),
]

cells.append(step_md(
    "Paso C4.2", "Lista de modelos (modo sesión)",
    "Muestra qué algoritmos se evaluarán según `MODO_LIGERO`.",
    "Lista impresa de códigos LR, LDA, …",
))

cells.append(code("""if MODO_LIGERO:
    bench_list = [
        ('LR', LogisticRegression(max_iter=3000)),
        ('LDA', LinearDiscriminantAnalysis()),
        ('CART', DecisionTreeClassifier(max_depth=12, random_state=1)),
        ('GBM', GradientBoostingClassifier(n_estimators=50, random_state=1)),
        ('RF', RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=1)),
    ]
else:
    bench_list = [
        ('LR', LogisticRegression(max_iter=3000)),
        ('LDA', LinearDiscriminantAnalysis()),
        ('KNN', KNeighborsClassifier()),
        ('CART', DecisionTreeClassifier()),
        ('NB', GaussianNB()),
        ('NN', MLPClassifier(max_iter=400)),
        ('AB', AdaBoostClassifier()),
        ('GBM', GradientBoostingClassifier()),
        ('RF', RandomForestClassifier(n_jobs=-1)),
    ]
print('Modelos:', [b[0] for b in bench_list])
"""))

for tag, _expr, ml_tip, tr_tip in MODELS_LIGERO:
    cells.append(step_md(
        f"Paso C4.3-{tag}", f"CV — {tag}",
        f"Un fold a la vez: entrena **{tag}** en k−1 particiones y mide la métrica `scoring` en la restante.",
        f"Línea `{tag}: media (std)` — anota la media.",
        trading=tr_tip,
        ml=ml_tip,
    ))
    cells.append(code(f"""_found = False
for _code, _model in bench_list:
    if _code == '{tag}':
        cv_results = cross_val_score(_model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append('{tag}')
        print('{tag}: %f (%f)' % (cv_results.mean(), cv_results.std()))
        _found = True
        break
if not _found:
    print('{tag}: no incluido en bench_list (omitido).')
"""))

for tag, ml_tip, tr_tip in [
    ("KNN", "Clasificador por distancia en el espacio de features.", "Vecinos cercanos en RSI/EMA/etc."),
    ("NB", "Generativo con supuesto de independencia.", "Baseline probabilístico rápido."),
    ("NN", "MLP con capas ocultas.", "Captura no linealidades."),
    ("AB", "AdaBoost sobre clasificadores débiles.", "Secuencia de modelos correctoras."),
]:
    cells.append(step_md(
        f"Paso C4.3-{tag}", f"CV — {tag} (solo modo completo)",
        f"Evalúa **{tag}** cuando `MODO_LIGERO = False`.",
        "En modo ligero verás mensaje de omitido.",
        ml=ml_tip,
        trading=tr_tip,
    ))
    cells.append(code(f"""if MODO_LIGERO:
    print('{tag}: omitido en MODO_LIGERO')
else:
    for _code, _model in bench_list:
        if _code == '{tag}':
            cv_results = cross_val_score(_model, X_train, Y_train, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append('{tag}')
            print('{tag}: %f (%f)' % (cv_results.mean(), cv_results.std()))
            break
"""))

cells.append(step_md(
    "Paso C4.4", "Boxplot del benchmark",
    "Compara distribución de scores CV entre algoritmos.",
    "Cajas estrechas = desempeño estable entre folds.",
    ml="Elige el ganador (suele ser RF/GBM) para el Grid Search.",
))

cells.append(code("""fig = plt.figure()
fig.suptitle('Comparación de algoritmos')
ax = fig.add_subplot(111)
plt.boxplot(results, labels=names)
ax.set_ylabel(scoring)
fig.set_size_inches(15, 8)
plt.xticks(rotation=45)
plt.show()
"""))

cells.append(md("""**Interpretación (10 min discusión):** ¿Cuál gana en **media CV**? ¿La caja es estrecha (estable)? En clase, el ganador suele ser **RF o GBM**; anoten el nombre para el grid.

> **Actividad C1 (5 min):** cambien `scoring` a `'recall'` y vuelvan a correr **solo** la celda del benchmark. ¿Cambia el ranking? ¿Por qué importa para una estrategia long?
"""))

cells.append(block_intro(
    "D",
    "Afinamos **Random Forest** (Grid Search), entrenamos el modelo final, lo evaluamos en **validation** y vemos **qué indicadores pesan más**.",
    "Los mejores hiperparámetros, accuracy en validation, y qué significa cada celda de la matriz de confusión en dinero simulado.",
    "¿El Grid optimiza ganancias o solo accuracy? (respuesta: solo la métrica `scoring`, por defecto accuracy).",
))
cells.append(md("""<a id='5'></a>
# 6. Ajuste fino y Grid Search

### Paso a paso — Grid Search (modelos)

1. **Elegir familia:** Random Forest (suele ganar el benchmark).
2. **Definir malla:** `n_estimators`, `max_depth`, `criterion` (Gini vs entropía).
3. **Escalar X_train** con `StandardScaler` (solo para el Grid Search).
4. **GridSearchCV** prueba cada combinación × cada fold → guarda `best_params_`.
5. **Interpretar heatmap:** celdas similares = zona robusta; pico aislado = cuidado.

### Paso a paso — hiperparámetros RF en lenguaje Bitcoin

- **`n_estimators`:** más árboles → menos varianza, más tiempo de cómputo.
- **`max_depth`:** árboles más profundos → captan micro-patrones del minuto (riesgo de sobreajuste).
- **`criterion`:** forma de medir “pureza” de cada split (Gini vs entropía); impacto suele ser menor que la profundidad.
"""))

cells.append(step_md(
    "Paso D1", "Ajustar StandardScaler",
    "Calcula media/desvío de cada feature en train y transforma X_train.",
    "Media ~0 en la primera feature tras escalar.",
    ml="Solo para Grid Search RF; el modelo final no usa escala.",
))

cells.append(code("""scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
print('Media tras escalar (1ª feature, muestra):', rescaledX[:, 0].mean().round(4))
"""))

cells.append(step_md(
    "Paso D2a", "Hiperparámetro n_estimators",
    "Lista de candidatos de número de árboles (según modo clase).",
    "Vector `n_estimators` impreso.",
    trading="Más árboles → señal más estable, más tiempo de cómputo.",
))

cells.append(code("""if MODO_LIGERO:
    n_estimators = [20, 50]
else:
    n_estimators = [20, 80]
print('n_estimators:', n_estimators)
"""))

cells.append(step_md(
    "Paso D2b", "Hiperparámetros max_depth y criterion",
    "Profundidad máxima de cada árbol y criterio de impureza (Gini/entropía).",
    "Listas `max_depth` y `criterion`.",
    ml="max_depth bajo → menos sobreajuste al ruido minuto a minuto.",
))

cells.append(code("""max_depth = [5, 10]
criterion = ['gini'] if MODO_LIGERO else ['gini', 'entropy']
print('max_depth:', max_depth, '| criterion:', criterion)
"""))

cells.append(step_md(
    "Paso D2c", "Armar param_grid",
    "Diccionario con todas las combinaciones a probar.",
    "Número total de combinaciones.",
))

cells.append(code("""param_grid = dict(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion)
print('Combinaciones a probar:', len(n_estimators) * len(max_depth) * len(criterion))
"""))

cells.append(step_md(
    "Paso D3", "GridSearchCV — entrenamiento",
    "Prueba cada combinación con CV sobre train escalado.",
    "`best_score_` y `best_params_` impresos.",
    ml="Elige hiperparámetros maximizando scoring en CV, no en validation.",
))

cells.append(code("""model = RandomForestClassifier(n_jobs=-1, random_state=1)
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold, n_jobs=-1)
grid_result = grid.fit(rescaledX, Y_train)
print('Best CV score:', grid_result.best_score_.round(4))
print('Best params:', grid_result.best_params_)
"""))

cells.append(step_md(
    "Paso D4", "Ranking de combinaciones",
    "Lista cada combinación con rank, media CV y desviación.",
    "Rank #1 coincide con `best_params_`.",
))

cells.append(code("""means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
ranks = grid_result.cv_results_['rank_test_score']
for mean, stdev, param, rank in zip(means, stds, params, ranks):
    print('#%d %f (%f) with: %r' % (rank, mean, stdev, param))
"""))

cells.append(step_md(
    "Paso D4b", "Preparar tabla para heatmap",
    "Convierte resultados del grid en DataFrame.",
    "Filas = una fila por combinación probada.",
))

cells.append(code("""cv_df = pd.DataFrame(grid_result.cv_results_)
print('Filas en cv_df:', len(cv_df))
"""))

cells.append(step_md(
    "Paso D4c", "Heatmap Grid RF",
    "Mapa de calor: max_depth vs n_estimators (por cada criterion).",
    "Celdas similares = zona robusta de hiperparámetros.",
    trading="Optimizamos accuracy CV, no PnL — el heatmap no garantiza ganancias.",
))

cells.append(code("""for crit in criterion:
    sub = cv_df[cv_df['param_criterion'] == crit]
    pivot = sub.pivot(index='param_max_depth', columns='param_n_estimators', values='mean_test_score')
    plt.figure(figsize=(5, 4))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis')
    plt.title('Grid RF — criterion=%s' % crit)
    plt.show()
"""))

cells.append(md("""<a id='6'></a>
# 7. Finalizar el modelo
"""))

cells.append(md("""<a id='6.1'></a>
## 7.1 Resultados en el conjunto de validación

### Paso a paso — evaluación final (ML + trading)

1. **Reentrenar** RF con `best_params_` sobre todo **X_train** (sin escalar).
2. **Predecir** en **X_validation** → vector `predictions`.
3. **Accuracy / reporte:** calidad global de clasificación.
4. **Matriz de confusión:** traducir TP/FP a “compras acertadas” vs “falsas alarmas”.
5. **Importancia de variables:** qué indicadores usa el bosque para imitar/mejorar la SMA.

Parámetros ganadores del grid; entrenamiento sobre **X_train sin escalar**.
"""))

cells.append(step_md(
    "Paso D5", "Modelo final en train",
    "Instancia RandomForest con `best_params_` y ajusta en X_train **sin escalar**.",
    "Mensaje `Entrenado con:` y dict de parámetros.",
    ml="Escala solo en grid; no en el fit final.",
))

cells.append(code("""# Mejores hiperparámetros del Grid Search
bp = grid_result.best_params_
model = RandomForestClassifier(
    criterion=bp.get('criterion', 'gini'),
    n_estimators=bp.get('n_estimators', 80),
    max_depth=bp.get('max_depth', 10),
    n_jobs=-1,
    random_state=1,
)
model.fit(X_train, Y_train)
print('Entrenado con:', bp)
"""))

cells.append(step_md(
    "Paso D6", "Predicciones ŷ en validation",
    "Aplica `predict` en X_validation.",
    "Accuracy impresa vs Y_validation.",
    trading="Cada 1 predicho ≈ minuto donde el modelo recomienda estar largo.",
))

cells.append(code("""predictions = model.predict(X_validation)
print('Accuracy validation:', accuracy_score(Y_validation, predictions))
"""))

cells.append(step_md(
    "Paso D7a", "Matriz de confusión (números)",
    "Cuenta TP, TN, FP, FN entre real y predicho.",
    "Matriz 2×2 impresa.",
    trading="FP = falsa compra; FN = subida perdida.",
))

cells.append(code("""print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
"""))

cells.append(step_md(
    "Paso D7b", "Heatmap de confusión",
    "Visualiza la misma matriz (más fácil de discutir en clase).",
    "Diagonal fuerte = buen acuerdo con etiqueta SMA.",
))

cells.append(code("""df_cm = pd.DataFrame(
    confusion_matrix(Y_validation, predictions),
    columns=np.unique(Y_validation),
    index=np.unique(Y_validation),
)
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
sns.heatmap(df_cm, cmap='Blues', annot=True, annot_kws={'size': 16})
plt.show()
"""))

cells.append(md("""**Matriz de confusión — traducción explícita (long-only)**

| | **Predicho 0** (no largo) | **Predicho 1** (largo) |
|---|---------------------------|-------------------------|
| **Real 0** | TN — acierto, evitamos compra mala | **FP — falsa compra** (el modelo compra cuando SMA decía no) |
| **Real 1** | **FN — perdimos subida** | TP — acierto, coincidimos con SMA alcista |

- **Accuracy alta** puede ocultar muchos **FP** (compras malas).
- **Precision** de clase 1 ≈ “cuando digo compra, ¿cuántas veces acierto?”.
- **Recall** de clase 1 ≈ “de todos los minutos buenos (SMA=1), ¿cuántos capturo?”.
"""))

cells.append(md("""<a id='6.2'></a>
## 7.2 Intuición de variables / importancia de features
"""))

cells.append(step_md(
    "Paso D8a", "Importancias del bosque",
    "Extrae `feature_importances_` del RF (suma 100% en el gráfico).",
    "Series ordenada por importancia.",
    ml="Impurity decrease — no es causalidad ni estabilidad temporal.",
))

cells.append(code("""Importance = pd.DataFrame({'Importance': model.feature_importances_ * 100}, index=X.columns)
Importance.sort_values('Importance', ascending=False).head(8)
"""))

cells.append(step_md(
    "Paso D8b", "Gráfico de barras",
    "Visualiza qué indicadores usa más el modelo.",
    "Barras largas = splits más frecuentes en el bosque.",
    trading="Si dominan RSI/ROC, el ML imita lógica momentum/tendencia.",
))

cells.append(code("""Importance.sort_values('Importance', axis=0, ascending=True).plot(kind='barh', color='steelblue', figsize=(8, 6))
plt.xlabel('Importancia de variables (%)')
plt.tight_layout()
plt.show()
"""))

cells.append(md("""**Interpretación:** indicadores de **momentum** (RSI, ROC, estocástico) suelen aparecer arriba si la etiqueta SMA es tendencial. No implica causalidad ni estabilidad temporal: reentrena y compara importancias en distintas ventanas (walk-forward en ventanas distintas).
"""))

cells.append(md("""<a id='6.3'></a>
## 7.3 Guardar modelo para uso posterior
"""))

cells.append(step_md(
    "Paso D9", "Persistir modelo",
    "Guarda el RF entrenado con `pickle`.",
    "Archivo `finalized_model_bitcoin.sav` en el directorio de trabajo.",
    ml="En producción versionaría modelo + fecha + features usados.",
))

cells.append(code("""filename = 'finalized_model_bitcoin.sav'
dump(model, open(filename, 'wb'))
print('Modelo guardado:', filename)
"""))

cells.append(block_intro(
    "E",
    "Convertimos predicciones 0/1 en **retornos minuto a minuto** y comparamos **estrategia ML vs regla SMA** (con un minuto de retraso).",
    "Si la curva acumulada del ML queda arriba o abajo de la SMA, y qué falta (comisiones, split temporal) para confiar en dinero real.",
    "¿Por qué multiplicamos retorno × señal de t−1 y no de t?",
))
cells.append(md("""<a id='7'></a>
# 8. Backtesting

### Paso a paso — simulación (aplicación Bitcoin)

1. **`Market Returns`** = $\\frac{P_t - P_{t-1}}{P_{t-1}}$ (retorno del BTC en ese minuto).
2. **`signal_actual`** = etiqueta SMA (referencia / benchmark).
3. **`signal_pred`** = predicción del Random Forest.
4. **Posición con lag:** usamos `shift(1)` → la señal de $t-1$ multiplica el retorno de $t$ (evita usar la señal del mismo minuto del retorno en esta demo).
5. **`Strategy Returns`** = retorno de mercado × señal predicha (0 o 1) → simula estar largo solo cuando el modelo dice 1.
6. **`Actual Returns`** = mismo esquema con la señal SMA → comparar ML vs regla original.
7. **Gráfico acumulado:** suma simple de retornos (didáctico; en producción usar log-returns, comisiones y slippage).

Retorno × señal con **`.shift(1)`** (decisión en t−1, retorno en t).
"""))

cells.append(step_md(
    "Paso E1a", "Crear marco de backtest",
    "DataFrame indexado como validation.",
    "Tabla vacía con índice correcto.",
))

cells.append(code("""backtestdata = pd.DataFrame(index=X_validation.index)
print('Filas backtest:', len(backtestdata))
"""))

cells.append(step_md(
    "Paso E1b", "Señal predicha (ML)",
    "Columna `signal_pred` = salida del Random Forest.",
    "Valores 0/1 en `signal_pred`.",
    trading="1 = minuto donde el modelo estaría largo.",
))

cells.append(code("""backtestdata['signal_pred'] = predictions
backtestdata['signal_pred'].value_counts()
"""))

cells.append(step_md(
    "Paso E1c", "Señal real (SMA)",
    "Columna `signal_actual` = etiqueta SMA de referencia.",
    "Compara conteos con predicción.",
))

cells.append(code("""backtestdata['signal_actual'] = Y_validation
backtestdata[['signal_pred', 'signal_actual']].head(3)
"""))

cells.append(step_md(
    "Paso E2", "Retorno de mercado",
    "`pct_change()` del Close en validation.",
    "Primer valor NaN (no hay t−1).",
    trading="Retorno buy-and-hold minuto a minuto sin filtro.",
))

cells.append(code("""backtestdata['Market Returns'] = X_validation['Close'].pct_change()
backtestdata[['Market Returns']].head(3)
"""))

cells.append(step_md(
    "Paso E3", "Retorno estrategia SMA (benchmark)",
    "`Actual Returns = Market Returns * signal_actual.shift(1)`.",
    "Columna `Actual Returns`; primera fila suele ser NaN por el lag.",
    trading="**Interpretación:** solo gano el retorno del minuto t si **en t−1 la SMA decía estar largo (1).** Es la referencia SMA.",
    ml="No usamos ML aquí; es el baseline para comparar.",
    objetivo="Tener la curva de PnL de la regla SMA en el mismo hold-out que el modelo.",
    en_palabras="Si la SMA decía ‘sí’ el minuto anterior, me quedo expuesto al movimiento de este minuto.",
    anota="¿Actual Returns y Market Returns son iguales en algún minuto? ¿Cuándo son cero?",
))

cells.append(code("""backtestdata['Actual Returns'] = backtestdata['Market Returns'] * backtestdata['signal_actual'].shift(1)
backtestdata[['Market Returns', 'signal_actual', 'Actual Returns']].head(5)
"""))

cells.append(step_md(
    "Paso E4", "Retorno estrategia ML",
    "Market return × señal predicha en t−1.",
    "Columna `Strategy Returns`.",
    trading="Compara PnL simple ML vs SMA en el mismo hold-out.",
))

cells.append(code("""backtestdata['Strategy Returns'] = backtestdata['Market Returns'] * backtestdata['signal_pred'].shift(1)
backtestdata = backtestdata.reset_index()
backtestdata[['signal_pred', 'Strategy Returns']].head(5)
"""))

cells.append(step_md(
    "Paso E5a", "Retorno acumulado (números)",
    "Suma simple de retornos minuto a minuto (didáctico).",
    "Totales acumulados impresos para ML vs SMA.",
    trading="No incluye comisiones Bitstamp ni slippage.",
))

cells.append(code("""cum = backtestdata[['Strategy Returns', 'Actual Returns']].cumsum()
print('Acumulado final ML:', round(float(cum['Strategy Returns'].iloc[-1]), 4))
print('Acumulado final SMA:', round(float(cum['Actual Returns'].iloc[-1]), 4))
"""))

cells.append(step_md(
    "Paso E5b", "Histograma de caminos acumulados",
    "Distribución de retornos acumulados (exploración rápida).",
    "Forma del histograma — no es walk-forward.",
))

cells.append(code("""backtestdata[['Strategy Returns', 'Actual Returns']].cumsum().hist()
"""))

cells.append(step_md(
    "Paso E5c", "Serie temporal acumulada",
    "Gráfico principal: ¿ML supera a SMA en validation?",
    "Curva más alta al final = mejor en esta demo simplificada.",
))

cells.append(code("""backtestdata[['Strategy Returns', 'Actual Returns']].cumsum().plot(figsize=(12, 5))
plt.title('Retornos acumulados (suma simple) — estrategia ML vs señal SMA')
plt.legend(['Estrategia ML', 'Señal SMA'])
plt.show()
"""))

cells.append(md("""### Conclusión

1. **Formulación:** convertir el objetivo de inversión en etiquetas y features es el primer paso; aquí la etiqueta sigue una regla SMA interpretable.
2. **Feature engineering:** indicadores de tendencia y momentum aportan información frente a usar solo precio crudo.
3. **Métricas:** accuracy es útil si las clases están balanceadas; para **long-only** revisa **precision** (falsas compras) y **recall** (subidas perdidas).
4. **Backtesting:** simula rentabilidad **antes** de arriesgar capital — sin comisiones, slippage ni tamaño de posición.

**Próximo paso:** CSV más grande en Drive, split temporal, comisiones y validación walk-forward.

> **Cierre:** ¿La estrategia ML supera a la señal SMA en el gráfico? ¿Qué falta para confiar en capital real?
"""))

cells.append(md("""<a id='8'></a>
# 9. Modo con más filas (opcional)

1. Pon **`MODO_LIGERO = False`**, usa un CSV más grande en **`RUTA_CSV`** y ejecuta **Runtime → Run all**.
2. Compara accuracy CV y backtest con el modo ligero.
3. Entrega sugerida: boxplot, `best_params_`, matriz de confusión e interpretación precision vs recall.
"""))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "colab": {"provenance": []},
    },
    "cells": cells,
}

out = Path(__file__).parent / "Bitcoin_Estrategia_Clasificacion_Colab.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Wrote", out, "cells:", len(cells))
