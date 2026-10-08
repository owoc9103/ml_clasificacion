# -*- coding: utf-8 -*-
"""Genera el notebook alineado con fin-ml (Bitcoin + Master Template), en español."""
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


cells = []

cells.append(md("""# Estrategia Bitcoin con clasificación — **sesión 3 horas**

**Referencias:** [BitcoinTradingStrategy.ipynb](https://github.com/tatsath/fin-ml/blob/master/Chapter%206%20-%20Sup.%20Learning%20-%20Classification%20models/CaseStudy3%20-%20Bitcoin%20Trading%20Strategy/BitcoinTradingStrategy.ipynb) · [Classification-MasterTemplate.ipynb](https://github.com/tatsath/fin-ml/blob/master/Chapter%206%20-%20Sup.%20Learning%20-%20Classification%20models/Classification-MasterTemplate.ipynb) · *Machine Learning and Data Science Blueprints for Finance*.

---

## Agenda sugerida (~180 min)

| Bloque | Tiempo | Qué hacemos |
|--------|--------|-------------|
| **A** | 0:00 – 0:25 | Problema, datos, EDA breve |
| **B** | 0:25 – 1:15 | Limpieza, etiqueta SMA, indicadores, correlación |
| **C** | 1:15 – 1:55 | Métricas, hiperparámetros (gráficos), comparar modelos |
| **D** | 1:55 – 2:35 | Grid Search RF, validación, importancia de variables |
| **E** | 2:35 – 3:00 | Backtesting, cierre, tarea opcional (modo libro completo) |

**Antes de empezar:** ejecuta la celda **Configuración de la sesión** (`MODO_CLASE_3H = True` por defecto → menos filas y CV más rápida para terminar en 3 h en Colab).

**Pedagogía:** **una idea por celda** — markdown de interpretación → código corto → siguiente paso. Evita saltar bloques largos sin leer.

**Roles:** en A–B conviene ejecutar **celda a celda**; en C–D puedes usar **Run all** desde la partición train/val si el tiempo apremia.
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
* [9. Modo completo fin-ml (opcional / tarea)](#8)
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

**Señal de referencia (libro / fin-ml):**

- **1 = compra / largo:** SMA₁₀(Close) > SMA₆₀(Close).
- **0 = no largo:** en caso contrario.

El ML **no sustituye** esa idea: aprende una función $f(X_t) \\approx \\text{signal}_t$ usando muchos indicadores a la vez.

## 1.2 Enfoque de modelos (plantilla maestra de clasificación)

| Paso ML | Sección del notebook | Idea |
|---------|----------------------|------|
| 1 | EDA | Entender datos antes de modelar |
| 2 | Preparación | $Y$ = signal, $X$ = indicadores |
| 3 | Train / validation split | Medir generalización |
| 4 | Validación cruzada | Comparar algoritmos sin mirar el test |
| 5 | Grid Search | Ajustar hiperparámetros del ganador |
| 6 | Modelo final + métricas | Reportar en hold-out |
| 7 | Persistencia + backtest | Cerrar el ciclo “investigación → decisión” |

**Datos:** muestra en GitHub; números del PDF con [Kaggle — Bitstamp minutes](https://www.kaggle.com/mlfinancebook/bitstamp-bicoin-minutes-data).
"""))

cells.append(md("""<a id='1'></a>
# 2. Inicio — librerías y datos
<a id='1.1'></a>
## 2.1 Cargar librerías

Mismas familias que el notebook original: `pandas`, `sklearn`, visualización y (opcional) redes vía `MLPClassifier` — equivalente práctico a la red shallow del master template en Colab sin Keras.
"""))

cells.append(md("""**Paso 2.1a — Entorno Colab:** detecta si estamos en Colab e instala dependencias si hace falta.
"""))

cells.append(code("""try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    !pip -q install seaborn scikit-learn

print('Colab:', IN_COLAB)
"""))

cells.append(md("""**Paso 2.1b — Importar librerías:** mismas familias que fin-ml (`pandas`, `sklearn`, gráficos).
"""))

cells.append(code("""import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, validation_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from pickle import dump, load
"""))

cells.append(md("""<a id='cfg'></a>
## Configuración de la sesión (ejecutar primero)

- **`MODO_CLASE_3H = True`:** ~30 000 filas finales, 3 folds, 5 modelos en el benchmark, grid RF reducido → **cabe en ~3 h**.
- **`MODO_CLASE_3H = False`:** réplica del flujo del repo (100 000 filas, 10 folds, 9 modelos) → mejor para **tarea en casa** o sesión extendida.
"""))

cells.append(code("""# --- Ajuste docente ---
MODO_CLASE_3H = True

N_FILAS = 30_000 if MODO_CLASE_3H else 100_000
N_FOLDS = 3 if MODO_CLASE_3H else 10
CV_SEED = 7

print('Modo:', 'CLASE 3h' if MODO_CLASE_3H else 'COMPLETO fin-ml')
print('Filas para modelado:', N_FILAS, '| Folds CV:', N_FOLDS)
"""))

cells.append(md("""---
## ⏱ Bloque A · Problema y datos (~25 min)
<a id='1.2'></a>
### 2.2 Cargar datos

> **Nota del libro:** en GitHub la muestra es pequeña por límite de tamaño; los números del PDF se reproducen con el CSV completo en Kaggle.
"""))

cells.append(code("""from pathlib import Path

CANDIDATES = [
    Path('data/BitstampData_sample.csv'),
    Path('Ejemplo_Clase/data/BitstampData_sample.csv'),
    Path('/content/ml_clasificacion/Ejemplo_Clase/data/BitstampData_sample.csv'),
]
URLS = [
    'https://raw.githubusercontent.com/owoc9103/ml_clasificacion/main/Ejemplo_Clase/data/BitstampData_sample.csv',
    'https://raw.githubusercontent.com/tatsath/fin-ml/master/Chapter%206%20-%20Sup.%20Learning%20-%20Classification%20models/CaseStudy3%20-%20Bitcoin%20Trading%20Strategy/BitstampData_sample.csv',
]

dataset = None
for p in CANDIDATES:
    if p.exists():
        dataset = read_csv(p)
        print('Archivo local:', p)
        break
if dataset is None:
    for url in URLS:
        try:
            dataset = read_csv(url)
            print('Descargado:', url)
            break
        except Exception as exc:
            print('No disponible:', url, exc)

assert dataset is not None, 'Sube data/BitstampData_sample.csv o revisa las URLs'
print('Filas cargadas:', len(dataset))
"""))

cells.append(md("""### Paso a paso — qué es cada columna (Bitcoin)

| Columna | Significado en mercado |
|---------|-------------------------|
| `Open`, `High`, `Low`, `Close` | Precio en ese minuto (vela de 1 min) |
| `Volume_(BTC)` | Cantidad negociada en BTC (liquidez relativa) |
| `Volume_(Currency)` | Volumen en moneda fiat (se elimina más adelante en el repo) |
| `Weighted_Price` | Precio promedio ponderado del minuto |

Cada **fila = un minuto**. La estrategia del libro decide en cada minuto si el régimen es “alcista de corto plazo” (etiqueta) usando medias e indicadores **calculados solo con información hasta ese minuto** (cuidado con *look-ahead* en proyectos propios).
"""))

cells.append(code("""try:
    from IPython.display import display
except ImportError:
    display = print
set_option('display.width', 100)
set_option('precision', 3)
"""))

cells.append(md("""**Paso A3 — Tamaño del archivo:** filas × columnas. En la muestra GitHub hay menos filas que en Kaggle completo.
"""))

cells.append(code("""print('Shape (filas, columnas):', dataset.shape)
"""))

cells.append(md("""**Paso A4 — Últimas observaciones:** revisa que `Close` y volúmenes sean numéricos y que el orden temporal sea coherente (precios recientes al final).
"""))

cells.append(code("""display(dataset.tail(5))
"""))

cells.append(md("""**Paso A5 — Resumen estadístico:** fíjate en `count` (¿faltan valores?) y en min/max de `Close` (rango de precios en la muestra).
"""))

cells.append(code("""display(dataset.describe())
"""))

cells.append(md("""**Interpretación (5 min):** ¿Cuántos NaNs implícitos (`count` < filas)? ¿Rango de `Close`? En minutos hay millones de filas; modelaremos solo las **últimas `N_FILAS`** (ver config).

> **Actividad A1 (3 min):** en chat, respondan: *¿Por qué no entrenar con toda la serie desde 2012 en un laptop/Colab free?* (memoria, tiempo, régimen de mercado distinto).
"""))

cells.append(md("""---
## ⏱ Bloque B · Preparación (~50 min)
<a id='3'></a>
# 4. Preparación de datos
<a id='3.1'></a>
## 4.1 Limpieza de datos
"""))

cells.append(code("""# Checking for any null values
print('Null Values =', dataset.isnull().values.any())
"""))

cells.append(md("""Si hay nulos, el libro propone **forward fill** (último valor observado): coherente en series de precios, pero introduce suavizado en huecos largos.
"""))

cells.append(code("""dataset[dataset.columns.values] = dataset[dataset.columns.values].ffill()
"""))

cells.append(code("""dataset = dataset.drop(columns=['Timestamp'])
"""))

cells.append(md("""<a id='3.2'></a>
## 4.2 Datos categóricos (plantilla maestra)

En el master template (crédito alemán) se codifican variables categóricas. **En Bitcoin todas las entradas son numéricas** (OHLCV); no hay one-hot encoding. Si añadieras calendario (día de la semana), ahí aplicarías la sección 4.2 del template.
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

cells.append(md("""**Paso B3.1 — SMA corta (10 min):** media móvil simple del cierre; reacciona rápido a subidas/bajadas recientes.
"""))

cells.append(code("""dataset['short_mavg'] = dataset['Close'].rolling(window=10, min_periods=1, center=False).mean()
dataset[['Close', 'short_mavg']].tail(3)
"""))

cells.append(md("""**Paso B3.2 — SMA larga (60 min):** filtra ruido; representa el “consenso” de precio en la última hora.
"""))

cells.append(code("""dataset['long_mavg'] = dataset['Close'].rolling(window=60, min_periods=1, center=False).mean()
dataset[['Close', 'short_mavg', 'long_mavg']].tail(3)
"""))

cells.append(md("""**Paso B3.3 — Etiqueta binaria:** si la media corta está **por encima** de la larga, marcamos régimen alcista de corto plazo (`signal = 1`).
"""))

cells.append(code("""dataset['signal'] = np.where(dataset['short_mavg'] > dataset['long_mavg'], 1.0, 0.0)
"""))

cells.append(code("""dataset.tail()
"""))

cells.append(md("""**Interpretación:** si `signal` está muy desbalanceado (p. ej. 90 % unos), accuracy puede engañar: el modelo podría predecir siempre la clase mayoritaria. Anota las proporciones antes de entrenar.
"""))

cells.append(code("""dataset['signal'].value_counts(normalize=True)
"""))

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

A continuación: **una familia de indicadores por celda** (mismas fórmulas que fin-ml). Ejecuta en orden y lee la interpretación antes de seguir.
"""))

cells.append(md("""**Paso B4.1 — EMA (tendencia):** media exponencial del cierre; da más peso a precios recientes que la SMA.
"""))

cells.append(code("""def EMA(df, n):
    EMA = pd.Series(df['Close'].ewm(span=n, min_periods=n).mean(), name='EMA_' + str(n))
    return EMA

dataset['EMA10'] = EMA(dataset, 10)
dataset['EMA30'] = EMA(dataset, 30)
dataset['EMA200'] = EMA(dataset, 200)
dataset[['Close', 'EMA10', 'EMA30']].tail(2)
"""))

cells.append(md("""**Paso B4.2 — ROC (momentum %):** cambio porcentual del precio respecto a hace *n* minutos.
"""))

cells.append(code("""def ROC(df, n):
    M = df.diff(n - 1)
    N = df.shift(n - 1)
    ROC = pd.Series(((M / N) * 100), name='ROC_' + str(n))
    return ROC

dataset['ROC10'] = ROC(dataset['Close'], 10)
dataset['ROC30'] = ROC(dataset['Close'], 30)
dataset[['Close', 'ROC10', 'ROC30']].tail(2)
"""))

cells.append(md("""**Paso B4.3 — MOM (momentum absoluto):** diferencia de precio en *n* minutos (no en %).
"""))

cells.append(code("""def MOM(df, n):
    MOM = pd.Series(df.diff(n), name='Momentum_' + str(n))
    return MOM

dataset['MOM10'] = MOM(dataset['Close'], 10)
dataset['MOM30'] = MOM(dataset['Close'], 30)
dataset[['Close', 'MOM10']].tail(2)
"""))

cells.append(md("""**Paso B4.4 — RSI:** mide fuerza relativa de subidas vs bajadas (0–100). Valores altos → muchas subidas recientes (posible sobrecompra en análisis clásico).
"""))

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

cells.append(code("""dataset['RSI10'] = RSI(dataset['Close'], 10)
dataset['RSI30'] = RSI(dataset['Close'], 30)
dataset['RSI200'] = RSI(dataset['Close'], 200)
dataset[['Close', 'RSI10', 'RSI30']].tail(2)
"""))

cells.append(md("""**Paso B4.5 — Estocástico (%K, %D):** dónde cierra el precio dentro del rango High–Low de los últimos *n* minutos (0–100).
"""))

cells.append(code("""def STOK(close, low, high, n):
    STOK = ((close - low.rolling(n).min()) / (high.rolling(n).max() - low.rolling(n).min())) * 100
    return STOK

def STOD(close, low, high, n):
    STOK = ((close - low.rolling(n).min()) / (high.rolling(n).max() - low.rolling(n).min())) * 100
    STOD = STOK.rolling(3).mean()
    return STOD
"""))

cells.append(code("""dataset['%K10'] = STOK(dataset['Close'], dataset['Low'], dataset['High'], 10)
dataset['%D10'] = STOD(dataset['Close'], dataset['Low'], dataset['High'], 10)
dataset['%K30'] = STOK(dataset['Close'], dataset['Low'], dataset['High'], 30)
dataset['%D30'] = STOD(dataset['Close'], dataset['Low'], dataset['High'], 30)
dataset['%K200'] = STOK(dataset['Close'], dataset['Low'], dataset['High'], 200)
dataset['%D200'] = STOD(dataset['Close'], dataset['Low'], dataset['High'], 200)
dataset[['Close', '%K10', '%D10']].tail(2)
"""))

cells.append(md("""**Paso B4.6 — MA (medias simples):** en el repo los nombres son MA21/63/252 pero las ventanas son 10/30/200 minutos (igual que fin-ml).
"""))

cells.append(code("""def MA(df, n):
    MA = pd.Series(df['Close'].rolling(n, min_periods=n).mean(), name='MA_' + str(n))
    return MA

dataset['MA21'] = MA(dataset, 10)
dataset['MA63'] = MA(dataset, 30)
dataset['MA252'] = MA(dataset, 200)
"""))

cells.append(md("""**Paso B4.7 — Vista final:** tabla con precio, etiqueta e indicadores recientes.
"""))

cells.append(code("""dataset.tail()
"""))

cells.append(md("""<a id='3.5'></a>
## 4.5 Visualización de datos

Exploramos relaciones lineales entre indicadores (como en el caso Bitcoin del repo).
"""))

cells.append(md("""**Paso B5.1 — Quitar columnas crudas:** OHLC y las SMA de etiquetado ya no entran a $X$; evitamos duplicar información y “copiar” la regla SMA.
"""))

cells.append(code("""dataset = dataset.drop(['High', 'Low', 'Open', 'Volume_(Currency)', 'short_mavg', 'long_mavg'], axis=1)
print('Columnas tras drop:', list(dataset.columns))
"""))

cells.append(md("""**Paso B5.2 — Filas incompletas:** los indicadores con ventana larga generan NaN al inicio; las eliminamos (como fin-ml).
"""))

cells.append(code("""filas_antes = len(dataset)
dataset = dataset.dropna(axis=0)
print('Filas eliminadas por NaN:', filas_antes - len(dataset))
dataset.tail(3)
"""))

cells.append(md("""**Paso B5.3 — Correlación entre features:** detecta redundancia (varias medias del mismo precio suelen correlacionar fuerte).
"""))

cells.append(code("""correlation = dataset.corr()
figsize = (10, 10) if MODO_CLASE_3H else (15, 15)
plt.figure(figsize=figsize)
plt.title('Matriz de correlación')
sns.heatmap(correlation, vmax=1, square=True, annot=False, cmap='cubehelix')
plt.show()
"""))

cells.append(md("""**Interpretación:** pares con correlación > 0.9 (p. ej. distintas medias del mismo precio) aportan poca información nueva; en proyectos reales considerarías eliminar redundantes para estabilizar modelos lineales. Los árboles y bosques toleran colinealidad mejor que LDA o regresión logística.
"""))

cells.append(md("""<a id='3.6'></a>
## 4.6 Selección de variables (plantilla maestra)

El master template usa filtros y dominio del negocio. Aquí **mantenemos el set del repo** (precio, volumen, indicadores) y excluimos solo columnas crudas ya reemplazadas por features. La variable objetivo es `signal`.
"""))

cells.append(md("""<a id='3.7'></a>
## 4.7 Transformación — estandarización

En el libro, **StandardScaler** (media 0, varianza 1) ayuda a modelos sensibles a escala (KNN, redes, SVM). En el notebook original se usa al **Grid Search** del Random Forest; el modelo final se entrena sobre `X_train` sin escalar — replicamos ese comportamiento.
"""))

cells.append(md("""---
## ⏱ Bloque C · Modelos (~40 min)
<a id='4'></a>
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

Misma lógica que fin-ml: `random_state=1`.
"""))

cells.append(md("""**Paso C1.1 — Ventana reciente:** nos quedamos con las últimas `N_FILAS` filas (config al inicio).
"""))

cells.append(code("""n_use = min(N_FILAS, len(dataset))
subset_dataset = dataset.iloc[-n_use:]
print('Filas para modelado:', n_use)
subset_dataset[['Close', 'signal']].head(2)
"""))

cells.append(md("""**Paso C1.2 — Separar $Y$ y $X$:** objetivo = `signal`; predictores = resto de columnas numéricas.
"""))

cells.append(code("""Y = subset_dataset['signal']
X = subset_dataset.loc[:, subset_dataset.columns != 'signal']
print('Y (etiqueta):', Y.shape, '| X (features):', X.shape)
"""))

cells.append(md("""**Paso C1.3 — Partición 80/20:** entrenamiento para ajustar; validación como “futuro” hold-out (`random_state=1` como fin-ml).
"""))

cells.append(code("""validation_size = 0.2
seed = 1
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, Y, test_size=validation_size, random_state=1
)
print('Train:', X_train.shape, 'Validation:', X_validation.shape)
print('Proporción Y en train:', Y_train.mean())
"""))

cells.append(md("""**Interpretación:** usar solo el tramo final simula “entrenar con historia reciente”. No es walk-forward estricto (el split es aleatorio estratificado), pero coincide con el código del repositorio para comparar resultados en clase.
"""))

cells.append(md("""<a id='4.2'></a>
## 5.2 Opciones de prueba y métricas de evaluación
"""))

cells.append(code("""num_folds = N_FOLDS
seed = CV_SEED
scoring = 'accuracy'
# En clase: descomenta UNA métrica y comenta accuracy para discutir trading long-only
# scoring = 'precision'
# scoring = 'recall'
"""))

cells.append(md("""**Métricas (libro, cap. 6):**

- **Accuracy:** $\\frac{TP+TN}{N}$ — útil si clases balanceadas.
- **Precision (clase 1):** $\\frac{TP}{TP+FP}$ — penaliza falsas compras.
- **Recall (clase 1):** $\\frac{TP}{TP+FN}$ — penaliza perder subidas.
- **AUC-ROC:** calidad del ranking de probabilidades.

Para estrategias **long-only**, a menudo se discute precision vs recall (ver conclusión del notebook original).

### Paso a paso — validación cruzada (qué hace el código después)

1. Partir **train** en `N_FOLDS` bloques temporales aleatorios (KFold).
2. En cada fold: entrenar en k−1 bloques, medir accuracy en el bloque restante.
3. Promediar → estimación de error **sin usar validation**.
4. Repetir para cada algoritmo → **boxplot** al final del bloque C.
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

En **modo completo** (`MODO_CLASE_3H = False`) se añaden KNN, NB, NN, AdaBoost como en fin-ml.

| Código | Modelo | Idea (Blueprints) | Hiperparámetro clave | Si lo subes demasiado… |
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

cells.append(code("""from sklearn.pipeline import Pipeline

demo_cols = ['RSI10', 'ROC10']
demo = subset_dataset[demo_cols + ['signal']].dropna()
Xd = demo[demo_cols]
yd = demo['signal']
Xd_tr, _, yd_tr, _ = train_test_split(Xd, yd, test_size=0.3, random_state=1, stratify=yd)
print('Demo 2D — filas train:', len(Xd_tr))
"""))

cells.append(md("""**Paso C3.1 — Árbol (CART):** curva train vs CV al variar `max_depth` (solo RSI10 y ROC10 para poder visualizar).
"""))

cells.append(code("""depths = [2, 4, 6, 8, 10, 15] if MODO_CLASE_3H else [2, 3, 4, 5, 6, 8, 10, 15, 20]
pipe = Pipeline([('sc', StandardScaler()), ('clf', DecisionTreeClassifier(random_state=1))])
tr, va = validation_curve(pipe, Xd_tr, yd_tr, param_name='clf__max_depth', param_range=depths, cv=3, scoring='accuracy')
plt.figure(figsize=(8, 4))
plt.plot(depths, tr.mean(1), 'o-', label='Train')
plt.plot(depths, va.mean(1), 'o-', label='CV')
plt.xlabel('max_depth'); plt.ylabel('Accuracy'); plt.title('Árbol: profundidad vs desempeño')
plt.legend(); plt.show()
"""))

cells.append(md("""**Paso C3.2 — Regresión logística:** mismo ejercicio con el hiperparámetro `C` (inverso de la regularización).
"""))

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
4. **Elegir candidato** para Grid Search (en el libro: Random Forest).

- **Modo clase:** 5 modelos (~8–12 min Colab).
- **Modo completo:** 9 modelos (fin-ml).
"""))

cells.append(code("""models = []
if MODO_CLASE_3H:
    models.append(('LR', LogisticRegression(max_iter=3000)))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('CART', DecisionTreeClassifier(max_depth=12, random_state=1)))
    models.append(('GBM', GradientBoostingClassifier(n_estimators=50, random_state=1)))
    models.append(('RF', RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=1)))
else:
    models.append(('LR', LogisticRegression(max_iter=3000)))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('NN', MLPClassifier(max_iter=400)))
    models.append(('AB', AdaBoostClassifier()))
    models.append(('GBM', GradientBoostingClassifier()))
    models.append(('RF', RandomForestClassifier(n_jobs=-1)))
print('Modelos a evaluar:', [m[0] for m in models])
"""))

cells.append(md("""**Paso C4.2 — Validación cruzada en train:** cada modelo se entrena y evalúa `N_FOLDS` veces; imprimimos media y desviación.
"""))

cells.append(code("""results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = '%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())
    print(msg)
"""))

cells.append(md("""**Paso C4.3 — Boxplot:** cada caja muestra la dispersión de accuracy entre folds (estabilidad del modelo).
"""))

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

cells.append(md("""---
## ⏱ Bloque D · Grid y modelo final (~40 min)
<a id='5'></a>
# 6. Ajuste fino y Grid Search

### Paso a paso — Grid Search (modelos)

1. **Elegir familia:** Random Forest (suele ganar en el caso del libro).
2. **Definir malla:** `n_estimators`, `max_depth`, `criterion` (Gini vs entropía).
3. **Escalar X_train** con `StandardScaler` (como fin-ml en esta etapa).
4. **GridSearchCV** prueba cada combinación × cada fold → guarda `best_params_`.
5. **Interpretar heatmap:** celdas similares = zona robusta; pico aislado = cuidado.

### Paso a paso — hiperparámetros RF en lenguaje Bitcoin

- **`n_estimators`:** más árboles → menos varianza, más tiempo de cómputo.
- **`max_depth`:** árboles más profundos → captan micro-patrones del minuto (riesgo de sobreajuste).
- **`criterion`:** forma de medir “pureza” de cada split (Gini vs entropía); impacto suele ser menor que la profundidad.
"""))

cells.append(md("""**Paso D1 — Estandarizar solo para el grid:** `StandardScaler` sobre **X_train** (fin-ml hace lo mismo en esta etapa).
"""))

cells.append(code("""scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
print('Media tras escalar (1ª feature, muestra):', rescaledX[:, 0].mean().round(4))
"""))

cells.append(md("""**Paso D2 — Definir la malla** de hiperparámetros del Random Forest.
"""))

cells.append(code("""if MODO_CLASE_3H:
    n_estimators = [20, 50]
    max_depth = [5, 10]
    criterion = ['gini']
else:
    n_estimators = [20, 80]
    max_depth = [5, 10]
    criterion = ['gini', 'entropy']

param_grid = dict(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion)
print('Combinaciones a probar:', len(n_estimators) * len(max_depth) * len(criterion))
"""))

cells.append(md("""**Paso D3 — Ejecutar GridSearchCV:** cada combinación se evalúa con validación cruzada sobre train.
"""))

cells.append(code("""model = RandomForestClassifier(n_jobs=-1, random_state=1)
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold, n_jobs=-1)
grid_result = grid.fit(rescaledX, Y_train)
print('Best CV score:', grid_result.best_score_.round(4))
print('Best params:', grid_result.best_params_)
"""))

cells.append(md("""**Paso D4 — Tabla de ranks:** todas las combinaciones ordenadas por desempeño CV.
"""))

cells.append(code("""means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
ranks = grid_result.cv_results_['rank_test_score']
for mean, stdev, param, rank in zip(means, stds, params, ranks):
    print('#%d %f (%f) with: %r' % (rank, mean, stdev, param))
"""))

cells.append(code("""# Visual: superficie de CV para cada criterion
cv_df = pd.DataFrame(grid_result.cv_results_)
for crit in criterion:
    sub = cv_df[cv_df['param_criterion'] == crit]
    pivot = sub.pivot(index='param_max_depth', columns='param_n_estimators', values='mean_test_score')
    plt.figure(figsize=(5, 4))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis')
    plt.title('Grid RF — criterion=%s' % crit)
    plt.show()
"""))

cells.append(md("""**Interpretación:** el `best_params_` maximiza **accuracy media en CV**, no ganancias de trading. Un árbol más profundo puede subir accuracy in-sample pero empeorar estabilidad fuera de muestra. Usa el heatmap para ver si el óptimo es “meseta” (robusto) o un pico aislado (frágil).
"""))

cells.append(md("""<a id='6'></a>
# 7. Finalizar el modelo
"""))

cells.append(md("""<a id='6.1'></a>
## 7.1 Resultados en el conjunto de validación

### Paso a paso — evaluación final (ML + trading)

1. **Reentrenar** RF con `best_params_` sobre todo **X_train** (sin escalar → igual que fin-ml).
2. **Predecir** en **X_validation** → vector `predictions`.
3. **Accuracy / reporte:** calidad global de clasificación.
4. **Matriz de confusión:** traducir TP/FP a “compras acertadas” vs “falsas alarmas”.
5. **Importancia de variables:** qué indicadores usa el bosque para imitar/mejorar la SMA.

Parámetros ganadores del grid; entrenamiento sobre **X_train sin escalar** (mismo código que fin-ml).
"""))

cells.append(md("""**Paso D5 — Reentrenar Random Forest** con `best_params_` sobre todo el train (features sin escalar).
"""))

cells.append(code("""# Usar mejores hiperparámetros del grid (fin-ml fija a mano; aquí tomamos el grid)
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

cells.append(md("""**Paso D6 — Predicciones en validación:** el modelo nunca vio estas filas durante el ajuste.
"""))

cells.append(code("""predictions = model.predict(X_validation)
print('Accuracy validation:', accuracy_score(Y_validation, predictions))
"""))

cells.append(md("""**Paso D7 — Matriz y reporte:** desglose por clase (0 = no largo, 1 = largo).
"""))

cells.append(code("""print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
"""))

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

cells.append(md("""**Interpretación matriz de confusión:**

- **Falsos positivos (FP):** modelo dice compra y la regla SMA decía venta → operaciones perdedoras potenciales.
- **Falsos negativos (FN):** pierdes subidas.
- Si operas **long-only**, muchos FP pueden destruir el PnL aunque accuracy sea alta.
"""))

cells.append(md("""<a id='6.2'></a>
## 7.2 Intuición de variables / importancia de features
"""))

cells.append(code("""Importance = pd.DataFrame({'Importance': model.feature_importances_ * 100}, index=X.columns)
Importance.sort_values('Importance', axis=0, ascending=True).plot(kind='barh', color='r')
plt.xlabel('Importancia de variables (%)')
plt.show()
"""))

cells.append(md("""**Interpretación:** indicadores de **momentum** (RSI, ROC, estocástico) suelen aparecer arriba si la etiqueta SMA es tendencial. No implica causalidad ni estabilidad temporal: reentrena y compara importancias en distintas ventanas (walk-forward del libro).
"""))

cells.append(md("""<a id='6.3'></a>
## 7.3 Guardar modelo para uso posterior (master template)
"""))

cells.append(code("""filename = 'finalized_model_bitcoin.sav'
dump(model, open(filename, 'wb'))
print('Modelo guardado:', filename)
"""))

cells.append(md("""---
## ⏱ Bloque E · Backtesting y cierre (~25 min)
<a id='7'></a>
# 8. Backtesting

### Paso a paso — simulación (aplicación Bitcoin)

1. **`Market Returns`** = $\\frac{P_t - P_{t-1}}{P_{t-1}}$ (retorno del BTC en ese minuto).
2. **`signal_actual`** = etiqueta SMA (referencia “benchmark” del libro).
3. **`signal_pred`** = predicción del Random Forest.
4. **Posición con lag:** usamos `shift(1)` → la señal de $t-1$ multiplica el retorno de $t$ (evita usar la señal del mismo minuto del retorno en esta demo).
5. **`Strategy Returns`** = retorno de mercado × señal predicha (0 o 1) → simula estar largo solo cuando el modelo dice 1.
6. **`Actual Returns`** = mismo esquema con la señal SMA → comparar ML vs regla original.
7. **Gráfico acumulado:** suma simple de retornos (didáctico; en producción usar log-returns, comisiones y slippage).

Retorno × señal con **`.shift(1)`** como en fin-ml.
"""))

cells.append(md("""**Paso E1 — Tabla base:** alineamos predicciones y etiqueta SMA en el conjunto de validación.
"""))

cells.append(code("""backtestdata = pd.DataFrame(index=X_validation.index)
backtestdata['signal_pred'] = predictions
backtestdata['signal_actual'] = Y_validation
backtestdata.head(3)
"""))

cells.append(md("""**Paso E2 — Retorno de mercado:** variación minuto a minuto del `Close` (sin posición aún).
"""))

cells.append(code("""backtestdata['Market Returns'] = X_validation['Close'].pct_change()
backtestdata[['Market Returns']].head(3)
"""))

cells.append(md("""**Paso E3 — Retorno con señal SMA (referencia):** posición de *t−1* × retorno en *t* (`shift(1)`).
"""))

cells.append(code("""backtestdata['Actual Returns'] = backtestdata['Market Returns'] * backtestdata['signal_actual'].shift(1)
backtestdata[['Market Returns', 'signal_actual', 'Actual Returns']].head(5)
"""))

cells.append(md("""**Paso E4 — Retorno con señal del ML:** misma regla de lag; comparamos estrategia aprendida vs SMA.
"""))

cells.append(code("""backtestdata['Strategy Returns'] = backtestdata['Market Returns'] * backtestdata['signal_pred'].shift(1)
backtestdata = backtestdata.reset_index()
backtestdata[['signal_pred', 'Strategy Returns']].head(5)
"""))

cells.append(md("""**Paso E5 — Acumulados:** suma simple de retornos (didáctico; en producción: log-returns, fees, slippage).
"""))

cells.append(code("""backtestdata[['Strategy Returns', 'Actual Returns']].cumsum().hist()
"""))

cells.append(code("""backtestdata[['Strategy Returns', 'Actual Returns']].cumsum().plot(figsize=(12, 5))
plt.title('Retornos acumulados (suma simple) — estrategia ML vs señal SMA')
plt.legend(['Estrategia ML', 'Señal SMA'])
plt.show()
"""))

cells.append(md("""### Conclusión (adaptada del libro / repo)

1. **Formulación:** convertir el objetivo de inversión en etiquetas y features es el primer paso; aquí la etiqueta sigue una regla SMA interpretable.
2. **Feature engineering:** indicadores de tendencia y momentum aumentan poder predictivo frente a usar solo precio crudo.
3. **Métricas:** accuracy o AUC son razonables en clasificación; si priorizas **long** con pocos falsos positivos, mira **precision**; si no quieres perder subidas, **recall**.
4. **Backtesting:** permite analizar riesgo/rentabilidad **antes** de arriesgar capital — sin comisiones, slippage ni tamaño de posición (limitaciones didácticas).

**Próximo paso académico:** dataset completo Kaggle, split temporal, costos de transacción y validación walk-forward del capítulo de producción.

> **Cierre (5 min):** ¿La estrategia ML supera a la señal SMA en el gráfico? ¿Qué falta para confiar en capital real?
"""))

cells.append(md("""<a id='8'></a>
# 9. Modo completo fin-ml (opcional — tarea en casa)

1. Arriba, pon **`MODO_CLASE_3H = False`** y vuelve a ejecutar **Runtime → Run all** (≈45–90 min con muestra GitHub; más con Kaggle completo).
2. Compara accuracy CV y backtest con la sesión de 3 h.
3. Entrega sugerida: captura del boxplot, `best_params_`, matriz de confusión y una página de interpretación (precision vs recall).
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
