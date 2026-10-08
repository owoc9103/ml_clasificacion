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

cells.append(md("""# Estrategia de trading con Bitcoin — modelos de clasificación

**Referencia:** [BitcoinTradingStrategy.ipynb](https://github.com/tatsath/fin-ml/blob/master/Chapter%206%20-%20Sup.%20Learning%20-%20Classification%20models/CaseStudy3%20-%20Bitcoin%20Trading%20Strategy/BitcoinTradingStrategy.ipynb) y [Classification-MasterTemplate.ipynb](https://github.com/tatsath/fin-ml/blob/master/Chapter%206%20-%20Sup.%20Learning%20-%20Classification%20models/Classification-MasterTemplate.ipynb) (Tatsat, Puri & Lookabaugh, *Machine Learning and Data Science Blueprints for Finance*).

Este cuaderno sigue **el mismo orden y la misma lógica de código** que el repositorio oficial; las celdas en español explican *qué estás viendo* y *por qué importa* en finanzas y en ML.
"""))

cells.append(md("""## Contenido

* [1. Definición del problema](#0)
* [2. Inicio — librerías y datos](#1)
    * [2.1 Cargar librerías](#1.1)
    * [2.2 Cargar datos](#1.2)
* [3. Análisis exploratorio (EDA)](#2)
    * [3.1 Estadísticas descriptivas](#2.1)
* [4. Preparación de datos](#3)
    * [4.1 Limpieza](#3.1)
    * [4.2 Datos categóricos (plantilla maestra)](#3.2)
    * [4.3 Preparar etiqueta de clasificación](#3.3)
    * [4.4 Ingeniería de características — indicadores técnicos](#3.4)
    * [4.5 Visualización de datos](#3.5)
    * [4.6 Selección de variables](#3.6)
    * [4.7 Transformación — estandarización](#3.7)
* [5. Evaluar algoritmos y modelos](#4)
    * [5.1 Partición entrenamiento / validación](#4.1)
    * [5.2 Métricas y validación cruzada](#4.2)
    * [5.3 Guía de modelos e hiperparámetros (teoría + gráficos)](#4.3)
    * [5.4 Comparar modelos](#4.4)
* [6. Ajuste fino y Grid Search](#5)
* [7. Finalizar el modelo](#6)
    * [7.1 Resultados en validación](#6.1)
    * [7.2 Importancia de variables](#6.2)
    * [7.3 Guardar el modelo](#6.3)
* [8. Backtesting](#7)
"""))

cells.append(md("""<a id='0'></a>
# 1. Definición del problema

El caso de estudio plantea **clasificación binaria**:

- **1 (compra / posición larga):** la media móvil **corta** del precio está **por encima** de la media **larga** → se espera momentum alcista de corto plazo.
- **0 (venta / fuera del mercado):** lo contrario.

Los modelos no “adivinan” el futuro: aprenden a mapear **indicadores técnicos** (tendencia, momentum, sobrecompra/sobreventa) hacia esa regla de referencia. En producción real habría que añadir costos, desfase temporal estricto y validación walk-forward.

**Datos:** Bitstamp (Bitcoin), frecuencia minutos. En GitHub viene una **muestra**; el libro usa el archivo completo en [Kaggle — Bitstamp minutes](https://www.kaggle.com/mlfinancebook/bitstamp-bicoin-minutes-data).
"""))

cells.append(md("""<a id='1'></a>
# 2. Inicio — librerías y datos
<a id='1.1'></a>
## 2.1 Cargar librerías

Mismas familias que el notebook original: `pandas`, `sklearn`, visualización y (opcional) redes vía `MLPClassifier` — equivalente práctico a la red shallow del master template en Colab sin Keras.
"""))

cells.append(code("""# Google Colab
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    !pip -q install seaborn scikit-learn

import warnings
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

cells.append(md("""<a id='1.2'></a>
## 2.2 Cargar datos

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

assert dataset is not None
"""))

cells.append(code("""type(dataset)
"""))

cells.append(md("""<a id='2'></a>
# 3. Análisis exploratorio (EDA)
<a id='2.1'></a>
## 3.1 Estadísticas descriptivas

**Cómo leerlo:** fíjate en `Close` (rango de precios), volúmenes y si hay muchos missing implícitos (el libro rellena hacia adelante). Minutos → muchísimas filas: luego usaremos las **últimas 100 000** observaciones, igual que el notebook oficial.
"""))

cells.append(code("""# shape
dataset.shape
"""))

cells.append(code("""# peek at data
set_option('display.width', 100)
dataset.tail(5)
"""))

cells.append(code("""# describe data
set_option('precision', 3)
dataset.describe()
"""))

cells.append(md("""**Interpretación rápida:** compara media y desviación de `Close` con la escala del activo; volúmenes muy bajos en algunos minutos son normales en cripto. Si `count` < filas totales, hay NaNs que trataremos en limpieza.
"""))

cells.append(md("""<a id='3'></a>
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

Etiqueta según cruce de medias móviles simples (SMA), como en el repo:

$$\\text{signal}_t = \\mathbb{1}\\{\\text{SMA}_{10}(P)_t > \\text{SMA}_{60}(P)_t\\}$$
"""))

cells.append(code("""# Create short / long simple moving average
dataset['short_mavg'] = dataset['Close'].rolling(window=10, min_periods=1, center=False).mean()
dataset['long_mavg'] = dataset['Close'].rolling(window=60, min_periods=1, center=False).mean()
dataset['signal'] = np.where(dataset['short_mavg'] > dataset['long_mavg'], 1.0, 0.0)
"""))

cells.append(code("""dataset.tail()
"""))

cells.append(md("""**Interpretación:** si `signal` está muy desbalanceado (p. ej. 90 % unos), accuracy puede engañar: el modelo podría predecir siempre la clase mayoritaria. Anota las proporciones antes de entrenar.
"""))

cells.append(code("""dataset['signal'].value_counts(normalize=True)
"""))

cells.append(md("""<a id='3.4'></a>
## 4.4 Ingeniería de características — indicadores técnicos

Mismo bloque funcional que el notebook de GitHub: EMA, ROC, momentum, RSI, estocástico (%K, %D) y medias móviles adicionales (MA21, MA63, MA252).
"""))

cells.append(code("""# calculation of exponential moving average
def EMA(df, n):
    EMA = pd.Series(df['Close'].ewm(span=n, min_periods=n).mean(), name='EMA_' + str(n))
    return EMA

dataset['EMA10'] = EMA(dataset, 10)
dataset['EMA30'] = EMA(dataset, 30)
dataset['EMA200'] = EMA(dataset, 200)

# calculation of rate of change
def ROC(df, n):
    M = df.diff(n - 1)
    N = df.shift(n - 1)
    ROC = pd.Series(((M / N) * 100), name='ROC_' + str(n))
    return ROC

dataset['ROC10'] = ROC(dataset['Close'], 10)
dataset['ROC30'] = ROC(dataset['Close'], 30)

# Calculation of price momentum
def MOM(df, n):
    MOM = pd.Series(df.diff(n), name='Momentum_' + str(n))
    return MOM

dataset['MOM10'] = MOM(dataset['Close'], 10)
dataset['MOM30'] = MOM(dataset['Close'], 30)

# calculation of relative strength index (libro / repo)
def RSI(series, period):
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

dataset['RSI10'] = RSI(dataset['Close'], 10)
dataset['RSI30'] = RSI(dataset['Close'], 30)
dataset['RSI200'] = RSI(dataset['Close'], 200)

# stochastic oscillator
def STOK(close, low, high, n):
    STOK = ((close - low.rolling(n).min()) / (high.rolling(n).max() - low.rolling(n).min())) * 100
    return STOK

def STOD(close, low, high, n):
    STOK = ((close - low.rolling(n).min()) / (high.rolling(n).max() - low.rolling(n).min())) * 100
    STOD = STOK.rolling(3).mean()
    return STOD

dataset['%K10'] = STOK(dataset['Close'], dataset['Low'], dataset['High'], 10)
dataset['%D10'] = STOD(dataset['Close'], dataset['Low'], dataset['High'], 10)
dataset['%K30'] = STOK(dataset['Close'], dataset['Low'], dataset['High'], 30)
dataset['%D30'] = STOD(dataset['Close'], dataset['Low'], dataset['High'], 30)
dataset['%K200'] = STOK(dataset['Close'], dataset['Low'], dataset['High'], 200)
dataset['%D200'] = STOD(dataset['Close'], dataset['Low'], dataset['High'], 200)

# moving averages (nombres MA21/63/252 como en el repo; ventanas 10/30/200)
def MA(df, n):
    MA = pd.Series(df['Close'].rolling(n, min_periods=n).mean(), name='MA_' + str(n))
    return MA

dataset['MA21'] = MA(dataset, 10)
dataset['MA63'] = MA(dataset, 30)
dataset['MA252'] = MA(dataset, 200)
dataset.tail()
"""))

cells.append(md("""<a id='3.5'></a>
## 4.5 Visualización de datos

Exploramos relaciones lineales entre indicadores (como en el caso Bitcoin del repo).
"""))

cells.append(code("""# excluding columns not needed for prediction (igual que fin-ml)
dataset = dataset.drop(['High', 'Low', 'Open', 'Volume_(Currency)', 'short_mavg', 'long_mavg'], axis=1)
dataset = dataset.dropna(axis=0)
dataset.tail()
"""))

cells.append(code("""# correlation
correlation = dataset.corr()
plt.figure(figsize=(15, 15))
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

cells.append(md("""<a id='4'></a>
# 5. Evaluar algoritmos y modelos
<a id='4.1'></a>
## 5.1 Partición entrenamiento / validación

Igual que GitHub: **últimas 100 000 filas**, 80 % train / 20 % validation, `random_state=1`.
"""))

cells.append(code("""subset_dataset = dataset.iloc[-100000:]
Y = subset_dataset['signal']
X = subset_dataset.loc[:, subset_dataset.columns != 'signal']
validation_size = 0.2
seed = 1
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, Y, test_size=validation_size, random_state=1
)
print('Train:', X_train.shape, 'Validation:', X_validation.shape)
"""))

cells.append(md("""**Interpretación:** usar solo el tramo final simula “entrenar con historia reciente”. No es walk-forward estricto (el split es aleatorio estratificado), pero coincide con el código del repositorio para comparar resultados en clase.
"""))

cells.append(md("""<a id='4.2'></a>
## 5.2 Opciones de prueba y métricas de evaluación
"""))

cells.append(code("""num_folds = 10
seed = 7
scoring = 'accuracy'
# scoring = 'precision'
# scoring = 'recall'
# scoring = 'neg_log_loss'
# scoring = 'roc_auc'
"""))

cells.append(md("""**Métricas (libro, cap. 6):**

- **Accuracy:** $\\frac{TP+TN}{N}$ — útil si clases balanceadas.
- **Precision (clase 1):** $\\frac{TP}{TP+FP}$ — penaliza falsas compras.
- **Recall (clase 1):** $\\frac{TP}{TP+FN}$ — penaliza perder subidas.
- **AUC-ROC:** calidad del ranking de probabilidades.

Para estrategias **long-only**, a menudo se discute precision vs recall (ver conclusión del notebook original).
"""))

cells.append(md("""<a id='4.3'></a>
## 5.3 Guía de modelos e hiperparámetros

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

cells.append(code("""demo_cols = ['RSI10', 'ROC10']
demo = subset_dataset[demo_cols + ['signal']].dropna()
Xd = demo[demo_cols]
yd = demo['signal']
Xd_tr, Xd_va, yd_tr, yd_va = train_test_split(Xd, yd, test_size=0.3, random_state=1, stratify=yd)

from sklearn.pipeline import Pipeline

depths = [2, 3, 4, 5, 6, 8, 10, 15, 20]
pipe = Pipeline([('sc', StandardScaler()), ('clf', DecisionTreeClassifier(random_state=1))])
tr, va = validation_curve(pipe, Xd_tr, yd_tr, param_name='clf__max_depth', param_range=depths, cv=3, scoring='accuracy')
plt.figure(figsize=(8, 4))
plt.plot(depths, tr.mean(1), 'o-', label='Train')
plt.plot(depths, va.mean(1), 'o-', label='CV')
plt.xlabel('max_depth'); plt.ylabel('Accuracy'); plt.title('Árbol: profundidad vs desempeño')
plt.legend(); plt.show()

Cs = np.logspace(-3, 2, 8)
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
## 5.4 Comparar modelos y algoritmos

### 5.4.1 Modelos de clasificación comunes  
### 5.4.2 Modelos ensemble (boosting y bagging)

Lista **idéntica** al notebook Bitcoin del repo.
"""))

cells.append(code("""models = []
models.append(('LR', LogisticRegression(max_iter=3000)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('NN', MLPClassifier(max_iter=400)))
models.append(('AB', AdaBoostClassifier()))
models.append(('GBM', GradientBoostingClassifier()))
models.append(('RF', RandomForestClassifier(n_jobs=-1)))
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

cells.append(code("""fig = plt.figure()
fig.suptitle('Comparación de algoritmos')
ax = fig.add_subplot(111)
plt.boxplot(results, labels=names)
ax.set_ylabel(scoring)
fig.set_size_inches(15, 8)
plt.xticks(rotation=45)
plt.show()
"""))

cells.append(md("""**Interpretación:** en el libro, **Random Forest** y **Gradient Boosting** suelen encabezar en series financieras ruidosas; **LR/LDA** son baselines interpretables. **KNN** puede ser lento con 100k filas. Compara **media ± desv. estándar** de CV: preferimos modelos altos *y* estables (caja estrecha).
"""))

cells.append(md("""<a id='5'></a>
# 6. Ajuste fino y Grid Search

Como en el repo: se elige **Random Forest** y se explora la malla de hiperparámetros.
"""))

cells.append(code("""# Grid Search: Random Forest Classifier (comentarios del notebook original)
# n_estimators: número de árboles
# max_depth: profundidad máxima
# criterion: 'gini' o 'entropy'

scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
n_estimators = [20, 80]
max_depth = [5, 10]
criterion = ['gini', 'entropy']
param_grid = dict(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion)
model = RandomForestClassifier(n_jobs=-1)
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)

print('Best: %f using %s' % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
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

Parámetros ganadores del grid (como en el repo); entrenamiento sobre **X_train sin escalar** (mismo código que fin-ml).
"""))

cells.append(code("""model = RandomForestClassifier(
    criterion='gini', n_estimators=80, max_depth=10, n_jobs=-1
)
model.fit(X_train, Y_train)
"""))

cells.append(code("""predictions = model.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
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

cells.append(md("""<a id='7'></a>
# 8. Backtesting

Misma lógica que el notebook oficial: retorno del activo × señal **desplazada un periodo** (decisión con información previa al retorno).
"""))

cells.append(code("""backtestdata = pd.DataFrame(index=X_validation.index)
backtestdata['signal_pred'] = predictions
backtestdata['signal_actual'] = Y_validation
backtestdata['Market Returns'] = X_validation['Close'].pct_change()
backtestdata['Actual Returns'] = backtestdata['Market Returns'] * backtestdata['signal_actual'].shift(1)
backtestdata['Strategy Returns'] = backtestdata['Market Returns'] * backtestdata['signal_pred'].shift(1)
backtestdata = backtestdata.reset_index()
backtestdata.head()
"""))

cells.append(code("""backtestdata[['Strategy Returns', 'Actual Returns']].cumsum().hist()
backtestdata[['Strategy Returns', 'Actual Returns']].cumsum().plot(figsize=(12, 5))
plt.title('Retornos acumulados (suma simple) — estrategia ML vs señal SMA')
plt.show()
"""))

cells.append(md("""### Conclusión (adaptada del libro / repo)

1. **Formulación:** convertir el objetivo de inversión en etiquetas y features es el primer paso; aquí la etiqueta sigue una regla SMA interpretable.
2. **Feature engineering:** indicadores de tendencia y momentum aumentan poder predictivo frente a usar solo precio crudo.
3. **Métricas:** accuracy o AUC son razonables en clasificación; si priorizas **long** con pocos falsos positivos, mira **precision**; si no quieres perder subidas, **recall**.
4. **Backtesting:** permite analizar riesgo/rentabilidad **antes** de arriesgar capital — sin comisiones, slippage ni tamaño de posición (limitaciones didácticas).

**Próximo paso académico:** dataset completo Kaggle, split temporal, costos de transacción y validación walk-forward del capítulo de producción.
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
