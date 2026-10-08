"""Genera Bitcoin_Estrategia_Clasificacion_Colab.ipynb"""
import json
from pathlib import Path

def md(s: str):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}

def code(s: str):
    return {"cell_type": "code", "metadata": {}, "source": s.splitlines(keepends=True), "outputs": [], "execution_count": None}

cells = []

cells.append(md("""# Estrategia de trading Bitcoin con modelos de clasificación

**Curso:** ML en Finanzas — Clase 11  
**Basado en:** Tatsat, Puri & Lookabaugh — *Machine Learning and Data Science Blueprints for Finance* (Cap. 6) y el [Master Template de clasificación](https://github.com/tatsath/fin-ml).

**Objetivo:** Predecir señal **compra (1)** vs **venta/mantener corto (0)** comparando precio de corto vs largo plazo, usando el flujo completo de ML: EDA → preparación → comparación de modelos → ajuste → evaluación → *backtesting*.

> ⚠️ Uso académico. No es asesoría de inversión.
"""))

cells.append(md("""## Contenido

* [1. Introducción](#0)
* [2. Inicio — librerías y datos](#1)
    * [2.1 Cargar librerías](#1.1)
    * [2.2 Cargar datos (Colab)](#1.2)
* [3. Análisis exploratorio (EDA)](#2)
    * [3.1 Estadísticas descriptivas](#2.1)
    * [3.2 Visualización de datos](#2.2)
* [4. Preparación de datos](#3)
    * [4.1 Limpieza](#3.1)
    * [4.2 Etiqueta de clasificación](#3.2)
    * [4.3 Ingeniería de características (indicadores)](#3.3)
    * [4.4 Selección de variables y escalado](#3.4)
* [4A. Guía de modelos, fórmulas e hiperparámetros](#3A)
* [5. Evaluar algoritmos y modelos](#4)
    * [5.1 Partición train/validation](#4.1)
    * [5.2 Métricas de evaluación](#4.2)
    * [5.3 Comparar modelos](#4.3)
* [6. Ajuste fino y Grid Search](#5)
* [7. Finalizar el modelo](#6)
    * [7.1 Resultados en validación](#6.1)
    * [7.2 Intuición de variables / importancia](#6.2)
    * [7.3 Guardar el modelo](#6.3)
* [8. Backtesting](#7)
"""))

cells.append(md("""<a id='0'></a>
# 1. Introducción

El problema se plantea como **clasificación binaria**:

- **$y = 1$ (compra):** la media móvil corta está por encima de la larga → tendencia alcista de corto plazo.
- **$y = 0$ (venta / no largo):** lo contrario.

En el libro, la señal se construye con medias móviles simples (SMA) y se enriquece con indicadores técnicos (RSI, estocástico, ROC, momentum, EMA). Los modelos supervisados aprenden a aproximar esa regla (y patrones no lineales) a partir de features **sin usar** directamente las medias usadas para etiquetar en producción (se eliminan del set de predictores tras crear la etiqueta).

Aprenderás a:

1. Recorrer un caso de ML **de punta a punta** (plantilla maestra del capítulo 6).
2. Comparar clasificadores clásicos y *ensemble* con validación cruzada.
3. Interpretar **hiperparámetros** y su efecto en métricas y fronteras de decisión.
4. Simular una estrategia con **backtesting** simple sobre el conjunto de validación.
"""))

cells.append(md("""<a id='1'></a>
# 2. Inicio — librerías y datos
<a id='1.1'></a>
## 2.1 Cargar librerías
"""))

cells.append(code("""# Entorno Google Colab
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
import seaborn as sns
from pandas import set_option

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, validation_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from pickle import dump, load

plt.style.use('seaborn-v0_8-whitegrid')
set_option('display.max_columns', 20)
seed = 1
np.random.seed(seed)
"""))

cells.append(md("""<a id='1.2'></a>
## 2.2 Cargar datos

Datos: Bitstamp (Bitcoin), frecuencia minutos. Muestra incluida en `data/`; el libro usa el archivo completo en Kaggle.
"""))

cells.append(code("""from pathlib import Path

# Rutas: Colab (repo clonado), local Ejemplo_Clase, o descarga directa fin-ml / tu GitHub
CANDIDATE_PATHS = [
    Path('Ejemplo_Clase/data/BitstampData_sample.csv'),
    Path('data/BitstampData_sample.csv'),
    Path('/content/ml_clasificacion/Ejemplo_Clase/data/BitstampData_sample.csv'),
]
URLS = [
    'https://raw.githubusercontent.com/owoc9103/ml_clasificacion/main/Ejemplo_Clase/data/BitstampData_sample.csv',
    'https://raw.githubusercontent.com/tatsath/fin-ml/master/Chapter%206%20-%20Sup.%20Learning%20-%20Classification%20models/CaseStudy3%20-%20Bitcoin%20Trading%20Strategy/BitstampData_sample.csv',
]

dataset = None
for p in CANDIDATE_PATHS:
    if p.exists():
        dataset = pd.read_csv(p)
        print('Cargado desde', p.resolve())
        break
if dataset is None:
    for url in URLS:
        try:
            dataset = pd.read_csv(url)
            print('Descargado desde', url)
            break
        except Exception as e:
            print('Fallo', url, e)

assert dataset is not None, 'No se encontró BitstampData_sample.csv'
print('Shape:', dataset.shape)
dataset.tail(3)
"""))

cells.append(code("""# Opcional en Colab: usar submuestra para acelerar la clase (cambiar a False con datos completos)
USE_SUBSAMPLE = True
SUBSAMPLE_EVERY = 10  # cada 10 filas ≈ cada 10 minutos si la serie es minuto a minuto

if USE_SUBSAMPLE and len(dataset) > 100_000:
    dataset = dataset.iloc[::SUBSAMPLE_EVERY].reset_index(drop=True)
    print('Submuestra para práctica:', dataset.shape)
"""))

cells.append(md("""<a id='2'></a>
# 3. Análisis exploratorio (EDA)
<a id='2.1'></a>
## 3.1 Estadísticas descriptivas
"""))

cells.append(code("""set_option('precision', 3)
dataset.describe().T
"""))

cells.append(md("""<a id='2.2'></a>
## 3.2 Visualización de datos
"""))

cells.append(code("""if 'Timestamp' in dataset.columns:
    dataset['datetime'] = pd.to_datetime(dataset['Timestamp'], unit='s')
else:
    dataset['datetime'] = pd.RangeIndex(len(dataset))

fig, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
ax[0].plot(dataset['datetime'], dataset['Close'], color='steelblue', lw=0.8)
ax[0].set_title('Precio de cierre Bitcoin (Bitstamp)')
ax[0].set_ylabel('USD')
if 'Volume_(BTC)' in dataset.columns:
    ax[1].bar(dataset['datetime'], dataset['Volume_(BTC)'], width=0.8, color='gray', alpha=0.5)
    ax[1].set_ylabel('Volumen BTC')
plt.tight_layout()
plt.show()
"""))

cells.append(md("""<a id='3'></a>
# 4. Preparación de datos
<a id='3.1'></a>
## 4.1 Limpieza
"""))

cells.append(code("""print('¿Hay nulos?', dataset.isnull().values.any())
dataset[dataset.columns] = dataset[dataset.columns].ffill()
if 'Timestamp' in dataset.columns:
    dataset = dataset.drop(columns=['Timestamp'])
"""))

cells.append(md("""<a id='3.2'></a>
## 4.2 Etiqueta de clasificación

Regla del caso de estudio (SMA corta vs larga):

$$\\text{signal}_t = \\mathbb{1}\\{\\text{SMA}_{10}(P)_t > \\text{SMA}_{60}(P)_t\\}$$

con $P$ el precio de cierre.
"""))

cells.append(code("""dataset['short_mavg'] = dataset['Close'].rolling(window=10, min_periods=1).mean()
dataset['long_mavg'] = dataset['Close'].rolling(window=60, min_periods=1).mean()
dataset['signal'] = np.where(dataset['short_mavg'] > dataset['long_mavg'], 1.0, 0.0)
dataset['signal'].value_counts(normalize=True)
"""))

cells.append(md("""<a id='3.3'></a>
## 4.3 Ingeniería de características — indicadores técnicos

Como en el libro: EMA, ROC, momentum, RSI y estocástico (%K, %D).
"""))

cells.append(code("""def EMA(df, n):
    return pd.Series(df['Close'].ewm(span=n, min_periods=n).mean(), name='EMA_' + str(n))

def ROC(close, n):
    M = close.diff(n - 1)
    N = close.shift(n - 1)
    return pd.Series((M / N) * 100, name='ROC_' + str(n))

def MOM(close, n):
    return pd.Series(close.diff(n), name='Momentum_' + str(n))

def RSI(series, period):
    delta = series.diff().dropna()
    u = delta.clip(lower=0)
    d = (-delta.clip(upper=0))
    u_avg = u.ewm(com=period - 1, adjust=False).mean()
    d_avg = d.ewm(com=period - 1, adjust=False).mean()
    rs = u_avg / d_avg.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def STOK(close, low, high, n):
    return ((close - low.rolling(n).min()) / (high.rolling(n).max() - low.rolling(n).min())) * 100

def STOD(close, low, high, n):
    k = STOK(close, low, high, n)
    return k.rolling(3).mean()

for n in (10, 30, 200):
    dataset[f'EMA{n}'] = EMA(dataset, n)
dataset['ROC10'] = ROC(dataset['Close'], 10)
dataset['ROC30'] = ROC(dataset['Close'], 30)
dataset['MOM10'] = MOM(dataset['Close'], 10)
dataset['MOM30'] = MOM(dataset['Close'], 30)
for n in (10, 30, 200):
    dataset[f'RSI{n}'] = RSI(dataset['Close'], n)
for n in (10, 30, 200):
    dataset[f'%K{n}'] = STOK(dataset['Close'], dataset['Low'], dataset['High'], n)
    dataset[f'%D{n}'] = STOD(dataset['Close'], dataset['Low'], dataset['High'], n)

dataset = dataset.replace([np.inf, -np.inf], np.nan).ffill().bfill()
"""))

cells.append(md("""<a id='3.4'></a>
## 4.4 Selección de variables y escalado

Eliminamos columnas crudas y las SMA usadas solo para etiquetar. Estandarizamos antes de modelos sensibles a escala (SVM, KNN, redes).
"""))

cells.append(code("""drop_cols = ['High', 'Low', 'Open', 'Volume_(Currency)', 'short_mavg', 'long_mavg', 'datetime']
drop_cols = [c for c in drop_cols if c in dataset.columns]
dataset_model = dataset.drop(columns=drop_cols)

Y = dataset_model['signal'].astype(int)
X = dataset_model.drop(columns=['signal'])
feature_names = list(X.columns)
print('Features:', len(feature_names))
X.head()
"""))

cells.append(md("""<a id='3A'></a>
# 4A. Guía de modelos, fórmulas e hiperparámetros

Referencia: *Blueprints for Finance*, Cap. 6 (clasificación supervisada).

| Modelo | Idea | Hiperparámetro clave | Efecto intuitivo |
|--------|------|----------------------|------------------|
| **Regresión logística** | $P(y=1\\mid x)=\\sigma(w^\\top x + b)$, $\\sigma(z)=1/(1+e^{-z})$ | `C` (inverso de regularización L2) | `C` grande → ajuste más flexible, riesgo de sobreajuste |
| **LDA** | Asume clases Gaussianas con misma covarianza; frontera lineal | — | Baseline lineal interpretable |
| **KNN** | Voto de los *k* vecinos más cercanos | `n_neighbors` | *k* pequeño → frontera muy irregular; *k* grande → suaviza |
| **Árbol (CART)** | Particiones que minimizan impureza (Gini/entropía) | `max_depth`, `min_samples_leaf` | Profundidad alta → memoriza ruido |
| **Naive Bayes** | $P(x\\mid y)=\\prod_j P(x_j\\mid y)$ (independencia condicional) | — | Rápido; supuesto fuerte |
| **MLP** | Capas densas + activación no lineal | `hidden_layer_sizes`, `alpha` | Más neuronas → más capacidad |
| **AdaBoost** | Combina clasificadores débiles con pesos | `n_estimators`, `learning_rate` | Más estimadores → mejor ajuste hasta saturar |
| **Gradient Boosting** | Boosting por gradiente sobre pérdida | `max_depth`, `n_estimators` | Similar a RF pero secuencial |
| **Random Forest** | Promedio de árboles sobre *bootstrap* | `n_estimators`, `max_depth`, `criterion` | Más árboles → menor varianza |

**Métricas (clasificación):**

$$\\text{Accuracy} = \\frac{TP+TN}{TP+TN+FP+FN}, \\quad
\\text{Precision}_1 = \\frac{TP}{TP+FP}, \\quad
\\text{Recall}_1 = \\frac{TP}{TP+FN}$$

En trading, **recall** de la clase compra puede importar si priorizas capturar subidas; **precision** si quieres pocas falsas alarmas.
"""))

cells.append(md("""### Demostración visual: efecto de hiperparámetros (2 features)

Usamos solo `RSI10` y `ROC10` para graficar fronteras y curvas de validación.
"""))

cells.append(code("""from sklearn.pipeline import Pipeline

X_demo = X[['RSI10', 'ROC10']].dropna()
Y_demo = Y.loc[X_demo.index]

X_tr, X_te, y_tr, y_te = train_test_split(X_demo, Y_demo, test_size=0.3, random_state=seed, stratify=Y_demo)

# Curva: profundidad del árbol
pipe_tree = Pipeline([('scaler', StandardScaler()), ('clf', DecisionTreeClassifier(random_state=seed))])
depths = [2, 3, 4, 5, 6, 8, 10, 15, 20]
train_scores, val_scores = validation_curve(
    pipe_tree, X_tr, y_tr, param_name='clf__max_depth', param_range=depths, cv=3, scoring='accuracy', n_jobs=-1
)
plt.figure(figsize=(8, 4))
plt.plot(depths, train_scores.mean(axis=1), 'o-', label='Train')
plt.plot(depths, val_scores.mean(axis=1), 'o-', label='CV')
plt.xlabel('max_depth'); plt.ylabel('Accuracy'); plt.title('Árbol de decisión: profundidad vs desempeño')
plt.legend(); plt.show()

# Curva: C en regresión logística
pipe_lr = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=2000, random_state=seed))])
Cs = np.logspace(-3, 2, 8)
tr, va = validation_curve(pipe_lr, X_tr, y_tr, param_name='clf__C', param_range=Cs, cv=3, scoring='accuracy', n_jobs=-1)
plt.figure(figsize=(8, 4))
plt.semilogx(Cs, tr.mean(axis=1), 'o-', label='Train')
plt.semilogx(Cs, va.mean(axis=1), 'o-', label='CV')
plt.xlabel('C'); plt.ylabel('Accuracy'); plt.title('Logistic Regression: regularización')
plt.legend(); plt.show()
"""))

cells.append(md("""<a id='4'></a>
# 5. Evaluar algoritmos y modelos
<a id='4.1'></a>
## 5.1 Partición train / validation
"""))

cells.append(code("""validation_size = 0.2
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, Y, test_size=validation_size, random_state=seed, stratify=Y
)
print(X_train.shape, X_validation.shape)
"""))

cells.append(md("""<a id='4.2'></a>
## 5.2 Opciones de prueba y métricas
"""))

cells.append(code("""num_folds = 5
scoring = 'accuracy'  # alternativas: 'precision', 'recall', 'roc_auc', 'neg_log_loss'
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
"""))

cells.append(md("""<a id='4.3'></a>
## 5.3 Comparar modelos y algoritmos

### 5.3.1 Modelos de clasificación comunes  
### 5.3.2 Modelos ensemble (*boosting* y *bagging*)
"""))

cells.append(code("""models = [
    ('LR', LogisticRegression(max_iter=2000, random_state=seed)),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier(random_state=seed)),
    ('NB', GaussianNB()),
    ('NN', MLPClassifier(max_iter=400, random_state=seed)),
    ('AB', AdaBoostClassifier(random_state=seed)),
    ('GBM', GradientBoostingClassifier(random_state=seed)),
    ('RF', RandomForestClassifier(random_state=seed, n_jobs=-1)),
    ('ET', ExtraTreesClassifier(random_state=seed, n_jobs=-1)),
]

results, names = [], []
for name, model in models:
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring, n_jobs=-1)
    results.append(cv_results)
    names.append(name)
    print(f'{name}: {cv_results.mean():.4f} ({cv_results.std():.4f})')
"""))

cells.append(code("""fig, ax = plt.subplots(figsize=(12, 6))
ax.boxplot(results, labels=names)
ax.set_title('Comparación de algoritmos (validación cruzada)')
ax.set_ylabel(scoring)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""))

cells.append(md("""<a id='5'></a>
# 6. Ajuste fino y Grid Search

Seleccionamos **Random Forest** (suele ser fuerte en el caso del libro) y exploramos la malla de hiperparámetros.
"""))

cells.append(code("""scaler = StandardScaler()
rescaledX = scaler.fit_transform(X_train)

param_grid = {
    'n_estimators': [20, 80],
    'max_depth': [5, 10],
    'criterion': ['gini', 'entropy'],
}
model = RandomForestClassifier(random_state=seed, n_jobs=-1)
grid = GridSearchCV(model, param_grid, scoring=scoring, cv=kfold, n_jobs=-1)
grid_result = grid.fit(rescaledX, Y_train)

print('Mejor:', grid_result.best_score_, grid_result.best_params_)
for mean, std, param, rank in zip(
    grid_result.cv_results_['mean_test_score'],
    grid_result.cv_results_['std_test_score'],
    grid_result.cv_results_['params'],
    grid_result.cv_results_['rank_test_score'],
):
    print(f'#{rank} {mean:.4f} ({std:.4f}) {param}')
"""))

cells.append(code("""# Heatmap visual: n_estimators vs max_depth (mejor criterion)
cv_df = pd.DataFrame(grid_result.cv_results_)
for crit in ['gini', 'entropy']:
    sub = cv_df[cv_df['param_criterion'] == crit]
    if sub.empty:
        continue
    pivot = sub.pivot_table(index='param_max_depth', columns='param_n_estimators', values='mean_test_score')
    plt.figure(figsize=(5, 4))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis')
    plt.title(f'Grid RF — criterion={crit}')
    plt.show()
"""))

cells.append(md("""<a id='6'></a>
# 7. Finalizar el modelo
<a id='6.1'></a>
## 7.1 Resultados en el conjunto de validación
"""))

cells.append(code("""best = grid_result.best_params_
final_model = RandomForestClassifier(
    criterion=best['criterion'],
    n_estimators=best['n_estimators'],
    max_depth=best['max_depth'],
    random_state=seed,
    n_jobs=-1,
)
final_model.fit(scaler.transform(X_train), Y_train)
pred = final_model.predict(scaler.transform(X_validation))

print('Accuracy validación:', accuracy_score(Y_validation, pred))
print(confusion_matrix(Y_validation, pred))
print(classification_report(Y_validation, pred, digits=4))
"""))

cells.append(md("""<a id='6.2'></a>
## 7.2 Intuición de variables / importancia de features
"""))

cells.append(code("""imp = pd.Series(final_model.feature_importances_, index=feature_names).sort_values(ascending=False)
imp.head(15).plot(kind='barh', figsize=(8, 6), title='Importancia (Random Forest)')
plt.tight_layout()
plt.show()
"""))

cells.append(md("""<a id='6.3'></a>
## 7.3 Guardar modelo para uso posterior
"""))

cells.append(code("""artifact = {'model': final_model, 'scaler': scaler, 'features': feature_names}
with open('modelo_bitcoin_rf.pkl', 'wb') as f:
    dump(artifact, f)
print('Guardado modelo_bitcoin_rf.pkl')
"""))

cells.append(md("""<a id='7'></a>
# 8. Backtesting

Simulación simplificada (como en el notebook original): retorno de mercado multiplicado por la señal predicha (desplazada un periodo para evitar *look-ahead* obvio en la demo).
"""))

cells.append(code("""# Necesitamos Close en validación
close_val = dataset.loc[X_validation.index, 'Close']
back = pd.DataFrame(index=X_validation.index)
back['signal_pred'] = pred
back['signal_actual'] = Y_validation.values
back['Market Returns'] = close_val.pct_change()
back['Actual Returns'] = back['Market Returns'] * back['signal_actual'].shift(1)
back['Strategy Returns'] = back['Market Returns'] * back['signal_pred'].shift(1)
back[['Strategy Returns', 'Actual Returns']].fillna(0).cumsum().plot(figsize=(12, 5))
plt.title('Retornos acumulados: estrategia ML vs señal SMA (actual)')
plt.ylabel('Retorno acumulado (suma simple)')
plt.show()
"""))

cells.append(md("""---

## Cierre de la clase

1. Repasa la **plantilla maestra**: EDA → preparación → benchmark → *grid search* → evaluación → persistencia.
2. Relaciona métricas con el objetivo de trading (precisión vs recall).
3. Para producción académica: validación temporal (*walk-forward*), costos de transacción y dataset completo de Kaggle.

**Dataset completo:** [Kaggle — Bitstamp minutes (libro)](https://www.kaggle.com/mlfinancebook/bitstamp-bicoin-minutes-data)
"""))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "colab": {"provenance": [], "name": "Bitcoin_Estrategia_Clasificacion_Colab.ipynb"},
    },
    "cells": cells,
}

out = Path(__file__).parent / "Bitcoin_Estrategia_Clasificacion_Colab.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Wrote", out)
