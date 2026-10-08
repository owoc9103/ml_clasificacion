"""Registro de clasificadores supervisados para el laboratorio académico."""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42

MODEL_NAMES: tuple[str, ...] = (
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "svm",
    "knn",
    "naive_bayes",
)

DISPLAY_NAMES: dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "xgboost": "XGBoost",
    "svm": "SVM",
    "knn": "KNN",
    "naive_bayes": "Naive Bayes",
}

SCALED_MODELS = {"logistic_regression", "svm", "knn"}


def build_model(name: str):
    """Crea un estimador sklearn (Pipeline) sin ajustar."""
    key = name.strip().lower()
    if key == "logistic_regression":
        est = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
    elif key == "decision_tree":
        est = DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=10, random_state=RANDOM_STATE
        )
    elif key == "random_forest":
        est = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=10,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    elif key == "gradient_boosting":
        est = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        )
    elif key == "xgboost":
        est = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=1,
            tree_method="hist",
            random_state=RANDOM_STATE,
        )
    elif key == "svm":
        est = CalibratedClassifierCV(
            estimator=SVC(kernel="rbf", C=1.0, random_state=RANDOM_STATE),
            method="sigmoid",
            cv=TimeSeriesSplit(n_splits=3),
        )
    elif key == "knn":
        est = KNeighborsClassifier(n_neighbors=15)
    elif key == "naive_bayes":
        est = GaussianNB()
    else:
        raise ValueError(f"Modelo desconocido: {name}. Opciones: {', '.join(MODEL_NAMES)}")

    steps: list[tuple] = []
    if key in SCALED_MODELS:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", est))
    return Pipeline(steps)


def predict_positive_proba(model, features) -> float:
    """P(Y=1) para una fila o matriz; devuelve vector si hay varias filas."""
    import numpy as np

    proba = model.predict_proba(features)
    estimator = model.named_steps["model"] if hasattr(model, "named_steps") else model
    classes = list(getattr(estimator, "classes_", [0, 1]))
    idx = classes.index(1) if 1 in classes else classes.index(1.0)
    return np.asarray(proba[:, idx], dtype=float)
