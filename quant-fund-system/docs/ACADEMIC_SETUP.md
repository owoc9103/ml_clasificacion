# Despliegue académico (sin Docker / sin AWS)

## Qué cambió

- **Docker y AWS** dejan de ser obligatorios. `Dockerfile` y `deploy/` quedan como referencia opcional.
- **Automatización principal:** GitHub Actions (`.github/workflows/academic-pipeline.yml`).
- **Modelos:** además de XGBoost, puede entrenar Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, KNN y Naive Bayes.
- **Universo:** se mantiene `configs/universe.yaml` (13 símbolos actuales).

## Modelos y artefactos

Cada modelo se guarda en:

```text
artifacts/1d/models/<model_key>/model_all.joblib
artifacts/1d/models/<model_key>/calibrator.joblib
artifacts/1d/models/<model_key>/feature_meta.json
```

XGBoost también copia a la ruta legacy `artifacts/1d/model_all.joblib`.

El modelo en paper trading se elige con `live_model` en `configs/train.yaml` (por defecto `xgboost`).

## Comandos locales (PowerShell)

```powershell
cd quant-fund-system
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"

python scripts/fetch_data.py --intervals 1d
python scripts/make_dataset.py --intervals 1d
```

`configs/data.yaml` usa `eod_end: today` → siempre descarga hasta la fecha actual.

**Windows (cada mañana):** `scripts\run_daily_data.ps1` en el Programador de tareas. Tras actualizar datos, ejecuta `scripts/maybe_retrain.py`: reentrena todos los modelos de `configs/train.yaml` si hace más de `retrain_every_days` (por defecto 7) desde el último entrenamiento.

**Briefing + datos:** desde `signal_journal`, `scripts\run_daily_briefing.ps1` (incluye lo anterior + CSV del briefing).

Entrenamiento manual (forzar o comparar modelos):

```powershell
python scripts/maybe_retrain.py --interval 1d --force
python scripts/train_models.py --start 2018-01-01 --end 2025-12-31 --interval 1d
python scripts/train_model.py --start 2018-01-01 --end 2025-12-31 --model random_forest

# Comparación en reports/model_comparison_1d.json
```

## GitHub Actions

1. Suba este repo a GitHub (raíz = carpeta `quant-fund-system` o incluya el workflow en la raíz del remoto).
2. **Settings → Secrets → Actions:**
   - `ALPACA_API_KEY`
   - `ALPACA_SECRET_KEY`
3. Ejecute **workflow_dispatch** en *Academic Trading Pipeline* o espere el cron (UTC).

El job **entrena todos los modelos en cada ejecución programada** (`train_models.py` con `--end` = hoy) y deja artefactos en *Artifacts* del run. En local, `maybe_retrain.py` usa el mismo entrenamiento pero solo si pasó `retrain_every_days`.

## Cron local (sin Docker)

En Linux/macOS puede usar `scripts/run_scheduled.sh` con cron. En Windows, Task Scheduler llamando al mismo script vía Git Bash o el workflow de GitHub.
