# Signal Journal (Streamlit)

App **separada** de `quant-fund-system`: no modifica ese proyecto.

- Calcula señales y pesos sugeridos con el **mismo modelo** (`live_model` en `configs/train.yaml`).
- Muestra **indicadores** clave por ticker.
- **No ejecuta órdenes**; el usuario decide si opera en Alpaca.
- Acumula cada consulta en `data/signal_journal.csv` (y `signal_journal.xlsx`).

## Instalación

```powershell
cd "...\Clase_11\signal_journal"
pip install -r requirements.txt
pip install -e "..\quant-fund-system"
```

## Uso diario

1. Actualizar datos en el repo quant (fetch + make_dataset).
2. `streamlit run app.py`
3. Pulsar **Generar señales de hoy**.

Automatización sin UI:

```powershell
# Datos hasta HOY + indicadores + CSV del briefing (recomendado cada mañana)
..\quant-fund-system\scripts\run_daily_data.ps1
python scripts/collect_morning.py --replace

# O todo en uno:
.\scripts\run_daily_briefing.ps1
```

Programa `run_daily_briefing.ps1` en **Task Scheduler** (lun–vie, ~7:00–8:00).

GitHub: workflow `daily-data.yml` (si subes `Clase_11` completo al repo).
