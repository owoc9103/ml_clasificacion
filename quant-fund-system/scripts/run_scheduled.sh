#!/bin/bash
# Ejecución programada SIN Docker (GitHub Actions o cron local).
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${PROJECT_DIR}/logs/live"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/trading_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

echo "========================================" | tee -a "${LOG_FILE}"
echo "Starting trading run at $(date)" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

cd "${PROJECT_DIR}"

PYTHON="${PYTHON:-python}"
if [ -x "${PROJECT_DIR}/.venv/bin/python" ]; then
  PYTHON="${PROJECT_DIR}/.venv/bin/python"
elif [ -x "${PROJECT_DIR}/.venv/Scripts/python.exe" ]; then
  PYTHON="${PROJECT_DIR}/.venv/Scripts/python.exe"
fi

# Actualizar datos y datasets (universo en configs/universe.yaml)
"${PYTHON}" scripts/fetch_data.py --intervals 1d >> "${LOG_FILE}" 2>&1
"${PYTHON}" scripts/make_dataset.py --intervals 1d >> "${LOG_FILE}" 2>&1
"${PYTHON}" scripts/maybe_retrain.py --interval 1d >> "${LOG_FILE}" 2>&1

# Paper trading (requiere ALPACA_* en el entorno)
if "${PYTHON}" scripts/run_live.py --mode paper >> "${LOG_FILE}" 2>&1; then
  echo "SUCCESS: Trading run completed at $(date)" | tee -a "${LOG_FILE}"
  EXIT_CODE=0
else
  echo "ERROR: Trading run failed at $(date)" | tee -a "${LOG_FILE}"
  EXIT_CODE=1
fi

find "${LOG_DIR}" -name "trading_*.log" -mtime +30 -delete 2>/dev/null || true
exit $EXIT_CODE
