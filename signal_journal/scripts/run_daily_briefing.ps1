# Datos quant-fund + briefing CSV (sin órdenes). Ejecutar cada mañana.

$ErrorActionPreference = "Stop"
$JournalDir = Split-Path -Parent $PSScriptRoot
$QuantDir = Join-Path (Split-Path -Parent $JournalDir) "quant-fund-system"

& "$QuantDir\scripts\run_daily_data.ps1"
if ($LASTEXITCODE -ne 0) { exit 1 }

Push-Location $JournalDir
$Python = "python"
if (Test-Path "$JournalDir\.venv\Scripts\python.exe") {
    $Python = "$JournalDir\.venv\Scripts\python.exe"
}
& $Python scripts/collect_morning.py --replace
$code = $LASTEXITCODE
Pop-Location

if ($code -ne 0) { exit $code }
Write-Host "Briefing diario listo. Abre Streamlit: streamlit run app.py"
exit 0
