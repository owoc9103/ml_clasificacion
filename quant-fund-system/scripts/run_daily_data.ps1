# Actualización diaria: precios + indicadores (sin trading).
# Programar en Task Scheduler (Windows) a primera hora, lun–vie.

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectDir

$Python = "python"
if (Test-Path "$ProjectDir\.venv\Scripts\python.exe") {
    $Python = "$ProjectDir\.venv\Scripts\python.exe"
}

$LogDir = Join-Path $ProjectDir "logs\data"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile = Join-Path $LogDir ("daily_{0:yyyyMMdd_HHmmss}.log" -f (Get-Date))

function Log($msg) {
    $line = "{0} | {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Add-Content -Path $LogFile -Value $line
    Write-Host $line
}

Log "Inicio actualización diaria de datos"
& $Python scripts/fetch_data.py --intervals 1d 2>&1 | Tee-Object -FilePath $LogFile -Append
if ($LASTEXITCODE -ne 0) { Log "ERROR fetch_data"; exit 1 }

& $Python scripts/make_dataset.py --intervals 1d 2>&1 | Tee-Object -FilePath $LogFile -Append
if ($LASTEXITCODE -ne 0) { Log "ERROR make_dataset"; exit 1 }

Log "Comprobando reentrenamiento (configs/train.yaml: retrain_every_days)"
& $Python scripts/maybe_retrain.py --interval 1d 2>&1 | Tee-Object -FilePath $LogFile -Append
if ($LASTEXITCODE -ne 0) { Log "ERROR maybe_retrain"; exit 1 }

Log "Datos, indicadores y modelos (si tocaba) OK"
exit 0
