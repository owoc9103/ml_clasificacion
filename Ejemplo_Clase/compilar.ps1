# Render Clase 11 — Modelos supervisados (revealjs + pandas-mono)
Set-Location $PSScriptRoot

quarto render Clase_11_Modelos_Supervisados.qmd

if (Test-Path "Clase_11_Modelos_Supervisados.html") {
    Write-Host "OK: Clase_11_Modelos_Supervisados.html generado."
} else {
    Write-Error "Render failed."
    exit 1
}
