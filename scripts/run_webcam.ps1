param(
    [string]$Config = "config\\webcam.example.json"
)

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Error "Önce .\scripts\setup_webcam.ps1 çalıştırın."
    exit 1
}

$env:PYTHONPATH = "src"
& .\.venv\Scripts\python.exe -m helmetai_fcw webcam-run --config $Config
