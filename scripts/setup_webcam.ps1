$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv (Join-Path $projectRoot '.venv')
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv (Join-Path $projectRoot '.venv')
    } else {
        throw 'Python is missing. Install Python 3.10 or newer with PATH support, then reopen PowerShell.'
    }
    if ($LASTEXITCODE -ne 0) { throw 'Creating the Python environment failed.' }
}
& $python -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'
if ($LASTEXITCODE -ne 0) { throw 'This project requires Python 3.10 or newer.' }
& $python -m pip install -r (Join-Path $projectRoot 'requirements-webcam.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Setup is NOT complete.' }
& $python -c 'import cv2, numpy; print(cv2.__version__, numpy.__version__)'
if ($LASTEXITCODE -ne 0) { throw 'Dependency import check failed. Setup is NOT complete.' }
Write-Host 'Setup complete. Use run_image_check.ps1, run_video_demo.ps1 or run_log_report.ps1.'
