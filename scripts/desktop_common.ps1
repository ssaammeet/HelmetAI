# Shared helpers. No administrator rights, camera access or GPIO changes.
$ErrorActionPreference = 'Stop'
$script:HelmetProjectRoot = Split-Path -Parent $PSScriptRoot

function Resolve-HelmetInput([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "File not found: $Path"
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

function Get-HelmetOutputPath([string]$Path) {
    return $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path)
}

function Get-HelmetNewResultPath([string]$Kind) {
    $parent = Join-Path $script:HelmetProjectRoot 'results'
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    $name = '{0}-{1}-{2}' -f $Kind, (Get-Date -Format 'yyyyMMdd-HHmmss'), ([guid]::NewGuid().ToString('N').Substring(0,8))
    return Join-Path $parent $name
}

function Invoke-HelmetPython([string[]]$ModuleArguments) {
    $python = Join-Path $script:HelmetProjectRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        throw "Python environment missing. First run scripts\setup_webcam.ps1 in this V7 project."
    }
    $previousPythonPath = $env:PYTHONPATH
    $previousPythonIOEncoding = $env:PYTHONIOENCODING
    $previousConsoleEncoding = [Console]::OutputEncoding
    try {
        $env:PYTHONPATH = Join-Path $script:HelmetProjectRoot 'src'
        # Native Python may have redirected stdout even in an interactive PS5
        # session. Do not let a Unicode result path turn success into an error.
        $env:PYTHONIOENCODING = 'utf-8'
        [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
        & $python @ModuleArguments
        if ($LASTEXITCODE -ne 0) { throw "HelmetAI command failed (exit $LASTEXITCODE). Read the error above." }
    } finally {
        $env:PYTHONPATH = $previousPythonPath
        $env:PYTHONIOENCODING = $previousPythonIOEncoding
        [Console]::OutputEncoding = $previousConsoleEncoding
    }
}
