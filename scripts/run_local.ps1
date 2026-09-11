param(
    [ValidateSet("demo", "rear-demo", "demo-dashboard", "test", "monocular-calibration")]
    [string]$Mode = "demo"
)

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$env:PYTHONPATH = "src"

switch ($Mode) {
    "demo" { py -3 -m helmetai_fcw demo }
    "rear-demo" { py -3 -m helmetai_fcw rear-demo }
    "demo-dashboard" { py -3 -m helmetai_fcw demo-dashboard }
    "test" { py -3 -m unittest discover -s tests -v }
    "monocular-calibration" {
        Write-Host "Örnek: .\scripts\run_local.ps1 monocular-calibration"
        Write-Host "Sonra: py -3 -m helmetai_fcw calibrate-monocular --reference-distance 10 --object-width 1.8 --pixel-width 104"
    }
}
