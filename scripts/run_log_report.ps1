param(
    [Parameter(Mandatory=$true)][string]$Runtime,
    [string]$Before = '',
    [string]$After = '',
    [string]$OutputDir = ''
)
. (Join-Path $PSScriptRoot 'desktop_common.ps1')
$runtimePath = Resolve-HelmetInput $Runtime
if (-not $OutputDir) { $OutputDir = Get-HelmetNewResultPath 'logs' }
$OutputDir = Get-HelmetOutputPath $OutputDir
$moduleArguments = @('-m','helmetai_fcw.log_report','--runtime',$runtimePath,'--output-dir',$OutputDir)
if ($Before) { $moduleArguments += @('--before',(Resolve-HelmetInput $Before)) }
if ($After) { $moduleArguments += @('--after',(Resolve-HelmetInput $After)) }
Invoke-HelmetPython $moduleArguments
Write-Host "Results: $OutputDir"
Write-Host 'A completed run is not a passed safety/accuracy test. See summary.md.'
