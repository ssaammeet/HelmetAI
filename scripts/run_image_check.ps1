param(
    [Parameter(Mandatory=$true)][string]$Image,
    [ValidateSet('front','rear')][string]$Direction = 'front',
    [string]$OutputDir = '',
    [string]$Model = '',
    [ValidateSet(0,90,180,270)][int]$Rotate = 0,
    [ValidateRange(1,4)][int]$CpuThreads = 3
)
. (Join-Path $PSScriptRoot 'desktop_common.ps1')
$imagePath = Resolve-HelmetInput $Image
if (-not $Model) { $Model = Join-Path $script:HelmetProjectRoot 'models\detector.onnx' }
$modelPath = Resolve-HelmetInput $Model
if (-not $OutputDir) { $OutputDir = Get-HelmetNewResultPath 'image' }
$OutputDir = Get-HelmetOutputPath $OutputDir
Invoke-HelmetPython @('-m','helmetai_fcw.image_check','--image',$imagePath,'--model',$modelPath,
    '--direction',$Direction,'--rotate',"$Rotate",'--cpu-threads',"$CpuThreads",'--output-dir',$OutputDir)
Write-Host "Results: $OutputDir"
Write-Host 'Open annotated.jpg to inspect the boxes. Detection confidence is NOT accuracy or collision probability.'
