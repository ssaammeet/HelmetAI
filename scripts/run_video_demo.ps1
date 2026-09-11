param(
    [Parameter(Mandatory = $true)]
    [string]$Video,
    [ValidateSet("front", "rear")]
    [string]$Direction = "front",
    [string]$Output = "",
    [switch]$DemoVisualAlerts,
    [string]$ReportDir = '',
    [ValidateRange(0,100000)][int]$MaxFrames = 300,
    [ValidateRange(0,86400)][double]$StartSeconds = 0,
    [ValidateRange(1,4)][int]$CpuThreads = 3,
    [switch]$NoPreview
)

. (Join-Path $PSScriptRoot 'desktop_common.ps1')
$videoPath = Resolve-HelmetInput $Video
if ($DemoVisualAlerts -and $Direction -ne 'front') {
    throw 'The existing illustrative visual-alert profile is front-only. Omit -DemoVisualAlerts for rear evidence.'
}

if ($DemoVisualAlerts -and $Direction -eq "front") {
    $config = "config\webcam_video_demo_unverified.json"
    Write-Host 'ILLUSTRATIVE DEMO ONLY: not validated metric distance, TTC or road safety.'
} else {
    $config = if ($Direction -eq "rear") { "config\webcam_rear.example.json" } else { "config\webcam.example.json" }
}
$config = Join-Path $script:HelmetProjectRoot $config
if (-not $ReportDir) { $ReportDir = Get-HelmetNewResultPath 'video' }
$ReportDir = Get-HelmetOutputPath $ReportDir
if (-not $Output) { $Output = "$ReportDir-annotated.mp4" }
$Output = Get-HelmetOutputPath $Output
$startSecondsText = $StartSeconds.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$moduleArguments = @('-m','helmetai_fcw','video-run','--config',$config,'--video',$videoPath,
    '--output',$Output,'--report-dir',$ReportDir,'--start-seconds',$startSecondsText,'--cpu-threads',"$CpuThreads")
$moduleArguments += @('--max-frames',"$MaxFrames")
if ($NoPreview) { $moduleArguments += '--no-preview' }
Invoke-HelmetPython $moduleArguments
Write-Host "Annotated video: $Output"
Write-Host "Evidence report: $ReportDir"
Write-Host 'The default limit is 300 processed frames. Use -MaxFrames 0 only when you want the entire file.'
