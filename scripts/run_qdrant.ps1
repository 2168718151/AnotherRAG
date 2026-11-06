param([ValidateRange(1, 65534)][int]$Port = 6333)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$executable = Join-Path $projectRoot "tools\qdrant\qdrant.exe"
if (-not (Test-Path -LiteralPath $executable)) {
    throw "Run scripts\install_qdrant.ps1 first."
}
foreach ($servicePort in @($Port, ($Port + 1))) {
    if (Get-NetTCPConnection -LocalPort $servicePort -State Listen -ErrorAction SilentlyContinue) {
        throw "Port $servicePort is occupied. Check the existing service before starting Qdrant."
    }
}

$logs = Join-Path $projectRoot "logs"
$storage = Join-Path $projectRoot "data\qdrant"
New-Item -ItemType Directory -Force -Path $logs, $storage | Out-Null
$variables = @{
    QDRANT__SERVICE__HOST = "127.0.0.1"
    QDRANT__SERVICE__HTTP_PORT = "$Port"
    QDRANT__SERVICE__GRPC_PORT = "$($Port + 1)"
    QDRANT__STORAGE__STORAGE_PATH = $storage
    QDRANT__TELEMETRY_DISABLED = "true"
}
$previous = @{}
foreach ($name in $variables.Keys) {
    $previous[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
    [Environment]::SetEnvironmentVariable($name, $variables[$name], "Process")
}
try {
    $process = Start-Process -FilePath $executable -WorkingDirectory $projectRoot `
        -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $logs "qdrant.stdout.log") `
        -RedirectStandardError (Join-Path $logs "qdrant.stderr.log")
} finally {
    foreach ($name in $previous.Keys) {
        [Environment]::SetEnvironmentVariable($name, $previous[$name], "Process")
    }
}
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    if ($process.HasExited) {
        throw "Qdrant exited. See logs\qdrant.stderr.log and logs\qdrant.stdout.log."
    }
    try {
        $null = Invoke-RestMethod "http://127.0.0.1:$Port/readyz" -TimeoutSec 1
        Write-Host "Qdrant ready at http://127.0.0.1:$Port (PID $($process.Id))."
        Write-Host "Stop this instance with: Stop-Process -Id $($process.Id)"
        exit 0
    } catch {
        Start-Sleep -Milliseconds 500
    }
}
Stop-Process -Id $process.Id
throw "Qdrant did not become ready. See logs\qdrant.stdout.log."
