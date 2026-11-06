$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required. Install it before running this script."
}

uv sync --group dev
uv lock

$directories = @("data\qdrant", "data\docstore", "logs", "logs\traces", "reports")
foreach ($directory in $directories) {
    New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot $directory) | Out-Null
}

if (-not (Test-Path -LiteralPath (Join-Path $projectRoot ".git"))) {
    git init
}

Write-Host "Bootstrap complete." # noqa: PSUseWriteHost
