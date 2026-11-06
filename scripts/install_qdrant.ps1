$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$installDir = Join-Path $projectRoot "tools\qdrant"
$executable = Join-Path $installDir "qdrant.exe"
$version = "1.19.2"
$sha256 = "7d86596f16c6e85d45a50312f5e16ccb51e059b62e3d7308110cb65b4c799a4d"

if (Test-Path -LiteralPath $executable) {
    & $executable --version
    exit 0
}

New-Item -ItemType Directory -Force -Path $installDir | Out-Null
$archive = Join-Path $installDir "qdrant-$version.zip"
$url = "https://github.com/qdrant/qdrant/releases/download/v$version/qdrant-x86_64-pc-windows-msvc.zip"
Invoke-WebRequest -Uri $url -OutFile $archive
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $sha256) {
    throw "Qdrant archive SHA256 does not match the official release digest."
}
Expand-Archive -LiteralPath $archive -DestinationPath $installDir
if (-not (Test-Path -LiteralPath $executable)) {
    throw "Qdrant executable was not found in the release archive."
}
& $executable --version
