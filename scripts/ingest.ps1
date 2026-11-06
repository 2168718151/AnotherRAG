param(
    [Parameter(Mandatory = $true)]
    [string]$Path,
    [string]$Database,
    [string]$DatabaseId
)

$ErrorActionPreference = "Stop"
if (([bool]$Database) -eq ([bool]$DatabaseId)) {
    throw "Specify exactly one of Database and DatabaseId"
}
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$arguments = @('run', 'story-rag', 'ingest', '--path', $Path)
if ($Database) { $arguments += @('--db', $Database) }
if ($DatabaseId) { $arguments += @('--db-id', $DatabaseId) }
uv @arguments
exit $LASTEXITCODE
