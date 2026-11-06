param(
    [string]$Database,
    [string]$DatabaseId,
    [string]$PostgresSchema = "story_rag"
)

$ErrorActionPreference = "Stop"
if ($Database -and $DatabaseId) { throw "Specify Database or DatabaseId, not both" }

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

if (-not $env:STORY_RAG_POSTGRES_DSN) {
    $dsnPath = Join-Path $projectRoot "data\postgres\session-url.txt"
    if (Test-Path -LiteralPath $dsnPath) {
        $env:STORY_RAG_POSTGRES_DSN = (Get-Content -LiteralPath $dsnPath -Raw -Encoding UTF8).Trim()
    }
}
if ($env:STORY_RAG_POSTGRES_DSN) {
    $env:STORY_RAG_POSTGRES_DSN = $env:STORY_RAG_POSTGRES_DSN.Replace(
        "postgresql+asyncpg://", "postgresql://"
    )
}
$env:STORY_RAG_POSTGRES_SCHEMA = $PostgresSchema

$arguments = @('run', 'story-rag-api')
if ($Database) { $arguments += @('--db', $Database) }
if ($DatabaseId) { $arguments += @('--db-id', $DatabaseId) }
uv @arguments
exit $LASTEXITCODE
