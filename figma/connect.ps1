[CmdletBinding()]
param(
    [switch]$Login
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$codex = Get-Command codex -ErrorAction Stop

# The override lets the check and login work before project trust is configured.
# It does not register a server in the global Codex configuration.
$serverConfig = 'mcp_servers.figma.url="https://mcp.figma.com/mcp"'

Push-Location $projectRoot
try {
    if ($Login) {
        & $codex.Source -c $serverConfig mcp login figma
        if ($LASTEXITCODE -ne 0) {
            throw "Figma OAuth login failed or was cancelled."
        }
    }

    $raw = & $codex.Source -c $serverConfig mcp list --json
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to inspect Codex MCP configuration."
    }

    $server = @($raw | ConvertFrom-Json) | Where-Object name -eq "figma"
    if (-not $server) {
        throw "Figma MCP configuration was not found."
    }

    [pscustomobject]@{
        Server = $server.name
        Url = $server.transport.url
        Enabled = $server.enabled
        AuthStatus = $server.auth_status
        ProjectConfig = Join-Path $projectRoot ".codex\config.toml"
        FigmaToolsVerified = $false
    }

    Write-Host "This checks configuration and credential presence, not live Figma tool access."
    if ($server.auth_status -eq "not_logged_in") {
        Write-Host "OAuth is pending. Run this script with -Login to sign in yourself."
    }
    Write-Host "Open this project in Codex and verify whoami before using Figma files."
}
finally {
    Pop-Location
}
