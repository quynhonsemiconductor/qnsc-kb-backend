param(
    [Parameter(Mandatory=$true)][string]$DatabaseBackup,
    [string]$SourceArchive,
    [string]$PostgresUser = $env:POSTGRES_ADMIN_USER,
    [string]$PostgresDb = $env:POSTGRES_DB
)
if (-not $PostgresUser) { $PostgresUser = "postgres" }
if (-not $PostgresDb)   { $PostgresDb = "qnsc_kb" }

$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $DatabaseBackup)) { throw "Database backup not found: $DatabaseBackup" }

$confirmation = Read-Host "This replaces the current database ($PostgresDb). Type RESTORE to continue"
if ($confirmation -ne "RESTORE") { throw "Restore cancelled." }

Write-Host "Restoring PostgreSQL ($PostgresDb) from $DatabaseBackup"
# Pipe raw bytes: Get-Content -Raw re-encodes under Windows PowerShell 5.1 and
# can re-introduce a BOM mid-stream.
cmd /c "type `"$DatabaseBackup`"" | docker compose exec -T db psql -U $PostgresUser -d $PostgresDb
if ($LASTEXITCODE -ne 0) { throw "psql restore failed." }

if ($SourceArchive) {
    if (-not (Test-Path -LiteralPath $SourceArchive)) { throw "Source archive not found: $SourceArchive" }
    $volume = (docker volume ls --format '{{.Name}}' | Where-Object { $_ -like '*_connector_data' } | Select-Object -First 1)
    if (-not $volume) { throw "Connector volume was not found." }
    docker run --rm -v "${volume}:/source" -v "$(Resolve-Path $SourceArchive):/backup/archive.tar.gz:ro" alpine:3.20 sh -c "rm -rf /source/* && tar -xzf /backup/archive.tar.gz -C /source"
}

Write-Host "Restore completed. Verify /health/ready and a permission-checked read before returning the stack to service."
