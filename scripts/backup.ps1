param(
    [string]$OutputDirectory = "./backups",
    # Override to match the deployment: dev compose uses postgres/qnsc_kb.
    [string]$PostgresUser = $env:POSTGRES_ADMIN_USER,
    [string]$PostgresDb = $env:POSTGRES_DB
)
if (-not $PostgresUser) { $PostgresUser = "postgres" }
if (-not $PostgresDb)   { $PostgresDb = "qnsc_kb" }

$ErrorActionPreference = "Stop"
$resolvedOutput = (New-Item -ItemType Directory -Force -Path $OutputDirectory).FullName
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$dbFile = Join-Path $resolvedOutput "qnsc_kb-$stamp.sql"
$sourceArchive = Join-Path $resolvedOutput "sources-$stamp.tar.gz"

Write-Host "Backing up PostgreSQL ($PostgresDb) to $dbFile"
# -Encoding utf8NoBOM: Windows PowerShell 5.1's plain utf8 emits a BOM which
# psql rejects on the first statement during restore.
docker compose exec -T db pg_dump -U $PostgresUser -d $PostgresDb --format=plain --no-owner --no-privileges > $dbFile
if ($LASTEXITCODE -ne 0) { throw "pg_dump failed." }

# The local connector volume is the only file storage that lives on this host;
# article sources live in Cloudflare R2, which must be backed up separately
# (lifecycle rules or `rclone sync` against the bucket).
$volume = (docker volume ls --format '{{.Name}}' | Where-Object { $_ -like '*_connector_data' } | Select-Object -First 1)
if ($volume) {
    Write-Host "Backing up connector volume $volume to $sourceArchive"
    docker run --rm -v "${volume}:/source:ro" -v "${resolvedOutput}:/backup" alpine:3.20 tar -czf "/backup/$(Split-Path -Leaf $sourceArchive)" -C /source .
} else {
    Write-Warning "Connector volume not found; database backup completed without local file storage."
}

# Retention: keep the newest 14 backup sets so ad-hoc runs cannot fill the disk.
Get-ChildItem -LiteralPath $resolvedOutput -Filter "qnsc_kb-*.sql" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -Skip 14 |
    Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem -LiteralPath $resolvedOutput -Filter "sources-*.tar.gz" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -Skip 14 |
    Remove-Item -Force -ErrorAction SilentlyContinue

Write-Host "Backup completed. Copy archives off-host (encrypted); production sources in R2 need a separate bucket-level backup."
