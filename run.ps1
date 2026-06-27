# run.ps1 — SkyRoute launcher
#
# SAFE TO COMMIT: this script contains no secrets.
# All API keys live in .env (gitignored). Copy .env.example → .env and fill in your keys.
#
# Usage:
#   .\run.ps1            # start the Streamlit app
#   .\run.ps1 --check    # verify .env keys without starting the app

param([switch]$check)

$envFile = Join-Path $PSScriptRoot ".env"

# ── Load .env into this process ───────────────────────────────────────────────
if (-not (Test-Path $envFile)) {
    Write-Warning ".env not found. Run:  Copy-Item .env.example .env  then fill in your keys."
    exit 1
}

$loaded = @()
foreach ($line in Get-Content $envFile) {
    # Skip blank lines and comments
    if ([string]::IsNullOrWhiteSpace($line) -or $line.TrimStart().StartsWith("#")) { continue }
    if ($line -match "^([^=]+)=(.*)$") {
        $k = $Matches[1].Trim()
        $v = $Matches[2].Trim()
        if ($v) {
            [System.Environment]::SetEnvironmentVariable($k, $v, "Process")
            $loaded += $k
        }
    }
}

Write-Host "Loaded from .env: $($loaded -join ', ')" -ForegroundColor DarkGray

# ── Key presence report ───────────────────────────────────────────────────────
$required = @{
    "TOMTOM_API_KEY" = "TomTom Routing (traffic-aware ground routes — free 2500/day, no credit card)"
    "GROQ_API_KEY"   = "Groq LLM API (Route Assistant natural language — free 14400/day, no credit card)"
}
$optional = @{
    "ANTHROPIC_API_KEY" = "Claude API (LLM validation — Phase 2 HIE)"
    "FAA_API_KEY"        = "FAA NOTAM live feed"
    "MAPBOX_TOKEN"       = "Enhanced map tiles"
}

$missing = @()
foreach ($k in $required.Keys) {
    $val = [System.Environment]::GetEnvironmentVariable($k, "Process")
    if ($val) {
        Write-Host "  [OK]      $k" -ForegroundColor Green
    } else {
        Write-Host "  [MISSING] $k  — $($required[$k])" -ForegroundColor Red
        $missing += $k
    }
}
foreach ($k in $optional.Keys) {
    $val = [System.Environment]::GetEnvironmentVariable($k, "Process")
    $status = if ($val) { "[OK]     " } else { "[not set]" }
    $color  = if ($val) { "Green"    } else { "DarkGray" }
    Write-Host "  $status $k  — $($optional[$k])" -ForegroundColor $color
}

if ($check) { exit ($missing.Count -gt 0 ? 1 : 0) }

if ($missing.Count -gt 0) {
    Write-Warning "Required keys missing — the app will fall back to OSRM (no traffic). Add them to .env to enable full functionality."
}

# ── Launch ────────────────────────────────────────────────────────────────────
Write-Host "`nStarting SkyRoute..." -ForegroundColor Cyan
streamlit run app.py
