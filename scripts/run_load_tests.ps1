# Load test the API behind nginx with 1, 2 and 3 replicas. Requires: pip install locust
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

foreach ($n in 1, 2, 3) {
    Write-Host "=== $n replica(s) ==="
    docker compose down          # nginx resolves the api replicas at startup, so restart the stack
    docker compose up -d --build --scale api=$n

    $ok = $false
    for ($i = 0; $i -lt 90 -and -not $ok; $i++) {
        try { $ok = (Invoke-WebRequest "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200 }
        catch { Start-Sleep -Seconds 2 }
    }
    if (-not $ok) { throw "API did not become healthy with $n replica(s)" }

    $dir = "results/locust_$n"
    New-Item -ItemType Directory -Force $dir | Out-Null
    python -m locust -f locustfile.py --headless --host http://localhost:8000 `
        -u 50 -r 10 -t 1m --csv "$dir/locust" --only-summary
}
docker compose down
python scripts/summarize_results.py
