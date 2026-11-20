param(
    [string]$Url = "https://192.168.1.237/v1/predict",
    [string]$CaCert = ".\velov.crt",
    [int]$Duration = 20
)

$ErrorActionPreference = "Stop"
$outDir = Join-Path $PSScriptRoot "results\w1"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

foreach ($users in 10, 25, 50, 100, 150) {
    Write-Host "Palier $users utilisateurs..."
    python (Join-Path $PSScriptRoot "run_paliers.py") `
        --url $Url --cacert $CaCert --users $users --duration $Duration `
        --out (Join-Path $outDir "w$users.jsonl")
}

python (Join-Path $PSScriptRoot "summary.py") `
    (Get-ChildItem $outDir -Filter "w*.jsonl" | Sort-Object Name | ForEach-Object { $_.FullName })
