# Test all 3 modalities together
Write-Host "`n=== Testing Full Multimodal Fusion ===" -ForegroundColor Cyan

# Get script directory to build proper paths
$scriptDir = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $scriptDir) { $scriptDir = Get-Location }

# Test 1: Rust + Critical Environment
Write-Host "`n--- Rust Image + Critical Environmental Conditions ---" -ForegroundColor Red
$imagePath = Join-Path $scriptDir "tests\images\rust\rust1.jpg"
$b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($imagePath))
$req = Get-Content (Join-Path $scriptDir "samples\env_critical.json") -Raw | ConvertFrom-Json
$bodyObj = @{
  asset_id = "multimodal_rust_critical"
  timestamp = (Get-Date).ToString("o")
  sensor_window = $req.sensor_window
  image_base64 = $b64
  environmental = $req.environmental
}
$body = $bodyObj | ConvertTo-Json -Depth 50
$resp1 = Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -Body $body
$resp1 | ConvertTo-Json -Depth 10

# Test 2: No Rust + Favorable Environment
Write-Host "`n--- Clean Image + Favorable Environmental Conditions ---" -ForegroundColor Green
$imagePath = Join-Path $scriptDir "tests\images\no_rust\clean1.jpg"
$b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($imagePath))
$req = Get-Content (Join-Path $scriptDir "samples\env_favorable.json") -Raw | ConvertFrom-Json
$bodyObj = @{
  asset_id = "multimodal_clean_favorable"
  timestamp = (Get-Date).ToString("o")
  sensor_window = $req.sensor_window
  image_base64 = $b64
  environmental = $req.environmental
}
$body = $bodyObj | ConvertTo-Json -Depth 50
$resp2 = Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -Body $body
$resp2 | ConvertTo-Json -Depth 10

Write-Host "`n=== Comparison ===" -ForegroundColor Cyan
Write-Host ("Rust + Critical:      p_fail = {0:N6}, fault = {1}" -f $resp1.failure_probability, $resp1.predicted_fault_type)
Write-Host ("Clean + Favorable:    p_fail = {0:N6}, fault = {1}" -f $resp2.failure_probability, $resp2.predicted_fault_type)
