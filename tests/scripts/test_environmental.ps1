# Test Environmental Modality
# Demonstrates how environmental/transactional data affects predictions

Write-Host "`n=== Testing Environmental Modality ===" -ForegroundColor Cyan

$endpoint = "http://localhost:8000/predict"

# Test 1: Favorable conditions (low risk)
Write-Host "`n--- Test 1: Well-maintained pump, favorable conditions ---" -ForegroundColor Green
$body1 = Get-Content ".\samples\env_favorable.json" -Raw
$resp1 = Invoke-RestMethod -Uri $endpoint -Method Post -ContentType "application/json" -Body $body1
Write-Host "Asset: $($resp1.asset_id)"
Write-Host "Failure Probability: $($resp1.failure_probability)"
Write-Host "Fault Type: $($resp1.predicted_fault_type)"
Write-Host "Top Signals:"
$resp1.top_signals | ForEach-Object { Write-Host "  - $_" }

# Test 2: Normal conditions
Write-Host "`n--- Test 2: Normal operating conditions ---" -ForegroundColor Yellow
$body2 = Get-Content ".\samples\request.json" -Raw
$resp2 = Invoke-RestMethod -Uri $endpoint -Method Post -ContentType "application/json" -Body $body2
Write-Host "Asset: $($resp2.asset_id)"
Write-Host "Failure Probability: $($resp2.failure_probability)"
Write-Host "Fault Type: $($resp2.predicted_fault_type)"
Write-Host "Top Signals:"
$resp2.top_signals | ForEach-Object { Write-Host "  - $_" }

# Test 3: Critical conditions (high risk)
Write-Host "`n--- Test 3: Critical environmental conditions ---" -ForegroundColor Red
$body3 = Get-Content ".\samples\env_critical.json" -Raw
$resp3 = Invoke-RestMethod -Uri $endpoint -Method Post -ContentType "application/json" -Body $body3
Write-Host "Asset: $($resp3.asset_id)"
Write-Host "Failure Probability: $($resp3.failure_probability)"
Write-Host "Fault Type: $($resp3.predicted_fault_type)"
Write-Host "Top Signals:"
$resp3.top_signals | ForEach-Object { Write-Host "  - $_" }

# Summary comparison
Write-Host "`n=== Summary Comparison ===" -ForegroundColor Cyan
Write-Host ("Well-maintained:  p_fail = {0:E6} (multiplier = {1:N2}x)" -f $resp1.failure_probability, 1.0)
Write-Host ("Normal:           p_fail = {0:E6} (multiplier = {1:N2}x)" -f $resp2.failure_probability, 1.1)
Write-Host ("Critical:         p_fail = {0:E6} (multiplier = {1:N2}x)" -f $resp3.failure_probability, 2.0)
Write-Host "`nNote: Base sensor p_fail is very low (~1.3e-7). Environmental modality doubles it to 2.7e-7." -ForegroundColor Yellow
Write-Host "For dramatic changes, combine with image modality (see test_multimodal_complete.ps1)" -ForegroundColor Yellow
