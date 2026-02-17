# 🚀 Quick Start Guide

**Complete step-by-step instructions to run the Predictive Maintenance Agent.**

> 💡 **New to the project?** Start here! This guide provides all commands needed to get up and running in 5 minutes.

---

## ✅ Prerequisites

- Windows PowerShell
- Docker Desktop installed and running (optional, but recommended)
- Python 3.10+ installed
- Git (optional, for cloning)

---

## 📥 Step 1: Get the Project

```powershell
# If from Git repository
git clone <repository-url>
cd predictive-agent

# OR if from ZIP file
# Extract ZIP, then:
cd predictive-agent
```

---

## 🐳 Step 2: Build Docker Image

```powershell
# Build the Docker image (takes 2-3 minutes first time)
docker build -f docker\Dockerfile -t predictive-agent-api .

# Expected output: "=> exporting to image"
# Image size: ~500 MB
```

---

## 🖥️ Step 3: Start API Server

**Option A: Docker (Recommended for Production)**

**Open Terminal 1** (PowerShell):

```powershell
# Start with MobileNetV3 (default - fast)
docker run --rm -p 8000:8000 --name predictive-agent-api predictive-agent-api

# OR start with CLIP model (better accuracy)
docker run --rm -p 8000:8000 -e RUST_MODEL_TYPE=clip --name predictive-agent-api predictive-agent-api
```

**Option B: Direct Python (Development)**

```powershell
# Start with default model (MobileNetV3)
uvicorn app.main:app --host 0.0.0.0 --port 8000

# OR specify CLIP model
$env:RUST_MODEL_TYPE="clip"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Expected Output:**
```
MAIN.PY LOADED - MULTIMODAL VERSION ✅
RUST_MODEL_TYPE = mobilenet
RUST_ONNX     = artifacts/rust_model.onnx
RUST_LABELS   = artifacts/rust_labels.json
✅ Loaded MOBILENET rust detection model: artifacts/rust_model.onnx
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

**✅ SUCCESS**: Leave this terminal running!

---

## 🎨 Step 4: Start UI Server

**Open Terminal 2** (PowerShell):

```powershell
# Navigate to UI folder
cd ui

# Start the UI server
python serve_ui.py
```

**Expected Output:**
```
🌐 Serving UI at http://localhost:8080/ui/ui.html
📁 Directory: C:\predictive-agent

✅ Open in browser: http://localhost:8080/ui/ui.html
⚠️  Make sure API is running at http://localhost:8000

Press Ctrl+C to stop
```

**✅ SUCCESS**: Leave this terminal running too!

---

## 🌐 Step 5: Open Web UI

**Open Terminal 3** (PowerShell):

```powershell
# Open UI in default browser
Start-Process "http://localhost:8080/ui/ui.html"
```

**OR** manually navigate to: **http://localhost:8080/ui/ui.html**

---

## 🧪 Step 6: Test the System

In the browser:

1. ✅ Verify "Connected" status (green circle) next to "Check Health"
2. Click **"▶ Run All Tests"** button
3. Watch as all 8 tests run and show results:
   - Health Check
   - Sensor Only
   - Environmental - Critical
   - Environmental - Favorable
   - Image - Rust Detected
   - Image - Clean Surface
   - Multimodal Complete
   - Batch Prediction

**Expected**: All tests show ✅ Success with prediction metrics!

---

## 🎉 Success Indicators

### API Server (Terminal 1)
```
INFO:     172.17.0.1:xxxxx - "POST /predict HTTP/1.1" 200 OK
```
✅ API responding to requests

### UI Server (Terminal 2)
```
127.0.0.1 - - [date/time] "GET /ui/ui.html HTTP/1.1" 200 -
127.0.0.1 - - [date/time] "GET /tests/images/rust/rust1.jpg HTTP/1.1" 200 -
```
✅ UI serving files correctly

### Browser
- ✅ Green "Connected" indicator
- ✅ All test cards showing success
- ✅ Predictions with metrics displayed
- ✅ Latency < 150ms per request

---

## 🛑 Stopping the System

### Stop UI Server (Terminal 2)
```powershell
# Press Ctrl+C
Ctrl+C
```

### Stop API Server (Terminal 1)
```powershell
# Press Ctrl+C (stops and removes container)
Ctrl+C
```

---

## 🔄 Restarting After Changes

### If you modified Python code:

**Terminal 1:**
```powershell
# Stop running container (Ctrl+C)

# Rebuild Docker image
docker build -t predictive-agent -f docker\Dockerfile .

# Start container again
docker run --rm -p 8000:8000 --name predictive-agent predictive-agent
```

### If you modified UI (ui.html):

**No restart needed!** Just refresh browser (Ctrl+R or F5)

---

## 🧪 Alternative: Test with Command Line

If you prefer testing without the UI:

```powershell
# Test health endpoint
Invoke-RestMethod "http://localhost:8000/health" | ConvertTo-Json

# Test single prediction
$body = Get-Content .\samples\request.json -Raw
Invoke-RestMethod -Uri "http://localhost:8000/predict" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body | ConvertTo-Json -Depth 10

# Test batch prediction
$body = Get-Content .\samples\batch.json -Raw
Invoke-RestMethod -Uri "http://localhost:8000/predict/batch" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body | ConvertTo-Json -Depth 10
```

---

## 🐛 Troubleshooting

### Issue: "Port 8000 already in use"

```powershell
# Find and stop the container
docker ps
docker stop predictive-agent

# Or stop all containers
docker stop $(docker ps -q)
```

---

### Issue: "Port 8080 already in use"

```powershell
# Find process on port 8080
$pid = (Get-NetTCPConnection -LocalPort 8080).OwningProcess
Stop-Process -Id $pid -Force
```

---

### Issue: "Docker daemon not running"

1. Open Docker Desktop
2. Wait for it to start (whale icon in system tray)
3. Try again

---

### Issue: "Failed to fetch" in UI

**Check API is running:**
```powershell
docker ps | Select-String "predictive-agent"
```

**If not running, restart it:**
```powershell
docker run --rm -p 8000:8000 --name predictive-agent predictive-agent
```

---

### Issue: Python not found

**Install Python 3.10+** from: https://www.python.org/downloads/

**Verify installation:**
```powershell
python --version
# Should show: Python 3.10.x or higher
```

---

## 📊 What Each Test Does

| Test | What It Tests | Expected Result |
|------|--------------|----------------|
| **Health Check** | API is running, models loaded | Status: ok |
| **Sensor Only** | Basic LightGBM prediction | Low failure probability |
| **Env - Critical** | High-risk environmental conditions | Environmental stress fault |
| **Env - Favorable** | Normal operating conditions | Reduced risk |
| **Image - Rust** | Rust detection + fusion | Corrosion/rust fault detected |
| **Image - Clean** | No rust detection | Below fusion threshold |
| **Multimodal** | All 3 modalities combined | Multiple fault signals |
| **Batch** | Process 3 assets at once | 3 predictions returned |

---

## 🎯 Next Steps

Once everything works:

1. ✅ Review predictions and metrics
2. ✅ Check latency (should be < 150ms)
3. ✅ Try modifying sample requests
4. ✅ Read documentation:
   - [README.md](../README.md) - Full documentation
   - [ARCHITECTURE.md](../ARCHITECTURE.md) - System design
   - [EVALUATION_REPORT.md](../EVALUATION_REPORT.md) - Model performance

---

## 📞 Need Help?

- **Check Logs**: Look at Terminal 1 (API) and Terminal 2 (UI) for errors
- **Browser Console**: Press F12 in browser, check Console tab
- **Docker Logs**: `docker logs predictive-agent`
- **Verify Ports**: `Get-NetTCPConnection -LocalPort 8000,8080`

---

## ✅ Complete Command Reference

### Full Startup (3 Terminals)

**Terminal 1:**
```powershell
cd C:\predictive-agent
docker build -t predictive-agent -f docker\Dockerfile .
docker run --rm -p 8000:8000 --name predictive-agent predictive-agent
```

**Terminal 2:**
```powershell
cd C:\predictive-agent\ui
python serve_ui.py
```

**Terminal 3:**
```powershell
Start-Process "http://localhost:8080/ui/ui.html"
```

**That's it! 🎉**

---

