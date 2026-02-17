# 🎨 Predictive Maintenance Agent - Test UI

**Simple web-based testing interface for the Predictive Maintenance API**

This UI provides a visual way to test all API endpoints and multimodal capabilities without writing code.

---

## 📋 Features

- ✅ **9 Pre-configured Test Cases** - All modality combinations ready to run
- ✅ **VLM Testing** - Compare MobileNetV3 vs CLIP vision-language model
- ✅ **Visual Results Display** - See predictions, metrics, and explanations instantly
- ✅ **Image Preview** - View uploaded images alongside rust detection results
- ✅ **Batch Testing** - Run all tests sequentially or individually
- ✅ **Live Health Check** - Real-time API connection status
- ✅ **No Installation Required** - Pure HTML/JavaScript frontend

---

## 🚀 Quick Start

### Prerequisites

1. **Docker** installed and running
2. **Python 3.10+** installed

### Step 1: Start the API Server

Open a terminal and run:

```powershell
# Navigate to project root
cd C:\predictive-agent

# Build Docker image (if not already built)
docker build -f docker\Dockerfile -t predictive-agent-api .

# Start API container with MobileNetV3 (default - fast)
docker run --rm -p 8000:8000 --name predictive-agent-api predictive-agent-api

# OR with CLIP model (better accuracy)
docker run --rm -p 8000:8000 -e RUST_MODEL_TYPE=clip --name predictive-agent-api predictive-agent-api

# OR run directly with Python (no Docker)
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Expected Output:**
```
MAIN.PY LOADED - MULTIMODAL VERSION ✅
RUST_MODEL_TYPE = mobilenet
✅ Loaded MOBILENET rust detection model: artifacts/rust_model.onnx
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

Leave this terminal running.

---

### Step 2: Start the UI Server

Open a **NEW terminal** and run:

```powershell
# Navigate to UI folder
cd C:\predictive-agent\ui

# Start UI server
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

---

### Step 3: Open in Browser

Navigate to: **http://localhost:8080/ui/ui.html**

Or run from terminal:
```powershell
Start-Process "http://localhost:8080/ui/ui.html"
```

---

## 🧪 Available Test Cases

| Test | Modalities | Description |
|------|-----------|-------------|
| **Health Check** | - | Verify API is running and models are loaded |
| **Sensor Only** | 📊 Sensor | Basic prediction using sensor data only |
| **Environmental - Critical** | 📊 Sensor + 🌡️ Env | High-risk conditions (overdue maintenance, extreme temp) |
| **Environmental - Favorable** | 📊 Sensor + 🌡️ Env | Normal operating conditions with recent maintenance |
| **Image - Rust Detected** | 📊 Sensor + 🖼️ Image | Visual inspection detects corrosion |
| **Image - Clean Surface** | 📊 Sensor + 🖼️ Image | No rust detected, below fusion threshold |
| **Multimodal Complete** | 📊 Sensor + 🖼️ Image + 🌡️ Env | All three modalities combined |
| **VLM Multimodal (CLIP)** | 📊 Sensor + 🖼️ Image + 🌡️ Env + 🧠 VLM | Test CLIP vision-language model with all modalities (demonstrates false positive issue) |
| **Batch Prediction** | 📦 Batch | Process multiple assets in one request |

---

## 🎯 How to Use

1. **Check Connection**: Click "Check Health" button to verify API is running ✅
2. **Run Individual Tests**: Click "Run Test" on any test card
3. **Run All Tests**: Click the green "▶ Run All Tests" button at the top
4. **View Results**: Each test shows:
   - ✅ Success with prediction metrics
   - ❌ Error with detailed message
   - ⏳ Loading state while processing

---

## 📊 Understanding Results

Each prediction shows:

- **Failure Probability**: 0-100% likelihood of pump failure
- **Time to Breakdown (TTB)**: Estimated hours until failure
- **Fault Type**: Predicted fault category (if any)
  - `bearing_failure`
  - `seal_leak`
  - `corrosion_rust`
  - `environmental_stress`
- **Fault Confidence**: Model confidence in fault prediction
- **Latency**: Response time in milliseconds
- **Top Signals**: Key contributing factors to the prediction
- **Explanation**: Human-readable reasoning for the prediction

---

## 🧠 VLM Test Case Explained

The **VLM Multimodal (CLIP)** test demonstrates vision-language model capabilities:

**What it tests:**
- Uses a **clean pump image** (no rust) with normal sensor + environmental data
- Tests CLIP model's ability to correctly classify clean surfaces
- Compares against MobileNetV3 baseline performance

**Expected behavior:**
- **With MobileNetV3 (default)**: Correctly identifies as `no_rust` (~70-80%)
- **With CLIP (VLM)**: May show **false positive** - incorrectly predicts rust (77-99%)

**Why this matters:**
This test showcases why MobileNetV3 is the default production model. While CLIP is more sophisticated (335 MB vs 6 MB), it suffers from training data imbalance causing false positives on clean pumps.

**To test CLIP model:**
1. Stop current API server
2. Start with CLIP: `docker run --rm -p 8000:8000 -e RUST_MODEL_TYPE=clip --name predictive-agent-api predictive-agent-api`
3. Run the VLM test case to see the false positive

See [VLM_EVALUATION.md](../docs/VLM_EVALUATION.md) for full comparison and test results.

---

## 🛠️ Troubleshooting

### Issue: "Failed to fetch"

**Cause**: API server not running or CORS not configured

**Solution**:
```powershell
# 1. Check if API is running
docker ps | Select-String "predictive-agent"

# 2. If not running, start it
docker run --rm -p 8000:8000 --name predictive-agent-api predictive-agent-api

# 3. Test API directly
Invoke-RestMethod "http://localhost:8000/health" | ConvertTo-Json
```

---

### Issue: "Port 8080 already in use"

**Cause**: UI server already running

**Solution**:
```powershell
# Find process using port 8080
$pid = (Get-NetTCPConnection -LocalPort 8080).OwningProcess
Stop-Process -Id $pid -Force

# Then restart UI server
python serve_ui.py
```

---

### Issue: Images not loading

**Cause**: Server not serving from correct directory

**Solution**: Run `serve_ui.py` from the `ui/` folder, not the root:
```powershell
cd C:\predictive-agent\ui
python serve_ui.py
```

---

### Issue: Errors showing "Cannot read properties of undefined"

**Cause**: API response format doesn't match UI expectations

**Solution**: 
1. Check API logs for actual errors
2. Verify you're running the latest Docker image:
```powershell
docker stop predictive-agent-api
docker build -t predictive-agent-api -f docker\Dockerfile .
docker run --rm -p 8000:8000 --name predictive-agent-api predictive-agent-api
```

---

## 🔧 Configuration

### Change API URL

If your API is running on a different host/port, update it in the UI:

1. Open the UI in browser
2. Find "API Configuration" section at the top
3. Change the "Base URL" field
4. Click "Check Health" to verify

---

## 📁 Files in This Folder

```
ui/
├── ui.html          # Main UI file (HTML + CSS + JavaScript)
├── serve_ui.py      # Simple Python HTTP server with CORS
└── README.md        # This file
```

**Note**: The UI requires access to parent directory files:
- `tests/images/` - Sample images for rust detection tests
- `samples/batch.json` - Sample batch prediction payload

---

## 🎓 Technical Details

### Architecture

```
Browser (localhost:8080)
    ↓
Python HTTP Server (serve_ui.py)
    ↓
Static Files (ui.html, tests/images/, etc.)
    ↓
JavaScript Fetch API
    ↓
FastAPI Server (localhost:8000)
    ↓
ML Models (LightGBM, ONNX, Environmental)
```

### CORS Configuration

The API includes CORS middleware to allow cross-origin requests from `localhost:8080` → `localhost:8000`. This is configured in `app/main.py`:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Security Note

⚠️ **Development Only**: The CORS configuration allows all origins (`*`). In production, restrict to specific domains:

```python
allow_origins=["https://yourdomain.com"]
```

---

## 📝 Sample Commands (Complete Workflow)

### Terminal 1 - API Server
```powershell
cd C:\predictive-agent
docker build -t predictive-agent-api -f docker\Dockerfile .
docker run --rm -p 8000:8000 --name predictive-agent-api predictive-agent-api
```

### Terminal 2 - UI Server
```powershell
cd C:\predictive-agent\ui
python serve_ui.py
```

### Terminal 3 - Open Browser & Test
```powershell
Start-Process "http://localhost:8080/ui/ui.html"
```

---

## 🎉 Success Indicators

✅ **API Ready**: Docker logs show "Application startup complete"  
✅ **UI Ready**: Browser shows green "Connected" status  
✅ **Tests Pass**: All 8 tests return success with predictions  
✅ **Fast Response**: Latency < 100ms for most requests  

---

## 📞 Support

If you encounter issues:

1. Check Docker container logs:
   ```powershell
   docker logs predictive-agent-api
   ```

2. Check UI server logs in the terminal running `serve_ui.py`

3. Open browser DevTools (F12) → Console tab for JavaScript errors

4. Verify ports are free:
   ```powershell
   Get-NetTCPConnection -LocalPort 8000,8080 -State Listen
   ```

---

**Happy Testing! 🚀**
