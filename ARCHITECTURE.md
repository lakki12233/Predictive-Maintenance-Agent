# System Architecture

## Overview

The system predicts pump failures using 3 data sources (modalities):
1. **Sensor data** → LightGBM ML models
2. **Images** → ONNX rust detection (MobileNetV3 or CLIP, switchable)
3. **Environmental data** → Rule-based risk scoring

---

## System Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                       Client Request                        │
│   {sensor_window, image_base64?, environmental?}            │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Server (:8000)                    │
│   /health  │  /predict  │  /predict/batch                   │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  Multimodal Orchestrator                    │
└───────┬─────────────────┬─────────────────┬─────────────────┘
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│ Sensor Module │ │ Image Module  │ │  Env Module   │
│   (LightGBM)  │ │    (ONNX)     │ │ (Rule-based)  │
└───────┬───────┘ │ MobileNetV3 or│ └───────┬───────┘
        │         │     CLIP      │         │
        │         └───────┬───────┘         │
        │                 │                 │
        └─────────────────┼─────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     Fusion Engine                           │
│   p_fail = sensor_prob + rust_boost × env_multiplier        │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      JSON Response                          │
│   {failure_probability, fault_type, top_signals, ...}       │
└─────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Sensor Module
- **Input**: Time-series sensor readings (5-30 samples)
- **Process**: Extract features (mean, std, min, max, slope) → LightGBM prediction
- **Output**: Failure probability, time-to-breakdown, fault type

### 2. Image Module
- **Input**: Base64-encoded image (JPEG/PNG)
- **Process**: Decode → Resize to 640x640 → ONNX inference
- **Models Available**:
  - **MobileNetV3** (Default): 6 MB, 19-55ms inference, 100% test accuracy
  - **CLIP**: 335 MB, 64-127ms inference, 70% test accuracy (not recommended)
- **Switchable**: Set `RUST_MODEL_TYPE` env variable (mobilenet|clip)
- **Output**: "rust" or "no_rust" with confidence score

### 3. Environmental Module
- **Input**: Operating hours, maintenance days, temperature, humidity, load
- **Process**: Apply rule-based thresholds
- **Output**: Risk multiplier (0.5x to 2.0x)

### 4. Fusion Engine
Combines all modality outputs:
```
1. Start with sensor failure probability
2. If rust detected (>85% confidence): add 25% boost
3. Multiply by environmental risk factor
4. Clamp result to 0-100%
5. Assign fault type based on strongest signal
```

---

## Files

| File | Purpose |
|------|---------|
| `app/main.py` | API endpoints + fusion logic |
| `app/modalities/sensor.py` | Sensor inference |
| `app/modalities/image.py` | ONNX rust detection (model switching logic) |
| `app/modalities/environmental.py` | Risk scoring |
| `artifacts/*.joblib` | LightGBM models |
| `artifacts/rust_model.onnx` | MobileNetV3 image model (default, 6 MB) |
| `artifacts/rust_clip.onnx` | CLIP image model (experimental, 335 MB) |

---

## Performance

| Metric | Value |
|--------|-------|
| Throughput | ~100 requests/second |
| Latency (p50) | ~88 ms (with MobileNetV3) |
| Latency (p95) | ~147 ms |
| Memory per worker | ~500 MB (MobileNetV3) / ~1.2 GB (CLIP) |

**Bottleneck**: Image inference
- **MobileNetV3**: 42ms average (52% of total time)
- **CLIP**: 100ms average (would increase total latency significantly)
- Can speed up with INT8 quantization or GPU acceleration

---

## Deployment

**Current**: Docker container running Uvicorn (4 workers)

```powershell
# Default (MobileNetV3)
docker build -t oxmaint-api -f docker/Dockerfile .
docker run -p 8000:8000 oxmaint-api

# With CLIP model
docker run -p 8000:8000 -e RUST_MODEL_TYPE=clip oxmaint-api
```

**Production recommendation**: AWS ECS or GCP Cloud Run with auto-scaling

---


