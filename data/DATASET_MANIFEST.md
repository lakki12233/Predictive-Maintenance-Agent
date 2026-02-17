# Dataset Manifest

**Overview of all datasets used in the Predictive Maintenance Agent.**

---

## Summary

| Modality | Source | License | Type | Status |
|----------|--------|---------|------|--------|
| **Sensor** | [Kaggle](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data) | CC0 Public Domain | Real industrial data | ✅ Used |
| **Image** | [Roboflow](https://universe.roboflow.com/test-stage/rust-detection-t8vza) | CC BY 4.0 | Real inspection photos | ✅ Used |
| **Environmental** | Synthetic (domain rules) | N/A | Synthetic | ✅ Used |

---

## 1. Sensor Data

**Source:** Kaggle - Pump Sensor Data for Predictive Maintenance  
**License:** CC0: Public Domain  
**File:** `data/sensor.csv`  
**Size:** 220,320 samples × 52 sensors  
**Labels:** NORMAL, BROKEN, RECOVERING

### Usage
- Training LightGBM models for failure prediction
- Time-to-breakdown (TTB) regression
- Feature extraction: mean, std, slope, min/max

### Quality Checks
- ✅ Removed rows with >20% missing values
- ✅ Validated timestamps (ISO 8601)
- ✅ Removed duplicates
- ✅ Sorted chronologically

---

## 2. Image Data (Rust Detection)

**Source (Training):** Roboflow Universe - Rust Detection Dataset  
**Source (Testing):** Manual pump photos in `test_images/`  
**License:** CC BY 4.0  
**Models:** 
- `artifacts/rust_model.onnx` (MobileNetV3 - 6MB)
- `artifacts/rust_clip.onnx` (CLIP VLM - 335MB)

### Test Images

| File | Label | Purpose |
|------|-------|---------|
| `test_images/rust/rust1.jpg` | rust | Integration test |
| `test_images/rust/rust2.jpg` | rust | Integration test |
| `test_images/no_rust/clean1.jpg` | no_rust | Integration test |
| `test_images/no_rust/clean2.jpg` | no_rust | Integration test |

### Preprocessing
- Resize to 224×224 (MobileNetV3) or 224×224 (CLIP)
- Normalize RGB to [0, 1]
- Convert HWC → CHW for ONNX

---

## 3. Environmental Data

**Source:** Synthetic (domain expertise)  
**File:** `samples/env_critical.json`, `samples/env_favorable.json`

### Fields

| Field | Type | Range | Purpose |
|-------|------|-------|---------|
| `operating_hours` | float | 0 - 10,000+ | Usage metric |
| `days_since_last_maintenance` | float | 0 - 365+ | Maintenance lag |
| `ambient_temperature_c` | float | -10 to 60°C | Environmental stress |
| `ambient_humidity_percent` | float | 0 - 100% | Corrosion factor |
| `load_factor` | float | 0.0 - 1.5 | Operating load |
| `maintenance_overdue` | bool | true/false | Alert flag |

### Usage
Risk multiplier calculation (0.5x - 2.0x adjustment to failure probability)

---

## Data Validation (Runtime)

### Sensor Validation
- At least 1 sample in sensor_window
- Numeric values for all sensors
- Valid ISO 8601 timestamps

### Image Validation
- Valid Base64 encoding
- Decodable to JPEG/PNG
- RGB format (3 channels)
- Minimum 32×32 pixels

### Environmental Validation
- Non-negative numeric values
- load_factor capped at 2.0
- Realistic ranges for operating parameters

---

## Missing Data Handling

| Modality | Behavior |
|----------|----------|
| Sensor | Uses available features, masks NaN columns |
| Image | Skipped, sensor-only prediction |
| Environmental | Multiplier defaults to 1.0 (no adjustment) |

---

## Citation

```bibtex
@misc{predictive_maintenance_agent,
  title={Predictive Maintenance Agent},
  year={2026},
  howpublished={GitHub Repository}
}
```
