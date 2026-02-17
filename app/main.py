from __future__ import annotations

import os
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import PredictRequest, PredictResponse
from app.modalities.sensor import SensorInference
from app.modalities.image import ImageInference
from app.modalities.environmental import EnvironmentalInference

# -----------------------------
# Debug prints (optional)
# -----------------------------
print("MAIN.PY LOADED - MULTIMODAL VERSION ✅")
print("FAILURE_MODEL =", os.getenv("FAILURE_MODEL", "artifacts/pump_failure_lgbm.joblib"))
print("TTB_MODEL     =", os.getenv("TTB_MODEL", "artifacts/pump_ttb_lgbm.joblib"))
print("SCHEMA_PATH   =", os.getenv("SCHEMA_PATH", "artifacts/feature_schema.json"))

# Rust detection model selection (mobilenet or clip)
RUST_MODEL_TYPE = os.getenv("RUST_MODEL_TYPE", "mobilenet").lower()

# -----------------------------
# Paths
# -----------------------------
FAILURE_MODEL = os.getenv("FAILURE_MODEL", "artifacts/pump_failure_lgbm.joblib")
TTB_MODEL = os.getenv("TTB_MODEL", "artifacts/pump_ttb_lgbm.joblib")
SCHEMA_PATH = os.getenv("SCHEMA_PATH", "artifacts/feature_schema.json")

# Auto-select rust model based on type (or use explicit env var)
# Default: mobilenet uses rust_model.onnx, clip uses rust_clip.onnx
if "RUST_ONNX" in os.environ:
    RUST_ONNX = os.getenv("RUST_ONNX")
else:
    if RUST_MODEL_TYPE == "clip":
        RUST_ONNX = "artifacts/rust_clip.onnx"
    else:
        RUST_ONNX = "artifacts/rust_model.onnx"  # mobilenet (default)

RUST_LABELS = os.getenv("RUST_LABELS", "artifacts/rust_labels.json")

print("RUST_ONNX     =", RUST_ONNX)
print("RUST_LABELS   =", RUST_LABELS)

# -----------------------------
# Fusion knobs (tunable)
# -----------------------------
RUST_FUSE_THRESH = float(os.getenv("RUST_FUSE_THRESH", "0.85"))  # stricter to reduce false positives
RUST_PFAIL_BUMP = float(os.getenv("RUST_PFAIL_BUMP", "0.25"))    # p_fail += bump * rust_prob (clamped)

# -----------------------------
# App
# -----------------------------
app = FastAPI(title="Predictive Maintenance Agent API", version="2.2.0")

# -----------------------------
# CORS Middleware (allow UI access)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load Sensor Model (mandatory)
# -----------------------------
sensor_inf = SensorInference(FAILURE_MODEL, TTB_MODEL, SCHEMA_PATH)

# -----------------------------
# Try Load Image Model (optional modality)
# -----------------------------
image_inf: Optional[ImageInference] = None
image_enabled = False
image_error: Optional[str] = None

try:
    image_inf = ImageInference(RUST_ONNX, RUST_LABELS, model_type=RUST_MODEL_TYPE)
    image_enabled = True
    print(f"✅ Rust image model loaded: {RUST_MODEL_TYPE.upper()}")
except Exception as e:
    image_inf = None
    image_enabled = False
    image_error = f"{type(e).__name__}: {e}"
    print("⚠️ Rust image model NOT loaded:", image_error)

# -----------------------------
# Load Environmental Module (rule-based, always available)
# -----------------------------
env_inf = EnvironmentalInference()
print("✅ Environmental inference loaded.")

# -----------------------------
# Helpers
# -----------------------------
def risk_bucket(p: float) -> str:
    if p >= 0.8:
        return "failure_risk_very_high"
    if p >= 0.5:
        return "failure_risk_high"
    if p >= 0.2:
        return "failure_risk_medium"
    return "failure_risk_low"


def refresh_risk_signal(signals: list[str], p_fail: float) -> list[str]:
    signals = [s for s in signals if not s.startswith("failure_risk_")]
    signals.insert(0, risk_bucket(p_fail))
    return signals


def img_signal_from_probs(probs_dict: dict, pred_label: str) -> str:
    """
    Neat, stable formatting:
      img:<label>(p=<prob>)
    Uses canonical keys if available: {'no_rust':..., 'rust':...}
    """
    lab = str(pred_label).lower()
    if isinstance(probs_dict, dict):
        if lab in probs_dict:
            p = float(probs_dict[lab])
        else:
            # fallback to whichever key exists
            p = float(probs_dict.get("rust", probs_dict.get("no_rust", 0.0)))
    else:
        p = 0.0

    if lab not in ("rust", "no_rust"):
        lab = "unknown"

    return f"img:{lab}(p={p:.3f})"


def ensure_img_signal_visible(signals: list[str]) -> list[str]:
    """
    Keep img:* within top 5 by placing it early if present.
    """
    img_idx = next((i for i, s in enumerate(signals) if s.startswith("img:")), None)
    if img_idx is not None and img_idx > 1:
        sig = signals.pop(img_idx)
        signals.insert(1, sig)  # after failure_risk_*
    return signals


def finalize_fault_label(fault: str) -> Optional[str]:
    """
    If sensor baseline says 'unknown', return None (cleaner contract).
    """
    if fault is None:
        return None
    if str(fault).lower() == "unknown":
        return None
    return str(fault)


def generate_explanation(p_fail: float, ttb: float, fault: Optional[str], 
                         signals: list[str], env_mult: float = 1.0) -> str:
    """
    Generate a human-readable explanation of the prediction.
    This provides transparency into why the system made its prediction.
    """
    parts = []
    
    # Risk level interpretation
    if p_fail >= 0.8:
        parts.append(f"CRITICAL: Very high failure risk ({p_fail*100:.1f}%).")
    elif p_fail >= 0.5:
        parts.append(f"WARNING: Elevated failure risk ({p_fail*100:.1f}%).")
    elif p_fail >= 0.2:
        parts.append(f"CAUTION: Moderate failure risk ({p_fail*100:.1f}%).")
    else:
        parts.append(f"Normal operation with low failure risk ({p_fail*100:.2f}%).")
    
    # Fault type explanation
    if fault:
        fault_explanations = {
            "corrosion_rust": "Visual inspection detected rust/corrosion on pump components.",
            "environmental_stress": "Harsh operating conditions are accelerating wear.",
            "bearing_failure": "Sensor patterns indicate potential bearing degradation.",
            "seal_leak": "Pressure anomalies suggest possible seal integrity issues.",
            "impeller_damage": "Flow/vibration patterns consistent with impeller issues.",
        }
        parts.append(fault_explanations.get(fault, f"Detected fault condition: {fault}."))
    
    # Time to breakdown
    if ttb < 24:
        parts.append(f"Estimated breakdown: within {ttb:.0f} hours. Immediate attention recommended.")
    elif ttb < 168:  # 1 week
        parts.append(f"Estimated breakdown: ~{ttb/24:.0f} days. Schedule maintenance soon.")
    else:
        parts.append(f"Estimated breakdown: {ttb:.0f} hours. Continue monitoring.")
    
    # Environmental factors
    if env_mult > 1.5:
        parts.append("Environmental conditions are significantly increasing risk.")
    elif env_mult > 1.2:
        parts.append("Operating conditions are contributing to elevated risk.")
    elif env_mult < 0.8:
        parts.append("Favorable operating conditions are reducing risk profile.")
    
    # Top contributing signals
    contributing = [s for s in signals[:3] if not s.startswith("failure_risk_")]
    if contributing:
        parts.append(f"Key factors: {', '.join(contributing)}.")
    
    return " ".join(parts)


# =============================
# Health Endpoint
# =============================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": getattr(sensor_inf, "version", "sensor_lgbm_v1"),
        "image_model": "enabled" if image_enabled else "disabled",
        "image_model_type": RUST_MODEL_TYPE if image_enabled else None,
        "image_error": image_error,
        "paths": {
            "RUST_MODEL_TYPE": RUST_MODEL_TYPE,
            "RUST_ONNX": RUST_ONNX,
            "RUST_LABELS": RUST_LABELS,
            "FAILURE_MODEL": FAILURE_MODEL,
            "TTB_MODEL": TTB_MODEL,
            "SCHEMA_PATH": SCHEMA_PATH,
        },
        "fusion": {
            "RUST_FUSE_THRESH": RUST_FUSE_THRESH,
            "RUST_PFAIL_BUMP": RUST_PFAIL_BUMP,
        },
    }


# =============================
# Single Prediction
# =============================
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    try:
        # ---- Sensor inference
        p_fail, ttb, fault, fconf, signals = sensor_inf.predict(req)

        # ---- Image inference (if provided)
        if image_enabled and image_inf is not None and getattr(req, "image_base64", None):
            pred_label, pred_prob, probs_dict = image_inf.predict(req.image_base64)

            # Neat signal (one line), no big dict in top_signals
            signals.append(img_signal_from_probs(probs_dict, pred_label))

            # Fusion only when rust is confidently predicted
            rust_p = float(probs_dict.get("rust", pred_prob)) if isinstance(probs_dict, dict) else float(pred_prob)
            if str(pred_label).lower() == "rust" and rust_p >= RUST_FUSE_THRESH:
                signals.append("rust_detected")
                p_fail = min(1.0, float(p_fail) + RUST_PFAIL_BUMP * rust_p)
                fault = "corrosion_rust"
                fconf = max(float(fconf), rust_p)

        # ---- Environmental inference (if provided)
        env_multiplier = 1.0
        if getattr(req, "environmental", None) is not None:
            env_multiplier, env_signals = env_inf.predict(req.environmental)
            signals.extend(env_signals)
            
            # Apply environmental risk multiplier to failure probability
            p_fail = min(1.0, float(p_fail) * env_multiplier)
            
            # Adjust fault type if environmental conditions are critical
            if env_multiplier >= 1.5 and fault in (None, "unknown"):
                fault = "environmental_stress"
                fconf = max(float(fconf), 0.6)

        # recompute risk bucket after fusion so it matches final p_fail
        signals = refresh_risk_signal(signals, float(p_fail))
        signals = ensure_img_signal_visible(signals)

        dt_ms = int((time.perf_counter() - t0) * 1000)
        
        # Generate human-readable explanation
        explanation = generate_explanation(
            p_fail=float(p_fail),
            ttb=float(ttb),
            fault=finalize_fault_label(fault),
            signals=signals,
            env_mult=env_multiplier
        )

        return PredictResponse(
            asset_id=req.asset_id,
            failure_probability=float(p_fail),
            estimated_time_to_breakdown_hours=float(ttb),
            predicted_fault_type=finalize_fault_label(fault),  # unknown -> None
            fault_confidence=float(fconf),
            top_signals=signals[:5],
            inference_ms=dt_ms,
            model_version="sensor_lgbm_v2_multimodal",
            explanation=explanation,
        )

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"{type(e).__name__}: {e}")


# =============================
# Batch Prediction
# =============================
@app.post("/predict/batch")
def predict_batch(reqs: list[PredictRequest]):
    t0 = time.perf_counter()
    results = []

    try:
        for r in reqs:
            p_fail, ttb, fault, fconf, signals = sensor_inf.predict(r)

            if image_enabled and image_inf is not None and getattr(r, "image_base64", None):
                pred_label, pred_prob, probs_dict = image_inf.predict(r.image_base64)

                signals.append(img_signal_from_probs(probs_dict, pred_label))

                rust_p = float(probs_dict.get("rust", pred_prob)) if isinstance(probs_dict, dict) else float(pred_prob)
                if str(pred_label).lower() == "rust" and rust_p >= RUST_FUSE_THRESH:
                    signals.append("rust_detected")
                    p_fail = min(1.0, float(p_fail) + RUST_PFAIL_BUMP * rust_p)
                    fault = "corrosion_rust"
                    fconf = max(float(fconf), rust_p)

            # Environmental inference
            env_multiplier = 1.0
            if getattr(r, "environmental", None) is not None:
                env_multiplier, env_signals = env_inf.predict(r.environmental)
                signals.extend(env_signals)
                p_fail = min(1.0, float(p_fail) * env_multiplier)
                
                if env_multiplier >= 1.5 and fault in (None, "unknown"):
                    fault = "environmental_stress"
                    fconf = max(float(fconf), 0.6)

            signals = refresh_risk_signal(signals, float(p_fail))
            signals = ensure_img_signal_visible(signals)
            
            # Generate explanation for batch item
            explanation = generate_explanation(
                p_fail=float(p_fail),
                ttb=float(ttb),
                fault=finalize_fault_label(fault),
                signals=signals,
                env_mult=env_multiplier
            )

            results.append(
                {
                    "asset_id": r.asset_id,
                    "failure_probability": float(p_fail),
                    "estimated_time_to_breakdown_hours": float(ttb),
                    "predicted_fault_type": finalize_fault_label(fault),  # unknown -> None
                    "fault_confidence": float(fconf),
                    "top_signals": signals[:5],
                    "model_version": "sensor_lgbm_v2_multimodal",
                    "explanation": explanation,
                }
            )

        dt_ms = int((time.perf_counter() - t0) * 1000)
        return {"count": len(results), "inference_ms_total": dt_ms, "results": results}

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"{type(e).__name__}: {e}")
