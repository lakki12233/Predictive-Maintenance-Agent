from __future__ import annotations

import json
import time
import warnings
from typing import List, Tuple, Any

import numpy as np
import pandas as pd
import joblib


class SensorInference:
    """
    Optimized inference:
    - Builds features directly from the provided sensor_window using NumPy
    - Supports sensor_window items that are dicts OR Pydantic models (SensorSample)
    - Returns tuple (p_fail, ttb, fault, fconf, signals) as expected by app/main.py
    """

    def __init__(self, failure_model_path: str, ttb_model_path: str, schema_path: str):
        self.clf = joblib.load(failure_model_path)
        self.reg = joblib.load(ttb_model_path)

        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        self.window = int(schema["window"])
        self.sensor_columns: List[str] = list(schema["sensor_columns"])
        self.feature_columns: List[str] = list(schema["feature_columns"])

        self.version = "sensor_lgbm_v1"

    @staticmethod
    def _as_dict(x: Any) -> dict:
        # pydantic v2
        if hasattr(x, "model_dump"):
            return x.model_dump()
        # pydantic v1
        if hasattr(x, "dict"):
            return x.dict()
        # already dict-like
        if isinstance(x, dict):
            return x
        # fallback: try vars()
        try:
            return dict(vars(x))
        except Exception:
            return {}

    def _features_from_window_numpy(self, samples: List[Any]) -> np.ndarray:
        # Keep only last N
        samples = samples[-self.window :]

        # Convert to dicts (handles SensorSample objects)
        rows = [self._as_dict(s) for s in samples]

        n_t = len(rows)
        n_s = len(self.sensor_columns)

        mat = np.full((n_t, n_s), np.nan, dtype=np.float32)

        for i, r in enumerate(rows):
            for j, c in enumerate(self.sensor_columns):
                v = r.get(c, None)
                if v is None:
                    continue
                try:
                    mat[i, j] = float(v)
                except Exception:
                    mat[i, j] = np.nan

        # Aggregates across time axis=0
        # Suppress expected warnings for empty/all-NaN slices
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
            warnings.filterwarnings('ignore', category=RuntimeWarning, message='Degrees of freedom')
            warnings.filterwarnings('ignore', category=RuntimeWarning, message='All-NaN slice encountered')
            
            mean = np.nanmean(mat, axis=0)
            std = np.nanstd(mat, axis=0)
            mn = np.nanmin(mat, axis=0)
            mx = np.nanmax(mat, axis=0)
        
        last = mat[-1, :] if n_t > 0 else np.full(n_s, np.nan)
        first = mat[0, :] if n_t > 0 else np.full(n_s, np.nan)
        denom = max(1, n_t - 1)
        slope = (last - first) / denom

        feat_map = {}
        for j, c in enumerate(self.sensor_columns):
            feat_map[f"mean__{c}"] = mean[j]
            feat_map[f"std__{c}"] = std[j]
            feat_map[f"min__{c}"] = mn[j]
            feat_map[f"max__{c}"] = mx[j]
            feat_map[f"last__{c}"] = last[j]
            feat_map[f"slope__{c}"] = slope[j]

        x = np.array([feat_map.get(fc, np.nan) for fc in self.feature_columns], dtype=np.float32)
        return x

    def predict(self, req) -> Tuple[float, float, str, float, List[str]]:
        t0 = time.perf_counter()

        samples = req.sensor_window or []
        if not samples:
            raise ValueError("sensor_window is required and cannot be empty.")

        x = self._features_from_window_numpy(samples)

        # Keep DataFrame for the sklearn pipeline compatibility
        X = pd.DataFrame([x], columns=self.feature_columns)

        p_fail = float(self.clf.predict_proba(X)[0, 1])

        ttb = float(self.reg.predict(X)[0])
        if np.isnan(ttb) or ttb < 0:
            ttb = 0.0

        # ---- Signals (cheap heuristics)
        signals: List[str] = []

        if p_fail >= 0.8:
            signals.append("failure_risk_very_high")
        elif p_fail >= 0.5:
            signals.append("failure_risk_high")
        elif p_fail >= 0.2:
            signals.append("failure_risk_medium")
        else:
            signals.append("failure_risk_low")

        std_idx = [i for i, name in enumerate(self.feature_columns) if name.startswith("std__")]
        agg_std = float(np.nanmean(x[std_idx])) if std_idx else 0.0

        if agg_std > 50:
            signals.append("high_vibration_variance")
        elif agg_std > 20:
            signals.append("moderate_vibration_variance")
        else:
            signals.append("stable_vibration")

        last_idx = [i for i, name in enumerate(self.feature_columns) if name.startswith("last__")]
        if last_idx:
            mx_last = float(np.nanmax(x[last_idx]))
            if mx_last > 2000:
                signals.append("sensor_spike_detected")
            else:
                signals.append("no_sensor_spike")

        slope_idx = [i for i, name in enumerate(self.feature_columns) if name.startswith("slope__")]
        if slope_idx:
            mx_slope = float(np.nanmax(np.abs(x[slope_idx])))
            if mx_slope > 50:
                signals.append("strong_trend_change")
            elif mx_slope > 10:
                signals.append("moderate_trend_change")
            else:
                signals.append("weak_trend_change")

        # ---- Fault heuristic (baseline)
        fault = None
        fconf = 0.30

        if p_fail >= 0.7 and agg_std > 50:
            fault = "bearing_failure"
            fconf = 0.65
        elif p_fail >= 0.5 and agg_std > 20:
            fault = "cavitation"
            fconf = 0.55
        elif p_fail >= 0.5 and ttb <= 72:
            fault = "seal_leak"
            fconf = 0.50
        elif p_fail >= 0.9:
            fault = "impeller_damage"
            fconf = 0.45

        _ = int((time.perf_counter() - t0) * 1000)  # endpoint measures full time

        return p_fail, ttb, fault, fconf, signals[:5]
