from __future__ import annotations

import base64
import io
import json
from typing import Dict, Tuple, List

import numpy as np
from PIL import Image
import onnxruntime as ort


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-12)


class ImageInference:
    """
    Rust / No-rust classifier inference using ONNXRuntime.
    
    Supports multiple model architectures:
    - MobileNetV3 (default): ImageNet preprocessing
    - CLIP: Vision-language model preprocessing
    
    Goals:
    - Always return a stable probs_dict with keys {"rust","no_rust"} (when possible)
    - Avoid label-order bugs by normalizing labels into a canonical mapping
    - Work with common ONNX output shapes ([1,2], [2], or something flattenable to 2)

    Returns from predict():
        (pred_label, pred_prob, probs_dict)
    Where probs_dict always tries to be:
        {"no_rust": p_no_rust, "rust": p_rust}
    """

    def __init__(
        self, 
        onnx_path: str, 
        labels_path: str, 
        image_size: int = 224,
        model_type: str = "mobilenet"
    ):
        raw_labels: List[str] = json.load(open(labels_path, "r", encoding="utf-8"))
        if len(raw_labels) != 2:
            raise ValueError(f"Expected 2 labels, got {len(raw_labels)}: {raw_labels}")
        
        self.model_type = model_type.lower()

        self.model_type = model_type.lower()

        # Normalize label strings for robust matching
        self.raw_labels = raw_labels
        self.norm_labels = [self._norm_label(x) for x in raw_labels]

        # Build a mapping from model-index -> canonical label ("no_rust" or "rust")
        # This prevents accidental swaps in labels.json from silently breaking logic.
        self.index_to_canonical = self._build_index_to_canonical(self.norm_labels)

        self.image_size = int(image_size)

        # CPU provider works everywhere (Windows + Docker)
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [o.name for o in self.sess.get_outputs()]
        
        print(f"✅ Loaded {self.model_type.upper()} rust detection model: {onnx_path}")

    # ---------------------------
    # Label normalization
    # ---------------------------
    @staticmethod
    def _norm_label(s: str) -> str:
        s = str(s).strip().lower()
        s = s.replace("-", "_").replace(" ", "_")
        return s

    @staticmethod
    def _build_index_to_canonical(norm_labels: List[str]) -> Dict[int, str]:
        """
        Given normalized labels from labels.json (length=2),
        determine which index corresponds to 'rust' and which to 'no_rust'.

        Accepts common variants:
        - "rust" / "no_rust"
        - "norust" / "no-rust" / "no rust"
        - "clean" / "healthy" (treated as no_rust)
        """
        def canonicalize(lab: str) -> str:
            lab2 = lab.replace("__", "_")
            if lab2 in ("rust", "corrosion", "corroded"):
                return "rust"
            if lab2 in ("no_rust", "norust", "no_rust_", "clean", "healthy", "normal", "ok"):
                return "no_rust"
            # Try partial matches
            if "rust" in lab2 and "no" not in lab2:
                return "rust"
            if "no" in lab2 and "rust" in lab2:
                return "no_rust"
            return lab2  # fallback

        canon = [canonicalize(x) for x in norm_labels]

        # Ensure we can map to canonical pair
        if set(canon) == {"rust", "no_rust"}:
            return {0: canon[0], 1: canon[1]}

        # If ambiguous, keep order but warn via fallback mapping.
        # (We still return something deterministic.)
        return {0: canon[0], 1: canon[1]}

    # ---------------------------
    # Preprocess
    # ---------------------------
    def _preprocess(self, b64: str) -> np.ndarray:
        img_bytes = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        img = img.resize((self.image_size, self.image_size))

        x = np.asarray(img).astype(np.float32) / 255.0  # [H,W,C]
        x = np.transpose(x, (2, 0, 1))                  # [C,H,W]

        # Model-specific preprocessing
        if self.model_type == "clip":
            # CLIP preprocessing (OpenAI)
            mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)[:, None, None]
            std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)[:, None, None]
        else:
            # MobileNet/ImageNet preprocessing (default)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        
        x = (x - mean) / std

        return x[None, :, :, :]  # [1,C,H,W]

    # ---------------------------
    # Output extraction
    # ---------------------------
    def _extract_class_vector(self, outputs: List[np.ndarray]) -> np.ndarray:
        """
        Pick the first output that can be coerced into (2,).
        """
        candidates = []

        for out in outputs:
            arr = np.array(out)

            if arr.ndim == 2 and arr.shape[0] == 1 and arr.shape[1] == 2:
                candidates.append(arr[0])
            elif arr.ndim == 1 and arr.shape[0] == 2:
                candidates.append(arr)
            elif arr.ndim >= 2 and arr.shape[-1] == 2:
                candidates.append(arr.reshape(-1, 2)[0])

        if not candidates:
            shapes = [np.array(o).shape for o in outputs]
            raise RuntimeError(f"Could not find class vector of length 2 in outputs. Shapes: {shapes}")

        return np.asarray(candidates[0], dtype=np.float32)

    @staticmethod
    def _to_probs(vec2: np.ndarray) -> np.ndarray:
        """
        Convert a length-2 vector to probabilities.
        If it already looks like probs (sum~1, each in [0,1]), keep it.
        Else apply softmax.
        """
        vec2 = vec2.astype(np.float32)
        s = float(np.sum(vec2))
        if (np.min(vec2) >= -1e-3) and (np.max(vec2) <= 1.0 + 1e-3) and (abs(s - 1.0) < 1e-2):
            probs = vec2
        else:
            probs = _softmax(vec2)

        # numerical safety
        probs = np.clip(probs, 0.0, 1.0).astype(np.float32)
        probs = probs / (np.sum(probs) + 1e-12)
        return probs

    # ---------------------------
    # Public API
    # ---------------------------
    def predict(self, image_base64: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Returns:
          (pred_label, pred_prob, probs_dict)

        probs_dict is canonical whenever possible:
          {"no_rust": p_no_rust, "rust": p_rust}
        """
        x = self._preprocess(image_base64)

        outputs = self.sess.run(None, {self.input_name: x})
        vec = self._extract_class_vector(outputs)
        probs = self._to_probs(vec)

        # Build canonical probs dict (robust to labels.json swaps)
        # First, map model indices -> canonical label using index_to_canonical
        p0 = float(probs[0])
        p1 = float(probs[1])

        canon0 = self.index_to_canonical.get(0, self.norm_labels[0])
        canon1 = self.index_to_canonical.get(1, self.norm_labels[1])

        # Initialize with safe defaults
        probs_dict: Dict[str, float] = {}

        # If we can form the canonical pair, do it.
        if set([canon0, canon1]) == {"rust", "no_rust"}:
            probs_dict[canon0] = p0
            probs_dict[canon1] = p1

            # Ensure both keys exist in standard spelling
            p_rust = float(probs_dict.get("rust", 0.0))
            p_no = float(probs_dict.get("no_rust", 0.0))
            probs_dict = {"no_rust": p_no, "rust": p_rust}
        else:
            # Fallback: expose both raw labels as a dict
            # (still deterministic, but not guaranteed canonical)
            probs_dict = {self.raw_labels[0]: p0, self.raw_labels[1]: p1}

            # Best-effort canonical projection if substring match exists
            # (helps if labels are like ["clean","rust"])
            p_rust = 0.0
            p_no = 0.0
            for i, raw in enumerate(self.raw_labels):
                n = self._norm_label(raw)
                if ("rust" in n) and ("no" not in n):
                    p_rust = float(probs[i])
                elif ("no" in n and "rust" in n) or (n in ("clean", "healthy", "normal", "ok")):
                    p_no = float(probs[i])
            if (p_rust + p_no) > 0:
                probs_dict = {"no_rust": p_no, "rust": p_rust}

        # Prediction based on canonical rust/no_rust if available, else argmax
        if "rust" in probs_dict and "no_rust" in probs_dict:
            pred_label = "rust" if probs_dict["rust"] >= probs_dict["no_rust"] else "no_rust"
            pred_prob = float(max(probs_dict["rust"], probs_dict["no_rust"]))
        else:
            pred_idx = int(np.argmax(probs))
            pred_label = self.raw_labels[pred_idx]
            pred_prob = float(probs[pred_idx])

        return pred_label, pred_prob, probs_dict
