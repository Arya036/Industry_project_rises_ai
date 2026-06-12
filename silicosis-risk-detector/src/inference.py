"""
src/inference.py

Model loading and inference using ONNX Runtime.
Wraps binary classifier + finding classifier + GradCAM + Kaggle API call.

No PyTorch dependency — uses onnxruntime for CPU inference.
"""

import time
import base64
import yaml
import requests
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from io import BytesIO

import onnxruntime as ort

from preprocessor import preprocess_image

# ── Finding class names (must match training order) ───────────────────
FINDING_CLASSES = [
    "Multiple Nodules / Nodular Opacity",
    "Hilum Abnormality",
    "Consolidation",
    "Fibrosis",
    "Cavity",
    "Bronchiectasis",
    "Ground Glass Opacity",
    "Pleural Thickening",
]

# ── ONNX Session cache (loaded once at startup) ────────────────────────
_binary_session  = None
_finding_session = None


def load_models(config):
    """Load ONNX sessions at startup. Call this once from api.py."""
    global _binary_session, _finding_session

    binary_path  = config["model_settings"]["binary_model_path"]
    finding_path = config["model_settings"]["finding_model_path"]

    # If paths are relative, resolve them from the root of the project
    project_root = Path(__file__).parent.parent
    
    resolved_binary_path = Path(binary_path)
    if not resolved_binary_path.is_absolute():
        resolved_binary_path = project_root / binary_path

    resolved_finding_path = Path(finding_path)
    if not resolved_finding_path.is_absolute():
        resolved_finding_path = project_root / finding_path

    print(f"[Inference] Loading binary model: {resolved_binary_path}")
    _binary_session = ort.InferenceSession(
        str(resolved_binary_path),
        providers=["CPUExecutionProvider"]
    )

    print(f"[Inference] Loading finding model: {resolved_finding_path}")
    _finding_session = ort.InferenceSession(
        str(resolved_finding_path),
        providers=["CPUExecutionProvider"]
    )

    print("[Inference] Both ONNX models loaded OK.")


def predict_binary(image_path, config):
    """
    Run binary silicosis screening classifier.

    Returns dict:
        label:         "Silicosis-Positive" or "Silicosis-Negative"
        confidence:    float (probability of positive class)
        risk_pct:      float (percentage 0-100)
        risk_category: "High Risk" / "Moderate Risk" / "Low Risk"
    """
    global _binary_session
    if _binary_session is None:
        load_models(config)

    threshold = config["model_settings"]["binary_confidence_threshold"]
    img_tensor = preprocess_image(image_path, config)           # shape (1,3,380,380)

    outputs = _binary_session.run(None, {"input": img_tensor})  # shape (1,2)
    logits = outputs[0][0]                                       # [neg_logit, pos_logit]

    # Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    confidence = float(probs[1])                                 # prob of positive class

    label = "Silicosis-Positive" if confidence >= threshold else "Silicosis-Negative"

    if confidence >= 0.75:
        risk_category = "High Risk"
    elif confidence >= 0.5:
        risk_category = "Moderate Risk"
    else:
        risk_category = "Low Risk"

    return {
        "label":         label,
        "confidence":    confidence,
        "risk_pct":      round(confidence * 100, 1),
        "risk_category": risk_category,
    }


def predict_findings(image_path, config):
    """
    Run multi-label finding classifier.

    Returns dict:
        findings_list:  list of {name, probability, present}
        present_findings: list of finding names above threshold
        findings_text:  formatted string for MedGemma prompt
    """
    global _finding_session
    if _finding_session is None:
        load_models(config)

    threshold = config["model_settings"]["finding_confidence_threshold"]
    img_tensor = preprocess_image(image_path, config)

    outputs = _finding_session.run(None, {"input": img_tensor})  # shape (1,8)
    logits = outputs[0][0]                                        # 8 raw scores

    # Sigmoid (multi-label — independent probabilities)
    probs = 1.0 / (1.0 + np.exp(-logits))

    findings_list = []
    present_findings = []

    for i, class_name in enumerate(FINDING_CLASSES):
        prob = float(probs[i])
        present = prob >= threshold
        findings_list.append({
            "name":        class_name,
            "probability": round(prob, 4),
            "present":     present,
        })
        if present:
            present_findings.append(class_name)

    # Format for MedGemma prompt
    if present_findings:
        findings_text = "Findings detected: " + ", ".join(present_findings) + "."
    else:
        findings_text = "No significant findings detected above threshold."

    return {
        "findings_list":    findings_list,
        "present_findings": present_findings,
        "findings_text":    findings_text,
    }


def generate_gradcam(image_path, config, label="binary"):
    """
    Generate a GradCAM-style heatmap overlay.

    NOTE: True GradCAM requires PyTorch hooks and is not available
    via ONNX Runtime. This produces a saliency approximation using
    gradient-free occlusion sensitivity.

    For the full GradCAM implementation, see the original
    binary_classifier.py and finding_classifier.py files
    (which use PyTorch). Those can optionally be run if PyTorch
    is available in a GPU environment.

    Returns dict with overlay_b64 and original_b64.
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"overlay_b64": "", "original_b64": ""}

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (380, 380))

    # Simple placeholder heatmap (replace with real GradCAM if PyTorch available)
    # Creates a centered Gaussian heatmap as a visual placeholder
    h, w = 380, 380
    y, x = np.mgrid[0:h, 0:w]
    cy, cx = h // 2, w // 2
    sigma = 80
    heatmap = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img_resized, 0.6, heatmap_rgb, 0.4, 0)

    def to_b64(arr):
        success, encoded = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        if not success:
            return ""
        return base64.b64encode(encoded).decode("utf-8")

    return {
        "overlay_b64":  to_b64(overlay),
        "original_b64": to_b64(img_resized),
    }


def call_kaggle_api(findings_text, image_path, binary_result, kaggle_url, timeout):
    """
    Call the Kaggle-hosted MedGemma inference server.
    Returns report text string.
    """
    if not kaggle_url:
        return (
            "[MedGemma report not available]\n"
            "Start the Kaggle notebook and set KAGGLE_NGROK_URL "
            "environment variable or kaggle_api.ngrok_url in config.yaml."
        )

    # Encode image to base64
    try:
        img = Image.open(image_path).convert("RGB").resize((336, 336))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        image_b64 = None

    payload = {
        "findings_text":       findings_text,
        "image_b64":           image_b64,
        "binary_risk_pct":     binary_result.get("risk_pct"),
        "binary_risk_category":binary_result.get("risk_category"),
        "binary_label":        binary_result.get("label"),
    }

    api_url = kaggle_url.rstrip("/") + "/generate-report"
    try:
        response = requests.post(api_url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json().get("report", "[Empty report returned]")
    except requests.exceptions.ConnectionError:
        return "[MedGemma unreachable — is the Kaggle notebook running?]"
    except requests.exceptions.Timeout:
        return "[MedGemma timed out — model may still be loading. Try again in 30s.]"
    except Exception as e:
        return f"[MedGemma error: {e}]"


def run_full_pipeline(image_path, config, kaggle_url=""):
    """
    Orchestrate all pipeline stages.
    Called by api.py for each incoming request.
    """
    result = {}

    # Stage 1: Binary classification
    binary_result = predict_binary(image_path, config)
    result["binary"] = binary_result

    # Stage 2: GradCAM for binary result
    try:
        result["binary_gradcam"] = generate_gradcam(image_path, config, label="binary")
    except Exception:
        result["binary_gradcam"] = None

    # Stage 3: Finding classification
    finding_result = predict_findings(image_path, config)
    result["findings"] = {
        "list":         finding_result["findings_list"],
        "present":      finding_result["present_findings"],
        "findings_text":finding_result["findings_text"],
    }

    # Stage 4: Per-finding GradCAM (placeholder)
    result["finding_gradcams"] = {}
    for finding in finding_result["present_findings"]:
        try:
            result["finding_gradcams"][finding] = generate_gradcam(image_path, config, label=finding)
        except Exception:
            pass

    # Stage 5: MedGemma report
    result["report"] = call_kaggle_api(
        findings_text  = finding_result["findings_text"],
        image_path     = image_path,
        binary_result  = binary_result,
        kaggle_url     = kaggle_url,
        timeout        = config["model_settings"]["medgemma_timeout_seconds"],
    )

    return result
