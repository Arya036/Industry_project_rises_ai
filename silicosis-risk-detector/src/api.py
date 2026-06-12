"""
src/api.py

Silicosis Detection System — FastAPI backend
Orchestrates the full diagnostic pipeline:
  1. Binary silicosis classifier  (local ONNX, CPU)
  2. Multi-label finding classifier (local ONNX, CPU)
  3. GradCAM visualizations         (local)
  4. MedGemma report generation     (remote — Kaggle ngrok API)

POST /predict  — Main inference endpoint
GET  /health   — Server liveness check
GET  /kaggle-status — Check if MedGemma API is reachable
"""

import os
import sys
import time
import yaml
import json
import base64
import tempfile
import requests
import traceback

import cv2
import numpy as np
from pathlib import Path
from io import BytesIO
from PIL import Image

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

# Add src/ to path so sibling modules are importable
sys.path.insert(0, str(Path(__file__).parent))

from inference import run_full_pipeline
from preprocessor import preprocess_image

# ── Load config ───────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Allow ngrok URL to be overridden via environment variable at runtime
KAGGLE_NGROK_URL = os.environ.get(
    "KAGGLE_NGROK_URL",
    config["kaggle_api"]["ngrok_url"]
).strip()

KAGGLE_TIMEOUT = config["model_settings"]["medgemma_timeout_seconds"]
REPORT_ENDPOINT = config["kaggle_api"]["report_endpoint"]

# ── FastAPI app ───────────────────────────────────────────────────────
app = FastAPI(
    title="Silicosis Detection API",
    description=(
        "Three-stage diagnostic pipeline: binary silicosis screening, "
        "multi-label finding detection, and AI-generated radiology report."
    ),
    version="1.0.0",
)


@app.get("/health")
def health_check():
    return {"status": "online", "service": "silicosis-detection-api"}


@app.get("/kaggle-status")
def kaggle_status():
    """Check whether the Kaggle MedGemma server is reachable."""
    url = KAGGLE_NGROK_URL
    if not url:
        return {"status": "not_configured"}
    try:
        r = requests.get(
            url.rstrip("/") + config["kaggle_api"]["health_endpoint"],
            timeout=6
        )
        if r.status_code == 200:
            return {"status": "online", "url": url}
        return {"status": "error", "code": r.status_code}
    except Exception as e:
        return {"status": "offline", "error": str(e)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Main diagnostic endpoint.

    Input:  multipart/form-data with field 'file' (JPEG or PNG chest X-ray)
    Output: JSON with prediction, confidence_score, inference_time_ms,
            visualizations, and additional_inference_metadata
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    allowed_types = {"image/jpeg", "image/png", "image/jpg"}
    
    # We check content type but also fallback to checking file extension in case content type is unset
    is_allowed = file.content_type in allowed_types
    if not is_allowed:
        suffix = Path(file.filename).suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png"}:
            is_allowed = True
            
    if not is_allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Send JPEG or PNG."
        )

    # Save upload to temp file
    suffix = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        start_time = time.time()
        result = run_full_pipeline(tmp_path, config, KAGGLE_NGROK_URL)
        elapsed_ms = int((time.time() - start_time) * 1000)

        # ── Format response to match handover spec ─────────────────
        binary = result.get("binary", {})
        findings = result.get("findings", {})
        gradcams = result.get("finding_gradcams", {})
        binary_cam = result.get("binary_gradcam") or {}

        # Primary prediction comes from binary classifier risk category to match handover schema
        prediction = binary.get("risk_category", "Unknown")
        confidence = binary.get("confidence", 0.0)

        # Build visualizations dict
        visualizations = {}
        if binary_cam.get("overlay_b64"):
            visualizations["gradcam_overlay_base64"] = binary_cam["overlay_b64"]
        if binary_cam.get("original_b64"):
            visualizations["original_image_base64"] = binary_cam["original_b64"]

        # Add per-finding GradCAMs
        for finding_name, cam_data in gradcams.items():
            safe_key = finding_name.lower().replace(" ", "_").replace("/", "_")
            visualizations[f"gradcam_{safe_key}_base64"] = cam_data.get("overlay_b64", "")

        # NOTE: Segmentation mask is not produced by our pipeline currently.
        # If you integrate segmentation in a future version, add it here.
        visualizations["segmentation_mask_base64"] = ""

        response = {
            "prediction": prediction,
            "confidence_score": round(float(confidence), 4),
            "inference_time_ms": elapsed_ms,
            "visualizations": visualizations,
            "additional_inference_metadata": {
                "binary_label": binary.get("label", ""),
                "risk_category": binary.get("risk_category", ""),
                "risk_pct": binary.get("risk_pct", 0.0),
                "youden_threshold": config["model_settings"]["binary_youden_threshold"],
                "findings_detected": findings.get("present", []),
                "findings_probabilities": {
                    item["name"]: round(item["probability"], 4)
                    for item in findings.get("list", [])
                },
                "clinical_report": result.get("report", ""),
                "medgemma_status": (
                    "online" if KAGGLE_NGROK_URL else "not_configured"
                ),
                "model_notes": (
                    "Binary AUC 0.9078 (internal validation). "
                    "Hilum and Bronchiectasis findings have low reliability. "
                    "Not validated for clinical use."
                ),
            },
        }

        return JSONResponse(content=response)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()}
        )
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
