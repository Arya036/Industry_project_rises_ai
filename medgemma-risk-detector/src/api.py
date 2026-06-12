"""
src/api.py

MedGemma / VLM Risk Detector API.
Provides:
  - POST /predict: Handles multipart/form-data image file uploads.
  - GET  /health: Liveness health check.

Usage:
    uvicorn src.api:app --host 0.0.0.0 --port 8000
    python src/api.py --test  # Self-test mode
"""

import os
import sys
import argparse
import tempfile
import yaml
import json
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

# Add src/ to path so siblings are importable
sys.path.insert(0, str(Path(__file__).parent))

from inference import run_pipeline, load_models

# ── Load Config ───────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# ── FastAPI app ───────────────────────────────────────────────────────
app = FastAPI(
    title="MedGemma Risk Detector API",
    description="Vision-Language Model diagnostic pipeline for occupational lung disease classification.",
    version="1.0.0",
)


@app.on_event("startup")
def startup_event():
    """Load ONNX models at server startup."""
    try:
        load_models(config)
        print("[API] Models loaded successfully on startup.")
    except Exception as e:
        print(f"[API] Error loading models: {e}")
        # We do not crash startup in case placeholders need to be generated at runtime
        pass


@app.get("/health")
def health_check():
    """Liveness check endpoint."""
    return {"status": "online", "service": "medgemma-risk-detector-api"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Diagnostic classification endpoint.
    
    Input:
        multipart/form-data with field 'file' containing the chest X-ray image (JPEG or PNG).
        
    Output:
        JSON object containing prediction (abnormal/normal), confidence_score,
        inference_time_ms, and visualizations.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    allowed_types = {"image/jpeg", "image/png", "image/jpg"}
    is_allowed = file.content_type in allowed_types
    if not is_allowed:
        suffix = Path(file.filename).suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png"}:
            is_allowed = True

    if not is_allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Upload JPEG or PNG."
        )

    # Save uploaded image to temp file
    suffix = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # Run inference pipeline
        result = run_pipeline(tmp_path, config)
        return JSONResponse(content=result)
    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()}
        )
    finally:
        # Cleanup temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ── Self-Test / CLI Entrypoint ────────────────────────────────────────
def run_self_test():
    """Runs a self-test of the pipeline against the sample images."""
    print("=" * 60)
    print("MEDGEMMA RISK DETECTOR — API SELF TEST")
    print("=" * 60)
    
    # Load models
    load_models(config)
    
    # Locate a sample image in data/ folder
    data_dir = Path(__file__).parent.parent / "data"
    samples = list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png"))
    
    if not samples:
        print(f"Error: No sample images found in {data_dir}. Place some images first.")
        sys.exit(1)
        
    sample_img = samples[0]
    print(f"Running self-test on sample image: {sample_img.name}")
    
    try:
        start_time = time.time()
        result = run_pipeline(str(sample_img), config)
        elapsed = (time.time() - start_time) * 1000
        
        print("\nAPI Response Output:")
        print(json.dumps(result, indent=2))
        
        # Verify keys
        required_keys = {"prediction", "confidence_score", "inference_time_ms", "visualizations"}
        missing_keys = required_keys - set(result.keys())
        
        if missing_keys:
            print(f"\n[FAIL] Missing required keys in response: {missing_keys}")
            sys.exit(1)
            
        vis_keys = {"gradcam_overlay_base64"}
        missing_vis_keys = vis_keys - set(result["visualizations"].keys())
        if missing_vis_keys:
            print(f"\n[FAIL] Missing keys inside visualizations: {missing_vis_keys}")
            sys.exit(1)
            
        if not result["visualizations"]["gradcam_overlay_base64"]:
            print("\n[FAIL] Visualizations field 'gradcam_overlay_base64' is empty.")
            sys.exit(1)
            
        print("\n[SUCCESS] Self-test complete! API response schema is 100% compliant.")
        
    except Exception as e:
        print(f"\n[FAIL] Self-test encountered an error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import time  # imported for CLI test timing
    parser = argparse.ArgumentParser(description="MedGemma Risk Detector API Startup")
    parser.add_argument("--test", action="store_true", help="Run self-test on sample images and exit")
    args = parser.parse_args()

    if args.test:
        run_self_test()
    else:
        import uvicorn
        uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
