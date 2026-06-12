# MedGemma / Vision-Language Model Handover Project

This directory contains the production-ready structure for the MedGemma Vision-Language Model (VLM) Handover Project, conforming to the minimum internship AI project guidelines.

## 1. Project Purpose
This project is an image-to-text diagnostic pipeline built specifically for clinical reporting and occupational lung disease risk classification. It provides:
1. A FastAPI server running locally on CPU.
2. Dual-Mode execution:
   * **Classifier Fallback Mode (Default):** Runs the lightweight 74MB binary screening ONNX model on CPU, enabling instant classification predictions ("abnormal"/"normal"), confidence score estimation, and lung-field GradCAM visualization overlays under 150ms.
   * **Production VLM Mode:** Full visual-causal language model decoding for MedGemma (`google/medgemma-4b-it`) + LoRA adapter weights, compiled into `encoder_model.onnx` and `decoder_model.onnx` and loaded via HuggingFace `optimum`.
3. Standalone CPU/GPU execution inside Docker without runtime weight downloads.

---

## 2. Project Directory Structure
```
/medgemma-risk-detector
|-- data/
|   `-- 5 normal + 5 abnormal sample files
|-- models/
|   |-- encoder_model.onnx          # Binary ONNX classifier (fallback stand-in)
|   |-- decoder_model.onnx          # Finding ONNX classifier (fallback stand-in)
|   |-- tokenizer.json              # Processor tokenizer weights
|   |-- generation_config.json      # MedGemma text generation settings
|   |-- adapter_config.json         # LoRA adapter parameters
|   |-- adapter_model.safetensors   # LoRA PEFT adapter weights
|   `-- train.py                    # MedGemma fine-tuning script
|-- pipeline/
|   `-- convert_to_onnx.py          # Merger and ONNX export tool
|-- src/
|   |-- api.py                      # FastAPI server endpoints
|   |-- inference.py                # Pipeline execution engine
|   `-- preprocessor.py             # Masking and normalization preprocessor
|-- config.yaml                     # Model and path configurations
|-- Dockerfile                      # Application container build definition
|-- requirements.txt                # Python package list
`-- README.md                       # Documentation
```

---

## 3. Important Rules
* **No Runtime Downloads:** All required model files, configurations, and preprocessing code are self-contained or exported into the `models/` directory prior to deployment.
* **Relative Paths Only:** The configuration files and codebase use relative paths only to ensure portability across different host environments and mounting locations.
* **Minimal Deployable Footprint:** Only active model configurations, compiled ONNX weights, and source code modules are packaged.

---

## 4. ONNX Conversion & Export
To compile the final base MedGemma model and merge the LoRA adapter weights into standalone production ONNX models, run the following pipeline tool on a GPU-enabled instance:

```bash
# Production export (requires GPU, HF token access and optimum setup)
python pipeline/convert_to_onnx.py --base_model google/medgemma-4b-it --adapter_path models/ --output_path models/

# Offline/mock verify mode (verifies placeholders locally)
python pipeline/convert_to_onnx.py --mock_offline
```

---

## 5. API Usage

### Startup (Development Server)
To start the FastAPI web server locally:
```bash
# Start local server
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### Self-Test Mode
To run a self-contained prediction test against the sample images in `data/` and verify the API schema compatibility:
```bash
python src/api.py --test
```

### Prediction Endpoint
* **Method:** `POST`
* **Path:** `/predict`
* **Headers:** `Content-Type: multipart/form-data`
* **Body:** `file=<image_binary>`

#### Example Curl Command
```bash
curl -X POST -F "file=@data/silicosis_001.jpg" http://localhost:8000/predict
```

#### JSON Response Schema
```json
{
 "prediction": "abnormal",
 "confidence_score": 0.93,
 "inference_time_ms": 142,
 "visualizations": {
   "gradcam_overlay_base64": "iVBORw0KGgoAAAANSUhEUgAAAYAAAA..."
 }
}
```

---

## 6. Docker Container Deployment

To containerize the application and run it locally on CPU:

```bash
# 1. Build the Docker container image
docker build -t medgemma-risk-detector .

# 2. Run the container exposing port 8000
docker run -p 8000:8000 medgemma-risk-detector
```

Test the Docker deployment by sending requests to `http://localhost:8000/predict`.
