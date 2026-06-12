"""
MedGemma API Server — Run this in a Kaggle Notebook (T4 GPU)
=============================================================

SETUP INSTRUCTIONS:
1. Create a NEW Kaggle notebook with T4 GPU accelerator
2. Attach your adapter as a dataset:
   - Go to "Add Data" → "Datasets" → upload your medgemma_silicosis_v2/adapter/ folder
   - OR use the Kaggle CLI:  kaggle datasets create -p ./medgemma_silicosis_v2
3. Paste this entire file into a single code cell and run it
4. Copy the ngrok URL printed at the end
5. Paste it into the app at http://localhost:5000

DATASET PATHS (update if different):
  - Base model : google/medgemma-4b-it  (downloaded automatically from HuggingFace)
  - Adapter    : /kaggle/input/<YOUR-DATASET-NAME>/adapter/

NGROK TOKEN:
  - Sign up free at https://ngrok.com
  - Copy your auth token from https://dashboard.ngrok.com/get-started/your-authtoken
  - Paste it below

Run time: ~3-4 minutes to load model + start server
"""

# ── Cell 1: Install dependencies ──────────────────────────────────────────────
import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

install("pyngrok")
install("flask")
install("bitsandbytes")
install("peft")
install("transformers>=4.46.0")
install("accelerate")

# ── Cell 2: Imports & Config ──────────────────────────────────────────────────
import os
import base64
import traceback
from io import BytesIO
from threading import Thread

import torch
from PIL import Image
from flask import Flask, request, jsonify

from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel

# ── !! UPDATE THESE !! ────────────────────────────────────────────────────────
NGROK_AUTH_TOKEN = "YOUR_NGROK_AUTH_TOKEN_HERE"   # <- paste your ngrok token

# Path to the adapter folder inside your Kaggle dataset
# Example: if you named your dataset "medgemma-silicosis-adapter",
#   the path is: /kaggle/input/medgemma-silicosis-adapter/adapter
ADAPTER_PATH = "/kaggle/input/medgemma-silicosis-adapter/adapter"

# HuggingFace token — TWO options (use whichever is easier):
#
# OPTION A (Recommended — secure): Add your HF token as a Kaggle Secret
#   Notebook → Add-ons → Secrets → "HF_TOKEN" → paste your token → Attach
#   Then leave the line below as None:
HF_TOKEN = None   # ← set to None to use Kaggle Secret (recommended)
#
# OPTION B (Quick): Paste your token directly here instead:
# HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
#
# Get your token from: https://huggingface.co/settings/tokens
# Make sure you have accepted MedGemma terms at:
#   https://huggingface.co/google/medgemma-4b-it

MODEL_ID = "google/medgemma-4b-it"
# ─────────────────────────────────────────────────────────────────────────────

# Resolve HF token (Kaggle Secret takes priority)
try:
    from kaggle_secrets import UserSecretsClient
    _secret_client = UserSecretsClient()
    HF_TOKEN = _secret_client.get_secret("HF_TOKEN")
    print("[Auth] HF token loaded from Kaggle Secrets")
except Exception:
    if HF_TOKEN is None:
        raise RuntimeError(
            "HF_TOKEN not found. Either:\n"
            "  A) Add it via Notebook -> Add-ons -> Secrets (name: HF_TOKEN)\n"
            "  B) Set HF_TOKEN = 'hf_...' directly in the config section above"
        )
    print("[Auth] HF token loaded from config (consider using Kaggle Secrets)")

# Log in to HuggingFace Hub explicitly — required for gated repos like MedGemma
from huggingface_hub import login as hf_login
hf_login(token=HF_TOKEN, add_to_git_credential=False)
print(f"[Auth] HuggingFace login OK")


USER_PROMPT_TEMPLATE = """You are an expert radiologist reviewing a frontal chest X-ray from a stone worker with occupational silica dust exposure.

=== AI CLASSIFIER OUTPUTS ===
Primary Silicosis Risk : {binary_risk_pct}% ({binary_risk_category}) — {binary_label}

Secondary Radiological Findings:
{findings_text}

=== MANDATORY RULES ===
1. Every finding listed as CONFIRMED PRESENT above MUST be explicitly named in your FINDINGS section.
2. Do NOT contradict CONFIRMED PRESENT findings. If the classifier says a finding is PRESENT, you MUST report it as present.
3. The Primary Risk Score drives the IMPRESSION:
   - HIGH (>=84%): conclude consistent with silicosis.
   - BORDERLINE (50-84%): note uncertainty, recommend HRCT.
   - LOW (<50%): do not suggest silicosis.
4. Silicosis reference: bilateral upper-lobe rounded opacities (ILO p/q/r), +/- hilar eggshell calcification, +/- progressive massive fibrosis.

Write a concise structured report with exactly these three sections:

EXAMINATION:
Frontal chest X-ray. Indication: occupational lung disease screening — stone worker with silica dust history.

FINDINGS:
[State each CONFIRMED PRESENT finding by name and describe its radiographic appearance (zone, distribution, character). Then briefly describe cardiac silhouette, mediastinum, pleura, and diaphragm.]

IMPRESSION:
[State whether findings are consistent or inconsistent with silicosis based on the primary risk score. Recommend appropriate next steps: HRCT chest, pulmonary function tests, occupational health referral, or 12-month follow-up CXR.]"""


def load_medgemma():
    """Load MedGemma 4B in 4-bit quantization + LoRA adapter."""
    print(f"\n{'='*60}")
    print("Loading MedGemma 4B (4-bit quantized) ...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading processor from adapter path...")
    processor = AutoProcessor.from_pretrained(
        ADAPTER_PATH,
        token=HF_TOKEN,
    )

    print("Downloading MedGemma base model (this takes ~5 min on first run)...")
    base_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    print("Applying LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    print("MedGemma + adapter loaded successfully!")
    print(f"{'='*60}\n")
    return model, processor


# Load model
model, processor = load_medgemma()


def generate_report(image_pil, findings_text,
                    binary_risk_pct=None, binary_risk_category=None,
                    binary_label=None, max_new_tokens=500):
    """Generate a clinical radiology report from image + findings + binary risk."""
    if binary_risk_pct is None:
        binary_risk_pct = "N/A"
    if binary_risk_category is None:
        binary_risk_category = "N/A"
    if binary_label is None:
        binary_label = "N/A"

    user_prompt = USER_PROMPT_TEMPLATE.format(
        findings_text        = findings_text,
        binary_risk_pct      = binary_risk_pct,
        binary_risk_category = binary_risk_category,
        binary_label         = binary_label,
    )

    image = image_pil.convert("RGB").resize((336, 336))

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": user_prompt},
        ],
    }]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens       = max_new_tokens,
            do_sample            = True,
            temperature          = 0.3,          # low temp → focused but not greedy
            top_p                = 0.9,
            repetition_penalty   = 1.3,          # penalise repeated phrases heavily
            no_repeat_ngram_size = 5,            # forbid repeating any 5-gram
        )

    report = processor.decode(
        out[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )
    return report.strip()


# ── Flask API ─────────────────────────────────────────────────────────────────
api = Flask(__name__)


@api.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "medgemma-4b-it + silicosis-lora"})


@api.route("/generate-report", methods=["POST"])
def generate_report_endpoint():
    """
    POST body (JSON):
      findings_text : str   (required)
      image_b64     : str   (optional, base64 JPEG)
      zones_text    : str   (optional)
    """
    try:
        data = request.get_json(force=True)

        findings_text = data.get("findings_text", "")
        if not findings_text:
            return jsonify({"error": "findings_text is required"}), 400

        # Binary classifier risk (primary silicosis assessment)
        binary_risk_pct      = data.get("binary_risk_pct",      "N/A")
        binary_risk_category = data.get("binary_risk_category", "N/A")
        binary_label         = data.get("binary_label",         "N/A")

        # Decode image if provided
        image_b64 = data.get("image_b64", None)
        if image_b64:
            img_bytes = base64.b64decode(image_b64)
            image_pil = Image.open(BytesIO(img_bytes)).convert("RGB")
        else:
            image_pil = Image.new("RGB", (336, 336), color=(128, 128, 128))

        report = generate_report(
            image_pil,
            findings_text,
            binary_risk_pct      = binary_risk_pct,
            binary_risk_category = binary_risk_category,
            binary_label         = binary_label,
        )

        return jsonify({"report": report, "status": "ok"})

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[API ERROR] {e}\n{tb}")
        return jsonify({"error": str(e), "trace": tb}), 500


# ── Start ngrok tunnel ────────────────────────────────────────────────────────
from pyngrok import ngrok, conf

print("Starting ngrok tunnel ...")
conf.get_default().auth_token = NGROK_AUTH_TOKEN

# Kill any existing ngrok processes
ngrok.kill()

# Open tunnel on port 5001 (avoid conflict with Kaggle's port 5000)
tunnel = ngrok.connect(5001, bind_tls=True)
public_url = tunnel.public_url   # extract the clean https:// string

print(f"\n{'='*70}")
print(f"  *** MedGemma API is LIVE ***")
print(f"")
print(f"  YOUR URL (copy this exactly):")
print(f"  {public_url}")
print(f"")
print(f"  Paste it into: http://localhost:5000 -> Settings (top-right) -> Kaggle API URL")
print(f"")
print(f"  Quick test: {public_url}/health")
print(f"{'='*70}\n")

# Run Flask in background thread
def run_flask():
    api.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)

flask_thread = Thread(target=run_flask, daemon=True)
flask_thread.start()

# Keep the Kaggle cell alive (prevents session from dying)
print("Server is running. Keep this cell alive by checking output periodically.")
print("The API will remain active as long as this Kaggle session is open.")

# Import time and keep alive
import time
try:
    while True:
        time.sleep(300)  # ping every 5 minutes
        try:
            import requests as req
            req.get(f"http://localhost:5001/health", timeout=5)
            print(f"[{time.strftime('%H:%M:%S')}] Server heartbeat OK")
        except Exception:
            pass
except KeyboardInterrupt:
    print("Server stopped.")
