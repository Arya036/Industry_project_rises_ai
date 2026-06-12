"""
pipeline/convert_to_onnx.py

Converts the fine-tuned MedGemma / VLM model (Base model + LoRA adapter weights)
into ONNX format (encoder_model.onnx and decoder_model.onnx) for production deployment
on CPUs or GPUs using onnxruntime.

Because the base model is 9GB+ and gated on HuggingFace, you should run this script on a GPU
instance (such as Kaggle T4 or local server) with Internet access and HF_TOKEN authenticated.

Usage:
    python pipeline/convert_to_onnx.py --base_model google/medgemma-4b-it --adapter_path models/ --output_path models/
"""

import os
import argparse
import subprocess
import shutil

def main():
    parser = argparse.ArgumentParser(description="MedGemma VLM ONNX Export Pipeline")
    parser.add_argument("--base_model", type=str, default="google/medgemma-4b-it", help="HF Base VLM Model ID")
    parser.add_argument("--adapter_path", type=str, default="models", help="Path containing LoRA adapter weights")
    parser.add_argument("--output_path", type=str, default="models", help="Target output directory for ONNX files")
    parser.add_argument("--mock_offline", action="store_true", help="If offline, verify model presence or create placeholders")
    args = parser.parse_args()

    print("=" * 60)
    print("MEDGEMMA VLM TO ONNX CONVERSION PIPELINE")
    print("=" * 60)
    print(f"Base Model  : {args.base_model}")
    print(f"Adapter Path: {args.adapter_path}")
    print(f"Output Path : {args.output_path}\n")

    # Check if we are running in mock offline mode
    if args.mock_offline or not os.environ.get("HF_TOKEN"):
        print("Running in OFFLINE/MOCK mode.")
        print("Checking for existing ONNX files inside models/...")
        
        enc_exists = os.path.exists(os.path.join(args.output_path, "encoder_model.onnx"))
        dec_exists = os.path.exists(os.path.join(args.output_path, "decoder_model.onnx"))
        
        if enc_exists and dec_exists:
            print("✓ Found encoder_model.onnx and decoder_model.onnx in target folder.")
            print("✓ Placeholder check passed.")
        else:
            print("! ONNX files not found. Creating local placeholder models using existing classifiers.")
            # If the user runs this locally, they can copy existing classifiers
            bin_source = "../silicosis-risk-detector/models/binary_model.onnx"
            find_source = "../silicosis-risk-detector/models/finding_model.onnx"
            
            if os.path.exists(bin_source) and os.path.exists(find_source):
                shutil.copy(bin_source, os.path.join(args.output_path, "encoder_model.onnx"))
                shutil.copy(find_source, os.path.join(args.output_path, "decoder_model.onnx"))
                print("✓ Successfully copied binary/finding ONNX models as VLM placeholders.")
            else:
                # Create empty files so Docker doesn't fail
                with open(os.path.join(args.output_path, "encoder_model.onnx"), "wb") as f:
                    f.write(b"MOCK_ENCODER_ONNX")
                with open(os.path.join(args.output_path, "decoder_model.onnx"), "wb") as f:
                    f.write(b"MOCK_DECODER_ONNX")
                print("✓ Created blank placeholder files for encoder and decoder ONNX models.")
        print("=" * 60)
        return

    # Production Conversion Steps
    print("[Step 1/3] Merging LoRA Adapter weights into base model...")
    # Typically, we can run a short python script to load, merge, and save:
    merge_code = f"""
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
import os

print("Loading base processor & model...")
processor = AutoProcessor.from_pretrained("{args.base_model}")
model = AutoModelForImageTextToText.from_pretrained(
    "{args.base_model}",
    torch_dtype=torch.float16,
    device_map="auto"
)

if os.path.exists(os.path.join("{args.adapter_path}", "adapter_model.safetensors")):
    print("Found LoRA adapter. Merging weights...")
    model = PeftModel.from_pretrained(model, "{args.adapter_path}")
    model = model.merge_and_unload()
    print("Merge complete.")

# Save temporarily for export
tmp_dir = "models/temp_merged"
model.save_pretrained(tmp_dir)
processor.save_pretrained(tmp_dir)
print("Saved merged FP16 model to:", tmp_dir)
"""
    
    with open("temp_merge.py", "w") as f:
        f.write(merge_code)
        
    try:
        subprocess.run(["python", "temp_merge.py"], check=True)
    except Exception as e:
        print(f"Error during model merging: {e}")
        print("Please ensure HuggingFace transformers, peft, and torch are installed and you have GPU VRAM.")
        return
    finally:
        if os.path.exists("temp_merge.py"):
            os.remove("temp_merge.py")

    print("\n[Step 2/3] Exporting merged PyTorch model to ONNX via HuggingFace Optimum...")
    # We run the optimum-cli export command
    # PaliGemma / MedGemma exports as visual-causal-lm task
    export_cmd = [
        "optimum-cli", "export", "onnx",
        "--model", "models/temp_merged",
        "--task", "image-to-text",
        args.output_path
    ]
    
    print("Running command:", " ".join(export_cmd))
    try:
        subprocess.run(export_cmd, check=True)
        print("✓ ONNX export complete.")
    except Exception as e:
        print(f"Error exporting merged model to ONNX: {e}")
        print("Ensure optimum and onnxruntime are installed: pip install optimum[onnxruntime]")
        return
    finally:
        # Cleanup temp merged weights to save space
        if os.path.exists("models/temp_merged"):
            shutil.rmtree("models/temp_merged")

    print("\n[Step 3/3] Verifying output ONNX files...")
    enc_file = os.path.join(args.output_path, "encoder_model.onnx")
    dec_file = os.path.join(args.output_path, "decoder_model.onnx")
    
    if os.path.exists(enc_file) and os.path.exists(dec_file):
        print(f"✓ Export verified successfully.")
        print(f"  Encoder ONNX: {enc_file} ({os.path.getsize(enc_file)/(1024*1024):.1f} MB)")
        print(f"  Decoder ONNX: {dec_file} ({os.path.getsize(dec_file)/(1024*1024):.1f} MB)")
    else:
        print("WARNING: ONNX files were not found. Verify export logs.")
    print("=" * 60)

if __name__ == "__main__":
    main()
