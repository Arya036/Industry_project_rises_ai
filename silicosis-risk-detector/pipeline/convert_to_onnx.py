"""
pipeline/convert_to_onnx.py

Converts trained EfficientNet-B4 PyTorch weights (.pth) to ONNX format
for CPU deployment via onnxruntime inside the Docker container.

Run this ONCE on your machine (or on Kaggle) before final delivery.
Requires: torch, torchvision, efficientnet_pytorch (or timm)

Usage:
    python pipeline/convert_to_onnx.py
"""

import sys

# Force standard streams to use UTF-8 on Windows to avoid emoji/Unicode encoding crashes
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

import torch
import torch.nn as nn
from collections import OrderedDict


# ── Helper: Build the EfficientNet-B4 architecture ───────────────────
# This must match EXACTLY how you defined the model during training.

def build_binary_model():
    """
    Binary silicosis classifier.
    Architecture: EfficientNet-B4, ImageNet pretrained, binary output.
    Classifier head: features → Dropout(0.5) → Linear → Dropout(0.3) → Linear(2)
    """
    from torchvision.models import efficientnet_b4
    model = efficientnet_b4(weights=None)   # no pretrained weights — we load our own

    # Replace classifier head to match what you trained with
    in_features = model.classifier[1].in_features   # 1792 for B4
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, 2)
    )
    return model


def build_finding_model():
    """
    Multi-label finding classifier.
    Architecture: EfficientNet-B4, ImageNet pretrained, 8-class output.
    Classifier head: features → Dropout(0.5) → Linear(1792,512) → ReLU → Dropout(0.3) → Linear(512,8)
    """
    from torchvision.models import efficientnet_b4
    model = efficientnet_b4(weights=None)

    in_features = model.classifier[1].in_features   # 1792 for B4
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, 8)
    )
    return model


# ── Conversion function ───────────────────────────────────────────────

def convert_to_onnx(model, weights_path, output_path, input_size=380, model_name="model"):
    """
    Load weights into model and export to ONNX.

    Args:
        model:        PyTorch nn.Module (architecture already defined)
        weights_path: path to .pth file
        output_path:  path for output .onnx file
        input_size:   image size (both width and height — 380 for EfficientNet-B4)
        model_name:   label for print messages
    """
    print(f"\n[{model_name}] Loading weights from: {weights_path}")

    # Load checkpoint — handle both raw state_dict and wrapped checkpoints
    checkpoint = torch.load(weights_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint   # assume it is the raw state dict

    # Handle optional DataParallel "module." prefix
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    state_dict = new_state_dict

    model.load_state_dict(state_dict)
    model.eval()
    print(f"[{model_name}] Weights loaded OK.")

    # Dummy input: batch=1, RGB, 380x380
    dummy_input = torch.randn(1, 3, input_size, input_size)

    print(f"[{model_name}] Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        }
    )
    print(f"[{model_name}] Export complete → {output_path}")

    # Consolidate external weights back into a single self-contained .onnx file
    import onnx
    import os
    print(f"[{model_name}] Consolidating external weight data into a single ONNX file...")
    onnx_model = onnx.load(output_path)
    onnx.save_model(onnx_model, output_path, save_as_external_data=False)
    
    # Remove the temporary external data file if it exists
    data_file = output_path + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)
        print(f"[{model_name}] Cleaned up temporary external data file: {data_file}")

    # Quick sanity check
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[{model_name}] Consolidated file size: {size_mb:.1f} MB")
    if size_mb < 5:
        print(f"[{model_name}] WARNING: File is very small ({size_mb:.1f} MB). "
              "Weights may not have exported. Check state_dict loading above.")
    else:
        print(f"[{model_name}] Size looks correct. Run Netron check to confirm.")


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Binary model ──────────────────────────────────────────────────
    # EDIT THESE PATHS if your .pth files are elsewhere
    BINARY_WEIGHTS = "models/efficientnet_b4_merged_best.pth"
    BINARY_ONNX    = "models/binary_model.onnx"

    binary_model = build_binary_model()
    convert_to_onnx(
        model        = binary_model,
        weights_path = BINARY_WEIGHTS,
        output_path  = BINARY_ONNX,
        input_size   = 380,
        model_name   = "BinaryClassifier"
    )

    # ── Finding model ─────────────────────────────────────────────────
    # EDIT THIS PATH if your .pth file has a different name
    FINDING_WEIGHTS = "models/finding_model_final.pth"
    FINDING_ONNX    = "models/finding_model.onnx"

    finding_model = build_finding_model()
    convert_to_onnx(
        model        = finding_model,
        weights_path = FINDING_WEIGHTS,
        output_path  = FINDING_ONNX,
        input_size   = 380,
        model_name   = "FindingClassifier"
    )

    print("\n" + "="*60)
    print("ONNX conversion complete.")
    print("Next step: drag both .onnx files into netron.app to verify.")
    print("Expected file sizes: ~70-80 MB each.")
    print("="*60)
