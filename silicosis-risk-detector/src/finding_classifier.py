"""
Multi-label Finding Classifier
Architecture: EfficientNet-B4, custom head → 8-class sigmoid
Model file  : finding_model_final.pth

Outputs 8 independent probabilities, one per radiological finding.
Per-class thresholds from K-fold AUC reliability analysis.
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from torchvision import transforms
from PIL import Image

from .gradcam import generate_gradcam_b64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 380
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

FINDING_CLASSES = [
    "Multiple Nodules",
    "Hilum Abnormality",
    "Consolidation",
    "Fibrosis",
    "Cavity",
    "Bronchiectasis",
    "Ground Glass Opacity",
    "Pleural Thickening",
]
NUM_FINDINGS = len(FINDING_CLASSES)

# Per-class thresholds — calibrated from K-fold AUC scores
PER_CLASS_THRESHOLDS = {
    "Multiple Nodules"    : 0.50,
    "Hilum Abnormality"   : 0.90,
    "Consolidation"       : 0.65,
    "Fibrosis"            : 0.55,
    "Cavity"              : 0.50,
    "Bronchiectasis"      : 0.65,
    "Ground Glass Opacity": 0.70,
    "Pleural Thickening"  : 0.60,
}

# K-fold cross-validation AUC for each finding
# (used to determine model reliability — NOT clinical specificity for silicosis)
FINDING_AUC = {
    "Consolidation"       : 0.84,   # highest AUC → HIGH
    "Cavity"              : 0.78,   # second highest → HIGH
    "Fibrosis"            : 0.68,   # moderate AUC → MODERATE
    "Multiple Nodules"    : 0.66,   # moderate AUC → MODERATE
    "Bronchiectasis"      : 0.65,   # moderate AUC → MODERATE  (was LOW — corrected)
    "Pleural Thickening"  : 0.67,   # moderate AUC → MODERATE  (was LOW — corrected)
    "Ground Glass Opacity": 0.55,   # near-chance → LOW
    "Hilum Abnormality"   : 0.48,   # near-random → LOW (below chance!)
}

# Reliability tier: HIGH ≥ 0.75 | MODERATE 0.60–0.75 | LOW < 0.60
FINDING_CONFIDENCE = {
    "Consolidation"       : "HIGH",
    "Cavity"              : "HIGH",
    "Fibrosis"            : "MODERATE",
    "Multiple Nodules"    : "MODERATE",
    "Bronchiectasis"      : "MODERATE",   # corrected from LOW
    "Pleural Thickening"  : "MODERATE",   # corrected from LOW
    "Ground Glass Opacity": "LOW",
    "Hilum Abnormality"   : "LOW",
}

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def _build_finding_model():
    model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, NUM_FINDINGS),
    )
    return model


def load_finding_classifier(model_path):
    model = _build_finding_model()
    state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    print(f"[FindingClassifier] Loaded from {model_path}")
    return model


def predict_findings(model, image_path):
    """
    Run multi-label finding classification.
    Returns:
        findings_text  : str  — formatted string for MedGemma prompt
        findings_list  : list of dicts — detailed per-finding data for frontend
        prob_dict      : dict — {finding: probability}
        present_findings : list of str — names of findings above threshold
    """
    img = Image.open(image_path).convert("RGB")
    tensor = VAL_TRANSFORM(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0].cpu().tolist()

    findings_list    = []
    present_lines    = []
    absent_lines     = []
    present_findings = []
    prob_dict        = {}

    for i, fname in enumerate(FINDING_CLASSES):
        p    = float(probs[i])
        thr  = PER_CLASS_THRESHOLDS[fname]
        conf = FINDING_CONFIDENCE[fname]
        auc  = FINDING_AUC[fname]
        present = p >= thr

        if present:
            present_findings.append(fname)
            present_lines.append(f"  - {fname}: PRESENT [{conf} confidence] ({p:.1%})")
        else:
            absent_lines.append(f"  - {fname}: not detected ({p:.1%})")

        prob_dict[fname] = round(p, 4)
        findings_list.append({
            "name"       : fname,
            "prob"       : round(p, 4),
            "prob_pct"   : round(p * 100, 1),
            "threshold"  : thr,
            "present"    : present,
            "confidence" : conf,
            "auc"        : auc,
        })

    # Structured text: PRESENT findings listed prominently so MedGemma cannot miss them
    if present_lines:
        findings_text = (
            "CONFIRMED PRESENT (MUST be stated in your FINDINGS section):\n"
            + "\n".join(present_lines)
            + "\n\nNOT detected (below threshold — do NOT report as present):\n"
            + "\n".join(absent_lines)
        )
    else:
        findings_text = (
            "CONFIRMED PRESENT: None above detection threshold.\n\n"
            "NOT detected:\n"
            + "\n".join(absent_lines)
        )

    return {
        "findings_text"    : findings_text,
        "findings_list"    : findings_list,
        "prob_dict"        : prob_dict,
        "present_findings" : present_findings,
    }


def finding_gradcam(model, image_path, finding_name):
    """
    Generate GradCAM overlay focused on a specific finding.
    class_idx = index of finding in FINDING_CLASSES.
    """
    if finding_name not in FINDING_CLASSES:
        raise ValueError(f"Unknown finding: {finding_name}")
    class_idx = FINDING_CLASSES.index(finding_name)

    return generate_gradcam_b64(
        model=model,
        image_path=image_path,
        device=DEVICE,
        transform=VAL_TRANSFORM,
        class_idx=class_idx,
        img_size=IMG_SIZE,
    )


def finding_gradcam_all_present(model, image_path, present_findings):
    """
    Generate GradCAM for all present findings.
    Returns dict: {finding_name: overlay_b64}
    """
    results = {}
    for fname in present_findings:
        try:
            cam_data = finding_gradcam(model, image_path, fname)
            results[fname] = cam_data["overlay_b64"]
        except Exception as e:
            print(f"[FindingGradCAM] Failed for {fname}: {e}")
    return results
