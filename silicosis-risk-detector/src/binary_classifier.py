"""
Binary Silicosis Classifier
Architecture: EfficientNet-B4, custom head → 2-class CrossEntropy
Model file  : efficientnet_b4_vindr_best.pth

Output interpretation:
  softmax(logits)[0] = P(Normal/Non-Silicosis)
  softmax(logits)[1] = P(Silicosis)  ← silicosis_risk_score
"""

import cv2
import numpy as np
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


class LungMaskTransform:
    """
    Custom PIL → PIL transform applied AFTER resizing for inference.
    Suppresses non-lung pixels to neutral gray (128).
    """
    def __init__(self, fallback_border: float = 0.10):
        self.fallback_border = fallback_border

    def __call__(self, img_pil: Image.Image) -> Image.Image:
        w, h   = img_pil.size
        img_np = np.array(img_pil.convert("L"))

        img_norm = cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_norm)
        inverted = cv2.bitwise_not(enhanced)

        _, thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        k_c = max(h // 15, 15)
        k_o = max(h // 25, 10)
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_c, k_c))
        k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_o, k_o))
        thresh  = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_close)
        thresh  = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  k_open)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = h * w * 0.025
        mid_x    = w // 2
        left_cnts, right_cnts = [], []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            (left_cnts if cx < mid_x else right_cnts).append((area, cnt))

        mask = np.zeros((h, w), dtype=np.uint8)
        for bucket in (left_cnts, right_cnts):
            if bucket:
                _, best = max(bucket, key=lambda x: x[0])
                cv2.drawContours(mask, [best], -1, 255, cv2.FILLED)

        if mask.max() == 0:
            pad_x = int(w * self.fallback_border)
            pad_y = int(h * self.fallback_border)
            mask[pad_y: h - pad_y, pad_x: w - pad_x] = 255

        blur_k = max(h // 10, 11)
        if blur_k % 2 == 0:
            blur_k += 1
        mask_f = cv2.GaussianBlur(mask.astype(np.float32), (blur_k, blur_k), 0)
        if mask_f.max() > 0:
            mask_f /= mask_f.max()

        rgb    = np.array(img_pil).astype(np.float32)
        bg     = np.full_like(rgb, 128.0)
        m3     = mask_f[..., np.newaxis]
        masked = (m3 * rgb + (1.0 - m3) * bg).astype(np.uint8)
        return Image.fromarray(masked)


# Inference transform matches training: Resize FIRST, then apply LungMaskTransform
VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    LungMaskTransform(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def _build_binary_model():
    model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, 2),
    )
    return model


def load_binary_classifier(model_path):
    model = _build_binary_model()
    state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    print(f"[BinaryClassifier] Loaded from {model_path}")
    return model


def predict_binary(model, image_path):
    """
    Run binary classification.
    Returns:
        risk_score   : float [0,1] — probability of silicosis
        label        : str  — 'Silicosis Suspected' | 'Low Risk'
        risk_category: str  — 'HIGH' | 'BORDERLINE' | 'LOW'
        probs        : dict — {normal: float, silicosis: float}
    """
    img = Image.open(image_path).convert("RGB")
    tensor = VAL_TRANSFORM(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()

    normal_prob   = probs[0]
    silicosis_prob = probs[1]

    # Thresholds adjusted to Youden optimal cutoff (0.8423)
    if silicosis_prob >= 0.8423:
        label = "Silicosis Suspected"
        category = "HIGH"
    elif silicosis_prob >= 0.50:
        label = "Borderline — Review Recommended"
        category = "BORDERLINE"
    else:
        label = "Low Risk"
        category = "LOW"

    return {
        "risk_score"    : round(silicosis_prob, 4),
        "risk_pct"      : round(silicosis_prob * 100, 1),
        "label"         : label,
        "risk_category" : category,
        "probs"         : {"normal": round(normal_prob, 4), "silicosis": round(silicosis_prob, 4)},
    }


def binary_gradcam(model, image_path):
    """
    Generate GradCAM overlay for the binary classifier (class_idx=1 = silicosis).
    Returns dict with overlay_b64, original_b64, probs.
    """
    return generate_gradcam_b64(
        model=model,
        image_path=image_path,
        device=DEVICE,
        transform=VAL_TRANSFORM,
        class_idx=1,  # Always visualise silicosis class
        img_size=IMG_SIZE,
    )
