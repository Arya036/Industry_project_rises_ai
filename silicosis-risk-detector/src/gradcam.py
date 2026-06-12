"""
GradCAM implementation for EfficientNet-B4
Works on both binary classifier (2-class) and finding classifier (8-class).

Post-hoc lung masking is applied after CAM generation so heatmap
attention is suppressed in non-lung regions (mediastinum, background,
image markers, body silhouette).
"""

import numpy as np
import cv2
import torch
import base64
from io import BytesIO
from PIL import Image


# ─── Lung mask extraction ──────────────────────────────────────────────────────

def extract_lung_mask(pil_img, img_size):
    """
    Estimate the lung region from a chest X-ray using classical image processing.

    Strategy:
      1. Convert to grayscale and enhance contrast with CLAHE
      2. Invert image (lungs are dark on CXR → bright after inversion)
      3. Otsu threshold to separate lung from background
      4. Morphological cleanup (close small gaps, open noise)
      5. Find the two largest blobs in the left and right halves
      6. Smooth mask edges with Gaussian blur for natural look

    If detection fails completely (all-black mask), returns a fallback
    that covers the central 70% of the image.

    Returns:
        lung_mask: float32 array [0, 1], shape (img_size, img_size)
    """
    # Resize to working size, convert to grayscale
    img_gray = np.array(pil_img.resize((img_size, img_size)).convert("L"))
    h, w = img_gray.shape

    # ── Step 1: Normalize & CLAHE ──
    img_norm = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_norm)

    # ── Step 2: Invert (lungs dark → bright) ──
    inverted = cv2.bitwise_not(enhanced)

    # ── Step 3: Otsu threshold ──
    _, thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ── Step 4: Morphological cleanup ──
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_close)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  k_open)

    # ── Anatomical pre-filter: strip frame/marker regions BEFORE contour search ──
    # Scanner annotations (L/R markers, dates, borders) sit in the outer 12% top,
    # 15% bottom, and 4% left/right strips. Zeroing these prevents them from being
    # mistaken for a lung candidate contour.
    thresh[: int(h * 0.12), :]  = 0   # above clavicles  (marker zone)
    thresh[int(h * 0.85) :, :] = 0   # below diaphragm
    thresh[:, : int(w * 0.06)] = 0   # left film border (6% — tighter lateral)
    thresh[:, int(w * 0.94) :] = 0   # right film border

    # ── Step 5: Find left/right lung candidates ──
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = h * w * 0.025   # must be at least 2.5% of image area
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
            cv2.drawContours(mask, [best], -1, 255, thickness=cv2.FILLED)

    # ── Fallback: centre-crop if detection failed ──
    if mask.max() == 0:
        fallback = np.zeros((h, w), dtype=np.uint8)
        pad_x = int(w * 0.10)
        pad_y = int(h * 0.08)
        fallback[pad_y: h - pad_y, pad_x: w - pad_x] = 255
        mask = fallback

    # ── Anatomical constraint: harden mask at non-lung zones ──
    # Belt-and-suspenders: even if a stray contour slipped through,
    # zero the same strips in the filled mask before blurring.
    mask[: int(h * 0.12), :]  = 0
    mask[int(h * 0.85) :, :] = 0
    mask[:, : int(w * 0.06)] = 0
    mask[:, int(w * 0.94) :] = 0

    # ── Step 6: Smooth edges (tighter kernel — less boundary bleed) ──
    mask_f = cv2.GaussianBlur(mask.astype(np.float32), (25, 25), 0)
    if mask_f.max() > 0:
        mask_f /= mask_f.max()

    return mask_f


# ─── GradCAM core ─────────────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Target layer: model.features[-1] (last convolutional block of EfficientNet-B4).
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        """
        Generate a GradCAM heatmap.
        Args:
            input_tensor : (1, 3, H, W) preprocessed tensor on device
            class_idx    : which output neuron to backprop through
                           None = argmax (auto-select predicted class)
        Returns:
            cam       : (H, W) float array normalized to [0, 1]
            class_idx : int
            probs     : numpy array of raw model outputs
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        # Global-average-pool gradients → channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, h, w]
        cam = torch.relu(cam).squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        probs = output.detach().cpu().numpy()[0]
        return cam, class_idx, probs


# ─── Overlay helpers ───────────────────────────────────────────────────────────

def overlay_heatmap(original_pil, cam, img_size=380, alpha=0.50, lung_mask=None):
    """
    Resize CAM, apply optional lung mask, then overlay as JET heatmap.

    Args:
        original_pil : PIL Image (original X-ray)
        cam          : (H, W) float array [0, 1]
        img_size     : output square size
        alpha        : heatmap blend strength
        lung_mask    : float array [0,1] same size as output, or None

    Returns:
        overlay : (img_size, img_size, 3) uint8 numpy array
    """
    img_np = np.array(original_pil.resize((img_size, img_size)))
    if img_np.ndim == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 4:
        img_np = img_np[:, :, :3]

    cam_resized = cv2.resize(cam, (img_size, img_size))

    # ── Apply lung mask to suppress non-lung heatmap ──
    # Use mask**3 for even steeper falloff: boundary values of 0.3 → 0.027
    # (vs 0.09 with ^2), making lateral pleural edge bleed nearly invisible.
    # Interior lung values (0.9) become 0.73 — still strong.
    if lung_mask is not None:
        cam_resized = cam_resized * (lung_mask ** 3)

    # Re-normalize after masking so colour scale uses full range inside lungs
    if cam_resized.max() > 0:
        cam_resized = cam_resized / cam_resized.max()

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap  = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend: outside mask → show only original; inside mask → show heatmap
    if lung_mask is not None:
        alpha_map = (lung_mask * alpha)[..., np.newaxis]          # (H, W, 1)
        overlay   = (alpha_map * heatmap + (1 - alpha_map) * img_np).astype(np.uint8)
    else:
        overlay = (alpha * heatmap + (1 - alpha) * img_np).astype(np.uint8)

    return overlay


def numpy_to_base64(img_np):
    """Convert numpy RGB image to base64 PNG string."""
    pil = Image.fromarray(img_np)
    buf = BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def pil_to_base64(pil_img, fmt="PNG"):
    """Convert PIL image to base64 string."""
    buf = BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ─── Full pipeline ─────────────────────────────────────────────────────────────

def generate_gradcam_b64(model, image_path, device, transform,
                         class_idx=None, img_size=380):
    """
    Full pipeline: image_path → lung-masked GradCAM overlay → base64 PNG.

    1. Load image
    2. Run GradCAM
    3. Extract lung mask from the original X-ray
    4. Apply mask to suppress non-lung heatmap regions
    5. Return overlay + original as base64 strings

    Returns dict with keys: overlay_b64, original_b64, class_idx, probs
    """
    original_pil = Image.open(image_path).convert("RGB")
    input_tensor = transform(original_pil).unsqueeze(0).to(device)

    target_layer = model.features[-1]
    gradcam = GradCAM(model, target_layer)

    cam, pred_class, probs = gradcam.generate(input_tensor, class_idx=class_idx)

    # Extract lung mask
    try:
        lung_mask = extract_lung_mask(original_pil, img_size)
    except Exception as e:
        print(f"[GradCAM] Lung mask extraction failed ({e}), skipping mask.")
        lung_mask = None

    overlay_np = overlay_heatmap(original_pil, cam,
                                 img_size=img_size,
                                 lung_mask=lung_mask)

    overlay_b64  = numpy_to_base64(overlay_np)
    original_b64 = pil_to_base64(original_pil.resize((img_size, img_size)))

    return {
        "overlay_b64" : overlay_b64,
        "original_b64": original_b64,
        "class_idx"   : int(pred_class),
        "probs"        : probs.tolist(),
    }
