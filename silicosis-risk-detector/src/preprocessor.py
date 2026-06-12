"""
src/preprocessor.py

Image preprocessing pipeline for Silicosis Detection models.
Applies resize, custom LungMaskTransform (segmentation), and ImageNet normalization.

Pure Python/NumPy/OpenCV implementation (no PyTorch dependencies).
"""

import cv2
import numpy as np
from PIL import Image

class LungMaskTransform:
    """
    Custom image preprocessing to isolate lung fields and mask out the background.
    Matches the training preprocessor exactly but uses OpenCV and PIL/NumPy.
    """
    def __init__(self, fallback_border: float = 0.10):
        self.fallback_border = fallback_border

    def __call__(self, img_pil: Image.Image) -> Image.Image:
        w, h = img_pil.size
        img_np = np.array(img_pil.convert("L"))   # grayscale for mask detection

        # 1. Normalize -> CLAHE
        img_norm = cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_norm)

        # 2. Invert (lungs are dark -> make them bright)
        inverted = cv2.bitwise_not(enhanced)

        # 3. Otsu threshold
        _, thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 4. Morphological cleanup
        k_c = max(h // 15, 15)
        k_o = max(h // 25, 10)
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_c, k_c))
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_o, k_o))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_close)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k_open)

        # 5. Contour search - pick largest blob in each half
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = h * w * 0.025
        mid_x = w // 2
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

        # Fallback: mask the outer border if detection failed
        if mask.max() == 0:
            pad_x = int(w * self.fallback_border)
            pad_y = int(h * self.fallback_border)
            mask[pad_y: h - pad_y, pad_x: w - pad_x] = 255

        # 6. Smooth mask edges
        blur_k = max(h // 10, 11)
        if blur_k % 2 == 0:
            blur_k += 1
        mask_f = cv2.GaussianBlur(mask.astype(np.float32), (blur_k, blur_k), 0)
        if mask_f.max() > 0:
            mask_f /= mask_f.max()

        # 7. Apply mask: lung -> original pixel, background -> neutral gray (128)
        rgb = np.array(img_pil).astype(np.float32)
        bg = np.full_like(rgb, 128.0)
        m3 = mask_f[..., np.newaxis]
        masked = (m3 * rgb + (1.0 - m3) * bg).astype(np.uint8)
        return Image.fromarray(masked)


def preprocess_image(image_path: str, config: dict) -> np.ndarray:
    """
    Main preprocessing entrypoint for ONNX Runtime.
    
    Args:
        image_path: Path to the input image file
        config: Dict loaded from config.yaml
        
    Returns:
        np.ndarray: Preprocessed image tensor with shape (1, 3, target_height, target_width)
                    and dtype float32, normalized to ImageNet statistics.
    """
    target_w = config["image_settings"]["target_width"]
    target_h = config["image_settings"]["target_height"]

    # 1. Load image using PIL (RGB format)
    img = Image.open(image_path).convert("RGB")

    # 2. Resize first (matching the fast preprocessing path of training)
    img_resized = img.resize((target_w, target_h), Image.Resampling.BILINEAR)

    # 3. Apply custom lung mask segmentation
    mask_transform = LungMaskTransform()
    img_masked = mask_transform(img_resized)

    # 4. Convert to NumPy float32 and scale to [0, 1]
    img_np = np.array(img_masked).astype(np.float32) / 255.0

    # 5. Normalize with ImageNet statistics
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_normalized = (img_np - mean) / std

    # 6. Transpose from HWC (Height, Width, Channels) to CHW (Channels, Height, Width)
    img_chw = np.transpose(img_normalized, (2, 0, 1))

    # 7. Add batch dimension -> (1, 3, target_height, target_width)
    img_batch = np.expand_dims(img_chw, axis=0)

    return img_batch
