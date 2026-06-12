"""
pipeline/data_prep.py

Dataset loading, custom LungMaskTransform (segmentation), and data loader setup.
Extracted from training notebook for cleaner project structure.

Designed to run in training environment with PyTorch.
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class LungMaskTransform:
    """
    Custom PIL -> PIL transform applied BEFORE any resize/normalisation.
    Isolates lung region and fills background with neutral gray.
    """
    def __init__(self, fallback_border: float = 0.10):
        self.fallback_border = fallback_border

    def __call__(self, img_pil: Image.Image) -> Image.Image:
        w, h = img_pil.size
        img_np = np.array(img_pil.convert("L"))

        img_norm = cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_norm)
        inverted = cv2.bitwise_not(enhanced)

        _, thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        k_c = max(h // 15, 15)
        k_o = max(h // 25, 10)
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_c, k_c))
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_o, k_o))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_close)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k_open)

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

        rgb = np.array(img_pil).astype(np.float32)
        bg = np.full_like(rgb, 128.0)
        m3 = mask_f[..., np.newaxis]
        masked = (m3 * rgb + (1.0 - m3) * bg).astype(np.uint8)
        return Image.fromarray(masked)


class BinaryDataset(Dataset):
    """Dataset for training and validating binary silicosis classifier."""
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)
