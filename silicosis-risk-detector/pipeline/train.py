# NOTE: This script is designed to run on Kaggle with T4 GPU.
# Local execution requires CUDA GPU with 16GB+ VRAM.

"""
EfficientNet-B4 Binary Silicosis Classifier — Updated
Changes vs previous best (AUC 0.9078):
  1. VinDr "No finding" + safe non-silicosis images added as extra negatives
  2. VinDr negatives capped at 1500 to avoid overwhelming Silicodata
  3. Everything else identical to the run that gave AUC 0.9078
     (no CLAHE, same augmentation, same LR, same architecture)

Target: Improve specificity from 67.2% → 73-80% without hurting sensitivity.
"""

import os, re, copy, pickle
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT    = "/kaggle/input/datasets/aryajadhav03/silico/Silicodata_Updated_Feb2025/Silicodata_Updated_Feb2025"
SET_A_TRAIN  = os.path.join(DATA_ROOT, "set_A_folder", "train_images")
SET_A_TEST   = os.path.join(DATA_ROOT, "set_A_folder", "test_images")
SET_B_IMAGES = os.path.join(DATA_ROOT, "set_B_folder", "set_B_images")
SET_B_CSV    = os.path.join(DATA_ROOT, "set_B_folder", "Silicodata_SetB_labels.csv")

# VinDr JPG dataset — already attached to the notebook
VINDR_ROOT   = "/kaggle/input/datasets/sunghyunjun/vinbigdata-1024-jpg-dataset"
VINDR_LABELS = os.path.join(VINDR_ROOT, "train.csv")
VINDR_IMAGES = os.path.join(VINDR_ROOT, "train")

SAVE_DIR = "/kaggle/working/saved_models"
LOG_DIR  = "/kaggle/working/logs"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE       = 380
BATCH_SIZE     = 16
LR             = 1e-4
WEIGHT_DECAY   = 1e-4
MAX_EPOCHS     = 50
PATIENCE_LIMIT = 15
VAL_SPLIT      = 0.20
VINDR_NEG_CAP  = 1500   # max VinDr negatives to add — keep Silicodata dominant

# VinDr labels that are SAFE to use as Silicosis-Negative
# Do NOT include: Pulmonary fibrosis, Nodule/Mass, ILD, Infiltration, Consolidation
# — these overlap visually with silicosis
VINDR_SAFE_NEGATIVES = {
    "No finding",
    "Aortic enlargement",
    "Cardiomegaly",
    "Pleural effusion",
    "Atelectasis",
    "Pneumothorax",
    "Calcification",
    "Other lesion",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

print(f"Device     : {DEVICE}")
print(f"Image size : {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch size : {BATCH_SIZE}")

# ── Set B label mapping ───────────────────────────────────────────────────────
def impression_to_label(imp):
    """Returns 1 (positive) / 0 (negative) / -1 (exclude)."""
    if pd.isna(imp):
        return -1
    imp = str(imp).strip().lower()
    if "poor image" in imp:
        return -1
    if "silicosis" in imp or "silico" in imp:
        return 1
    if "tuberculosis" in imp or "tb" in imp or "hilar lymph" in imp:
        return 0
    if "normal" in imp:
        return 0
    return -1

# ── Lung Mask Transform ───────────────────────────────────────────────────────
class LungMaskTransform:
    """
    Custom PIL → PIL transform applied BEFORE any resize/normalisation.

    Steps:
      1. CLAHE for contrast enhancement
      2. Invert image  (lungs are dark on CXR, we want them bright)
      3. Otsu threshold
      4. Morphological close + open to remove noise
      5. Keep only the largest blob in the LEFT half + largest in RIGHT half
      6. Soft Gaussian-blurred mask edges
      7. Blend: lung region → original pixel, background → neutral gray (128)

    If detection fails completely, falls back to masking the outer 10% border
    so at minimum the image margins are suppressed.
    """
    def __init__(self, fallback_border: float = 0.10):
        self.fallback_border = fallback_border

    def __call__(self, img_pil: Image.Image) -> Image.Image:
        w, h   = img_pil.size
        img_np = np.array(img_pil.convert("L"))   # grayscale for mask detection

        # 1. Normalise → CLAHE
        img_norm = cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_norm)

        # 2. Invert (lungs are dark → make them bright)
        inverted = cv2.bitwise_not(enhanced)

        # 3. Otsu threshold
        _, thresh = cv2.threshold(inverted, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 4. Morphological cleanup — kernel size scales with image height
        k_c = max(h // 15, 15)
        k_o = max(h // 25, 10)
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_c, k_c))
        k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_o, k_o))
        thresh  = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_close)
        thresh  = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  k_open)

        # 5. Pick largest blob in each half
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
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

        # Fallback: suppress outer border if detection failed
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

        # 7. Apply: lung → original pixel, background → neutral gray
        rgb    = np.array(img_pil).astype(np.float32)          # (H, W, 3)
        bg     = np.full_like(rgb, 128.0)                       # neutral gray
        m3     = mask_f[..., np.newaxis]                        # (H, W, 1)
        masked = (m3 * rgb + (1.0 - m3) * bg).astype(np.uint8)
        return Image.fromarray(masked)


# ── Dataset ───────────────────────────────────────────────────────────────────
class BinaryDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

# IMPORTANT: Resize FIRST, then LungMaskTransform.
# LungMaskTransform on a 2048px original image takes 2-5s (huge kernel ops).
# On a 380px/412px image it takes <100ms — same mask quality, 50x faster.
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),     # resize to small first
    LungMaskTransform(),                                    # <-- NEW: mask non-lung
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),                # resize to small first
    LungMaskTransform(),                                    # <-- NEW: mask non-lung
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ── Data loading ──────────────────────────────────────────────────────────────
def load_set_a(split_dir):
    """Load Set A folder structure → (path, label) list."""
    folder_map = {
        "folder_normal"   : 0,
        "folder_TB"       : 0,
        "folder_silicosis": 1,
        "folder_STB"      : 1,
    }
    samples = []
    for folder, label in folder_map.items():
        d = os.path.join(split_dir, folder)
        if not os.path.isdir(d):
            print(f"  WARNING: {d} not found")
            continue
        count = 0
        for fname in os.listdir(d):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((os.path.join(d, fname), label))
                count += 1
        arrow = "POSITIVE" if label == 1 else "NEGATIVE"
        print(f"  {folder} → {arrow} : {count} images")
    return samples


def load_set_b():
    """Load Set B via CSV Impression column → (path, label) list."""
    df = pd.read_csv(SET_B_CSV)
    available = {
        os.path.splitext(f)[0].lower(): os.path.join(SET_B_IMAGES, f)
        for f in os.listdir(SET_B_IMAGES)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    }
    samples, skipped = [], 0
    for _, row in df.iterrows():
        pid   = str(row["patient_id"]).strip().lower()
        label = impression_to_label(row.get("Impression", ""))
        if label == -1 or pid not in available:
            skipped += 1
            continue
        samples.append((available[pid], label))
    print(f"  Set B: {len(samples)} usable  ({skipped} excluded)")
    pos = sum(1 for _, l in samples if l == 1)
    neg = len(samples) - pos
    print(f"    Positive: {pos}  Negative: {neg}")
    return samples


def load_vindr_negatives(cap=VINDR_NEG_CAP):
    """
    Load VinDr 'safe' negative images.
    Only images where ALL annotations for that image are in VINDR_SAFE_NEGATIVES.
    Caps at `cap` images to keep Silicodata dominant.
    """
    if not os.path.isfile(VINDR_LABELS):
        print("  VinDr labels not found — skipping.")
        return []

    df = pd.read_csv(VINDR_LABELS)

    # For each image_id, get all class names annotated
    image_classes = df.groupby("image_id")["class_name"].apply(set).to_dict()

    # Determine which image dir to use
    img_dir = VINDR_IMAGES if os.path.isdir(VINDR_IMAGES) else VINDR_ROOT
    available = {
        os.path.splitext(f)[0]: os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    }
    print(f"  VinDr image files found: {len(available)}")

    samples = []
    for iid, classes in image_classes.items():
        if iid not in available:
            continue
        # All annotations must be in safe set
        if classes.issubset(VINDR_SAFE_NEGATIVES):
            samples.append((available[iid], 0))   # label 0 = Silicosis-Negative
        if len(samples) >= cap:
            break

    print(f"  VinDr negatives selected: {len(samples)} (cap={cap})")
    return samples


def build_dataloaders():
    print("\n[1] Loading data ...")
    print("SET A TRAIN:")
    train_a = load_set_a(SET_A_TRAIN)
    print("\nSET B:")
    train_b = load_set_b()
    print("\nVINDR:")
    train_v = load_vindr_negatives(cap=VINDR_NEG_CAP)

    all_train_raw = train_a + train_b + train_v

    # Separate test set (Set A test — never used in training)
    print("\nSET A TEST:")
    test_samples = load_set_a(SET_A_TEST)

    # Stratified val split from merged train pool
    paths  = [s[0] for s in all_train_raw]
    labels = [s[1] for s in all_train_raw]
    tr_paths, vl_paths, tr_labels, vl_labels = train_test_split(
        paths, labels, test_size=VAL_SPLIT, stratify=labels, random_state=42
    )

    train_samples = list(zip(tr_paths, tr_labels))
    val_samples   = list(zip(vl_paths, vl_labels))

    print(f"\n  After val split:")
    print(f"    Train : {len(train_samples)}")
    print(f"    Val   : {len(val_samples)}")
    print(f"    Test  : {len(test_samples)}")
    print(f"  Train class: Pos={sum(1 for _,l in train_samples if l==1)}  "
          f"Neg={sum(1 for _,l in train_samples if l==0)}")

    # Weighted sampler to handle imbalance
    pos_count = sum(1 for _, l in train_samples if l == 1)
    neg_count = len(train_samples) - pos_count
    class_weights = [1.0 / neg_count if l == 0 else 1.0 / pos_count
                     for _, l in train_samples]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=class_weights, num_samples=len(train_samples), replacement=True
    )

    train_ds = BinaryDataset(train_samples, train_transform)
    val_ds   = BinaryDataset(val_samples,   val_test_transform)
    test_ds  = BinaryDataset(test_samples,  val_test_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    print("\n[2] Building EfficientNet-B4 ...")
    model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if any(l in name for l in ["features.6", "features.7", "features.8", "classifier"]):
            param.requires_grad = True

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, 2),
    )

    model = model.to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,}")
    return model


# ── Training ──────────────────────────────────────────────────────────────────
def train(model, train_loader, val_loader):
    print("\n[3] Training ...")
    criterion  = nn.CrossEntropyLoss()
    optimizer  = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    scaler     = GradScaler()

    best_auc     = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    patience     = 0
    history      = {"tr_loss": [], "vl_loss": [], "vl_auc": []}

    for epoch in range(1, MAX_EPOCHS + 1):
        # Train
        model.train()
        tr_loss = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                loss = criterion(model(imgs), lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_loss += loss.item() * imgs.size(0)
        tr_loss /= len(train_loader.dataset)

        # Validate
        model.eval()
        vl_loss, all_probs, all_lbls = 0.0, [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                with autocast():
                    out = model(imgs)
                vl_loss += criterion(out, lbls).item() * imgs.size(0)
                all_probs.append(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
                all_lbls.append(lbls.cpu().numpy())

        vl_loss  /= len(val_loader.dataset)
        all_probs = np.concatenate(all_probs)
        all_lbls  = np.concatenate(all_lbls)
        vl_auc    = roc_auc_score(all_lbls, all_probs)

        scheduler.step()
        history["tr_loss"].append(tr_loss)
        history["vl_loss"].append(vl_loss)
        history["vl_auc"].append(vl_auc)

        print(f"Epoch [{epoch:02d}/{MAX_EPOCHS}]  "
              f"TrLoss={tr_loss:.4f}  VlLoss={vl_loss:.4f}  VlAUC={vl_auc:.4f}")

        if vl_auc > best_auc:
            best_auc     = vl_auc
            best_weights = copy.deepcopy(model.state_dict())
            save_path    = os.path.join(SAVE_DIR, "efficientnet_b4_vindr_best.pth")
            torch.save(best_weights, save_path)
            print(f"    ✓ Best saved (AUC={best_auc:.4f}) → {save_path}")
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE_LIMIT:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_weights)
    return model, history


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, test_loader, threshold=0.5):
    model.eval()
    all_probs, all_preds, all_lbls = [], [], []

    with torch.no_grad():
        for imgs, lbls in test_loader:
            with autocast():
                out = model(imgs.to(DEVICE))
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            all_probs.append(probs)
            all_lbls.append(lbls.numpy())

    all_probs = np.concatenate(all_probs)
    all_lbls  = np.concatenate(all_lbls)
    all_preds = (all_probs >= threshold).astype(int)

    auc = roc_auc_score(all_lbls, all_probs)
    cm  = confusion_matrix(all_lbls, all_preds)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy    = (tp + tn) / len(all_lbls)

    print(f"\n{'='*60}")
    print("TEST RESULTS")
    print(f"{'='*60}")
    print(f"AUC          : {auc:.4f}")
    print(f"Accuracy     : {accuracy:.1%}")
    print(f"Sensitivity  : {sensitivity:.1%}  (TP={tp}, FN={fn})")
    print(f"Specificity  : {specificity:.1%}  (TN={tn}, FP={fp})")

    # Youden's J — find optimal threshold
    fpr, tpr, thresholds = roc_curve(all_lbls, all_probs)
    j_scores  = tpr - fpr
    best_idx  = np.argmax(j_scores)
    best_thr  = thresholds[best_idx]
    best_preds = (all_probs >= best_thr).astype(int)
    tn2, fp2, fn2, tp2 = confusion_matrix(all_lbls, best_preds).ravel()
    print(f"\nYouden threshold : {best_thr:.4f}")
    print(f"  Sensitivity    : {tp2/(tp2+fn2):.1%}  Specificity: {tn2/(tn2+fp2):.1%}")

    # Save ROC plot
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.scatter([fpr[best_idx]], [tpr[best_idx]], color="red",
                label=f"Youden thr={best_thr:.3f}", zorder=5)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("1 - Specificity"); plt.ylabel("Sensitivity")
    plt.title("Binary Silicosis Classifier — ROC")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "roc_binary_vindr.png"), dpi=150)
    plt.show()

    return auc, best_thr


# ── TTA inference ─────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_with_tta(model, image_path, n=10, threshold=0.811):
    """
    Test-time augmentation — averages n augmented passes.
    More stable on external data than single-pass inference.
    Use threshold=0.811 for clinical, 0.5 for mass screening.
    """
    img = Image.open(image_path).convert("RGB")
    probs_list = []
    for _ in range(n):
        tensor = train_transform(img).unsqueeze(0).to(DEVICE)
        with autocast():
            prob = torch.softmax(model(tensor), dim=1)[0, 1].item()
        probs_list.append(prob)
    avg = float(np.mean(probs_list))
    return {
        "silicosis_confidence": avg,
        "prediction": "Silicosis-Positive" if avg >= threshold else "Silicosis-Negative",
        "threshold_used": threshold,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_loader, val_loader, test_loader = build_dataloaders()
    model                                 = build_model()
    model, history                        = train(model, train_loader, val_loader)
    auc, youden_thr                       = evaluate(model, test_loader, threshold=0.5)

    print(f"\nFinal model saved: {SAVE_DIR}/efficientnet_b4_vindr_best.pth")
    print(f"Use threshold 0.5 for mass screening.")
    print(f"Use threshold {youden_thr:.3f} (Youden) for clinical use.")
