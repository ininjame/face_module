"""
Fine-tune NKF on personal calibration data.

Usage:
  python finetune_nkf.py \
      --weights nkf_best.pth \
      --data    calibration_data/ \
      --out     models/nkf_finetuned.pth \
      --onnx    models/nkf_finetuned.onnx

Calibration data folder layout (produced by calibration_collect.py):
  calibration_data/
      neutral/   *.jpg
      happy/     *.jpg
      angry/     *.jpg
      sad/       *.jpg
      surprise/  *.jpg
      disgust/   *.jpg
      fear/      *.jpg
      contempt/  *.jpg

Each subfolder only needs to exist if you collected samples for it.
20-50 samples per expression is enough for personalization.
"""

import argparse
import os
import pickle
import sys

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

# ── import shared config ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    ALL_LANDMARK_INDICES, CLASSES, N_CLASSES, N_LANDMARKS, RKFA_INDEX,
    IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, PERTURB_PIXELS, ROTATION_DEG,
    DROPOUT, FEAT_DIM,
)

PERTURB_NORM = PERTURB_PIXELS / IMG_SIZE   # normalised [0,1] units


# ── Model (mirrors nkf_model_v2 / nkf_model_update1) ─────────────────────────

class GlobalFeatureNetwork(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.stem   = nn.Sequential(backbone.conv1, backbone.bn1,
                                    backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4   # kept in module but not used in forward
        self.conv_layer2 = nn.Conv2d(512,  256, 3, stride=2, padding=1, bias=False)
        self.conv_layer3 = nn.Conv2d(1024, 512, 3, padding=1,            bias=False)
        self.conv_fuse   = nn.Conv2d(768,  768, 3, padding=1,            bias=False)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        f2 = self.layer2(x)            # 512 × 24 × 24
        f3 = self.layer3(f2)           # 1024 × 12 × 12
        f2_down = self.conv_layer2(f2) # 256  × 12 × 12
        f3_proj = self.conv_layer3(f3) # 512  × 12 × 12
        return self.conv_fuse(torch.cat([f2_down, f3_proj], dim=1))  # 768 × 12 × 12


def extract_keypoint_features(global_feat, landmark_coords, perturbation=0.0):
    B, C, H, W = global_feat.shape
    coords = landmark_coords.clone()
    if perturbation > 0 and global_feat.requires_grad:
        coords = coords + torch.zeros_like(coords).uniform_(-perturbation, perturbation)
    coords = coords.clamp(0, 1)
    grid = (coords * 2 - 1).unsqueeze(2)          # (B, N, 1, 2)
    kp   = F.grid_sample(global_feat, grid, mode='bilinear',
                          padding_mode='border', align_corners=True)  # (B, C, N, 1)
    return kp.squeeze(-1).permute(0, 2, 1)         # (B, N, C)


class RKFA(nn.Module):
    def __init__(self, feature_dim=FEAT_DIM, n_landmarks=N_LANDMARKS,
                 nasal_base_idx=RKFA_INDEX):
        super().__init__()
        self.nasal_base_idx = nasal_base_idx
        self.n_landmarks    = n_landmarks
        self.fc_attention   = nn.Linear(feature_dim * 2, n_landmarks)

    def forward(self, kp):
        B, N, C = kp.shape
        nasal   = kp[:, self.nasal_base_idx:self.nasal_base_idx+1, :].expand(B, N, C)
        flat    = torch.cat([nasal, kp], dim=-1).reshape(B * N, 2 * C)
        scores  = self.fc_attention(flat).view(B, N, N)
        weights = F.softmax(
            torch.stack([scores[:, i, i] for i in range(N)], dim=1), dim=1
        ).unsqueeze(-1)
        return kp + weights * kp


class NKF(nn.Module):
    def __init__(self, n_classes=N_CLASSES, n_landmarks=N_LANDMARKS,
                 feature_dim=FEAT_DIM, perturbation=PERTURB_NORM, pretrained=False):
        super().__init__()
        self.n_landmarks  = n_landmarks
        self.feature_dim  = feature_dim
        self.perturbation = perturbation
        self.global_net   = GlobalFeatureNetwork(pretrained=pretrained)
        self.rkfa         = RKFA(feature_dim, n_landmarks)
        self.classifier   = nn.Sequential(
            nn.Linear(n_landmarks * feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(feature_dim, n_classes),
        )

    def forward(self, images, landmark_coords):
        gf   = self.global_net(images)
        p    = self.perturbation if self.training else 0.0
        kp   = extract_keypoint_features(gf, landmark_coords, p)
        ref  = self.rkfa(kp)
        return self.classifier(ref.reshape(ref.shape[0], -1))


# ── Landmark extraction ───────────────────────────────────────────────────────

def make_mp_face():
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.4,
    )


def extract_landmarks(img_bgr, mp_face):
    rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result = mp_face.process(rgb)
    if not result.multi_face_landmarks:
        return None
    lm  = result.multi_face_landmarks[0].landmark
    pts = np.array([[lm[i].x, lm[i].y] for i in ALL_LANDMARK_INDICES], dtype=np.float32)
    return np.clip(pts, 0.0, 1.0)


def precompute_landmarks_for_dir(root_dir, cache_path):
    """Run MediaPipe once over all images; save .pkl cache."""
    mp_face = make_mp_face()
    cache, failed = {}, 0

    for cls_name in CLASSES:
        folder = os.path.join(root_dir, cls_name)
        if not os.path.isdir(folder):
            continue
        for fname in tqdm(os.listdir(folder), desc=f"landmarks/{cls_name}"):
            if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            p = os.path.join(folder, fname)
            img = cv2.imread(p)
            if img is None:
                continue
            pts = extract_landmarks(img, mp_face)
            if pts is None:
                pts = np.zeros((N_LANDMARKS, 2), dtype=np.float32)
                failed += 1
            cache[p] = pts

    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)
    print(f"Saved {len(cache)} entries to {cache_path}  (no-face: {failed})")
    return cache


# ── Dataset ───────────────────────────────────────────────────────────────────

class CalibrationDataset(Dataset):
    def __init__(self, root_dir, transform, landmark_cache):
        self.samples        = []
        self.transform      = transform
        self.landmark_cache = landmark_cache

        for label_idx, cls_name in enumerate(CLASSES):
            folder = os.path.join(root_dir, cls_name)
            if not os.path.isdir(folder):
                continue
            for fname in os.listdir(folder):
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.samples.append((os.path.join(folder, fname), label_idx))

        print(f"Dataset: {len(self.samples)} images across "
              f"{len(set(s[1] for s in self.samples))} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return (torch.zeros(3, IMG_SIZE, IMG_SIZE),
                    torch.zeros(N_LANDMARKS, 2), label)

        pts = self.landmark_cache.get(img_path,
              np.zeros((N_LANDMARKS, 2), dtype=np.float32))

        img_pil    = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        img_tensor = self.transform(img_pil)
        return img_tensor, torch.from_numpy(pts), label


# ── Fine-tuning ───────────────────────────────────────────────────────────────

def finetune(
    weights_path,
    data_dir,
    out_pth,
    out_onnx,
    epochs,
    lr,
    batch_size,
    freeze_backbone,
    device_str,
):
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}   Landmarks: {N_LANDMARKS}   Classes: {N_CLASSES}")

    for _p in (out_pth, out_onnx):
        if os.path.dirname(_p):
            os.makedirs(os.path.dirname(_p), exist_ok=True)

    # ── load pretrained weights ──
    print(f"Loading weights from {weights_path} …")
    model = NKF(pretrained=False)
    state = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model = model.to(device)

    # ── freeze backbone if requested ──
    if freeze_backbone:
        for name, p in model.named_parameters():
            if name.startswith('global_net'):
                p.requires_grad_(False)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        print(f"Backbone frozen.  Trainable params: {trainable:,} / {total:,}")
    else:
        print("Full model unfrozen (all parameters trainable)")

    # ── landmark cache ──
    cache_path = os.path.join(data_dir, '_landmarks_cache.pkl')
    if os.path.exists(cache_path):
        print(f"Loading existing landmark cache from {cache_path}")
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        # validate shape
        sample = next(iter(cache.values()))
        if sample.shape[0] != N_LANDMARKS:
            print(f"Cache mismatch ({sample.shape[0]} vs {N_LANDMARKS}) — rebuilding")
            cache = precompute_landmarks_for_dir(data_dir, cache_path)
    else:
        cache = precompute_landmarks_for_dir(data_dir, cache_path)

    # ── transforms ──
    # NOTE: no flip / rotation — they move the image but not the landmarks,
    # so the keypoint features would be sampled from the wrong places.
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # ── train/val split (80/20) ──
    full_ds  = CalibrationDataset(data_dir, train_tf, cache)
    n        = len(full_ds)
    val_n    = max(1, int(0.2 * n))
    train_n  = n - val_n
    g        = torch.Generator().manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_n, val_n], generator=g)

    # override val transform on the val split
    val_base    = CalibrationDataset(data_dir, val_tf, cache)
    val_indices = val_ds.indices
    val_ds      = torch.utils.data.Subset(val_base, val_indices)

    # ── weighted sampler for class balance ──
    labels_all = [full_ds.samples[i][1] for i in range(n)]
    train_labels = [labels_all[i] for i in train_ds.indices]
    class_counts = np.bincount(train_labels, minlength=N_CLASSES).astype(float)
    class_counts = np.where(class_counts == 0, 1, class_counts)
    weights      = 1.0 / class_counts
    sample_wts   = torch.tensor([weights[l] for l in train_labels], dtype=torch.float)
    sampler      = WeightedRandomSampler(sample_wts, len(sample_wts), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    # ── optimizer — only update trainable params ──
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    def evaluate(loader):
        """Return (accuracy, confusion matrix [true, pred])."""
        model.eval()
        cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
        with torch.no_grad():
            for imgs, lms, lbls in loader:
                preds = model(imgs.to(device), lms.to(device)).argmax(1).cpu().numpy()
                for t, p_ in zip(lbls.numpy(), preds):
                    cm[t, p_] += 1
        total = cm.sum()
        return (np.trace(cm) / total if total else 0.0), cm

    def print_cm(cm, title):
        print(f"\n{title}  (rows = true, cols = predicted)")
        short = [c[:5] for c in CLASSES]
        print("          " + " ".join(f"{s:>5}" for s in short))
        for i, row in enumerate(cm):
            print(f"{short[i]:>8}  " + " ".join(f"{v:>5}" for v in row))

    # ── baseline: pretrained model, before any fine-tuning ──
    all_loader = DataLoader(CalibrationDataset(data_dir, val_tf, cache),
                            batch_size=batch_size, shuffle=False, num_workers=0)
    base_acc, base_cm = evaluate(all_loader)
    print(f"\nBaseline (pretrained, no fine-tune) accuracy on ALL calibration data: "
          f"{base_acc:.4f}   (chance = {1/N_CLASSES:.4f})")
    print_cm(base_cm, "Baseline confusion matrix")
    print()

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        if freeze_backbone:
            # keep frozen BatchNorm layers on their pretrained running stats
            model.global_net.eval()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for imgs, lms, lbls in tqdm(train_loader, desc=f"Ep {epoch:3d}/{epochs} train",
                                    leave=False):
            imgs, lms, lbls = imgs.to(device), lms.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = model(imgs, lms)
            loss   = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            tr_loss    += loss.item()
            tr_correct += (logits.argmax(1) == lbls).sum().item()
            tr_total   += lbls.size(0)

        # ── val ──
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for imgs, lms, lbls in val_loader:
                imgs, lms, lbls = imgs.to(device), lms.to(device), lbls.to(device)
                preds      = model(imgs, lms).argmax(1)
                v_correct += (preds == lbls).sum().item()
                v_total   += lbls.size(0)

        scheduler.step()
        val_acc = v_correct / v_total if v_total else 0.0
        print(f"Ep {epoch:3d}/{epochs}  "
              f"loss: {tr_loss/len(train_loader):.4f}  "
              f"train_acc: {tr_correct/tr_total:.4f}  "
              f"val_acc: {val_acc:.4f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_pth)
            print(f"  ↳ saved {out_pth}  (val_acc={val_acc:.4f})")

    print(f"\nBest val accuracy: {best_val_acc:.4f}")

    model.load_state_dict(torch.load(out_pth, map_location=device, weights_only=False))
    _, final_cm = evaluate(val_loader)
    print_cm(final_cm, "Fine-tuned confusion matrix (val split)")

    # ── export ONNX ──
    print(f"\nExporting to {out_onnx} …")
    model.eval()
    dummy_img = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    dummy_lm  = torch.zeros(1, N_LANDMARKS, 2,        device=device)
    torch.onnx.export(
        model,
        (dummy_img, dummy_lm),
        out_onnx,
        input_names=['image', 'landmarks'],
        output_names=['logits'],
        dynamic_axes={'image': {0: 'batch'}, 'landmarks': {0: 'batch'}},
        opset_version=14,
    )
    print(f"Done.  {out_onnx}  ({N_LANDMARKS} landmarks, {N_CLASSES} classes)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune NKF on personal calibration data")
    p.add_argument('--weights',          default='models/nkf_best.pth',
                   help='Path to pretrained .pth weights')
    p.add_argument('--data',             default='calibration_data',
                   help='Root folder of calibration images (subfolders = class names)')
    p.add_argument('--out',              default='models/nkf_finetuned.pth',
                   help='Output .pth path')
    p.add_argument('--onnx',             default='models/nkf_finetuned.onnx',
                   help='Output .onnx path')
    p.add_argument('--epochs',     type=int,   default=30)
    p.add_argument('--lr',         type=float, default=1e-4,
                   help='Learning rate (lower than original training; default 1e-4)')
    p.add_argument('--batch',      type=int,   default=8,
                   help='Batch size (small for tiny personal datasets)')
    p.add_argument('--freeze',     action='store_true', default=True,
                   help='Freeze ResNet50 backbone (default: True)')
    p.add_argument('--no-freeze',  dest='freeze', action='store_false',
                   help='Unfreeze full model (more capacity, needs more data)')
    p.add_argument('--device',     default='cuda',
                   help='cuda or cpu')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    finetune(
        weights_path   = args.weights,
        data_dir       = args.data,
        out_pth        = args.out,
        out_onnx       = args.onnx,
        epochs         = args.epochs,
        lr             = args.lr,
        batch_size     = args.batch,
        freeze_backbone= args.freeze,
        device_str     = args.device,
    )