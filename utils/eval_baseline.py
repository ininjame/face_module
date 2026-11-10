"""
Sanity-check a pretrained NKF checkpoint on a labelled image folder
(e.g. the AffectNet test split it was trained on).

Usage:
  python utils/eval_baseline.py --data /media/ininjame/JensFormat/facemesh_test/affectnet/Test --per-class 100
"""
import argparse
import os
import random
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# allow running as `python utils/<script>.py` from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CLASSES, N_CLASSES, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD
from finetune_nkf import NKF, make_mp_face, extract_landmarks

# AffectNet folder names that differ from config.CLASSES
ALIASES = {"angry": ["angry", "anger"]}


def find_folder(root, cls):
    for name in ALIASES.get(cls, [cls]):
        p = os.path.join(root, name)
        if os.path.isdir(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/nkf_best.pth")
    ap.add_argument("--data", required=True)
    ap.add_argument("--per-class", type=int, default=100)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NKF(pretrained=False)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=False))
    model.to(device).eval()

    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    mp_face = make_mp_face()
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    no_face = 0
    sizes = []

    for label, cls in enumerate(CLASSES):
        folder = find_folder(args.data, cls)
        if folder is None:
            print(f"  (no folder for {cls})")
            continue
        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        random.Random(0).shuffle(files)
        for f in files[: args.per_class]:
            img = cv2.imread(os.path.join(folder, f))
            if img is None:
                continue
            sizes.append(img.shape[:2])
            pts = extract_landmarks(img, mp_face)
            if pts is None:
                no_face += 1
                continue
            x = tf(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
            lm = torch.from_numpy(pts).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = int(model(x, lm).argmax(1))
            cm[label, pred] += 1

    acc = np.trace(cm) / max(cm.sum(), 1)
    print(f"\nAccuracy: {acc:.4f}  on {cm.sum()} images  (no-face skipped: {no_face}, chance = {1/N_CLASSES:.4f})")
    if sizes:
        hs, ws = zip(*sizes)
        print(f"Image sizes: h {min(hs)}-{max(hs)}, w {min(ws)}-{max(ws)}")
    short = [c[:5] for c in CLASSES]
    print("\n(rows = true, cols = predicted)")
    print("          " + " ".join(f"{s:>5}" for s in short))
    for i, row in enumerate(cm):
        print(f"{short[i]:>8}  " + " ".join(f"{v:>5}" for v in row))


if __name__ == "__main__":
    main()
