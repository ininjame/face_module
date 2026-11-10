"""
Convert full-frame calibration images into face crops, using exactly the
same crop as face_tracker_manual.py (MediaPipe mesh bounding box + CROP_PADDING).

Originals are left untouched; crops go to a separate folder.

Usage:
  python utils/recrop_calibration.py                        # calibration_data -> calibration_data_cropped
  python utils/recrop_calibration.py --src X --dst Y
Then:
  python utils/finetune_nkf.py --data calibration_data_cropped
"""
import argparse
import os
import sys

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# allow running as `python utils/<script>.py` from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C


def get_face_crop(frame, pts, padding=None, square=None):
    """Must stay identical to face_tracker_manual.get_face_crop."""
    padding = C.CROP_PADDING if padding is None else padding
    square  = getattr(C, "CROP_SQUARE", False) if square is None else square
    h, w = frame.shape[:2]
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    px, py = (x1 - x0) * padding, (y1 - y0) * padding
    x0, y0, x1, y1 = x0 - px, y0 - py, x1 + px, y1 + py
    if square:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        s = max(x1 - x0, y1 - y0) / 2
        x0, y0, x1, y1 = cx - s, cy - s, cx + s, cy + s
    x0, y0 = int(max(0, x0)), int(max(0, y0))
    x1, y1 = int(min(w, x1)), int(min(h, y1))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    return frame[y0:y1, x0:x1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="calibration_data")
    ap.add_argument("--dst", default="calibration_data_cropped")
    ap.add_argument("--padding", type=float, default=None,
                    help="override config.CROP_PADDING")
    ap.add_argument("--square", action="store_true", default=None,
                    help="square crop (overrides config.CROP_SQUARE)")
    args = ap.parse_args()

    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=False, min_detection_confidence=0.4)

    total, skipped = 0, 0
    for cls in C.CLASSES:
        src_dir = os.path.join(args.src, cls)
        if not os.path.isdir(src_dir):
            continue
        dst_dir = os.path.join(args.dst, cls)
        os.makedirs(dst_dir, exist_ok=True)
        files = sorted(f for f in os.listdir(src_dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
        for f in tqdm(files, desc=cls):
            img = cv2.imread(os.path.join(src_dir, f))
            if img is None:
                skipped += 1
                continue
            h, w = img.shape[:2]
            res = mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not res.multi_face_landmarks:
                skipped += 1
                continue
            pts = np.array([[p.x * w, p.y * h]
                            for p in res.multi_face_landmarks[0].landmark], np.float32)
            crop = get_face_crop(img, pts, args.padding, args.square)
            if crop is None:
                skipped += 1
                continue
            cv2.imwrite(os.path.join(dst_dir, f), crop)
            total += 1

    print(f"\nSaved {total} crops to {args.dst}/  (skipped: {skipped})")


if __name__ == "__main__":
    main()
