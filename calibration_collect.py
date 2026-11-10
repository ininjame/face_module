"""
Calibration data collection for NKF fine-tuning.

Controls:
  1–8   : set current expression label (starts capturing)
  0     : pause capture (no label active)
  SPACE : save one frame manually
  Q     : quit

Auto-captures one frame every CAPTURE_EVERY frames while a label is active.
Each saved frame is cropped to the face exactly like the tracker does
(CROP_PADDING / CROP_SQUARE in config.py):
  calibration_data_cropped/<expression>/frame_XXXXX.jpg   -> use for finetune_nkf.py
With --keep-full, the uncropped frame is also saved (lets you re-crop later
with recrop_calibration.py):
  calibration_data/<expression>/frame_XXXXX.jpg

Usage:
  python calibration_collect.py [--keep-full]
"""

import argparse
import os

import cv2
import mediapipe as mp
import numpy as np

import config as C
from utils.recrop_calibration import get_face_crop   # single source of truth for the crop

# ── Config ────────────────────────────────────────────────────────────────────
CROP_DIR       = "calibration_data_cropped"
FULL_DIR       = "calibration_data"
CAPTURE_EVERY  = 10          # auto-save one frame every N frames while labelling
CAM_INDEX      = C.CAM_INDEX
CAM_W, CAM_H   = C.CAM_W, C.CAM_H
TARGET_PER_CLS = 50          # show a progress bar toward this many per class

CLASSES = C.CLASSES          # same order as the tracker's override keys
KEY_MAP = {ord(str(i+1)): name for i, name in enumerate(CLASSES)}  # '1'→'neutral', etc.
OUTPUT_DIR = CROP_DIR        # counts / progress are based on the cropped set

PROMPT_MESSAGES = {
    "neutral":  "Look straight ahead, relaxed face",
    "happy":    "Smile! Show your teeth if natural",
    "angry":    "Frown, brow down, jaw tight",
    "sad":      "Frown, mouth corners down, look down slightly",
    "surprise": "Raise eyebrows high, open mouth wide",
    "disgust":  "Wrinkle nose, raise upper lip",
    "fear":     "Raise brows, widen eyes, open mouth slightly",
    "contempt": "One-sided smirk, slight head tilt",
}

# Colours
COL_BG     = (30, 30, 30)
COL_ACTIVE = (0, 200, 80)
COL_PAUSE  = (180, 180, 180)
COL_DONE   = (0, 140, 255)
COL_WHITE  = (255, 255, 255)
COL_DARK   = (50, 50, 50)

# ── Helpers ───────────────────────────────────────────────────────────────────

def ensure_dirs(keep_full):
    for cls in CLASSES:
        os.makedirs(os.path.join(CROP_DIR, cls), exist_ok=True)
        if keep_full:
            os.makedirs(os.path.join(FULL_DIR, cls), exist_ok=True)


def count_saved():
    counts = {}
    for cls in CLASSES:
        folder = os.path.join(OUTPUT_DIR, cls)
        counts[cls] = len([f for f in os.listdir(folder)
                           if f.lower().endswith(('.jpg', '.png'))])
    return counts


def next_frame_path(cls_name, counts):
    n = counts.get(cls_name, 0)
    return os.path.join(OUTPUT_DIR, cls_name, f"frame_{n:05d}.jpg")


def save_frame(frame, crop, cls_name, counts, keep_full):
    """Save the face crop (and optionally the full frame). Returns path or None."""
    if crop is None:
        return None
    path = next_frame_path(cls_name, counts)
    cv2.imwrite(path, crop)
    if keep_full:
        cv2.imwrite(os.path.join(FULL_DIR, cls_name, os.path.basename(path)), frame)
    counts[cls_name] = counts.get(cls_name, 0) + 1
    return path


def draw_hud(canvas, frame, active_cls, counts, frame_idx, capture_every, flash):
    h, w = canvas.shape[:2]
    fh, fw = frame.shape[:2]

    # paste webcam feed (left side)
    canvas[:fh, :fw] = frame

    # right panel background
    panel_x = fw + 10
    canvas[:, fw:] = COL_BG

    y = 20
    # title
    cv2.putText(canvas, "NKF CALIBRATION", (panel_x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_WHITE, 2)
    y += 30
    cv2.line(canvas, (panel_x, y), (w - 10, y), COL_DARK, 1)
    y += 18

    # current label
    if active_cls:
        col = COL_ACTIVE if not flash else (0, 255, 255)
        cv2.putText(canvas, f"CAPTURING: {active_cls.upper()}", (panel_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        y += 22
        prompt = PROMPT_MESSAGES.get(active_cls, "")
        # word-wrap prompt into ~28-char lines
        words, line = prompt.split(), ""
        for word in words:
            if len(line) + len(word) + 1 > 28:
                cv2.putText(canvas, line, (panel_x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, COL_PAUSE, 1)
                y += 18
                line = word
            else:
                line = (line + " " + word).strip()
        if line:
            cv2.putText(canvas, line, (panel_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, COL_PAUSE, 1)
            y += 18
    else:
        cv2.putText(canvas, "PAUSED  (press 1-8)", (panel_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_PAUSE, 1)
        y += 22

    y += 8
    cv2.line(canvas, (panel_x, y), (w - 10, y), COL_DARK, 1)
    y += 16

    # progress bars per class
    cv2.putText(canvas, "Progress:", (panel_x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_WHITE, 1)
    y += 18
    bar_w = w - panel_x - 10
    for i, cls in enumerate(CLASSES):
        n   = counts.get(cls, 0)
        pct = min(n / TARGET_PER_CLS, 1.0)
        key = str(i + 1)
        is_active = (cls == active_cls)

        # label
        label_col = COL_ACTIVE if is_active else COL_WHITE
        cv2.putText(canvas, f"{key}:{cls[:8]:<8} {n:3d}", (panel_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, label_col, 1)

        # bar
        bx = panel_x + 110
        by = y - 10
        cv2.rectangle(canvas, (bx, by), (bx + bar_w - 115, by + 10), COL_DARK, -1)
        filled = int((bar_w - 115) * pct)
        bar_col = COL_DONE if n >= TARGET_PER_CLS else (COL_ACTIVE if is_active else (80, 150, 80))
        if filled > 0:
            cv2.rectangle(canvas, (bx, by), (bx + filled, by + 10), bar_col, -1)

        y += 20

    y += 6
    cv2.line(canvas, (panel_x, y), (w - 10, y), COL_DARK, 1)
    y += 16

    # controls
    cv2.putText(canvas, "Controls:", (panel_x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_WHITE, 1)
    y += 18
    for text in ["1-8 : set expression", "0   : pause", "SPC : save one frame", "Q   : quit"]:
        cv2.putText(canvas, text, (panel_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_PAUSE, 1)
        y += 16

    # auto-capture countdown
    if active_cls:
        countdown = capture_every - (frame_idx % capture_every)
        cv2.putText(canvas, f"next auto-save in {countdown} frames",
                    (panel_x, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_ACTIVE, 1)

    # flash border when saving
    if flash:
        cv2.rectangle(canvas, (0, 0), (fw - 1, fh - 1), (0, 255, 255), 4)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Collect cropped calibration frames")
    ap.add_argument("--keep-full", action="store_true",
                    help=f"also save uncropped frames to {FULL_DIR}/")
    args = ap.parse_args()

    ensure_dirs(args.keep_full)
    counts = count_saved()
    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        refine_landmarks=False, min_detection_confidence=0.5)
    print(f"Crops -> {CROP_DIR}/   (padding={C.CROP_PADDING}, "
          f"square={getattr(C, 'CROP_SQUARE', False)})")
    if args.keep_full:
        print(f"Full frames -> {FULL_DIR}/")
    print(f"Existing counts: { {k: v for k, v in counts.items() if v > 0} }")

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    if not cap.isOpened():
        print(f"ERROR: cannot open camera {CAM_INDEX}")
        return

    active_cls  = None
    frame_idx   = 0
    flash_until = 0   # show flash indicator for N frames after save

    panel_w = 240
    canvas  = None

    print("\nPress 1-8 to start capturing for an expression, 0 to pause, Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed — exiting")
            break

        if C.FRAME_MIRRORED:
            frame = cv2.flip(frame, 1)   # mirror, same as the tracker
        fh, fw = frame.shape[:2]

        # face crop, computed on the clean frame
        res  = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        crop = None
        display = frame.copy()
        if res.multi_face_landmarks:
            pts = np.array([[p.x * fw, p.y * fh]
                            for p in res.multi_face_landmarks[0].landmark], np.float32)
            crop = get_face_crop(frame, pts)
            if crop is not None:
                # draw the crop box on the display copy only
                x0, y0 = pts.min(0); x1, y1 = pts.max(0)
                cv2.rectangle(display, (int(x0), int(y0)), (int(x1), int(y1)), (0, 200, 80), 1)
        if crop is None:
            cv2.putText(display, "NO FACE - not saving", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if canvas is None:
            canvas = cv2.UMat(fh, fw + panel_w + 10, cv2.CV_8UC3) \
                     if False else \
                     __import__('numpy').zeros((fh, fw + panel_w + 10, 3),
                                               dtype=__import__('numpy').uint8)

        flash = frame_idx < flash_until

        # auto-capture
        saved_this_frame = False
        if active_cls and (frame_idx % CAPTURE_EVERY == 0):
            if save_frame(frame, crop, active_cls, counts, args.keep_full):
                flash_until = frame_idx + 3
                saved_this_frame = True

        draw_hud(canvas, display, active_cls, counts, frame_idx, CAPTURE_EVERY,
                 flash or saved_this_frame)
        cv2.imshow("NKF Calibration Collector", canvas)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord('0'):
            active_cls = None
            print("Paused")
        elif key == ord(' '):
            if active_cls:
                path = save_frame(frame, crop, active_cls, counts, args.keep_full)
                if path:
                    flash_until = frame_idx + 3
                    print(f"  Saved {path}  (total {active_cls}: {counts[active_cls]})")
                else:
                    print("  No face detected — not saved")
            else:
                print("  No label active — press 1-8 first")
        elif key in KEY_MAP:
            active_cls = KEY_MAP[key]
            print(f"Label → {active_cls}  ({counts.get(active_cls, 0)} saved so far)")

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    print("\n── Final counts ──")
    total = 0
    for cls in CLASSES:
        n = counts.get(cls, 0)
        bar = "█" * (n // 2) + ("░" * max(0, TARGET_PER_CLS // 2 - n // 2))
        print(f"  {cls:<10} {n:4d}  {bar}")
        total += n
    print(f"\n  Total: {total} images saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
