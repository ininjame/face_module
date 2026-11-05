# Face Tracker (GPU)

Real-time facial expression tracker using MediaPipe landmarks + NKF (Neural Keypoint Fusion) expression classifier. Outputs per-frame face state as JSON — over stdout or UDP.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download model weights

Download the following files and place them in this folder:

| File | Description |
| --- | --- |
| `nkf_best.pth` | Pretrained NKF weights (PyTorch) — needed for fine-tuning |
| `nkf.onnx` | Pretrained NKF model (ONNX) — used directly by the face tracker |

> If you fine-tune on your own face (step 2 below), the tracker will use `nkf_finetuned.onnx` instead.

---

## Order of Execution

### Step 1 — Collect calibration data (optional, for personalization)

```bash
python calibration_collect.py
```

Sit in front of your webcam and record yourself doing each expression.

**Controls:**

| Key | Action |
| --- | --- |
| `1`–`8` | Set active expression label (neutral, happy, angry, sad, surprise, disgust, fear, contempt) |
| `0` | Pause capture |
| `Space` | Save one frame manually |
| `Q` / `ESC` | Quit |

Auto-saves one frame every 10 frames while a label is active. Aim for **~50 images per expression**. You can do them in any order and switch freely. Progress bars are shown in the HUD — capture continues past 100% if you want more diversity.

Output is saved to `calibration_data/<expression>/frame_XXXXX.jpg`.

---

### Step 2 — Fine-tune the model on your face (optional)

```bash
python finetune_nkf.py \
    --weights nkf_best.pth \
    --data    calibration_data/ \
    --out     nkf_finetuned.pth \
    --onnx    nkf_finetuned.onnx
```

**All arguments (defaults shown):**

| Argument | Default | Description |
| --- | --- | --- |
| `--weights` | `nkf_best.pth` | Pretrained weights to start from |
| `--data` | `calibration_data/` | Folder with collected images |
| `--out` | `nkf_finetuned.pth` | Where to save the fine-tuned checkpoint |
| `--onnx` | `nkf_finetuned.onnx` | Where to export the final ONNX model |
| `--epochs` | `30` | Number of training epochs |
| `--lr` | `1e-4` | Learning rate |
| `--batch` | `8` | Batch size |
| `--freeze` / `--no-freeze` | `--freeze` | Freeze ResNet50 backbone (recommended) |
| `--device` | `cuda` | `cuda` or `cpu` |

The script freezes the ResNet50 backbone by default and only updates the RKFA attention + classifier head — suitable for small personal datasets. Exports `nkf_finetuned.onnx` automatically at the end.

---

### Step 3 — Run the face tracker

```bash
# With base model (no fine-tuning)
python face_tracker_manual.py --model nkf.onnx

# With your fine-tuned model
python face_tracker_manual.py --model nkf_finetuned.onnx

# Stream FaceState JSON over UDP
python face_tracker_manual.py --model nkf_finetuned.onnx --udp 127.0.0.1:5005

# Headless (no GUI window)
python face_tracker_manual.py --model nkf_finetuned.onnx --no-gui --udp 127.0.0.1:5005

# Reduce output rate (emit every Nth frame)
python face_tracker_manual.py --model nkf_finetuned.onnx --udp 127.0.0.1:5005 --output-every 3
```

**Operator controls (GUI mode):**

| Key | Action |
| --- | --- |
| `1`–`8` | Manual expression override (shown in orange, confidence = 1.0) |
| `0` | Clear override — return to auto classifier (shown in green) |
| `R` | Toggle video recording → `recording_YYYYMMDD_HHMMSS.mp4` |
| `Q` / `ESC` | Quit |

---

## Output Format

Each frame with a detected face emits one JSON object:

```json
{
  "face_detected": true,
  "expression": "happy",
  "confidence": 0.87,
  "eye_open_left": 0.82,
  "eye_open_right": 0.79,
  "blinking_left": false,
  "blinking_right": false,
  "brow_raise_left": 0.31,
  "brow_raise_right": 0.28,
  "mouth_open": false,
  "mouth_open_norm": 0.12,
  "mouth_width_norm": 0.65
}
```

If no face is detected, no output is emitted for that frame — consumers should hold the last known state.

---

## Configuration

All thresholds and camera settings are in `config.py`. Key values:

| Parameter | Description |
| --- | --- |
| `CLASSIFY_EVERY_N` | Run NKF classifier every N frames (default 15, ~2 fps) |
| `MIN_CONFIDENCE` | Minimum confidence to accept a new expression (default 0.4) |
| `CAM_INDEX` | Camera index (default 0) |
| `CAM_W` / `CAM_H` / `CAM_FPS` | Camera resolution and frame rate |
| `FRAME_MIRRORED` | Flip frame horizontally (true for front-facing webcam) |
