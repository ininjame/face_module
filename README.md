# Face Tracker (GPU)

Real-time facial expression tracker using MediaPipe landmarks + NKF (Neural Keypoint Fusion) expression classifier. Outputs per-frame face state as JSON — over stdout or UDP.

Detailed documentation: [Notion](https://app.notion.com/p/Facial-expression-module-354ab3a8232680d6b865f3903a646e5e?source=copy_link)

Support files: [Drive](https://drive.google.com/drive/folders/15odwPd6iYvTcjMY5yauyeUygpzeto9Zq?usp=drive_link)

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download model weights

Download the following files from the support file folder and place them in this folder:

| File | Description |
| --- | --- |
| `nkf.onnx.data` | Pretrained NKF model — weights (sidecar, must be in the same folder as `nkf.onnx`) |

---

## Run the face tracker

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

## License

MIT License — see LICENSE for details.
