# Face Tracker (GPU)

Real-time facial expression tracker using MediaPipe landmarks + NKF (Neural Keypoint Fusion) expression classifier. Outputs per-frame face state as JSON over UDP.

## Project layout

```
face_module_gpu/
├── face_tracker_manual.py   # real-time tracker (UDP output)
├── calibration_collect.py   # collect cropped calibration frames
├── finetune_nkf.py          # fine-tune the model on calibration data
├── config.py                # shared settings
├── models/                  # model files (.pth / .onnx / .onnx.data)
└── utils/                   # optional helpers (evaluation, re-cropping)
```

---

## Change notes

**2026-11-10**
- Added the personal calibration + fine-tuning workflow (`calibration_collect.py` → `finetune_nkf.py`). The collector saves face crops directly, using the same crop as the tracker.
- Added `MODEL_PROFILE` switch in `config.py`. Default is `"update1"`, which matches `nkf_best.pth` / `nkf_model_update1.onnx` (105 landmarks).
- Face crop is now square with 10% padding (`CROP_SQUARE = True`, `CROP_PADDING = 0.10`), used identically by the tracker and the fine-tuning data.
- Fixed: the tracker in GUI mode cropped the face *after* drawing the mesh overlay, so the classifier saw mesh lines on the face.
- Fixed in `finetune_nkf.py`: flip/rotation augmentation no longer used (it moved the image but not the landmarks), and the frozen backbone keeps its BatchNorm statistics.
- Pinned dependency versions (see Setup).
- Moved model files to `models/` and helper scripts to `utils/`.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` pins versions that are known to work together:

```
mediapipe==0.10.21          # newer releases removed the mp.solutions API used here
numpy<2
protobuf<5
onnx==1.17.0
opencv-contrib-python==4.11.0.86
onnxruntime-gpu
onnxscript                  # needed by torch.onnx.export in finetune_nkf.py
torch, torchvision, tqdm
```

Only have **one** OpenCV package installed (`opencv-contrib-python`). Having `opencv-python` too breaks `cv2`.

### 2. Model files

Ready-to-use models should be put into the `models/` folder, so you can run the tracker straight away.
Download the following files from this Drive link: [Drive](https://drive.google.com/drive/folders/15odwPd6iYvTcjMY5yauyeUygpzeto9Zq?usp=sharing)

| File | Description |
| --- | --- |
| `nkf_finetuned.onnx` | Base model fine-tuned on the author's face. Works for other people too, but best for the author. |
| `nkf_finetuned.onnx.data` | Data for fine-tuned base model. |
| `nkf_finetuned_sq.onnx` | A more square crop of the fine-tuned model. May work better (please test yourself.) |
| `nkf_finetuned_sq.onnx.data` | Data for fine-tuned square cropped model. |
| `nkf_best.pth` | Base NKF model trained on AffectNet (PyTorch weights). Starting point for fine-tuning. |

If an `.onnx` file comes with a `.onnx.data` file of the same name, keep both in `models/`. The `.data` file holds the weights.

For the best accuracy on **your** face, camera and lighting, fine-tune your own model (see [Personal calibration & fine-tuning](#personal-calibration--fine-tuning)). It takes about 10 minutes.

> **Model profile:** these models require `MODEL_PROFILE = "update1"` in `config.py` (the default). Don't change it unless you switch to a different model.

### 3. Select the camera

Set `CAM_INDEX` in `config.py`. To find the right index on Linux:

```bash
v4l2-ctl --list-devices       # sudo apt install v4l-utils
```

Use the **first** `/dev/videoN` listed under your camera. Laptop cameras often take `video0`–`video3`, so an external USB webcam may be `4`.

---

## Run the face tracker

```bash
# Base Fine-tuned model (--model optional, as tracker uses finetuned model by default)
python face_tracker_manual.py --model models/nkf_finetuned.onnx

# Stream FaceState JSON over UDP
python face_tracker_manual.py --model models/nkf_finetuned.onnx --udp 127.0.0.1:5005

# Headless (no GUI window)
python face_tracker_manual.py --model models/nkf_finetuned.onnx --no-gui --udp 127.0.0.1:5005

# Reduce output rate (emit every Nth frame)
python face_tracker_manual.py --model models/nkf_finetuned.onnx --udp 127.0.0.1:5005 --output-every 3
```

> **Integration note:** UDP is the current output method. A ROS 2 topic + message interface is planned as a future replacement.

To receive the output, run a UDP listener on the same port:

```python
import socket, json

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 5005))

while True:
    data, _ = sock.recvfrom(4096)
    state = json.loads(data.decode())
    print(state["expression"])
```

If running on the **same machine**, use `127.0.0.1`. If on **different machines on the same network**, use the sender's local IP (e.g. `192.168.1.x`).

**Operator controls (GUI mode):**

| Key | Action |
| --- | --- |
| `1`–`8` | Manual expression override (shown in orange, confidence = 1.0) |
| `0` | Clear override — return to auto classifier (shown in green) |
| `R` | Toggle video recording → `recording_YYYYMMDD_HHMMSS.mp4` |
| `Q` / `ESC` | Quit |

---

## Personal calibration & fine-tuning

The base model was trained on AffectNet photos of many people. Fine-tuning adapts it to one person's face, camera and lighting using a few hundred labelled frames. In testing this raised accuracy on the operator's face from ~20% (base model) to ~75%.

### Step 1 — Collect calibration frames

```bash
python calibration_collect.py
```

- Press `1`–`8` to choose an expression (same keys as the tracker's override), then hold that expression. A frame is saved automatically every 10 frames.
- Press `0` to pause, `SPACE` to save a single frame, `Q` to quit.
- Aim for **~50 frames per expression** (8 expressions). Move your head slightly and vary the intensity of the expression.
- Each frame is cropped to the face exactly like the tracker does (`CROP_PADDING` and `CROP_SQUARE` in `config.py`) and saved to `calibration_data_cropped/<expression>/`. Frames with no detected face are skipped ("NO FACE" on screen).
- The camera (`CAM_INDEX`) and mirroring (`FRAME_MIRRORED`) come from `config.py`.

Tips:
- Use the **same camera and lighting** you will use with the tracker.
- Recording across 2–3 short sessions generalises better than one long session.

### Step 2 — Fine-tune

```bash
python finetune_nkf.py --data calibration_data_cropped
```

- Starts from `models/nkf_best.pth`, freezes the ResNet50 backbone, and trains the attention + classifier head for 30 epochs (takes a few minutes on GPU).
- Before training it prints the **baseline** accuracy of the untouched model on your data. After training it prints a confusion matrix.
- Outputs `models/nkf_finetuned.pth` (PyTorch) and `models/nkf_finetuned.onnx` (for the tracker).

Common options:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--weights` | `models/nkf_best.pth` | Model to start from |
| `--data` | `calibration_data` | Folder with one subfolder per expression |
| `--out` / `--onnx` | `models/nkf_finetuned.pth` / `.onnx` | Output file names |
| `--epochs` | 30 | Training passes over the data |
| `--lr` | 1e-4 | Learning rate |
| `--batch` | 8 | Batch size |
| `--no-freeze` | (frozen) | Also train the backbone — only with much more data |

Use `--out` / `--onnx` to avoid overwriting the provided fine-tuned model, e.g. `--out models/nkf_finetuned_myname.pth --onnx models/nkf_finetuned_myname.onnx`.

### Step 3 — Run the tracker with your model

```bash
python face_tracker_manual.py --model models/nkf_finetuned.onnx
```

### Full workflow as one script

```bash
#!/usr/bin/env bash
set -e
python calibration_collect.py                 # interactive; press Q when done
python finetune_nkf.py --data calibration_data_cropped
python face_tracker_manual.py --model models/nkf_finetuned.onnx
```

> **Keep crop settings consistent.** A model works best with the crop settings it was fine-tuned with. If you change `CROP_PADDING` or `CROP_SQUARE`, collect new calibration data and fine-tune again.

---

## Utilities (`utils/`)

Optional helper scripts. Run them from the project root.

**`eval_baseline.py`**: score a model on any labelled image folder (one subfolder per expression). Prints accuracy and a confusion matrix.

```bash
python utils/eval_baseline.py --weights models/nkf_best.pth --data calibration_data_cropped --per-class 100
```

Useful for checking how well the base model already fits your data, or for testing it on AffectNet test images.

**`recrop_calibration.py`**: crop a folder of full camera frames into face crops, using `CROP_PADDING` / `CROP_SQUARE` from `config.py` (or `--padding` / `--square` to override).

```bash
python utils/recrop_calibration.py --src my_full_frames --dst my_crops --padding 0.1 --square
```

Together with `eval_baseline.py` this lets you compare crop settings: a higher baseline score means the crops look more like the base model's training data.

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

If no face is detected, no output is emitted for that frame. Consumers should hold the last known state.

---

## Configuration

All thresholds and camera settings are in `config.py`. Key values:

| Parameter | Description |
| --- | --- |
| `MODEL_PROFILE` | `"update1"` (default, 105 landmarks) or `"v2"` (97 landmarks, for `nkf.onnx`) |
| `CAM_INDEX` | Camera index (see [Select the camera](#3-select-the-camera)) |
| `CAM_W` / `CAM_H` / `CAM_FPS` | Camera resolution and frame rate |
| `FRAME_MIRRORED` | Flip frame horizontally (true for front-facing webcam) |
| `CLASSIFY_EVERY_N` | Run NKF classifier every N frames (default 15, ~2 fps) |
| `MIN_CONFIDENCE` | Minimum confidence to accept a new expression (default 0.4) |
| `CROP_PADDING` | Padding around the face bounding box (default 0.10) |
| `CROP_SQUARE` | Make the face crop square (default true) |

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `module 'mediapipe' has no attribute 'solutions'` | `pip install mediapipe==0.10.21` |
| `[camera] opened=False` / 0 frames | Wrong `CAM_INDEX`, or camera not connected. See [Select the camera](#3-select-the-camera). |
| `Failed to create CUDAExecutionProvider` | onnxruntime falls back to CPU. The tracker still works, since the classifier only runs every 15th frame. |
| `size mismatch` when loading weights | `MODEL_PROFILE` doesn't match the model file. Use `"update1"` for `nkf_best.pth`. |
| `cv2.__version__` shows an unexpected version | `pip install --force-reinstall --no-deps opencv-contrib-python==4.11.0.86` |

---

## License

MIT License — see [LICENSE](LICENSE) for details.