import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from PIL import Image
import mediapipe as mp
import os
from tqdm import tqdm
import pickle

LABELS = ['neutral', 'happy', 'anger', 'sad', 'surprise', 'fear', 'disgust', 'contempt']
N_CLASSES  = len(LABELS)

# ---------------------------------------------------------------
# Landmark index definitions
# ---------------------------------------------------------------

# original 68 expression landmarks
BASE_LANDMARKS = list(range(68))

# --- nose ---
NOSE_NOSTRIL_LEFT    = [102, 129]          # nostril flare left  (AU9 disgust/anger)
NOSE_NOSTRIL_RIGHT   = [331, 358]          # nostril flare right (AU9 disgust/anger)
NOSE_NASOLABIAL_LEFT = [49, 131]           # nasolabial furrow left  (disgust/sad)
NOSE_NASOLABIAL_RIGHT= [279, 360]          # nasolabial furrow right (disgust/sad)
NOSE_TIP             = [4]                 # general anchor

# --- cheeks ---
CHEEK_LEFT           = [50, 187, 207]      # AU6 cheek raise (genuine happy)
CHEEK_RIGHT          = [280, 411, 427]     # AU6 cheek raise (genuine happy)
CHEEK_BONE_LEFT      = [234]              # face width reference
CHEEK_BONE_RIGHT     = [454]              # face width reference

# --- jaw and chin ---
CHIN_CENTER          = [152, 175]          # AU17 chin raise (disgust/sad)
JAW_LEFT             = [172, 136]          # jaw drop (surprise/fear)
JAW_RIGHT            = [397, 365]          # jaw drop (surprise/fear)
CHIN_LEFT            = [176, 149]          # asymmetric expressions
CHIN_RIGHT           = [400, 378]          # asymmetric expressions

# --- lips (extra) ---
LIP_UPPER_LEFT       = [39, 37]            # lip pucker AU18
LIP_UPPER_RIGHT      = [269, 267]          # lip pucker AU18
LIP_LOWER_CENTER     = [17, 84]            # AU17 chin/lip interaction
LIP_CORNERS_FULL     = [76, 306]           # AU12/AU15 corner pull/depress

# --- forehead ---
FOREHEAD_CENTER      = [10]               # head pose reference
BETWEEN_BROWS        = [9]               # AU4 brow furrow (anger/fear)

# combine all extra landmarks
EXTRA_LANDMARKS = (
    NOSE_NOSTRIL_LEFT + NOSE_NOSTRIL_RIGHT +
    NOSE_NASOLABIAL_LEFT + NOSE_NASOLABIAL_RIGHT + NOSE_TIP +
    CHEEK_LEFT + CHEEK_RIGHT + CHEEK_BONE_LEFT + CHEEK_BONE_RIGHT +
    CHIN_CENTER + JAW_LEFT + JAW_RIGHT + CHIN_LEFT + CHIN_RIGHT +
    LIP_UPPER_LEFT + LIP_UPPER_RIGHT + LIP_LOWER_CENTER + LIP_CORNERS_FULL +
    FOREHEAD_CENTER + BETWEEN_BROWS
)

# deduplicate while preserving order
seen = set()
EXTRA_LANDMARKS_DEDUPED = []
for idx in EXTRA_LANDMARKS:
    if idx not in seen:
        seen.add(idx)
        EXTRA_LANDMARKS_DEDUPED.append(idx)

# final landmark set: base 68 + extra
ALL_LANDMARK_INDICES = BASE_LANDMARKS + EXTRA_LANDMARKS_DEDUPED
N_LANDMARKS = len(ALL_LANDMARK_INDICES)

print(f"Total landmarks: {N_LANDMARKS}  "
      f"(base: {len(BASE_LANDMARKS)}, extra: {len(EXTRA_LANDMARKS_DEDUPED)})")

NASAL_BASE_IDX = 33  # within the ALL_LANDMARK_INDICES list


# ---------------------------------------------------------------
# Landmark extraction helper
# ---------------------------------------------------------------

def extract_landmarks_from_result(result):
    """
    Extract ALL_LANDMARK_INDICES from a MediaPipe result.
    Returns (N_LANDMARKS, 2) float32 array or None.
    """
    if not result.multi_face_landmarks:
        return None
    lm     = result.multi_face_landmarks[0].landmark
    coords = np.array(
        [[lm[i].x, lm[i].y] for i in ALL_LANDMARK_INDICES],
        dtype=np.float32
    )
    return coords


def precompute_landmarks(root_dir, output_path):
    """Run MediaPipe once on all images and save results."""
    mp_face = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5
    )
    cache  = {}
    failed = 0

    for label_name in LABELS:
        folder = os.path.join(root_dir, label_name)
        if not os.path.isdir(folder):
            print(f"Warning: {label_name} folder not found — skipping")
            continue

        for fname in tqdm(os.listdir(folder), desc=label_name):
            if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            img_path = os.path.join(folder, fname)
            img_bgr  = cv2.imread(img_path)
            if img_bgr is None:
                continue

            rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            result = mp_face.process(rgb)
            coords = extract_landmarks_from_result(result)

            if coords is None:
                coords = np.zeros((N_LANDMARKS, 2), dtype=np.float32)
                failed += 1

            cache[img_path] = coords

    with open(output_path, 'wb') as f:
        pickle.dump(cache, f)

    print(f"Saved {len(cache)} landmarks to {output_path}  "
          f"(failed: {failed})")
    return cache


# ---------------------------------------------------------------
# 1. Global Feature Network (ResNet50 backbone)
# ---------------------------------------------------------------

class GlobalFeatureNetwork(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )

        self.stem   = nn.Sequential(backbone.conv1, backbone.bn1,
                                    backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.conv_layer2 = nn.Conv2d(512, 256, kernel_size=3,
                                     stride=2, padding=1, bias=False)
        self.conv_layer3 = nn.Conv2d(1024, 512, kernel_size=3,
                                     padding=1, bias=False)
        self.conv_fuse   = nn.Conv2d(768, 768, kernel_size=3,
                                     padding=1, bias=False)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        f2 = self.layer2(x)             # 512 x 24 x 24
        f3 = self.layer3(f2)            # 1024 x 12 x 12

        f2_down = self.conv_layer2(f2)  # 256 x 12 x 12
        f3_proj = self.conv_layer3(f3)  # 512 x 12 x 12

        fused = torch.cat([f2_down, f3_proj], dim=1)
        global_feat = self.conv_fuse(fused)  # 768 x 12 x 12

        return global_feat


# ---------------------------------------------------------------
# 2. Keypoint Feature Extraction
# ---------------------------------------------------------------

def extract_keypoint_features(global_feat, landmark_coords, perturbation=0.0):
    B, C, H, W = global_feat.shape
    N = landmark_coords.shape[1]

    coords = landmark_coords.clone()

    if perturbation > 0 and global_feat.requires_grad:
        noise  = torch.zeros_like(coords).uniform_(-perturbation, perturbation)
        coords = coords + noise

    coords = coords.clamp(0, 1)

    grid            = coords * 2 - 1
    grid            = grid.unsqueeze(2)
    grid_for_sample = grid.view(B, N, 1, 2)

    keypoint_feats = F.grid_sample(
        global_feat,
        grid_for_sample,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )  # (B, C, N, 1)

    keypoint_feats = keypoint_feats.squeeze(-1).permute(0, 2, 1)  # (B, N, C)

    return keypoint_feats


# ---------------------------------------------------------------
# 3. RKFA
# ---------------------------------------------------------------

class RKFA(nn.Module):
    def __init__(self, feature_dim=768, n_landmarks=N_LANDMARKS,
                 nasal_base_idx=NASAL_BASE_IDX):
        super().__init__()
        self.nasal_base_idx = nasal_base_idx
        self.n_landmarks    = n_landmarks
        self.fc_attention   = nn.Linear(feature_dim * 2, n_landmarks)

    def forward(self, keypoint_feats):
        B, N, C = keypoint_feats.shape

        nasal_feat      = keypoint_feats[:, self.nasal_base_idx:self.nasal_base_idx+1, :]
        nasal_broadcast = nasal_feat.expand(B, N, C)
        concat          = torch.cat([nasal_broadcast, keypoint_feats], dim=-1)

        concat_flat = concat.reshape(B * N, 2 * C)
        attn_scores = self.fc_attention(concat_flat)  # (B*N, N)

        attn_scores  = attn_scores.view(B, N, N)
        attn_weights = torch.stack(
            [attn_scores[:, i, i] for i in range(N)], dim=1
        )  # (B, N)

        attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(-1)
        refined      = keypoint_feats + attn_weights * keypoint_feats

        return refined


# ---------------------------------------------------------------
# 4. Full NKF Model
# ---------------------------------------------------------------

class NKF(nn.Module):
    def __init__(self, n_classes=N_CLASSES, n_landmarks=N_LANDMARKS,
                 feature_dim=768, perturbation=0.5, pretrained=True):
        super().__init__()

        self.n_landmarks  = n_landmarks
        self.feature_dim  = feature_dim
        self.perturbation = perturbation

        self.global_net = GlobalFeatureNetwork(pretrained=pretrained)
        self.rkfa       = RKFA(feature_dim, n_landmarks)

        self.classifier = nn.Sequential(
            nn.Linear(n_landmarks * feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, n_classes)
        )

    def forward(self, images, landmark_coords):
        global_feat = self.global_net(images)

        perturbation = self.perturbation if self.training else 0.0
        kp_feats     = extract_keypoint_features(
            global_feat, landmark_coords, perturbation
        )

        refined = self.rkfa(kp_feats)
        flat    = refined.reshape(refined.shape[0], -1)
        logits  = self.classifier(flat)

        return logits


# ---------------------------------------------------------------
# 5. Dataset
# ---------------------------------------------------------------

class FERDataset(Dataset):
    def __init__(self, root_dir, transform=None, img_size=96,
                 landmark_cache=None):
        self.samples        = []
        self.transform      = transform
        self.img_size       = img_size
        self.landmark_cache = landmark_cache or {}
        self.mp_face        = None

        for label_idx, label_name in enumerate(tqdm(LABELS, desc="Scanning dataset")):
            folder = os.path.join(root_dir, label_name)
            if not os.path.isdir(folder):
                continue
            for fname in os.listdir(folder):
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.samples.append((
                        os.path.join(folder, fname), label_idx
                    ))

    def __len__(self):
        return len(self.samples)

    def get_landmarks(self, img_bgr, img_path=None):
        if img_path in self.landmark_cache:
            return self.landmark_cache[img_path]

        if self.mp_face is None:
            self.mp_face = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5
            )

        rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        result = self.mp_face.process(rgb)
        coords = extract_landmarks_from_result(result)

        if coords is None:
            coords = np.zeros((N_LANDMARKS, 2), dtype=np.float32)

        return coords

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img_bgr = cv2.imread(img_path)

        if img_bgr is None:
            return (torch.zeros(3, self.img_size, self.img_size),
                    torch.zeros(N_LANDMARKS, 2), label)

        coords  = self.get_landmarks(img_bgr, img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = transforms.ToTensor()(img_pil)

        return img_tensor, torch.from_numpy(coords), label


# ---------------------------------------------------------------
# 6. Training
# ---------------------------------------------------------------

def train_nkf(data_root, epochs=50, batch_size=32, lr=1e-3,
              use_gpu=True, perturbation=0.5, val_split=0.1):

    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Training with {N_LANDMARKS} landmarks per face")

    train_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    with open("landmarks_train.pkl", "rb") as f:
        train_cache = pickle.load(f)

    train_full = FERDataset(os.path.join(data_root, 'Train'),
                            train_transform, landmark_cache=train_cache)
    val_full   = FERDataset(os.path.join(data_root, 'Train'),
                            val_transform,   landmark_cache=train_cache)

    n        = len(train_full)
    val_size = int(val_split * n)
    indices  = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()

    train_dataset = torch.utils.data.Subset(train_full, indices[val_size:])
    val_dataset   = torch.utils.data.Subset(val_full,   indices[:val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    model     = NKF(perturbation=perturbation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_bar = tqdm(train_loader,
                         desc=f"Epoch {epoch+1:3d}/{epochs} [train]",
                         leave=False)

        for images, landmarks, labels in train_bar:
            images    = images.to(device)
            landmarks = landmarks.to(device)
            labels    = labels.to(device)

            optimizer.zero_grad()
            logits = model(images, landmarks)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += labels.size(0)
            train_loss    += loss.item()

            train_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc':  f"{train_correct/train_total:.4f}"
            })

        train_acc = train_correct / train_total

        model.eval()
        val_correct, val_total = 0, 0
        val_bar = tqdm(val_loader,
                       desc=f"Epoch {epoch+1:3d}/{epochs} [val]  ",
                       leave=False)

        with torch.no_grad():
            for images, landmarks, labels in val_bar:
                images    = images.to(device)
                landmarks = landmarks.to(device)
                labels    = labels.to(device)

                logits = model(images, landmarks)
                preds  = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

                val_bar.set_postfix({'acc': f"{val_correct/val_total:.4f}"})

        val_acc = val_correct / val_total
        scheduler.step()

        print(f"Epoch {epoch+1:3d}/{epochs}  "
              f"loss: {train_loss/len(train_loader):.4f}  "
              f"train_acc: {train_acc:.4f}  val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "nkf_best.pth")
            print(f"  -> saved best model (val_acc={val_acc:.4f})")

    print(f"\nBest val accuracy: {best_val_acc:.4f}")

    # --- test evaluation ---
    print("\nEvaluating on test set...")
    with open("landmarks_test.pkl", "rb") as f:
        test_cache = pickle.load(f)

    test_dataset = FERDataset(os.path.join(data_root, 'Test'),
                              val_transform, landmark_cache=test_cache)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    model.load_state_dict(torch.load("nkf_best.pth", map_location=device))
    model.eval()
    test_correct, test_total = 0, 0

    with torch.no_grad():
        for images, landmarks, labels in test_loader:
            images    = images.to(device)
            landmarks = landmarks.to(device)
            labels    = labels.to(device)

            logits = model(images, landmarks)
            preds  = logits.argmax(dim=1)
            test_correct += (preds == labels).sum().item()
            test_total   += labels.size(0)

    test_acc = test_correct / test_total
    print(f"Test accuracy: {test_acc:.4f}")

    return model


# ---------------------------------------------------------------
# 7. Inference
# ---------------------------------------------------------------

def load_nkf_model(weights_path="nkf_best.pth", device='cpu'):
    model = NKF(pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device,
                                     weights_only=False))
    model.eval()
    return model.to(device)


def predict_nkf(model, face_img_bgr, landmark_coords, device='cpu'):
    """
    Args:
        face_img_bgr    : cropped face image (BGR numpy array)
        landmark_coords : (N_LANDMARKS, 2) normalized coords from MediaPipe
    """
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img_pil = Image.fromarray(cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB))
    img_t   = transform(img_pil).unsqueeze(0).to(device)
    lm_t    = torch.from_numpy(landmark_coords).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_t, lm_t)
        probs  = F.softmax(logits, dim=1)[0]

    pred_idx   = int(probs.argmax())
    confidence = float(probs[pred_idx])
    return LABELS[pred_idx], confidence


# ---------------------------------------------------------------
# 8. Run
# ---------------------------------------------------------------

if __name__ == "__main__":
    # rebuild landmark caches since N_LANDMARKS changed
    if os.path.exists("landmarks_train.pkl"):
        # check if cache matches current N_LANDMARKS
        with open("landmarks_train.pkl", "rb") as f:
            sample_cache = pickle.load(f)
        sample_coords = next(iter(sample_cache.values()))
        if sample_coords.shape[0] != N_LANDMARKS:
            print(f"Cache has {sample_coords.shape[0]} landmarks but "
                  f"model expects {N_LANDMARKS} — rebuilding...")
            precompute_landmarks("affectnet/Train", "landmarks_train.pkl")
        else:
            print("Train landmark cache OK — skipping precomputation.")
    else:
        precompute_landmarks("affectnet/Train", "landmarks_train.pkl")

    if os.path.exists("landmarks_test.pkl"):
        with open("landmarks_test.pkl", "rb") as f:
            sample_cache = pickle.load(f)
        sample_coords = next(iter(sample_cache.values()))
        if sample_coords.shape[0] != N_LANDMARKS:
            print(f"Cache has {sample_coords.shape[0]} landmarks but "
                  f"model expects {N_LANDMARKS} — rebuilding...")
            precompute_landmarks("affectnet/Test", "landmarks_test.pkl")
        else:
            print("Test landmark cache OK — skipping precomputation.")
    else:
        precompute_landmarks("affectnet/Test", "landmarks_test.pkl")

    model = train_nkf(
        data_root="affectnet/",
        epochs=50,
        batch_size=32,
        lr=1e-3,
        use_gpu=True,
        perturbation=0.5
    )

    model.eval()
    device    = next(model.parameters()).device
    dummy_img = torch.zeros(1, 3, 96, 96).to(device)
    dummy_lm  = torch.zeros(1, N_LANDMARKS, 2).to(device)

    torch.onnx.export(
        model,
        (dummy_img, dummy_lm),
        "nkf_model.onnx",
        input_names=['image', 'landmarks'],
        output_names=['logits'],
        dynamic_axes={'image': {0: 'batch'}, 'landmarks': {0: 'batch'}}
    )
    print(f"Exported nkf_model.onnx with {N_LANDMARKS} landmarks")