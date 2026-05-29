import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import random_split
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import cv2
from PIL import Image
import mediapipe as mp
import os
from tqdm import tqdm
import pickle


# LABELS = ['neutral', 'happy', 'angry', 'sad', 'surprise', 'disgust', 'fear', 'contempt']
LABELS = ['neutral', 'happy','anger', 'sad', 'surprise', 'fear', 'disgust', 'contempt']
N_CLASSES  = len(LABELS)
N_LANDMARKS = 68
NASAL_BASE_IDX = 33  # mediapipe landmark 1 = nose tip, maps to index 33 in our 68-subset

# mediapipe landmark indices we use (68 most expressive, excluding iris)
LANDMARK_INDICES = list(range(68))  # first 68 of mediapipe's 468


def precompute_landmarks(root_dir, output_path):
    """Run MediaPipe once on all images and save results."""
    mp_face = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5
    )
    cache = {}
    for label_idx, label_name in enumerate(LABELS):
        folder = os.path.join(root_dir, label_name)
        if not os.path.isdir(folder):
            print(f"Warning: {label_name} data not found")
            return
        for fname in tqdm(os.listdir(folder), desc=label_name):
            if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            img_path = os.path.join(folder, fname)
            img_bgr  = cv2.imread(img_path)
            if img_bgr is None:
                continue
            rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            result = mp_face.process(rgb)
            if result.multi_face_landmarks:
                lm = result.multi_face_landmarks[0].landmark
                coords = np.array([[p.x, p.y] for p in lm[:68]], dtype=np.float32)
            else:
                coords = np.zeros((68, 2), dtype=np.float32)
            cache[img_path] = coords

    with open(output_path, 'wb') as f:
        pickle.dump(cache, f)
    print(f"Saved {len(cache)} landmarks to {output_path}")
    return cache




# ---------------------------------------------------------------
# 1. Global Feature Network (ResNet50 backbone)
# ---------------------------------------------------------------

class GlobalFeatureNetwork(nn.Module):
    """
    ResNet50 backbone that outputs intermediate feature maps
    at 14x14 spatial resolution, matching the paper's design.
    Feature maps from layer2 (28x28 -> downsampled) and
    layer3 (14x14) are fused as global features.
    """
    def __init__(self, pretrained=True):
        super().__init__()

        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # split resnet into stages
        self.stem   = nn.Sequential(backbone.conv1, backbone.bn1,
                                    backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1   # 256 x 56 x 56
        self.layer2 = backbone.layer2   # 512 x 28 x 28
        self.layer3 = backbone.layer3   # 1024 x 14 x 14
        self.layer4 = backbone.layer4   # 2048 x 7 x 7

        # fusion convolutions — match paper's 768 channel global feature
        # downsample layer2 output to 14x14
        self.conv_layer2 = nn.Conv2d(512, 256, kernel_size=3,
                                     stride=2, padding=1, bias=False)
        # project layer3 output
        self.conv_layer3 = nn.Conv2d(1024, 512, kernel_size=3,
                                     padding=1, bias=False)
        # fuse: 256 + 512 = 768 channels at 14x14
        self.conv_fuse   = nn.Conv2d(768, 768, kernel_size=3,
                                     padding=1, bias=False)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        f2 = self.layer2(x)           # 512 x 28 x 28
        f3 = self.layer3(f2)          # 1024 x 14 x 14

        f2_down = self.conv_layer2(f2)  # 256 x 14 x 14
        f3_proj = self.conv_layer3(f3)  # 512 x 14 x 14

        # concatenate and fuse -> 768 x 14 x 14
        fused = torch.cat([f2_down, f3_proj], dim=1)
        global_feat = self.conv_fuse(fused)   # 768 x 14 x 14

        return global_feat


# ---------------------------------------------------------------
# 2. Keypoint Feature Extraction with bilinear interpolation
# ---------------------------------------------------------------

def extract_keypoint_features(global_feat, landmark_coords, perturbation=0.0):
    """
    Extract features from global_feat at landmark coordinates
    using bilinear interpolation (as described in paper Eq. 2-3).

    Args:
        global_feat     : (B, C, H, W) feature maps
        landmark_coords : (B, N, 2) normalized landmark coords in [0, 1]
        perturbation    : max random offset for regularization (Section 3.3)

    Returns:
        keypoint_feats  : (B, N, C)
    """
    B, C, H, W = global_feat.shape
    N = landmark_coords.shape[1]

    coords = landmark_coords.clone()

    # landmark perturbation regularization (Section 3.3)
    if perturbation > 0 and global_feat.requires_grad:  # only during training
        noise = torch.zeros_like(coords).uniform_(-perturbation, perturbation)
        coords = coords + noise

    coords = coords.clamp(0, 1)

    # convert normalized [0,1] coords to [-1,1] for grid_sample
    # landmark_coords: (B, N, 2) -> grid: (B, N, 1, 2)
    grid = coords * 2 - 1  # map [0,1] -> [-1,1]
    grid = grid.unsqueeze(2)  # (B, N, 1, 2)

    # grid_sample expects (B, C, H_out, W_out) with grid (B, H_out, W_out, 2)
    # reshape for batch grid_sample
    grid_for_sample = grid.view(B, N, 1, 2)  # (B, N, 1, 2)

    # sample features at each landmark using bilinear interpolation
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
# 3. Representative Keypoint Feature Attention (RKFA)
# ---------------------------------------------------------------

class RKFA(nn.Module):
    """
    Representative Keypoint Feature Attention using nasal base landmark.
    Section 3.4 of the paper, Equation 5.

    The nasal base keypoint feature is broadcast to all N landmarks,
    concatenated, passed through FC to get attention weights,
    then applied as residual attention.
    """
    def __init__(self, feature_dim=768, n_landmarks=N_LANDMARKS,
                 nasal_base_idx=NASAL_BASE_IDX):
        super().__init__()
        self.nasal_base_idx = nasal_base_idx
        self.n_landmarks    = n_landmarks

        # FC layer: takes [nasal_feat || each_landmark_feat] -> scalar attention
        # input: feature_dim * 2 (concat of nasal + each landmark)
        # output: n_landmarks attention scores
        self.fc_attention = nn.Linear(feature_dim * 2, n_landmarks)

    def forward(self, keypoint_feats):
        """
        Args:
            keypoint_feats : (B, N, C)
        Returns:
            refined_feats  : (B, N, C)
        """
        B, N, C = keypoint_feats.shape

        # extract nasal base feature -> (B, 1, C)
        nasal_feat = keypoint_feats[:, self.nasal_base_idx:self.nasal_base_idx+1, :]

        # broadcast nasal to all N landmarks -> (B, N, C)
        nasal_broadcast = nasal_feat.expand(B, N, C)

        # concatenate along feature dim -> (B, N, 2C)
        concat = torch.cat([nasal_broadcast, keypoint_feats], dim=-1)

        # compute attention scores -> (B, N, N) -> take diagonal as per-landmark weight
        # simplified: FC over each position independently
        concat_flat = concat.view(B * N, 2 * C)
        # flat   = refined.reshape(refined.shape[0], -1)  # was .view
        attn_scores = self.fc_attention(concat_flat)  # (B*N, N)

        # take the diagonal — each landmark's own attention weight
        attn_scores = attn_scores.view(B, N, N)
        attn_weights = torch.stack(
            [attn_scores[:, i, i] for i in range(N)], dim=1
        )  # (B, N)

        # softmax normalize
        attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(-1)  # (B, N, 1)

        # residual attention (Eq. 5: F_R = F_K + 1 * softmax(...) * F_K)
        refined = keypoint_feats + attn_weights * keypoint_feats

        return refined


# ---------------------------------------------------------------
# 4. Full NKF Model
# ---------------------------------------------------------------

class NKF(nn.Module):
    """
    Network for utilizing Keypoint Features (NKF).
    Adapted from So & Han (2025) for use with MediaPipe landmarks.
    """
    def __init__(self, n_classes=N_CLASSES, n_landmarks=N_LANDMARKS,
                 feature_dim=768, perturbation=0.5, pretrained=True):
        super().__init__()

        self.n_landmarks  = n_landmarks
        self.feature_dim  = feature_dim
        self.perturbation = perturbation  # c in Eq. 4

        self.global_net = GlobalFeatureNetwork(pretrained=pretrained)
        self.rkfa       = RKFA(feature_dim, n_landmarks)

        # classifier: two FC layers as described in Section 3.2
        # input: N * C flattened keypoint features
        self.classifier = nn.Sequential(
            nn.Linear(n_landmarks * feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, n_classes)
        )

    def forward(self, images, landmark_coords):
        """
        Args:
            images          : (B, 3, 112, 112) normalized face images
            landmark_coords : (B, N, 2) landmark coords normalized to [0, 1]

        Returns:
            logits          : (B, n_classes)
        """
        # 1. extract global features
        global_feat = self.global_net(images)   # (B, 768, 14, 14)

        # 2. extract keypoint features at landmark coords
        perturbation = self.perturbation if self.training else 0.0
        kp_feats = extract_keypoint_features(
            global_feat, landmark_coords, perturbation
        )   # (B, N, 768)

        # 3. apply RKFA
        refined = self.rkfa(kp_feats)   # (B, N, 768)

        # 4. flatten and classify
        # flat   = refined.view(refined.shape[0], -1)   # (B, N * 768)
        flat   = refined.reshape(refined.shape[0], -1)  # (B, N * 768)
        logits = self.classifier(flat)                 # (B, n_classes)

        return logits


# ---------------------------------------------------------------
# 5. Dataset — loads images + extracts MediaPipe landmarks
# ---------------------------------------------------------------

class FERDataset(Dataset):
    """
    Loads face images and extracts MediaPipe landmarks on the fly.
    Expects folder structure:
        root/
          neutral/  *.jpg
          happy/    *.jpg
          ...
    """
    def __init__(self, root_dir, transform=None, img_size=96, landmark_cache=None):
        self.samples   = []
        self.transform = transform
        self.img_size  = img_size
        self.landmark_cache = landmark_cache or {}
        self.mp_face = None  # will initialize in get_landmarks to avoid re-creating for each image
        
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
        """Returns (68, 2) normalized landmark coords or None."""
        # use cache if available — no MediaPipe needed
        if img_path in self.landmark_cache:
            return self.landmark_cache[img_path]
        # fallback to live MediaPipe if not in cache
        if self.mp_face is None:
            self.mp_face = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5
            )
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        result = self.mp_face.process(rgb)
        if not result.multi_face_landmarks:
            return None
        lm = result.multi_face_landmarks[0].landmark
        coords = np.array([[p.x, p.y] for p in lm[:68]], dtype=np.float32)
        return coords  # already normalized to [0, 1]

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img_bgr = cv2.imread(img_path)

        if img_bgr is None:
            # return zeros if image can't be loaded
            img_tensor  = torch.zeros(3, self.img_size, self.img_size)
            lm_tensor   = torch.zeros(N_LANDMARKS, 2)
            return img_tensor, lm_tensor, label

        # get landmarks
        coords = self.get_landmarks(img_bgr, img_path)
        if coords is None:
            coords = np.zeros((N_LANDMARKS, 2), dtype=np.float32)

        # convert image
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = transforms.ToTensor()(img_pil)

        lm_tensor = torch.from_numpy(coords)
        return img_tensor, lm_tensor, label


# ---------------------------------------------------------------
# 6. Training
# ---------------------------------------------------------------

def train_nkf(data_root, epochs=50, batch_size=32, lr=1e-3,
              use_gpu=True, perturbation=0.5, val_split=0.1):

    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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

    train_full = FERDataset(os.path.join(data_root, 'Train'), train_transform, landmark_cache=train_cache)
    val_full   = FERDataset(os.path.join(data_root, 'Train'), val_transform, landmark_cache=train_cache)

    n        = len(train_full)
    val_size = int(val_split * n)
    indices  = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()

    train_dataset = torch.utils.data.Subset(train_full, indices[val_size:])
    val_dataset   = torch.utils.data.Subset(val_full,   indices[:val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    model = NKF(perturbation=perturbation).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(epochs):
        # --- train ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{epochs} [train]",
                     leave=False)

        for images, landmarks, labels in train_loader:
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

            # update bar with live loss and acc
            train_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc':  f"{train_correct/train_total:.4f}"
            })

        train_acc = train_correct / train_total

        # --- validate ---
        model.eval()
        val_correct, val_total = 0, 0

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1:3d}/{epochs} [val]  ",
                   leave=False)

        with torch.no_grad():
            for images, landmarks, labels in val_loader:
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
                            
    test_dataset = FERDataset(os.path.join(data_root, 'Test'), val_transform, landmark_cache=test_cache)
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
# 7. Inference — use in your live face tracker
# ---------------------------------------------------------------

def load_nkf_model(weights_path="nkf_best.pth", device='cpu'):
    model = NKF(pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model.to(device)

def predict_nkf(model, face_img_bgr, landmark_coords, device='cpu'):
    """
    Args:
        face_img_bgr    : cropped face image (BGR numpy array)
        landmark_coords : (68, 2) normalized coords from MediaPipe

    Returns:
        label, confidence
    """
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img_pil = Image.fromarray(cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB))
    img_t   = transform(img_pil).unsqueeze(0).to(device)  # (1, 3, 112, 112)
    lm_t    = torch.from_numpy(landmark_coords).unsqueeze(0).to(device)  # (1, 68, 2)

    with torch.no_grad():
        logits = model(img_t, lm_t)
        probs  = F.softmax(logits, dim=1)[0]

    pred_idx   = int(probs.argmax())
    confidence = float(probs[pred_idx])
    return LABELS[pred_idx], confidence


# ---------------------------------------------------------------
# 8. Run training
# ---------------------------------------------------------------

if __name__ == "__main__":
    # expects data/train/ and data/val/ with subfolders per class
    # run once before training
    if os.path.exists("landmarks_train.pkl"):
        print("Train landmark caches found, skipping precomputation.")
    else:
        precompute_landmarks("affectnet/Train", "landmarks_train.pkl")  
    if os.path.exists("landmarks_test.pkl"):
        print("Test landmark cache found, skipping precomputation.")
    else:
        precompute_landmarks("affectnet/Test",  "landmarks_test.pkl")

    model = train_nkf(
        data_root="affectnet/",
        epochs=50,
        batch_size=32,
        lr=1e-3,
        use_gpu=True,
        perturbation=0.5
    )

    dummy_img = torch.zeros(1, 3, 96, 96) # was 112, change to 96 to match training input size
    dummy_lm  = torch.zeros(1, 68, 2)

    # switch to eval mode — disables dropout and landmark perturbation
    # so the exported graph is the inference graph, not the training graph
    model.eval()

    device = next(model.parameters()).device  # get whatever device model is on
    dummy_img = torch.zeros(1, 3, 96, 96).to(device) # was 112, change to 96 to match training input size
    dummy_lm  = torch.zeros(1, 68, 2).to(device)

    torch.onnx.export(
        model,
        (dummy_img, dummy_lm),
        "nkf_model.onnx",
        input_names=['image', 'landmarks'],
        output_names=['logits'],
        dynamic_axes={'image': {0: 'batch'}, 'landmarks': {0: 'batch'}}
    )

    