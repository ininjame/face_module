import pandas as pd
import numpy as np
from tqdm import tqdm

FLIP_PAIRS = [
    (33, 263), (7, 249), (163, 390), (144, 373), (145, 374),
    (153, 380), (154, 381), (155, 382), (133, 362), (173, 398),
    (157, 384), (158, 385), (159, 386), (160, 387), (161, 388),
    (246, 466), (46, 276), (53, 283), (52, 282), (65, 295),
    (55, 285), (70, 300), (63, 293), (105, 334), (107, 336),
    (61, 291), (39, 269), (40, 270), (37, 267), (0, 0),
    (17, 17),  (84, 314), (91, 321), (78, 308), (81, 311),
    (13, 13),  (311, 81),
]

def extract_au_features(lm_flat):
    pts   = lm_flat.reshape(-1, 3)
    nose  = pts[1].copy()
    pts  -= nose
    scale = np.max(np.abs(pts)) + 1e-6
    pts  /= scale
    p     = pts

    # reference distance — outer eye corners, stable across expressions
    # use this to normalize all distance features
    face_width = np.linalg.norm(p[33] - p[263]) + 1e-6

    feats = {}
    feats['ear_right']          = np.linalg.norm(p[159] - p[145]) / face_width
    feats['ear_left']           = np.linalg.norm(p[386] - p[374]) / face_width
    feats['brow_raise_right']   = (p[159][1] - p[52][1])          / face_width
    feats['brow_raise_left']    = (p[386][1] - p[282][1])         / face_width
    feats['brow_inner_raise']   = ((p[159][1] - p[65][1] +
                                    p[386][1] - p[295][1]) / 2)   / face_width
    feats['brow_furrow']        = np.linalg.norm(p[55]  - p[285]) / face_width
    feats['mar']                = np.linalg.norm(p[13]  - p[14])  / face_width
    feats['mouth_width']        = np.linalg.norm(p[61]  - p[291]) / face_width
    feats['lip_corner_right_x'] = p[61][0]   / face_width
    feats['lip_corner_left_x']  = p[291][0]  / face_width
    feats['lip_corner_right_y'] = p[61][1]   / face_width
    feats['lip_corner_left_y']  = p[291][1]  / face_width
    feats['upper_lip_raise']    = p[0][1]    / face_width
    feats['nose_width']         = np.linalg.norm(p[98]  - p[327]) / face_width
    feats['jaw_drop']           = p[152][1]  / face_width
    feats['eye_squint_right']   = np.linalg.norm(p[33]  - p[133]) / face_width
    feats['eye_squint_left']    = np.linalg.norm(p[362] - p[263]) / face_width
    feats['cheek_raise']        = (p[50][1] + p[280][1]) / 2      / face_width

    au_array = np.array(list(feats.values()))
    return np.concatenate([pts.flatten(), au_array])


def augment_landmarks(X, y, factor=3):
    X_aug_list = [X]
    y_aug_list = [y]

    n_samples, n_feats = X.shape
    n_landmarks = n_feats // 3

    for _ in range(factor):
        X_new = X.copy().reshape(n_samples, n_landmarks, 3)

        angles = np.random.uniform(-15, 15, n_samples) * np.pi / 180
        cos_a  = np.cos(angles)
        sin_a  = np.sin(angles)
        x_rot  =  cos_a[:, None] * X_new[:, :, 0] - sin_a[:, None] * X_new[:, :, 1]
        y_rot  =  sin_a[:, None] * X_new[:, :, 0] + cos_a[:, None] * X_new[:, :, 1]
        X_new[:, :, 0] = x_rot
        X_new[:, :, 1] = y_rot

        scales = np.random.uniform(0.9, 1.1, n_samples)
        X_new  = X_new * scales[:, None, None]

        tx = np.random.uniform(-0.05, 0.05, n_samples)
        ty = np.random.uniform(-0.05, 0.05, n_samples)
        X_new[:, :, 0] += tx[:, None]
        X_new[:, :, 1] += ty[:, None]

        X_new += np.random.normal(0, 0.005, X_new.shape)
        X_aug_list.append(X_new.reshape(n_samples, n_feats))
        y_aug_list.append(y)

        # horizontal flip
        X_flip = X.copy().reshape(n_samples, n_landmarks, 3)
        X_flip[:, :, 0] = -X_flip[:, :, 0]
        for left_idx, right_idx in FLIP_PAIRS:
            if left_idx != right_idx:
                tmp = X_flip[:, left_idx, :].copy()
                X_flip[:, left_idx, :] = X_flip[:, right_idx, :]
                X_flip[:, right_idx, :] = tmp
        X_aug_list.append(X_flip.reshape(n_samples, n_feats))
        y_aug_list.append(y)

    X_out = np.vstack(X_aug_list)
    y_out = np.concatenate(y_aug_list)
    idx   = np.random.permutation(len(X_out))
    return X_out[idx], y_out[idx]


def build_au_csv(input_csv, output_csv):
    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)

    meta_cols = ["label", "label_name"]
    lm_cols   = [c for c in df.columns if c not in meta_cols]

    print(f"Samples: {len(df)}  Landmark features: {len(lm_cols)}")

    # get expected feature size from first sample
    n_features = len(extract_au_features(
        df[lm_cols].iloc[0].values.astype(np.float32)
    ))

    records = []
    failed  = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting AU features"):
        lm_flat = row[lm_cols].values.astype(np.float32)
        try:
            features = extract_au_features(lm_flat)
            records.append({
                "label":      row["label"],
                "label_name": row["label_name"],
                **{f"f{i}": v for i, v in enumerate(features)}
            })
        except Exception as e:
            failed += 1

    print(f"Failed: {failed}")

    out_df = pd.DataFrame(records)
    out_df.to_csv(output_csv, index=False)
    print(f"Saved {len(out_df)} samples → {output_csv}")
    print(f"Feature dimensions: {n_features}")
    print("\nClass distribution:")
    print(out_df["label_name"].value_counts())
    return out_df


def build_au_csv_with_augmentation(input_csv, output_csv, augment_factor=3):
    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)

    meta_cols = ["label", "label_name"]
    lm_cols   = [c for c in df.columns if c not in meta_cols]

    print(f"Samples: {len(df)}  Landmark features: {len(lm_cols)}")

    X = df[lm_cols].values.astype(np.float32)
    y = df["label"].values

    label_map = dict(zip(df["label"], df["label_name"]))

    print(f"Augmenting landmarks (factor={augment_factor})...")
    X_aug, y_aug = augment_landmarks(X, y, factor=augment_factor)
    label_names_aug = np.array([label_map[i] for i in y_aug])

    print(f"After augmentation: {len(X_aug)} samples")

    # get expected feature size from one good sample
    n_features = len(extract_au_features(X_aug[0]))

    records = []
    failed  = 0

    for i in tqdm(range(len(X_aug)), desc="Extracting AU features"):
        try:
            features = extract_au_features(X_aug[i])
            records.append({
                "label":      y_aug[i],
                "label_name": label_names_aug[i],
                **{f"f{j}": v for j, v in enumerate(features)}
            })
        except Exception as e:
            failed += 1

    print(f"Failed AU extraction: {failed}")

    out_df = pd.DataFrame(records)
    out_df.to_csv(output_csv, index=False)
    print(f"Saved {len(out_df)} samples → {output_csv}")
    print(f"Feature dimensions: {n_features}")
    print("\nFinal class distribution:")
    print(out_df["label_name"].value_counts())
    return out_df

if __name__ == "__main__":
    # run it
    build_au_csv(
        input_csv="keypoints_dataset_affectnet_balanced.csv",
        output_csv="keypoints_dataset_affectnet_balanced_au.csv"
    )

    build_au_csv_with_augmentation(
        input_csv="keypoints_dataset_affectnet_balanced.csv",
        output_csv="keypoints_dataset_affectnet_balanced_au_aug.csv"
    )