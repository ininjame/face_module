import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from helper import *



def train_classifier_xgb(source_csv: str, keyword: str="", use_gpu: bool=False, test_csv: str=None):

    # if balanced:
    #     print("Using balanced dataset with aug")
    #     csv_file = "keypoints_dataset_affectnet_balanced_au_aug.csv"
    #     keyword = "balanced"
    # else:
    #     print("Using original dataset")
    #     csv_file = "keypoints_dataset_affectnet_balanced_au.csv"
    #     keyword = ""
    csv_file = source_csv
    print(f"Loading data from {csv_file}...")
    
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(csv_file)

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric='mlogloss',
        early_stopping_rounds=20,
        random_state=42,
        device='cuda' if use_gpu else 'cpu'
    )

    # --- train ---
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )

    # --- evaluate ---
    y_pred = model.predict(X_test)
    acc    = np.mean(y_pred == y_test)
    print(f"\nTest accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=LABELS))

    # --- confusion matrix ---
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f',
                xticklabels=LABELS, yticklabels=LABELS,
                cmap='Blues')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title(f'XGBoost Confusion Matrix  (acc={acc:.3f})')
    plt.tight_layout()
    plt.savefig("confusion_matrix_xgb.png")
    print("Saved confusion_matrix_xgb.png")

    # --- save model ---
    with open("xgb_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Saved xgb_model.pkl")

    # save scaler too — needed for inference
    # note: scaler is already saved inside prepare_data as scaler.pkl
    # but save a keyword-specific copy to match the model
    scaler_path = f"scaler_xgb_{keyword}.pkl"
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to {scaler_path}")

    return acc, cm


if __name__ == "__main__":
    # print("Training on original dataset...")
    acc, cm = train_classifier_xgb(source_csv="keypoints_dataset_affectnet_balanced_au.csv", keyword="", use_gpu=True)
    print(f"\nTest accuracy: {acc:.4f}")

    # print("\nTraining on balanced dataset...")
    acc_balanced, cm_balanced = train_classifier_xgb(source_csv="keypoints_dataset_affectnet_balanced_au.csv", keyword="balanced", use_gpu=True)
    print(f"\nTest accuracy (balanced): {acc_balanced:.4f}")

    print("\nConfusion Matrix non-balanced:")
    print(pd.DataFrame(cm, index=LABELS, columns=LABELS))
    print("\nConfusion Matrix balanced:")
    print(pd.DataFrame(cm_balanced, index=LABELS, columns=LABELS))