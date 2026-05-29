import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from helper import *


def train_classifier_mlp(source_csv: str, keyword: str=""):

    # if balanced:
    #     print("Using balanced dataset")
    #     csv_file = "keypoints_dataset_affectnet_balanced_au_aug.csv"
    #     keyword = "balanced"
    # else:
    #     print("Using original dataset")
    #     csv_file = "keypoints_dataset_affectnet_balanced_au.csv"
    #     keyword = ""
    csv_file = source_csv
    print(f"Loading data from {csv_file}...")
    
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(csv_file=csv_file)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    class_weight_dict = dict(enumerate(class_weights))
    print("Class weights:", class_weight_dict)

    # --- model ---
    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(512, activation='relu'),  # added second 512 layer
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(len(LABELS), activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # --- callbacks ---
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(
            "best_model.keras", monitor='val_accuracy', save_best_only=True)
    ]

    # --- train ---
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        class_weight=class_weight_dict
    )

    # --- evaluate ---
    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=LABELS))

    # --- plot training curves ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'],     label='train')
    ax1.plot(history.history['val_accuracy'], label='val')
    ax1.set_title('Accuracy'); ax1.legend()
    ax2.plot(history.history['loss'],     label='train')
    ax2.plot(history.history['val_loss'], label='val')
    ax2.set_title('Loss'); ax2.legend()
    plt.savefig(f"training_curves_{keyword}.png")
    print(f"Saved training_curves_{keyword}.png")

    # --- confusion matrix ---
    cm = confusion_matrix(y_test, y_pred)
    # print("\nConfusion Matrix:")
    # print(pd.DataFrame(cm, index=LABELS, columns=LABELS))

    # --- save model ---
    model_path = f"mlp_model_{keyword}.keras"
    model.save(model_path)
    print(f"Saved model to {model_path}")

    # save scaler too — needed for inference
    # note: scaler is already saved inside prepare_data as scaler.pkl
    # but save a keyword-specific copy to match the model
    scaler_path = f"scaler_mlp_{keyword}.pkl"
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to {scaler_path}")

    return acc, cm


if __name__ == "__main__":
    # print("Training on original dataset...")
    acc, cm = train_classifier_mlp(source_csv="keypoints_dataset_affectnet_balanced_au_aug.csv", keyword="")
    print(f"\nTest accuracy: {acc:.4f}")
    print("\nConfusion Matrix non-balanced:")
    print(pd.DataFrame(cm, index=LABELS, columns=LABELS))

    # print("\nTraining on balanced dataset...")
    # acc_balanced, cm_balanced = train_classifier_mlp(balanced=True)
    # print(f"\nTest accuracy (balanced): {acc_balanced:.4f}")
    # print("\nConfusion Matrix balanced:")
    # print(pd.DataFrame(cm_balanced, index=LABELS, columns=LABELS))