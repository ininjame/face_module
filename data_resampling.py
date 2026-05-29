from sklearn.utils import resample
import pandas as pd

TARGET = 2000  # target samples per class
OUTPUT_CSV = "keypoints_dataset_affectnet_balanced.csv"

df = pd.read_csv("keypoints_dataset_affectnet.csv")
balanced = []
for label in df["label_name"].unique():
    subset = df[df["label_name"] == label]
    if len(subset) < TARGET:
        # oversample minority classes
        subset = resample(subset, n_samples=TARGET,
                         replace=True, random_state=42)
    else:
        # undersample majority classes
        subset = resample(subset, n_samples=TARGET,
                         replace=False, random_state=42)
    balanced.append(subset)

df = pd.concat(balanced).sample(frac=1, random_state=42)
print("Balanced dataset:")
print(df["label_name"].value_counts())
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {len(df)} samples to {OUTPUT_CSV}")