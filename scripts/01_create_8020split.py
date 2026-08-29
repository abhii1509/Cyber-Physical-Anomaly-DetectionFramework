import os
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUTPUT = ROOT / "data" / "8020"

os.makedirs(OUTPUT, exist_ok=True)

# LOAD SYNCHRONIZED DATA

Xp = pd.read_csv(f"{DATA}/Xp_common.csv")
Xd = pd.read_csv(f"{DATA}/Xd_common.csv")
Xc = pd.read_csv(f"{DATA}/Xc_common.csv")
labels = pd.read_csv(f"{DATA}/Actor_Level_Evaluation.csv")

print("Loaded Data")
print("Xp :", Xp.shape)
print("Xd :", Xd.shape)
print("Xc :", Xc.shape)

# CREATE TRAIN-TEST INDICES

indices = range(len(labels))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
    stratify=labels["GroundTruth"]
)

# SPLIT DATA INTO TRAINING AND TESTING SETS

Xp_train = Xp.iloc[train_idx].reset_index(drop=True)
Xp_test  = Xp.iloc[test_idx].reset_index(drop=True)

Xd_train = Xd.iloc[train_idx].reset_index(drop=True)
Xd_test  = Xd.iloc[test_idx].reset_index(drop=True)

Xc_train = Xc.iloc[train_idx].reset_index(drop=True)
Xc_test  = Xc.iloc[test_idx].reset_index(drop=True)

labels_train = labels.iloc[train_idx].reset_index(drop=True)
labels_test  = labels.iloc[test_idx].reset_index(drop=True)

# SAVE TRAINING AND TESTING DATA

Xp_train.to_csv(f"{OUTPUT}/Xp_train.csv", index=False)
Xp_test.to_csv(f"{OUTPUT}/Xp_test.csv", index=False)

Xd_train.to_csv(f"{OUTPUT}/Xd_train.csv", index=False)
Xd_test.to_csv(f"{OUTPUT}/Xd_test.csv", index=False)

Xc_train.to_csv(f"{OUTPUT}/Xc_train.csv", index=False)
Xc_test.to_csv(f"{OUTPUT}/Xc_test.csv", index=False)

labels_train.to_csv(f"{OUTPUT}/TrainLabels.csv", index=False)
labels_test.to_csv(f"{OUTPUT}/TestLabels.csv", index=False)

print("\nSaved 80-20 Split")

print("\nTraining Distribution")
print(labels_train["GroundTruth"].value_counts())

print("\nTesting Distribution")
print(labels_test["GroundTruth"].value_counts())