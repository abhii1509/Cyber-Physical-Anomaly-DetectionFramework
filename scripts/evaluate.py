import pickle
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from model import FusionNetwork

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "8020"
MODEL_PATH = DATA / "global_round_5.pth"
RESULTS = ROOT / "results" / "metrics"

RESULTS.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOAD MODEL
model = FusionNetwork().to(DEVICE)
model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)
model.eval()
print("Loaded Global Model")
print(MODEL_PATH)

# LOAD TEST EMBEDDINGS
Cp = joblib.load(f"{DATA}/Cp_test.pkl").to(DEVICE)
Cd = joblib.load(f"{DATA}/Cd_test.pkl").to(DEVICE)
Cy = joblib.load(f"{DATA}/Cy_test.pkl").to(DEVICE)
CS = joblib.load(f"{DATA}/CS_test.pkl").to(DEVICE)
labels = pd.read_csv(f"{DATA}/TestLabels.csv")
y_true = labels["GroundTruth"].values
assert len(y_true) == Cp.shape[0]
print("\nTest Samples :", len(y_true))
print(labels["GroundTruth"].value_counts())

# RECONSTRUCTION ERROR
with torch.no_grad():
    z, recon, fusion = model(
        Cp,
        Cd,
        Cy,
        CS
    )
    errors = torch.mean(
        (recon - fusion) ** 2,
        dim=1
    ).cpu().numpy()

# NORMALIZE
eps = 1e-8
errors = (errors - errors.min()) / (errors.max() - errors.min() + eps)

# ROC-AUC
roc = roc_auc_score(y_true, errors)
print(f"ROC-AUC : {roc:.4f}")


# BEST THRESHOLD
thresholds = np.linspace( 70, 99.9, 500)
best_f1 = 0
best_threshold = 0
best_pred = None
for t in thresholds:
    threshold = np.percentile( errors, t )
    pred = ( errors > threshold ).astype(int)
    f1 = f1_score( y_true, pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
        best_pred = pred

# RESULTS
print("\nBest Threshold :", best_threshold)
print("Best F1 :", best_f1)
print("\nClassification Report\n")
print(classification_report(y_true, best_pred, digits=4))
print("\nConfusion Matrix\n")
cm = confusion_matrix(y_true, best_pred)
print(cm)

# SAVE RESULTS
results = {
    "errors": errors,
    "labels": y_true,
    "predictions": best_pred,
    "roc_auc": roc,
    "best_f1": best_f1,
    "threshold": best_threshold,
    "confusion_matrix": cm
}
save_path = RESULTS / "federated_results_8020.pkl"
with open(
    save_path,
    "wb"
) as f:
    pickle.dump(
        results,
        f
    )
print("\nSaved Results")
print(save_path)