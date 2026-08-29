import pickle

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from config import (
    EXPERIMENT,
    DATA,
    GLOBAL_DIR,
    RESULTS_FILE
)

from model import FusionNetwork


# EVALUATION CONFIGURATION

EVALUATION_ROUND = 10

MODEL_PATH = (
    GLOBAL_DIR
    / f"global_round_{EVALUATION_ROUND}.pth"
)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# LOAD GLOBAL MODEL

model = FusionNetwork().to(
    DEVICE
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print("\n")
print(
    f"Experiment: {EXPERIMENT}"
)
print(
    f"Evaluation Round: "
    f"{EVALUATION_ROUND}"
)
print(
    f"Loaded Model: "
    f"{MODEL_PATH}"
)

# LOAD TEST EMBEDDINGS

Cp = joblib.load(DATA / "Cp_test.pkl").to(DEVICE)
Cd = joblib.load(DATA / "Cd_test.pkl").to(DEVICE)
Cy = joblib.load(DATA / "Cy_test.pkl").to(DEVICE)
CS = joblib.load(DATA / "CS_test.pkl").to(DEVICE)

# LOAD TEST LABELS

labels = pd.read_csv(DATA / "TestLabels.csv")

y_true = labels["GroundTruth"].values

assert (len(y_true) == Cp.shape[0])
print(
    f"\nTest Samples: "
    f"{len(y_true)}"
)

print(
    labels[
        "GroundTruth"
    ].value_counts()
)


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


# NORMALIZATION

errors = (

    errors - errors.min()

) / (

    errors.max()
    - errors.min()
    + 1e-8

)


# ROC-AUC

roc = roc_auc_score(
    y_true,
    errors
)


# BEST F1 THRESHOLD

best_f1 = 0.0

best_threshold = 0.0

best_pred = None


for percentile in np.linspace(
    70,
    99.9,
    500
):

    threshold = np.percentile(
        errors,
        percentile
    )

    pred = (
        errors > threshold
    ).astype(int)

    f1 = f1_score(
        y_true,
        pred
    )

    if f1 > best_f1:

        best_f1 = f1

        best_threshold = threshold

        best_pred = pred


# CONFUSION MATRIX

cm = confusion_matrix(
    y_true,
    best_pred
)


# DISPLAY RESULTS

print("\n")

print(
    f"{EXPERIMENT} RESULTS"
)

print("\n")

print(
    f"ROC-AUC : "
    f"{roc:.4f}"
)

print(
    f"F1-score: "
    f"{best_f1:.4f}"
)

print(
    f"Threshold: "
    f"{best_threshold:.6f}"
)

print("\nConfusion Matrix:")

print(cm)

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        best_pred,
        digits=4
    )
)


# SAVE RESULTS

results = {

    "experiment":
        EXPERIMENT,

    "evaluation_round":
        EVALUATION_ROUND,

    "roc_auc":
        roc,

    "best_f1":
        best_f1,

    "threshold":
        best_threshold,

    "confusion_matrix":
        cm,

    "errors":
        errors,

    "labels":
        y_true,

    "predictions":
        best_pred
}


with open(
    RESULTS_FILE,
    "wb"
) as f:

    pickle.dump(
        results,
        f
    )


print(
    "\nResults saved to:"
)

print(
    RESULTS_FILE
)