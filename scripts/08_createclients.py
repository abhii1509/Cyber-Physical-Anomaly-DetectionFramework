import os
import pickle
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "8020"

CLIENT_DIR = os.path.join(DATA, "clients")

os.makedirs(CLIENT_DIR, exist_ok=True)

# LOAD TRAIN EMBEDDINGS

Cp = joblib.load(f"{DATA}/Cp_train.pkl")
Cd = joblib.load(f"{DATA}/Cd_train.pkl")
Cy = joblib.load(f"{DATA}/Cy_train.pkl")
CS = joblib.load(f"{DATA}/CS_train.pkl")

print("Training Embeddings")
print("Cp :", Cp.shape)
print("Cd :", Cd.shape)
print("Cy :", Cy.shape)
print("CS :", CS.shape)

# SANITY CHECK

assert len(Cp) == len(Cd) == len(Cy) == len(CS), \
    "Embedding sizes do not match."

# SPLIT INTO 3 CLIENTS

N = len(Cp)
split = N // 3
indices = [
    (0, split),
    (split, 2 * split),
    (2 * split, N)
]

# SAVE CLIENT DATA

for i, (start, end) in enumerate(indices):
    client = {
        "Cp": Cp[start:end],
        "Cd": Cd[start:end],
        "Cy": Cy[start:end],
        "CS": CS[start:end]
    }

    save_path = os.path.join(
        CLIENT_DIR,
        f"client_{i+1}.pkl"
    )

    with open(save_path, "wb") as f:
        pickle.dump(client, f)

    print(f"\nClient {i+1}")
    print("Samples :", len(client["Cp"]))
    print("Cp :", client["Cp"].shape)
    print("Cd :", client["Cd"].shape)
    print("Cy :", client["Cy"].shape)
    print("CS :", client["CS"].shape)
print("Client Creation Completed Successfully")
print(f"Total Samples : {N}")
print(f"Clients Saved : {CLIENT_DIR}")