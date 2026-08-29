import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "8020"

# LOAD TRAINING AND TESTING DATA
Xp_train = pd.read_csv(f"{DATA}/Xp_train.csv")
Xp_test  = pd.read_csv(f"{DATA}/Xp_test.csv")

Xd_train = pd.read_csv(f"{DATA}/Xd_train.csv")
Xd_test  = pd.read_csv(f"{DATA}/Xd_test.csv")

Xc_train = pd.read_csv(f"{DATA}/Xc_train.csv")
Xc_test  = pd.read_csv(f"{DATA}/Xc_test.csv")

# REMOVE TIMESTAMP
for df in [
    Xp_train,
    Xp_test,
    Xd_train,
    Xd_test,
    Xc_train,
    Xc_test
]:

    if "Window_Start" in df.columns:
        df.drop(columns=["Window_Start"], inplace=True)

# HANDLE MISSING VALUES
Xp_train = Xp_train.fillna(0)
Xp_test  = Xp_test.fillna(0)

Xd_train = Xd_train.fillna(0)
Xd_test  = Xd_test.fillna(0)

Xc_train = Xc_train.fillna(0)
Xc_test  = Xc_test.fillna(0)

# STANDARDIZE
# FIT ONLY ON TRAINING DATA
scaler_p = StandardScaler()
scaler_d = StandardScaler()
scaler_c = StandardScaler()

Xp_train = scaler_p.fit_transform(Xp_train)
Xp_test  = scaler_p.transform(Xp_test)

Xd_train = scaler_d.fit_transform(Xd_train)
Xd_test  = scaler_d.transform(Xd_test)

Xc_train = scaler_c.fit_transform(Xc_train)
Xc_test  = scaler_c.transform(Xc_test)

# SAVE SCALERS
joblib.dump(scaler_p, f"{DATA}/Scaler_Physical.pkl")
joblib.dump(scaler_d, f"{DATA}/Scaler_Device.pkl")
joblib.dump(scaler_c, f"{DATA}/Scaler_Cyber.pkl")

# SAVE SCALED DATA
joblib.dump(Xp_train, f"{DATA}/Xp_train_scaled.pkl")
joblib.dump(Xp_test,  f"{DATA}/Xp_test_scaled.pkl")

joblib.dump(Xd_train, f"{DATA}/Xd_train_scaled.pkl")
joblib.dump(Xd_test,  f"{DATA}/Xd_test_scaled.pkl")

joblib.dump(Xc_train, f"{DATA}/Xc_train_scaled.pkl")
joblib.dump(Xc_test,  f"{DATA}/Xc_test_scaled.pkl")

# SUMMARY
print("Scaled Data Saved Successfully")

print("\nPhysical")
print("Train :", Xp_train.shape)
print("Test  :", Xp_test.shape)

print("\nDevice")
print("Train :", Xd_train.shape)
print("Test  :", Xd_test.shape)

print("\nCyber")
print("Train :", Xc_train.shape)
print("Test  :", Xc_test.shape)

print("\nNaN Check")

print("Physical :", np.isnan(Xp_train).sum())
print("Device   :", np.isnan(Xd_train).sum())
print("Cyber    :", np.isnan(Xc_train).sum())
print("\nInf Check")
print("Physical :", np.isinf(Xp_train).sum())
print("Device   :", np.isinf(Xd_train).sum())
print("Cyber    :", np.isinf(Xc_train).sum())
print("\nFinished.")