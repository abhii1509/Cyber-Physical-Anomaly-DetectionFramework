import os
from pathlib import Path


# EXPERIMENT CONFIGURATION
USE_DP = os.environ.get("PCCF_USE_DP", "").strip() == "1"

if USE_DP:
    EXPERIMENT = "E2_FL_DP"
else:
    EXPERIMENT = "E1_No_DP"


# TRAINING CONFIGURATION
NUM_CLIENTS = 3
NUM_ROUNDS = 10
BATCH_SIZE = 128
LOCAL_EPOCHS = 1
LEARNING_RATE = 2e-4
FEDPROX_MU = 0.01

# DIFFERENTIAL PRIVACY CONFIGURATION
DP_NOISE_MULTIPLIER = 0.8
DP_MAX_GRAD_NORM = 1.0
DP_DELTA = 1e-5

# PROJECT PATHS
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "8020"
CLIENT_DIR = DATA / "clients"
INITIAL_MODEL = ( DATA/ "models"/ "FusionNetwork_8020.pth")

# EXPERIMENT OUTPUT PATHS
RESULTS_ROOT = (ROOT/ "results"/ "metrics")
EXPERIMENT_DIR = (RESULTS_ROOT/ EXPERIMENT)
GLOBAL_DIR = (EXPERIMENT_DIR/ "global_models")
RESULTS_FILE = (EXPERIMENT_DIR/ f"{EXPERIMENT}_results.pkl")
GLOBAL_DIR.mkdir(parents=True, exist_ok=True)

# DISPLAY CONFIGURATION
print("PCCF DIFFERENTIAL PRIVACY ABLATION")
print(f"Experiment : {EXPERIMENT}")
print(f"DP Enabled : {USE_DP}")
print(f"Clients    : {NUM_CLIENTS}")
print(f"Rounds     : {NUM_ROUNDS}")
