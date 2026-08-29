import flwr as fl
import torch
from pathlib import Path
from model import FusionNetwork, get_parameters

# CONFIGURATION
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "8020"
CHECKPOINT = DATA / "models" / "FusionNetwork_8020.pth"
NUM_ROUNDS = 20

# LOAD INITIAL GLOBAL MODEL
global_model = FusionNetwork().to(DEVICE)
global_model.load_state_dict(torch.load(CHECKPOINT,map_location=DEVICE))
print("Loaded Centralized Fusion Network")

# INITIAL PARAMETERS
initial_parameters = fl.common.ndarrays_to_parameters(get_parameters(global_model))

# SAVE GLOBAL MODEL AFTER EACH ROUND
def get_evaluate_fn():
    def evaluate(server_round, parameters, config):
        state_dict = global_model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(
                v,
                dtype=state_dict[k].dtype
        )
        global_model.load_state_dict(state_dict)
        save_path = DATA / f"global_round_{server_round}.pth"
        torch.save(
            global_model.state_dict(),
            save_path
        )
        print(f"Saved Global Model : Round {server_round}")
        return 0.0, {}
    return evaluate

# STRATEGY
strategy = fl.server.strategy.FedProx(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
    proximal_mu=0.01,
    initial_parameters=initial_parameters,
    evaluate_fn=get_evaluate_fn()
)

# START SERVER
if __name__ == "__main__":
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(
            num_rounds=NUM_ROUNDS
        ),
        strategy=strategy
    )