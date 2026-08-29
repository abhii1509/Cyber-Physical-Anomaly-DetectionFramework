import torch
import flwr as fl

from config import (
    EXPERIMENT,
    INITIAL_MODEL,
    GLOBAL_DIR,
    NUM_ROUNDS,
    NUM_CLIENTS,
    FEDPROX_MU
)

from model import (
    FusionNetwork,
    get_parameters
)


# DEVICE

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# LOAD INITIAL GLOBAL MODEL

global_model = FusionNetwork().to(
    DEVICE
)

global_model.load_state_dict(
    torch.load(
        INITIAL_MODEL,
        map_location=DEVICE
    )
)

print(
    f"\nLoaded centralized PCCF model"
)

print(
    f"Experiment: {EXPERIMENT}"
)


# INITIAL PARAMETERS

initial_parameters = (
    fl.common.ndarrays_to_parameters(
        get_parameters(global_model)
    )
)


# SAVE GLOBAL MODEL AFTER EACH ROUND

def evaluate_fn(
    server_round,
    parameters,
    config
):

    state_dict = (
        global_model.state_dict()
    )

    for key, value in zip(
        state_dict.keys(),
        parameters
    ):

        state_dict[key] = torch.tensor(
            value,
            dtype=state_dict[key].dtype
        )

    global_model.load_state_dict(
        state_dict
    )

    save_path = (
        GLOBAL_DIR
        / f"global_round_{server_round}.pth"
    )

    torch.save(
        global_model.state_dict(),
        save_path
    )

    print(
        f"Saved {EXPERIMENT} "
        f"Global Model: "
        f"Round {server_round}"
    )

    return 0.0, {}


# FEDPROX STRATEGY

strategy = fl.server.strategy.FedProx(

    fraction_fit=1.0,

    fraction_evaluate=1.0,

    min_fit_clients=NUM_CLIENTS,

    min_evaluate_clients=NUM_CLIENTS,

    min_available_clients=NUM_CLIENTS,

    proximal_mu=FEDPROX_MU,

    initial_parameters=initial_parameters,

    evaluate_fn=evaluate_fn
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