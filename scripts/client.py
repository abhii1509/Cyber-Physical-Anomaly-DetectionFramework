import sys
import pickle
import torch
from pathlib import Path
import flwr as fl
from opacus import PrivacyEngine
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from model import (
    FusionNetwork,
    get_parameters,
    set_parameters,
    train_local,
    evaluate,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "8020"

CLIENT_DIR = DATA / "clients"
CHECKPOINT = DATA / "models" / "FusionNetwork_8020.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# CLIENT ID

cid = int(sys.argv[1])
print(f"\nStarting Client {cid+1}")

# LOAD CLIENT DATA

with open(
    f"{CLIENT_DIR}/client_{cid+1}.pkl",
    "rb"
) as f:
    data = pickle.load(f)
loader = DataLoader(
    TensorDataset(
        data["Cp"],
        data["Cd"],
        data["Cy"],
        data["CS"]
    ),
    batch_size=128,
    shuffle=True
)

# FLOWER CLIENT

class PCCFClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = FusionNetwork().to(DEVICE)
        self.model.load_state_dict(
            torch.load(
                CHECKPOINT,
                map_location=DEVICE
            )
        )
        for name, param in self.model.named_parameters():
            print(name, param.requires_grad)
        print("Loaded Centralized Model")
    def get_parameters(self, config):
        return get_parameters(self.model)
    def set_parameters(self, parameters):
        set_parameters(
            self.model,
            parameters
        )
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=2e-4
        )
        global_params = [
            p.detach().clone()
            for p in self.model.parameters()
        ]

        train_loader = loader
        self.model.train()
        privacy_engine = PrivacyEngine()
        self.model, optimizer, train_loader = privacy_engine.make_private(
            module=self.model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=0.8,
            max_grad_norm=1.0
        )    
        loss = train_local(
            self.model,
            train_loader,
            optimizer,
            DEVICE,
            mu=0.01,
            global_params=global_params,
            epochs=1
        )
        with torch.no_grad():
            x = torch.cat(
                [
                    data["Cp"].to(DEVICE),
                    data["Cd"].to(DEVICE),
                    data["Cy"].to(DEVICE),
                    data["CS"].to(DEVICE)
                ],
                dim=1
            )
            w = torch.softmax(
                self.model.gate.fc(x),
                dim=1
            )
            print("\nAverage Gate Weights")
            print("---------------------")
            print("Physical :", w[:,0].mean().item())
            print("Device   :", w[:,1].mean().item())
            print("Cyber    :", w[:,2].mean().item())
            print("CS        :", w[:,3].mean().item())
        delta = 1e-5
        epsilon = privacy_engine.get_epsilon(
            delta=delta
        )
        print(f"Epsilon : {epsilon:.2f}")
        print(f"Delta   : {delta}")
        return (
            get_parameters(self.model),
            len(train_loader.dataset),
            {
                "loss": float(loss)
            }
        )
    def evaluate(
        self,
        parameters,
        config
    ):
        self.set_parameters(parameters)
        errors = evaluate(
            self.model,
            loader,
            DEVICE
        )
        loss = sum(errors)/len(errors)
        return (
            float(loss),
            len(loader.dataset),
            {}
        )

# START CLIENT

fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=PCCFClient().to_client()
)