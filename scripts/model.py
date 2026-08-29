import torch
import torch.nn as nn

# Modality Gate

class ModalityGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(99,4)
    def forward(self,x):
        weights = torch.softmax(
            self.fc(x),
            dim=1
        )
        Cp = x[:,:32] * weights[:,0:1]
        Cd = x[:,32:64] * weights[:,1:2]
        Cy = x[:,64:96] * weights[:,2:3]
        CS = x[:,96:] * weights[:,3:4]
        fusion = torch.cat(
            [
                Cp,
                Cd,
                Cy,
                CS
            ],
            dim=1
        )
        return fusion

# Fusion Autoencoder

class FusionAutoencoder(nn.Module):
    def __init__(self, input_dim=99):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16,32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,input_dim)
        )
    def forward(self,x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return z,recon

# Complete PCCF Fusion Network

class FusionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = ModalityGate()
        self.fusion = FusionAutoencoder(99)
    def forward(self, Cp, Cd, Cy, CS):
        # Concatenate modality embeddings
        x = torch.cat(
            [Cp, Cd, Cy, CS],
            dim=1
        )
        # Apply learnable gate
        fusion_input = self.gate(x)
        # Fusion Autoencoder
        z, recon = self.fusion(fusion_input)
        return z, recon, fusion_input

# Parameter Utilities

def get_parameters(model):
    return [
        val.cpu().numpy()
        for _,val in model.state_dict().items()
    ]

def set_parameters(model,parameters):
    params_dict = zip(
        model.state_dict().keys(),
        parameters
    )
    state_dict = {
        k:torch.tensor(v)
        for k,v in params_dict
    }
    model.load_state_dict(
        state_dict,
        strict=True
    )

# LOCAL TRAINING (FedProx)

def train_local(
    model,
    loader,
    optimizer,
    device,
    mu=0.01,
    global_params=None,
    epochs=1
):
    model.train()
    total_loss = 0
    for epoch in range(epochs):
        epoch_loss = 0
        for Cp, Cd, Cy, CS in loader:
            Cp = Cp.to(device)
            Cd = Cd.to(device)
            Cy = Cy.to(device)
            CS = CS.to(device)
            optimizer.zero_grad()
            z, recon, fusion = model(
                Cp,
                Cd,
                Cy,
                CS
            )
            mse = torch.mean(
                (recon - fusion) ** 2,
                dim=1
            )
            reconstruction_loss = mse.mean()
            anomaly_loss = mse.std()
            lambda_std = 0.5
            loss = reconstruction_loss + lambda_std * anomaly_loss

            # FedProx Regularization

            if global_params is not None:
                prox = torch.tensor(
                    0.0,
                    device=device
                )
                for w, w_global in zip(
                    model.parameters(),
                    global_params
                ):
                    if w.requires_grad:
                        prox += torch.sum(
                            (w - w_global.to(device)) ** 2
                        )
                loss += (mu / 2.0) * prox
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(loader)
        print(
            f"Epoch {epoch+1}/{epochs}  Loss : {epoch_loss:.6f}"
        )
        total_loss += epoch_loss
    return total_loss / epochs

# Evaluation

@torch.no_grad()
def evaluate(
    model,
    loader,
    device
):
    model.eval()
    errors = []
    for Cp,Cd,Cy,CS in loader:
        Cp = Cp.to(device)
        Cd = Cd.to(device)
        Cy = Cy.to(device)
        CS = CS.to(device)
        z,recon,fusion = model(
            Cp,
            Cd,
            Cy,
            CS
        )
        err = torch.mean(
            (recon-fusion)**2,
            dim=1
        )
        errors.extend(
            err.cpu().numpy()
        )
    return errors
