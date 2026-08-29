import torch
import torch.nn as nn


# MODALITY GATE

class ModalityGate(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc = nn.Linear(99, 4)

    def forward(self, x):

        weights = torch.softmax(
            self.fc(x),
            dim=1
        )

        Cp = (
            x[:, :32]
            * weights[:, 0:1]
        )

        Cd = (
            x[:, 32:64]
            * weights[:, 1:2]
        )

        Cy = (
            x[:, 64:96]
            * weights[:, 2:3]
        )

        CS = (
            x[:, 96:]
            * weights[:, 3:4]
        )

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


# FUSION AUTOENCODER

class FusionAutoencoder(nn.Module):

    def __init__(self, input_dim=99):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(
                input_dim,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                16
            )
        )

        self.decoder = nn.Sequential(

            nn.Linear(
                16,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                input_dim
            )
        )

    def forward(self, x):

        z = self.encoder(x)

        recon = self.decoder(z)

        return z, recon


# COMPLETE PCCF FUSION NETWORK

class FusionNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.gate = ModalityGate()

        self.fusion = FusionAutoencoder(99)

    def forward(
        self,
        Cp,
        Cd,
        Cy,
        CS
    ):

        x = torch.cat(
            [
                Cp,
                Cd,
                Cy,
                CS
            ],
            dim=1
        )

        fusion_input = self.gate(x)

        z, recon = self.fusion(
            fusion_input
        )

        return (
            z,
            recon,
            fusion_input
        )


# FLOWER PARAMETER UTILITIES

def get_parameters(model):

    return [

        value.detach()
        .cpu()
        .numpy()

        for value in model.state_dict().values()

    ]


def set_parameters(
    model,
    parameters
):

    state_dict = {}

    for key, value in zip(
        model.state_dict().keys(),
        parameters
    ):

        state_dict[key] = torch.tensor(
            value,
            dtype=model.state_dict()[key].dtype
        )

    model.load_state_dict(
        state_dict,
        strict=True
    )


# FEDPROX LOCAL TRAINING

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

    total_loss = 0.0

    for epoch in range(epochs):

        epoch_loss = 0.0

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

            loss = (
                reconstruction_loss
                + 0.5 * anomaly_loss
            )

            # ------------------------------------------------
            # FedProx Regularization
            # ------------------------------------------------

            if global_params is not None:

                prox = torch.zeros(
                    (),
                    device=device
                )

                for w, w_global in zip(
                    model.parameters(),
                    global_params
                ):

                    wg = w_global.to(
                        device=device,
                        dtype=w.dtype
                    )

                    prox += torch.sum(
                        (w - wg) ** 2
                    )

                loss += (
                    mu / 2.0
                ) * prox

            # ------------------------------------------------
            # Numerical Safety
            # ------------------------------------------------

            if not torch.isfinite(loss):

                print(
                    "WARNING: "
                    "Non-finite loss detected."
                )

                optimizer.zero_grad()

                continue

            loss.backward()

            # Same clipping used in original E4
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            # ------------------------------------------------
            # Parameter Safety
            # ------------------------------------------------

            for name, param in model.named_parameters():

                if not torch.isfinite(param).all():

                    raise RuntimeError(
                        "Non-finite parameter after "
                        f"optimizer step: {name}"
                    )

            epoch_loss += loss.item()

        epoch_loss /= len(loader)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Loss: {epoch_loss:.6f}"
        )

        total_loss += epoch_loss

    return total_loss / epochs


# LOCAL EVALUATION

@torch.no_grad()
def evaluate(
    model,
    loader,
    device
):

    model.eval()

    errors = []

    for Cp, Cd, Cy, CS in loader:

        Cp = Cp.to(device)
        Cd = Cd.to(device)
        Cy = Cy.to(device)
        CS = CS.to(device)

        z, recon, fusion = model(
            Cp,
            Cd,
            Cy,
            CS
        )

        error = torch.mean(
            (recon - fusion) ** 2,
            dim=1
        )

        errors.extend(
            error.cpu().numpy()
        )

    return errors