import joblib
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "8020"
MODEL = DATA / "models"

device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")

# LOAD EMBEDDINGS

Cp = joblib.load(f"{DATA}/Cp_train.pkl").float().to(device)
Cd = joblib.load(f"{DATA}/Cd_train.pkl").float().to(device)
Cy = joblib.load(f"{DATA}/Cy_train.pkl").float().to(device)
CS = joblib.load(f"{DATA}/CS_train.pkl").float().to(device)

print(Cp.shape)
print(Cd.shape)
print(Cy.shape)
print(CS.shape)

# CROSSMODAL ATTENTION

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
        fusion = torch.cat([Cp,Cd,Cy,CS],dim=1)
        return fusion, weights

# FUSION AUTOENCODER

class FusionAutoencoder(nn.Module):
    def __init__(self,input_dim=99):
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

# COMPLETE MODEL

class FusionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = ModalityGate()
        self.fusion = FusionAutoencoder()
    def forward(self,Cp,Cd,Cy,CS):
        x = torch.cat([Cp,Cd,Cy,CS],dim=1)
        fusion_input, weights = self.gate(x)
        z,recon = self.fusion(fusion_input)
        return z,recon,fusion_input,weights

# MODEL

model = FusionNetwork().to(device)
optimizer = torch.optim.Adam( model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)
epochs = 90
lambda_std = 0.5
best_loss = float("inf")
patience = 10
counter = 0
history = []
print("\nStarting Fusion Training...\n")

# TRAIN

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    z, recon, fusion, weights = model(
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
    loss = reconstruction_loss + lambda_std * anomaly_loss
    loss.backward()
    optimizer.step()
    scheduler.step(loss)
    lr = optimizer.param_groups[0]["lr"]
    history.append({
        "Epoch": epoch + 1,
        "ReconLoss": reconstruction_loss.item(),
        "StdLoss": anomaly_loss.item(),
        "TotalLoss": loss.item(),
        "LearningRate": lr
    })
    print(
        f"Epoch {epoch+1:03d}"
        f" | Recon {reconstruction_loss.item():.6f}"
        f" | Std {anomaly_loss.item():.6f}"
        f" | Total {loss.item():.6f}"
        f" | LR {lr:.6f}"
    )

    if loss.item() < best_loss:
        best_loss = loss.item()
        counter = 0
        torch.save(
            model.state_dict(),
            f"{MODEL}/FusionNetwork_8020.pth"
        )

        torch.save(
            model.fusion.state_dict(),
            f"{MODEL}/FusionAutoencoder_8020.pth"
        )
        print("Saved Best Model")

    else:
        counter += 1

    if counter >= patience:
        print("\nEarly stopping triggered.")
        break

# SAVE TRAINING HISTORY

history = pd.DataFrame(history)
history.to_csv(f"{MODEL}/FusionTraining_8020_history.csv", index=False)
plt.figure(figsize=(8,5))
plt.plot( history["Epoch"], history["TotalLoss"], linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Fusion Network Training Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig( f"{MODEL}/FusionTraining_8020_loss.png", dpi=300)
plt.close()
print("\nTraining history saved.")
print("Loss curve saved.")

# GATE WEIGHTS
model.eval()
with torch.no_grad():
    _,_,_,weights = model(
        Cp,
        Cd,
        Cy,
        CS
    )

print("\nAverage Gate Weights")
print("-----------------------")
print("Physical :",weights[:,0].mean().item())
print("Device   :",weights[:,1].mean().item())
print("Cyber    :",weights[:,2].mean().item())
print("CS        :",weights[:,3].mean().item())

