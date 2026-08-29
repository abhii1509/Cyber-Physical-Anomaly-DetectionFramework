import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "8020"
MODEL = DATA / "models"
os.makedirs(MODEL, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOAD TRAINING DATA
Xp = joblib.load(f"{DATA}/Xp_train_scaled.pkl")
Xp = torch.tensor(Xp,dtype=torch.float32)
loader = DataLoader(
    TensorDataset(Xp),
    batch_size=256,
    shuffle=True
)
print("Physical Training Data")
print(Xp.shape)

# MODEL
class PhysicalEncoder(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(input_dim, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 32)

        )

        self.decoder = nn.Sequential(

            nn.Linear(32, 128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.ReLU(),

            nn.Linear(256, input_dim)

        )

    def forward(self, x):

        z = self.encoder(x)

        recon = self.decoder(z)

        return z, recon


model = PhysicalEncoder(
    Xp.shape[1]
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

criterion = nn.MSELoss()

# TRAIN
epochs = 90
best_loss = float("inf")
patience = 10
counter = 0
history = []
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3,
)

print("\nStarting Training...\n")

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    for (x,) in loader:

        x = x.to(device)

        optimizer.zero_grad()

        z, recon = model(x)

        loss = criterion(recon, x)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    running_loss /= len(loader)

    scheduler.step(running_loss)

    lr = optimizer.param_groups[0]["lr"]

    history.append({
        "Epoch": epoch + 1,
        "Loss": running_loss,
        "LearningRate": lr
    })

    print(
        f"Epoch {epoch+1:03d} | "
        f"Loss: {running_loss:.6f} | "
        f"LR: {lr:.6f}"
    )

    if running_loss < best_loss:

        best_loss = running_loss
        counter = 0

        torch.save(
            model.state_dict(),
            f"{MODEL}/PhysicalEncoder_8020.pth"
        )

        print("✓ Saved Best Model")

    else:

        counter += 1

    if counter >= patience:

        print("\nEarly stopping triggered.")
        break

print("Training Finished")
print(f"Best Loss : {best_loss:.6f}")
print(f"Model Saved : {MODEL}\\PhysicalEncoder_8020.pth")

# SAVE TRAINING HISTORY

history = pd.DataFrame(history)
history.to_csv(
    f"{MODEL}/PhysicalEncoder_8020_history.csv",
    index=False
)
print("Training history saved.")

# SAVE LOSS CURVE

plt.figure(figsize=(8,5))
plt.plot(history["Epoch"], history["Loss"], linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Reconstruction Loss")
plt.title("Physical Encoder Training Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig( f"{MODEL}/PhysicalEncoder_8020_loss.png", dpi=300)
plt.close()
print("Loss curve saved.")