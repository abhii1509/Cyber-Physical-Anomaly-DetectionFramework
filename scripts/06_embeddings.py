import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "8020"
MODEL = DATA / "models"

device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")

# LOAD DATA

Xp_train = torch.tensor(
    joblib.load(f"{DATA}/Xp_train_scaled.pkl"),
    dtype=torch.float32
).to(device)
Xp_test = torch.tensor(
    joblib.load(f"{DATA}/Xp_test_scaled.pkl"),
    dtype=torch.float32
).to(device)
Xd_train = torch.tensor(
    joblib.load(f"{DATA}/Xd_train_scaled.pkl"),
    dtype=torch.float32
).to(device)
Xd_test = torch.tensor(
    joblib.load(f"{DATA}/Xd_test_scaled.pkl"),
    dtype=torch.float32
).to(device)
Xc_train = torch.tensor(
    joblib.load(f"{DATA}/Xc_train_scaled.pkl"),
    dtype=torch.float32
).to(device)
Xc_test = torch.tensor(
    joblib.load(f"{DATA}/Xc_test_scaled.pkl"),
    dtype=torch.float32
).to(device)

# PHYSICAL ENCODER

class PhysicalEncoder(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32,128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,input_dim)
        )
    def forward(self,x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return z,recon

# DEVICE ENCODER

class DeviceEncoder(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            64,
            batch_first=True
        )
        self.fc = nn.Linear(64,32)
        self.decoder = nn.Sequential(
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,input_dim)
        )
    def forward(self,x):
        _,h = self.gru(x)
        h = h[-1]
        z = self.fc(h)
        recon = self.decoder(z)
        return z,recon

# CYBER ENCODER

class CyberEncoder(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1,32,3,padding=1),
            nn.ReLU(),
            nn.Conv1d(32,64,3,padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(64,32)
        self.decoder = nn.Sequential(
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,input_dim)
        )
    def forward(self,x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.squeeze(-1)
        z = self.fc(x)
        recon = self.decoder(z)
        return z,recon

# LOAD MODELS

model_p = PhysicalEncoder( Xp_train.shape[1]).to(device)
model_d = DeviceEncoder(Xd_train.shape[1]).to(device)
model_c = CyberEncoder(Xc_train.shape[1]).to(device)
model_p.load_state_dict(torch.load(f"{MODEL}/PhysicalEncoder_8020.pth",map_location=device))
model_d.load_state_dict(torch.load(f"{MODEL}/DeviceEncoder_8020.pth",map_location=device))
model_c.load_state_dict(torch.load(f"{MODEL}/CyberEncoder_8020.pth",map_location=device))
for model in [model_p, model_d, model_c]:
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

# NORMALIZATION

def normalize(z):
    return z / (
        torch.norm(
            z,
            dim=1,
            keepdim=True
        ) + 1e-8
    )

# TRAIN EMBEDDINGS

with torch.no_grad():
    Cp_train,_ = model_p(Xp_train)
    Cd_train,_ = model_d(Xd_train.unsqueeze(1))
    Cy_train,_ = model_c(Xc_train)
Cp_train = normalize(Cp_train)
Cd_train = normalize(Cd_train)
Cy_train = normalize(Cy_train)

CS_train = torch.cat([
    F.cosine_similarity(Cp_train,Cd_train).unsqueeze(1),
    F.cosine_similarity(Cp_train,Cy_train).unsqueeze(1),
    F.cosine_similarity(Cd_train,Cy_train).unsqueeze(1)
],dim=1)

# TEST EMBEDDINGS

with torch.no_grad():
    Cp_test,_ = model_p(Xp_test)
    Cd_test,_ = model_d(Xd_test.unsqueeze(1))
    Cy_test,_ = model_c(Xc_test)
Cp_test = normalize(Cp_test)
Cd_test = normalize(Cd_test)
Cy_test = normalize(Cy_test)
CS_test = torch.cat([
    F.cosine_similarity(Cp_test,Cd_test).unsqueeze(1),
    F.cosine_similarity(Cp_test,Cy_test).unsqueeze(1),
    F.cosine_similarity(Cd_test,Cy_test).unsqueeze(1)
],dim=1)

# SAVE

joblib.dump(Cp_train.cpu(),f"{DATA}/Cp_train.pkl")
joblib.dump(Cd_train.cpu(),f"{DATA}/Cd_train.pkl")
joblib.dump(Cy_train.cpu(),f"{DATA}/Cy_train.pkl")
joblib.dump(CS_train.cpu(),f"{DATA}/CS_train.pkl")

joblib.dump(Cp_test.cpu(),f"{DATA}/Cp_test.pkl")
joblib.dump(Cd_test.cpu(),f"{DATA}/Cd_test.pkl")
joblib.dump(Cy_test.cpu(),f"{DATA}/Cy_test.pkl")
joblib.dump(CS_test.cpu(),f"{DATA}/CS_test.pkl")

print("\nTrain Shapes")
print(Cp_train.shape)
print(Cd_train.shape)
print(Cy_train.shape)
print(CS_train.shape)

print("\nTest Shapes")
print(Cp_test.shape)
print(Cd_test.shape)
print(Cy_test.shape)
print(CS_test.shape)

print("\nEmbeddings Saved Successfully")