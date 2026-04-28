import pandas as pd
import torch
from torch.utils.data import DataLoader
import joblib
import numpy as np

from dataset import SKABDataset
from model import TransformerAE
from utils import compute_errors, unpack_batch

# ===== CONFIG =====
TRAIN_FILE = "data/test.csv"
VAL_FILE   = "data/6.csv"

MODEL_PATH = "models/best_model.pth"
SCALER_PATH = "models/scaler.pkl"

EPOCHS = 50
PATIENCE = 5
# ==================

def train_model(model, train_loader, val_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in train_loader:
            x, _ = unpack_batch(batch)
            x = x.to(device)

            out = model(x)
            loss = loss_fn(out, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # ===== VALIDATION =====
        val_errors, _ = compute_errors(model, val_loader, device)
        val_loss = np.mean(val_errors)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # ===== EARLY STOPPING =====
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0

            torch.save(model.state_dict(), MODEL_PATH)
            print("===== Best model saved =====")

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("===== Early stopping triggered =====")
            break

    print("Training finished")


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(TRAIN_FILE, sep=';')
    val_df   = pd.read_csv(VAL_FILE, sep=';')

    # ===== SAFE NORMAL FILTER =====
    if "anomaly" in train_df.columns:
        train_df = train_df[train_df["anomaly"] == 0]

    train_dataset = SKABDataset(train_df, seq_len=60)
    scaler = train_dataset.scaler
    joblib.dump(scaler, SCALER_PATH)

    val_dataset = SKABDataset(val_df, seq_len=60, scaler=scaler)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=32)

    model = TransformerAE(input_dim=train_dataset.X.shape[2]).to(device)

    train_model(model, train_loader, val_loader, device)