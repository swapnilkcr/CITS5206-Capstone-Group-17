import pandas as pd
import torch
from torch.utils.data import DataLoader
import joblib
import numpy as np

from dataset import SKABDataset
from model import TransformerAE
from utils import compute_errors

TEST_FILE = "data/7.csv"

MODEL_PATH = "models/best_model.pth"
SCALER_PATH = "models/scaler.pkl"

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = joblib.load(SCALER_PATH)

    df = pd.read_csv(TEST_FILE, sep=';')
    dataset = SKABDataset(df, seq_len=60, scaler=scaler)
    loader = DataLoader(dataset, batch_size=32)

    model = TransformerAE(input_dim=dataset.X.shape[2])
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    errors, labels = compute_errors(model, loader, device)

    threshold = np.percentile(errors, 41)
    preds = (errors > threshold).astype(int)

    print("Threshold:", threshold)

    if labels is not None:
        from sklearn.metrics import precision_score, recall_score, f1_score

        print("Precision:", precision_score(labels, preds))
        print("Recall:", recall_score(labels, preds))
        print("F1:", f1_score(labels, preds))
    else:
        print("No labels, only anomaly scores computed.")