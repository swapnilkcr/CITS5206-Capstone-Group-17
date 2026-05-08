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


    # thresholds = np.linspace(errors.min(), errors.max(), 100)
    low = np.percentile(errors, 0)
    high = np.percentile(errors, 70)#train on test:66

    thresholds = np.linspace(low, high, 100)

    precisions = []
    recalls = []
    f1s = []
    accs = []

    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

    for th in thresholds:
        preds = (errors > th).astype(int)

        precisions.append(precision_score(labels, preds))
        recalls.append(recall_score(labels, preds))
        f1s.append(f1_score(labels, preds))
        accs.append(accuracy_score(labels, preds))

    # best_idx = np.argmax(f1s)
    best_idx = np.argmax(recalls)
    best_th = thresholds[best_idx]

    print("Best Threshold:", best_th)
    print("Best Precision:", precisions[best_idx])
    print("Best Recall:", recalls[best_idx])
    print("Best F1:", f1s[best_idx])
    print("Best Accuracy:", accs[best_idx])

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(thresholds, f1s, label="F1")
    plt.plot(thresholds, recalls, label="Recall")
    plt.plot(thresholds, precisions, label="Precision")
    plt.plot(thresholds, accs, label="Accuracy")

    # best threshold
    plt.axvline(best_th, linestyle="--", label="Best Threshold")

    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Performance vs Threshold")
    plt.legend()
    plt.grid()

    plt.show()
    # threshold = np.percentile(errors, 41)
    # preds = (errors > threshold).astype(int)

    # print("Threshold:", threshold)

    # if labels is not None:
    #     from sklearn.metrics import precision_score, recall_score, f1_score

    #     print("Precision:", precision_score(labels, preds))
    #     print("Recall:", recall_score(labels, preds))
    #     print("F1:", f1_score(labels, preds))
    # else:
    #     print("No labels, only anomaly scores computed.")