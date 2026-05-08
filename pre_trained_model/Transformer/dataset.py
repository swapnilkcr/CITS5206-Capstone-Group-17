import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class SKABDataset(Dataset):
    def __init__(self, df, seq_len=60, scaler=None):

        df = df.copy()

        # ===== LABEL SAFE =====
        self.has_label = "anomaly" in df.columns
        self.labels = df["anomaly"].values if self.has_label else None

        # ===== DROP =====
        drop_cols = [c for c in ["datetime", "anomaly", "changepoint"] if c in df.columns]
        df = df.drop(columns=drop_cols)

        # ===== SCALER =====
        if scaler is None:
            self.scaler = StandardScaler()
            data = self.scaler.fit_transform(df.values)
        else:
            self.scaler = scaler
            data = self.scaler.transform(df.values)

        # ===== SEQUENCE =====
        self.X, self.y = [], []

        for i in range(len(data) - seq_len):
            self.X.append(data[i:i+seq_len])

            if self.has_label:
                self.y.append(self.labels[i+seq_len-1])

        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32) if self.has_label else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]