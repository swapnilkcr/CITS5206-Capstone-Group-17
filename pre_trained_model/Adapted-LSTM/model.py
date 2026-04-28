import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        z = self.fc(h[-1])
        return z


class ALSS_SVDD_CR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = LSTMEncoder(cfg.input_dim, cfg.hidden_dim, cfg.latent_dim)
        self.center = None

    def forward(self, x):
        return self.encoder(x)