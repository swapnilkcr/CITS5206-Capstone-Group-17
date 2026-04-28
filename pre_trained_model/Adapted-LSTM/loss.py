import torch

def svdd_loss(z, center):
    return ((z - center) ** 2).sum(dim=1).mean()


def anomaly_margin_loss(z, center, margin):
    dist = ((z - center) ** 2).sum(dim=1)
    loss = torch.relu(margin - dist) ** 2
    return loss.mean()