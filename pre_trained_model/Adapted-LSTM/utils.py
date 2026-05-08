import torch
import torch.nn.functional as F

# ---------------------------
# Contrastive Loss
# ---------------------------
def contrastive_loss(z, y, temperature=0.2):
    """
    simple batch-wise contrastive loss
    positive: same class
    negative: different class
    """

    z = F.normalize(z, dim=1)

    sim = torch.matmul(z, z.T) / temperature

    labels = y.unsqueeze(1) == y.unsqueeze(0)
    labels = labels.float()

    exp_sim = torch.exp(sim)

    loss = -torch.log(
        (exp_sim * labels).sum(dim=1) /
        (exp_sim.sum(dim=1) + 1e-8)
    )

    return loss.mean()


# ---------------------------
# SVDD loss
# ---------------------------
def svdd_loss(z, center):
    return ((z - center) ** 2).sum(dim=1).mean()


# ---------------------------
# anomaly margin loss
# ---------------------------
def anomaly_loss(z, center, margin):
    dist = ((z - center) ** 2).sum(dim=1)
    return torch.relu(margin - dist).pow(2).mean()