import torch

def compute_score(model, x):
    z = model(x)
    dist = ((z - model.center) ** 2).sum(dim=1)
    return dist


def predict(model, x, threshold):
    score = compute_score(model, x)
    pred = (score > threshold).int()
    return score, pred