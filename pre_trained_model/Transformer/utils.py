import numpy as np
import torch

def unpack_batch(batch):
    if isinstance(batch, (list, tuple)):
        return batch[0], batch[1] if len(batch) > 1 else None
    return batch, None


def compute_errors(model, loader, device):
    model.eval()

    errors, labels = [], []

    with torch.no_grad():
        for batch in loader:
            x, y = unpack_batch(batch)
            x = x.to(device)

            recon = model(x)
            err = ((x - recon) ** 2).mean(dim=(1, 2))

            errors.extend(err.cpu().numpy())

            if y is not None:
                labels.extend(y.numpy())

    return np.array(errors), np.array(labels) if len(labels) > 0 else None