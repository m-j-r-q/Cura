import torch
import numpy as np
from dataset import DISEASES, NUM_CLASSES


def enable_dropout(model):
    """Enable dropout layers during inference."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def mc_dropout_ensemble( models: list, input_tensor: torch.Tensor, device, n_passes: int = 10) -> dict:
    all_probs = []

    for model in models:
        model.eval()
        enable_dropout(model)

        with torch.no_grad():
            for _ in range(n_passes):
                logits = model(input_tensor.to(device))
                probs  = torch.sigmoid(logits)
                all_probs.append(probs.cpu().numpy())

        model.eval()

    all_probs = np.stack(all_probs, axis=0)

    mean_probs = all_probs.mean(axis=0).squeeze()
    variance   = all_probs.var(axis=0).squeeze()

    uncertainty = np.clip(variance * 10, 0, 1)

    def confidence_label(u):
        if u < 0.1:  return "High"
        if u < 0.3:  return "Moderate"
        return "Low"

    return {
        'mean_probs':  mean_probs.tolist(),
        'uncertainty': uncertainty.tolist(),
        'n_passes':    len(all_probs),
        'per_disease': {
            DISEASES[i]: {
                'probability':  round(float(mean_probs[i]), 4),
                'uncertainty':  round(float(uncertainty[i]), 4),
                'confidence':   confidence_label(uncertainty[i]),
            }
            for i in range(NUM_CLASSES)
        }
    }