import torch
import numpy as np
from dataset import DISEASES, NUM_CLASSES


def enable_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


from torch.amp import autocast

def mc_dropout_ensemble(models, input_tensor, device, n_passes=10):
    all_probs = []

    for model in models:
        model.eval()
        enable_dropout(model)

        with torch.no_grad():
            for _ in range(n_passes):
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    logits = model(input_tensor.to(device))
                    probs  = torch.sigmoid(logits).float()
                
                if torch.isnan(probs).any():
                    print(f"Warning: NaN detected in probs, skipping pass")
                    continue
                    
                all_probs.append(probs.cpu().numpy())

    if len(all_probs) == 0:
        n = input_tensor.shape[0]
        return {
            'mean_probs':  [0.5] * NUM_CLASSES,
            'uncertainty': [0.0] * NUM_CLASSES,
            'n_passes':    0,
            'per_disease': {
                DISEASES[i]: {
                    'probability': 0.5,
                    'uncertainty': 0.0,
                    'confidence':  'Low'
                }
                for i in range(NUM_CLASSES)
            }
        }

    all_probs  = np.stack(all_probs, axis=0)
    mean_probs = all_probs.mean(axis=0).squeeze()
    variance   = all_probs.var(axis=0).squeeze()

    variance   = np.nan_to_num(variance, nan=0.0)
    uncertainty = np.clip(variance * 10, 0, 1)

    def confidence_label(u):
        if u < 0.1: return "High"
        if u < 0.3: return "Moderate"
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